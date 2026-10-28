"""
pipeline.py — three-stage extraction wrapper for the frontend.

Stage 1: PDF → raw markdown (Docling by default, or LiteParse)
Stage 2: raw markdown → cleaned markdown
Stage 3: cleaned markdown → extracted JSON samples (LLM)
"""

import os
import queue
import re
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_FIXED_MODEL_FAMILY = "gpt-5.4"

# Strategy → (phase, variant) plus a human label for the UI log.
_STRATEGY_MAP: Dict[str, Dict[str, Any]] = {
    "multi-property":  {"phase": 2, "variant": None,
                        "label": "Single pass"},
    "agentic":         {"phase": 4, "variant": "variant_E",
                        "label": "Agentic pass (review + evidence verification)"},
}

# Providers supported by the UI. "ollama" is phase 2 only (single workflow).
_OLLAMA_FAMILY = "ollama"

_PARSER_META: Dict[str, Dict[str, str]] = {
    "docling": {
        "label": "Docling",
        "start": "Running OCR, layout, and table detection...",
        "heartbeat": "   Still running Docling... ({elapsed:.0f}s elapsed)",
    },
    "liteparse": {
        "label": "LiteParse",
        "start": "Running lightweight spatial parsing and OCR...",
        "heartbeat": "   Still running LiteParse... ({elapsed:.0f}s elapsed)",
    },
}

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b"),
    re.compile(r"(?i)\bBearer\s+\S+"),
    re.compile(r"(?i)\b(api[_ -]?key)\s*[:=]\s*\S+"),
)


def _redact_text(value: Any) -> str:
    text = str(value or "No detail provided")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    text = re.sub(r"(?:/private)?/var/folders/\S+", "[temporary file]", text)
    return re.sub(r"/tmp/\S+", "[temporary file]", text)


def _sanitize_detail(value: Any, limit: int = 360) -> str:
    """Return a compact diagnostic that is safe to show in the session log."""
    detail = " ".join(_redact_text(value).split())
    if len(detail) > limit:
        detail = f"{detail[: limit - 1]}…"
    return detail


def diagnostic_for_exception(error: Exception) -> str:
    """Format an exception for the browser log without exposing credentials."""
    return f"{type(error).__name__}: {_sanitize_detail(error)}"


def _print_exception(stage: str, error: Exception) -> None:
    """Write a redacted exception and traceback to the server log."""
    print(f"[{stage} error] {diagnostic_for_exception(error)}", file=sys.stderr)
    trace = _redact_text(traceback.format_exc())
    if trace and trace.strip() != "NoneType: None":
        print(trace[:8000], file=sys.stderr)


def _failure_response(
    stage: str,
    message: str,
    *,
    diagnostic: Optional[Any] = None,
    raw_text: str = "",
) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "success": False,
        "samples": [],
        "error": message,
        "failure_stage": stage,
        "raw_text": raw_text,
    }
    if diagnostic is not None:
        response["diagnostic"] = diagnostic
    return response


def _start_heartbeat(log_fn: Callable[[str], None], message_template: str,
                     interval: int = 15) -> threading.Event:
    """Start a daemon thread that emits an elapsed-time message every `interval`
    seconds via `log_fn`. Returns an Event whose `.set()` stops the heartbeat.

    `message_template` should contain `{elapsed:.0f}` to insert seconds.
    """
    stop = threading.Event()
    def beat() -> None:
        start = time.monotonic()
        while not stop.wait(interval):
            log_fn(message_template.format(elapsed=time.monotonic() - start))
    threading.Thread(target=beat, daemon=True).start()
    return stop


def _safe_error(
    stage: str,
    error: Exception,
    log_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """Map a raw exception to a user-safe message.

    A redacted traceback goes to stderr. The returned string reaches the
    browser and therefore never includes credentials or temporary paths.
    """
    diagnostic = diagnostic_for_exception(error)
    _print_exception(stage, error)
    if log_fn is not None:
        log_fn(f"ERROR · {stage.capitalize()} · {diagnostic}")

    s = (str(error) + " " + type(error).__name__).lower()
    if any(k in s for k in ("401", "authentication", "incorrect api key",
                             "invalid_api_key", "invalid api key")):
        return ("OpenAI rejected the API key. Please check the key in Run "
                "configuration and try again.")
    if any(k in s for k in ("429", "rate limit", "rate_limit", "quota", "insufficient_quota")):
        return ("OpenAI rate limit or quota reached. Please wait a moment and "
                "try again, or check your account quota.")
    if any(k in s for k in ("context length", "maximum context", "context_length_exceeded",
                             "too many tokens")):
        return "The document is too large for the model's context window."
    if any(k in s for k in ("timeout", "timed out")):
        return "The request timed out. Please try again."
    if any(k in s for k in ("connection", "network", "unreachable", "name or service not known")):
        return "Network error reaching the model provider. Please try again."
    return f"{stage.capitalize()} failed due to an internal error. Please try again."

def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        print(f"[configuration warning] {name} must be a positive integer; using {default}")
        return default


# Parser singletons avoid repeatedly loading OCR/layout resources. Neither
# backend is assumed thread-safe, so a shared semaphore (default 1) caps all
# document conversions.
_CONVERTERS: Dict[str, Any] = {}
_CONVERTER_LOCK = threading.Lock()
_MAX_CONCURRENT_CONVERSIONS = _positive_int_env(
    "POLYCOMPIE_MAX_CONCURRENT_CONVERSIONS", 1
)
_CONVERSION_SEMAPHORE = threading.Semaphore(_MAX_CONCURRENT_CONVERSIONS)


def _get_converter(parser: str = "docling") -> Any:
    if parser not in _PARSER_META:
        raise ValueError(f"Unsupported document parser: {parser}")
    if parser not in _CONVERTERS:
        with _CONVERTER_LOCK:
            if parser not in _CONVERTERS:
                from polycompie.ingestion.pdf_converter import setup_converter
                _CONVERTERS[parser] = setup_converter(do_ocr=True, parser=parser)
    return _CONVERTERS[parser]


def warmup_converter() -> None:
    """Pre-load the default Docling backend so the first user doesn't wait."""
    _get_converter("docling")


def _parse_with_fallback(
    pdf_path: Path,
    selected_parser: str,
    log: Callable[[str], None],
    cancelled: Callable[[], bool],
) -> Dict[str, Any]:
    """Parse with the selected backend, then retry once with the alternative."""
    from polycompie.ingestion.pdf_converter import convert_pdf

    parser_order = [
        selected_parser,
        next(name for name in _PARSER_META if name != selected_parser),
    ]
    diagnostics: Dict[str, str] = {}

    for attempt, current_parser in enumerate(parser_order, 1):
        if cancelled():
            log("   Cancelled before the next parser attempt.")
            return {"success": False, "cancelled": True}

        meta = _PARSER_META[current_parser]
        if attempt > 1:
            log(f"   Fallback · retrying with {meta['label']}...")
        log(f"   {meta['start']}")

        heartbeat = _start_heartbeat(log, meta["heartbeat"])
        try:
            converter = _get_converter(current_parser)
            result = convert_pdf(
                str(pdf_path),
                converter,
                parser=current_parser,
            )
            if not isinstance(result, dict):
                raise TypeError(
                    f"{meta['label']} returned {type(result).__name__}; expected a result object"
                )
            if result.get("success"):
                markdown = result.get("text")
                if not isinstance(markdown, str) or not markdown.strip():
                    raise ValueError(f"{meta['label']} returned an empty document")
                runtime = float(result.get("runtime_sec") or 0.0)
                log(
                    f"   Parsed with {meta['label']} "
                    f"({len(markdown):,} characters in {runtime:.1f}s)"
                )
                if attempt > 1:
                    primary_label = _PARSER_META[selected_parser]["label"]
                    log(
                        f"   Recovered · {meta['label']} succeeded after "
                        f"{primary_label} failed."
                    )
                return {
                    "success": True,
                    "text": markdown,
                    "runtime_sec": runtime,
                    "parser": current_parser,
                    "fallback_from": selected_parser if attempt > 1 else None,
                }

            error_type = _sanitize_detail(result.get("error_type") or "ParserError")
            detail = f"{error_type}: {_sanitize_detail(result.get('error'))}"
        except Exception as error:
            detail = diagnostic_for_exception(error)
            _print_exception(f"{current_parser} parser", error)
        finally:
            heartbeat.set()

        diagnostics[current_parser] = detail
        log(f"   Parser failure · {meta['label']} · {detail}")

    labels = " and ".join(_PARSER_META[name]["label"] for name in parser_order)
    log(f"ERROR · Parsing · {labels} both failed for this document.")
    return {
        "success": False,
        "cancelled": False,
        "error": f"Parsing failed with both {labels}. See Pipeline activity for details.",
        "failure_stage": "parsing",
        "diagnostic": diagnostics,
    }


def run_extraction(
    pdf_bytes: bytes,
    pdf_name: str,
    property_list: List[str],
    log_queue: queue.Queue,
    api_key: Optional[str] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    strategy: str = "multi-property",
    provider: str = "openai",
    ollama_model: Optional[str] = None,
    parser: str = "docling",
) -> Dict[str, Any]:
    """Run the full extraction pipeline. Returns {success, samples, error, raw_text}.

    raw_text is the unparsed model response, populated only when provider="ollama".
    """

    def log(msg: str) -> None:
        log_queue.put(msg)

    def cancelled() -> bool:
        return bool(is_cancelled and is_cancelled())

    def cancel_response() -> Dict[str, Any]:
        return _failure_response("cancelled", "Cancelled by user.")

    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        message = "Input validation failed: the uploaded PDF is empty."
        log(f"ERROR · Input · {message}")
        return _failure_response("input", message)

    normalized_properties = [
        str(prop).strip() for prop in property_list if str(prop).strip()
    ]
    if not normalized_properties:
        message = "Input validation failed: no properties were selected."
        log(f"ERROR · Input · {message}")
        return _failure_response("input", message)
    property_list = normalized_properties

    provider = str(provider or "openai").strip().lower()
    if provider not in {"openai", "ollama"}:
        message = f"Unsupported model provider: {provider}"
        log(f"ERROR · Configuration · {message}")
        return _failure_response("configuration", message)

    strategy = str(strategy or "multi-property").strip().lower()
    if strategy not in _STRATEGY_MAP:
        log(
            f"WARNING · Configuration · Unknown strategy '{strategy}'; "
            "using Single pass."
        )
        strategy = "multi-property"

    # Browser-provided names are untrusted input. Keep only the basename before
    # writing into the temporary directory so a crafted name cannot escape it.
    safe_pdf_name = Path(pdf_name).name or "document.pdf"
    article_id = Path(safe_pdf_name).stem
    parser = str(parser or "docling").strip().lower()
    if parser not in _PARSER_META:
        message = f"Unsupported document parser: {parser}"
        log(f"ERROR · Configuration · {message}")
        return _failure_response("configuration", message)
    parser_meta = _PARSER_META[parser]

    # ----- Stage 1: PDF → markdown ------------------------------------------
    size_mb = len(pdf_bytes) / (1024 * 1024)
    log(
        f"Parsing PDF ({size_mb:.1f} MB) with {parser_meta['label']}..."
    )

    acquired = False
    waiting_logged = False
    while not acquired:
        if cancelled():
            log("   Cancelled while waiting for a parsing slot.")
            return cancel_response()
        acquired = _CONVERSION_SEMAPHORE.acquire(timeout=0.25)
        if not acquired and not waiting_logged:
            log(
                "   Server busy — waiting for an available parsing slot "
                f"(limit={_MAX_CONCURRENT_CONVERSIONS})..."
            )
            waiting_logged = True

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / safe_pdf_name
            pdf_path.write_bytes(pdf_bytes)
            parse_result = _parse_with_fallback(
                pdf_path,
                parser,
                log,
                cancelled,
            )
            if parse_result.get("cancelled"):
                return cancel_response()
            if not parse_result.get("success"):
                return _failure_response(
                    "parsing",
                    parse_result.get("error")
                    or "Both document parsers failed. See Pipeline activity for details.",
                    diagnostic=parse_result.get("diagnostic"),
                )
            raw_markdown = parse_result["text"]
            parser_used = parse_result["parser"]
            parser_fallback_from = parse_result.get("fallback_from")
    except Exception as error:
        message = _safe_error("parsing", error, log)
        return _failure_response(
            "parsing",
            message,
            diagnostic=diagnostic_for_exception(error),
        )
    finally:
        _CONVERSION_SEMAPHORE.release()

    if cancelled():
        log("   Cancelled after Stage 1.")
        return cancel_response()

    # ----- Stage 2: clean markdown ------------------------------------------
    log("Cleaning document...")
    try:
        from polycompie.ingestion.cleaner import MarkdownCleaner
        cleaned, stats = MarkdownCleaner().clean(raw_markdown)
        if not isinstance(cleaned, str) or not cleaned.strip():
            raise ValueError("Document cleaning produced an empty document")
        removed_pct = stats.get("reduction_percent", 0)
        log(f"   Cleaned ({removed_pct:.1f}% non-informative content removed)")
    except Exception as error:
        message = _safe_error("cleaning", error, log)
        return _failure_response(
            "cleaning",
            message,
            diagnostic=diagnostic_for_exception(error),
        )

    if cancelled():
        log("   Cancelled after Stage 2 (skipping LLM call).")
        return cancel_response()

    # ----- Stage 3: LLM extraction ------------------------------------------
    # Ollama is phase 2 only — agentic strategy is silently downgraded.
    if provider == "ollama":
        strategy = "multi-property"
    s = _STRATEGY_MAP.get(strategy, _STRATEGY_MAP["multi-property"])
    phase, variant = s["phase"], s["variant"]

    if provider == "ollama":
        model_family = _OLLAMA_FAMILY
        model_label = f"Ollama / {ollama_model or '?'}"
    else:
        model_family = _FIXED_MODEL_FAMILY
        model_label = "GPT-5.4"

    log(f"Extracting properties: {', '.join(property_list)}...")
    log(f"   Model: {model_label}  |  Strategy: {s['label']}")

    try:
        from polycompie.extraction.prompts.prompt_manager import PromptManager
        from polycompie.extraction.runtime import (
            apply_variant_config,
            load_schema,
            make_extractor,
        )
        from polycompie.orchestrator.orchestrator import OrchestratorRunner

        log("   Preparing extraction request...")
        config = PromptManager().get_phase_config(
            phase=phase,
            model_family=model_family,
            property_list=property_list,
        )
        if provider == "ollama":
            if not ollama_model:
                message = (
                    "No local model selected. Pick an Ollama model in Run configuration."
                )
                log(f"ERROR · Configuration · {message}")
                return _failure_response("configuration", message)
            config["model_string"] = ollama_model
        if variant:
            apply_variant_config(config, variant)
        extractor = make_extractor(
            model_family=model_family,
            model_string=config["model_string"],
            temperature=config["temperature"],
            max_out=config["max_output_tokens"],
            api_key=api_key,
        )

        log("   Sending to model (this may take 1–2 minutes)...")

        # Heartbeat only useful for non-agentic phases. Agentic mode emits its
        # own per-step messages through log_callback, so an extra heartbeat
        # would just add noise.
        hb = None
        if strategy != "agentic":
            hb = _start_heartbeat(log, "   Still waiting on model... ({elapsed:.0f}s elapsed)")

        try:
            schema = load_schema()
            output = OrchestratorRunner(
                extractor=extractor,
                schema=schema,
                log_callback=lambda m: log(m),
            ).run(
                article_id=article_id,
                markdown=cleaned,
                phase=phase,
                model_family=model_family,
                config=config,
            )
        finally:
            if hb is not None:
                hb.set()

        status = str(output.get("status") or "unknown")
        samples = output.get("output")
        if samples is None or status == "failed":
            warnings = output.get("warnings") or []
            detail = warnings[0].get("message") if warnings and isinstance(warnings[0], dict) else None
            message = detail or "The extraction model did not return a usable result."
            diagnostic = f"Orchestrator status={status}: {_sanitize_detail(message)}"
            log(f"ERROR · Extraction · {diagnostic}")
            return _failure_response(
                "extraction",
                message,
                diagnostic=diagnostic,
                raw_text=getattr(extractor, "last_raw_response", "") or "",
            )
        if not isinstance(samples, list):
            message = "The extraction model returned an unexpected result shape."
            diagnostic = f"Expected list, received {type(samples).__name__}"
            log(f"ERROR · Extraction · {diagnostic}")
            return _failure_response(
                "extraction",
                message,
                diagnostic=diagnostic,
                raw_text=getattr(extractor, "last_raw_response", "") or "",
            )
        for s in samples:
            if isinstance(s, dict):
                s["_source_pdf"] = pdf_name
        n = len(samples)
        log(f"   Extracted {n} sample{'s' if n != 1 else ''}")
        # Capture last raw response from Ollama extractor for UI download.
        raw_text = getattr(extractor, "last_raw_response", "") or ""
        return {
            "success": True,
            "samples": samples,
            "error": None,
            "raw_text": raw_text,
            "warnings": output.get("warnings") or [],
            "parser": parser_used,
            "parser_fallback_from": parser_fallback_from,
        }

    except Exception as error:
        message = _safe_error("extraction", error, log)
        return _failure_response(
            "extraction",
            message,
            diagnostic=diagnostic_for_exception(error),
        )

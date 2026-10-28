from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class AgentResult:
    data: Any
    attempts: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.data is not None


class BaseAgent:
    """Shared LLM call and JSON parse; subclasses implement run() and use _call_and_parse()."""

    def __init__(
        self,
        extractor: Any,
        verbose: bool = True,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.extractor = extractor
        self.verbose = verbose
        self.log_callback = log_callback

    def _log(self, msg: str) -> None:
        full = f"[{self.__class__.__name__}] {msg}"
        if self.log_callback is not None:
            self.log_callback(full)
        elif self.verbose:
            print(full)

    def _call_and_parse(
        self,
        *,
        article_id: str,
        markdown: str,
        prompt: str,
        max_retries: int,
        max_out: int,
        temperature: Optional[float],
    ) -> Tuple[str, Any, List[Dict[str, Any]]]:
        """LLM call with retries; returns (raw_text, parsed_data_or_None, attempts)."""
        from ..utils import parse_json

        attempts: List[Dict[str, Any]] = []
        last_text = ""

        for attempt in range(1, max_retries + 1):
            self._log(f"LLM attempt {attempt}/{max_retries} (chars={len(markdown):,})")
            try:
                text = self.extractor.generate(
                    prompt=prompt,
                    document=markdown,
                    article_id=article_id,
                    max_output_tokens=max_out,
                    temperature=temperature,
                )
                last_text = text or ""

                if not isinstance(text, str) or not text.strip():
                    self._log("empty_response: model returned empty content")
                    attempts.append({"attempt": attempt, "status": "empty_response"})
                    continue

                data, err = parse_json(last_text)
                if err:
                    self._log(f"parse_failed: {err}")
                    # Show head + tail of raw response so we can diagnose
                    # malformed output (markdown fences, prose preamble,
                    # truncation, etc.) — open-source models love these.
                    head = last_text[:400].replace("\n", "\\n")
                    tail = last_text[-200:].replace("\n", "\\n") if len(last_text) > 400 else ""
                    self._log(f"raw_head[400]: {head}")
                    if tail:
                        self._log(f"raw_tail[200]: {tail}")
                    attempts.append({"attempt": attempt, "status": "parse_failed", "error": err, "text_len": len(last_text)})
                    continue

                self._log("parse_ok")
                attempts.append({"attempt": attempt, "status": "success", "text_len": len(last_text)})
                return last_text, data, attempts

            except Exception as e:
                self._log(f"llm_failed: {e}")
                attempts.append({"attempt": attempt, "status": "llm_failed", "error": str(e)})

        return last_text, None, attempts

    def run(self, **kwargs) -> AgentResult:
        raise NotImplementedError
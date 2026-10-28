"""Run extraction (single or batch): ledger + PromptManager + OrchestratorRunner; writes output.json and agentic_logs."""
import argparse
import hashlib
import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..ledger.ledger_utils import LedgerManager
from ..orchestrator.orchestrator import OrchestratorRunner
from .experiment_tracker import ExperimentTracker
from .prompts.prompt_manager import PromptManager
from .runtime import (
    apply_variant_config,
    load_schema,
    make_extractor,
)


def _count_samples(result: Any) -> int:
    """Return number of samples in the extraction result (list length)."""
    return len(result) if isinstance(result, list) else 0


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _safe_artifact_component(value: Any) -> str:
    """Return a stable, single-directory name for model-produced identifiers."""
    raw = str(value)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    if safe == raw and safe not in {"", ".", ".."} and len(safe) <= 80:
        return safe
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{(safe or 'missing')[:64]}-{digest}"


def _persist_agentic_logs(
    output_dir: Path,
    *,
    phase: int,
    workflow_name: str,
    run_log: Dict[str, Any],
) -> Dict[str, str]:
    """Write run artifacts to output_dir/agentic_logs/. Returns label -> path."""
    artifacts = (run_log or {}).get("artifacts", {})
    logs_dir = output_dir / "agentic_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    run_log_path = logs_dir / "run_log.json"
    _write_json(
        run_log_path,
        {
            "phase": phase,
            "workflow": workflow_name,
            "run_log": run_log or {},
        },
    )
    written["run_log"] = str(run_log_path)

    if str(workflow_name).lower() != "agentic":
        return written
    if not isinstance(artifacts, dict) or not artifacts:
        return written

    if phase == 3:
        if "inventory" in artifacts:
            p = logs_dir / "inventory.json"
            _write_json(p, artifacts["inventory"])
            written["inventory"] = str(p)
        return written

    if phase == 4:
        # 1. Global extractor output
        if "extract" in artifacts:
            p = logs_dir / "extract.json"
            _write_json(p, artifacts["extract"])
            written["extract"] = str(p)

        # 2. Self-review output (if present)
        if "self_review" in artifacts:
            p = logs_dir / "self_review.json"
            _write_json(p, artifacts["self_review"])
            written["self_review"] = str(p)

        samples_data = artifacts.get("samples", {})
        if not isinstance(samples_data, dict) or not samples_data:
            return written

        # 3. Summary — one row per sample
        summary = []
        for sid, sdata in samples_data.items():
            if not isinstance(sdata, dict):
                continue
            iterations = sdata.get("iterations") or []
            last_iter = iterations[-1] if iterations else {}
            summary.append({
                "sample_id": sdata.get("sample_id", sid),
                "sample_log_key": sid,
                "sample_index": sdata.get("sample_index"),
                "confidence": sdata.get("confidence", "unknown"),
                "iterations_run": len(iterations),
                "final_flagged_count": last_iter.get("flagged_count", "n/a"),
            })
        p = logs_dir / "summary.json"
        _write_json(p, summary)
        written["summary"] = str(p)

        # 4. Per-sample iteration files
        samples_dir = logs_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        for sid, sdata in samples_data.items():
            if not isinstance(sdata, dict):
                continue

            artifact_id = _safe_artifact_component(sid)
            sample_dir = samples_dir / f"sample_{artifact_id}"
            sample_dir.mkdir(exist_ok=True)

            for iter_data in sdata.get("iterations") or []:
                if not isinstance(iter_data, dict):
                    continue
                n = iter_data.get("iteration", "?")

                if "with_evidence" in iter_data:
                    p = sample_dir / f"iter_{n}_with_evidence.json"
                    _write_json(p, iter_data["with_evidence"])
                    written[f"sample_{artifact_id}_iter_{n}_with_evidence"] = str(p)

                if "verdict" in iter_data:
                    p = sample_dir / f"iter_{n}_verdict.json"
                    _write_json(p, iter_data["verdict"])
                    written[f"sample_{artifact_id}_iter_{n}_verdict"] = str(p)

                if "repaired" in iter_data:
                    p = sample_dir / f"iter_{n}_repaired.json"
                    _write_json(p, iter_data["repaired"])
                    written[f"sample_{artifact_id}_iter_{n}_repaired"] = str(p)

        return written

    return written


def extract_single_paper(
    article_id: str,
    phase: int,
    model_family: str,
    ledger_path: str,
    experiments_root: str,
    experiment_log_path: str,
    force: bool = False,
    variant: Optional[str] = None,
    seed: Optional[int] = None,
) -> bool:
    """Run extraction for one article; returns True on success."""
    ledger_manager = LedgerManager(ledger_path)
    exp_tracker = ExperimentTracker(
        log_path=experiment_log_path,
        experiments_root=experiments_root,
    )

    pm = PromptManager()

    entry = ledger_manager.find_by_article_id(article_id)
    if not entry:
        print(f"Article not found in ledger: {article_id}")
        return False

    # Variant- and seed-aware model key for paths and tracking
    model_key = f"{model_family}/{variant}" if variant else model_family
    if seed is not None:
        model_key = f"{model_key}/seed_{seed}"

    if not force and exp_tracker.is_completed(article_id, phase, model_key):
        print("Already completed, skipping (use --force to redo)")
        return True

    if entry.clean_status != "ok":
        print(f"Not cleaned yet: {article_id} (clean_status={entry.clean_status})")
        return False

    clean_path = Path(entry.get_clean_path(base_dir="data"))
    if not clean_path.exists():
        print(f"Cleaned file not found: {clean_path}")
        return False

    paths = exp_tracker.get_output_paths(article_id, phase, model_key)
    output_dir = Path(paths["output_dir"])
    output_file = Path(paths["output_file"])
    log_file = Path(paths["log_file"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now().isoformat()
    base_log: Dict[str, Any] = {
        "article_id": article_id,
        "phase": phase,
        "model_family": model_family,
        "variant": variant,
        "timestamp_start": start_time,
    }
    failure_stage = "read_cleaned_article"
    resolved_config: Dict[str, Any] = {"seed": seed}

    try:
        markdown = clean_path.read_text(encoding="utf-8")
        input_chars = len(markdown)
        print(f"Input: {input_chars:,} chars")

        failure_stage = "resolve_configuration"
        config = pm.get_phase_config(phase=phase, model_family=model_family)

        # Merge variant ablation config if specified
        if variant:
            variant_cfg = apply_variant_config(config, variant)
            enabled = config["orchestrator"].get("ablation", {}).get("enabled_steps", [])
            print(f"Variant: {variant} — enabled_steps={enabled}")

            baseline_variant = variant_cfg.get("baseline_variant")
            if baseline_variant:
                failure_stage = "load_baseline_extraction"
                baseline_key = f"{model_family}/{baseline_variant}"
                baseline_paths = exp_tracker.get_output_paths(article_id, phase, baseline_key)
                baseline_file = Path(baseline_paths["output_file"])
                if not baseline_file.exists():
                    raise FileNotFoundError(
                        f"baseline_variant '{baseline_variant}' required by '{variant}' "
                        f"but {baseline_file} does not exist. Run '{baseline_variant}' first."
                    )
                baseline_data = json.loads(baseline_file.read_text(encoding="utf-8"))
                config["baseline_extraction"] = baseline_data
                print(f"Loaded baseline from {baseline_variant}: {len(baseline_data) if isinstance(baseline_data, list) else '?'} samples")

        workflow_name = str(config.get("workflow") or "single")
        model_string = str(config.get("model_string") or "").strip()
        raw_temp = config.get("temperature", None)
        temperature: Optional[float] = float(raw_temp) if raw_temp is not None else None
        max_out = int(config.get("max_output_tokens", 8192))
        max_retries = int(config.get("max_retries", 2))
        reasoning_effort = config.get("reasoning_effort") or None
        resolved_config.update({
            "model_string": model_string,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "seed": seed,
            "max_output_tokens": max_out,
            "max_retries": max_retries,
        })

        print(f"Model family: {model_family}")
        print(f"Model string: {model_string}")
        print(f"Workflow: {workflow_name}")
        print(
            f"Temp: {temperature} | ReasoningEffort: {reasoning_effort} | "
            f"MaxOut: {max_out} | MaxRetries: {max_retries} | Seed: {seed}"
        )

        if not model_string:
            raise ValueError("model_string is missing from the resolved phase configuration")

        failure_stage = "initialize_model_adapter"
        extractor = make_extractor(
            model_family=model_family,
            model_string=model_string,
            temperature=temperature,
            max_out=max_out,
            reasoning_effort=reasoning_effort,
            seed=seed,
        )

        schema = load_schema()

        runner = OrchestratorRunner(extractor=extractor, schema=schema)

        failure_stage = "run_orchestrator"
        run_result: Dict[str, Any] = runner.run(
            article_id=article_id,
            markdown=markdown,
            phase=phase,
            model_family=model_family,
            config=config,
        )

        end_time = datetime.now().isoformat()

        result = run_result.get("output")
        status = run_result.get("status", "unknown")
        warnings = run_result.get("warnings", [])
        run_log = run_result.get("log", {}) or {}

        failure_stage = "persist_agentic_logs"
        agentic_logs_written = _persist_agentic_logs(
            output_dir,
            phase=phase,
            workflow_name=workflow_name,
            run_log=run_log,
        )

        if result is None:
            log_data = {
                **base_log,
                "timestamp_end": end_time,
                "status": status,
                "workflow": workflow_name,
                "prompt_yaml": pm.yaml_path,
                "config": resolved_config,
                "input_chars": input_chars,
                "warnings": warnings,
                "orchestrator_log": run_log,
                "agentic_logs_written": agentic_logs_written,
            }
            _write_json(log_file, log_data)
            exp_tracker.record_failure(article_id, phase, model_key, status or "no_output")
            ledger_manager.update_entry(article_id, extract_status="failed")
            print(f"Failed: {status}")
            return False

        failure_stage = "write_extraction_output"
        _write_json(output_file, result)

        num_samples = _count_samples(result)
        print(f"Extracted {num_samples} samples")

        log_data = {
            **base_log,
            "timestamp_end": end_time,
            "status": status,
            "workflow": workflow_name,
            "prompt_yaml": pm.yaml_path,
            "config": resolved_config,
            "input_chars": input_chars,
            "output": {
                "num_samples": num_samples,
                "output_file": str(output_file),
            },
            "phase3": {
                "inventory_file": agentic_logs_written.get("inventory") if phase == 3 else None
            } if phase == 3 else None,
            "warnings": warnings,
            "orchestrator_log": run_log,
            "agentic_logs_written": agentic_logs_written,
        }
        failure_stage = "write_run_log"
        _write_json(log_file, log_data)

        if status in ("success", "ok", "completed", "completed_with_warnings"):
            exp_tracker.record_success(article_id, phase, model_key)
            ledger_manager.update_entry(article_id, extract_status="ok")
            return True

        exp_tracker.record_failure(article_id, phase, model_key, status or "unknown")
        ledger_manager.update_entry(article_id, extract_status="failed")
        return False

    except Exception as e:
        end_time = datetime.now().isoformat()
        err = str(e)
        print(f"Unexpected error: {err}")
        traceback.print_exc()

        log_data = {
            **base_log,
            "timestamp_end": end_time,
            "status": "error",
            "failed_stage": failure_stage,
            "exception_type": type(e).__name__,
            "error": err,
            "config": resolved_config,
            "traceback": traceback.format_exc(),
        }
        try:
            _write_json(log_file, log_data)
        except OSError as log_err:
            print(f"[warn] could not write failure log to {log_file}: {log_err}")

        exp_tracker.record_failure(article_id, phase, model_key, err)
        ledger_manager.update_entry(article_id, extract_status="failed")
        return False


def batch_extract(
    collection: str,
    phase: int,
    model_family: str,
    *,
    limit: Optional[int] = None,
    ledger_path: str,
    experiments_root: str,
    experiment_log_path: str,
    force: bool = False,
    variant: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """Run extraction for all ledger entries in the given collection with clean_status ok."""
    manager = LedgerManager(ledger_path)
    entries = manager.load()

    to_process = [e for e in entries if e.collection == collection and e.clean_status == "ok"]
    if limit:
        to_process = to_process[:limit]

    print("=" * 70)
    print("Batch Extraction")
    print("=" * 70)
    print(f"Collection: {collection}")
    print(f"Phase: {phase}")
    print(f"Model: {model_family}")
    if variant:
        print(f"Variant: {variant}")
    if seed is not None:
        print(f"Seed: {seed}")
    print(f"Papers: {len(to_process)}")
    if force:
        print("Force: Enabled (will redo completed experiments)")
    print("=" * 70)

    results = {"success": 0, "failed": 0, "errors": []}

    for i, entry in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] {entry.article_id}")
        ok = extract_single_paper(
            article_id=entry.article_id,
            phase=phase,
            model_family=model_family,
            ledger_path=ledger_path,
            experiments_root=experiments_root,
            experiment_log_path=experiment_log_path,
            force=force,
            variant=variant,
            seed=seed,
        )
        if ok:
            results["success"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(entry.article_id)

    print("\n" + "=" * 70)
    print("Batch Complete")
    print("=" * 70)
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")
    print("=" * 70)

    if results["errors"]:
        print("\nFailed articles:")
        for aid in results["errors"]:
            print(f"  - {aid}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run extraction experiments")

    parser.add_argument("--article-id", type=str, help="Single article to process")
    parser.add_argument("--batch", type=str, help="Batch process collection by name")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt-5.2", "gpt-5.4", "claude-sonnet-4.6", "gemini-2.5-pro"],
        help="Model family to use",
    )
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4], help="Experiment phase")
    parser.add_argument("--variant", type=str, default=None,
                        help="Ablation variant from variants.yaml (e.g., variant_A, variant_E). Phase 4 only.")
    parser.add_argument("--limit", type=int, help="Limit batch size (for testing)")
    parser.add_argument("--ledger", type=str, default="data/ledger.csv", help="Path to ledger CSV file")
    parser.add_argument("--experiments-root", type=str, default="data/experiments", help="Root directory for experiment outputs")
    parser.add_argument("--experiment-log", type=str, default="data/experiments/experiment_log.json", help="Path to experiment tracker JSON log")
    parser.add_argument("--force", action="store_true", help="Force re-extraction even if already completed")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the model. When set, outputs are stored under .../seed_<N>/ "
             "so multiple seeds for the same (article, phase, model) coexist.",
    )

    args = parser.parse_args()

    if args.variant and args.phase != 4:
        print("[warn] --variant is only used with --phase 4. Ignoring.")
        args.variant = None

    if args.phase == 4 and not args.variant:
        print("[info] Phase 4 without --variant runs the full pipeline (equivalent to variant_E).")
        print("[info] Use --variant variant_A/B/C/D/E for ablation experiments.")

    if args.article_id:
        ok = extract_single_paper(
            article_id=args.article_id,
            phase=args.phase,
            model_family=args.model,
            ledger_path=args.ledger,
            experiments_root=args.experiments_root,
            experiment_log_path=args.experiment_log,
            force=args.force,
            variant=args.variant,
            seed=args.seed,
        )
        return 0 if ok else 1

    if args.batch:
        batch_extract(
            collection=args.batch,
            phase=args.phase,
            model_family=args.model,
            limit=args.limit,
            ledger_path=args.ledger,
            experiments_root=args.experiments_root,
            experiment_log_path=args.experiment_log,
            force=args.force,
            variant=args.variant,
            seed=args.seed,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

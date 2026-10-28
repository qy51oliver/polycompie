"""Route extraction by workflow: single (one LLM call) or agentic (phase3: inventory→attach; phase4: extract → self_review → per-sample [evidence→verify→repair] loop)."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Union

from jsonschema import Draft202012Validator

from .agents import (
    ExtractorAgent,
    EvidenceAdderAgent,
    VerifierAgent,
    RepairerAgent,
    SelfReviewAgent,
    InventoryAgent,
    AttachPropertiesAgent,
)
from .agents.verifier_agent import _all_supported, _count_flagged
from .agents.self_review_agent import extract_confidence, strip_confidence


@dataclass
class OrchestratorResult:
    output: Any
    status: str
    warnings: List[Dict[str, Any]]
    log: Dict[str, Any]


class OrchestratorRunner:
    def __init__(
        self,
        extractor: Any,
        schema: Optional[Dict[str, Any]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.extractor = extractor
        self.schema = schema
        self.log_callback = log_callback
        # _verbose is set per-run from config in _set_verbose().
        self._verbose: bool = False

    def run(
        self,
        *,
        article_id: str,
        markdown: str,
        phase: int,
        model_family: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._set_verbose(config)
        workflow = self._normalize_workflow(config.get("workflow"))
        self._p(f"START article={article_id} phase={phase} model_family={model_family} workflow={workflow}")

        if workflow == "single":
            res = self._run_single(article_id=article_id, markdown=markdown, config=config)
        elif workflow == "agentic":
            res = self._run_agentic(article_id=article_id, markdown=markdown, phase=phase, config=config)
        else:
            res = OrchestratorResult(
                output=None,
                status="failed",
                warnings=[{"type": "bad_workflow", "message": f"Unknown workflow '{workflow}'"}],
                log={"workflow": workflow, "phase": phase, "model_family": model_family},
            )

        self._p(f"END status={res.status} warnings={len(res.warnings)}")
        return {"output": res.output, "status": res.status, "warnings": res.warnings, "log": res.log}


    def _run_single(self, *, article_id: str, markdown: str, config: Dict[str, Any]) -> OrchestratorResult:
        prompt = (config.get("prompt") or "").strip()
        if not prompt:
            return OrchestratorResult(
                output=None,
                status="failed",
                warnings=[{"type": "missing_prompt", "message": "workflow=single but config.prompt is empty"}],
                log={"workflow": "single"},
            )

        max_retries = int(config.get("max_retries", 2))
        raw_temp = config.get("temperature")
        temperature = float(raw_temp) if raw_temp is not None else None
        max_out = int(config.get("max_output_tokens", 8192))
        self._p(f"single workflow max_retries={max_retries} temp={temperature} max_out={max_out}")

        result = ExtractorAgent(self.extractor, verbose=self._verbose, log_callback=self.log_callback).run(
            article_id=article_id,
            markdown=markdown,
            prompt=prompt,
            max_retries=max_retries,
            max_out=max_out,
            temperature=temperature,
        )

        if not result.success:
            return OrchestratorResult(
                output=None,
                status="failed",
                warnings=[{"type": "single_failed", "message": "All retries failed"}],
                log={"workflow": "single", "attempts": result.attempts},
            )

        warnings = list(result.warnings) + self._schema_warnings(result.data)
        return OrchestratorResult(
            output=result.data,
            status="completed_with_warnings" if warnings else "success",
            warnings=warnings,
            log={"workflow": "single", "attempts": result.attempts},
        )


    def _run_agentic(self, *, article_id: str, markdown: str, phase: int, config: Dict[str, Any]) -> OrchestratorResult:
        steps = config.get("agentic_steps") or {}
        if not isinstance(steps, dict) or not steps:
            return OrchestratorResult(
                output=None,
                status="failed",
                warnings=[{"type": "missing_agentic_steps", "message": "workflow=agentic but agentic_steps missing/empty"}],
                log={"workflow": "agentic", "phase": phase},
            )

        if "inventory" in steps and "attach_properties" in steps:
            self._p("agentic phase3: inventory -> attach_properties")
            return self._run_phase3(article_id=article_id, markdown=markdown, config=config, steps=steps)

        if "extract" in steps:
            self._p("agentic phase4: extract -> [self_review] -> per-sample [evidence -> verify -> repair] loop")
            return self._run_phase4(article_id=article_id, markdown=markdown, config=config, steps=steps)

        return OrchestratorResult(
            output=None,
            status="failed",
            warnings=[{"type": "unknown_agentic_plan", "message": f"Unrecognized agentic_steps keys: {sorted(steps.keys())}"}],
            log={"workflow": "agentic", "phase": phase},
        )


    def _run_phase3(self, *, article_id: str, markdown: str, config: Dict[str, Any], steps: Dict[str, str]) -> OrchestratorResult:
        prompt_inventory = (steps.get("inventory") or "").strip()
        prompt_attach = (steps.get("attach_properties") or "").strip()

        if not prompt_inventory or not prompt_attach:
            return OrchestratorResult(
                output=None,
                status="failed",
                warnings=[{"type": "missing_step_prompt", "message": "agentic_steps.inventory or attach_properties is empty"}],
                log={"workflow": "agentic", "plan": "phase3_two_step"},
            )

        shared = self._shared_params(config)
        log: Dict[str, Any] = {"workflow": "agentic", "plan": "phase3_two_step", "steps": {}, "artifacts": {}}
        warnings: List[Dict[str, Any]] = []

        inv_result = InventoryAgent(self.extractor, verbose=self._verbose, log_callback=self.log_callback).run(
            article_id=article_id, markdown=markdown, prompt=prompt_inventory, **shared
        )
        log["steps"]["inventory"] = {"attempts": inv_result.attempts}
        if not inv_result.success:
            return OrchestratorResult(
                output=None, status="failed",
                warnings=[{"type": "phase3_inventory_failed", "message": "Could not parse inventory JSON"}],
                log=log,
            )
        log["artifacts"]["inventory"] = inv_result.data
        warnings.extend([{"step": "inventory", **w} for w in self._schema_warnings(inv_result.data)])

        att_result = AttachPropertiesAgent(self.extractor, verbose=self._verbose, log_callback=self.log_callback).run(
            article_id=article_id, inventory=inv_result.data, markdown=markdown,
            prompt=prompt_attach, **shared
        )
        log["steps"]["attach_properties"] = {"attempts": att_result.attempts}
        if not att_result.success:
            return OrchestratorResult(
                output=None, status="failed",
                warnings=[{"type": "phase3_attach_failed", "message": "Could not parse attach_properties JSON"}],
                log=log,
            )
        warnings.extend([{"step": "attach_properties", **w} for w in self._schema_warnings(att_result.data)])

        return OrchestratorResult(
            output=att_result.data,
            status="completed_with_warnings" if warnings else "success",
            warnings=warnings,
            log=log,
        )


    def _run_phase4(self, *, article_id: str, markdown: str, config: Dict[str, Any], steps: Dict[str, str]) -> OrchestratorResult:
        prompt_extract = (steps.get("extract") or "").strip()
        prompt_self_review = (steps.get("self_review") or "").strip()
        prompt_add_ev = (steps.get("add_evidence") or "").strip()
        prompt_verify = (steps.get("verify_evidence") or "").strip()
        prompt_repair = (steps.get("repair_unsupported") or "").strip()

        if not prompt_extract:
            return OrchestratorResult(
                output=None, status="failed",
                warnings=[{"type": "missing_step_prompt", "message": "phase4 requires extract step prompt"}],
                log={"workflow": "agentic", "plan": "phase4_grounded"},
            )

        orch_cfg = config.get("orchestrator") or {}
        ablation_cfg = orch_cfg.get("ablation") if isinstance(orch_cfg.get("ablation"), dict) else {}
        enabled_steps = [str(s).strip().lower() for s in (ablation_cfg.get("enabled_steps") or [
            "extract", "self_review", "add_evidence", "verify_evidence", "repair_unsupported"
        ])]
        enabled_set = set(enabled_steps)
        early_stop = bool(ablation_cfg.get("early_stop", True))
        routing_mode = str(ablation_cfg.get("routing_mode", "rule_router")).strip().lower() or "rule_router"

        if "extract" not in enabled_set:
            return OrchestratorResult(
                output=None, status="failed",
                warnings=[{"type": "bad_ablation_config", "message": "phase4 requires 'extract' in enabled_steps"}],
                log={"workflow": "agentic", "plan": "phase4_grounded"},
            )

        # Validate prompts for enabled steps
        for step_name, prompt_text in [
            ("self_review", prompt_self_review),
            ("add_evidence", prompt_add_ev),
            ("verify_evidence", prompt_verify),
            ("repair_unsupported", prompt_repair),
        ]:
            if step_name in enabled_set and not prompt_text:
                return OrchestratorResult(
                    output=None, status="failed",
                    warnings=[{"type": "missing_step_prompt", "message": f"phase4 requires {step_name} prompt when enabled"}],
                    log={"workflow": "agentic", "plan": "phase4_grounded"},
                )

        max_iterations = int(ablation_cfg.get("max_iterations", orch_cfg.get("max_iterations", 3)))
        if "verify_evidence" not in enabled_set:
            max_iterations = 1

        keep_low_confidence = bool(orch_cfg.get("keep_low_confidence", True))
        shared = self._shared_params(config)

        log: Dict[str, Any] = {
            "workflow": "agentic",
            "plan": "phase4_grounded",
            "max_iterations": max_iterations,
            "ablation": {
                "enabled_steps": sorted(enabled_set),
                "early_stop": early_stop,
                "routing_mode": routing_mode,
            },
            "steps": {},
            "artifacts": {"extract": None, "self_review": None, "samples": {}},
        }
        warnings: List[Dict[str, Any]] = []

        # step 1: extract (fresh call, or load from baseline variant if provided)
        baseline_data = config.get("baseline_extraction")
        if baseline_data is not None:
            extracted = baseline_data if isinstance(baseline_data, list) else []
            self._p(f"phase4 step=extract (from baseline, {len(extracted)} samples)")
            log["steps"]["extract"] = {"source": "baseline_variant", "n_samples": len(extracted)}
        else:
            self._p("phase4 step=extract (global)")
            ext_result = ExtractorAgent(self.extractor, verbose=self._verbose, log_callback=self.log_callback).run(
                article_id=article_id, markdown=markdown, prompt=prompt_extract, **shared
            )
            log["steps"]["extract"] = {"attempts": ext_result.attempts}

            if not ext_result.success:
                return OrchestratorResult(
                    output=None, status="failed",
                    warnings=[{"type": "phase4_extract_failed", "message": "Could not parse extract JSON"}],
                    log=log,
                )

            extracted = ext_result.data if isinstance(ext_result.data, list) else []

        log["artifacts"]["extract"] = extracted
        warnings.extend([{"step": "extract", **w} for w in self._schema_warnings(extracted)])
        self._p(f"phase4 extracted {len(extracted)} samples")

        # step 2: self-review (global, optional)
        reviewed_samples = extracted
        if "self_review" in enabled_set:
            self._p("phase4 step=self_review (global)")
            sr_agent = SelfReviewAgent(self.extractor, verbose=self._verbose, log_callback=self.log_callback)
            sr_result = sr_agent.run(
                article_id=article_id,
                samples=extracted,
                markdown=markdown,
                prompt=prompt_self_review,
                **shared,
            )
            log["steps"]["self_review"] = {"attempts": sr_result.attempts}

            if sr_result.success:
                reviewed_samples = sr_result.data if isinstance(sr_result.data, list) else extracted
                log["artifacts"]["self_review"] = reviewed_samples
                if sr_result.warnings:
                    warnings.extend([{"step": "self_review", **w} for w in sr_result.warnings])

                high_count = sum(1 for s in reviewed_samples if isinstance(s, dict) and extract_confidence(s) == "HIGH")
                low_count = sum(1 for s in reviewed_samples if isinstance(s, dict) and extract_confidence(s) != "HIGH")
                self._p(f"phase4 self_review: {high_count} HIGH, {low_count} LOW confidence")
            else:
                self._p("phase4 self_review failed — proceeding with original extraction")
                warnings.append({"step": "self_review", "type": "phase4_self_review_failed",
                                  "message": "Self-review failed; using original extraction."})
                log["steps"]["self_review"]["status"] = "failed"
        else:
            self._p("phase4 self_review skipped (not in enabled_steps)")
            log["steps"]["self_review"] = {"skipped": True}

        # step 3: per-sample verification loop
        final_samples: List[Dict[str, Any]] = []
        has_verification = "add_evidence" in enabled_set or "verify_evidence" in enabled_set

        # fast path: no verification steps enabled
        if not has_verification:
            self._p("phase4 no verification steps enabled — collecting samples directly")
            for sample_index, raw_sample in enumerate(reviewed_samples, start=1):
                if not isinstance(raw_sample, dict):
                    continue
                sample_id = raw_sample.get("sample_id")
                sample_clean = strip_confidence(raw_sample)
                conf = extract_confidence(raw_sample) if "self_review" in enabled_set else "extract_only"
                _record_sample_log(log["artifacts"]["samples"], sample_id, sample_index, {
                    "confidence": conf,
                    "skipped_verification": True,
                    "iterations": [],
                })
                final_samples.append(sample_clean)

        # full path: per-sample evidence → verify → route → repair
        else:
            ev_agent = EvidenceAdderAgent(self.extractor, verbose=self._verbose, log_callback=self.log_callback)
            ver_agent = VerifierAgent(self.extractor, verbose=self._verbose, log_callback=self.log_callback)
            rep_agent = RepairerAgent(self.extractor, verbose=self._verbose, log_callback=self.log_callback)

            for sample_index, raw_sample in enumerate(reviewed_samples, start=1):
                if not isinstance(raw_sample, dict):
                    continue

                sample_id = raw_sample.get("sample_id")
                confidence = extract_confidence(raw_sample)
                sample_clean = strip_confidence(raw_sample)

                # HIGH confidence from self-review: skip the per-sample loop
                if confidence == "HIGH" and "self_review" in enabled_set:
                    self._p(f"phase4 sample_id={sample_id} HIGH confidence — skipping verification loop")
                    sample_log: Dict[str, Any] = {
                        "confidence": "high_self_review",
                        "skipped_verification": True,
                        "iterations": [],
                    }
                    _record_sample_log(
                        log["artifacts"]["samples"], sample_id, sample_index, sample_log
                    )
                    final_samples.append(sample_clean)
                    continue

                # LOW confidence or no self-review: enter verification loop
                self._p(f"phase4 sample_id={sample_id} {confidence} confidence — entering verification loop (max_iterations={max_iterations})")

                sample_log = {"confidence": "low_confidence", "iterations": []}
                current = sample_clean
                prev_verdict: Any = None
                prev_flagged: Optional[FrozenSet[Tuple[str, str]]] = None

                for iteration in range(1, max_iterations + 1):
                    self._p(f"phase4 sample_id={sample_id} iteration={iteration}/{max_iterations}")
                    iter_log: Dict[str, Any] = {"iteration": iteration}
                    with_ev: Dict[str, Any] = current

                    # Step A: add evidence
                    if "add_evidence" in enabled_set:
                        ev_result = ev_agent.run(
                            article_id=article_id,
                            sample=current,
                            markdown=markdown,
                            prompt=prompt_add_ev,
                            prev_verdict=prev_verdict,
                            **shared,
                        )
                        iter_log["add_evidence"] = {"attempts": ev_result.attempts}

                        if not ev_result.success:
                            self._p(f"phase4 sample_id={sample_id} iteration={iteration} add_evidence failed — keeping current")
                            warnings.append({
                                "type": "phase4_add_evidence_failed",
                                "sample_id": sample_id, "iteration": iteration,
                                "message": "EvidenceAdder failed; sample kept at current state.",
                            })
                            sample_log["iterations"].append(iter_log)
                            break

                        if ev_result.warnings:
                            warnings.extend([{"sample_id": sample_id, "iteration": iteration, **w} for w in ev_result.warnings])
                            if any(w.get("type") == "phase4_illegal_edits" for w in ev_result.warnings):
                                iter_log["illegal_edits_patched"] = True

                        with_ev = ev_result.data
                    else:
                        iter_log["add_evidence"] = {"skipped": True}
                    iter_log["with_evidence"] = with_ev

                    # Step B: verify
                    if "verify_evidence" not in enabled_set:
                        iter_log["verify_evidence"] = {"skipped": True}
                        iter_log["router"] = {"action": "accept", "reason": "verify_disabled"}
                        current = with_ev
                        sample_log["iterations"].append(iter_log)
                        break

                    ver_result = ver_agent.run(
                        article_id=article_id,
                        sample_with_evidence=with_ev,
                        prompt=prompt_verify,
                        **shared,
                    )
                    iter_log["verify_evidence"] = {"attempts": ver_result.attempts}

                    if not ver_result.success:
                        self._p(f"phase4 sample_id={sample_id} iteration={iteration} verify failed — keeping current")
                        warnings.append({
                            "type": "phase4_verify_failed",
                            "sample_id": sample_id, "iteration": iteration,
                            "message": "Verifier failed; sample kept at current state.",
                        })
                        sample_log["iterations"].append(iter_log)
                        break

                    verdict = ver_result.data
                    prev_verdict = verdict
                    iter_log["verdict"] = verdict
                    flagged_count = _count_flagged(verdict)
                    flagged_keys = _get_flagged_keys(verdict)
                    iter_log["flagged_count"] = flagged_count

                    # Step C: check exit — all supported
                    if _all_supported(verdict):
                        self._p(f"phase4 sample_id={sample_id} iteration={iteration} all_supported — done")
                        sample_log["confidence"] = "verified"
                        current = with_ev
                        iter_log["router"] = {"action": "accept", "reason": "all_supported"}
                        sample_log["iterations"].append(iter_log)
                        break

                    self._p(f"phase4 sample_id={sample_id} iteration={iteration} flagged={flagged_count}")

                    # Step D: route — decide what to do
                    action, reason = _route_action(
                        verdict=verdict,
                        current_flagged=flagged_keys,
                        prev_flagged=prev_flagged,
                        iteration=iteration,
                        max_iterations=max_iterations,
                        early_stop=early_stop,
                        routing_mode=routing_mode,
                        repair_enabled=("repair_unsupported" in enabled_set),
                        re_evidence_enabled=("add_evidence" in enabled_set),
                    )
                    iter_log["router"] = {"action": action, "reason": reason}
                    prev_flagged = flagged_keys

                    # Step E: execute routed action
                    if action == "repair":
                        rep_result = rep_agent.run(
                            article_id=article_id,
                            sample_with_evidence=with_ev,
                            verdict=verdict,
                            markdown=markdown,
                            prompt=prompt_repair,
                            **shared,
                        )
                        iter_log["repair"] = {"attempts": rep_result.attempts}

                        if rep_result.warnings:
                            warnings.extend([{"sample_id": sample_id, "iteration": iteration, **w} for w in rep_result.warnings])

                        current = rep_result.data
                        iter_log["repaired"] = current
                    elif action == "filter":
                        # Remove non-SUPPORTED properties; keep composition as-is
                        filtered, n_dropped = _filter_unsupported_properties(with_ev, verdict)
                        self._p(f"phase4 sample_id={sample_id} filtered {n_dropped} unsupported properties")
                        iter_log["filter"] = {"properties_dropped": n_dropped}
                        current = filtered
                        sample_log["confidence"] = "filtered"
                        sample_log["iterations"].append(iter_log)
                        break
                    elif action == "re_evidence":
                        current = with_ev
                    else:
                        # accept
                        current = with_ev
                        if iteration >= max_iterations:
                            self._p(f"phase4 sample_id={sample_id} max_iterations reached — marking low_confidence")
                        sample_log["iterations"].append(iter_log)
                        break

                    sample_log["iterations"].append(iter_log)

                # End of loop for this sample
                sample_log["final"] = current
                _record_sample_log(
                    log["artifacts"]["samples"], sample_id, sample_index, sample_log
                )

                if sample_log["confidence"] in ("verified", "high_self_review", "filtered") or keep_low_confidence:
                    final_samples.append(current)
                else:
                    self._p(f"phase4 sample_id={sample_id} dropped (low_confidence, keep_low_confidence=False)")

        # final assembly
        verified_count = sum(
            1 for s in log["artifacts"]["samples"].values()
            if s.get("confidence") not in ("low_confidence",)
        )
        low_conf_count = len(reviewed_samples) - verified_count
        self._p(f"phase4 complete: verified={verified_count} low_confidence={low_conf_count} output_samples={len(final_samples)}")

        warnings.extend([{"step": "final", **w} for w in self._schema_warnings(final_samples)])

        return OrchestratorResult(
            output=_strip_evidence(final_samples),
            status="completed_with_warnings" if warnings else "success",
            warnings=warnings,
            log=log,
        )


    def _shared_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        raw_temp = config.get("temperature")
        return {
            "max_retries": int(config.get("max_retries", 2)),
            "max_out": int(config.get("max_output_tokens", 8192)),
            "temperature": float(raw_temp) if raw_temp is not None else None,
        }

    def _set_verbose(self, config: Dict[str, Any]) -> None:
        orch_cfg = config.get("orchestrator") or {}
        if isinstance(orch_cfg, dict):
            self._verbose = bool(orch_cfg.get("verbose", True))

    def _p(self, msg: str) -> None:
        line = f"[orchestrator] {msg}"
        if self.log_callback is not None:
            self.log_callback(line)
        elif self._verbose:
            print(line)

    def _normalize_workflow(self, workflow_val: Union[str, Dict[str, Any], None]) -> str:
        if workflow_val is None:
            return "single"
        if isinstance(workflow_val, str):
            return workflow_val.strip().lower() or "single"
        if isinstance(workflow_val, dict):
            return str(workflow_val.get("name", "single")).strip().lower() or "single"
        return "single"

    def _schema_warnings(self, data: Any) -> List[Dict[str, Any]]:
        if self.schema is None:
            return []
        try:
            v = Draft202012Validator(self.schema)
            errors = sorted(v.iter_errors(data), key=lambda e: list(e.absolute_path))
            out = []
            for e in errors:
                path = "$"
                for p in e.absolute_path:
                    path += f"[{p}]" if isinstance(p, int) else f".{p}"
                out.append({"type": "schema", "path": path, "message": e.message})
            return out
        except Exception as ex:
            return [{"type": "schema", "message": f"schema_validation_error: {str(ex)}"}]


def _strip_evidence(obj: Any) -> Any:
    """Remove 'evidence' and '_confidence' keys from output for storage."""
    if isinstance(obj, list):
        return [_strip_evidence(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _strip_evidence(v) for k, v in obj.items()
                if str(k).strip().lower() not in ("evidence", "_confidence")}
    return obj


def _record_sample_log(
    samples_log: Dict[str, Any],
    sample_id: Any,
    sample_index: int,
    sample_log: Dict[str, Any],
) -> None:
    """Record one sample trace without losing duplicate or missing IDs.

    ``sample_id`` is model-produced data and is not guaranteed to be unique.
    The extraction output intentionally preserves it as returned; only the
    provenance-log key is disambiguated here.
    """
    base_key = str(sample_id)
    key = base_key
    duplicate_index = 2
    while key in samples_log:
        key = f"{base_key}#{duplicate_index}"
        duplicate_index += 1

    sample_log["sample_id"] = sample_id
    sample_log["sample_index"] = sample_index
    samples_log[key] = sample_log


def _filter_unsupported_properties(sample: Dict[str, Any], verdict: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Remove properties with non-SUPPORTED verdicts from the sample.
    Composition fields are kept regardless (too aggressive to drop the whole sample).
    Returns (filtered_sample, number_of_properties_dropped).
    """
    if not isinstance(sample, dict) or not isinstance(verdict, dict):
        return sample, 0

    supported_keys: set = set()
    for pv in verdict.get("property_verdicts") or []:
        if not isinstance(pv, dict):
            continue
        status = str(pv.get("status", "")).strip().upper()
        key = str(pv.get("property_key", "")).strip()
        if status == "SUPPORTED" and key:
            supported_keys.add(key)

    props = sample.get("properties")
    if not isinstance(props, dict) or not props:
        return sample, 0

    filtered_props = {}
    dropped = 0
    for k, v in props.items():
        if k in supported_keys:
            filtered_props[k] = v
        else:
            dropped += 1

    out = dict(sample)
    out["properties"] = filtered_props
    return out, dropped


def _get_flagged_keys(verdict: Any) -> FrozenSet[Tuple[str, str]]:
    """Return stable set of flagged verdict keys for early-stopping and routing."""
    out: List[Tuple[str, str]] = []
    if not isinstance(verdict, dict):
        return frozenset(out)

    comp = verdict.get("composition_verdict")
    if isinstance(comp, dict):
        status = str(comp.get("status", "")).strip().upper()
        if status and status != "SUPPORTED":
            out.append(("composition_verdict", status))

    for pv in verdict.get("property_verdicts") or []:
        if not isinstance(pv, dict):
            continue
        status = str(pv.get("status", "")).strip().upper()
        if not status or status == "SUPPORTED":
            continue
        prop_key = str(pv.get("property_key") or "").strip()
        out.append((f"properties.{prop_key}", status))

    return frozenset(sorted(out))


def _route_action(
    *,
    verdict: Any,
    current_flagged: FrozenSet[Tuple[str, str]],
    prev_flagged: Optional[FrozenSet[Tuple[str, str]]],
    iteration: int,
    max_iterations: int,
    early_stop: bool,
    routing_mode: str,
    repair_enabled: bool,
    re_evidence_enabled: bool,
) -> Tuple[str, str]:
    """Decide next action for the phase4 loop: accept, repair, or re_evidence."""
    if iteration >= max_iterations:
        return "accept", "max_iterations_reached"

    if early_stop and prev_flagged is not None and current_flagged == prev_flagged:
        return "accept", "early_stop_same_flagged"

    if not repair_enabled:
        return "filter", "repair_disabled_filter_unsupported"

    if routing_mode != "rule_router":
        return "repair", "fixed_mode_default_repair"

    statuses = [status for _, status in current_flagged]
    if statuses and all(s == "NEEDS_MORE_EVIDENCE" for s in statuses) and re_evidence_enabled:
        return "re_evidence", "all_flagged_need_more_evidence"

    return "repair", "rule_router_default_repair"

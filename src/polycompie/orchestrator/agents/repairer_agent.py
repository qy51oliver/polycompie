import json
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent, AgentResult


class RepairerAgent(BaseAgent):
    """Phase 4 step 4 (per-sample): repair sample for flagged evidence; removes unsupportable property keys only."""

    def run(
        self,
        *,
        article_id: str,
        sample_with_evidence: Dict[str, Any],
        verdict: Dict[str, Any],
        markdown: str,
        prompt: str,
        max_retries: int,
        max_out: int,
        temperature: Optional[float],
    ) -> AgentResult:
        sample_id = sample_with_evidence.get("sample_id")
        self._log(f"Repairing sample_id={sample_id}.")

        sample_json = json.dumps(sample_with_evidence, ensure_ascii=False, indent=2)
        verdict_json = json.dumps(verdict, ensure_ascii=False, indent=2)

        doc = (
            f"INPUT A (sample with evidence):\n{sample_json}"
            f"\n\n---\n\n"
            f"INPUT B (verifier verdict):\n{verdict_json}"
            f"\n\n---\n\n"
            f"INPUT C (full article text):\n{markdown}"
        )

        _, repaired, attempts = self._call_and_parse(
            article_id=article_id,
            markdown=doc,
            prompt=prompt,
            max_retries=max_retries,
            max_out=max_out,
            temperature=temperature,
        )

        if repaired is None:
            self._log("Failed (no parseable JSON) — keeping pre-repair sample.")
            return AgentResult(data=sample_with_evidence, attempts=attempts, warnings=[{
                "type": "phase4_repair_parse_failed",
                "message": f"Repairer failed to produce valid JSON for sample_id={sample_id}. Kept pre-repair version.",
            }])

        if isinstance(repaired, list):
            if len(repaired) == 1 and isinstance(repaired[0], dict):
                self._log("Model returned a list — unwrapping single element.")
                repaired = repaired[0]
            else:
                self._log("Model returned unexpected list shape — keeping pre-repair sample.")
                return AgentResult(data=sample_with_evidence, attempts=attempts, warnings=[{
                    "type": "phase4_repair_bad_shape",
                    "message": f"Repairer returned list of length {len(repaired)} for sample_id={sample_id}. Kept pre-repair version.",
                }])

        enforced, guardrail_warnings = _enforce_supported_unchanged(
            pre_repair=sample_with_evidence,
            repaired=repaired,
            verdict=verdict,
        )
        return AgentResult(data=enforced, attempts=attempts, warnings=guardrail_warnings)


def _enforce_supported_unchanged(
    *,
    pre_repair: Dict[str, Any],
    repaired: Dict[str, Any],
    verdict: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Patch repaired output so verifier-SUPPORTED fields are preserved from pre_repair.
    """
    if not isinstance(pre_repair, dict) or not isinstance(repaired, dict) or not isinstance(verdict, dict):
        return repaired, []

    patched = dict(repaired)
    warnings: List[Dict[str, Any]] = []
    changed_paths: List[str] = []

    comp_verdict = verdict.get("composition_verdict") or {}
    comp_status = str(comp_verdict.get("status", "")).strip().upper()
    if comp_status == "SUPPORTED":
        for k in ("matrix_name", "fillers", "sample_thickness"):
            if k in pre_repair:
                if repaired.get(k) != pre_repair.get(k):
                    changed_paths.append(k)
                patched[k] = pre_repair.get(k)

    prop_status: Dict[str, str] = {}
    for pv in verdict.get("property_verdicts") or []:
        if not isinstance(pv, dict):
            continue
        key = str(pv.get("property_key") or "").strip()
        status = str(pv.get("status") or "").strip().upper()
        if key:
            prop_status[key] = status

    pre_props = pre_repair.get("properties") or {}
    rep_props = repaired.get("properties") or {}
    if isinstance(pre_props, dict) and isinstance(rep_props, dict):
        merged_props = dict(rep_props)
        for key, status in prop_status.items():
            if status != "SUPPORTED":
                continue
            if key in pre_props:
                if rep_props.get(key) != pre_props.get(key):
                    changed_paths.append(f"properties.{key}")
                merged_props[key] = pre_props.get(key)
        patched["properties"] = merged_props

    if changed_paths:
        warnings.append({
            "type": "phase4_repair_supported_guardrail",
            "message": f"Repairer modified {len(changed_paths)} SUPPORTED fields; restored original values.",
            "patched_paths": changed_paths[:200],
        })

    return patched, warnings
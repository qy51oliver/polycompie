import json
from typing import Any, Dict, Optional

from .base_agent import BaseAgent, AgentResult


class VerifierAgent(BaseAgent):
    """Phase 4 step 3 (per-sample): verify evidence supports claimed values. Returns verdict with composition_verdict and property_verdicts (SUPPORTED | NEEDS_MORE_EVIDENCE | CONTRADICTED)."""

    def run(
        self,
        *,
        article_id: str,
        sample_with_evidence: Dict[str, Any],
        prompt: str,
        max_retries: int,
        max_out: int,
        temperature: Optional[float],
    ) -> AgentResult:
        sample_id = sample_with_evidence.get("sample_id")
        self._log(f"Verifying evidence for sample_id={sample_id}.")

        sample_json = json.dumps(sample_with_evidence, ensure_ascii=False, indent=2)
        doc = f"INPUT (sample with evidence):\n{sample_json}"

        _, verdict, attempts = self._call_and_parse(
            article_id=article_id,
            markdown=doc,
            prompt=prompt,
            max_retries=max_retries,
            max_out=max_out,
            temperature=temperature,
        )

        if verdict is None:
            self._log("Failed (no parseable JSON).")
            return AgentResult(data=None, attempts=attempts)

        flagged_count = _count_flagged(verdict)
        self._log(f"flagged_items={flagged_count}")

        return AgentResult(data=verdict, attempts=attempts)


def _count_flagged(verdict: Any) -> int:
    """Count non-SUPPORTED items in verdict."""
    if not isinstance(verdict, dict):
        return 0

    count = 0

    comp = verdict.get("composition_verdict")
    if isinstance(comp, dict):
        if str(comp.get("status", "")).upper().strip() != "SUPPORTED":
            count += 1

    for pv in verdict.get("property_verdicts") or []:
        if not isinstance(pv, dict):
            continue
        if str(pv.get("status", "")).upper().strip() != "SUPPORTED":
            count += 1

    return count


def _all_supported(verdict: Any) -> bool:
    """True iff every verdict entry is SUPPORTED."""
    if not isinstance(verdict, dict):
        return False

    comp = verdict.get("composition_verdict")
    if not isinstance(comp, dict):
        return False
    if str(comp.get("status", "")).upper().strip() != "SUPPORTED":
        return False

    for pv in verdict.get("property_verdicts") or []:
        if not isinstance(pv, dict):
            return False
        if str(pv.get("status", "")).upper().strip() != "SUPPORTED":
            return False

    return True
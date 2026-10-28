import json
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentResult


_EVIDENCE_KEY = "evidence"


def _is_evidence_key(k: str) -> bool:
    return str(k).strip().lower() == _EVIDENCE_KEY


class EvidenceAdderAgent(BaseAgent):
    """Phase 4 step 2 (per-sample): add evidence snippets to one sample. With prev_verdict, only re-grounds NEEDS_MORE_EVIDENCE/CONTRADICTED; only 'evidence' keys may change."""

    def run(
        self,
        *,
        article_id: str,
        sample: Dict[str, Any],
        markdown: str,
        prompt: str,
        max_retries: int,
        max_out: int,
        temperature: Optional[float],
        prev_verdict: Any = None,
    ) -> AgentResult:
        iteration_type = "fresh" if prev_verdict is None else "targeted"
        self._log(f"Adding evidence for sample_id={sample.get('sample_id')} ({iteration_type}).")

        sample_json = json.dumps(sample, ensure_ascii=False, indent=2)

        if prev_verdict is not None:
            verdict_json = json.dumps(prev_verdict, ensure_ascii=False, indent=2)
            verdict_section = (
                f"\n\n---\n\n"
                f"INPUT C (verifier verdict from previous iteration):\n{verdict_json}\n"
                f"Fields marked SUPPORTED in INPUT C already have correct evidence — do NOT re-ground them.\n"
                f"Only re-ground fields marked NEEDS_MORE_EVIDENCE or CONTRADICTED."
            )
        else:
            verdict_section = ""

        doc = (
            f"INPUT A (sample JSON):\n{sample_json}"
            f"\n\n---\n\n"
            f"INPUT B (article text):\n{markdown}"
            f"{verdict_section}"
        )

        _, with_ev, attempts = self._call_and_parse(
            article_id=article_id,
            markdown=doc,
            prompt=prompt,
            max_retries=max_retries,
            max_out=max_out,
            temperature=temperature,
        )

        if with_ev is None:
            self._log("Failed (no parseable JSON).")
            return AgentResult(data=None, attempts=attempts)

        if isinstance(with_ev, list):
            if len(with_ev) == 1 and isinstance(with_ev[0], dict):
                self._log("Model returned a list — unwrapping single element.")
                with_ev = with_ev[0]
            else:
                self._log("Model returned unexpected list shape — falling back to input sample.")
                return AgentResult(data=sample, attempts=attempts, warnings=[{
                    "type": "phase4_evidence_bad_shape",
                    "message": f"EvidenceAdder returned list of length {len(with_ev)}, expected single object. Using original sample.",
                }])

        warnings: List[Dict[str, Any]] = []
        illegal = _diff_illegal_edits(sample, with_ev)
        if illegal:
            self._log(f"illegal_edits_detected={len(illegal)} (patching to evidence-only)")
            warnings.append({
                "type": "phase4_illegal_edits",
                "message": f"EvidenceAdder modified non-evidence fields; patched. count={len(illegal)}",
                "illegal_edits": illegal[:200],
            })
            with_ev = _patch_evidence_only(sample, with_ev)

        return AgentResult(data=with_ev, attempts=attempts, warnings=warnings)



def _diff_illegal_edits(baseline: Any, candidate: Any, path: str = "$") -> List[Dict[str, Any]]:
    """Diffs where candidate changed non-evidence keys vs baseline."""
    diffs: List[Dict[str, Any]] = []
    if baseline is candidate:
        return diffs
    if type(baseline) is not type(candidate):
        diffs.append({"path": path, "reason": "type_mismatch"})
        return diffs

    if isinstance(baseline, dict):
        b_keys, c_keys = set(baseline), set(candidate)
        for k in sorted(b_keys - c_keys):
            if not _is_evidence_key(k):
                diffs.append({"path": f"{path}.{k}", "reason": "missing_key_in_candidate"})
        for k in sorted(c_keys - b_keys):
            if not _is_evidence_key(k):
                diffs.append({"path": f"{path}.{k}", "reason": "new_non_evidence_key"})
        for k in sorted(b_keys & c_keys):
            if _is_evidence_key(k):
                continue
            diffs.extend(_diff_illegal_edits(baseline[k], candidate[k], path=f"{path}.{k}"))
        return diffs

    if isinstance(baseline, list):
        if len(baseline) != len(candidate):
            diffs.append({"path": path, "reason": "list_length_mismatch"})
            return diffs
        for i, (b, c) in enumerate(zip(baseline, candidate)):
            diffs.extend(_diff_illegal_edits(b, c, path=f"{path}[{i}]"))
        return diffs

    if baseline != candidate:
        diffs.append({"path": path, "reason": "value_changed", "baseline": baseline, "candidate": candidate})
    return diffs


def _patch_evidence_only(baseline: Any, candidate: Any) -> Any:
    """Merge candidate's evidence keys into baseline; leave other keys unchanged."""
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        out: Dict[str, Any] = {}
        for k, b_v in baseline.items():
            if _is_evidence_key(k):
                out[k] = candidate.get(k, b_v)
            else:
                out[k] = _patch_evidence_only(b_v, candidate.get(k, b_v))
        for k, c_v in candidate.items():
            if _is_evidence_key(k) and k not in out:
                out[k] = c_v
        return out

    if isinstance(baseline, list) and isinstance(candidate, list):
        if len(baseline) != len(candidate):
            return baseline
        return [_patch_evidence_only(b, c) for b, c in zip(baseline, candidate)]

    return baseline
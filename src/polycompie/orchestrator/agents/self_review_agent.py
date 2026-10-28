import json
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentResult


class SelfReviewAgent(BaseAgent):
    """Phase 4 step 2 (global): review extraction output for internal consistency.
    
    Receives the full extraction output and article text. Reviews for:
    - Composition errors (amounts don't match article)
    - Missing samples
    - Wrong property assignments
    
    Outputs: corrected sample list with per-sample confidence (HIGH/LOW).
    """

    def run(
        self,
        *,
        article_id: str,
        samples: List[Dict[str, Any]],
        markdown: str,
        prompt: str,
        max_retries: int,
        max_out: int,
        temperature: Optional[float],
    ) -> AgentResult:
        n_samples = len(samples) if isinstance(samples, list) else 0
        self._log(f"Self-reviewing {n_samples} extracted samples.")

        samples_json = json.dumps(samples, ensure_ascii=False, indent=2)
        doc = (
            f"INPUT A (extracted samples):\n{samples_json}"
            f"\n\n---\n\n"
            f"INPUT B (article text):\n{markdown}"
        )

        _, reviewed, attempts = self._call_and_parse(
            article_id=article_id,
            markdown=doc,
            prompt=prompt,
            max_retries=max_retries,
            max_out=max_out,
            temperature=temperature,
        )

        if reviewed is None:
            self._log("Failed (no parseable JSON) — keeping original samples with LOW confidence.")
            # Fall back: return originals, all marked LOW
            fallback = _tag_all_low(samples)
            return AgentResult(data=fallback, attempts=attempts, warnings=[{
                "type": "phase4_self_review_parse_failed",
                "message": "Self-review failed to produce valid JSON. All samples tagged LOW confidence.",
            }])

        # Handle model returning a dict with "samples" key
        if isinstance(reviewed, dict) and "samples" in reviewed:
            reviewed = reviewed["samples"]

        if not isinstance(reviewed, list):
            self._log(f"Unexpected shape ({type(reviewed).__name__}) — keeping originals with LOW confidence.")
            fallback = _tag_all_low(samples)
            return AgentResult(data=fallback, attempts=attempts, warnings=[{
                "type": "phase4_self_review_bad_shape",
                "message": f"Self-review returned {type(reviewed).__name__}, expected list. All samples tagged LOW.",
            }])

        for s in reviewed:
            if not isinstance(s, dict):
                continue
            conf = str(s.get("_confidence", "")).strip().upper()
            if conf not in ("HIGH", "LOW"):
                s["_confidence"] = "LOW"

        high_count = sum(1 for s in reviewed if isinstance(s, dict) and s.get("_confidence") == "HIGH")
        low_count = sum(1 for s in reviewed if isinstance(s, dict) and s.get("_confidence") == "LOW")
        self._log(f"Self-review complete: {high_count} HIGH, {low_count} LOW confidence samples.")

        return AgentResult(data=reviewed, attempts=attempts, warnings=[])


def _tag_all_low(samples: Any) -> List[Dict[str, Any]]:
    """Tag all samples with LOW confidence (fallback)."""
    if not isinstance(samples, list):
        return []
    out = []
    for s in samples:
        if isinstance(s, dict):
            tagged = dict(s)
            tagged["_confidence"] = "LOW"
            out.append(tagged)
    return out


def extract_confidence(sample: Dict[str, Any]) -> str:
    """Get confidence label from a self-reviewed sample."""
    return str(sample.get("_confidence", "LOW")).strip().upper()


def strip_confidence(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the _confidence field from a sample."""
    return {k: v for k, v in sample.items() if k != "_confidence"}
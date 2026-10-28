from typing import Optional

from .base_agent import BaseAgent, AgentResult


class ExtractorAgent(BaseAgent):
    """Phase 4 step 1: single-pass extraction of all samples."""

    def run(
        self,
        *,
        article_id: str,
        markdown: str,
        prompt: str,
        max_retries: int,
        max_out: int,
        temperature: Optional[float],
    ) -> AgentResult:
        self._log("Extracting samples from full article.")
        _, data, attempts = self._call_and_parse(
            article_id=article_id,
            markdown=markdown,
            prompt=prompt,
            max_retries=max_retries,
            max_out=max_out,
            temperature=temperature,
        )
        if data is None:
            self._log("Failed (no parseable JSON).")
        elif isinstance(data, dict) and "samples" in data:
            self._log("Model returned dict — unwrapping 'samples' key.")
            data = data["samples"]
        return AgentResult(data=data, attempts=attempts)
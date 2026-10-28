import json
from typing import Any, Optional

from .base_agent import BaseAgent, AgentResult


class InventoryAgent(BaseAgent):
    """Phase 3 step 1: enumerate distinct composite samples (no properties)."""

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
        self._log("Enumerating samples from full text.")
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


class AttachPropertiesAgent(BaseAgent):
    """Phase 3 step 2: attach measured properties to sample inventory."""

    def run(
        self,
        *,
        article_id: str,
        inventory: Any,
        markdown: str,
        prompt: str,
        max_retries: int,
        max_out: int,
        temperature: Optional[float],
    ) -> AgentResult:
        self._log("Attaching properties to inventory using article context.")

        inv_json = json.dumps(inventory, ensure_ascii=False, indent=2)
        step2_doc = (
            f"INPUT A (inventory JSON):\n{inv_json}\n\n---\n\nINPUT B (article text):\n{markdown}"
        )

        _, data, attempts = self._call_and_parse(
            article_id=article_id,
            markdown=step2_doc,
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
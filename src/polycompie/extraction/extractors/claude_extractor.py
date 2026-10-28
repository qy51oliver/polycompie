import os
from typing import Any, Dict, Optional

from anthropic import Anthropic

from .base_extractor import BaseExtractor


class ClaudeExtractor(BaseExtractor):
    """Direct Anthropic Messages API extractor."""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4.6",
        api_key: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        max_output_tokens: int = 8192,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            seed=seed,
        )
        key = (api_key or os.getenv("ANTHROPIC_API_KEY", "")).strip()
        if not key:
            raise ValueError("Missing required API key: ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=key)

    def _call_api(self, *, req: Dict[str, Any]) -> str:
        model = req["model"]
        messages = self._adapt_messages(req["messages"])
        temperature = req.get("temperature", self.temperature)
        max_out = req.get("max_output_tokens", self.max_output_tokens)

        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_out,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = self.client.messages.create(**kwargs)

        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text or ""
        return ""

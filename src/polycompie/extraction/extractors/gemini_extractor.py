import os
from typing import Any, Dict, Optional

from google import genai
from google.genai import types

from .base_extractor import BaseExtractor


class GeminiExtractor(BaseExtractor):
    """Direct Google GenAI extractor for Gemini models."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
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
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Missing required API key: GOOGLE_API_KEY or GEMINI_API_KEY")
        self.client = genai.Client(api_key=key.strip())

    def _call_api(self, *, req: Dict[str, Any]) -> str:
        model = req["model"]
        messages = self._adapt_messages(req["messages"])
        temperature = req.get("temperature", self.temperature)
        max_out = req.get("max_output_tokens", self.max_output_tokens)
        seed = req.get("seed", self.seed)

        contents = "\n\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )

        config_kwargs: Dict[str, Any] = {"max_output_tokens": max_out}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if seed is not None:
            config_kwargs["seed"] = seed

        resp = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return resp.text or ""

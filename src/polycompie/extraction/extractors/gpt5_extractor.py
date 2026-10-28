import os
import sys
from typing import Any, Dict, Optional

from openai import OpenAI

from .base_extractor import BaseExtractor


class GPT5Extractor(BaseExtractor):
    """OpenAI Chat Completions; routes reasoning vs chat models distinctly."""

    def __init__(
        self,
        model_name: str = "gpt-5.4-2026-03-05",
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
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        return "gpt-5" in model or model.startswith("o")

    def _call_api(self, *, req: Dict[str, Any]) -> str:
        model = req["model"]
        messages = self._adapt_messages(req["messages"])
        max_out = req.get("max_output_tokens", self.max_output_tokens)
        seed = req.get("seed", self.seed)
        reasoning_effort = req.get("reasoning_effort", self.reasoning_effort)

        api_params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if self._is_reasoning_model(model):
            # Reasoning models reject temperature, top_p, etc. — sampling is
            # internal. Tune behavior via reasoning_effort instead.
            api_params["max_completion_tokens"] = max_out
            if reasoning_effort:
                api_params["reasoning_effort"] = reasoning_effort
        else:
            api_params["max_tokens"] = max_out
            temperature = req.get("temperature", self.temperature)
            if temperature is not None:
                api_params["temperature"] = temperature

        if seed is not None:
            api_params["seed"] = seed

        resp = self.client.chat.completions.create(**api_params)

        choice = resp.choices[0]
        msg = choice.message

        if (msg is None) or (msg.content is None) or (not str(msg.content).strip()):
            fr = getattr(choice, "finish_reason", None)
            tc = getattr(msg, "tool_calls", None) if msg else None
            print(
                f"[GPT5Extractor] empty content. finish_reason={fr} "
                f"tool_calls={bool(tc)} model={model}",
                file=sys.stderr,
            )

        return (msg.content or "") if msg else ""

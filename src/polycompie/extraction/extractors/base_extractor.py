from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseExtractor(ABC):
    """LLM interface: build_messages, generate(); subclasses implement _call_api."""

    def __init__(
        self,
        model_name: str,
        temperature: Optional[float] = 0.0,
        max_output_tokens: int = 8192,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.seed = seed

    def build_messages(self, prompt: str, document: str) -> List[Dict[str, str]]:
        content = f"{prompt}\n\n---\n\n{document}"
        return [{"role": "user", "content": content}]

    def _adapt_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return messages

    def generate(
        self,
        *,
        prompt: str,
        document: str,
        article_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        messages = self.build_messages(prompt=prompt, document=document)
        req = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_output_tokens": self.max_output_tokens if max_output_tokens is None else max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "seed": self.seed,
            "article_id": article_id,
        }
        return self._call_api(req=req) or ""

    @abstractmethod
    def _call_api(self, *, req: Dict[str, Any]) -> str: ...

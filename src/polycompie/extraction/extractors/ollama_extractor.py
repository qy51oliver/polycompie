"""Ollama (local-model) extractor — talks to a local Ollama server's /api/chat.

Captures the last raw response text on the instance so the UI can offer a
"download raw output" button (open-source models often return text that's
close to but not quite valid JSON, and inspecting the raw response is the
fastest way to diagnose).
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base_extractor import BaseExtractor


_DEFAULT_BASE_URL = "http://localhost:11434"
_REQUEST_TIMEOUT_S = 600  # 10 minutes — local models can be slow
# Ollama defaults to num_ctx=4096 which silently truncates long article
# prompts. 32768 fits qwen2.5 / llama3.x natively; override via OLLAMA_NUM_CTX.
_DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))


class OllamaExtractor(BaseExtractor):
    """Direct HTTP client for a local Ollama server."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        max_output_tokens: int = 16384,
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
        self.base_url = (base_url or os.getenv("OLLAMA_HOST") or _DEFAULT_BASE_URL).rstrip("/")
        # Captured per call so the UI can expose a "Download raw output" button.
        # OSS models often return text that's close-but-not-quite valid JSON,
        # and inspecting the raw response is the fastest way to diagnose.
        self.last_raw_response: str = ""

    def _call_api(self, *, req: Dict[str, Any]) -> str:
        model = req["model"]
        messages = self._adapt_messages(req["messages"])
        temperature = req.get("temperature", self.temperature)
        max_out = req.get("max_output_tokens", self.max_output_tokens)

        options: Dict[str, Any] = {
            "num_predict": int(max_out),
            "num_ctx": _DEFAULT_NUM_CTX,
        }
        if temperature is not None:
            options["temperature"] = float(temperature)

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/api/chat"
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as response:
                resp_bytes = response.read()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama request failed at {url}: {e}") from e

        try:
            resp_data = json.loads(resp_bytes.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Ollama returned non-JSON response: {e}") from e

        text = (resp_data.get("message") or {}).get("content", "") or ""
        self.last_raw_response = text
        return text


def list_local_models(base_url: Optional[str] = None, timeout: float = 2.0) -> List[str]:
    """Return list of installed model names from Ollama's /api/tags.

    Returns an empty list on any failure (server down, network error, parse
    error). Callers can use ``len(list_local_models(...)) > 0`` as a probe
    for whether a local Ollama server is reachable AND has models pulled.
    """
    url = ((base_url or os.getenv("OLLAMA_HOST") or _DEFAULT_BASE_URL).rstrip("/")
           + "/api/tags")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError, TimeoutError):
        return []
    return [m.get("name", "") for m in (data.get("models") or []) if m.get("name")]

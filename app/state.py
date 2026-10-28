"""Per-client UI state for the PolyCompIE frontend."""

import queue
from typing import Any, Dict, List, Optional, Tuple


class State:
    """One instance per browser tab. Held in a closure inside @ui.page."""

    def __init__(self) -> None:
        # List of (filename, raw bytes) — one entry per uploaded PDF. Processed
        # sequentially by the worker thread.
        self.pdf_files: List[Tuple[str, bytes]] = []
        self.selected_props: List[str] = []
        self.samples: List[Dict[str, Any]] = []
        self.running: bool = False
        self.log_queue: queue.Queue = queue.Queue()
        self.log_lines: List[str] = []
        self.api_key: Optional[str] = None
        self.cancel_requested: bool = False
        self.strategy: str = "multi-property"
        # PDF parser backend. Docling stays the production-quality default;
        # LiteParse is an opt-in fast path for text-native documents.
        self.parser: str = "docling"
        # Model provider: "openai" (hosted GPT-5.4) or "ollama" (local model).
        # When "ollama", the agentic toggle is disabled (phase 2 only) and the
        # raw-output download button appears in the results panel.
        self.model_provider: str = "openai"
        self.ollama_model: Optional[str] = None
        # (pdf_name, raw_text) per article, populated only when provider=ollama.
        # Used by the "Download raw output" button to dump unparsed model
        # responses for diagnosis when OSS models return invalid JSON.
        self.raw_outputs: List[Tuple[str, str]] = []


def fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n/1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n/(1024**2):.1f} MB"
    return f"{n/(1024**3):.1f} GB"

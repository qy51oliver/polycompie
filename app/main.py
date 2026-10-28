"""
main.py — PolyCompIE frontend (NiceGUI). Entry point.

Run with:
    pip install -e ".[app]"
    python -m app.main

Each user supplies their own OpenAI API key via the sidebar; no environment
key is required. Hosted platforms inject PORT to override the default 8080.
"""

import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from nicegui import app, ui

load_dotenv(Path(__file__).parent / ".env")

# Import after loading local overrides because pipeline module constants read
# their environment configuration at import time.
from .page import App  # noqa: E402
from .pipeline import warmup_converter  # noqa: E402
from .styles import STYLES  # noqa: E402


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">
<rect width="40" height="40" rx="3" fill="#123D7A"/>
<path d="M7 12.5 13.5 8.5 20 12.5 26.5 8.5 33 12.5"
 fill="none" stroke="#fff" stroke-width="1.7" stroke-linecap="round"
 stroke-linejoin="round"/>
<g fill="#fff">
<circle cx="7" cy="12.5" r="1.8"/><circle cx="13.5" cy="8.5" r="1.8"/>
<circle cx="20" cy="12.5" r="1.8"/><circle cx="26.5" cy="8.5" r="1.8"/>
<circle cx="33" cy="12.5" r="1.8"/>
</g>
<path d="M8 23.5h20M8 28.5h15M8 33.5h22" fill="none" stroke="#fff"
 stroke-width="1.7" stroke-linecap="round"/>
<g fill="#9DC6E7">
<rect x="30" y="21.5" width="3.8" height="3.8" rx="0.8"/>
<rect x="25" y="26.6" width="3.8" height="3.8" rx="0.8"/>
<rect x="32" y="31.6" width="3.8" height="3.8" rx="0.8"/>
</g>
</svg>"""


@app.on_startup
def _warmup() -> None:
    if os.getenv("POLYCOMPIE_SKIP_PARSER_WARMUP", "0") == "1":
        return

    def load_default_parser() -> None:
        try:
            warmup_converter()
        except Exception as exc:
            # Keep the UI available so users can still select LiteParse or see
            # a conversion-stage error instead of losing the whole service.
            import sys

            print(f"[parser warmup warning] {type(exc).__name__}: {exc}", file=sys.stderr)

    threading.Thread(target=load_default_parser, daemon=True).start()


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe. Hosted platforms hit this every ~30s to verify the
    container is alive and route traffic away if not."""
    return {"status": "ok"}


@ui.page("/")
def index() -> None:
    ui.colors(
        primary="#123D7A",
        secondary="#1469AA",
        accent="#1469AA",
        positive="#246A4B",
        negative="#9C3C38",
        warning="#94651B",
    )
    ui.add_head_html(f"<style>{STYLES}</style>")
    App().render()


ui.run(
    title="PolyCompIE",
    favicon=FAVICON_SVG,
    host=os.getenv("HOST", "127.0.0.1"),
    port=int(os.getenv("PORT", "8080")),
    dark=False,
    reload=False,
)

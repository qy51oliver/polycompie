"""Page rendering and extraction lifecycle for the PolyCompIE frontend.

One ``App`` instance is created per browser tab. Each extraction receives an
immutable request snapshot, a private queue, and a monotonically increasing run
ID so a cancelled worker can never publish into a later run.
"""

import asyncio
import html
import json
import queue
import sys
import threading
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nicegui import ui

from .export import samples_to_excel
from .pipeline import diagnostic_for_exception, run_extraction
from .state import State, fmt_size


PROPERTY_SUGGESTIONS = (
    "shielding effectiveness",
    "electrical conductivity",
    "thermal conductivity",
    "peak heat release rate",
    "time to ignition",
    "melting temperature",
    "tensile strength",
    "elongation at break",
    "impact strength",
    "Young's modulus",
    "glass transition temperature",
    "limiting oxygen index",
)


@dataclass(frozen=True)
class RunRequest:
    """Immutable inputs for one background extraction run."""

    pdf_files: Tuple[Tuple[str, bytes], ...]
    properties: Tuple[str, ...]
    parser: str
    strategy: str
    provider: str
    ollama_model: Optional[str]
    api_key: Optional[str] = field(repr=False)


class App:
    def __init__(self) -> None:
        self.state = State()
        self.refs: Dict[str, Any] = {}
        self._run_counter = 0
        self._active_run_id: Optional[int] = None
        self._last_run_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Top-level layout
    # ------------------------------------------------------------------
    def render(self) -> None:
        ui.context.client.on_disconnect(self._on_disconnect)
        with ui.element("div").classes("app-shell"):
            self._render_topbar()
            ui.html('<h1 class="sr-only">PolyCompIE extraction workspace</h1>')
            with ui.element("main").classes("workbench-grid"):
                with ui.element("div").classes("workflow-column"):
                    self._render_upload_section()
                    self._render_properties_section()
                self._render_config_panel()
            self._render_log_panel()
            self._render_results_panel()
            self._render_footer()

    def _on_disconnect(self) -> None:
        if self.state.running:
            self.state.cancel_requested = True
            self.state.running = False
            self._active_run_id = None

    def _render_topbar(self) -> None:
        with ui.element("header").classes("topbar"):
            with ui.element("div").classes("brand-lockup"):
                ui.html(
                    '<div class="brand-mark" aria-hidden="true">'
                    '<svg class="brand-glyph" viewBox="0 0 40 40" '
                    'xmlns="http://www.w3.org/2000/svg">'
                    '<path class="polymer-bond" d="M7 12.5 13.5 8.5 20 12.5 '
                    '26.5 8.5 33 12.5" />'
                    '<circle class="polymer-node" cx="7" cy="12.5" r="1.8" />'
                    '<circle class="polymer-node" cx="13.5" cy="8.5" r="1.8" />'
                    '<circle class="polymer-node" cx="20" cy="12.5" r="1.8" />'
                    '<circle class="polymer-node" cx="26.5" cy="8.5" r="1.8" />'
                    '<circle class="polymer-node" cx="33" cy="12.5" r="1.8" />'
                    '<path class="data-row" d="M8 23.5h20M8 28.5h15M8 33.5h22" />'
                    '<rect class="data-cell" x="30" y="21.5" width="3.8" height="3.8" rx="0.8" />'
                    '<rect class="data-cell" x="25" y="26.6" width="3.8" height="3.8" rx="0.8" />'
                    '<rect class="data-cell" x="32" y="31.6" width="3.8" height="3.8" rx="0.8" />'
                    '</svg></div>'
                )
                with ui.element("div"):
                    ui.label("PolyCompIE").classes("brand-name")
                    ui.label(
                        "Structured data extraction from polymer composite literature"
                    ).classes("brand-subtitle")

    # ------------------------------------------------------------------
    # Input documents
    # ------------------------------------------------------------------
    def _render_upload_section(self) -> None:
        with ui.element("section").classes("panel workflow-panel"):
            self._section_heading(
                "1",
                "Input documents",
                "PDF · up to 10 files",
            )

            async def on_upload(e) -> None:
                if self.state.running:
                    return
                content = await e.file.read()
                name = Path(e.file.name).name
                if not content.startswith(b"%PDF-"):
                    ui.notify(
                        f"{name} does not appear to be a valid PDF.",
                        type="warning",
                        timeout=5000,
                    )
                elif any(existing.casefold() == name.casefold() for existing, _ in self.state.pdf_files):
                    ui.notify(f"{name} is already in this run.", type="warning")
                else:
                    self.state.pdf_files.append((name, content))
                    self._refresh_file_list()
                    self._refresh_run_summary()
                self.refs["upload_widget"].reset()

            def on_rejected(_e) -> None:
                ui.notify("File rejected. Upload PDFs only.", type="warning", timeout=5000)

            with ui.element("div").classes("upload-zone-wrap"):
                drop_visual = ui.element("div").classes("drop-zone")
                with drop_visual:
                    with ui.element("div").classes("drop-copy"):
                        ui.label("Drop PDFs here").classes("drop-title")
                        ui.label("or choose files from your computer").classes(
                            "drop-subtitle"
                        )
                    ui.label("Choose PDFs").classes("browse-action")
                upload_widget = ui.upload(
                    on_upload=on_upload,
                    on_rejected=on_rejected,
                    auto_upload=True,
                    multiple=True,
                    max_files=10,
                    max_file_size=50 * 1024 * 1024,
                ).props("accept=.pdf").classes("hidden-uploader")
                drop_visual.on("click", lambda: upload_widget.run_method("pickFiles"))

            with ui.element("div").classes("file-list-header") as file_list_header:
                file_count = ui.label("0 documents").classes("file-count")
                clear_files = ui.label("Clear all").classes("text-action")
                clear_files.on("click", self._clear_files)
            file_list_header.set_visibility(False)

            file_list = ui.element("div").classes("file-list")
            self.refs.update(
                {
                    "upload_widget": upload_widget,
                    "file_list_header": file_list_header,
                    "file_count": file_count,
                    "clear_files": clear_files,
                    "file_list": file_list,
                }
            )

    def _section_heading(self, number: str, title: str, description: str) -> None:
        with ui.element("div").classes("section-heading"):
            ui.label(number).classes("step-index")
            with ui.element("div"):
                ui.label(title).classes("section-title")
                ui.label(description).classes("section-description")

    def _clear_files(self) -> None:
        if self.state.running:
            return
        self.state.pdf_files.clear()
        self._refresh_file_list()
        self._refresh_run_summary()

    def _refresh_file_list(self) -> None:
        container = self.refs["file_list"]
        container.clear()
        count = len(self.state.pdf_files)
        self.refs["file_list_header"].set_visibility(count > 0)
        self.refs["file_count"].set_text(
            f"{count} document{'s' if count != 1 else ''} ready"
        )
        self.refs["clear_files"].set_visibility(count > 1)
        with container:
            for index, (name, content) in enumerate(self.state.pdf_files):
                with ui.element("div").classes("file-row"):
                    ui.html('<div class="file-type">PDF</div>')
                    with ui.element("div").classes("file-meta"):
                        ui.label(name).classes("file-name")
                        ui.label(fmt_size(len(content))).classes("file-size")

                    def remove(idx=index) -> None:
                        if self.state.running:
                            return
                        if 0 <= idx < len(self.state.pdf_files):
                            self.state.pdf_files.pop(idx)
                            self._refresh_file_list()
                            self._refresh_run_summary()

                    ui.button(icon="close", on_click=remove).props(
                        "flat round dense aria-label='Remove document'"
                    ).classes("remove-file")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def _render_properties_section(self) -> None:
        with ui.element("section").classes("panel workflow-panel"):
            self._section_heading(
                "2",
                "Properties of interest",
                "Add the measurements to capture.",
            )

            with ui.element("div").classes("property-entry"):
                prop_input = ui.input(
                    placeholder="e.g. shielding effectiveness"
                ).props("outlined dense").classes("property-input")
                add_button = ui.button("Add property", on_click=self._add_property_from_input).classes(
                    "secondary-btn"
                )
            prop_input.on("keydown.enter", self._add_property_from_input)

            ui.label("Common in polymer composite studies").classes("field-caption")
            with ui.element("div").classes("suggestion-list"):
                suggestion_buttons = []
                for suggestion in PROPERTY_SUGGESTIONS:
                    button = ui.button(
                        f"+ {suggestion}",
                        on_click=partial(self._add_property, suggestion),
                    ).props("flat no-caps").classes("suggestion-chip")
                    suggestion_buttons.append(button)

            selected_container = ui.element("div").classes("selected-properties")
            self.refs.update(
                {
                    "prop_input": prop_input,
                    "add_property_btn": add_button,
                    "suggestion_buttons": suggestion_buttons,
                    "selected_properties": selected_container,
                }
            )
            self._refresh_properties()

    def _add_property_from_input(self, _event=None) -> None:
        raw = (self.refs["prop_input"].value or "").strip()
        self._add_property(raw)
        self.refs["prop_input"].set_value("")

    def _add_property(self, raw: str, _event=None) -> None:
        if self.state.running:
            return
        value = str(raw or "").strip()
        if not value:
            return
        existing = {prop.casefold() for prop in self.state.selected_props}
        if value.casefold() in existing:
            ui.notify("That property is already selected.", type="info")
            return
        self.state.selected_props.append(value)
        self._refresh_properties()
        self._refresh_run_summary()

    def _remove_property(self, value: str, _event=None) -> None:
        if self.state.running:
            return
        if value in self.state.selected_props:
            self.state.selected_props.remove(value)
        self._refresh_properties()
        self._refresh_run_summary()

    def _refresh_properties(self) -> None:
        container = self.refs["selected_properties"]
        container.clear()
        with container:
            if not self.state.selected_props:
                ui.label("No properties selected yet.").classes("empty-selection")
            for value in self.state.selected_props:
                with ui.element("div").classes("selected-property"):
                    ui.label(value)
                    ui.label("×").classes("chip-remove").on(
                        "click", partial(self._remove_property, value)
                    )

    # ------------------------------------------------------------------
    # Run configuration
    # ------------------------------------------------------------------
    def _render_config_panel(self) -> None:
        with ui.element("aside").classes("panel config-panel"):
            self._section_heading(
                "3",
                "Run configuration",
                "Parser and extraction settings.",
            )

            self._field_label("Document parser")
            parser_toggle = ui.toggle(
                {"docling": "Docling", "liteparse": "LiteParse"},
                value=self.state.parser,
                on_change=self._on_parser_change,
            ).props("unelevated no-caps").classes("segmented-control")
            parser_help = ui.label(
                "Best for academic layouts and tables. Falls back to LiteParse if needed."
            ).classes("field-help")

            self._field_label("Model provider")
            provider_toggle = ui.toggle(
                {"openai": "OpenAI", "ollama": "Ollama"},
                value=self.state.model_provider,
                on_change=self._on_provider_change,
            ).props("unelevated no-caps").classes("segmented-control")

            openai_panel = ui.element("div").classes("provider-panel")
            with openai_panel:
                ui.label("OpenAI API key").classes("input-label")
                ui.input(
                    placeholder="sk-...",
                    password=True,
                    password_toggle_button=True,
                    on_change=self._on_api_key_change,
                ).props("outlined dense").classes("api-key-input")
                ui.label(
                    "Stored for this browser session."
                ).classes("field-help")

            ollama_panel = ui.element("div").classes("provider-panel")
            with ollama_panel:
                ollama_status = ui.label("Select Ollama to scan for local models.").classes(
                    "provider-status"
                )
                ollama_select = ui.select(
                    options=[],
                    label="Local model",
                    on_change=self._on_ollama_model_change,
                ).props("outlined dense").classes("ollama-select")
                ollama_select.set_visibility(False)
                ui.button("Re-scan local models", on_click=self._refresh_ollama).props(
                    "flat no-caps"
                ).classes("text-button")
            ollama_panel.set_visibility(False)

            self._field_label("Extraction strategy")
            strategy_toggle = ui.toggle(
                {"multi-property": "Single pass", "agentic": "Agentic pass"},
                value=self.state.strategy,
                on_change=self._on_strategy_change,
            ).props("unelevated no-caps").classes("segmented-control")
            strategy_help = ui.label(
                "Agentic pass adds evidence review and repair."
            ).classes("field-help")

            with ui.element("div").classes("run-summary"):
                ui.label("RUN SUMMARY").classes("summary-kicker")
                run_summary = ui.label("Add documents and properties to begin.").classes(
                    "run-summary-text"
                )

            extract_btn = ui.button(on_click=self.start_extraction).classes("extract-btn")
            with extract_btn:
                btn_label = ui.html('<span>Run extraction</span><span aria-hidden="true">→</span>')
            cancel_btn = ui.button("Cancel run", on_click=self.cancel_extraction).classes(
                "cancel-btn"
            )
            cancel_btn.set_visibility(False)

            self.refs.update(
                {
                    "parser_toggle": parser_toggle,
                    "parser_help": parser_help,
                    "provider_toggle": provider_toggle,
                    "openai_panel": openai_panel,
                    "ollama_panel": ollama_panel,
                    "ollama_status": ollama_status,
                    "ollama_select": ollama_select,
                    "strategy_toggle": strategy_toggle,
                    "strategy_help": strategy_help,
                    "run_summary": run_summary,
                    "btn": extract_btn,
                    "btn_label": btn_label,
                    "cancel_btn": cancel_btn,
                }
            )
            self._refresh_run_summary()

    @staticmethod
    def _field_label(text: str) -> None:
        ui.label(text).classes("field-label")

    def _on_parser_change(self, e) -> None:
        self.state.parser = e.value or "docling"
        if self.state.parser == "liteparse":
            help_text = "Faster for text PDFs. Falls back to Docling if needed."
        else:
            help_text = (
                "Best for academic layouts and tables. Falls back to LiteParse if needed."
            )
        self.refs["parser_help"].set_text(help_text)
        self._refresh_run_summary()

    def _on_provider_change(self, e) -> None:
        self.state.model_provider = e.value or "openai"
        is_ollama = self.state.model_provider == "ollama"
        self.refs["openai_panel"].set_visibility(not is_ollama)
        self.refs["ollama_panel"].set_visibility(is_ollama)
        strategy_toggle = self.refs.get("strategy_toggle")
        if strategy_toggle is not None:
            if is_ollama:
                self.state.strategy = "multi-property"
                strategy_toggle.set_value("multi-property")
                strategy_toggle.disable()
                self.refs["strategy_help"].set_text(
                    "Ollama uses the single-pass strategy in this release."
                )
            else:
                strategy_toggle.enable()
                self.refs["strategy_help"].set_text(
                    "Agentic pass adds evidence review and repair."
                )
        if is_ollama:
            self._refresh_ollama()
        self._refresh_run_summary()

    def _on_api_key_change(self, e) -> None:
        value = (e.value or "").strip()
        self.state.api_key = value or None

    def _on_ollama_model_change(self, e) -> None:
        self.state.ollama_model = e.value or None
        self._refresh_run_summary()

    def _on_strategy_change(self, e) -> None:
        self.state.strategy = e.value or "multi-property"
        self._refresh_run_summary()

    def _refresh_ollama(self) -> None:
        status = self.refs["ollama_status"]
        select = self.refs["ollama_select"]
        status.set_text("Scanning http://localhost:11434…")
        try:
            from polycompie.extraction.extractors.ollama_extractor import list_local_models

            models = list_local_models()
        except Exception:
            models = []
        if models:
            status.set_text(
                f"{len(models)} local model{'s' if len(models) != 1 else ''} available"
            )
            status.classes(replace="provider-status provider-status-success")
            select.options = list(models)
            select.set_visibility(True)
            selected = self.state.ollama_model if self.state.ollama_model in models else models[0]
            self.state.ollama_model = selected
            select.value = selected
            select.update()
        else:
            status.set_text("No running Ollama server or installed models detected.")
            status.classes(replace="provider-status provider-status-error")
            select.set_visibility(False)
            self.state.ollama_model = None
        self._refresh_run_summary()

    def _refresh_run_summary(self) -> None:
        summary = self.refs.get("run_summary")
        if summary is None:
            return
        doc_count = len(self.state.pdf_files)
        prop_count = len(self.state.selected_props)
        parser = "Docling" if self.state.parser == "docling" else "LiteParse"
        if self.state.model_provider == "ollama":
            model = self.state.ollama_model or "Ollama model not selected"
        else:
            model = "GPT-5.4"
        strategy = "Agentic pass" if self.state.strategy == "agentic" else "Single pass"
        summary.set_text(
            f"{doc_count} document{'s' if doc_count != 1 else ''} · "
            f"{prop_count} propert{'ies' if prop_count != 1 else 'y'} · "
            f"{parser} · {model} · {strategy}"
        )

    # ------------------------------------------------------------------
    # Logs and results
    # ------------------------------------------------------------------
    def _render_log_panel(self) -> None:
        log_section = ui.element("section").classes("panel output-panel log-section")
        with log_section:
            with ui.element("div").classes("output-heading"):
                with ui.element("div"):
                    ui.label("Pipeline activity").classes("section-title")
                    ui.label("Live progress.").classes(
                        "section-description"
                    )
                ui.html('<span class="live-badge"><span></span> LIVE</span>')
            log_el = ui.html("<div class='log-box'></div>")
        log_section.set_visibility(False)
        self.refs["log_section"] = log_section
        self.refs["log"] = log_el

    def _render_results_panel(self) -> None:
        results_section = ui.element("section").classes("panel output-panel results-section")
        results_section.set_visibility(False)
        self.refs["results_section"] = results_section

    def _render_footer(self) -> None:
        with ui.element("footer").classes("footer"):
            ui.label("PolyCompIE · Research software for polymer composite literature").classes(
                "footer-copy"
            )
            ui.label("Validate results against source papers.").classes(
                "footer-note"
            )

    def update_log(self) -> None:
        content = "\n".join(self.state.log_lines)
        self.refs["log"].set_content(f"<div class='log-box'>{html.escape(content)}</div>")
        ui.run_javascript(
            "var el=document.querySelector('.log-box');if(el)el.scrollTop=el.scrollHeight;"
        )

    # ------------------------------------------------------------------
    # Extraction lifecycle
    # ------------------------------------------------------------------
    def start_extraction(self) -> None:
        if self.state.running:
            return
        if self.state.model_provider == "openai" and not self.state.api_key:
            ui.notify("Enter an OpenAI API key in Run configuration.", type="warning", timeout=5000)
            return
        if self.state.model_provider == "ollama" and not self.state.ollama_model:
            ui.notify("Select a local Ollama model before starting.", type="warning", timeout=5000)
            return
        if not self.state.pdf_files:
            ui.notify("Upload at least one PDF document.", type="warning")
            return
        if not self.state.selected_props:
            ui.notify("Add at least one property of interest.", type="warning")
            return

        request = RunRequest(
            pdf_files=tuple((name, bytes(content)) for name, content in self.state.pdf_files),
            properties=tuple(self.state.selected_props),
            parser=self.state.parser,
            strategy=self.state.strategy,
            provider=self.state.model_provider,
            ollama_model=self.state.ollama_model,
            api_key=self.state.api_key,
        )
        self._run_counter += 1
        run_id = self._run_counter
        run_queue: queue.Queue = queue.Queue()
        self._active_run_id = run_id
        self.state.running = True
        self.state.cancel_requested = False
        self.state.samples = []
        self.state.raw_outputs = []
        self.state.log_lines = [
            f"Run {run_id} initialized · {len(request.pdf_files)} document(s) · "
            f"parser={request.parser}"
        ]
        self.state.log_queue = run_queue

        self.refs["btn"].disable()
        self.refs["btn_label"].set_content(
            "<span><span class='spinner-icon'>↻</span> Running pipeline</span>"
        )
        self.refs["cancel_btn"].set_visibility(True)
        self.refs["log_section"].set_visibility(True)
        self.refs["results_section"].set_visibility(False)
        self.refs["results_section"].clear()
        self._set_form_enabled(False)
        self.update_log()

        threading.Thread(
            target=self._worker,
            args=(run_id, request, run_queue),
            daemon=True,
        ).start()
        ui.timer(0.1, partial(self._poll, run_id, run_queue), once=True)

    def _set_form_enabled(self, enabled: bool) -> None:
        names = (
            "upload_widget",
            "prop_input",
            "add_property_btn",
            "parser_toggle",
            "provider_toggle",
        )
        for name in names:
            control = self.refs.get(name)
            if control is not None:
                control.enable() if enabled else control.disable()
        for control in self.refs.get("suggestion_buttons", []):
            control.enable() if enabled else control.disable()
        strategy = self.refs.get("strategy_toggle")
        if strategy is not None:
            if enabled and self.state.model_provider != "ollama":
                strategy.enable()
            else:
                strategy.disable()

    def cancel_extraction(self) -> None:
        if not self.state.running:
            return
        self.state.cancel_requested = True
        self.state.running = False
        self._active_run_id = None
        self.state.log_lines.append(
            "Cancellation requested. The active parser or model call may finish in the background, "
            "but its result will be discarded."
        )
        self.update_log()
        self._reset_run_controls()

    def _worker(self, run_id: int, request: RunRequest, run_queue: queue.Queue) -> None:
        all_samples: List[Dict[str, Any]] = []
        raw_outputs: List[Tuple[str, str]] = []
        parsers_used = set()
        successful_files = 0
        failed_files = 0
        total = len(request.pdf_files)
        final: Dict[str, Any] = {
            "success": False,
            "samples": [],
            "error": "The background worker stopped unexpectedly.",
            "cancelled": False,
        }

        try:
            for index, (name, content) in enumerate(request.pdf_files, 1):
                if self._active_run_id != run_id:
                    break
                run_queue.put(f"━━━ [{index}/{total}] {name} ━━━")
                try:
                    result = run_extraction(
                        pdf_bytes=content,
                        pdf_name=name,
                        property_list=list(request.properties),
                        log_queue=run_queue,
                        api_key=request.api_key,
                        is_cancelled=lambda rid=run_id: self._active_run_id != rid,
                        strategy=request.strategy,
                        provider=request.provider,
                        ollama_model=request.ollama_model,
                        parser=request.parser,
                    )
                    if not isinstance(result, dict):
                        raise TypeError(
                            "Pipeline returned "
                            f"{type(result).__name__}; expected a result object"
                        )
                    if result.get("success") and not isinstance(
                        result.get("samples"), list
                    ):
                        raise TypeError("Successful pipeline result did not contain a sample list")
                except Exception as error:
                    diagnostic = diagnostic_for_exception(error)
                    print(f"[worker error] {diagnostic}", file=sys.stderr)
                    run_queue.put(f"ERROR · Worker · {diagnostic}")
                    result = {
                        "success": False,
                        "samples": [],
                        "error": "Unexpected worker error. See Pipeline activity for details.",
                        "failure_stage": "worker",
                        "diagnostic": diagnostic,
                    }

                raw_text = result.get("raw_text") or ""
                if raw_text:
                    raw_outputs.append((name, str(raw_text)))
                if result.get("success"):
                    successful_files += 1
                    samples = result.get("samples") or []
                    all_samples.extend(samples)
                    parsers_used.add(str(result.get("parser") or request.parser))
                    warning_count = len(result.get("warnings") or [])
                    suffix = (
                        f" · {warning_count} validation warning(s)"
                        if warning_count
                        else ""
                    )
                    run_queue.put(
                        f"   [{index}/{total}] Complete · {len(samples)} sample"
                        f"{'s' if len(samples) != 1 else ''}{suffix}"
                    )
                else:
                    failed_files += 1
                    stage = result.get("failure_stage") or "unknown stage"
                    run_queue.put(
                        f"   [{index}/{total}] Failed at {stage} · "
                        f"{result.get('error') or 'Unknown error'}"
                    )

            cancelled = self._active_run_id != run_id
            if cancelled:
                final = {
                    "success": False,
                    "samples": all_samples,
                    "error": "Cancelled by user.",
                    "cancelled": True,
                }
            elif successful_files:
                final = {
                    "success": True,
                    "samples": all_samples,
                    "error": None,
                    "cancelled": False,
                }
            else:
                final = {
                    "success": False,
                    "samples": [],
                    "error": "No documents completed successfully. Review Pipeline activity.",
                    "cancelled": False,
                }
        except Exception as error:
            diagnostic = diagnostic_for_exception(error)
            print(f"[worker lifecycle error] {diagnostic}", file=sys.stderr)
            run_queue.put(f"ERROR · Worker lifecycle · {diagnostic}")
            final["diagnostic"] = diagnostic
        finally:
            parser_summary = (
                next(iter(parsers_used))
                if len(parsers_used) == 1
                else "mixed" if parsers_used else request.parser
            )
            final.update(
                {
                    "raw_outputs": raw_outputs,
                    "successful_files": successful_files,
                    "failed_files": failed_files,
                    "parser": parser_summary,
                    "parsers_used": sorted(parsers_used),
                }
            )
            run_queue.put(("__done__", final))

    async def _poll(self, run_id: int, run_queue: queue.Queue) -> None:
        while self._active_run_id == run_id:
            try:
                message = run_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.25)
                continue
            if isinstance(message, tuple) and message[0] == "__done__":
                self._finish(run_id, message[1])
                return
            self.state.log_lines.append(str(message))
            self.update_log()
            await asyncio.sleep(0)

    def _finish(self, run_id: int, result: Dict[str, Any]) -> None:
        if self._active_run_id != run_id:
            return
        self._active_run_id = None
        self.state.running = False
        self.state.raw_outputs = result.get("raw_outputs") or []
        self._last_run_stats = result
        self._reset_run_controls()

        if result.get("success"):
            self.state.samples = result.get("samples") or []
            count = len(self.state.samples)
            self.state.log_lines.append(
                f"Run complete · {count} sample{'s' if count != 1 else ''} extracted."
            )
            self.update_log()
            self._render_results(self.state.samples)
        else:
            error = result.get("error") or "Extraction failed."
            self.state.log_lines.append(f"Run ended · {error}")
            self.update_log()
            if not result.get("cancelled"):
                ui.notify(error, type="negative", timeout=8000)

    def _reset_run_controls(self) -> None:
        self.refs["btn"].enable()
        self.refs["btn_label"].set_content(
            '<span>Run extraction</span><span aria-hidden="true">→</span>'
        )
        self.refs["cancel_btn"].set_visibility(False)
        self._set_form_enabled(True)

    def _render_results(self, samples: List[Dict[str, Any]]) -> None:
        section = self.refs["results_section"]
        section.set_visibility(True)
        section.clear()
        with section:
            with ui.element("div").classes("output-heading"):
                with ui.element("div"):
                    ui.label("Extraction results").classes("section-title")
                    ui.label("Structured records ready for review and export.").classes(
                        "section-description"
                    )
                ui.html('<span class="complete-badge">COMPLETE</span>')

            property_count = sum(
                len(sample.get("properties") or {})
                for sample in samples
                if isinstance(sample, dict)
            )
            document_count = int(self._last_run_stats.get("successful_files", 0))
            failed_count = int(self._last_run_stats.get("failed_files", 0))
            parser_label = {
                "docling": "Docling",
                "liteparse": "LiteParse",
                "mixed": "Mixed",
            }.get(str(self._last_run_stats.get("parser")), "Unknown")
            with ui.element("div").classes("metrics-grid"):
                self._metric_card(str(len(samples)), "Samples")
                self._metric_card(str(property_count), "Property records")
                self._metric_card(str(document_count), "Documents processed")
                self._metric_card(parser_label, "Parser")

            if failed_count:
                ui.label(
                    f"{failed_count} document{'s' if failed_count != 1 else ''} did not complete. "
                    "See Pipeline activity for details."
                ).classes("partial-warning")

            if samples:
                json_str = json.dumps(samples, indent=2, ensure_ascii=False)
                with ui.expansion("Inspect structured JSON", icon="data_object").classes(
                    "json-expansion"
                ):
                    ui.html(f"<pre class='json-box'>{html.escape(json_str)}</pre>")
            else:
                json_str = "[]"
                ui.label(
                    "The pipeline completed successfully but found no matching samples for the "
                    "selected properties."
                ).classes("empty-results")

            with ui.element("div").classes("export-row"):
                ui.label("Export dataset").classes("export-label")
                with ui.element("div").classes("export-actions"):
                    ui.button(
                        "JSON",
                        icon="download",
                        on_click=lambda: ui.download(
                            json_str.encode("utf-8"),
                            filename="polycompie_extraction.json",
                            media_type="application/json",
                        ),
                    ).props("outline no-caps").classes("json-btn")

                    def do_excel() -> None:
                        try:
                            data = samples_to_excel(samples)
                            ui.download(
                                data,
                                filename="polycompie_extraction.xlsx",
                                media_type=(
                                    "application/vnd.openxmlformats-officedocument."
                                    "spreadsheetml.sheet"
                                ),
                            )
                        except Exception as exc:
                            print(
                                f"[excel export error] {type(exc).__name__}: {exc}",
                                file=sys.stderr,
                            )
                            ui.notify(
                                "Excel export failed. Use the JSON export instead.",
                                type="negative",
                            )

                    ui.button("Excel", icon="table_view", on_click=do_excel).props(
                        "no-caps"
                    ).classes("export-btn")

                    if self.state.model_provider == "ollama" and self.state.raw_outputs:

                        def do_raw() -> None:
                            parts = [
                                f"========== {name} ==========\n{text}\n"
                                for name, text in self.state.raw_outputs
                            ]
                            ui.download(
                                "\n".join(parts).encode("utf-8"),
                                filename="polycompie_raw_model_output.txt",
                                media_type="text/plain",
                            )

                        ui.button("Raw output", icon="terminal", on_click=do_raw).props(
                            "outline no-caps"
                        ).classes("json-btn")

    @staticmethod
    def _metric_card(value: str, label: str) -> None:
        with ui.element("div").classes("metric-card"):
            ui.label(value).classes("metric-value")
            ui.label(label).classes("metric-label")

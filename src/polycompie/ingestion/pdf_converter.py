"""Convert PDFs to Markdown with a selectable local parsing backend.

Docling remains the default because it is the most capable option for dense
academic layouts and tables. LiteParse is available as a faster, lightweight
alternative for documents with a reliable text layer.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..ledger.ledger_utils import LedgerManager


PARSER_CHOICES = ("docling", "liteparse")
PARSER_LABELS = {
    "docling": "Docling",
    "liteparse": "LiteParse",
}


def normalize_parser(parser: str) -> str:
    """Return a normalized parser name or raise a user-actionable error."""
    normalized = str(parser or "docling").strip().lower()
    if normalized not in PARSER_CHOICES:
        raise ValueError(
            f"Unsupported parser '{parser}'. Choose one of: {', '.join(PARSER_CHOICES)}"
        )
    return normalized


def _setup_docling(num_threads: int, do_ocr: bool) -> Any:
    """Create Docling DocumentConverter (auto device: CUDA > MPS > CPU)."""
    import torch
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        EasyOcrOptions,
        PdfPipelineOptions,
    )

    if torch.cuda.is_available():
        device = AcceleratorDevice.CUDA
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = AcceleratorDevice.MPS
        print("Using Apple MPS (Metal)")
    else:
        device = AcceleratorDevice.CPU
        print("Using CPU")

    accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=device,
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    if do_ocr:
        ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter


def _setup_liteparse(num_threads: int, do_ocr: bool) -> Any:
    """Create a LiteParse parser configured for LLM-ready Markdown output."""
    try:
        from liteparse import LiteParse
    except ImportError as exc:
        raise RuntimeError(
            "LiteParse is not installed. Install the project dependencies or run "
            "`pip install liteparse`."
        ) from exc

    return LiteParse(
        ocr_enabled=do_ocr,
        output_format="markdown",
        image_mode="placeholder",
        extract_links=True,
        keep_headers_footers=False,
        num_workers=max(1, num_threads),
        quiet=True,
    )


def setup_converter(
    num_threads: int = 4,
    do_ocr: bool = True,
    parser: str = "docling",
) -> Any:
    """Create the requested parser backend; defaults to Docling."""
    parser = normalize_parser(parser)
    if parser == "liteparse":
        return _setup_liteparse(num_threads=num_threads, do_ocr=do_ocr)
    return _setup_docling(num_threads=num_threads, do_ocr=do_ocr)


def convert_pdf(pdf_path: str, converter: Any, parser: str = "docling") -> dict:
    """Convert one PDF to Markdown.

    Returns ``{text, runtime_sec, success, error, parser}`` for either backend.
    The ``parser`` argument defaults to Docling for backwards compatibility.
    """
    start = time.perf_counter()
    parser = normalize_parser(parser)

    try:
        if parser == "liteparse":
            result = converter.parse(pdf_path)
            markdown = result.text
        else:
            result = converter.convert(pdf_path)
            markdown = result.document.export_to_markdown()

        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError(f"{PARSER_LABELS[parser]} returned an empty document")

        runtime = time.perf_counter() - start
        return {
            "text": markdown,
            "runtime_sec": round(runtime, 2),
            "success": True,
            "error": None,
            "error_type": None,
            "parser": parser,
        }

    except Exception as e:
        runtime = time.perf_counter() - start
        return {
            "text": "",
            "runtime_sec": round(runtime, 2),
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "parser": parser,
        }


def find_pdf_folders(input_dir: Path) -> list:
    """Return [(pdf_path, folder_path), ...] for each dir with a PDF.

    Prefers {folder_name}.pdf if it exists, otherwise takes the first PDF
    alphabetically.
    """
    results = []

    for folder in sorted(input_dir.iterdir()):
        if not folder.is_dir():
            continue

        pdfs = sorted(folder.glob("*.pdf"))
        if not pdfs:
            continue

        # Prefer the PDF whose stem matches the folder name
        preferred = next((p for p in pdfs if p.stem == folder.name), pdfs[0])
        results.append((preferred, folder))

    return results


def save_result(result: dict, output_dir: Path, pdf_path: Path):
    """Write .md and metrics.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = pdf_path.stem
    md_path = output_dir / f"{pdf_name}.md"
    md_path.write_text(result["text"], encoding="utf-8")

    metrics = {
        "parser": result.get("parser", "docling"),
        "runtime_sec": result["runtime_sec"],
        "char_count": len(result["text"]),
        "success": result["success"],
        "error": result["error"],
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown using Docling or LiteParse"
    )

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input directory with PDF folders"
    )
    parser.add_argument(
        "--output-subdir",
        "-o",
        type=str,
        default="outputs",
        help="Output subdirectory name (default: outputs)",
    )
    parser.add_argument(
        "--num-threads", "-t", type=int, default=4, help="CPU threads (default: 4)"
    )
    parser.add_argument(
        "--parser",
        choices=PARSER_CHOICES,
        default="docling",
        help="PDF parser backend (default: docling)",
    )
    parser.add_argument(
        "--no-ocr", action="store_true", help="Disable OCR (faster for digital PDFs)"
    )
    parser.add_argument(
        "--ledger",
        type=str,
        default=None,
        help="Optional ledger CSV path to update convert stage status",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name in ledger (defaults to input dir name)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    collection_name = args.collection or input_dir.name
    ledger_manager = LedgerManager(args.ledger) if args.ledger else None

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    pdf_folders = find_pdf_folders(input_dir)
    if not pdf_folders:
        print(f"No PDF folders found in {input_dir}")
        return 1

    try:
        print(f"{PARSER_LABELS[args.parser]} Converter")
        print("=" * 50)
        print(f"Found {len(pdf_folders)} PDFs")
        print()

        print("Setting up converter...")
        converter = setup_converter(
            num_threads=args.num_threads,
            do_ocr=not args.no_ocr,
            parser=args.parser,
        )
        print()

        success_count = 0
        total_time = 0

        for i, (pdf_path, folder_path) in enumerate(pdf_folders, 1):
            print(
                f"[{i}/{len(pdf_folders)}] {folder_path.name}...", end=" ", flush=True
            )

            result = convert_pdf(str(pdf_path), converter, parser=args.parser)

            output_dir = folder_path / args.output_subdir / args.parser
            save_result(result, output_dir, pdf_path)
            article_id = folder_path.name

            if result["success"]:
                print(f"OK {result['runtime_sec']}s, {len(result['text']):,} chars")
                success_count += 1
                if ledger_manager:
                    ledger_manager.update_entry(
                        article_id,
                        collection=collection_name,
                        convert_status="ok",
                        convert_path=str(
                            Path("outputs") / args.parser / f"{pdf_path.stem}.md"
                        ),
                    )
            else:
                print(f"FAILED {result['error'][:50]}")
                if ledger_manager:
                    ledger_manager.update_entry(
                        article_id,
                        collection=collection_name,
                        convert_status="failed",
                        convert_path=str(
                            Path("outputs") / args.parser / f"{pdf_path.stem}.md"
                        ),
                    )

            total_time += result["runtime_sec"]

        print()
        print("=" * 50)
        print(f"Done: {success_count}/{len(pdf_folders)} successful")
        print(f"Total time: {total_time:.1f}s")

    except KeyboardInterrupt:
        print("Interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

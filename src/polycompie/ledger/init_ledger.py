"""Build ledger CSV by scanning user-selected collections for PDFs and/or existing outputs."""
from pathlib import Path
from typing import List, Optional
from .schema import ArticleLedger
from .ledger_utils import LedgerManager
import argparse


def _pick_docling_md(article_dir: Path, article_id: str) -> Optional[Path]:
    candidates = [article_dir / "outputs" / "docling" / f"{article_id}.md"]
    candidates.extend(sorted((article_dir / "outputs" / "docling").glob("*.md")))
    for c in candidates:
        if c.exists():
            return c
    return None


def _pick_cleaned_md(article_dir: Path, article_id: str) -> Optional[Path]:
    candidates = [
        article_dir / "outputs" / "cleaned" / f"{article_id}.md",
        article_dir / "outputs" / "cleaned" / "paper.md",  # legacy fallback
    ]
    candidates.extend(sorted((article_dir / "outputs" / "cleaned").glob("*.md")))
    for c in candidates:
        if c.exists():
            return c
    return None


def scan_collection(collection_dir: Path, collection_name: str) -> List[ArticleLedger]:
    """Scan collection/{article_id} folders and infer pipeline statuses from existing artifacts."""
    entries = []
    article_dirs = [p for p in sorted(collection_dir.iterdir()) if p.is_dir()]

    print(f"Scanning {collection_name}... folders={len(article_dirs)}")

    for article_dir in article_dirs:
        article_id = article_dir.name
        pdfs = sorted(article_dir.glob("*.pdf"))
        pdf = next((p for p in pdfs if p.stem == article_id), pdfs[0] if pdfs else None)
        docling_md = _pick_docling_md(article_dir, article_id)
        cleaned_md = _pick_cleaned_md(article_dir, article_id)

        # Enforce PDF-first contract for all collections.
        if not pdf:
            continue

        entry = ArticleLedger(
            article_id=article_id,
            collection=collection_name,
        )

        if docling_md:
            entry.convert_status = "ok"
            entry.convert_path = str(docling_md.relative_to(article_dir))

        if cleaned_md:
            entry.clean_status = "ok"
            entry.clean_path = str(cleaned_md.relative_to(article_dir))

        entries.append(entry)

    return entries


def initialize_ledger(
    *,
    data_dir: Path,
    ledger_path: Path,
    collections: List[str],
    force: bool,
) -> None:
    """Scan selected collections and write ledger CSV (use --force to overwrite)."""
    if ledger_path.exists() and not force:
        print(f"Ledger already exists: {ledger_path}")
        print("Use --force to overwrite")
        return

    print("=" * 70)
    print("Initializing Pipeline Ledger")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Ledger path: {ledger_path}")

    all_entries = []

    for collection_name in collections:
        collection_dir = data_dir / collection_name
        if not collection_dir.exists():
            print(f"Directory not found: {collection_dir}")
            continue
        entries = scan_collection(collection_dir, collection_name)
        all_entries.extend(entries)
        print(f"Added {len(entries)} entries from {collection_name}")

    if not all_entries:
        print("No entries found. Check your directory structure and --collections value.")
        print(f"Expected: {data_dir}/<collection>/<article_id>/<article_id>.pdf")
        return

    manager = LedgerManager(str(ledger_path))
    manager.save(all_entries)

    print("=" * 70)
    print("Ledger Initialized")
    print("=" * 70)
    print(f"Total entries: {len(all_entries)}")

    convert_ok = sum(1 for e in all_entries if e.convert_status == "ok")
    convert_pending = sum(1 for e in all_entries if e.convert_status == "pending")

    clean_ok = sum(1 for e in all_entries if e.clean_status == "ok")
    clean_pending = sum(1 for e in all_entries if e.clean_status == "pending")

    print(f"Conversion: OK={convert_ok}, Pending={convert_pending}")
    print(f"Cleaning:   OK={clean_ok}, Pending={clean_pending}")
    print(f"Ledger saved to: {ledger_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Initialize pipeline ledger")

    parser.add_argument(
        "--data-dir", type=str, required=True, help="Base data directory to scan"
    )
    parser.add_argument(
        "--ledger",
        type=str,
        default="data/ledger.csv",
        help="Ledger output CSV path (default: data/ledger.csv)",
    )
    parser.add_argument(
        "--collections",
        type=str,
        required=True,
        help="Comma-separated collection folders under data-dir",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing ledger"
    )

    args = parser.parse_args()

    try:
        collections = [c.strip() for c in args.collections.split(",") if c.strip()]
        initialize_ledger(
            data_dir=Path(args.data_dir),
            ledger_path=Path(args.ledger),
            collections=collections,
            force=args.force,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
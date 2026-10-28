"""Clean converted markdown: remove parser artifacts and truncate at References."""
import re
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import argparse
from ..ledger.ledger_utils import LedgerManager


PARSER_CHOICES = ("docling", "liteparse")


class MarkdownCleaner:
    """Clean converted Markdown and return (text, stats)."""

    def __init__(self):
        self.glyph_pattern = re.compile(
            r"glyph(?:<|&lt;)[^>]*(?:>|&gt;)", re.IGNORECASE
        )
        self.image_pattern = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)
        self.formula_pattern = re.compile(
            r"<!--\s*formula-not-decoded\s*-->", re.IGNORECASE
        )

    def clean(self, raw_markdown: str) -> Tuple[str, Dict]:
        """Return (cleaned_text, stats_dict)."""
        stats = {
            "original_chars": len(raw_markdown),
            "original_lines": raw_markdown.count("\n") + 1,
            "glyphs_removed": 0,
            "glyphs_chars_removed": 0,
            "images_removed": 0,
            "images_chars_removed": 0,
            "formulas_removed": 0,
            "formulas_chars_removed": 0,
            "references_truncated": False,
            "references_chars_removed": 0,
        }

        text = raw_markdown

        glyph_matches = self.glyph_pattern.findall(text)
        stats["glyphs_removed"] = len(glyph_matches)
        stats["glyphs_chars_removed"] = sum(len(match) for match in glyph_matches)
        text = self.glyph_pattern.sub("", text)

        image_matches = self.image_pattern.findall(text)
        stats["images_removed"] = len(image_matches)
        stats["images_chars_removed"] = sum(len(match) for match in image_matches)
        text = self.image_pattern.sub("", text)

        formula_matches = self.formula_pattern.findall(text)
        stats["formulas_removed"] = len(formula_matches)
        stats["formulas_chars_removed"] = sum(len(match) for match in formula_matches)
        text = self.formula_pattern.sub("", text)

        text, truncated, ref_chars_removed = self._truncate_at_references(text)
        stats["references_truncated"] = truncated
        stats["references_chars_removed"] = ref_chars_removed

        stats["cleaned_chars"] = len(text)
        stats["cleaned_lines"] = text.count("\n") + 1
        stats["chars_removed"] = stats["original_chars"] - stats["cleaned_chars"]
        stats["reduction_percent"] = (
            stats["chars_removed"] / stats["original_chars"] * 100
            if stats["original_chars"] > 0
            else 0
        )

        return text, stats

    def _truncate_at_references(self, text: str) -> Tuple[str, bool, int]:
        """Truncate at References section and return chars removed."""
        pattern = r"\n(?:#{1,3})\s*.*?references.*$"
        parts = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)

        if len(parts) > 1:
            chars_removed = len(parts[1])
            return parts[0], True, chars_removed
        return text, False, 0


def find_all_parser_files(
    base_dir: Path,
    parser_backend: str = "docling",
) -> List[Path]:
    """Find converted Markdown files for one supported parser backend."""
    parser_backend = str(parser_backend).strip().lower()
    if parser_backend not in PARSER_CHOICES:
        raise ValueError(
            f"Unsupported parser '{parser_backend}'. "
            f"Choose one of: {', '.join(PARSER_CHOICES)}"
        )
    return sorted(base_dir.glob(f"*/outputs/{parser_backend}/*.md"))


def find_all_docling_files(base_dir: Path) -> List[Path]:
    """Find Docling Markdown files (backwards-compatible helper)."""
    return find_all_parser_files(base_dir, "docling")


def clean_all_files(
    annotated_dir: Path,
    *,
    ledger_manager: Optional[LedgerManager] = None,
    collection_name: Optional[str] = None,
    parser_backend: str = "docling",
):
    """Clean converted Markdown files from one parser backend."""
    md_files = find_all_parser_files(annotated_dir, parser_backend)

    cleaner = MarkdownCleaner()
    article_stats = {}

    for md_file in md_files:
        subfolder = md_file.parent.parent.parent
        subfolder_name = subfolder.name

        cleaned_dir = subfolder / "outputs" / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        if subfolder_name not in article_stats:
            article_stats[subfolder_name] = {
                "subfolder": subfolder_name,
                "files": [],
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "total_original_chars": 0,
                "total_cleaned_chars": 0,
                "total_references_chars_removed": 0,
            }

        output_file = cleaned_dir / md_file.name

        file_stats = {
            "input_file": str(md_file.relative_to(annotated_dir)),
            "output_file": str(output_file.relative_to(annotated_dir)),
            "filename": md_file.name,
            "status": "success",
            "error": None,
        }

        try:
            with open(md_file, "r", encoding="utf-8") as f:
                raw = f.read()

            cleaned, stats = cleaner.clean(raw)
            file_stats.update(stats)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned)

            if ledger_manager:
                ledger_manager.update_entry(
                    subfolder_name,
                    collection=collection_name,
                    clean_status="ok",
                    clean_path=str(Path("outputs") / "cleaned" / output_file.name),
                )

            article_stats[subfolder_name]["total_original_chars"] += stats[
                "original_chars"
            ]
            article_stats[subfolder_name]["total_cleaned_chars"] += stats[
                "cleaned_chars"
            ]
            article_stats[subfolder_name][
                "total_references_chars_removed"
            ] += stats.get("references_chars_removed", 0)
            article_stats[subfolder_name]["successful"] += 1

        except Exception as e:
            article_stats[subfolder_name]["failed"] += 1
            file_stats["status"] = "failed"
            file_stats["error"] = str(e)
            if ledger_manager:
                ledger_manager.update_entry(
                    subfolder_name,
                    collection=collection_name,
                    clean_status="failed",
                    clean_path=str(Path("outputs") / "cleaned" / output_file.name),
                )

        article_stats[subfolder_name]["files"].append(file_stats)
        article_stats[subfolder_name]["total_files"] += 1

    for subfolder_name, stats in article_stats.items():
        subfolder = annotated_dir / subfolder_name
        cleaned_dir = subfolder / "outputs" / "cleaned"

        total_original = stats["total_original_chars"]
        total_cleaned = stats["total_cleaned_chars"]
        total_chars_removed = total_original - total_cleaned
        total_references_chars_removed = stats.get("total_references_chars_removed", 0)
        total_reduction_percent = (
            (total_original - total_cleaned) / total_original * 100
            if total_original > 0
            else 0
        )

        print(f"{subfolder_name}:")
        print(f"  Total original chars: {total_original:,}")
        print(f"  Total cleaned chars: {total_cleaned:,}")
        print(f"  Total chars removed: {total_chars_removed:,}")
        print(f"  From references: {total_references_chars_removed:,}")
        print(f"  Total reduction: {total_reduction_percent:.2f}%")

        stats_to_save = {
            k: v
            for k, v in stats.items()
            if k
            not in [
                "total_original_chars",
                "total_cleaned_chars",
                "total_chars_removed",
                "total_reduction_percent",
                "total_references_chars_removed",
            ]
        }

        stats_file = cleaned_dir / "cleaning_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_to_save, f, indent=2)

    total_files = len(md_files)
    total_successful = sum(s["successful"] for s in article_stats.values())
    total_failed = sum(s["failed"] for s in article_stats.values())

    return {
        "total_files": total_files,
        "total_successful": total_successful,
        "total_failed": total_failed,
        "articles_processed": len(article_stats),
    }


def main():
    parser = argparse.ArgumentParser(description="Clean converted Markdown files")

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input directory with converted Markdown files",
    )
    parser.add_argument(
        "--ledger",
        type=str,
        default=None,
        help="Optional ledger CSV path to update clean stage status",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name in ledger (defaults to input dir name)",
    )
    parser.add_argument(
        "--parser",
        dest="parser_backend",
        choices=PARSER_CHOICES,
        default="docling",
        help="Parser output to clean (default: docling)",
    )

    args = parser.parse_args()

    annotated_dir = Path(args.input)
    collection_name = args.collection or annotated_dir.name
    ledger_manager = LedgerManager(args.ledger) if args.ledger else None

    if not annotated_dir.exists():
        print(f"Error: Input directory does not exist: {annotated_dir}")
        return 1

    try:
        stats = clean_all_files(
            annotated_dir,
            ledger_manager=ledger_manager,
            collection_name=collection_name,
            parser_backend=args.parser_backend,
        )
        print(
            f"Processed {stats['total_successful']}/{stats['total_files']} files "
            f"across {stats['articles_processed']} articles"
        )
        if stats["total_failed"] > 0:
            print(f"Failed: {stats['total_failed']} files")
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

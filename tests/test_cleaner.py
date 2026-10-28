import tempfile
import unittest
from pathlib import Path

from polycompie.ingestion.cleaner import clean_all_files, find_all_parser_files


class CleanerTests(unittest.TestCase):
    def test_liteparse_output_is_discovered_and_cleaned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = Path(tmpdir)
            source_dir = collection / "article-1" / "outputs" / "liteparse"
            source_dir.mkdir(parents=True)
            source = source_dir / "article-1.md"
            source.write_text(
                "# Results\nUseful content.\n<!-- image -->\n## References\nCitation",
                encoding="utf-8",
            )

            self.assertEqual(
                find_all_parser_files(collection, "liteparse"),
                [source],
            )

            stats = clean_all_files(collection, parser_backend="liteparse")
            cleaned = (
                collection
                / "article-1"
                / "outputs"
                / "cleaned"
                / "article-1.md"
            ).read_text(encoding="utf-8")

            self.assertEqual(stats["total_successful"], 1)
            self.assertIn("Useful content.", cleaned)
            self.assertNotIn("<!-- image -->", cleaned)
            self.assertNotIn("References", cleaned)


if __name__ == "__main__":
    unittest.main()

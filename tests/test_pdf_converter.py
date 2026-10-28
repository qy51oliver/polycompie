import tempfile
import unittest
from pathlib import Path

from polycompie.ingestion.pdf_converter import (
    convert_pdf,
    normalize_parser,
    save_result,
)


class _DoclingDocument:
    def export_to_markdown(self):
        return "# Docling output"


class _DoclingResult:
    document = _DoclingDocument()


class _DoclingConverter:
    def convert(self, _path):
        return _DoclingResult()


class _LiteParseResult:
    text = "# LiteParse output"


class _LiteParseConverter:
    def parse(self, _path):
        return _LiteParseResult()


class PdfConverterTests(unittest.TestCase):
    def test_docling_is_default(self):
        self.assertEqual(normalize_parser(""), "docling")
        result = convert_pdf("paper.pdf", _DoclingConverter())
        self.assertTrue(result["success"])
        self.assertEqual(result["parser"], "docling")
        self.assertEqual(result["text"], "# Docling output")

    def test_liteparse_returns_same_markdown_contract(self):
        result = convert_pdf("paper.pdf", _LiteParseConverter(), parser="liteparse")
        self.assertTrue(result["success"])
        self.assertEqual(result["parser"], "liteparse")
        self.assertEqual(result["text"], "# LiteParse output")

    def test_empty_parser_output_is_failure(self):
        converter = _LiteParseConverter()
        converter.parse = lambda _path: type("Result", (), {"text": "  "})()
        result = convert_pdf("paper.pdf", converter, parser="liteparse")
        self.assertFalse(result["success"])
        self.assertIn("empty document", result["error"])
        self.assertEqual(result["error_type"], "ValueError")

    def test_unknown_parser_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported parser"):
            normalize_parser("mystery")

    def test_saved_metrics_record_parser_provenance(self):
        result = convert_pdf("paper.pdf", _LiteParseConverter(), parser="liteparse")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            save_result(result, output_dir, Path("paper.pdf"))
            metrics = (output_dir / "metrics.json").read_text(encoding="utf-8")
            self.assertIn('"parser": "liteparse"', metrics)


if __name__ == "__main__":
    unittest.main()

import queue
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app import pipeline


class AppPipelineTests(unittest.TestCase):
    def test_parser_failure_falls_back_to_alternative_backend(self):
        logs = queue.Queue()

        def fake_convert(_path, _converter, parser):
            if parser == "docling":
                return {
                    "success": False,
                    "error": "layout model unavailable",
                    "error_type": "RuntimeError",
                }
            return {
                "success": True,
                "text": "# Parsed with LiteParse",
                "runtime_sec": 0.1,
                "parser": parser,
            }

        with patch.object(pipeline, "_get_converter", return_value=object()), patch(
            "polycompie.ingestion.pdf_converter.convert_pdf",
            side_effect=fake_convert,
        ):
            result = pipeline._parse_with_fallback(
                Path("paper.pdf"),
                "docling",
                logs.put,
                lambda: False,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["parser"], "liteparse")
        self.assertEqual(result["fallback_from"], "docling")
        log_text = "\n".join(str(item) for item in list(logs.queue))
        self.assertIn("Parser failure · Docling · RuntimeError", log_text)
        self.assertIn("Recovered · LiteParse succeeded", log_text)

    def test_both_parser_failures_are_diagnostic_and_redacted(self):
        logs = queue.Queue()

        with patch.object(
            pipeline,
            "_get_converter",
            side_effect=RuntimeError("API key=sk-supersecret123 unavailable"),
        ):
            result = pipeline._parse_with_fallback(
                Path("paper.pdf"),
                "liteparse",
                logs.put,
                lambda: False,
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "parsing")
        self.assertEqual(set(result["diagnostic"]), {"docling", "liteparse"})
        log_text = "\n".join(str(item) for item in list(logs.queue))
        self.assertIn("ERROR · Parsing", log_text)
        self.assertIn("RuntimeError", log_text)
        self.assertNotIn("sk-supersecret123", log_text)

    def test_invalid_parser_fails_before_conversion(self):
        result = pipeline.run_extraction(
            pdf_bytes=b"%PDF-test",
            pdf_name="paper.pdf",
            property_list=["thermal conductivity"],
            log_queue=queue.Queue(),
            parser="unknown",
        )
        self.assertFalse(result["success"])
        self.assertIn("Unsupported document parser", result["error"])

    def test_cancellation_interrupts_parser_queue_wait(self):
        log_queue = queue.Queue()
        with patch.object(pipeline, "_CONVERSION_SEMAPHORE", threading.Semaphore(0)):
            result = pipeline.run_extraction(
                pdf_bytes=b"%PDF-test",
                pdf_name="paper.pdf",
                property_list=["thermal conductivity"],
                log_queue=log_queue,
                parser="docling",
                is_cancelled=lambda: True,
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Cancelled by user.")

    def test_uploaded_filename_is_reduced_to_basename(self):
        observed = {}

        def fake_convert(path, _converter, parser):
            observed["path"] = path
            return {
                "success": False,
                "error": "test stop",
                "text": "",
                "runtime_sec": 0.0,
                "parser": parser,
            }

        with patch.object(pipeline, "_get_converter", return_value=object()), patch(
            "polycompie.ingestion.pdf_converter.convert_pdf", side_effect=fake_convert
        ):
            result = pipeline.run_extraction(
                pdf_bytes=b"%PDF-test",
                pdf_name="../../unsafe.pdf",
                property_list=["thermal conductivity"],
                log_queue=queue.Queue(),
            )

        self.assertFalse(result["success"])
        self.assertEqual(Path(observed["path"]).name, "unsafe.pdf")
        self.assertNotIn("..", Path(observed["path"]).parts)


if __name__ == "__main__":
    unittest.main()

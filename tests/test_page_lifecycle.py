import queue
import unittest
from unittest.mock import patch

try:
    from app import page
except ModuleNotFoundError as exc:
    if exc.name == "nicegui":
        page = None
    else:
        raise


@unittest.skipIf(page is None, "NiceGUI app extra is not installed")
class PageLifecycleTests(unittest.TestCase):
    def test_property_suggestions_cover_common_database_measurements(self):
        expected = {
            "peak heat release rate",
            "time to ignition",
            "melting temperature",
            "elongation at break",
            "limiting oxygen index",
            "impact strength",
        }

        self.assertTrue(expected.issubset(page.PROPERTY_SUGGESTIONS))

    def _request(self):
        return page.RunRequest(
            pdf_files=(("paper.pdf", b"%PDF-test"),),
            properties=("thermal conductivity",),
            parser="liteparse",
            strategy="multi-property",
            provider="openai",
            ollama_model=None,
            api_key="test-key",
        )

    @staticmethod
    def _final_from(run_queue):
        while True:
            item = run_queue.get_nowait()
            if isinstance(item, tuple) and item[0] == "__done__":
                return item[1]

    def test_successful_zero_sample_document_is_not_reported_as_failure(self):
        app = page.App()
        app._active_run_id = 1
        run_queue = queue.Queue()
        with patch.object(
            page,
            "run_extraction",
            return_value={"success": True, "samples": [], "warnings": []},
        ):
            app._worker(1, self._request(), run_queue)

        final = self._final_from(run_queue)
        self.assertTrue(final["success"])
        self.assertEqual(final["successful_files"], 1)
        self.assertEqual(final["samples"], [])

    def test_stale_worker_cannot_execute_after_new_run_starts(self):
        app = page.App()
        app._active_run_id = 2
        run_queue = queue.Queue()
        with patch.object(page, "run_extraction") as extraction:
            app._worker(1, self._request(), run_queue)

        extraction.assert_not_called()
        final = self._final_from(run_queue)
        self.assertTrue(final["cancelled"])

    def test_unexpected_pipeline_exception_finishes_with_diagnostic(self):
        app = page.App()
        app._active_run_id = 1
        run_queue = queue.Queue()
        with patch.object(
            page,
            "run_extraction",
            side_effect=RuntimeError("unexpected parser crash"),
        ):
            app._worker(1, self._request(), run_queue)

        items = list(run_queue.queue)
        final = next(item[1] for item in items if isinstance(item, tuple))
        log_text = "\n".join(str(item) for item in items if not isinstance(item, tuple))
        self.assertFalse(final["success"])
        self.assertEqual(final["failed_files"], 1)
        self.assertIn("ERROR · Worker · RuntimeError: unexpected parser crash", log_text)

    def test_actual_fallback_parser_is_reported_in_final_stats(self):
        app = page.App()
        app._active_run_id = 1
        run_queue = queue.Queue()
        with patch.object(
            page,
            "run_extraction",
            return_value={
                "success": True,
                "samples": [],
                "warnings": [],
                "parser": "docling",
            },
        ):
            app._worker(1, self._request(), run_queue)

        final = self._final_from(run_queue)
        self.assertEqual(final["parser"], "docling")
        self.assertEqual(final["parsers_used"], ["docling"])


if __name__ == "__main__":
    unittest.main()

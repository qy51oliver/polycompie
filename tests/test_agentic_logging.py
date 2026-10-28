import json
import tempfile
import unittest
from pathlib import Path

from polycompie.extraction.run_extraction import _persist_agentic_logs
from polycompie.orchestrator.orchestrator import _record_sample_log


class AgenticLoggingTests(unittest.TestCase):
    def test_duplicate_sample_ids_keep_distinct_traces(self):
        samples = {}
        _record_sample_log(samples, "sample/one", 1, {"iterations": []})
        _record_sample_log(samples, "sample/one", 2, {"iterations": []})

        self.assertEqual(list(samples), ["sample/one", "sample/one#2"])
        self.assertEqual(samples["sample/one"]["sample_index"], 1)
        self.assertEqual(samples["sample/one#2"]["sample_index"], 2)

    def test_model_sample_id_cannot_create_nested_artifact_paths(self):
        run_log = {
            "artifacts": {
                "samples": {
                    "PBT/POE (60/5/35)#1": {
                        "sample_id": "PBT/POE (60/5/35)",
                        "sample_index": 1,
                        "confidence": "verified",
                        "iterations": [
                            {
                                "iteration": 1,
                                "with_evidence": {"sample_id": "PBT/POE (60/5/35)"},
                                "verdict": {"composition_verdict": {"status": "SUPPORTED"}},
                            }
                        ],
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            written = _persist_agentic_logs(
                output_dir,
                phase=4,
                workflow_name="agentic",
                run_log=run_log,
            )

            summary = json.loads(
                (output_dir / "agentic_logs" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary[0]["sample_id"], "PBT/POE (60/5/35)")
            for path_text in written.values():
                self.assertTrue(Path(path_text).resolve().is_relative_to(output_dir.resolve()))
            sample_dirs = list((output_dir / "agentic_logs" / "samples").iterdir())
            self.assertEqual(len(sample_dirs), 1)
            self.assertNotIn("/", sample_dirs[0].name)


if __name__ == "__main__":
    unittest.main()

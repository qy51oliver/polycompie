import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from polycompie.evaluation import build_gt_synonyms
from polycompie.evaluation.normalize import PROPERTY_SET_SELECTED


class EvaluationToolTests(unittest.TestCase):
    def test_synonym_builder_passes_selected_keys_to_gt_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotated_dir = Path(tmpdir) / "annotated"
            annotated_dir.mkdir()
            out_path = Path(tmpdir) / "draft.py"
            argv = [
                "polycompie-synonyms",
                "--annotated-dir",
                str(annotated_dir),
                "--out",
                str(out_path),
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
                patch.object(build_gt_synonyms, "OpenAI", return_value=object()),
                patch.object(
                    build_gt_synonyms,
                    "scan",
                    return_value={"property_keys": {}, "n_files": 0},
                ) as scan,
            ):
                self.assertEqual(build_gt_synonyms.main(), 0)

            scan.assert_called_once_with(
                annotated_dir,
                list(PROPERTY_SET_SELECTED),
            )


if __name__ == "__main__":
    unittest.main()

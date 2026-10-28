import json
from pathlib import Path
from unittest.mock import patch

import pytest

from polycompie.evaluation.evaluate import eval_article, make_metrics, run_evaluation


EMI = ["shielding_effectiveness"]


def sample(value=20, key="shielding_effectiveness", unit="dB", thickness=None):
    result = {
        "matrix_name": [{"name": "polyethylene"}],
        "fillers": [],
        "properties": {key: {"value": value, "unit": unit}},
    }
    if thickness is not None:
        result["sample_thickness"] = {"value": thickness, "unit": "mm"}
    return result


def score(gt, pred):
    return eval_article("paper", gt, pred, EMI, None, "unused")


def test_reference_negative_predictions_are_sample_and_property_false_positives():
    row = score([sample(5, "tensile_strength", "MPa")], [sample()])
    assert not row["skipped"]
    assert (row["n_gt_samples"], row["n_pr_samples"]) == (0, 1)
    assert row["_counts"]["comp"] == {
        "tp_s": 0, "fp_s": 1, "fn_s": 0, "tp_r": 0, "fp_r": 1, "fn_r": 0,
    }
    for suffix in ("", "_s"):
        counts = row["_counts"]["prop"]
        assert [counts[k + suffix] for k in ("tp", "fp", "fn")] == [0, 1, 0]


def test_empty_reference_and_prediction_are_scored_with_zero_denominators():
    row = score([], [])
    assert not row["skipped"]
    assert (row["n_gt_samples"], row["n_pr_samples"]) == (0, 0)
    metrics = make_metrics(row["_counts"])
    for component in ("composition", "property"):
        for gate in ("strict", "relaxed"):
            assert all(metrics[component][gate][m] == 0 for m in ("precision", "recall", "f1"))


def test_empty_prediction_counts_all_reference_samples_and_properties_as_missed():
    row = score([sample(), sample(value=30, thickness=2)], [])
    assert not row["skipped"]
    assert (row["n_gt_samples"], row["n_pr_samples"]) == (2, 0)
    assert row["_counts"]["comp"]["fn_r"] == row["_counts"]["comp"]["fn_s"] == 2
    assert row["_counts"]["prop"]["fn"] == row["_counts"]["prop"]["fn_s"] == 2


def test_property_relevance_filter_applies_only_to_ground_truth():
    out_of_scope = sample(5, "tensile_strength", "MPa", thickness=2)
    row = score([sample(), out_of_scope], [sample(), out_of_scope])
    assert (row["n_gt_samples"], row["n_pr_samples"]) == (1, 2)
    assert row["_counts"]["comp"]["tp_r"] == 1
    assert row["_counts"]["comp"]["fp_r"] == 1
    assert row["_counts"]["prop"]["tp"] == 1
    assert row["_counts"]["prop"]["fp"] == 0


def test_reference_negative_still_counts_prediction_without_target_properties():
    row = score([], [sample(5, "tensile_strength", "MPa")])
    assert row["n_gt_samples"] == 0
    assert row["n_pr_samples"] == 1
    assert row["_counts"]["comp"]["fp_s"] == row["_counts"]["comp"]["fp_r"] == 1
    assert row["_counts"]["prop"]["fp"] == 0


def write_article(annotations, predictions, article_id, gt, pred=None):
    folder = annotations / article_id
    folder.mkdir(parents=True)
    (folder / "ground_truth.txt").write_text(json.dumps(gt))
    if pred is not None:
        folder = predictions / article_id / "phase2" / "model"
        folder.mkdir(parents=True)
        (folder / "output.json").write_text(json.dumps(pred))


def test_all_59_articles_include_missing_empty_predictions_and_reference_negatives(tmp_path: Path):
    annotations, predictions = tmp_path / "annotations", tmp_path / "outputs"
    predictions.mkdir()
    for i in range(59):
        gt = [sample()] if i < 56 else []
        pred = [] if i == 0 else [sample()] if i == 58 else None
        write_article(annotations, predictions, str(i), gt, pred)
    run_evaluation(annotations, predictions, 2, "model", EMI, tmp_path / "scores", False, "unused")
    summary = json.loads((tmp_path / "scores/summary_table.json").read_text())
    details = json.loads((tmp_path / "scores/details.json").read_text())
    assert summary["evaluated_articles"] == 59
    assert summary["reference_positive_articles"] == 56
    assert summary["reference_negative_articles"] == 3
    assert summary["missing_prediction_articles"] == 57
    assert summary["empty_prediction_articles"] == 1
    assert summary["skipped_articles"] == 0
    assert len(details) == 59
    assert summary["property_matching"]["relaxed"]["fn"] == 56
    assert summary["property_matching"]["relaxed"]["fp"] == 1
    assert all(not row["skipped"] for row in details)
    assert {r["prediction_status"] for r in details} == {"missing", "empty", "present"}


@pytest.mark.parametrize("bad_prediction", ['{"samples":', '{"samples":null}', '[null]', '42'])
def test_invalid_predictions_raise_an_explicit_article_error(tmp_path, bad_prediction):
    annotations, predictions = tmp_path / "annotations", tmp_path / "outputs"
    write_article(annotations, predictions, "broken-paper", [sample()], [])
    (predictions / "broken-paper/phase2/model/output.json").write_text(bad_prediction)
    with pytest.raises(ValueError, match="broken-paper: prediction"):
        run_evaluation(annotations, predictions, 2, "model", EMI, tmp_path / "scores", False, "unused")


def test_frozen_property_mappings_and_material_judgments_bypass_resolvers(tmp_path):
    annotations, predictions = tmp_path / "annotations", tmp_path / "outputs"
    pred = sample(key="a_custom_property")
    pred["matrix_name"][0]["name"] = "PE"
    write_article(annotations, predictions, "paper", [sample()], [pred])
    with patch("polycompie.evaluation.evaluate._resolve_pr_key_synonyms", side_effect=AssertionError), \
         patch("polycompie.evaluation.evaluate.OpenAI", side_effect=AssertionError):
        run_evaluation(
            annotations, predictions, 2, "model", EMI, tmp_path / "scores", True, "unused",
            pr_extra_synonyms={"a_custom_property": "shielding_effectiveness"},
            material_judgments={("polyethylene", "PE", "polymer"): True},
        )
    summary = json.loads((tmp_path / "scores/summary_table.json").read_text())
    assert summary["property_matching"]["relaxed"]["tp"] == 1
    assert summary["composition_matching"]["strict"]["f1"] == 1

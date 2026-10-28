"""Regressions for the original scoring method and narrowly scoped fixes."""

import unittest
from unittest.mock import patch

from polycompie.evaluation.matching import (
    _collect_name_pairs,
    extract_measurements,
    filler_amounts_match,
    greedy_match,
    measurements_equal,
    merge_samples,
    sample_similarity,
)


def sample(matrices=(), fillers=(), properties=None):
    return {
        "matrix_name": [{"name": name} for name in matrices],
        "fillers": [
            {"name": name, "composition": {"amount": 10, "type": "wt%"}}
            for name in fillers
        ],
        "properties": properties or {},
    }


class MinimalMatchingTests(unittest.TestCase):
    def test_similarity_retains_ground_truth_denominators(self):
        gt = sample(["PE"], ["carbon"])
        prediction = sample(["PE", "extra polymer"], ["carbon", "extra filler"])
        judgments = {pair: pair[0] == pair[1] for pair in _collect_name_pairs(gt, prediction)}
        score, details = sample_similarity(gt, prediction, "paper", None, "unused", judgments)
        self.assertEqual(score, 1.0)
        self.assertEqual(details["matrix_score"], 1.0)
        self.assertEqual(details["filler_score"], 1.0)

    def test_constituent_matching_remains_greedy(self):
        gt = sample(["A", "B"], ["A", "B"])
        prediction = sample(["X", "Y"], ["X", "Y"])
        # A can use either prediction; B can use only X. Greedy matching
        # consumes X for A, although a maximum matching could match both.
        judgments = {
            pair: pair[:2] in {("A", "X"), ("A", "Y"), ("B", "X")}
            for pair in _collect_name_pairs(gt, prediction)
        }
        score, details = sample_similarity(gt, prediction, "paper", None, "unused", judgments)
        self.assertEqual(details["matrix_score"], 0.5)
        self.assertEqual(details["filler_score"], 0.5)
        self.assertAlmostEqual(score, 0.6)

    def test_property_matching_remains_greedy(self):
        gt = [("tensile_strength", value, "mpa") for value in (0, 1.5e-6)]
        predictions = [("tensile_strength", value, "mpa") for value in (0.75e-6, -0.75e-6)]
        self.assertEqual(greedy_match(gt, predictions), (
            1, 1, 1,
            {"tensile_strength": 1},
            {"tensile_strength": 1},
            {"tensile_strength": 1},
        ))

    def test_explicit_judgments_bypass_cache_and_judge(self):
        gt, prediction = sample(["PE"]), sample(["polyethylene"])
        with patch("polycompie.evaluation.matching._get_matches", side_effect=AssertionError):
            score, _ = sample_similarity(
                gt, prediction, "paper", object(), "unused",
                {("PE", "polyethylene", "polymer"): True},
            )
        self.assertEqual(score, 1.0)

    def test_explicit_false_is_not_overridden_by_exact_match(self):
        record = sample(["PE"])
        score, details = sample_similarity(
            record, record, "paper", None, "unused", {("PE", "PE", "polymer"): False},
        )
        self.assertEqual(details["matrix_score"], 0.0)
        self.assertAlmostEqual(score, 0.6)

    def test_incomplete_judgments_raise_without_fallback(self):
        record = sample(["PE"])
        with patch("polycompie.evaluation.matching._get_matches", side_effect=AssertionError):
            with self.assertRaises(KeyError):
                sample_similarity(record, record, "paper", object(), "unused", {})

    def test_default_normalized_exact_matching_is_unchanged(self):
        with patch("polycompie.evaluation.matching._pair_cache", {}):
            score, _ = sample_similarity(sample([" PE "]), sample(["pe"]), "paper", None, "unused")
        self.assertEqual(score, 1.0)

    def test_merge_retains_original_composition_grouping(self):
        records = []
        for index, amount in enumerate((20, 80)):
            record = sample(["PE"], properties={"tensile_strength": {"value": amount, "unit": "MPa"}})
            record["sample_id"] = f"formulation-{index}"
            record["matrix_name"][0]["composition"] = {"amount": amount, "type": "wt%"}
            record["matrix_name"][0]["description"] = f"treatment-{index}"
            records.append(record)
        merged, audit = merge_samples(records, "gt")
        self.assertEqual(len(merged), 1)
        self.assertEqual(audit[0]["n_originals"], 2)
        self.assertEqual(set(audit[0]["composition_key"]), {"matrices", "fillers", "thickness"})

    def test_merge_suffixes_preserve_every_entry_and_metadata(self):
        keys = [
            "melting temperature 1", "melting temperature 1",
            "melting_temperature_1", "melting_temperature_2",
            "melting_temperature", "melting_temperature",
        ]
        values = [
            {"value": i, "unit": "°C", "conditions": [f"run-{i}"], "evidence": f"source-{i}"}
            for i in range(100, 106)
        ]
        records = [sample(properties={key: value}) for key, value in zip(keys, values)]
        for ordered in (records, list(reversed(records))):
            with self.subTest(order=[next(iter(record["properties"])) for record in ordered]):
                merged, _ = merge_samples(ordered, "gt")
                properties = merged[0]["properties"]
                self.assertEqual(len(properties), len(values))
                self.assertEqual(sorted(properties.values(), key=lambda value: value["value"]), values)
                measurements = extract_measurements(properties, ["melting_temperature"])
                self.assertEqual(sorted(value for _, value, _ in measurements), list(range(100, 106)))

    def test_filler_tolerance_is_absolute(self):
        def filler(amount):
            return {"composition": {"amount": amount, "type": "wt%"}}

        self.assertTrue(filler_amounts_match(filler(10), filler(10 + 0.5e-6)))
        self.assertFalse(filler_amounts_match(filler(10), filler(10 + 2e-6)))

    def test_property_tolerance_is_absolute_with_and_without_conversion(self):
        key = "tensile_strength"
        self.assertTrue(measurements_equal((key, 100, "mpa"), (key, 100 + 0.5e-6, "mpa")))
        self.assertFalse(measurements_equal((key, 100, "mpa"), (key, 100 + 2e-6, "mpa")))
        self.assertTrue(measurements_equal((key, 100, "mpa"), (key, 0.1, "gpa")))
        self.assertFalse(measurements_equal((key, 100, "mpa"), (key, 0.100000002, "gpa")))

    def test_conductivity_uses_original_absolute_tolerance(self):
        key = "electrical_conductivity"
        self.assertTrue(measurements_equal((key, 1e-13, "s/m"), (key, 1e-12, "s/m")))


if __name__ == "__main__":
    unittest.main()

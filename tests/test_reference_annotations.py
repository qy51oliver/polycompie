import json
import re
from pathlib import Path

import pytest

from polycompie.evaluation.ground_truth import parse_gt


ANNOTATIONS = Path(__file__).resolve().parents[1] / "annotations"


def parse_text(tmp_path, text):
    path = tmp_path / "ground_truth.txt"
    path.write_text(text)
    before = path.read_bytes()
    data, error = parse_gt(path)
    assert error is None
    assert path.read_bytes() == before
    return data


@pytest.mark.parametrize("wrapped", [False, True])
def test_repeated_measurements_keep_source_order_and_reserved_keys(tmp_path, wrapped):
    sample = '{"properties":{"SE":52,"SE_1":80,"SE":70}}'
    document = f'{{"samples":[{sample}]}}' if wrapped else f"[{sample}]"
    data = parse_text(tmp_path, document)
    samples = data["samples"] if wrapped else data
    assert list(samples[0]["properties"].items()) == [("SE_2", 52), ("SE_1", 80), ("SE_3", 70)]


def test_numbered_repeats_keep_the_property_name(tmp_path):
    data = parse_text(tmp_path, '[{"properties":{"tensile_strength_1":12,'
                      '"tensile_strength_1":15,"tensile_strength_2":20,'
                      '"elongation_30pct":2,"elongation_30pct":3}}]')
    assert list(data[0]["properties"].items()) == [
        ("tensile_strength_1", 12), ("tensile_strength_3", 15),
        ("tensile_strength_2", 20), ("elongation_30pct_1", 2),
        ("elongation_30pct_2", 3),
    ]


def test_suffix_allocation_does_not_collide_with_later_explicit_keys(tmp_path):
    data = parse_text(tmp_path, '[{"properties":{"SE":52,"SE":70,"SE_1":80,"SE_2":90}}]')
    assert list(data[0]["properties"].items()) == [("SE_3", 52), ("SE_4", 70), ("SE_1", 80), ("SE_2", 90)]


def test_duplicate_keys_outside_property_observations_are_rejected(tmp_path):
    path = tmp_path / "ground_truth.txt"
    path.write_text('[{"sample_id":"a","sample_id":"b","properties":{}}]')
    data, error = parse_gt(path)
    assert data is None
    assert "Repeated JSON key" in error


def test_historical_json_cleanup_preserves_measurement_values(tmp_path):
    data = parse_text(tmp_path, '[{"sample_id":"p\\t1","properties":{"SE":{"value":20,"unit":"dB",},},},]')
    assert data[0]["sample_id"] == "p\t1"
    assert data[0]["properties"]["SE"] == {"value": 20, "unit": "dB"}


def test_all_reference_files_parse_without_changing_source_bytes():
    files = sorted(ANNOTATIONS.glob("*/ground_truth.txt"))
    assert len(files) == 59
    for path in files:
        original_bytes = path.read_bytes()
        data, error = parse_gt(path)
        assert error is None, (path, error)
        assert isinstance(data, (list, dict))
        assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("article", [
    "s10973-016-5484-9",
    "1-s2.0-S0266353818314507-main",
    "1-s2.0-S0266353823004505-main",
    "1-s2.0-S0021979723011311-main",
])
def test_article_names_do_not_trigger_reference_corrections(article, tmp_path):
    path = ANNOTATIONS / article / "ground_truth.txt"
    raw = path.read_text()
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    cleaned = re.sub(r",(\s*[\]\}])", r"\1", cleaned)
    original_samples = json.loads(cleaned, object_pairs_hook=tuple)
    parsed, error = parse_gt(path)
    assert error is None
    def unpack(value):
        if isinstance(value, tuple):
            return {key: unpack(item) for key, item in value}
        if isinstance(value, list):
            return [unpack(item) for item in value]
        return value

    assert len(parsed) == len(original_samples)
    for source, actual in zip(original_samples, parsed):
        source_fields = dict(source)
        source_properties = source_fields.pop("properties", ())
        actual_fields = {key: value for key, value in actual.items() if key != "properties"}
        assert actual_fields == {key: unpack(value) for key, value in source_fields.items()}
        assert list(actual.get("properties", {}).values()) == [
            unpack(value) for _, value in source_properties
        ]
    copied_path = tmp_path / article / "ground_truth.txt"
    copied_path.parent.mkdir()
    copied_path.write_bytes(path.read_bytes())
    copied, error = parse_gt(copied_path)
    assert error is None
    assert copied == parsed

import importlib
import json
import unittest
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from polycompie.evaluation.normalize import PROPERTY_SET_SELECTED
from polycompie.extraction.prompts.prompt_manager import PromptManager
from polycompie.extraction.runtime import load_schema, make_extractor


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EVALUATION_PROPERTIES = {
    "electrical_conductivity",
    "elongation_at_break",
    "flammability",
    "glass_transition_temperature",
    "impact_strength",
    "limiting_oxygen_index",
    "melting_temperature",
    "peak_heat_release_rate",
    "shielding_effectiveness",
    "tensile_strength",
    "thermal_conductivity",
    "time_to_ignition",
    "youngs_modulus",
}


class PipelineContractTests(unittest.TestCase):
    def test_runtime_import_does_not_require_unused_provider_sdks(self):
        module = importlib.import_module("polycompie.extraction.runtime")
        self.assertIs(module.make_extractor, make_extractor)
        self.assertEqual(load_schema().get("type"), "array")

    def test_all_configured_model_phase_pairs_resolve(self):
        manager = PromptManager()
        configured = {
            1: ("gpt-5.2", "gpt-5.4", "claude-sonnet-4.6", "gemini-2.5-pro"),
            2: (
                "gpt-5.2",
                "gpt-5.4",
                "claude-sonnet-4.6",
                "gemini-2.5-pro",
                "ollama",
            ),
            3: ("gpt-5.2", "gpt-5.4", "claude-sonnet-4.6", "gemini-2.5-pro"),
            4: ("gpt-5.2", "gpt-5.4", "claude-sonnet-4.6", "gemini-2.5-pro"),
        }
        for phase, model_families in configured.items():
            for model_family in model_families:
                with self.subTest(phase=phase, model_family=model_family):
                    config = manager.get_phase_config(
                        phase=phase,
                        model_family=model_family,
                        property_list=["thermal conductivity"],
                    )
                    self.assertTrue(config["model_string"])
                    self.assertGreater(config["max_output_tokens"], 0)
                    if config["workflow"] == "single":
                        self.assertTrue(config["prompt"])
                        self.assertNotIn("{PROPERTY_LIST}", config["prompt"])
                    else:
                        self.assertTrue(config["agentic_steps"])
                        self.assertNotIn(
                            "{PROPERTY_LIST}",
                            "\n".join(config["agentic_steps"].values()),
                        )

    def test_extraction_schema_is_valid_draft_2020_12(self):
        schema_path = ROOT / "src/polycompie/extraction/schemas/extraction_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        self.assertEqual(schema.get("type"), "array")

    def test_fixed_thirteen_property_contract_is_shared_by_prompt_and_eval(self):
        manager = PromptManager()
        prompt_properties = manager.raw["property_sets"]["selected"]

        self.assertEqual(len(PROPERTY_SET_SELECTED), 13)
        self.assertEqual(len(prompt_properties), 13)
        self.assertEqual(set(PROPERTY_SET_SELECTED), EXPECTED_EVALUATION_PROPERTIES)
        self.assertEqual(set(prompt_properties), EXPECTED_EVALUATION_PROPERTIES)

    def test_every_ablation_variant_has_orchestrator_config(self):
        path = ROOT / "src/polycompie/extraction/prompts/variants.yaml"
        variants = yaml.safe_load(path.read_text(encoding="utf-8"))["variants"]
        self.assertEqual(set(variants), {f"variant_{letter}" for letter in "ABCDE"})
        for name, config in variants.items():
            with self.subTest(variant=name):
                self.assertIsInstance(config.get("orchestrator"), dict)


if __name__ == "__main__":
    unittest.main()

import copy
from typing import Any, Dict, List, Optional
from pathlib import Path

import yaml


class PromptManager:
    """Load prompts.yaml and return phase+model config."""

    def __init__(self, yaml_path: Optional[str] = None):
        self.yaml_path = yaml_path or str(self._default_yaml_path())
        self.raw = self._load_yaml(self.yaml_path)

    def _default_yaml_path(self) -> Path:
        return Path(__file__).resolve().parent / "prompts.yaml"

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Prompt YAML not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Prompt YAML must parse to a dict, got {type(data)}")
        return data

    def get_phase_config(
        self,
        *,
        phase: int,
        model_family: str,
        property_list: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return the full config dict for a (phase, model_family) combination."""
        phase_key = f"phase{phase}"
        if phase_key not in self.raw:
            raise KeyError(f"YAML missing key '{phase_key}'")

        phase_block = self.raw[phase_key]
        if not isinstance(phase_block, dict):
            raise ValueError(f"{phase_key} must be a dict, got {type(phase_block)}")

        workflow = str(phase_block.get("workflow", "single")).strip().lower()

        if property_list is None:
            property_list = self._resolve_property_set(phase_block)

        steps = phase_block.get("agentic_steps", {}) or {}
        if not isinstance(steps, dict):
            raise ValueError(f"{phase_key}.agentic_steps must be a dict")
        steps_filled = self._fill_steps(steps, property_list=property_list)

        models = phase_block.get("models", {}) or {}
        if not isinstance(models, dict):
            raise ValueError(f"{phase_key}.models must be a dict")

        requested_family = str(model_family or "").strip().lower()
        mf_cfg = models.get(requested_family, {}) or {}
        if not isinstance(mf_cfg, dict) or not mf_cfg:
            raise ValueError(
                f"{phase_key}.models.{requested_family} not found. "
                f"Available: {sorted(models.keys())}"
            )

        model_string = str(mf_cfg.get("model_string", "")).strip()
        if not model_string:
            raise ValueError(f"{phase_key}.models.{requested_family}.model_string is missing/empty")

        defaults = self.raw.get("defaults", {}) or {}
        max_output_tokens = int(mf_cfg.get("max_output_tokens", defaults.get("max_output_tokens", 16384)))
        max_retries = int(mf_cfg.get("max_retries", defaults.get("max_retries", 2)))

        # Temperature is opt-in: only set when the model entry declares it.
        # Reasoning models (gpt-5, o-series) must not receive temperature at all.
        raw_temp = mf_cfg.get("temperature", None)
        temperature = float(raw_temp) if raw_temp is not None else None

        raw_effort = mf_cfg.get("reasoning_effort", None)
        reasoning_effort = str(raw_effort).strip() if raw_effort else None

        orchestrator_cfg = mf_cfg.get("orchestrator", {}) or {}
        if not isinstance(orchestrator_cfg, dict):
            orchestrator_cfg = {}

        prompt = ""
        if workflow == "single":
            prompt = str(phase_block.get("prompt", "") or "").strip()
            if "{PROPERTY_LIST}" in prompt:
                prompt = prompt.replace("{PROPERTY_LIST}", self._format_property_list(property_list))

        return {
            "workflow": workflow,
            "agentic_steps": steps_filled,
            "prompt": prompt,
            "model_string": model_string,
            "max_output_tokens": max_output_tokens,
            "max_retries": max_retries,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "orchestrator": copy.deepcopy(orchestrator_cfg),
        }

    def _resolve_property_set(self, phase_block: Dict[str, Any]) -> Optional[List[str]]:
        ps_name = str(phase_block.get("property_set", "") or "").strip()
        if not ps_name:
            return None
        prop_sets = self.raw.get("property_sets", {}) or {}
        if not isinstance(prop_sets, dict):
            return None
        props = prop_sets.get(ps_name, None)
        if not isinstance(props, list):
            return None
        return [str(p) for p in props]

    def _fill_steps(self, steps: Dict[str, Any], *, property_list: Optional[List[str]]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        prop_str = self._format_property_list(property_list)

        for k, v in steps.items():
            if v is None:
                out[str(k)] = ""
                continue
            text = str(v)
            if "{PROPERTY_LIST}" in text:
                text = text.replace("{PROPERTY_LIST}", prop_str)
            out[str(k)] = text
        return out

    def _format_property_list(self, property_list: Optional[List[str]]) -> str:
        if not property_list:
            return ""
        return "\n".join([f"- {p}" for p in property_list])
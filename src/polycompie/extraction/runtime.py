"""Public runtime helpers shared by batch and interactive extraction.

Provider construction and packaged configuration loading live here so callers
do not need to reach into the command-line implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_schema(extraction_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load the packaged extraction schema when it is available."""
    base_dir = extraction_dir or Path(__file__).resolve().parent
    schema_path = base_dir / "schemas" / "extraction_schema.json"
    if not schema_path.exists():
        return None
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_variant_config(variant: str) -> Dict[str, Any]:
    """Load one named phase-4 ablation configuration."""
    variants_path = Path(__file__).resolve().parent / "prompts" / "variants.yaml"
    if not variants_path.exists():
        raise FileNotFoundError(f"variants.yaml not found: {variants_path}")

    data = yaml.safe_load(variants_path.read_text(encoding="utf-8")) or {}
    variants = data.get("variants", {})
    if variant not in variants:
        raise ValueError(
            f"Variant '{variant}' not found in variants.yaml. "
            f"Available: {sorted(variants.keys())}"
        )
    return variants[variant]


def apply_variant_config(config: Dict[str, Any], variant: str) -> Dict[str, Any]:
    """Apply a variant's orchestrator settings using the existing merge rule."""
    variant_cfg = load_variant_config(variant)
    orchestrator = config.get("orchestrator", {}) or {}
    orchestrator.update(variant_cfg.get("orchestrator", {}) or {})
    config["orchestrator"] = orchestrator
    return variant_cfg


def make_extractor(
    model_family: str,
    model_string: str,
    temperature: Optional[float],
    max_out: int,
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    seed: Optional[int] = None,
) -> Any:
    """Instantiate the provider adapter for a configured model family."""
    if model_family in ("gpt-5.2", "gpt-5.4"):
        from .extractors.gpt5_extractor import GPT5Extractor

        return GPT5Extractor(
            model_name=model_string,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_out,
            reasoning_effort=reasoning_effort,
            seed=seed,
        )
    if model_family == "claude-sonnet-4.6":
        from .extractors.claude_extractor import ClaudeExtractor

        return ClaudeExtractor(
            model_name=model_string,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_out,
            reasoning_effort=reasoning_effort,
            seed=seed,
        )
    if model_family == "gemini-2.5-pro":
        from .extractors.gemini_extractor import GeminiExtractor

        return GeminiExtractor(
            model_name=model_string,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_out,
            reasoning_effort=reasoning_effort,
            seed=seed,
        )
    if model_family == "ollama":
        from .extractors.ollama_extractor import OllamaExtractor

        return OllamaExtractor(
            model_name=model_string,
            temperature=temperature,
            max_output_tokens=max_out,
            seed=seed,
        )
    raise ValueError(f"Model family not implemented: {model_family}")


__all__ = [
    "apply_variant_config",
    "load_schema",
    "load_variant_config",
    "make_extractor",
]

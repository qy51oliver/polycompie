"""
Normalization functions and value helpers for the evaluation pipeline.
"""
import re
from typing import Any, Optional

import numpy as np

from .lookup import (
    PROPERTY_KEY_SYNONYMS,
    UNIT_ALIASES,
    COMPOSITION_TYPE_MAP,
    THICKNESS_UNIT_MAP,
)


# ---------------------------------------------------------------------------
# Property sets
# ---------------------------------------------------------------------------

PROPERTY_SET_EMI = ["shielding_effectiveness"]

PROPERTY_SET_ALL = [
    "breaking_strain", "breaking_strength", "compression_strength",
    "compressive_modulus", "electrical_conductivity", "elongation_at_break",
    "flammability", "glass_transition_temperature", "impact_strength",
    "limiting_oxygen_index", "maximum_compressive_stress", "melting_temperature",
    "peak_heat_release_rate", "shielding_effectiveness", "tensile_strength",
    "tensile_stress", "thermal_conductivity", "time_to_ignition",
    "ultimate_compressive_strain", "ultimate_compressive_strength",
    "volume_conductivity", "youngs_modulus",
]

PROPERTY_SET_SELECTED = [
    "electrical_conductivity", "elongation_at_break", "flammability",
    "glass_transition_temperature", "impact_strength", "limiting_oxygen_index",
    "melting_temperature", "peak_heat_release_rate", "shielding_effectiveness",
    "tensile_strength", "thermal_conductivity", "time_to_ignition", "youngs_modulus",
]


# ---------------------------------------------------------------------------
# Unicode superscript/subscript replacement tables
# ---------------------------------------------------------------------------

_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
})


def _normalize_unit_string(unit: str) -> str:
    """
    Shared preprocessing for unit strings:
    1. Strip whitespace
    2. Replace unicode operators: · ⋅ → removed, − → -
    3. Replace unicode superscripts: ⁻¹ → -1, ² → 2, etc.
    4. Remove spaces
    5. Lowercase
    """
    if not isinstance(unit, str):
        return ""
    u = unit.strip()
    u = u.replace("·", "").replace("⋅", "").replace("−", "-")
    u = u.translate(_SUPERSCRIPT_MAP)
    u = u.replace(" ", "")
    u = u.lower()
    return u


# ---------------------------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------------------------

def base_normalize_property_key(k: str) -> str:
    """Base normalization only: strip, lowercase, space/hyphen → _, drop trailing _N.
    Does NOT apply the synonym lookup — use normalize_property_key for that."""
    k = k.strip().lower()
    k = k.replace(" ", "_").replace("-", "_")
    return re.sub(r"_\d+$", "", k)


def normalize_property_key(k: str) -> str:
    """
    Normalize a property key:
    1. Lowercase, replace spaces/hyphens with underscores
    2. Strip trailing _N suffix (e.g., shielding_effectiveness_1 → shielding_effectiveness)
    3. Lookup in PROPERTY_KEY_SYNONYMS
    """
    nk = base_normalize_property_key(k)
    return PROPERTY_KEY_SYNONYMS.get(nk, nk)


def normalize_property_unit(unit: str) -> str:
    """
    Normalize a property unit string:
    1. Shared preprocessing (unicode, spaces, lowercase)
    2. Lookup in UNIT_ALIASES
    """
    u = _normalize_unit_string(unit)
    return UNIT_ALIASES.get(u, u)


def normalize_composition_type(t: str) -> str:
    """
    Normalize composition type strings:
    1. Lowercase, remove spaces
    2. Lookup in COMPOSITION_TYPE_MAP
    """
    if not isinstance(t, str):
        return ""
    u = t.strip().lower().replace(" ", "")
    return COMPOSITION_TYPE_MAP.get(u, u)


def normalize_thickness_unit(unit: str) -> str:
    """
    Normalize thickness unit strings:
    1. Strip, lowercase
    2. Replace unicode mu variants
    3. Lookup in THICKNESS_UNIT_MAP
    """
    if not isinstance(unit, str):
        return ""
    u = unit.strip().lower()
    # Normalize unicode mu variants before lookup
    u = u.replace("μ", "µ")  # Greek mu (U+03BC) → micro sign (U+00B5)
    return THICKNESS_UNIT_MAP.get(u, u)


def normalize_name(name: str) -> str:
    """
    Normalize a material name for exact-match comparison.
    """
    if not isinstance(name, str):
        return ""
    n = name.lower()
    n = n.replace("–", " ").replace("—", " ").replace("_", " ")
    n = re.sub(r"\s+", " ", n).strip()
    return n


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def to_float(value: Any) -> Optional[float]:
    """
    Try to parse a value as a float.
    Returns None for unparseable values.
    """
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value)
        s = re.split(r"±|\+/-|\+-", s)[0]
        for tok in ("wt%", "vol%", "%", "um", "µm", "μm", "nm", "cm", "mm"):
            s = s.replace(tok, "")
        return float(s.strip())
    except Exception:
        return None


def is_specified(v: Any) -> bool:
    """
    Check if a value is not missing / unspecified.
    """
    if v is None:
        return False
    if isinstance(v, str) and v.strip().lower() in ("", "none", "not specified"):
        return False
    return True


# ---------------------------------------------------------------------------
# Thickness comparison
# ---------------------------------------------------------------------------

_THICKNESS_TO_UM = {"um": 1.0, "mm": 1000.0, "nm": 0.001, "cm": 10000.0}


def thicknesses_equal(v1: Any, u1: Any, v2: Any, u2: Any) -> bool:
    """
    Compare two thickness values accounting for unit conversion.
    Returns True if both are unspecified (both missing counts as match).
    Returns False if either unit is unrecognised after normalization.
    """
    u1 = normalize_thickness_unit(u1 or "")
    u2 = normalize_thickness_unit(u2 or "")

    def _unspecified(v: Any) -> bool:
        return v is None or str(v).strip().lower() in ("", "none", "not specified")

    if _unspecified(v1) and _unspecified(v2):
        return True
    if _unspecified(v1) or _unspecified(v2):
        return False

    f1, f2 = to_float(v1), to_float(v2)
    if f1 is None or f2 is None:
        return False

    if u1 not in _THICKNESS_TO_UM or u2 not in _THICKNESS_TO_UM:
        return False

    f1_um = f1 * _THICKNESS_TO_UM[u1]
    f2_um = f2 * _THICKNESS_TO_UM[u2]
    return bool(np.isclose(f1_um, f2_um, atol=1e-3))
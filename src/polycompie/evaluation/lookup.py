"""
lookup.py — normalization tables for the evaluation pipeline.

All synonym and unit mappings are derived from scanning 59 GT files
via analysis/scan_gt.py. The inclusion criteria for synonyms are strict:

  1. Same physical quantity — not just related, but identical
  2. Compatible units — directly comparable or convertible
  3. Prompt covers it — our extraction prompt targets the canonical key

Keys that represent different physical quantities (normalized SE, component SE,
enhancement percentages, time-to-peak vs peak rate) are intentionally excluded
even if they correlate with a canonical property.
"""

from typing import Dict, Optional, Set, Tuple


PROPERTY_KEY_SYNONYMS: Dict[str, str] = {

    # --- electrical_conductivity ---
    "direct_current_conductivity":           "electrical_conductivity",
    "drect_current_conductivity":            "electrical_conductivity",
    "electrical_conductiveiy":               "electrical_conductivity",
    "in_plane_electrical_conductivity":      "electrical_conductivity",
    "through_plane_electrical_conductivity": "electrical_conductivity",
    "volume_conductivity":                   "electrical_conductivity",

    # --- elongation_at_break ---
    "elongation":                            "elongation_at_break",
    "strain_at_break":                       "elongation_at_break",
    "breaking_strain":                       "elongation_at_break",

    # --- flammability ---
    "flamability":                           "flammability",

    # --- glass_transition_temperature ---
    "glass_transition_temperture":           "glass_transition_temperature",

    # --- impact_strength ---
    "charpy_impact_strength":                "impact_strength",
    "izod_impact_strength":                  "impact_strength",

    # --- limiting_oxygen_index ---
    "loi":                                   "limiting_oxygen_index",

    # --- melting_temperature ---
    "crystalline_melting_point":             "melting_temperature",
    "melting_point":                         "melting_temperature",
    "melting_temperatures":                  "melting_temperature",

    # --- peak_heat_release_rate ---
    "peak_of_heat_release_rate":             "peak_heat_release_rate",
    "peak_of_the_heat_release_rate":         "peak_heat_release_rate",


    # --- tensile_strength ---
    "ultimate_tensile_strength":             "tensile_strength",
    "maximum_tensile_strength":              "tensile_strength",

    # --- thermal_conductivity ---
    "in_plane_thermal_conductivity":         "thermal_conductivity",
    "through_plane_thermal_conductivity":    "thermal_conductivity",

    # --- time_to_ignition ---
    "ignition_time":                         "time_to_ignition",

    # --- youngs_modulus ---
    "tensile_modulus":                       "youngs_modulus",
    "young's_modulus":                       "youngs_modulus",
}


UNIT_ALIASES: Dict[str, str] = {

    # electrical_conductivity
    "scm^-1":       "s/cm",
    "scm-1":        "s/cm",
    "s*cm^-1":      "s/cm",
    "s*cm-1":       "s/cm",
    "sm^-1":        "s/m",
    "sm-1":         "s/m",
    "s*m^-1":       "s/m",
    "s*m-1":        "s/m",

    # temperature
    "◦c":           "°c",
    "oc":           "°c",
    "c":            "°c",

    # impact_strength
    "kj*m^-2":      "kj/m^2",
    "kj*m-2":       "kj/m^2",
    "kjm^-2":       "kj/m^2",
    "kjm-2":        "kj/m^2",
    "kj/m2":        "kj/m^2",

    # limiting_oxygen_index
    "vol%":         "%",

    # peak_heat_release_rate
    "kw*m^-2":      "kw/m^2",
    "kw*m-2":       "kw/m^2",
    "kwm^-2":       "kw/m^2",
    "kwm-2":        "kw/m^2",   # kW·m⁻² after unicode superscript expansion
    "kw/m2":        "kw/m^2",

    # shielding_effectiveness
    # NOT aliased: db/mm, db/(g/cm^3) — different quantities
    "-db":          "db",

    # thermal_conductivity
    "wm^-1k^-1":    "w/mk",
    "wm-1k-1":      "w/mk",
    "w*m^-1*k^-1":  "w/mk",
    "w*m-1*k-1":    "w/mk",
    "w/(mk)":       "w/mk",
    "w/m*k":        "w/mk",
    "w/m-k":        "w/mk",
    "w/m/k":        "w/mk",

    # categorical / no unit
    "notspecify":   "not specified",
    "notspecified": "not specified",
    "none":         "not specified",
    "n/a":          "not specified",
    "ul94":         "not specified",
    "ul-94":        "not specified",

    # youngs_modulus / tensile_strength
    "kgf*cm^-2":    "kgf/cm^2",
    "kgf/cm2":      "kgf/cm^2",
}


EXPECTED_UNITS: Dict[str, Optional[Set[str]]] = {
    "shielding_effectiveness":      {"db"},
    "electrical_conductivity":      {"s/m", "s/cm"},
    "thermal_conductivity":         {"w/mk"},
    "peak_heat_release_rate":       {"kw/m^2", "w/g"},  # cone calorimeter + MCC (both legitimate methods)
    "time_to_ignition":             {"s"},
    "elongation_at_break":          {"%"},
    "flammability":                 None,
    "limiting_oxygen_index":        {"%"},
    "tensile_strength":             {"mpa", "gpa", "kpa", "kgf/cm^2"},
    "melting_temperature":          {"°c"},
    "impact_strength":              {"j/m", "kj/m^2"}, 
    "glass_transition_temperature": {"°c"},
    "youngs_modulus":               {"mpa", "gpa", "kpa", "kgf/cm^2"},
}


CONVERTIBLE_UNITS: Dict[str, Tuple[str, float]] = {
    # electrical_conductivity — base: s/m
    "s/m":      ("conductivity", 1.0),
    "s/cm":     ("conductivity", 100.0),

    # youngs_modulus / tensile_strength — base: mpa
    "mpa":      ("pressure", 1.0),
    "gpa":      ("pressure", 1000.0),
    "kpa":      ("pressure", 0.001),
    "kgf/cm^2": ("pressure", 0.0981),   # 1 kgf/cm² = 0.0981 MPa; GT: x1

    # elongation_at_break / limiting_oxygen_index — base: %
    "%":        ("percent", 1.0),
}


COMPOSITION_TYPE_MAP: Dict[str, str] = {
    "wt%":              "wt%",
    "wt":               "wt%",
    "%":                "wt%",
    "wt.%":             "wt%",
    "weight%":          "wt%",
    "weightpercent":    "wt%",
    "vol%":             "vol%",
    "vol.%":            "vol%",
    "vol":              "vol%",
    "volume%":          "vol%",
    "volumepercent":    "vol%",
    "phr":              "phr",
    "massratio":        "mass ratio",
    "notspecified":     "not specified",
    "":                 "not specified",
}


THICKNESS_UNIT_MAP: Dict[str, str] = {
    "mm":           "mm",
    "millimeter":   "mm",
    "millimeters":  "mm",
    "um":           "um",
    "µm":           "um",
    "μm":           "um",
    "micrometer":   "um",
    "micrometers":  "um",
    "micron":       "um",
    "microns":      "um",
    "cm":           "cm",
    "nm":           "nm",
    "nanometer":    "nm",
    "nanometers":   "nm",
}
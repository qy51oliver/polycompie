"""
Sample matching, property comparison, and LLM-as-judge for the evaluation
pipeline.

Depends on normalize.py for all normalization helpers and lookup.py for
conversion tables.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .lookup import CONVERTIBLE_UNITS, EXPECTED_UNITS
from .normalize import (
    normalize_property_key,
    normalize_property_unit,
    normalize_composition_type,
    normalize_thickness_unit,
    normalize_name,
    to_float,
    is_specified,
    thicknesses_equal,
)


# ---------------------------------------------------------------------------
# Sample merging
# ---------------------------------------------------------------------------

def _composition_key(s: Dict) -> tuple:
    """
    Build a tuple key representing a sample's composition.
    Used to detect duplicate samples before merging.
    """
    matrices = tuple(sorted(
        (m.get("name") or "").strip().lower()
        for m in (s.get("matrix_name") or [])
    ))

    fillers = []
    for f in (s.get("fillers") or []):
        name = (f.get("name") or "").strip().lower()
        comp = f.get("composition") or {}
        amt = to_float(comp.get("amount") if "amount" in comp else f.get("value"))
        unit = normalize_composition_type(comp.get("type") or "")
        fillers.append(f"{name}:{amt}{unit}" if amt is not None else f"{name}:")

    t = s.get("sample_thickness") or {}
    tv = to_float(t.get("value"))
    tu = normalize_thickness_unit(t.get("unit", ""))
    return (matrices, tuple(sorted(fillers)), f"{tv}_{tu}")


def merge_samples(
    samples: List[Dict],
    dataset_label: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Collapse duplicate compositions into single records.

    Returns:
        merged:  list of merged sample dicts
        audit:   list of merge events (for merge_audit.json)
    """
    groups: Dict[Any, List[Dict]] = {}
    for s in (samples or []):
        groups.setdefault(_composition_key(s), []).append(s)

    merged: List[Dict] = []
    audit: List[Dict] = []

    for key, originals in groups.items():
        if len(originals) == 1:
            merged.append(originals[0])
            continue

        base: Dict[str, Any] = {
            "matrix_name":      originals[0].get("matrix_name", []),
            "fillers":          originals[0].get("fillers", []),
            "sample_thickness": originals[0].get("sample_thickness", {}),
            "properties":       {},
        }

        bucket: Dict[str, list] = {}
        for s in originals:
            for pk, pv in (s.get("properties") or {}).items():
                bucket.setdefault(pk, []).append(pv)

        for pk, vals in bucket.items():
            if len(vals) == 1:
                base["properties"][pk] = vals[0]
            else:
                for i, v in enumerate(vals, 1):
                    base["properties"][f"{pk}_{i}"] = (
                        {"value": v.get("value"), "unit": v.get("unit"),
                         "conditions": v.get("conditions", [])}
                        if isinstance(v, dict) else {"value": v, "unit": None}
                    )

        merged.append(base)

        audit.append({
            "dataset":         dataset_label,
            "composition_key": {
                "matrices": list(key[0]),
                "fillers":  list(key[1]),
                "thickness": key[2],
            },
            "n_originals": len(originals),
            "sample_ids":  [s.get("sample_id") for s in originals],
        })

    return merged, audit


# ---------------------------------------------------------------------------
# Placeholder filler detection
# ---------------------------------------------------------------------------

_PLACEHOLDER_NAMES = {"not specified", "none", "n/a", ""}


def _is_placeholder_filler(f: Dict) -> bool:
    """True if a filler entry is a placeholder for 'no filler' (neat polymer)."""
    name = (f.get("name") or "").strip().lower()
    if name not in _PLACEHOLDER_NAMES:
        return False
    comp = f.get("composition") or {}
    amt = to_float(comp.get("amount"))
    return amt is None or amt <= 0


# ---------------------------------------------------------------------------
# Filler amount matching
# ---------------------------------------------------------------------------

def filler_amounts_match(gt_f: Dict, pr_f: Dict) -> bool:
    """
    Check if two filler entries have the same composition amount and type.
    Both-unspecified counts as agreement; one-specified-one-not does not.
    """
    gc = gt_f.get("composition") or {}
    pc = pr_f.get("composition") or {}

    ga = to_float(gc.get("amount"))
    pa = to_float(pc.get("amount"))

    if ga is None and pa is None:
        return True
    if ga is None or pa is None:
        return False

    gt_type = normalize_composition_type(gc.get("type") or "")
    pr_type = normalize_composition_type(pc.get("type") or "")
    if gt_type != pr_type:
        return False

    return bool(np.isclose(ga, pa, atol=1e-6))


# ---------------------------------------------------------------------------
# LLM-as-judge for material name matching
# Results are cached globally to avoid redundant API calls.
# ---------------------------------------------------------------------------

_pair_cache: Dict[Tuple[str, str, str], bool] = {}
_llm_log: List[Dict[str, Any]] = []


def reset_caches() -> None:
    """Clear the LLM match cache and log (call at the start of each run)."""
    global _pair_cache, _llm_log
    _pair_cache = {}
    _llm_log = []


def get_llm_log() -> List[Dict[str, Any]]:
    """Return the LLM match log accumulated during this run."""
    return _llm_log


def _collect_name_pairs(s1: Dict, s2: Dict) -> List[Tuple[str, str, str]]:
    """
    Collect all (name1, name2, type) pairs to be judged.
    type is 'polymer' or 'filler'.
    """
    out: List[Tuple[str, str, str]] = []
    for m1 in (s1.get("matrix_name") or []):
        for m2 in (s2.get("matrix_name") or []):
            n1 = f"{m1.get('name', '')} {m1.get('description', '')}".strip()
            n2 = f"{m2.get('name', '')} {m2.get('description', '')}".strip()
            out.append((n1, n2, "polymer"))
    for f1 in (s1.get("fillers") or []):
        for f2 in (s2.get("fillers") or []):
            n1 = f"{f1.get('name', '')} {f1.get('description', '')}".strip()
            n2 = f"{f2.get('name', '')} {f2.get('description', '')}".strip()
            out.append((n1, n2, "filler"))
    return list(dict.fromkeys(out))


def _normalized_exact_match(a: str, b: str) -> bool:
    """Fallback: normalized exact match (lowercase, strip punctuation)."""
    return normalize_name(a) == normalize_name(b)


def _batch_llm_match(
    pairs: List[Tuple[str, str, str]],
    context: str,
    client: Any,
    model: str,
) -> Dict[Tuple[str, str, str], bool]:
    """
    Send a batch of name pairs to the LLM judge.
    Returns a dict mapping each pair to True/False.
    Falls back to normalized exact match on LLM failure.
    """
    if not pairs:
        return {}

    results: Dict[Tuple[str, str, str], bool] = {}

    def _log(a: str, b: str, etype: str, match: bool, reason: str) -> None:
        _llm_log.append({
            "context": context, "type": etype,
            "gt": a, "pred": b,
            "match": bool(match), "reason": reason,
        })

    if client is None:
        for a, b, etype in pairs:
            match = _normalized_exact_match(a, b)
            results[(a, b, etype)] = match
            _log(a, b, etype, match, "no_llm_normalized_exact")
        return results

    examples = [
        f'{i}. [{e}] GT="{a}" | PRED="{b}"'
        for i, (a, b, e) in enumerate(pairs, 1)
    ]
    pairs_block = "\n".join(examples)
    prompt = (
        "You are matching material names from polymer composite research papers.\n"
        "For each pair, decide if GT and PRED refer to the same material.\n\n"
        "MATCH when the two names describe the same chemical substance — for example\n"
        "when one uses an abbreviation, acronym, or trade name where the other uses\n"
        "the full chemical name, or when one adds descriptors for particle size,\n"
        "morphology, surface treatment, or product grade that do not change the\n"
        "underlying material identity.\n\n"
        "DO NOT MATCH when the names describe chemically distinct substances,\n"
        "or when one is a composite or mixture and the other is a pure component.\n\n"
        'Return ONLY JSON: {"pairs":[{"pair":1,"match":true},...]} — one entry per pair.\n'
        f"Pairs:\n{pairs_block}"
    )
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(res.choices[0].message.content)
        for r in parsed.get("pairs", []):
            idx = r.get("pair")
            if not isinstance(idx, int) or idx < 1 or idx > len(pairs):
                continue
            a, b, etype = pairs[idx - 1]
            match = bool(r.get("match"))
            if not match and _normalized_exact_match(a, b):
                match = True
                reason = "normalized_exact_fallback"
            else:
                reason = "llm_true" if match else "llm_false"
            results[(a, b, etype)] = match
            _log(a, b, etype, match, reason)
    except Exception as e:
        for a, b, etype in pairs:
            match = _normalized_exact_match(a, b)
            results[(a, b, etype)] = match
            _log(a, b, etype, match, f"llm_error_fallback: {e}")

    # If the LLM omitted any requested pair, fall back to normalized exact match
    # so callers always get one result per input pair.
    for a, b, etype in pairs:
        if (a, b, etype) not in results:
            match = _normalized_exact_match(a, b)
            results[(a, b, etype)] = match
            _log(a, b, etype, match, "llm_missing_fallback")

    return results


def _get_matches(
    pairs: List[Tuple[str, str, str]],
    context: str,
    client: Any,
    model: str,
) -> Dict[Tuple[str, str, str], bool]:
    """Return match results for pairs, using cache to avoid re-calling LLM."""
    unseen = [p for p in pairs if p not in _pair_cache]
    if unseen:
        _pair_cache.update(_batch_llm_match(unseen, context, client, model))
    return {p: _pair_cache[p] for p in pairs}


# ---------------------------------------------------------------------------
# Sample similarity scoring
# ---------------------------------------------------------------------------

def sample_similarity(
    gt: Dict,
    pr: Dict,
    context: str,
    client: Any,
    model: str,
) -> Tuple[float, Dict]:
    """
    Score similarity between a GT sample and a predicted sample.
    Returns (score, breakdown_dict).
    """
    pairs = _collect_name_pairs(gt, pr)
    matches = _get_matches(pairs, context, client, model)

    # Matrix score: fraction of GT matrices matched by a distinct PR matrix
    m1 = gt.get("matrix_name") or []
    m2 = pr.get("matrix_name") or []
    m_hits = 0
    m_used: set = set()
    for a in m1:
        a_name = f"{a.get('name', '')} {a.get('description', '')}".strip()
        for j, b in enumerate(m2):
            if j in m_used:
                continue
            b_name = f"{b.get('name', '')} {b.get('description', '')}".strip()
            if matches.get((a_name, b_name, "polymer"), False):
                m_hits += 1
                m_used.add(j)
                break
    if m1:
        m_score = m_hits / len(m1)
    else:
        m_score = 1.0 if not m2 else 0.0

    # Filler score: fraction of GT fillers where name AND amount match a distinct PR filler
    # Filter out placeholder entries (neat polymer convention: name="not specified", amount=0)
    f1 = [f for f in (gt.get("fillers") or []) if not _is_placeholder_filler(f)]
    f2 = [f for f in (pr.get("fillers") or []) if not _is_placeholder_filler(f)]
    f_hits = 0
    f_used: set = set()
    for gf in f1:
        gn = f"{gf.get('name', '')} {gf.get('description', '')}".strip()
        for j, pf in enumerate(f2):
            if j in f_used:
                continue
            pn = f"{pf.get('name', '')} {pf.get('description', '')}".strip()
            if matches.get((gn, pn, "filler"), False) and filler_amounts_match(gf, pf):
                f_hits += 1
                f_used.add(j)
                break
    if f1:
        f_score = f_hits / len(f1)
    else:
        f_score = 1.0 if not f2 else 0.0

    # Thickness score: binary
    t1 = gt.get("sample_thickness") or {}
    t2 = pr.get("sample_thickness") or {}
    thick_ok = thicknesses_equal(
        t1.get("value"), t1.get("unit"),
        t2.get("value"), t2.get("unit"),
    )

    score = 0.4 * m_score + 0.4 * f_score + (0.2 if thick_ok else 0.0)

    return score, {
        "matrix_score":     m_score,
        "filler_score":     f_score,
        "thickness_match":  thick_ok,
        "final_similarity": score,
    }


# ---------------------------------------------------------------------------
# Property matching
# ---------------------------------------------------------------------------

_TEMPERATURE_CANONICALS = frozenset({"glass_transition_temperature", "melting_temperature"})


def extract_measurements(
    props: Dict,
    target_keys: List[str],
    extra_synonyms: Optional[Dict[str, str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Any, str]]:
    """
    Extract (normalized_key, value, normalized_unit) tuples from a
    properties dict, keeping only keys in *target_keys*.

    extra_synonyms: optional runtime extension to PROPERTY_KEY_SYNONYMS,
        built by resolve_gt_key_synonyms() at eval startup using LLM judgment.
        Applied after the static synonym table, so any key not covered by the
        table gets a second chance to resolve to a canonical name.

    context: when set (e.g. {"dataset": "gt", "sample_id": ...}), prints
        diagnostic drops (unit mismatch, unspecified value). Pass None to
        suppress (e.g. for the GT filter loop where we only test emptiness).
    """
    out: List[Tuple[str, Any, str]] = []
    for k, v in (props or {}).items():
        nk = normalize_property_key(k)
        # Apply LLM-extended synonyms for keys the static table did not cover
        if extra_synonyms and nk not in target_keys:
            nk = extra_synonyms.get(nk, nk)
        if nk not in target_keys:
            continue

        val  = v.get("value") if isinstance(v, dict) else v
        unit = (v.get("unit") or "") if isinstance(v, dict) else ""

        if not is_specified(val):
            if context is not None:
                _ds = context.get("dataset", "?")
                _sid = context.get("sample_id", "?")
                print(f"  [unspecified_drop] {_ds} sample_id={_sid} {nk} raw_value={val!r}")
            continue

        norm_unit = normalize_property_unit(str(unit))

        # Kelvin → Celsius for temperature canonicals (offset conversion not
        # expressible in CONVERTIBLE_UNITS' multiplicative-only table).
        # Without this, PR-side Kelvin values get dropped while GT-side °C
        # values stay, creating fake FNs.
        if norm_unit in ("k", "kelvin") and nk in _TEMPERATURE_CANONICALS:
            fval = to_float(val)
            if fval is not None:
                val = fval - 273.15
                norm_unit = "°c"

        expected = EXPECTED_UNITS.get(nk)
        if expected is not None and norm_unit not in expected:
            # Tolerate "not specified" / empty unit ONLY when the canonical
            # has exactly one accepted unit — the inference is unambiguous.
            # Multi-unit canonicals (e.g. tensile_strength: mpa/gpa/kpa) stay
            # strict to avoid guessing the wrong scale.
            if norm_unit in ("", "not specified") and len(expected) == 1:
                norm_unit = next(iter(expected))
            else:
                if context is not None:
                    _ds = context.get("dataset", "?")
                    _sid = context.get("sample_id", "?")
                    print(f"  [unit_drop] {_ds} sample_id={_sid} {nk} value={val} unit={norm_unit!r} (expected: {sorted(expected)})")
                continue

        out.append((nk, val, norm_unit))
    return out


def measurements_equal(a: Tuple, b: Tuple) -> bool:
    """
    Check if two (key, value, unit) tuples represent the same measurement.
    Handles unit conversion for known convertible pairs (e.g. GPa <-> MPa).
    """
    ka, va, ua = a
    kb, vb, ub = b

    if ka != kb:
        return False

    fa, fb = to_float(va), to_float(vb)

    if ua == ub:
        if fa is not None and fb is not None:
            return bool(np.isclose(fa, fb, atol=1e-6))
        return str(va).strip().lower() == str(vb).strip().lower()

    # Units differ — check if they belong to the same convertible group
    conv_a = CONVERTIBLE_UNITS.get(ua)
    conv_b = CONVERTIBLE_UNITS.get(ub)
    if conv_a is None or conv_b is None or conv_a[0] != conv_b[0]:
        return False
    if fa is None or fb is None:
        return False
    return bool(np.isclose(fa * conv_a[1], fb * conv_b[1], atol=1e-6))


def greedy_match(
    gt_ms: List[Tuple],
    pr_ms: List[Tuple],
) -> tuple:
    """
    Greedily match GT measurements to predicted measurements.

    Returns (tp, fp, fn, per_key_tp, per_key_fp, per_key_fn) where the
    per_key dicts map property key -> count for TP / FP / FN respectively.
    """
    used = [False] * len(pr_ms)
    per_key_tp: Dict[str, int] = {}
    per_key_fn: Dict[str, int] = {}

    for g in gt_ms:
        matched = False
        for j, p in enumerate(pr_ms):
            if not used[j] and measurements_equal(g, p):
                used[j] = True
                matched = True
                per_key_tp[g[0]] = per_key_tp.get(g[0], 0) + 1
                break
        if not matched:
            per_key_fn[g[0]] = per_key_fn.get(g[0], 0) + 1

    per_key_fp: Dict[str, int] = {}
    for j, p in enumerate(pr_ms):
        if not used[j]:
            per_key_fp[p[0]] = per_key_fp.get(p[0], 0) + 1

    tp = sum(per_key_tp.values())
    fp = sum(per_key_fp.values())
    fn = sum(per_key_fn.values())
    return tp, fp, fn, per_key_tp, per_key_fp, per_key_fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 from counts."""
    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1
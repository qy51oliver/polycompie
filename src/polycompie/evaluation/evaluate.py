"""
Article-level and corpus-level evaluation of the extraction pipeline.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from .normalize import (
    PROPERTY_SET_EMI,
    PROPERTY_SET_ALL,
    PROPERTY_SET_SELECTED,
)
from .matching import (
    merge_samples,
    sample_similarity,
    extract_measurements,
    greedy_match,
    prf,
    reset_caches,
    get_llm_log,
)


# ---------------------------------------------------------------------------
# JSON / IO helpers
# ---------------------------------------------------------------------------

def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(obj), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_json_file(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"
    except Exception as e:
        return None, str(e)


def parse_gt(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    """
    Parse ground_truth.txt.
    Strips control characters and trailing commas before json.loads.
    """
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return None, str(e)
    if not raw:
        return None, "empty file"

    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    cleaned = re.sub(r",(\s*[\]\}])", r"\1", cleaned)

    try:
        return json.loads(cleaned), None
    except Exception as e:
        return None, f"JSON parse failed after cleanup: {e}"


def unwrap_samples(obj: Any) -> Any:
    """Handle both {'samples': [...]} and [...] formats."""
    if isinstance(obj, dict) and "samples" in obj:
        return obj["samples"]
    return obj


# ---------------------------------------------------------------------------
# Article-level evaluation
# ---------------------------------------------------------------------------

def eval_article(
    article_id: str,
    gt_samples: List[Dict],
    pr_samples: List[Dict],
    property_keys: List[str],
    client: Any,
    model: str,
    pr_extra_synonyms: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate pipeline predictions against GT for one article.

    Steps:
      1. Merge duplicate compositions in GT and predictions independently
      2. Filter GT to samples with at least one property in the target set
      3. Build similarity matrix between all GT and predicted samples
      4. Hungarian algorithm finds optimal one-to-one sample pairing
      5. For each pair: score composition match against strict (=1.0) and
         relaxed (>=0.8) thresholds. Properties get two parallel tracks:
           - relaxed-gated: greedy_match if pair clears 0.8, else FN/FP
           - strict-gated:  greedy_match if pair clears 1.0, else FN/FP
      6. Unmatched GT samples: composition FN + property FN
      7. Unmatched PR samples: composition FP + property FP
    """
    # Step 1: Merge
    gt_merged, gt_audit = merge_samples(gt_samples, "ground_truth")
    pr_merged, pr_audit = merge_samples(pr_samples, "prediction")
    merge_audit = gt_audit + pr_audit

    # Step 2: Filter GT to samples with at least one property in target set.
    # extract_measurements applies the full synonym lookup chain, so GT keys
    # like "in_plane_thermal_conductivity" correctly resolve to "thermal_conductivity".
    n_gt_pre_filter = len(gt_merged)
    gt_merged = [
        s for s in gt_merged
        if extract_measurements(s.get("properties") or {}, property_keys)
    ]
    n_gt_filtered = n_gt_pre_filter - len(gt_merged)
    if n_gt_filtered > 0:
        merge_audit.append({
            "dataset": "ground_truth",
            "filter": "property_relevance",
            "removed": n_gt_filtered,
            "kept": len(gt_merged),
            "reason": f"Removed {n_gt_filtered} GT samples with no properties in target set",
        })

    if not gt_merged or not pr_merged:
        return {
            "article_id": article_id,
            "skipped":    True,
            "reason":     f"empty after merge (gt={len(gt_merged)}, pr={len(pr_merged)})",
            "merge_audit": merge_audit,
        }

    # Step 3: Similarity matrix
    n_gt = len(gt_merged)
    n_pr = len(pr_merged)
    sim        = np.zeros((n_gt, n_pr))
    breakdowns = {}
    for i, g in enumerate(gt_merged):
        for j, p in enumerate(pr_merged):
            score, bd = sample_similarity(g, p, article_id, client, model)
            sim[i][j]          = score
            breakdowns[(i, j)] = (score, bd)

    # Step 4: Hungarian assignment
    gi, pj = linear_sum_assignment(1 - sim)

    # Composition counters (strict and relaxed)
    comp_tp_s = comp_fp_s = comp_fn_s = 0
    comp_tp_r = comp_fp_r = comp_fn_r = 0

    # Property counters — relaxed-gated (score >= 0.8) and strict-gated (score == 1.0)
    prop_tp = prop_fp = prop_fn = 0          # relaxed gate
    prop_tp_s = prop_fp_s = prop_fn_s = 0   # strict gate
    prop_tp_per_key: Dict[str, int] = {}
    prop_fp_per_key: Dict[str, int] = {}
    prop_fn_per_key: Dict[str, int] = {}
    prop_tp_per_key_s: Dict[str, int] = {}
    prop_fp_per_key_s: Dict[str, int] = {}
    prop_fn_per_key_s: Dict[str, int] = {}

    def _accum(dest: Dict[str, int], src: Dict[str, int]) -> None:
        for k, v in src.items():
            dest[k] = dest.get(k, 0) + v

    sample_details = []
    matched_gt = set(gi)
    matched_pr = set(pj)

    # Step 5: Evaluate each matched pair
    for g_idx, p_idx in zip(gi, pj):
        g = gt_merged[g_idx]
        p = pr_merged[p_idx]
        score, breakdown = breakdowns[(g_idx, p_idx)]

        # score max is exactly 1.0 (0.4+0.4+0.2); >= guards floating-point edge
        strict  = score >= 1.0
        relaxed = score >= 0.8

        # Composition counts
        if strict:
            comp_tp_s += 1
        else:
            comp_fp_s += 1
            comp_fn_s += 1
        if relaxed:
            comp_tp_r += 1
        else:
            comp_fp_r += 1
            comp_fn_r += 1

        # Property: count TPs only when composition is relaxed-matched (>=0.8).
        tp = fp = fn = 0
        gt_meas = extract_measurements(
            g.get("properties") or {}, property_keys,
            context={"dataset": "gt", "sample_id": g.get("sample_id")},
        )
        pr_meas = extract_measurements(
            p.get("properties") or {}, property_keys, pr_extra_synonyms,
            context={"dataset": "pr", "sample_id": p.get("sample_id")},
        )
        if relaxed:
            tp, fp, fn, per_tp, per_fp, per_fn = greedy_match(gt_meas, pr_meas)
            prop_tp += tp
            prop_fp += fp
            prop_fn += fn
            _accum(prop_tp_per_key, per_tp)
            _accum(prop_fp_per_key, per_fp)
            _accum(prop_fn_per_key, per_fn)
        else:
            # Below the relaxed threshold we cannot reliably attribute property values
            # to this specific sample, so treat them as FN (GT) and FP (PR) respectively.
            prop_fn += len(gt_meas)
            prop_fp += len(pr_meas)
            _accum(prop_fn_per_key, dict(Counter(m[0] for m in gt_meas)))
            _accum(prop_fp_per_key, dict(Counter(m[0] for m in pr_meas)))

        # Strict property gate: only credit if composition was perfectly matched
        if strict:
            prop_tp_s += tp
            prop_fp_s += fp
            prop_fn_s += fn
            _accum(prop_tp_per_key_s, per_tp)
            _accum(prop_fp_per_key_s, per_fp)
            _accum(prop_fn_per_key_s, per_fn)
        else:
            prop_fn_s += len(gt_meas)
            prop_fp_s += len(pr_meas)
            _accum(prop_fn_per_key_s, dict(Counter(m[0] for m in gt_meas)))
            _accum(prop_fp_per_key_s, dict(Counter(m[0] for m in pr_meas)))

        sample_details.append({
            "gt_idx":          int(g_idx),
            "pr_idx":          int(p_idx),
            "similarity":      round(score, 4),
            "strict_match":    strict,
            "relaxed_match":   relaxed,
            "matrix_score":    round(breakdown["matrix_score"], 4),
            "filler_score":    round(breakdown["filler_score"], 4),
            "thickness_match": breakdown["thickness_match"],
            "gt_sample_id":    g.get("sample_id"),
            "pr_sample_id":    p.get("sample_id"),
            "gt_n_properties": len(gt_meas),
            "pr_n_properties": len(pr_meas),
            "prop_tp": tp, "prop_fp": fp, "prop_fn": fn,
        })

    # Step 6: Unmatched GT → composition FN + property FN
    for g_idx, g in enumerate(gt_merged):
        if g_idx in matched_gt:
            continue
        comp_fn_s += 1
        comp_fn_r += 1
        gt_meas = extract_measurements(
            g.get("properties") or {}, property_keys,
            context={"dataset": "gt", "sample_id": g.get("sample_id")},
        )
        prop_fn += len(gt_meas)
        prop_fn_s += len(gt_meas)
        _accum(prop_fn_per_key, dict(Counter(m[0] for m in gt_meas)))
        _accum(prop_fn_per_key_s, dict(Counter(m[0] for m in gt_meas)))

    # Step 7: Unmatched PR → composition FP + property FP
    for p_idx, p in enumerate(pr_merged):
        if p_idx in matched_pr:
            continue
        comp_fp_s += 1
        comp_fp_r += 1
        pr_meas = extract_measurements(
            p.get("properties") or {}, property_keys, pr_extra_synonyms,
            context={"dataset": "pr", "sample_id": p.get("sample_id")},
        )
        prop_fp += len(pr_meas)
        prop_fp_s += len(pr_meas)
        _accum(prop_fp_per_key, dict(Counter(m[0] for m in pr_meas)))
        _accum(prop_fp_per_key_s, dict(Counter(m[0] for m in pr_meas)))

    return {
        "article_id":       article_id,
        "skipped":          False,
        "n_gt_samples":     n_gt,
        "n_pr_samples":     n_pr,
        "sample_details":   sample_details,
        "avg_similarity":   float(np.mean([d["similarity"] for d in sample_details])) if sample_details else 0.0,
        "strict_sample_matches":  sum(1 for d in sample_details if d["strict_match"]),
        "relaxed_sample_matches": sum(1 for d in sample_details if d["relaxed_match"]),
        "merge_audit":      merge_audit,
        "_counts": {
            "comp": {
                "tp_s": comp_tp_s, "fp_s": comp_fp_s, "fn_s": comp_fn_s,
                "tp_r": comp_tp_r, "fp_r": comp_fp_r, "fn_r": comp_fn_r,
            },
            "prop": {
                "tp": prop_tp, "fp": prop_fp, "fn": prop_fn,
                "tp_s": prop_tp_s, "fp_s": prop_fp_s, "fn_s": prop_fn_s,
                "per_key": {
                    k: {
                        "tp": prop_tp_per_key.get(k, 0),
                        "fp": prop_fp_per_key.get(k, 0),
                        "fn": prop_fn_per_key.get(k, 0),
                        "tp_s": prop_tp_per_key_s.get(k, 0),
                        "fp_s": prop_fp_per_key_s.get(k, 0),
                        "fn_s": prop_fn_per_key_s.get(k, 0),
                    }
                    for k in property_keys
                },
            },
        },
    }


def aggregate_counts(results: List[Dict]) -> Dict:
    agg: Dict[str, Any] = {
        "comp": {"tp_s": 0, "fp_s": 0, "fn_s": 0, "tp_r": 0, "fp_r": 0, "fn_r": 0},
        "prop": {"tp": 0, "fp": 0, "fn": 0, "tp_s": 0, "fp_s": 0, "fn_s": 0, "per_key": {}},
    }
    for r in results:
        if r.get("skipped"):
            continue
        c = r["_counts"]
        for k in ("tp_s", "fp_s", "fn_s", "tp_r", "fp_r", "fn_r"):
            agg["comp"][k] += c["comp"][k]
        for k in ("tp", "fp", "fn", "tp_s", "fp_s", "fn_s"):
            agg["prop"][k] += c["prop"][k]
        for prop_key, counts in (c["prop"].get("per_key") or {}).items():
            if prop_key not in agg["prop"]["per_key"]:
                agg["prop"]["per_key"][prop_key] = {"tp": 0, "fp": 0, "fn": 0, "tp_s": 0, "fp_s": 0, "fn_s": 0}
            for cnt_k in ("tp", "fp", "fn", "tp_s", "fp_s", "fn_s"):
                agg["prop"]["per_key"][prop_key][cnt_k] += counts[cnt_k]
    return agg


def make_metrics(counts: Dict) -> Dict:
    out: Dict[str, Any] = {}

    # Composition: strict and relaxed
    comp = counts["comp"]
    ps, rs, f1s = prf(comp["tp_s"], comp["fp_s"], comp["fn_s"])
    pr_, rr, f1r = prf(comp["tp_r"], comp["fp_r"], comp["fn_r"])
    out["composition"] = {
        "strict":  {"precision": round(ps, 4), "recall": round(rs, 4), "f1": round(f1s, 4)},
        "relaxed": {"precision": round(pr_, 4), "recall": round(rr, 4), "f1": round(f1r, 4)},
    }

    # Property: relaxed-gated (composition score >= 0.8) and strict-gated (score == 1.0)
    prop = counts["prop"]
    pp, rp, f1p = prf(prop["tp"], prop["fp"], prop["fn"])
    pp_s, rp_s, f1p_s = prf(prop["tp_s"], prop["fp_s"], prop["fn_s"])
    out["property"] = {
        "relaxed": {"precision": round(pp, 4), "recall": round(rp, 4), "f1": round(f1p, 4),
                     "tp": prop["tp"], "fp": prop["fp"], "fn": prop["fn"]},
        "strict":  {"precision": round(pp_s, 4), "recall": round(rp_s, 4), "f1": round(f1p_s, 4),
                     "tp": prop["tp_s"], "fp": prop["fp_s"], "fn": prop["fn_s"]},
    }

    # Per-property breakdown (both relaxed and strict)
    out["per_property"] = {}
    for key, c in sorted((counts["prop"].get("per_key") or {}).items()):
        pk, rk, f1k = prf(c["tp"], c["fp"], c["fn"])
        pk_s, rk_s, f1k_s = prf(c["tp_s"], c["fp_s"], c["fn_s"])
        out["per_property"][key] = {
            "relaxed": {"precision": round(pk, 4), "recall": round(rk, 4), "f1": round(f1k, 4),
                         "tp": c["tp"], "fp": c["fp"], "fn": c["fn"]},
            "strict":  {"precision": round(pk_s, 4), "recall": round(rk_s, 4), "f1": round(f1k_s, 4),
                         "tp": c["tp_s"], "fp": c["fp_s"], "fn": c["fn_s"]},
        }

    return out


# ---------------------------------------------------------------------------
# PR key synonym resolution (run once per eval run)
# GT key synonyms are built offline via `python -m polycompie.evaluation.build_gt_synonyms`
# and baked into lookup.py.
# ---------------------------------------------------------------------------

def _collect_pr_raw_keys(
    experiments_dir: Path,
    phase: int,
    model: str,
) -> Dict[str, int]:
    """
    Scan all PR output files for this (phase, model) run.
    Returns {base_normalized_key: occurrence_count} across all samples.
    """
    counts: Dict[str, int] = {}
    phase_str = f"phase{phase}"
    for article_dir in experiments_dir.iterdir():
        if not article_dir.is_dir():
            continue
        pr_path = article_dir / phase_str / model / "output.json"
        if not pr_path.exists():
            continue
        try:
            obj = json.loads(pr_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        obj = unwrap_samples(obj)
        if not isinstance(obj, list):
            continue
        for sample in obj:
            for k in (sample.get("properties") or {}):
                nk = k.strip().lower()
                nk = nk.replace(" ", "_").replace("-", "_")
                nk = re.sub(r"_\d+$", "", nk)
                counts[nk] = counts.get(nk, 0) + 1
    return counts


def _resolve_pr_key_synonyms(
    experiments_dir: Path,
    phase: int,
    model: str,
    property_keys: List[str],
    client: Any,
    llm_model: str,
) -> Dict[str, str]:
    """
    Collect all PR output property keys for this (phase, model) run.
    For keys not resolved by the static synonym table, ask LLM which
    canonical key they map to (if any).
    Returns {unknown_normalized_key: canonical_key}.
    Called once at eval startup; applied to PR extract_measurements calls only.
    GT keys are handled offline via build_gt_synonyms.py and lookup.py.
    """
    from .lookup import PROPERTY_KEY_SYNONYMS
    canonical_set = set(property_keys)

    pr_raw = _collect_pr_raw_keys(experiments_dir, phase, model)
    unknowns = []
    for k in sorted(pr_raw):
        resolved = PROPERTY_KEY_SYNONYMS.get(k, k)
        if resolved not in canonical_set:
            unknowns.append(k)

    if not unknowns:
        print("[key_synonyms] All PR keys covered by static synonym table.")
        return {}

    print(f"[key_synonyms] Found {len(unknowns)} PR key(s) not in static table: {unknowns}")

    if client is None:
        print("[key_synonyms] No LLM client — unknown PR keys will be dropped.")
        return {}

    canonical_list = "\n".join(f"  - {k}" for k in sorted(property_keys))
    examples = "\n".join(f'{i}. "{k}"' for i, k in enumerate(unknowns, 1))

    prompt = (
        "You are an expert in polymer composite material characterization.\n"
        "Below are property key names from a model's extraction output.\n"
        "For each key, determine whether it refers to the same physical measurement\n"
        "as one of the following canonical property keys:\n\n"
        f"{canonical_list}\n\n"
        "Rules:\n"
        "- MATCH if the key describes the same physical quantity as a canonical key,\n"
        "  even if phrased differently (different word order, level of specificity,\n"
        "  or synonymous terms).\n"
        "- DO NOT MATCH if the key describes a distinct physical quantity, even if related.\n"
        "- DO NOT MATCH if it is ambiguous which canonical key applies.\n"
        "- Each input key maps to at most ONE canonical key.\n\n"
        'Return ONLY JSON: {"mappings": [{"input": "<key>", "canonical": "<canonical_or_null>"}, ...]}\n\n'
        f"Keys to resolve:\n{examples}"
    )

    try:
        res = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(res.choices[0].message.content)
        extra: Dict[str, str] = {}
        for entry in parsed.get("mappings", []):
            inp = str(entry.get("input", "")).strip()
            canon = entry.get("canonical")
            if not inp:
                continue
            if isinstance(canon, str) and canon.strip() in canonical_set:
                extra[inp] = canon.strip()
                print(f"[key_synonyms]   PR: {inp!r} -> {canon.strip()!r}")
            else:
                print(f"[key_synonyms]   PR: {inp!r} -> no match")
        return extra
    except Exception as e:
        print(f"[key_synonyms] LLM call failed: {e} — unknown PR keys will be dropped.")
        return {}


# ---------------------------------------------------------------------------
# Article discovery
# ---------------------------------------------------------------------------

def find_articles(annotated_dir: Path, experiments_dir: Path, phase: int, model: str) -> List[Dict]:
    """
    Walk annotated_dir. For each article subfolder find:
      {annotated_dir}/{article_id}/ground_truth.txt
      {experiments_dir}/{article_id}/phase{N}/{model}/output.json
    """
    articles = []
    for folder in sorted(annotated_dir.iterdir()):
        if not folder.is_dir():
            continue
        article_id = folder.name
        gt_path = folder / "ground_truth.txt"
        pr_path = experiments_dir / article_id / f"phase{phase}" / model / "output.json"
        articles.append({
            "article_id": article_id,
            "gt_path":    gt_path,
            "pr_path":    pr_path,
            "has_gt":     gt_path.exists(),
            "has_pr":     pr_path.exists(),
        })
    return articles


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    annotated_dir: Path,
    experiments_dir: Path,
    phase: int,
    model: str,
    property_keys: List[str],
    out_dir: Path,
    use_llm: bool,
    llm_model: str,
) -> None:
    """
    Run full evaluation: load articles, evaluate each, aggregate, write outputs.
    """
    reset_caches()

    client = None
    if use_llm:
        if OpenAI is None:
            print("[warn] openai package not available — using normalized exact match fallback")
        else:
            key = os.getenv("OPENAI_API_KEY", "").strip()
            if not key:
                print("[warn] OPENAI_API_KEY not set — using normalized exact match fallback")
            else:
                client = OpenAI(api_key=key)

    # Collect all PR output keys for this (phase, model) run and resolve any
    # variants not covered by the static synonym table. Runs once before the
    # article loop so every article shares the same pr_extra_synonyms cache.
    # GT keys are NOT resolved here — they are handled offline by
    # build_gt_synonyms.py and baked into lookup.py.
    print("[key_synonyms] Resolving PR property key variants for this run...")
    pr_extra_synonyms = _resolve_pr_key_synonyms(
        experiments_dir, phase, model, property_keys, client, llm_model
    )
    print(f"[key_synonyms] {len(pr_extra_synonyms)} PR synonym(s) resolved for this run.\n")

    articles = find_articles(annotated_dir, experiments_dir, phase, model)
    print(f"Found {len(articles)} article folders (model={model})\n")

    results:      List[Dict] = []
    all_audits:   List[Dict] = []
    per_article:  List[Dict] = []
    skips:        List[str]  = []

    for art in articles:
        aid = art["article_id"]
        row: Dict[str, Any] = {"article_id": aid}

        if not art["has_gt"]:
            skips.append(f"{aid}: ground_truth.txt missing")
            row.update({"skipped": True, "reason": "no_gt"})
            per_article.append(row)
            continue

        gt_raw, gt_err = parse_gt(art["gt_path"])
        if gt_raw is None:
            skips.append(f"{aid}: GT parse failed — {gt_err}")
            row.update({"skipped": True, "reason": f"gt_parse_failed: {gt_err}"})
            per_article.append(row)
            continue

        gt_samples = unwrap_samples(gt_raw)
        if not isinstance(gt_samples, list) or not gt_samples:
            skips.append(f"{aid}: GT is empty or not a list")
            row.update({"skipped": True, "reason": "gt_empty"})
            per_article.append(row)
            continue

        if not art["has_pr"]:
            skips.append(f"{aid}: pipeline output.json missing")
            row.update({"skipped": True, "reason": "no_prediction"})
            per_article.append(row)
            continue

        pr_raw, pr_err = load_json_file(art["pr_path"])
        if pr_raw is None:
            skips.append(f"{aid}: prediction parse failed — {pr_err}")
            row.update({"skipped": True, "reason": f"pr_parse_failed: {pr_err}"})
            per_article.append(row)
            continue

        pr_samples = unwrap_samples(pr_raw)
        if not isinstance(pr_samples, list) or not pr_samples:
            row.update({"skipped": True, "reason": "empty_prediction"})
            per_article.append(row)
            continue

        print(f"  {aid}")
        r = eval_article(aid, gt_samples, pr_samples, property_keys, client, llm_model, pr_extra_synonyms)
        results.append(r)
        if r.get("merge_audit"):
            all_audits.extend(r["merge_audit"])

        row.update({
            "n_gt":           r.get("n_gt_samples"),
            "n_pr":           r.get("n_pr_samples"),
            "avg_sim":        round(r.get("avg_similarity", 0.0), 3),
            "strict_matches": r.get("strict_sample_matches"),
            "relaxed_matches":r.get("relaxed_sample_matches"),
            "skipped":        r.get("skipped", False),
        })
        if not r.get("skipped"):
            print(
                f"    gt={r['n_gt_samples']} pr={r['n_pr_samples']} "
                f"sim={r['avg_similarity']:.3f} "
                f"strict={r['strict_sample_matches']} relaxed={r['relaxed_sample_matches']}"
            )
        per_article.append(row)

    # Aggregate
    counts  = aggregate_counts(results)
    metrics = make_metrics(counts)

    avg_sim = float(np.mean([r["avg_similarity"] for r in results if not r.get("skipped")])) \
              if results else 0.0

    summary_table = {
        "annotated_dir":       str(annotated_dir),
        "experiments_dir":     str(experiments_dir),
        "phase":               phase,
        "model":               model,
        "property_set":        property_keys,
        "evaluated_articles":  len([r for r in results if not r.get("skipped")]),
        "skipped_articles":    len(skips),
        "avg_similarity":      round(avg_sim, 4),
        "pr_key_synonyms_extended": pr_extra_synonyms,
        "composition_matching":  metrics["composition"],
        "property_matching":    metrics["property"],
        "per_property_metrics": metrics["per_property"],
    }

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary_table.json",  summary_table)
    write_json(out_dir / "per_article.json",     per_article)
    write_json(out_dir / "details.json",         results)
    write_json(out_dir / "merge_audit.json",     all_audits)
    write_json(out_dir / "llm_match_log.json",   get_llm_log())
    write_json(out_dir / "pr_key_synonyms.json",  pr_extra_synonyms)
    (out_dir / "skips.txt").write_text("\n".join(skips), encoding="utf-8")

    # Print summary
    print()
    print("=" * 70)
    print(f"RESULTS — {annotated_dir.name}  (phase {phase}, model={model})")
    print("=" * 70)

    def _comp_row(label, m):
        print(f"  {label:<35} "
              f"strict  P={m['strict']['precision']:.3f} R={m['strict']['recall']:.3f} F1={m['strict']['f1']:.3f}  "
              f"relaxed P={m['relaxed']['precision']:.3f} R={m['relaxed']['recall']:.3f} F1={m['relaxed']['f1']:.3f}")

    print(f"  avg sample similarity: {avg_sim:.3f}")
    print()
    _comp_row("composition", metrics["composition"])
    pm = metrics["property"]
    pr_ = pm["relaxed"]
    ps_ = pm["strict"]
    print(f"  {'property (relaxed)':<35} P={pr_['precision']:.3f} R={pr_['recall']:.3f} F1={pr_['f1']:.3f}  "
          f"(TP={pr_['tp']} FP={pr_['fp']} FN={pr_['fn']})")
    print(f"  {'property (strict)':<35} P={ps_['precision']:.3f} R={ps_['recall']:.3f} F1={ps_['f1']:.3f}  "
          f"(TP={ps_['tp']} FP={ps_['fp']} FN={ps_['fn']})")
    print()
    print("  Per-property F1 (relaxed | strict):")
    for prop, m in sorted(metrics["per_property"].items()):
        mr = m["relaxed"]
        ms = m["strict"]
        print(f"    {prop:<40} "
              f"relaxed P={mr['precision']:.3f} R={mr['recall']:.3f} F1={mr['f1']:.3f}  "
              f"strict P={ms['precision']:.3f} R={ms['recall']:.3f} F1={ms['f1']:.3f}")
    print()
    print(f"  Evaluated: {summary_table['evaluated_articles']} articles  "
          f"Skipped: {summary_table['skipped_articles']}")
    print(f"  Output: {out_dir}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline extraction against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--annotated-dir", required=True,
        help="Root folder of GT annotations, e.g. pipeline/polm_ie/data/annotated",
    )
    parser.add_argument(
        "--experiments-dir", required=True,
        help="Root folder of pipeline outputs, e.g. pipeline/polm_ie/data/experiments",
    )
    parser.add_argument(
        "--phase", type=int, default=4,
        help="Evaluation phase",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model family directory name (e.g. gpt-5.2, claude-sonnet-4.6)",
    )
    parser.add_argument(
        "--property-set", default="selected", choices=["emi", "selected", "all"],
        help="Property set: 'emi' = SE only, 'selected' = 13 curated keys, 'all' = all properties",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for evaluation artifacts",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM judge for name matching; use normalized exact match only",
    )
    parser.add_argument(
        "--llm-model", default="gpt-4o",
        help="OpenAI model for LLM judge (default: gpt-4o)",
    )
    args = parser.parse_args()

    annotated_dir   = Path(args.annotated_dir)
    experiments_dir = Path(args.experiments_dir)
    for d, name in [(annotated_dir, "--annotated-dir"), (experiments_dir, "--experiments-dir")]:
        if not d.exists():
            print(f"[error] Directory not found ({name}): {d}")
            sys.exit(1)

    if args.property_set == "emi":
        property_keys = PROPERTY_SET_EMI
    elif args.property_set == "selected":
        property_keys = PROPERTY_SET_SELECTED
    else:
        property_keys = PROPERTY_SET_ALL

    out_dir = Path(args.out_dir)

    print("=" * 70)
    print("evaluate_collection")
    print("=" * 70)
    print(f"Annotated  : {annotated_dir}")
    print(f"Experiments: {experiments_dir}")
    print(f"Phase      : {args.phase}")
    print(f"Model      : {args.model}")
    print(f"Properties : {args.property_set} ({len(property_keys)} keys)")
    print(f"LLM judge  : {'disabled (normalized exact match)' if args.no_llm else f'enabled ({args.llm_model})'}")
    print(f"Output     : {out_dir}")
    print("=" * 70)
    print()

    run_evaluation(
        annotated_dir=annotated_dir,
        experiments_dir=experiments_dir,
        phase=args.phase,
        model=args.model,
        property_keys=property_keys,
        out_dir=out_dir,
        use_llm=not args.no_llm,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()

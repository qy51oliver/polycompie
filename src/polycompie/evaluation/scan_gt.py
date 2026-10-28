"""
Scan ground-truth annotations and report observed vocabulary.

Walks {annotated_dir}/{article_id}/ground_truth.txt, parses each file with
the same cleanup evaluate.py applies, and tallies:

  - property keys (normalized) with coverage status
  - property units per key (normalized)
  - composition types (matrix + filler)
  - thickness units

Use the output to spot variants that need new entries in lookup.py's
synonym/alias maps before evaluation.

Usage:
    python -m polycompie.evaluation.scan_gt --annotated-dir data/annotated --out scan_gt_report.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .evaluate import parse_gt, unwrap_samples
from .lookup import PROPERTY_KEY_SYNONYMS
from .matching import extract_measurements
from .normalize import (
    PROPERTY_SET_ALL,
    PROPERTY_SET_SELECTED,
    base_normalize_property_key,
    normalize_composition_type,
    normalize_property_key,
    normalize_property_unit,
    normalize_thickness_unit,
)


def scan(annotated_dir: Path, target_keys: List[str]) -> Dict[str, Any]:
    """Return a vocabulary report for every ground_truth.txt under annotated_dir.

    Emits two views of canonical counts:
      property_keys:           every GT entry that resolves to a canonical (raw,
                               post-synonym, pre-filter)
      evaluable_property_keys: what evaluate.py actually scores against —
                               computed via the same extract_measurements()
                               function eval uses, so it picks up Kelvin
                               conversion, EXPECTED_UNITS gating, "not specified"
                               tolerance, etc. Single source of truth.
    """
    prop_keys: Counter = Counter()
    eval_prop_keys: Counter = Counter()
    raw_keys: Counter = Counter()
    sample_coverage: Counter = Counter()
    eval_sample_coverage: Counter = Counter()
    prop_units: Dict[str, Counter] = {}
    comp_types: Counter = Counter()
    thickness_units: Counter = Counter()
    n_files = 0
    n_samples = 0
    skipped: List[str] = []

    for folder in sorted(annotated_dir.iterdir()):
        if not folder.is_dir():
            continue
        gt_path = folder / "ground_truth.txt"
        if not gt_path.exists():
            continue
        raw, err = parse_gt(gt_path)
        if raw is None:
            skipped.append(f"{folder.name}: {err}")
            continue
        samples = unwrap_samples(raw)
        if not isinstance(samples, list):
            skipped.append(f"{folder.name}: not a sample list")
            continue

        n_files += 1
        for s in samples:
            if not isinstance(s, dict):
                continue
            n_samples += 1

            props = s.get("properties") or {}
            canonicals_in_sample: set = set()
            for k, v in props.items():
                raw = base_normalize_property_key(k)
                nk = normalize_property_key(k)
                raw_keys[raw] += 1
                prop_keys[nk] += 1
                canonicals_in_sample.add(nk)
                if isinstance(v, dict):
                    unit = normalize_property_unit(str(v.get("unit") or ""))
                    prop_units.setdefault(nk, Counter())[unit] += 1
            for c in canonicals_in_sample:
                sample_coverage[c] += 1

            # Use extract_measurements as the single source of truth for what
            # eval actually scores against. Same Kelvin conv, EXPECTED_UNITS,
            # not-specified tolerance, target-set filter.
            for nk, _val, _unit in extract_measurements(props, target_keys):
                eval_prop_keys[nk] += 1
            for nk in {m[0] for m in extract_measurements(props, target_keys)}:
                eval_sample_coverage[nk] += 1

            for mat in (s.get("matrix_name") or []):
                comp = (mat.get("composition") or {}).get("type")
                if comp:
                    comp_types[normalize_composition_type(comp)] += 1
            for f in (s.get("fillers") or []):
                comp = (f.get("composition") or {}).get("type")
                if comp:
                    comp_types[normalize_composition_type(comp)] += 1

            t = s.get("sample_thickness") or {}
            tu = t.get("unit")
            if tu:
                thickness_units[normalize_thickness_unit(tu)] += 1

    return {
        "n_files": n_files,
        "n_samples": n_samples,
        "skipped": skipped,
        "property_keys": dict(prop_keys.most_common()),
        "evaluable_property_keys": dict(eval_prop_keys.most_common()),
        "raw_property_keys": dict(raw_keys.most_common()),
        "sample_coverage": dict(sample_coverage.most_common()),
        "evaluable_sample_coverage": dict(eval_sample_coverage.most_common()),
        "property_units": {k: dict(v.most_common()) for k, v in prop_units.items()},
        "composition_types": dict(comp_types.most_common()),
        "thickness_units": dict(thickness_units.most_common()),
    }


def _coverage_status(key: str, canonical: List[str]) -> str:
    if key in canonical:
        return "canonical"
    resolved = PROPERTY_KEY_SYNONYMS.get(key)
    if resolved in canonical:
        return f"-> {resolved} (synonym)"
    return "UNMAPPED"


def print_summary(report: Dict[str, Any], canonical: List[str]) -> None:
    print("=" * 78)
    print(f"GT scan — {report['n_files']} files, {report['n_samples']} samples")
    print("=" * 78)

    if report["skipped"]:
        print(f"\nSkipped {len(report['skipped'])} folder(s):")
        for s in report["skipped"]:
            print(f"  {s}")

    print(f"\nProperty keys ({len(report['property_keys'])} unique):")
    unmapped: List[str] = []
    for k, n in report["property_keys"].items():
        status = _coverage_status(k, canonical)
        print(f"  {n:>4}  {k:<50}  {status}")
        if status == "UNMAPPED":
            unmapped.append(k)

    print(f"\nComposition types  : {report['composition_types']}")
    print(f"Thickness units    : {report['thickness_units']}")

    if unmapped:
        print(f"\n{len(unmapped)} property key(s) need attention "
              f"(not canonical, not in PROPERTY_KEY_SYNONYMS):")
        for k in unmapped:
            print(f"  - {k}")
        print("\nRun build_gt_synonyms to propose LLM-classified mappings.")
    else:
        print("\nAll property keys covered by canonical set or synonym table.")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Scan ground-truth files and report observed vocabulary."
    )
    p.add_argument("--annotated-dir", required=True, type=Path,
                   help="Root folder of GT annotations, e.g. data/annotated")
    p.add_argument("--out", type=Path, default=None,
                   help="Write full JSON report (normalized counts) to this path.")
    p.add_argument("--raw-out", type=Path, default=None,
                   help="Write raw-key JSON (pre-synonym counts + canonical resolution) to this path.")
    p.add_argument("--property-set", default="selected",
                   choices=["selected", "all"],
                   help="Canonical property set to check coverage against.")
    args = p.parse_args()

    if not args.annotated_dir.exists():
        print(f"[error] not found: {args.annotated_dir}", file=sys.stderr)
        return 1

    canonical = PROPERTY_SET_SELECTED if args.property_set == "selected" else PROPERTY_SET_ALL
    report = scan(args.annotated_dir, list(canonical))
    print_summary(report, canonical)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nFull JSON report written to: {args.out}")

    if args.raw_out:
        canonical_set = set(canonical)
        flat = []
        by_canonical: Dict[str, Dict[str, Any]] = {}
        for raw, count in report["raw_property_keys"].items():
            canon = PROPERTY_KEY_SYNONYMS.get(raw, raw)
            if raw == canon and canon in canonical_set:
                status = "canonical"
            elif canon in canonical_set:
                status = "synonym"
            elif raw in canonical_set:
                status = "canonical"
            else:
                status = "unmapped"
            flat.append({"raw": raw, "count": count, "canonical": canon, "status": status})
            bucket = by_canonical.setdefault(canon, {"total": 0, "raw_keys": []})
            bucket["total"] += count
            bucket["raw_keys"].append({"raw": raw, "count": count, "status": status})

        # Sort each canonical's raw_keys by count desc
        for c in by_canonical.values():
            c["raw_keys"].sort(key=lambda x: -x["count"])
        # Sort by_canonical by total desc
        by_canonical = dict(sorted(by_canonical.items(), key=lambda kv: -kv[1]["total"]))

        raw_report = {
            "n_files": report["n_files"],
            "n_samples": report["n_samples"],
            "raw_property_keys": flat,
            "by_canonical": by_canonical,
        }
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_out.write_text(
            json.dumps(raw_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Raw-key JSON report written to: {args.raw_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

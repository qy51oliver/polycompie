"""
Propose canonical mappings for ground-truth property keys not covered by
the static synonym table.

For each unmapped key surfaced by scan_gt, ask an LLM whether it refers
to the same physical quantity as one of the canonical keys. Emit a draft
Python file (PROPERTY_KEY_SYNONYMS_GT_DRAFT) you review and hand-merge
into lookup.py.

The script never edits lookup.py directly so the human curation behind
the EXCLUDED: comments stays intact.

Usage:
    python -m polycompie.evaluation.build_gt_synonyms \
        --annotated-dir data/annotated \
        --out src/polycompie/evaluation/_gt_synonyms_draft.py
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from .lookup import PROPERTY_KEY_SYNONYMS
from .normalize import PROPERTY_SET_ALL, PROPERTY_SET_SELECTED
from .scan_gt import scan


def find_unknown_keys(prop_keys: Dict[str, int], canonical: List[str]) -> List[str]:
    """Return property keys not already mapped to a canonical key."""
    canonical_set = set(canonical)
    out: List[str] = []
    for k in sorted(prop_keys):
        resolved = PROPERTY_KEY_SYNONYMS.get(k, k)
        if resolved in canonical_set:
            continue
        out.append(k)
    return out


def ask_llm(
    unknowns: List[str],
    canonical: List[str],
    client: "OpenAI",
    model: str,
) -> Dict[str, Dict[str, str]]:
    """Return {input_key: {"canonical": <key or None>, "reason": <short>}}."""
    canonical_block = "\n".join(f"  - {k}" for k in sorted(canonical))
    keys_block = "\n".join(f'{i}. "{k}"' for i, k in enumerate(unknowns, 1))

    prompt = (
        "You are an expert in polymer composite material characterization.\n"
        "Below are property key names found in ground-truth annotations.\n"
        "For each key, decide whether it refers to the SAME physical measurement\n"
        "as one of these canonical property keys:\n\n"
        f"{canonical_block}\n\n"
        "Rules:\n"
        "- MATCH only if the key describes the same physical quantity as a canonical\n"
        "  key, even if phrased differently (different word order, level of\n"
        "  specificity, or synonymous terms).\n"
        "- DO NOT MATCH if the key describes a distinct physical quantity (different\n"
        "  units, normalization, or derived measurement), even if related.\n"
        "- DO NOT MATCH if the canonical mapping is ambiguous.\n"
        "- Each input key maps to at most ONE canonical key.\n\n"
        'Return ONLY JSON of the form:\n'
        '{"mappings": [{"input": "<key>", "canonical": "<canonical_or_null>", '
        '"reason": "<short justification>"}, ...]}\n\n'
        f"Keys to resolve:\n{keys_block}"
    )

    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(res.choices[0].message.content)

    canonical_set = set(canonical)
    out: Dict[str, Dict[str, str]] = {}
    for entry in parsed.get("mappings", []):
        inp = str(entry.get("input", "")).strip()
        canon = entry.get("canonical")
        reason = str(entry.get("reason") or "").strip()
        if not inp:
            continue
        if isinstance(canon, str) and canon.strip() in canonical_set:
            out[inp] = {"canonical": canon.strip(), "reason": reason}
        else:
            out[inp] = {"canonical": "", "reason": reason or "no canonical match"}
    return out


def write_draft(results: Dict[str, Dict[str, str]], out_path: Path) -> None:
    """Emit a Python snippet ready for review."""
    mapped = {k: v for k, v in results.items() if v.get("canonical")}
    skipped = {k: v for k, v in results.items() if not v.get("canonical")}

    lines = [
        '"""',
        "Auto-generated draft from build_gt_synonyms.",
        "Review each entry, then hand-merge into PROPERTY_KEY_SYNONYMS in lookup.py.",
        '"""',
        "",
        "PROPERTY_KEY_SYNONYMS_GT_DRAFT = {",
    ]
    for k in sorted(mapped):
        canon = mapped[k]["canonical"]
        reason = mapped[k]["reason"].replace("\n", " ")
        lines.append(f"    # {reason}" if reason else "")
        lines.append(f"    {k!r}: {canon!r},")
    lines.append("}")
    lines.append("")

    if skipped:
        lines.append("# Keys the LLM declined to map (review manually):")
        for k in sorted(skipped):
            reason = skipped[k]["reason"].replace("\n", " ") or "no reason provided"
            lines.append(f"#   {k!r}  // {reason}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a GT-key → canonical synonym draft via LLM."
    )
    p.add_argument("--annotated-dir", required=True, type=Path,
                   help="Root folder of GT annotations, e.g. data/annotated")
    p.add_argument("--out", required=True, type=Path,
                   help="Path for the draft Python file (will be created).")
    p.add_argument("--property-set", default="selected",
                   choices=["selected", "all"],
                   help="Canonical property set to map against.")
    p.add_argument("--llm-model", default="gpt-4o",
                   help="OpenAI model for the judge (default: gpt-4o).")
    args = p.parse_args()

    if not args.annotated_dir.exists():
        print(f"[error] not found: {args.annotated_dir}", file=sys.stderr)
        return 1

    if OpenAI is None:
        print("[error] openai package required", file=sys.stderr)
        return 1
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("[error] OPENAI_API_KEY not set", file=sys.stderr)
        return 1
    client = OpenAI(api_key=api_key)

    canonical = PROPERTY_SET_SELECTED if args.property_set == "selected" else PROPERTY_SET_ALL

    report = scan(args.annotated_dir, list(canonical))
    unknowns = find_unknown_keys(report["property_keys"], canonical)

    if not unknowns:
        print("All GT property keys already resolve to canonical via PROPERTY_KEY_SYNONYMS.")
        return 0

    print(f"Scanning {report['n_files']} GT file(s); "
          f"{len(unknowns)} unmapped key(s) to resolve.")
    for u in unknowns:
        print(f"  - {u}")

    print(f"\nAsking {args.llm_model} for canonical mappings...")
    results = ask_llm(unknowns, canonical, client, args.llm_model)

    mapped_n = sum(1 for v in results.values() if v.get("canonical"))
    skipped_n = len(results) - mapped_n
    print(f"\nLLM mapped {mapped_n} / {len(unknowns)} key(s); "
          f"{skipped_n} declined.")
    for k, v in sorted(results.items()):
        canon = v.get("canonical") or "<no match>"
        reason = v.get("reason") or ""
        print(f"  {k!r} -> {canon!r}  ({reason})")

    write_draft(results, args.out)
    print(f"\nDraft written to: {args.out}")
    print("Review entries, then merge into PROPERTY_KEY_SYNONYMS in lookup.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

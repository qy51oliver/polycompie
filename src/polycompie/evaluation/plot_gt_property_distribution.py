from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict, List

import numpy as np

from .normalize import PROPERTY_SET_SELECTED

# Display order for the 13 properties when --keep-order is set.
DISPLAY_ORDER: List[str] = list(PROPERTY_SET_SELECTED)

# Functional categories — used for color when --show-legend is set.
CATEGORY_ORDER: List[str] = [
    "EMI / shielding",
    "Electrical transport",
    "Thermal transport",
    "Mechanical",
    "Thermal transitions",
    "Fire safety",
]

PROPERTY_CATEGORY: Dict[str, str] = {
    "shielding_effectiveness":      "EMI / shielding",
    "electrical_conductivity":      "Electrical transport",
    "thermal_conductivity":         "Thermal transport",
    "tensile_strength":             "Mechanical",
    "elongation_at_break":          "Mechanical",
    "youngs_modulus":               "Mechanical",
    "impact_strength":              "Mechanical",
    "glass_transition_temperature": "Thermal transitions",
    "melting_temperature":          "Thermal transitions",
    "flammability":                 "Fire safety",
    "limiting_oxygen_index":        "Fire safety",
    "peak_heat_release_rate":       "Fire safety",
    "time_to_ignition":             "Fire safety",
}


def _draw_donut(*, keys, heights, share_pcts, colors, title, legend_handles,
                single_column=False):
    """Draw exact count proportions with evenly spaced external labels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 3.5) if single_column else (8.0, 4.5), dpi=300)
    wedges, _ = ax.pie(
        heights, colors=colors, startangle=30, counterclock=False,
        wedgeprops={"width": 0.43, "edgecolor": "white", "linewidth": 0.8},
    )
    ax.text(0, 0.08, f"n = {sum(heights):,}", ha="center", va="center",
            fontsize=10 if single_column else 15, fontweight="bold")
    ax.text(0, -0.23 if single_column else -0.10,
            "Property\ninstances" if single_column else "Property instances",
            ha="center", va="center", fontsize=7 if single_column else 9)

    anchors = []
    for i, wedge in enumerate(wedges):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        anchors.append((i, np.cos(angle), np.sin(angle)))

    for side in (-1, 1):
        rows = sorted(
            (p for p in anchors if (1 if p[1] >= 0 else -1) == side),
            key=lambda p: p[2],
        )
        extent = 1.85 if single_column else 1.12
        label_ys = np.linspace(-extent, extent, len(rows)) if len(rows) > 1 else [0.0]
        for (i, x, y), label_y in zip(rows, label_ys):
            # Move horizontally outside the circle before fanning out to labels.
            # This avoids crossing the donut or intersecting adjacent leaders.
            if heights[i]:
                ax.plot(
                    [x * 1.015, side * 1.045, side * 1.13, side * 1.17],
                    [y * 1.015, y * 1.015, label_y, label_y],
                    color="0.55", linewidth=0.6,
                )
            label = keys[i].replace("_", " ").capitalize().replace("Youngs", "Young’s")
            align = "left" if side == 1 else "right"
            if single_column:
                label = textwrap.fill(label, width=18, break_long_words=False)
                ax.annotate(label, (side * 1.21, label_y), xytext=(0, 2),
                            textcoords="offset points", ha=align, va="bottom",
                            fontsize=7.5, linespacing=1.0)
                ax.annotate(f"{heights[i]} ({share_pcts[i]:.1f}%)",
                            (side * 1.21, label_y), xytext=(0, -1),
                            textcoords="offset points", ha=align, va="top",
                            fontsize=7, fontweight="bold")
            else:
                ax.text(side * 1.21, label_y + 0.045, label,
                        ha=align, va="center", fontsize=9)
                ax.text(side * 1.21, label_y - 0.070,
                        f"{heights[i]} ({share_pcts[i]:.1f}%)",
                        ha=align, va="center", fontsize=8, fontweight="bold")

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim((-2.10, 2.40) if single_column else (-1.25, 1.25))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.01), ncol=2 if single_column else 3,
                  fontsize=6 if single_column else 7, frameon=False)
    fig.tight_layout(pad=0.4)
    return fig


def plot(
    *,
    counts: Dict[str, int],
    out_path: Path,
    title: str | None,
    bar_width: float,
    show_legend: bool,
    sort_by_count: bool,
    chart_type: str = "bar",
    single_column: bool = False,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman", "serif"],
        "font.size": 9,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    keys = list(DISPLAY_ORDER)
    if sort_by_count:
        keys.sort(key=lambda k: (-counts.get(k, 0), k))

    heights = [counts.get(k, 0) for k in keys]
    total = sum(heights)
    share_pcts = [100.0 * h / total if total else 0.0 for h in heights]
    if chart_type not in {"bar", "donut"}:
        raise ValueError(f"Unknown chart type: {chart_type}")
    if single_column and chart_type != "donut":
        raise ValueError("The single-column layout is available for donut charts only.")
    if chart_type == "donut" and (total <= 0 or any(h < 0 for h in heights)):
        raise ValueError("A donut chart requires nonnegative counts and a positive total.")

    # Colors
    if show_legend:
        cmap = {cat: plt.cm.tab10(i / max(len(CATEGORY_ORDER), 1))
                for i, cat in enumerate(CATEGORY_ORDER)}
        colors = [cmap[PROPERTY_CATEGORY[k]] for k in keys]
    else:
        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = [tab20[i % 20] for i in range(len(keys))]

    if chart_type == "donut":
        handles = []
        if show_legend:
            categories = dict.fromkeys(PROPERTY_CATEGORY[k] for k in keys)
            handles = [plt.Rectangle((0, 0), 1, 1, fc=cmap[cat],
                                     ec="0.25", lw=0.35, label=cat)
                       for cat in categories]
        fig = _draw_donut(keys=keys, heights=heights, share_pcts=share_pcts,
                          colors=colors, title=title, legend_handles=handles,
                          single_column=single_column)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        print(f"wrote {out_path}")
        print(f"  slices: {total} property rows across {len(keys)} canonical keys")
        print("  labels: count and share of selected GT (sums to 100%)")
        return

    n = len(keys)
    x = np.arange(n, dtype=float)
    ymax = max(1.0, max(heights) * 1.18) if heights else 1.0

    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=150)
    ax.bar(x, heights, width=bar_width, color=colors,
           edgecolor="0.25", linewidth=0.35, align="center", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=90, ha="center", va="top", fontsize=8)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=4))
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="0.75", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=10)

    # Bar labels: count + (share of selected GT %)
    for xi, h, pct in zip(x, heights, share_pcts):
        ax.text(xi, h + ymax * 0.015, f"{int(h)}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Right axis: share of total (%), aligned to the left axis. The 13 bars
    # sum to 100% by construction.
    if total:
        ax_right = ax.twinx()
        ax_right.set_ylim(0, ymax * 100.0 / total)
        ax_right.set_ylabel("Share of selected GT (%)", fontsize=10)
        ax_right.spines["top"].set_visible(False)
        ax_right.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))

    if show_legend:
        seen, handles = set(), []
        for k in keys:
            cat = PROPERTY_CATEGORY[k]
            if cat in seen:
                continue
            seen.add(cat)
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=cmap[cat],
                                         ec="0.25", lw=0.35, label=cat))
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  bars: {total} property rows across {len(keys)} canonical keys")
    print(f"  right axis: each bar's share of those {total} rows (sums to 100%)")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Bar or donut chart for the 13 selected properties in GT. "
                    "Counts are normalized property rows; percentages are shares "
                    "of the total selected row count (sums to 100%)."
    )
    p.add_argument("--report", type=Path, default=Path("scan_gt_report.json"),
                   help="scan_gt_report.json with property_keys field")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (PNG, PDF, SVG, etc.). Default: "
                        "data/figures/gt_property_distribution[_donut][_column].png")
    p.add_argument("--chart-type", choices=("bar", "donut"), default="bar",
                   help="Chart style (default: bar); both use the same fonts and colors")
    p.add_argument("--single-column", action="store_true",
                   help="Use a roughly 3.5-inch-wide donut with wrapped labels at 7.5 pt")
    p.add_argument("--title", type=str, default=None, help="Optional plot title")
    p.add_argument("--bar-width", type=float, default=0.42,
                   help="Bar width in x-units; applies only to bar charts (default 0.42)")
    p.add_argument("--show-legend", action="store_true",
                   help="Color by functional category and show legend")
    p.add_argument("--keep-order", action="store_true",
                   help="Keep PROPERTY_SET_SELECTED order; default sorts by descending count")
    p.add_argument("--use-raw", action="store_true",
                   help="Use raw property_keys (pre-unit-filter) instead of evaluable_property_keys")
    args = p.parse_args()
    if args.single_column and args.chart_type != "donut":
        p.error("--single-column requires --chart-type donut")

    if not args.report.exists():
        raise SystemExit(
            f"[error] {args.report} not found. Run "
            f"`python -m polycompie.evaluation.scan_gt --annotated-dir <dir> "
            f"--property-set selected --out {args.report}` first."
        )
    data = json.loads(args.report.read_text(encoding="utf-8"))

    # Default: evaluable counts (gated by EXPECTED_UNITS). Falls back to raw
    # property_keys if the report predates the evaluable_property_keys field.
    field = "property_keys" if args.use_raw else "evaluable_property_keys"
    counts = data.get(field) or data.get("property_keys", {})

    plot(
        counts=counts,
        out_path=args.out or Path(
            "data/figures/gt_property_distribution"
            + ("_donut" if args.chart_type == "donut" else "")
            + ("_column" if args.single_column else "") + ".png"
        ),
        title=args.title,
        bar_width=args.bar_width,
        show_legend=args.show_legend,
        sort_by_count=not args.keep_order,
        chart_type=args.chart_type,
        single_column=args.single_column,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

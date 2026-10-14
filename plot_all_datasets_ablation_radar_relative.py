"""Own, LCT and six-cell ablation radars on one shared relative-RMSE scale.

Reads the raw per-sample prediction files (own.csv, LCT.csv, 6cell.csv) and
selects every column BY NAME: ``Cell``, ``SOH_true`` and one column per method
in METHOD_ORDER. Column order in the CSV does not matter (the three files use
different orders); a missing, duplicated or unrecognised method column stops
the script instead of being silently mapped to the wrong method.

Per cell, each method's RMSE is divided by DRA's RMSE in that cell:

    ratio = RMSE_method / RMSE_DRA

so DRA is the 1x circle, a point outside it is worse than DRA and a point
inside it is better. All spokes in all three panels share one logarithmic
radial scale (ratios are multiplicative: 0.5x and 2x sit equally far from
1x), with rings labelled 1x, 1.5x, 2x, 3x ... Lower error is closer to the
centre. Spoke labels give DRA's absolute RMSE so the ratios can be turned
back into percentage points.

Usage (from outputs/Ablation study)::

    python plot_all_datasets_ablation_radar_relative.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATASETS = [("own", "Own"), ("LCT", "LCT"), ("6cell", "6cell")]
CELL_COLUMN = "Cell"
TRUE_COLUMN = "SOH_true"
BASELINE = "DRA"
METHOD_ORDER = ["DRA", "DRA-w/o CRA", "DRA-w/o Dilated Encoder", "DRA-w/o Anchor", "DRA-w/o Interaction"]
# Older exports used DPA-* names; map them onto the DRA names (by name, never by position).
ALIASES = {
    "DPA": "DRA", "DPA-AnchorOnly": "DRA-w/o CRA", "DPA-SingleScale": "DRA-w/o Dilated Encoder",
    "DPA-w/oAnchor": "DRA-w/o Anchor", "DPA-w/oInteract": "DRA-w/o Interaction",
}
NON_METHOD_COLUMNS = {CELL_COLUMN, TRUE_COLUMN, "Cycle"}
COLORS = dict(zip(METHOD_ORDER, ["#E7833A", "#8A7CC7", "#D9A441", "#2596A6", "#59A66F"]))
MARKERS = dict(zip(METHOD_ORDER, ["o", "D", "s", "^", "v"]))
RING_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
GRID_COLOR = "#B4C2C8"
OUTPUT_NAME = "all_datasets_ablation_rmse_radar_relative"


def natural_key(value):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(value))]


def load_predictions(path: Path) -> pd.DataFrame:
    """Read one raw prediction CSV and return it with canonical method names."""
    with open(path, newline="", encoding="utf-8-sig") as stream:
        header = [name.strip() for name in next(csv.reader(stream))]
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(f"{path.name}: duplicated column name(s) {duplicates}")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [ALIASES.get(name.strip(), name.strip()) for name in frame.columns]
    missing = [name for name in [CELL_COLUMN, TRUE_COLUMN, *METHOD_ORDER] if name not in frame.columns]
    unknown = [name for name in frame.columns if name not in NON_METHOD_COLUMNS and name not in METHOD_ORDER]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}; header is {list(frame.columns)}")
    if unknown:
        raise ValueError(f"{path.name}: unrecognised column(s) {unknown}; add them to METHOD_ORDER or ALIASES")
    numeric = [TRUE_COLUMN, *METHOD_ORDER]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if frame[[CELL_COLUMN, *numeric]].isna().any().any():
        raise ValueError(f"{path.name}: missing values in Cell/SOH_true/method columns")
    return frame


def rmse_pp(true: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean((predicted.to_numpy() - true.to_numpy()) ** 2)) * 100)


def cell_metrics(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for cell, group in frame.groupby(CELL_COLUMN):
        baseline = rmse_pp(group[TRUE_COLUMN], group[BASELINE])
        for method in METHOD_ORDER:
            value = rmse_pp(group[TRUE_COLUMN], group[method])
            rows.append({"Dataset": key, "Cell": cell, "Method": method, "N": len(group),
                         "RMSE (pp)": value, "Ratio to DRA": value / baseline})
    return pd.DataFrame(rows)


def pooled_rmse(frame: pd.DataFrame) -> dict[str, float]:
    return {method: rmse_pp(frame[TRUE_COLUMN], frame[method]) for method in METHOD_ORDER}


def radius(ratio, center: float, outer: float):
    return np.log(np.asarray(ratio) / center) / np.log(outer / center)


def label_alignment(angle: float) -> tuple[str, str]:
    # angle measured clockwise from north (theta_direction=-1, zero at N)
    x, y = np.sin(angle), np.cos(angle)
    ha = "center" if abs(x) < 0.25 else ("left" if x > 0 else "right")
    va = "center" if abs(y) < 0.25 else ("bottom" if y > 0 else "top")
    return ha, va


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / OUTPUT_NAME

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 8,
        "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
    })

    panels = []
    for key, title in DATASETS:
        frame = load_predictions(args.data_root / f"{key}.csv")
        metrics = cell_metrics(frame, key)
        cells = sorted(metrics["Cell"].unique(), key=natural_key)
        panels.append((key, title, cells, metrics, pooled_rmse(frame)))
        print(f"{key}: {len(frame)} samples, {len(cells)} cells, methods read by name: {METHOD_ORDER}")
    all_metrics = pd.concat([panel[3] for panel in panels], ignore_index=True)

    # One radial scale for every spoke of every panel.
    low, high = all_metrics["Ratio to DRA"].min(), all_metrics["Ratio to DRA"].max()
    center = min(0.4, low / 1.15)
    outer = high * 1.12
    rings = [value for value in RING_CANDIDATES if center < value < outer]

    fig, axes = plt.subplots(1, 3, figsize=(240 / 25.4, 100 / 25.4), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.16, top=0.80, wspace=0.50)

    for axis, (key, title, cells, metrics, pooled), letter in zip(axes, panels, "abc"):
        angles = np.linspace(0, 2 * np.pi, len(cells), endpoint=False)
        closed = np.r_[angles, angles[0]]
        dense = np.linspace(0, 2 * np.pi, 361)
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_xticks([])
        axis.grid(False)
        axis.spines["polar"].set_visible(False)
        axis.patch.set_visible(False)

        for value in rings:
            r = radius(value, center, outer)
            is_baseline = value == 1.0
            axis.plot(dense, np.full_like(dense, r), color=GRID_COLOR, lw=0.55,
                      ls="-" if is_baseline else (0, (2, 3)), zorder=0)
        for angle in angles:
            axis.plot([angle, angle], [0, 1], color=GRID_COLOR, lw=0.5, ls=(0, (2, 3)), zorder=0)

        table = metrics.pivot(index="Cell", columns="Method", values="Ratio to DRA").loc[cells]
        # Ring labels go in the gap between the two adjacent spokes with the least data reach.
        reach = table.max(axis=1).to_numpy()
        gap = int(np.argmin(np.maximum(reach, np.roll(reach, -1))))
        label_angle = angles[gap] + np.pi / len(cells)
        for value in rings:
            axis.text(label_angle, radius(value, center, outer), f"{value:g}×", fontsize=6.5, color="#5B6B72",
                      ha="center", va="center", zorder=6,
                      bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85})

        dra_rmse = metrics[metrics["Method"].eq(BASELINE)].set_index("Cell").loc[cells, "RMSE (pp)"]
        for method in METHOD_ORDER:
            values = radius(table[method].to_numpy(), center, outer)
            values = np.r_[values, values[0]]
            is_baseline = method == BASELINE
            axis.plot(closed, values, color=COLORS[method], marker=MARKERS[method],
                      lw=1.7 if is_baseline else 1.05, ms=3.6 if is_baseline else 3.0, markeredgewidth=0.6,
                      alpha=1 if is_baseline else 0.9, zorder=5 if is_baseline else 3, clip_on=False)
            if is_baseline:
                axis.fill(dense, np.full_like(dense, values[0]), color=COLORS[method], alpha=0.07, zorder=1)

        for angle, cell in zip(angles, cells):
            ha, va = label_alignment(angle)
            axis.text(angle, 1.10, f"{cell}\nDRA {dra_rmse[cell]:.2f}", ha=ha, va=va, fontsize=7.3,
                      linespacing=1.35, clip_on=False)

        ablations = [pooled[m] for m in METHOD_ORDER if m != BASELINE]
        position = axis.get_position()
        x_mid = (position.x0 + position.x1) / 2
        fig.text(position.x0 - 0.03, 0.955, letter, fontweight="bold", fontsize=10, ha="left")
        fig.text(x_mid, 0.955, f"{title} ({len(cells)} cells)", fontsize=10, fontweight="bold", ha="center")
        fig.text(x_mid, 0.912, f"Pooled RMSE: DRA {pooled[BASELINE]:.2f} pp · ablations "
                 f"{min(ablations):.2f}–{max(ablations):.2f} pp", fontsize=7, color="#4E626C", ha="center")

    handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], lw=1.7 if m == BASELINE else 1.05,
                      markersize=4, label=m) for m in METHOD_ORDER]
    fig.legend(handles=handles, ncol=5, loc="center", bbox_to_anchor=(0.5, 0.05), fontsize=8,
               handlelength=1.9, handletextpad=0.6, columnspacing=1.7)

    for extension, options in {".pdf": {}, ".svg": {}, ".png": {"dpi": 600},
                               ".tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}}.items():
        fig.savefig(str(base) + extension, **options)
    plt.close(fig)

    all_metrics.to_csv(str(base) + "_cell_metrics.csv", index=False, float_format="%.6g")
    print(f"Radial scale: log, centre {center:.3g}×, rim {outer:.3g}×, rings {rings}")
    print(f"Ratio range {low:.3f}–{high:.3f}; points inside 1× (ablation better than DRA):")
    better = all_metrics[all_metrics["Ratio to DRA"].lt(1)]
    print(better[["Dataset", "Cell", "Method", "RMSE (pp)", "Ratio to DRA"]].round(3).to_string(index=False)
          if len(better) else "  none")
    print(f"Saved: {base}.pdf/.svg/.png/.tiff and _cell_metrics.csv")


if __name__ == "__main__":
    main()

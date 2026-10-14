"""Run current and initial/current/delta Slot Attention on own corrected DRT."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
LCT_DIR = WORKSPACE / "data" / "LCT"
CURRENT_LCT_DIR = LCT_DIR / "Current" / "slot_attention"
for path in [LCT_DIR, CURRENT_LCT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import current_drt_slot_attention_lct as current_lct  # noqa: E402
import drt_soh_only_multislot_lct as initial_lct  # noqa: E402


SOURCE_ROOT = SCRIPT_DIR / "results" / "modeling" / "newresults" / "paper_method_own_8cell_analysis"
DEFAULT_DRT = (
    SOURCE_ROOT
    / "02_corrected_drt"
    / "DRT_Distribution_after_RL_CPE_removal_deboundary.csv"
)
DEFAULT_TAU = SOURCE_ROOT / "02_corrected_drt" / "tau_vector.csv"
RESULTS_ROOT = SCRIPT_DIR / "results" / "modeling" / "newresults"
OUTPUTS = {
    "current": RESULTS_ROOT / "Current" / "slot_attention",
    "initial": RESULTS_ROOT / "Initial" / "slot_attention",
}
CELLS = [f"No.{index}" for index in range(1, 9)]
MODEL_NAMES = {
    "current": "current_drt_anchor_k5_huber",
    "initial": "anchor_k5_huber",
}
SCENARIO_LABELS = {
    "current": "Current corrected DRT Slot Attention",
    "initial": "Initial/current/delta corrected DRT Slot Attention",
}


def valid_boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_data(
    drt_path: Path,
    tau_path: Path,
    min_soh: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = pd.read_csv(drt_path)
    tau = pd.read_csv(tau_path).sort_values("tau_index")
    gamma_columns = sorted(
        [column for column in source if column.startswith("gamma_")],
        key=lambda column: int(column.split("_")[1]),
    )
    required = {"Cell", "Cycle", "SOH", "FitSuccess"}
    missing = required.difference(source.columns)
    if missing or len(gamma_columns) != len(tau):
        raise ValueError(
            f"Invalid source: missing={sorted(missing)}, "
            f"gamma={len(gamma_columns)}, tau={len(tau)}"
        )
    numeric_gamma = source[gamma_columns].apply(pd.to_numeric, errors="coerce")
    valid = (
        source["Cell"].isin(CELLS)
        & pd.to_numeric(source["SOH"], errors="coerce").ge(min_soh)
        & valid_boolean(source["FitSuccess"])
        & np.isfinite(numeric_gamma.to_numpy()).all(axis=1)
    )
    frame = source.loc[valid, ["Cell", "Cycle", "SOH"]].copy()
    frame["Cycle"] = pd.to_numeric(frame["Cycle"], errors="raise").astype(int)
    frame["SOH"] = pd.to_numeric(frame["SOH"], errors="raise").astype(float)
    gamma = np.maximum(numeric_gamma.loc[valid].to_numpy(np.float32), 0.0)
    if frame.duplicated(["Cell", "Cycle"]).any():
        raise ValueError("Corrected DRT contains duplicate Cell/Cycle rows")
    order = np.lexsort((frame["Cycle"].to_numpy(), frame["Cell"].to_numpy()))
    frame = frame.iloc[order].reset_index(drop=True)
    gamma = gamma[order]
    missing_cells = sorted(set(CELLS).difference(frame["Cell"].unique()))
    if missing_cells:
        raise ValueError(f"No usable rows for cells: {missing_cells}")

    initial_gamma = np.empty_like(gamma)
    for cell in CELLS:
        indices = frame.index[frame["Cell"].eq(cell)].to_numpy()
        initial_index = int(frame.loc[indices].sort_values("Cycle").index[0])
        initial_gamma[indices] = gamma[initial_index]

    log_tau = (
        tau["log10_tau"].to_numpy(np.float32)
        if "log10_tau" in tau
        else np.log10(tau["tau_seconds"].to_numpy(np.float32))
    )
    position = (
        2.0 * (log_tau - log_tau.min()) / (log_tau.max() - log_tau.min()) - 1.0
    ).astype(np.float32)
    return frame, gamma, initial_gamma, position, log_tau


def metric_table(core, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = pd.DataFrame(
        [
            core.metric_row(model, group)
            for model, group in predictions.groupby("Model", sort=False)
        ]
    ).sort_values("RMSE_SOH")
    folds = pd.DataFrame(
        [
            {**core.metric_row(model, group), "TestCell": cell}
            for (model, cell), group in predictions.groupby(
                ["Model", "TestCell"], sort=False
            )
        ]
    )
    return aggregate, folds


def save_plots(
    output: Path,
    scenario: str,
    frame: pd.DataFrame,
    gamma: np.ndarray,
    log_tau: np.ndarray,
    aggregate: pd.DataFrame,
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    summaries: pd.DataFrame,
    attention: np.ndarray,
    attention_indices: np.ndarray,
    comparison: pd.DataFrame,
    display_label: str | None = None,
) -> None:
    label = display_label or SCENARIO_LABELS[scenario]
    model = str(aggregate.iloc[0]["Model"])
    selected_predictions = predictions[predictions["Model"].eq(model)]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    for axis, cell in zip(axes.flat, CELLS):
        data = selected_predictions[selected_predictions["Cell"].eq(cell)].sort_values(
            "Cycle"
        )
        axis.plot(data["Cycle"], data["SOH_true"], "o-", ms=3, lw=1.3, label="True")
        axis.plot(
            data["Cycle"],
            data["SOH_pred"],
            "s--",
            ms=3,
            lw=1.1,
            label="Predicted",
        )
        axis.set_title(cell)
        axis.set_xlabel("Cycle")
        axis.set_ylabel("SOH")
        axis.grid(alpha=0.2)
    axes.flat[-1].set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Own {label}: strict LOCO", fontsize=15)
    fig.savefig(output / "loco_soh_trends.png", dpi=180)
    plt.close(fig)

    ordered = folds.sort_values("RMSE_pct_point")
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.barh(ordered["TestCell"], ordered["RMSE_pct_point"], color="#4C78A8")
    axis.set_xlabel("LOCO RMSE / SOH percentage points")
    axis.set_title(f"Own {label}: error by held-out cell")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "fold_rmse_by_cell.png", dpi=180)
    plt.close(fig)

    n_slots = 5
    slot_data = summaries[summaries["Model"].eq(model)]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    for axis, cell in zip(axes.flat, CELLS):
        data = slot_data[slot_data["Cell"].eq(cell)].sort_values("Cycle")
        for slot in range(1, n_slots + 1):
            center = data[f"slot_{slot}_center_norm"].to_numpy()
            center_log_tau = np.interp(
                center, [-1, 1], [log_tau.min(), log_tau.max()]
            )
            size = 8 + 70 * data[f"slot_{slot}_importance"].to_numpy()
            axis.scatter(
                data["Cycle"], center_log_tau, s=size, alpha=0.65, label=f"Slot {slot}"
            )
        axis.set_title(cell)
        axis.set_xlabel("Cycle")
        axis.set_ylabel("Attention center log10(tau / s)")
        axis.grid(alpha=0.2)
    axes.flat[-1].set_visible(False)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Own {label}: SOH-sensitive Slot regions", fontsize=15)
    fig.savefig(output / "slot_attention_center_evolution.png", dpi=180)
    plt.close(fig)

    lookup = {int(index): row for row, index in enumerate(attention_indices)}
    examples = ["No.1", "No.3", "No.6"]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, n_slots))
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    for row_axes, cell in zip(axes, examples):
        rows = frame[frame["Cell"].eq(cell)].sort_values("Cycle")
        picks = [rows.iloc[0], rows.iloc[len(rows) // 2], rows.iloc[-1]]
        for axis, picked in zip(row_axes, picks):
            index = int(picked.name)
            spectrum = gamma[index]
            axis.plot(log_tau, spectrum, color="#202938", lw=1.8, label="DRT")
            masks = attention[lookup[index]].astype(np.float32)
            scale = max(float(spectrum.max()), 1e-6)
            for slot in range(n_slots):
                mask = masks[slot] / max(float(masks[slot].max()), 1e-8)
                axis.plot(
                    log_tau,
                    mask * scale,
                    color=colors[slot],
                    lw=1.0,
                    alpha=0.8,
                )
            axis.set_title(f"{cell} Cycle {int(picked['Cycle'])}")
            axis.set_xlabel("log10(tau / s)")
            axis.set_ylabel("gamma / attention")
            axis.grid(alpha=0.2)
    fig.suptitle(f"Own {label}: Slot masks over corrected DRT", fontsize=15)
    fig.savefig(output / "slot_attention_examples.png", dpi=180)
    plt.close(fig)

    colors = comparison["Type"].map(
        {"Slot Attention": "#4C78A8", "1D CNN": "#F58518"}
    )
    fig, axis = plt.subplots(figsize=(11, 5.8))
    axis.barh(comparison["Method"], comparison["RMSE_pct_point"], color=colors)
    axis.invert_yaxis()
    axis.set_xlabel("Strict LOCO RMSE / SOH percentage points")
    axis.set_title(f"Own {label}: model comparison")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=180)
    plt.close(fig)


def existing_cnn_rows(scenario: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = RESULTS_ROOT / scenario.title()
    sources = [
        (base / "corrected_drt_only_1dcnn" / "aggregate_metrics.csv", "Corrected DRT only 1D CNN"),
        (
            base / "corrected_drt_physics_1dcnn" / "aggregate_metrics.csv",
            "Corrected DRT + physical parameters 1D CNN",
        ),
    ]
    for path, label in sources:
        if not path.exists():
            continue
        result = pd.read_csv(path).iloc[0]
        rows.append(
            {
                "Method": label,
                "RMSE_pct_point": float(result["RMSE_pct_point"]),
                "MAE_pct_point": float(result["MAE_pct_point"]),
                "R2": float(result["R2"]),
                "Type": "1D CNN",
            }
        )
    return rows


def run_scenario(
    scenario: str,
    frame: pd.DataFrame,
    gamma: np.ndarray,
    initial_gamma: np.ndarray,
    position: np.ndarray,
    log_tau: np.ndarray,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if scenario == "current":
        module = current_lct
        core = current_lct.core
        core.make_inputs = current_lct.make_robust_inputs
    else:
        module = initial_lct
        core = initial_lct.core
        core.make_inputs = initial_lct.make_robust_inputs
    core.CELL_IDS[:] = CELLS
    model_name = MODEL_NAMES[scenario]
    config = core.Config(max_epochs=max_epochs, patience=patience, min_soh=0.70)
    core.set_seed(config.seed)
    output = OUTPUTS[scenario]
    output.mkdir(parents=True, exist_ok=True)

    prediction_parts = []
    summary_parts = []
    history_parts = []
    attention_parts = []
    attention_index_parts = []
    start = time.perf_counter()
    print(f"\n[{scenario}] model={model_name}, rows={len(frame)}, device={device}", flush=True)
    for fold_index, test_cell in enumerate(CELLS, start=1):
        fold_start = time.perf_counter()
        common_arguments = (
            model_name,
            core.MODEL_SPECS[model_name],
            config,
            frame,
            gamma,
        )
        if scenario == "current":
            predictions, summaries, history, attention, attention_indices = (
                core.fit_fold(
                    *common_arguments,
                    position,
                    test_cell,
                    device,
                )
            )
        else:
            predictions, summaries, history, attention, attention_indices = (
                core.fit_fold(
                    *common_arguments,
                    initial_gamma,
                    position,
                    test_cell,
                    device,
                )
            )
        prediction_parts.append(predictions)
        history_parts.append(history)
        if len(summaries):
            summary_parts.append(summaries)
        if attention is not None and attention_indices is not None:
            attention_parts.append(attention)
            attention_index_parts.append(attention_indices)
        metric = core.metric_row(model_name, predictions)
        print(
            f"  {fold_index:02d}/08 {test_cell}: "
            f"RMSE={metric['RMSE_pct_point']:.3f}, epochs={len(history)}, "
            f"time={time.perf_counter() - fold_start:.1f}s",
            flush=True,
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    summaries = pd.concat(summary_parts, ignore_index=True)
    history = pd.concat(history_parts, ignore_index=True)
    metadata = frame.reset_index().rename(columns={"index": "RowIndex"})[
        ["RowIndex", "Cell", "Cycle", "SOH"]
    ]
    predictions = predictions.merge(metadata, on="RowIndex", how="left")
    summaries = summaries.merge(metadata, on="RowIndex", how="left")
    aggregate, folds = metric_table(core, predictions)
    attention = np.concatenate(attention_parts)
    attention_indices = np.concatenate(attention_index_parts)

    predictions.to_csv(output / "loco_predictions.csv", index=False, float_format="%.9g")
    summaries.to_csv(output / "attention_region_summaries.csv", index=False, float_format="%.9g")
    history.to_csv(output / "training_history.csv", index=False, float_format="%.9g")
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False, float_format="%.9g")
    folds.to_csv(output / "fold_metrics.csv", index=False, float_format="%.9g")
    np.savez_compressed(
        output / f"attention_{model_name}.npz",
        attention=attention,
        row_indices=attention_indices,
    )
    core.save_attention_diagnostics(output, summaries)

    result = aggregate.iloc[0]
    comparison_rows = [
        {
            "Method": SCENARIO_LABELS[scenario],
            "RMSE_pct_point": float(result["RMSE_pct_point"]),
            "MAE_pct_point": float(result["MAE_pct_point"]),
            "R2": float(result["R2"]),
            "Type": "Slot Attention",
        },
        *existing_cnn_rows(scenario),
    ]
    comparison = pd.DataFrame(comparison_rows).sort_values("RMSE_pct_point")
    comparison.to_csv(output / "comparison_with_1dcnn.csv", index=False)
    save_plots(
        output,
        scenario,
        frame,
        gamma,
        log_tau,
        aggregate,
        folds,
        predictions,
        summaries,
        attention,
        attention_indices,
        comparison,
    )

    lines = [
        f"# Own {SCENARIO_LABELS[scenario]}",
        "",
        f"- Cells: {', '.join(CELLS)}.",
        f"- Samples: {len(frame)}.",
        "- Evaluation: strict leave-one-cell-out with fold-local scaling.",
        "- Model: five-anchor Slot Attention with SOH-only Huber loss.",
        (
            "- Input: current corrected DRT only."
            if scenario == "current"
            else "- Input: initial, current, and current-minus-initial corrected DRT."
        ),
        "- Physical RL/CPE parameters are not supplied to the Slot model.",
        "",
        "| RMSE (pct-point) | MAE (pct-point) | R2 |",
        "|---:|---:|---:|",
        f"| {result['RMSE_pct_point']:.3f} | {result['MAE_pct_point']:.3f} | {result['R2']:.3f} |",
        "",
        "## Cell errors",
        "",
        "| Cell | RMSE (pct-point) | Bias (pct-point) |",
        "|---|---:|---:|",
    ]
    for row in folds.sort_values("RMSE_pct_point", ascending=False).itertuples():
        lines.append(
            f"| {row.TestCell} | {row.RMSE_pct_point:.3f} | "
            f"{100.0 * row.Bias_SOH:.3f} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_info = {
        "scenario": scenario,
        "model": model_name,
        "model_spec": core.MODEL_SPECS[model_name],
        "config": asdict(config),
        "cells": CELLS,
        "rows": len(frame),
        "input_channels": (
            ["current_drt"]
            if scenario == "current"
            else ["initial_drt", "current_drt", "delta_drt"]
        ),
        "uses_physical_parameters": False,
        "device": str(device),
        "torch_version": torch.__version__,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output / "run_configuration.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(aggregate.to_string(index=False), flush=True)
    return aggregate, folds, predictions


def save_overall_comparison(results: dict[str, pd.DataFrame]) -> None:
    rows = []
    for scenario, aggregate in results.items():
        result = aggregate.iloc[0]
        rows.append(
            {
                "Scenario": scenario,
                "Method": SCENARIO_LABELS[scenario],
                "RMSE_pct_point": float(result["RMSE_pct_point"]),
                "MAE_pct_point": float(result["MAE_pct_point"]),
                "R2": float(result["R2"]),
                "Type": "Slot Attention",
            }
        )
        rows.extend(
            {"Scenario": scenario, **row}
            for row in existing_cnn_rows(scenario)
        )
    comparison = pd.DataFrame(rows).sort_values("RMSE_pct_point")
    comparison.to_csv(RESULTS_ROOT / "slot_attention_1dcnn_comparison.csv", index=False)
    colors = comparison["Type"].map(
        {"Slot Attention": "#4C78A8", "1D CNN": "#F58518"}
    )
    fig, axis = plt.subplots(figsize=(12, 6.5))
    labels = comparison["Scenario"].str.title() + ": " + comparison["Method"]
    axis.barh(labels, comparison["RMSE_pct_point"], color=colors)
    axis.invert_yaxis()
    axis.set_xlabel("Strict LOCO RMSE / SOH percentage points")
    axis.set_title("Own corrected DRT: Slot Attention and 1D CNN")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_ROOT / "slot_attention_1dcnn_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Own corrected DRT Slot Attention",
        "",
        "- Cells: No.1-No.8; samples: 175.",
        "- Evaluation: strict leave-one-cell-out with fold-local scaling.",
        "- Slot model: five position anchors and SOH-only Huber loss.",
        "- Slot Attention inputs contain corrected DRT only, without RL/CPE parameters.",
        "",
        "| Scenario | Method | RMSE (pct-point) | MAE (pct-point) | R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples():
        lines.append(
            f"| {row.Scenario} | {row.Method} | {row.RMSE_pct_point:.3f} | "
            f"{row.MAE_pct_point:.3f} | {row.R2:.3f} |"
        )
    lines.extend(
        [
            "",
            "Current and Initial Slot artifacts are stored in their respective scenario directories.",
        ]
    )
    (RESULTS_ROOT / "slot_attention_README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drt", type=Path, default=DEFAULT_DRT)
    parser.add_argument("--tau", type=Path, default=DEFAULT_TAU)
    parser.add_argument("--min-soh", type=float, default=0.70)
    parser.add_argument("--max-epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["current", "initial"],
        default=["current", "initial"],
    )
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    frame, gamma, initial_gamma, position, log_tau = load_data(
        args.drt.resolve(), args.tau.resolve(), args.min_soh
    )
    results = {}
    for scenario in args.scenarios:
        aggregate, _, _ = run_scenario(
            scenario,
            frame,
            gamma,
            initial_gamma,
            position,
            log_tau,
            args.max_epochs,
            args.patience,
            device,
        )
        results[scenario] = aggregate
    save_overall_comparison(results)
    print("\nSaved to:")
    for scenario in args.scenarios:
        print(OUTPUTS[scenario])


if __name__ == "__main__":
    main()

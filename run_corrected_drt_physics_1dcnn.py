"""Run matched corrected-DRT 1D CNN experiments on the own dataset.

Four models are evaluated with strict leave-one-cell-out validation:

1. Current corrected DRT plus current physical parameters.
2. Current corrected DRT only.
3. Initial/current/delta corrected DRT plus matching parameter evolution.
4. Initial/current/delta corrected DRT only.

The CNN implementation and training protocol are imported from the LCT runner
so the comparison differs only in dataset and input representation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
LCT_DIR = WORKSPACE / "data" / "LCT"
if str(LCT_DIR) not in sys.path:
    sys.path.insert(0, str(LCT_DIR))

import run_corrected_drt_physics_1dcnn as engine  # noqa: E402


SOURCE_ROOT = SCRIPT_DIR / "results" / "modeling" / "newresults" / "paper_method_own_8cell_analysis"
DEFAULT_DRT = (
    SOURCE_ROOT
    / "02_corrected_drt"
    / "DRT_Distribution_after_RL_CPE_removal.csv"
)
DEFAULT_PARAMETERS = (
    SOURCE_ROOT
    / "01_split_frequency_correction"
    / "split_frequency_model_parameters.csv"
)
DEFAULT_TAU = SOURCE_ROOT / "02_corrected_drt" / "tau_vector.csv"
RESULTS_ROOT = SCRIPT_DIR / "results" / "modeling" / "newresults"
CELLS = [f"No.{index}" for index in range(1, 9)]

OUTPUTS = {
    ("current", True): RESULTS_ROOT
    / "Current"
    / "corrected_drt_physics_1dcnn",
    ("current", False): RESULTS_ROOT
    / "Current"
    / "corrected_drt_only_1dcnn",
    ("initial", True): RESULTS_ROOT
    / "Initial"
    / "corrected_drt_physics_1dcnn",
    ("initial", False): RESULTS_ROOT
    / "Initial"
    / "corrected_drt_only_1dcnn",
}


def valid_boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_own_data(
    drt_path: Path,
    parameter_path: Path,
    tau_path: Path,
    min_soh: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    drt = pd.read_csv(drt_path)
    gamma_columns = sorted(
        [column for column in drt if column.startswith("gamma_")],
        key=lambda column: int(column.split("_")[1]),
    )
    required_drt = {"Cell", "Cycle", "SOH", "FitSuccess"}
    missing_drt = required_drt.difference(drt.columns)
    if missing_drt or not gamma_columns:
        raise ValueError(
            f"Invalid corrected DRT table; missing={sorted(missing_drt)}, "
            f"gamma_columns={len(gamma_columns)}"
        )
    gamma_numeric = drt[gamma_columns].apply(pd.to_numeric, errors="coerce")
    valid = (
        drt["Cell"].isin(CELLS)
        & pd.to_numeric(drt["SOH"], errors="coerce").ge(min_soh)
        & valid_boolean(drt["FitSuccess"])
        & np.isfinite(gamma_numeric.to_numpy()).all(axis=1)
    )
    source_indices = np.flatnonzero(valid.to_numpy())
    frame = drt.loc[valid, ["Cell", "Cycle", "SOH"]].copy()
    frame["Cycle"] = pd.to_numeric(frame["Cycle"], errors="raise").astype(int)
    frame["SOH"] = pd.to_numeric(frame["SOH"], errors="raise").astype(float)
    gamma = gamma_numeric.iloc[source_indices].to_numpy(np.float32)
    if frame.duplicated(["Cell", "Cycle"]).any():
        raise ValueError("Corrected DRT contains duplicate Cell/Cycle rows")
    sort_positions = np.lexsort(
        (frame["Cycle"].to_numpy(), frame["Cell"].to_numpy())
    )
    frame = frame.iloc[sort_positions].reset_index(drop=True)
    gamma = gamma[sort_positions]

    parameters = pd.read_csv(parameter_path)
    required_parameters = {
        "Cell",
        "Cycle",
        "SOH",
        "cpe_status",
        *engine.PHYSICAL_PARAMETER_COLUMNS,
    }
    missing_parameters = required_parameters.difference(parameters.columns)
    if missing_parameters:
        raise ValueError(f"Missing parameter columns: {sorted(missing_parameters)}")
    parameters = parameters[parameters["Cell"].isin(CELLS)].copy()
    parameters["Cycle"] = pd.to_numeric(
        parameters["Cycle"], errors="raise"
    ).astype(int)
    if parameters.duplicated(["Cell", "Cycle"]).any():
        raise ValueError("Parameter table contains duplicate Cell/Cycle rows")

    parameters["cpe_available"] = (
        parameters["cpe_status"].astype(str).str.lower().eq("ok").astype(float)
    )
    invalid_cpe = parameters["cpe_available"].lt(0.5)
    parameters.loc[
        invalid_cpe, ["cpe_alpha", "cpe_Q", "r_pol_ohm"]
    ] = np.nan

    aligned = frame.merge(
        parameters[
            [
                "Cell",
                "Cycle",
                "SOH",
                "cpe_status",
                *engine.PHYSICAL_PARAMETER_COLUMNS,
                "cpe_available",
            ]
        ],
        on=["Cell", "Cycle"],
        how="left",
        validate="one_to_one",
        suffixes=("_drt", "_parameter"),
        sort=False,
    )
    if aligned["SOH_parameter"].isna().any():
        missing = aligned.loc[
            aligned["SOH_parameter"].isna(), ["Cell", "Cycle"]
        ].to_dict("records")
        raise ValueError(f"Missing aligned parameters: {missing[:5]}")
    if np.max(
        np.abs(
            aligned["SOH_drt"].to_numpy(float)
            - aligned["SOH_parameter"].to_numpy(float)
        )
    ) > 1e-9:
        raise ValueError("DRT and parameter SOH labels disagree")

    parameter_values = aligned[engine.PARAMETER_NAMES].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(np.float32)
    available_index = engine.PARAMETER_NAMES.index("cpe_available")
    unavailable = parameter_values[:, available_index] < 0.5
    expected_missing = [
        engine.PARAMETER_NAMES.index(name)
        for name in ["cpe_alpha", "cpe_Q", "r_pol_ohm"]
    ]
    missing_mask = ~np.isfinite(parameter_values)
    allowed_missing = np.zeros_like(missing_mask)
    allowed_missing[
        np.ix_(np.flatnonzero(unavailable), expected_missing)
    ] = True
    if np.any(missing_mask & ~allowed_missing):
        locations = np.argwhere(missing_mask & ~allowed_missing)[:5]
        raise ValueError(f"Unexpected missing parameters: {locations.tolist()}")

    tau = pd.read_csv(tau_path).sort_values("tau_index")
    if len(tau) != gamma.shape[1]:
        raise ValueError(f"Gamma/tau mismatch: {gamma.shape[1]} vs {len(tau)}")
    log_tau = (
        tau["log10_tau"].to_numpy(np.float32)
        if "log10_tau" in tau
        else np.log10(tau["tau_seconds"].to_numpy(np.float32))
    )
    missing_cells = sorted(set(CELLS).difference(frame["Cell"].unique()))
    if missing_cells:
        raise ValueError(f"No usable rows for cells: {missing_cells}")

    audit = (
        aligned.assign(CPEAvailable=~unavailable)
        .groupby("Cell", as_index=False)
        .agg(
            Rows=("Cycle", "size"),
            MinCycle=("Cycle", "min"),
            MaxCycle=("Cycle", "max"),
            MinSOH=("SOH_drt", "min"),
            MaxSOH=("SOH_drt", "max"),
            ValidCPE=("CPEAvailable", "sum"),
        )
    )
    return frame, gamma, parameter_values, log_tau, audit


def save_comparison(
    results: dict[
        tuple[str, bool], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ],
    outputs: dict[tuple[str, bool], Path],
) -> None:
    rows = []
    for key, (aggregate, _, _) in results.items():
        scenario, use_parameters = key
        result = aggregate.iloc[0]
        rows.append(
            {
                "Scenario": scenario,
                "UsePhysicalParameters": use_parameters,
                "Method": engine.scenario_label(scenario, use_parameters),
                "Rows": int(result["Rows"]),
                "RMSE_pct_point": float(result["RMSE_pct_point"]),
                "MAE_pct_point": float(result["MAE_pct_point"]),
                "R2": float(result["R2"]),
            }
        )
    comparison = pd.DataFrame(rows).sort_values("RMSE_pct_point")
    for output in outputs.values():
        comparison.to_csv(output / "corrected_drt_1dcnn_comparison.csv", index=False)

    colors = np.where(
        comparison["UsePhysicalParameters"], "#F58518", "#4C78A8"
    )
    fig, axis = plt.subplots(figsize=(11, 5.8))
    axis.barh(comparison["Method"], comparison["RMSE_pct_point"], color=colors)
    axis.invert_yaxis()
    axis.set_xlabel("Strict LOCO RMSE / SOH percentage points")
    axis.set_title("Own dataset: corrected DRT 1D CNN comparison")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for output in outputs.values():
        fig.savefig(output / "corrected_drt_1dcnn_comparison.png", dpi=180)
    plt.close(fig)

    for key, (aggregate, folds, predictions) in results.items():
        scenario, use_parameters = key
        output = outputs[key]
        local_comparison = comparison[["Method", "RMSE_pct_point"]].copy()
        local_comparison["Type"] = np.where(
            local_comparison["Method"].eq(
                engine.scenario_label(scenario, use_parameters)
            ),
            "This run",
            "Other requested scenario",
        )
        local_comparison.to_csv(
            output / "comparison_with_requested_models.csv", index=False
        )
        engine.save_scenario_plots(
            output,
            scenario,
            aggregate,
            folds,
            predictions,
            local_comparison,
            use_parameters,
        )

    ablation_rows = []
    for scenario in ["current", "initial"]:
        with_physics = float(
            results[(scenario, True)][0].iloc[0]["RMSE_pct_point"]
        )
        drt_only = float(
            results[(scenario, False)][0].iloc[0]["RMSE_pct_point"]
        )
        ablation_rows.append(
            {
                "Scenario": scenario,
                "DRT_only_RMSE_pct_point": drt_only,
                "DRT_plus_physics_RMSE_pct_point": with_physics,
                "Physics_RMSE_reduction_pct_point": drt_only - with_physics,
                "Physics_relative_RMSE_reduction_pct": (
                    100.0 * (drt_only - with_physics) / drt_only
                ),
            }
        )
        only_folds = results[(scenario, False)][1][
            ["TestCell", "RMSE_pct_point"]
        ].rename(columns={"RMSE_pct_point": "DRT_only_RMSE_pct_point"})
        physics_folds = results[(scenario, True)][1][
            ["TestCell", "RMSE_pct_point"]
        ].rename(columns={"RMSE_pct_point": "DRT_plus_physics_RMSE_pct_point"})
        fold_comparison = only_folds.merge(
            physics_folds, on="TestCell", validate="one_to_one"
        )
        fold_comparison["Physics_RMSE_reduction_pct_point"] = (
            fold_comparison["DRT_only_RMSE_pct_point"]
            - fold_comparison["DRT_plus_physics_RMSE_pct_point"]
        )
        for use_parameters in [False, True]:
            fold_comparison.to_csv(
                outputs[(scenario, use_parameters)]
                / "physical_parameter_ablation_by_cell.csv",
                index=False,
            )

        positions = np.arange(len(fold_comparison))
        width = 0.38
        fig, axis = plt.subplots(figsize=(11, 5.8))
        axis.bar(
            positions - width / 2,
            fold_comparison["DRT_only_RMSE_pct_point"],
            width,
            label="Corrected DRT only",
            color="#4C78A8",
        )
        axis.bar(
            positions + width / 2,
            fold_comparison["DRT_plus_physics_RMSE_pct_point"],
            width,
            label="Corrected DRT + physical parameters",
            color="#F58518",
        )
        axis.set_xticks(positions, fold_comparison["TestCell"])
        axis.set_ylabel("LOCO RMSE / SOH percentage points")
        axis.set_title(f"Own {scenario}: physical-parameter ablation")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        for use_parameters in [False, True]:
            fig.savefig(
                outputs[(scenario, use_parameters)]
                / "physical_parameter_ablation_by_cell.png",
                dpi=180,
            )
        plt.close(fig)

    ablation = pd.DataFrame(ablation_rows)
    for output in outputs.values():
        ablation.to_csv(
            output / "corrected_drt_physical_parameter_ablation.csv",
            index=False,
        )
    lines = [
        "# Own corrected DRT 1D CNN experiments",
        "",
        "- Cells: No.1-No.8.",
        f"- Samples: {int(comparison['Rows'].iloc[0])} labeled EIS curves.",
        "- Evaluation: strict leave-one-cell-out.",
        "- Current and Initial results are stored separately.",
        "- Invalid CPE-derived alpha/Q/Rpol values are fold-locally imputed.",
        "",
        "| Method | RMSE (pct-point) | MAE (pct-point) | R2 |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples():
        lines.append(
            f"| {row.Method} | {row.RMSE_pct_point:.3f} | "
            f"{row.MAE_pct_point:.3f} | {row.R2:.3f} |"
        )
    readme_text = "\n".join(lines) + "\n"
    for output in outputs.values():
        (output / "corrected_drt_1dcnn_README.md").write_text(readme_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drt", type=Path, default=DEFAULT_DRT)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--tau", type=Path, default=DEFAULT_TAU)
    parser.add_argument("--min-soh", type=float, default=0.70)
    parser.add_argument("--max-epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    engine.CELLS = CELLS
    engine.core.CELLS = CELLS
    config = engine.core.Config(
        min_soh=args.min_soh,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    engine.core.set_seed(config.seed)
    frame, current_drt, current_parameters, log_tau, audit = load_own_data(
        args.drt.resolve(),
        args.parameters.resolve(),
        args.tau.resolve(),
        config.min_soh,
    )
    for output in OUTPUTS.values():
        output.mkdir(parents=True, exist_ok=True)
        audit.to_csv(output / "corrected_drt_1dcnn_input_audit.csv", index=False)

    results = {}
    for scenario in ["current", "initial"]:
        for use_parameters in [True, False]:
            key = (scenario, use_parameters)
            results[key] = engine.run_scenario(
                scenario,
                OUTPUTS[key],
                args.drt.resolve(),
                args.parameters.resolve(),
                config,
                frame,
                current_drt,
                current_parameters,
                log_tau,
                device,
                use_parameters,
            )
    save_comparison(results, OUTPUTS)
    run_info = {
        "drt_source": str(args.drt.resolve()),
        "parameter_source": str(args.parameters.resolve()),
        "tau_source": str(args.tau.resolve()),
        "cells": CELLS,
        "rows": len(frame),
        "device": str(device),
        "config": config.__dict__,
        "outputs": {f"{key[0]}_{key[1]}": str(path) for key, path in OUTPUTS.items()},
    }
    run_info_text = json.dumps(run_info, indent=2)
    for output in OUTPUTS.values():
        (output / "corrected_drt_1dcnn_run_configuration.json").write_text(
            run_info_text, encoding="utf-8"
        )
    print("\nSaved results to:")
    for output in OUTPUTS.values():
        print(" ", output)


if __name__ == "__main__":
    main()

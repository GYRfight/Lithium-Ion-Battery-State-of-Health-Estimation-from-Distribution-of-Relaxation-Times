"""Analyze optimized own-cell EIS data with raw DRT, ECM, and SOH models.

The pipeline reads EIS1_optimized.csv ... EIS8_optimized.csv, builds a raw
non-negative DRT representation and a physical equivalent-circuit fit, then
compares several feature sets under pooled and cross-cell validation.

DEPRECATED as a CLI entry point: running this file's ``main()`` re-derives
spectra from the raw per-cell CSVs and recomputes DRT/ECM/prediction in one
shot, duplicating what the staged ``pipeline_00_prepare_inputs.py`` ->
``pipeline_01_raw_drt.py`` -> ``pipeline_02_ecm.py`` -> ``pipeline_03_features.py``
-> ``pipeline_04_soh_prediction.py`` scripts now do (each reading the
previous stage's saved output instead of recomputing it). Prefer those, or
``pipeline_run_all.py`` to run all five in order.

This module's functions are kept unchanged and still imported directly by
``pipeline_00_prepare_inputs.py``/``pipeline_01_raw_drt.py``/
``pipeline_02_ecm.py``/``pipeline_03_features.py``/``pipeline_04_soh_prediction.py``
and by ``paper_method_8cell_analysis.py``, ``paper_method_ecm_soh_prediction.py``,
and ``data/EIS/paper_method_feature_extraction.py`` -- do not remove or
rename functions here without checking those callers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.signal import find_peaks
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EIS_ROOT = SCRIPT_DIR / "data" / "EIS"
CAPACITY_FILE = SCRIPT_DIR / "data" / "capacity.csv"
FREQUENCY_FILE = SCRIPT_DIR / "data" / "freq_index.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "modeling" / "newresults"
CELL_IDS = [f"No.{index}" for index in range(1, 9)]
N_FREQ = 80
RE_COLUMNS = [f"Zreal_{index}" for index in range(1, N_FREQ + 1)]
IM_COLUMNS = [f"Zimag_{index}" for index in range(1, N_FREQ + 1)]
RANDOM_STATE = 42

ECM_POSITIVE_PARAMETERS = [
    "Rs_ohm",
    "L_H",
    "R_SEI_ohm",
    "C_SE_F",
    "tau_SE_s",
    "R_CT_ohm",
    "Q_CT",
    "tau_CT_s",
    "Q_W",
    "f_SE_Hz",
    "f_CT_Hz",
]


@dataclass
class Spectrum:
    cell: str
    cycle: int
    frequency_hz: np.ndarray
    z_ohm: np.ndarray
    source_file: str
    capacity_mah: float = np.nan
    initial_capacity_mah: float = np.nan
    soh: float = np.nan

    def metadata(self) -> dict[str, object]:
        return {
            "Cell": self.cell,
            "Cycle": self.cycle,
            "SourceFile": self.source_file,
            "capacity_mAh": self.capacity_mah,
            "initial_capacity_mAh": self.initial_capacity_mah,
            "SOH": self.soh,
        }


@dataclass
class RawDRTFit:
    r_inf_ohm: float
    inductance_h: float
    gamma: np.ndarray
    z_fit: np.ndarray
    nrmse: float
    rmse_ohm: float
    cost: float
    success: bool
    message: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frequency(path: Path) -> np.ndarray:
    values: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)
        if len(numbers) >= 2:
            values.append((int(float(numbers[0])), float(numbers[1])))
    if len(values) != N_FREQ:
        raise ValueError(f"Expected {N_FREQ} frequencies in {path}, found {len(values)}")
    values.sort(key=lambda item: item[0])
    frequency = np.asarray([value for _, value in values], dtype=float)
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0):
        raise ValueError("Frequency vector contains invalid values")
    return frequency


def parse_capacity_cycle(value: object) -> float:
    text = str(value).strip()
    interval = re.search(r"-\s*(\d+)\s*$", text)
    if interval:
        return float(interval.group(1))
    chinese_date = re.fullmatch(r"\s*\d+\s*月\s*(\d+)\s*日\s*", text)
    if chinese_date:
        return float(chinese_date.group(1))
    return np.nan


def load_capacity_labels(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    source["Cycle"] = source["id"].map(parse_capacity_cycle)
    rows: list[pd.DataFrame] = []
    for cell in CELL_IDS:
        capacity = pd.to_numeric(source[cell], errors="coerce")
        cell_frame = pd.DataFrame(
            {
                "Cell": cell,
                "Cycle": source["Cycle"],
                "capacity_mAh": capacity,
            }
        ).dropna(subset=["Cycle", "capacity_mAh"])
        cell_frame["Cycle"] = cell_frame["Cycle"].round().astype(int)
        cell_frame = cell_frame.sort_values("Cycle").drop_duplicates("Cycle", keep="last")
        initial = float(cell_frame["capacity_mAh"].iloc[0])
        cell_frame["initial_capacity_mAh"] = initial
        cell_frame["SOH"] = cell_frame["capacity_mAh"] / initial
        rows.append(cell_frame)
    return pd.concat(rows, ignore_index=True)


def optimized_path(cell_number: int) -> Path:
    path = EIS_ROOT / f"No.{cell_number}" / f"EIS{cell_number}_optimized.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing optimized EIS file: {path}")
    return path


def load_spectra(
    frequency: np.ndarray,
    labels: pd.DataFrame,
    max_curves_per_cell: int | None,
) -> tuple[list[Spectrum], pd.DataFrame]:
    label_lookup = labels.set_index(["Cell", "Cycle"])
    spectra: list[Spectrum] = []
    audit_rows: list[dict[str, object]] = []
    for cell_number, cell in enumerate(CELL_IDS, start=1):
        path = optimized_path(cell_number)
        frame = pd.read_csv(path)
        missing = [column for column in ["Cycle", *RE_COLUMNS, *IM_COLUMNS] if column not in frame]
        if missing:
            raise ValueError(f"{path} missing columns: {missing[:10]}")
        frame["Cycle"] = pd.to_numeric(frame["Cycle"], errors="coerce")
        frame = frame.dropna(subset=["Cycle"]).copy()
        frame["Cycle"] = frame["Cycle"].round().astype(int)
        frame = frame.sort_values("Cycle").drop_duplicates("Cycle", keep="last")
        if max_curves_per_cell and len(frame) > max_curves_per_cell:
            positions = np.linspace(0, len(frame) - 1, max_curves_per_cell).round().astype(int)
            frame = frame.iloc[np.unique(positions)].copy()

        numeric = frame[RE_COLUMNS + IM_COLUMNS].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise ValueError(f"{path} contains non-numeric or missing impedance values")

        labeled = 0
        for row_index, row in frame.iterrows():
            cycle = int(row["Cycle"])
            key = (cell, cycle)
            if key in label_lookup.index:
                label = label_lookup.loc[key]
                capacity = float(label["capacity_mAh"])
                initial = float(label["initial_capacity_mAh"])
                soh = float(label["SOH"])
                labeled += 1
            else:
                capacity = initial = soh = np.nan
            z_real = numeric.loc[row_index, RE_COLUMNS].to_numpy(dtype=float)
            z_imag = numeric.loc[row_index, IM_COLUMNS].to_numpy(dtype=float)
            spectra.append(
                Spectrum(
                    cell=cell,
                    cycle=cycle,
                    frequency_hz=frequency.copy(),
                    z_ohm=z_real + 1j * z_imag,
                    source_file=str(path),
                    capacity_mah=capacity,
                    initial_capacity_mah=initial,
                    soh=soh,
                )
            )

        audit_rows.append(
            {
                "Cell": cell,
                "SourceFile": str(path),
                "SHA256": sha256(path),
                "Curves": len(frame),
                "LabeledCurves": labeled,
                "UnlabeledCurves": len(frame) - labeled,
                "CycleMin": int(frame["Cycle"].min()),
                "CycleMax": int(frame["Cycle"].max()),
                "FrequencyPoints": len(frequency),
                "FrequencyMinHz": float(frequency.min()),
                "FrequencyMaxHz": float(frequency.max()),
                "MissingImpedanceValues": int(numeric.isna().sum().sum()),
            }
        )
    return spectra, pd.DataFrame(audit_rows)


def second_difference(size: int, spacing: float) -> np.ndarray:
    if size < 3:
        return np.zeros((0, size))
    return np.diff(np.eye(size), n=2, axis=0) / (spacing**2)


def make_tau_grid(frequency: np.ndarray, count: int) -> np.ndarray:
    tau_min = 1.0 / (2.0 * np.pi * float(frequency.max()))
    tau_max = 1.0 / (2.0 * np.pi * float(frequency.min()))
    return np.logspace(np.log10(tau_min), np.log10(tau_max), count)


def raw_drt_matrix(frequency: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, float]:
    omega = 2.0 * np.pi * frequency
    dln_tau = float(np.mean(np.diff(np.log(tau))))
    product = omega[:, None] * tau[None, :]
    denominator = 1.0 + product**2
    real_kernel = dln_tau / denominator
    imag_kernel = -dln_tau * product / denominator

    rows = 2 * len(frequency)
    matrix = np.zeros((rows, len(tau) + 2), dtype=float)
    matrix[: len(frequency), 0] = 1.0
    matrix[len(frequency) :, 1] = omega * 1e-6
    matrix[: len(frequency), 2:] = real_kernel
    matrix[len(frequency) :, 2:] = imag_kernel
    return matrix, dln_tau


def fit_raw_drt(
    spectrum: Spectrum,
    tau: np.ndarray,
    lambda_drt: float,
) -> RawDRTFit:
    matrix, dln_tau = raw_drt_matrix(spectrum.frequency_hz, tau)
    target = np.concatenate([spectrum.z_ohm.real, spectrum.z_ohm.imag])
    scale = max(float(np.sqrt(np.mean(np.abs(spectrum.z_ohm) ** 2))), 1e-9)
    regularization = np.zeros((max(len(tau) - 2, 0), len(tau) + 2), dtype=float)
    if len(regularization):
        regularization[:, 2:] = np.sqrt(lambda_drt) * second_difference(len(tau), dln_tau)
    augmented_matrix = np.vstack([matrix / scale, regularization])
    augmented_target = np.concatenate([target / scale, np.zeros(len(regularization))])
    lower = np.zeros(len(tau) + 2)
    upper = np.full(len(tau) + 2, np.inf)
    upper[0] = 10.0
    upper[1] = 10000.0
    result = lsq_linear(
        augmented_matrix,
        augmented_target,
        bounds=(lower, upper),
        method="trf",
        tol=1e-9,
        max_iter=4000,
    )
    parameters = result.x
    prediction = matrix @ parameters
    count = len(spectrum.frequency_hz)
    z_fit = prediction[:count] + 1j * prediction[count:]
    residual = z_fit - spectrum.z_ohm
    rmse = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
    nrmse = rmse / scale
    return RawDRTFit(
        r_inf_ohm=float(parameters[0]),
        inductance_h=float(parameters[1] * 1e-6),
        gamma=parameters[2:].copy(),
        z_fit=z_fit,
        nrmse=nrmse,
        rmse_ohm=rmse,
        cost=float(result.cost),
        success=bool(result.success and np.all(np.isfinite(parameters))),
        message=str(result.message),
    )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    if cumulative[-1] <= 0:
        return np.nan
    return float(np.interp(quantile * cumulative[-1], cumulative, ordered_values))


def extract_drt_features(tau: np.ndarray, gamma: np.ndarray) -> dict[str, float]:
    gamma = np.clip(np.asarray(gamma, dtype=float), 0, None)
    smooth = gaussian_filter1d(gamma, sigma=1.0, mode="nearest")
    log_tau = np.log10(tau)
    ln_tau = np.log(tau)
    total_area = float(np.trapezoid(gamma, ln_tau))
    weights = np.maximum(gamma, 0) * float(np.mean(np.diff(ln_tau)))
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        weights = np.ones_like(gamma)
        weight_sum = float(weights.sum())
    probability = weights / weight_sum
    centroid = float(np.sum(probability * log_tau))
    spread = float(np.sqrt(np.sum(probability * (log_tau - centroid) ** 2)))
    entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-30))))
    dominant = int(np.argmax(smooth))

    features: dict[str, float] = {
        "drt_total_area_ohm": total_area,
        "drt_max_gamma_ohm": float(smooth[dominant]),
        "drt_dominant_tau_s": float(tau[dominant]),
        "drt_centroid_log10_tau": centroid,
        "drt_spread_log10_tau": spread,
        "drt_entropy": entropy,
        "drt_tau_q10_s": 10 ** weighted_quantile(log_tau, weights, 0.10),
        "drt_tau_q50_s": 10 ** weighted_quantile(log_tau, weights, 0.50),
        "drt_tau_q90_s": 10 ** weighted_quantile(log_tau, weights, 0.90),
    }

    edges = np.linspace(float(log_tau.min()), float(log_tau.max()), 13)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        mask = (log_tau >= left) & (log_tau <= right if index == 12 else log_tau < right)
        area = float(np.trapezoid(gamma[mask], ln_tau[mask])) if mask.sum() >= 2 else 0.0
        features[f"drt_bin_{index:02d}_area_ohm"] = area
        features[f"drt_bin_{index:02d}_area_fraction"] = area / total_area if total_area > 0 else 0.0

    prominence = max(float(smooth.max()) * 0.02, 1e-12)
    peaks, properties = find_peaks(smooth, prominence=prominence, distance=5)
    ranked = sorted(
        zip(peaks, properties.get("prominences", np.zeros(len(peaks)))),
        key=lambda pair: pair[1],
        reverse=True,
    )
    features["drt_detected_peak_count"] = float(len(peaks))
    for rank in range(1, 5):
        if rank <= len(ranked):
            peak_index, peak_prominence = ranked[rank - 1]
            features[f"drt_peak_{rank}_tau_s"] = float(tau[peak_index])
            features[f"drt_peak_{rank}_gamma_ohm"] = float(smooth[peak_index])
            features[f"drt_peak_{rank}_prominence_ohm"] = float(peak_prominence)
        else:
            features[f"drt_peak_{rank}_tau_s"] = np.nan
            features[f"drt_peak_{rank}_gamma_ohm"] = np.nan
            features[f"drt_peak_{rank}_prominence_ohm"] = np.nan
    return features


def run_raw_drt(
    spectra: list[Spectrum],
    tau: np.ndarray,
    lambda_drt: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, int], RawDRTFit]]:
    distribution_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    fits: dict[tuple[str, int], RawDRTFit] = {}
    total = len(spectra)
    for index, spectrum in enumerate(spectra, start=1):
        fit = fit_raw_drt(spectrum, tau, lambda_drt)
        fits[(spectrum.cell, spectrum.cycle)] = fit
        distribution_row = {
            **spectrum.metadata(),
            "drt_r_inf_ohm": fit.r_inf_ohm,
            "drt_L_H": fit.inductance_h,
        }
        distribution_row.update(
            {f"gamma_{tau_index:03d}": float(value) for tau_index, value in enumerate(fit.gamma)}
        )
        distribution_rows.append(distribution_row)
        feature_rows.append(
            {
                **spectrum.metadata(),
                "drt_r_inf_ohm": fit.r_inf_ohm,
                "drt_L_H": fit.inductance_h,
                **extract_drt_features(tau, fit.gamma),
            }
        )
        quality_rows.append(
            {
                **spectrum.metadata(),
                "drt_fit_success": fit.success,
                "drt_fit_nrmse": fit.nrmse,
                "drt_fit_rmse_ohm": fit.rmse_ohm,
                "drt_fit_cost": fit.cost,
                "drt_fit_message": fit.message,
            }
        )
        if index == 1 or index % 20 == 0 or index == total:
            print(f"  raw DRT {index}/{total}: {spectrum.cell} cycle {spectrum.cycle}", flush=True)

    distribution = pd.DataFrame(distribution_rows)
    features = pd.DataFrame(feature_rows)
    quality = pd.DataFrame(quality_rows)
    distribution.to_csv(output_dir / "raw_drt_distribution.csv", index=False, float_format="%.10g")
    features.to_csv(output_dir / "raw_drt_engineered_features.csv", index=False, float_format="%.10g")
    quality.to_csv(output_dir / "raw_drt_fit_quality.csv", index=False, float_format="%.10g")
    pd.DataFrame(
        {
            "tau_index": np.arange(len(tau)),
            "tau_s": tau,
            "log10_tau": np.log10(tau),
            "equivalent_frequency_hz": 1.0 / (2.0 * np.pi * tau),
        }
    ).to_csv(output_dir / "tau_vector.csv", index=False, float_format="%.10g")
    return distribution, features, quality, fits


def import_ecm_functions():
    module_dir = PROJECT_ROOT / "data" / "EIS"
    sys.path.insert(0, str(module_dir))
    from ecm_l_rs_sei_ct_cpew_soh_25C01 import circuit_impedance, fit_one

    return circuit_impedance, fit_one


def ecm_boundary_flags(row: dict[str, object]) -> tuple[int, str]:
    flags: list[str] = []
    if float(row["alpha_CT"]) <= 0.205 or float(row["alpha_CT"]) >= 0.995:
        flags.append("alpha_CT")
    if float(row["alpha_W"]) <= 0.205 or float(row["alpha_W"]) >= 0.995:
        flags.append("alpha_W")
    limits = {
        "Rs_ohm": (1.05e-5, 9.5),
        "L_H": (1.05e-10, 9.5e-3),
        "R_SEI_ohm": (1.05e-6, 19.0),
        "tau_SE_s": (1.05e-7, 9.5e4),
        "R_CT_ohm": (1.05e-6, 19.0),
        "tau_CT_s": (1.05e-7, 9.5e5),
        "Q_W": (1.05e-8, 9.5e5),
    }
    for parameter, (lower, upper) in limits.items():
        value = float(row[parameter])
        if value <= lower or value >= upper:
            flags.append(parameter)
    return len(flags), ";".join(flags)


def run_ecm(
    spectra: list[Spectrum],
    output_dir: Path,
    max_nrmse: float,
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    circuit_impedance, fit_one = import_ecm_functions()
    rows: list[dict[str, object]] = []
    fitted_spectra: dict[tuple[str, int], np.ndarray] = {}
    total = len(spectra)
    for index, spectrum in enumerate(spectra, start=1):
        try:
            fit = fit_one(spectrum.frequency_hz, spectrum.z_ohm)
            predicted = circuit_impedance(
                spectrum.frequency_hz,
                fit["Rs_ohm"],
                fit["L_H"],
                fit["R_SEI_ohm"],
                fit["tau_SE_s"],
                fit["R_CT_ohm"],
                fit["tau_CT_s"],
                fit["alpha_CT"],
                fit["Q_W"],
                fit["alpha_W"],
            )
            boundary_count, boundary_parameters = ecm_boundary_flags(fit)
            fit.update(
                {
                    **spectrum.metadata(),
                    "ecm_boundary_count": boundary_count,
                    "ecm_boundary_parameters": boundary_parameters,
                    "ecm_usable": bool(
                        fit["fit_success"]
                        and np.isfinite(fit["fit_nrmse"])
                        and float(fit["fit_nrmse"]) <= max_nrmse
                    ),
                }
            )
            fitted_spectra[(spectrum.cell, spectrum.cycle)] = predicted
            rows.append(fit)
        except Exception as error:
            rows.append(
                {
                    **spectrum.metadata(),
                    "fit_success": False,
                    "fit_message": f"{type(error).__name__}: {error}",
                    "ecm_usable": False,
                    "ecm_boundary_count": np.nan,
                    "ecm_boundary_parameters": "fit_exception",
                }
            )
        if index == 1 or index % 20 == 0 or index == total:
            print(f"  ECM {index}/{total}: {spectrum.cell} cycle {spectrum.cycle}", flush=True)

    parameters = pd.DataFrame(rows).sort_values(["Cell", "Cycle"]).reset_index(drop=True)
    parameters.to_csv(output_dir / "ecm_fitted_parameters.csv", index=False, float_format="%.10g")
    return parameters, fitted_spectra


def build_ecm_model_features(parameters: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = parameters[["Cell", "Cycle"]].copy()
    feature_columns: list[str] = []
    for parameter in ECM_POSITIVE_PARAMETERS:
        column = f"ecm_log10_{parameter}"
        values = pd.to_numeric(parameters.get(parameter), errors="coerce")
        frame[column] = np.log10(values.where(values > 0))
        feature_columns.append(column)
    for parameter in ["alpha_CT", "alpha_W"]:
        column = f"ecm_{parameter}"
        frame[column] = pd.to_numeric(parameters.get(parameter), errors="coerce")
        feature_columns.append(column)
    r_total = pd.to_numeric(parameters.get("R_SEI_ohm"), errors="coerce") + pd.to_numeric(
        parameters.get("R_CT_ohm"), errors="coerce"
    )
    frame["ecm_log10_R_SEI_plus_R_CT_ohm"] = np.log10(r_total.where(r_total > 0))
    feature_columns.append("ecm_log10_R_SEI_plus_R_CT_ohm")
    return frame, feature_columns


def add_initial_evolution_features(
    frame: pd.DataFrame,
    source_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    result = frame.sort_values(["Cell", "Cycle"]).copy()
    evolution_columns: list[str] = []
    for column in source_columns:
        initial_map: dict[str, float] = {}
        for cell, group in result.groupby("Cell", sort=False):
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values[np.isfinite(values)]
            initial_map[cell] = float(valid.iloc[0]) if len(valid) else np.nan
        evolution_column = f"evolution_delta_initial__{column}"
        result[evolution_column] = pd.to_numeric(result[column], errors="coerce") - result["Cell"].map(initial_map)
        evolution_columns.append(evolution_column)
    return result, evolution_columns


def make_models(n_train: int, n_features: int) -> dict[str, object]:
    ridge_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        (
            "clip_scaled",
            FunctionTransformer(
                np.clip,
                kw_args={"a_min": -10.0, "a_max": 10.0},
                feature_names_out="one-to-one",
            ),
        ),
    ]
    if n_features > 40:
        # Keep the linear baseline away from near-null PCA directions.  Those
        # directions can vary strongly in a held-out cell even though they have
        # almost no variance in a small training fold.
        components = max(2, min(10, n_features, n_train // 3))
        ridge_steps.append(("pca", PCA(n_components=components, random_state=RANDOM_STATE)))
    ridge_cv = max(2, min(5, n_train // 8))
    ridge_steps.append(
        (
            "model",
            RidgeCV(
                alphas=np.logspace(-2, 4, 19),
                cv=ridge_cv,
                scoring="neg_mean_squared_error",
            ),
        )
    )
    return {
        "Ridge": Pipeline(ridge_steps),
        "RandomForest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "ExtraTrees": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        max_features=0.7,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    return {
        "MAE_SOH": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_SOH": float(np.sqrt(np.mean(error**2))),
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "Bias_SOH": float(np.mean(error)),
        "MaxAbsError_SOH": float(np.max(np.abs(error))),
        "MAE_pct_point": float(mean_absolute_error(y_true, y_pred) * 100.0),
        "RMSE_pct_point": float(np.sqrt(np.mean(error**2)) * 100.0),
    }


def protocol_splits(frame: pd.DataFrame) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    indices = np.arange(len(frame))
    groups = frame["Cell"].to_numpy()
    logo = LeaveOneGroupOut()
    leave_cell = [
        (str(np.unique(groups[test])[0]), train, test)
        for train, test in logo.split(indices, groups=groups)
    ]
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pooled = [(f"fold_{fold}", train, test) for fold, (train, test) in enumerate(kfold.split(indices), start=1)]
    train, test = train_test_split(
        indices,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=groups,
    )
    return {
        "leave_one_cell_out": leave_cell,
        "pooled_5fold": pooled,
        "random_75_25": [("split_1", train, test)],
    }


def usable_feature_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    usable: list[str] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if len(finite) >= 3 and float(finite.var(ddof=0)) > 1e-18:
            usable.append(column)
    return usable


def evaluate_feature_sets(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    labeled = frame[frame["SOH"].notna()].sort_values(["Cell", "Cycle"]).reset_index(drop=True)
    y = labeled["SOH"].to_numpy(dtype=float)
    splits = protocol_splits(labeled)
    fold_metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    actual_features: dict[str, list[str]] = {}

    for feature_set, requested_columns in feature_sets.items():
        columns = usable_feature_columns(labeled, requested_columns)
        actual_features[feature_set] = columns
        if not columns:
            print(f"  skipping empty feature set: {feature_set}")
            continue
        x = labeled[columns].replace([np.inf, -np.inf], np.nan)
        print(f"  prediction feature set {feature_set}: {len(columns)} features", flush=True)
        for protocol, protocol_folds in splits.items():
            for fold_name, train_indices, test_indices in protocol_folds:
                models = make_models(len(train_indices), len(columns))
                for model_name, model in models.items():
                    model.fit(x.iloc[train_indices], y[train_indices])
                    predicted = model.predict(x.iloc[test_indices])
                    metrics = metric_values(y[test_indices], predicted)
                    fold_metric_rows.append(
                        {
                            "Protocol": protocol,
                            "Fold": fold_name,
                            "FeatureSet": feature_set,
                            "Model": model_name,
                            "TrainRows": len(train_indices),
                            "TestRows": len(test_indices),
                            "FeatureCount": len(columns),
                            **metrics,
                        }
                    )
                    for local_index, row_index in enumerate(test_indices):
                        metadata = labeled.iloc[row_index]
                        prediction_rows.append(
                            {
                                "Protocol": protocol,
                                "Fold": fold_name,
                                "FeatureSet": feature_set,
                                "Model": model_name,
                                "Cell": metadata["Cell"],
                                "Cycle": int(metadata["Cycle"]),
                                "SOH_true": float(y[row_index]),
                                "SOH_pred": float(predicted[local_index]),
                            }
                        )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    aggregate_rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(["Protocol", "FeatureSet", "Model"], sort=False):
        protocol, feature_set, model = keys
        aggregate_rows.append(
            {
                "Protocol": protocol,
                "FeatureSet": feature_set,
                "Model": model,
                "Rows": len(group),
                "FeatureCount": len(actual_features[feature_set]),
                **metric_values(group["SOH_true"].to_numpy(), group["SOH_pred"].to_numpy()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["Protocol", "RMSE_SOH", "MAE_SOH"]
    )
    fold_metrics.to_csv(output_dir / "prediction_fold_metrics.csv", index=False, float_format="%.10g")
    aggregate.to_csv(output_dir / "prediction_aggregate_metrics.csv", index=False, float_format="%.10g")
    predictions["Error"] = predictions["SOH_pred"] - predictions["SOH_true"]
    predictions.to_csv(output_dir / "soh_predictions.csv", index=False, float_format="%.10g")
    feature_manifest = pd.DataFrame(
        [
            {"FeatureSet": feature_set, "Feature": feature, "FeatureIndex": index}
            for feature_set, columns in actual_features.items()
            for index, feature in enumerate(columns, start=1)
        ]
    )
    feature_manifest.to_csv(output_dir / "model_feature_manifest.csv", index=False)
    return fold_metrics, aggregate, predictions, actual_features


def cross_cell_permutation_importance(
    frame: pd.DataFrame,
    aggregate: pd.DataFrame,
    actual_features: dict[str, list[str]],
    output_dir: Path,
) -> pd.DataFrame:
    candidates = aggregate[aggregate["Protocol"].eq("leave_one_cell_out")]
    best = candidates.sort_values(["RMSE_SOH", "MAE_SOH"]).iloc[0]
    feature_set = str(best["FeatureSet"])
    model_name = str(best["Model"])
    columns = actual_features[feature_set]
    labeled = frame[frame["SOH"].notna()].sort_values(["Cell", "Cycle"]).reset_index(drop=True)
    x = labeled[columns].replace([np.inf, -np.inf], np.nan)
    y = labeled["SOH"].to_numpy(dtype=float)
    groups = labeled["Cell"].to_numpy()
    rows: list[dict[str, object]] = []
    for train_indices, test_indices in LeaveOneGroupOut().split(x, y, groups):
        model = make_models(len(train_indices), len(columns))[model_name]
        model.fit(x.iloc[train_indices], y[train_indices])
        importance = permutation_importance(
            model,
            x.iloc[test_indices],
            y[test_indices],
            scoring="neg_root_mean_squared_error",
            n_repeats=12,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        test_cell = str(np.unique(groups[test_indices])[0])
        for feature, mean, std in zip(columns, importance.importances_mean, importance.importances_std):
            rows.append(
                {
                    "FeatureSet": feature_set,
                    "Model": model_name,
                    "TestCell": test_cell,
                    "Feature": feature,
                    "ImportanceMean": float(mean),
                    "ImportanceStd": float(std),
                }
            )
    fold_importance = pd.DataFrame(rows)
    summary = (
        fold_importance.groupby(["FeatureSet", "Model", "Feature"], as_index=False)
        .agg(
            ImportanceMean=("ImportanceMean", "mean"),
            ImportanceAcrossCellStd=("ImportanceMean", "std"),
            MeanRepeatStd=("ImportanceStd", "mean"),
        )
        .sort_values("ImportanceMean", ascending=False)
    )
    fold_importance.to_csv(output_dir / "permutation_importance_by_cell.csv", index=False, float_format="%.10g")
    summary.to_csv(output_dir / "permutation_importance_summary.csv", index=False, float_format="%.10g")
    return summary


def cell_axes(count: int, width: float = 18.0, height: float = 8.5) -> tuple[plt.Figure, np.ndarray]:
    columns = 4
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(width, height * rows / 2), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[count:]:
        axis.axis("off")
    return figure, axes


def representative_spectra(spectra: list[Spectrum]) -> list[Spectrum]:
    selected: list[Spectrum] = []
    for cell in CELL_IDS:
        cell_spectra = sorted([spectrum for spectrum in spectra if spectrum.cell == cell], key=lambda item: item.cycle)
        selected.append(cell_spectra[len(cell_spectra) // 2])
    return selected


def plot_drt_fits(
    spectra: list[Spectrum],
    fits: dict[tuple[str, int], RawDRTFit],
    output_path: Path,
) -> None:
    selected = representative_spectra(spectra)
    figure, axes = cell_axes(len(selected))
    for axis, spectrum in zip(axes, selected):
        fit = fits[(spectrum.cell, spectrum.cycle)]
        axis.plot(spectrum.z_ohm.real * 1000, -spectrum.z_ohm.imag * 1000, "o", ms=2.8, label="Measured")
        axis.plot(fit.z_fit.real * 1000, -fit.z_fit.imag * 1000, "-", lw=1.4, label="Raw DRT fit")
        axis.set_title(f"{spectrum.cell} Cycle {spectrum.cycle}, NRMSE={fit.nrmse:.3f}")
        axis.set_xlabel("Z' / mOhm")
        axis.set_ylabel("-Z'' / mOhm")
        axis.grid(True, ls="--", lw=0.35, alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    figure.suptitle("Optimized own EIS: raw DRT fitting examples", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_drt_evolution(
    distribution: pd.DataFrame,
    tau: np.ndarray,
    output_path: Path,
) -> None:
    gamma_columns = [column for column in distribution if column.startswith("gamma_")]
    figure, axes = cell_axes(len(CELL_IDS))
    for axis, cell in zip(axes, CELL_IDS):
        group = distribution[distribution["Cell"].eq(cell)].sort_values("Cycle")
        cycles = group["Cycle"].to_numpy(dtype=float)
        normalizer = mcolors.Normalize(cycles.min(), cycles.max() if cycles.max() > cycles.min() else cycles.min() + 1)
        color_map = plt.get_cmap("plasma")
        for _, row in group.iterrows():
            gamma = row[gamma_columns].to_numpy(dtype=float)
            axis.semilogx(tau, gamma, color=color_map(normalizer(float(row["Cycle"]))), lw=0.9, alpha=0.8)
        axis.set_title(cell)
        axis.set_xlabel("tau / s")
        axis.set_ylabel("gamma / ohm")
        axis.grid(True, which="both", ls="--", lw=0.35, alpha=0.45)
    figure.suptitle("Optimized own EIS: raw DRT evolution", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_ecm_fits(
    spectra: list[Spectrum],
    parameters: pd.DataFrame,
    fitted_spectra: dict[tuple[str, int], np.ndarray],
    output_path: Path,
) -> None:
    selected = representative_spectra(spectra)
    figure, axes = cell_axes(len(selected))
    lookup = parameters.set_index(["Cell", "Cycle"])
    for axis, spectrum in zip(axes, selected):
        key = (spectrum.cell, spectrum.cycle)
        predicted = fitted_spectra.get(key)
        row = lookup.loc[key]
        axis.plot(spectrum.z_ohm.real * 1000, -spectrum.z_ohm.imag * 1000, "o", ms=2.8, label="Measured")
        if predicted is not None:
            axis.plot(predicted.real * 1000, -predicted.imag * 1000, "-", lw=1.4, label="ECM fit")
        axis.set_title(f"{spectrum.cell} Cycle {spectrum.cycle}, NRMSE={row.get('fit_nrmse', np.nan):.3f}")
        axis.set_xlabel("Z' / mOhm")
        axis.set_ylabel("-Z'' / mOhm")
        axis.grid(True, ls="--", lw=0.35, alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    figure.suptitle("L-Rs-(R_SEI||C_SE)-(R_CT||CPE_CT)-CPE_W fitting examples", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_parameter_trends(parameters: pd.DataFrame, output_path: Path) -> None:
    columns = [
        "Rs_ohm",
        "L_H",
        "R_SEI_ohm",
        "C_SE_F",
        "R_CT_ohm",
        "Q_CT",
        "alpha_CT",
        "tau_CT_s",
        "Q_W",
        "alpha_W",
        "fit_nrmse",
        "ecm_boundary_count",
    ]
    figure, axes = plt.subplots(4, 3, figsize=(16, 14), constrained_layout=True)
    axes = axes.ravel()
    colors = plt.get_cmap("tab10")
    log_columns = {"L_H", "R_SEI_ohm", "C_SE_F", "R_CT_ohm", "Q_CT", "tau_CT_s", "Q_W", "fit_nrmse"}
    for axis, column in zip(axes, columns):
        for index, cell in enumerate(CELL_IDS):
            group = parameters[parameters["Cell"].eq(cell)].sort_values("Cycle")
            x = group["Cycle"]
            y = pd.to_numeric(group.get(column), errors="coerce")
            if column in log_columns:
                y = y.where(y > 0)
            axis.plot(x, y, "o-", ms=2.6, lw=1.0, color=colors(index), label=cell)
        axis.set_title(column)
        axis.set_xlabel("Cycle")
        if column in log_columns:
            axis.set_yscale("log")
        axis.grid(True, which="both", ls="--", lw=0.35, alpha=0.45)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=8, frameon=False)
    figure.suptitle("Optimized own EIS: ECM parameter and fit-quality trends", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_drt_feature_trends(features: pd.DataFrame, output_path: Path) -> None:
    columns = [
        "drt_r_inf_ohm",
        "drt_L_H",
        "drt_total_area_ohm",
        "drt_max_gamma_ohm",
        "drt_dominant_tau_s",
        "drt_centroid_log10_tau",
        "drt_spread_log10_tau",
        "drt_detected_peak_count",
        "drt_tau_q50_s",
    ]
    figure, axes = plt.subplots(3, 3, figsize=(15.5, 11.5), constrained_layout=True)
    axes = axes.ravel()
    colors = plt.get_cmap("tab10")
    log_columns = {"drt_L_H", "drt_total_area_ohm", "drt_max_gamma_ohm", "drt_dominant_tau_s", "drt_tau_q50_s"}
    for axis, column in zip(axes, columns):
        for index, cell in enumerate(CELL_IDS):
            group = features[features["Cell"].eq(cell)].sort_values("Cycle")
            y = pd.to_numeric(group[column], errors="coerce")
            if column in log_columns:
                y = y.where(y > 0)
            axis.plot(group["Cycle"], y, "o-", ms=2.6, lw=1.0, color=colors(index), label=cell)
        axis.set_title(column)
        axis.set_xlabel("Cycle")
        if column in log_columns:
            axis.set_yscale("log")
        axis.grid(True, which="both", ls="--", lw=0.35, alpha=0.45)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=8, frameon=False)
    figure.suptitle("Optimized own EIS: raw DRT feature trends", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison(aggregate: pd.DataFrame, output_path: Path) -> None:
    protocols = ["leave_one_cell_out", "pooled_5fold"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for axis, protocol in zip(axes, protocols):
        subset = aggregate[aggregate["Protocol"].eq(protocol)].copy()
        best = subset.loc[subset.groupby("FeatureSet")["RMSE_SOH"].idxmin()].sort_values("RMSE_SOH")
        axis.barh(best["FeatureSet"], best["RMSE_pct_point"], color="#4C78A8")
        for position, (_, row) in enumerate(best.iterrows()):
            axis.text(row["RMSE_pct_point"], position, f" {row['Model']}  {row['RMSE_pct_point']:.2f}", va="center", fontsize=8)
        axis.set_title(protocol)
        axis.set_xlabel("Best RMSE / SOH percentage points")
        axis.grid(True, axis="x", ls="--", lw=0.4, alpha=0.5)
    figure.suptitle("SOH prediction comparison by feature representation", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_loco_parity(aggregate: pd.DataFrame, predictions: pd.DataFrame, output_path: Path) -> None:
    subset = aggregate[aggregate["Protocol"].eq("leave_one_cell_out")]
    best_rows = subset.loc[subset.groupby("FeatureSet")["RMSE_SOH"].idxmin()].sort_values("RMSE_SOH")
    columns = 3
    rows = int(np.ceil(len(best_rows) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 4.8 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, (_, best) in zip(axes, best_rows.iterrows()):
        data = predictions[
            predictions["Protocol"].eq("leave_one_cell_out")
            & predictions["FeatureSet"].eq(best["FeatureSet"])
            & predictions["Model"].eq(best["Model"])
        ]
        for cell, group in data.groupby("Cell"):
            axis.scatter(group["SOH_true"], group["SOH_pred"], s=20, alpha=0.75, label=cell)
        low = min(float(data["SOH_true"].min()), float(data["SOH_pred"].min())) - 0.01
        high = max(float(data["SOH_true"].max()), float(data["SOH_pred"].max())) + 0.01
        axis.plot([low, high], [low, high], "k--", lw=1)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_title(f"{best['FeatureSet']} + {best['Model']}\nRMSE={best['RMSE_pct_point']:.2f} pct-point")
        axis.set_xlabel("True SOH")
        axis.set_ylabel("Predicted SOH")
        axis.grid(True, ls="--", lw=0.35, alpha=0.45)
    for axis in axes[len(best_rows) :]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=8, frameon=False)
    figure.suptitle("Leave-one-cell-out SOH parity", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_best_loco_trends(aggregate: pd.DataFrame, predictions: pd.DataFrame, output_path: Path) -> tuple[str, str]:
    best = aggregate[aggregate["Protocol"].eq("leave_one_cell_out")].sort_values(
        ["RMSE_SOH", "MAE_SOH"]
    ).iloc[0]
    data = predictions[
        predictions["Protocol"].eq("leave_one_cell_out")
        & predictions["FeatureSet"].eq(best["FeatureSet"])
        & predictions["Model"].eq(best["Model"])
    ]
    figure, axes = cell_axes(len(CELL_IDS))
    for axis, cell in zip(axes, CELL_IDS):
        group = data[data["Cell"].eq(cell)].sort_values("Cycle")
        axis.plot(group["Cycle"], group["SOH_true"], "o-", ms=3, lw=1.2, color="#111827", label="True")
        axis.plot(group["Cycle"], group["SOH_pred"], "s--", ms=3, lw=1.1, color="#2563EB", label="Predicted")
        axis.set_title(cell)
        axis.set_xlabel("Cycle")
        axis.set_ylabel("SOH")
        axis.grid(True, ls="--", lw=0.35, alpha=0.45)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    figure.suptitle(
        f"Best cross-cell model: {best['FeatureSet']} + {best['Model']} "
        f"(RMSE={best['RMSE_pct_point']:.2f} pct-point)",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(best["FeatureSet"]), str(best["Model"])


def plot_permutation_importance(summary: pd.DataFrame, output_path: Path) -> None:
    top = summary.head(25).sort_values("ImportanceMean")
    figure, axis = plt.subplots(figsize=(10.5, 8.5), constrained_layout=True)
    axis.barh(
        top["Feature"],
        top["ImportanceMean"],
        xerr=top["ImportanceAcrossCellStd"].fillna(0),
        color="#4C78A8",
        alpha=0.9,
    )
    axis.axvline(0, color="black", lw=0.8)
    axis.set_xlabel("Increase in held-out-cell RMSE when permuted")
    axis.set_title(
        f"Cross-cell permutation importance: {summary.iloc[0]['FeatureSet']} + {summary.iloc[0]['Model']}"
    )
    axis.grid(True, axis="x", ls="--", lw=0.4, alpha=0.5)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    output_path: Path,
    audit: pd.DataFrame,
    drt_quality: pd.DataFrame,
    ecm_parameters: pd.DataFrame,
    aggregate: pd.DataFrame,
    best_feature_set: str,
    best_model: str,
    elapsed_seconds: float,
) -> None:
    loco = aggregate[aggregate["Protocol"].eq("leave_one_cell_out")]
    best_by_set = loco.loc[loco.groupby("FeatureSet")["RMSE_SOH"].idxmin()].sort_values("RMSE_SOH")
    lines = [
        "# Optimized own EIS analysis",
        "",
        "## Input",
        f"- Cells: {len(audit)}",
        f"- EIS curves: {int(audit['Curves'].sum())}",
        f"- Curves with measured capacity/SOH: {int(audit['LabeledCurves'].sum())}",
        f"- Frequency range: {audit['FrequencyMinHz'].min():.6g} to {audit['FrequencyMaxHz'].max():.6g} Hz",
        "- SOH = capacity / first available capacity for each cell.",
        "- Cycle is metadata only and is not used as a model feature.",
        "",
        "## Raw DRT",
        f"- Successful fits: {int(drt_quality['drt_fit_success'].sum())} / {len(drt_quality)}",
        f"- Median fit NRMSE: {drt_quality['drt_fit_nrmse'].median():.6f}",
        f"- Maximum fit NRMSE: {drt_quality['drt_fit_nrmse'].max():.6f}",
        "- Representation: R_inf + j*omega*L + non-negative RC relaxation distribution.",
        "",
        "## Equivalent circuit",
        "- Circuit: L - Rs - (R_SEI || C_SE) - (R_CT || CPE_CT) - CPE_W.",
        f"- Successful fits: {int(ecm_parameters['fit_success'].fillna(False).sum())} / {len(ecm_parameters)}",
        f"- Usable fits (NRMSE threshold): {int(ecm_parameters['ecm_usable'].fillna(False).sum())} / {len(ecm_parameters)}",
        f"- Median fit NRMSE: {pd.to_numeric(ecm_parameters['fit_nrmse'], errors='coerce').median():.6f}",
        f"- Fits touching at least one parameter boundary: {int((pd.to_numeric(ecm_parameters['ecm_boundary_count'], errors='coerce') > 0).sum())}",
        "",
        "## Leave-one-cell-out SOH prediction",
        "",
        "| Feature set | Best model | RMSE (SOH) | RMSE (pct-point) | MAE (pct-point) | R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in best_by_set.iterrows():
        lines.append(
            f"| {row['FeatureSet']} | {row['Model']} | {row['RMSE_SOH']:.6f} | "
            f"{row['RMSE_pct_point']:.3f} | {row['MAE_pct_point']:.3f} | {row['R2']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Overall best cross-cell representation: **{best_feature_set} + {best_model}**.",
            "",
            "## Interpretation cautions",
            "- Pooled/random validation mixes cells and is expected to be more optimistic than leave-one-cell-out.",
            "- ECM boundary flags indicate weak parameter identifiability even when impedance fit error is low.",
            "- Raw DRT includes the measured low-frequency response; no diffusion/CPE tail is removed in this representation.",
            "- Cycle 0 curves without capacity labels are retained in fit/feature files but excluded from supervised metrics.",
            "",
            f"Runtime: {elapsed_seconds:.1f} seconds.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw DRT, ECM, and SOH prediction on optimized own EIS")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-tau", type=int, default=160)
    parser.add_argument("--lambda-drt", type=float, default=1e-4)
    parser.add_argument("--ecm-max-nrmse", type=float, default=0.20)
    parser.add_argument("--max-curves-per-cell", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    directories = {
        "audit": args.output_dir / "00_input_audit",
        "drt": args.output_dir / "01_raw_drt",
        "ecm": args.output_dir / "02_ecm",
        "features": args.output_dir / "03_features",
        "prediction": args.output_dir / "04_soh_prediction",
        "summary": args.output_dir / "05_summary",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    frequency = read_frequency(FREQUENCY_FILE)
    labels = load_capacity_labels(CAPACITY_FILE)
    spectra, audit = load_spectra(frequency, labels, args.max_curves_per_cell)
    audit.to_csv(directories["audit"] / "input_audit.csv", index=False)
    labels.to_csv(directories["audit"] / "capacity_soh_labels.csv", index=False, float_format="%.10g")
    print(f"Loaded {len(spectra)} optimized EIS curves from {len(CELL_IDS)} cells", flush=True)

    tau = make_tau_grid(frequency, args.n_tau)
    distribution, drt_features, drt_quality, drt_fits = run_raw_drt(
        spectra, tau, args.lambda_drt, directories["drt"]
    )
    plot_drt_fits(spectra, drt_fits, directories["drt"] / "raw_drt_fit_examples.png")
    plot_drt_evolution(distribution, tau, directories["drt"] / "raw_drt_evolution_all_cells.png")
    plot_drt_feature_trends(drt_features, directories["drt"] / "raw_drt_feature_trends.png")

    ecm_parameters, fitted_spectra = run_ecm(spectra, directories["ecm"], args.ecm_max_nrmse)
    plot_ecm_fits(spectra, ecm_parameters, fitted_spectra, directories["ecm"] / "ecm_fit_examples.png")
    plot_parameter_trends(ecm_parameters, directories["ecm"] / "ecm_parameter_trends.png")

    metadata_columns = ["Cell", "Cycle", "SourceFile", "capacity_mAh", "initial_capacity_mAh", "SOH"]
    drt_raw_columns = ["drt_r_inf_ohm", "drt_L_H"] + [column for column in distribution if column.startswith("gamma_")]
    engineered_columns = [
        column
        for column in drt_features.columns
        if column.startswith("drt_") and column not in {"drt_r_inf_ohm", "drt_L_H"}
    ]
    ecm_model_frame, ecm_model_columns = build_ecm_model_features(ecm_parameters)
    model_frame = distribution[metadata_columns + drt_raw_columns].merge(
        drt_features[["Cell", "Cycle"] + engineered_columns], on=["Cell", "Cycle"], how="left"
    )
    model_frame = model_frame.merge(ecm_model_frame, on=["Cell", "Cycle"], how="left")
    evolution_sources = engineered_columns + ecm_model_columns
    model_frame, evolution_columns = add_initial_evolution_features(model_frame, evolution_sources)
    model_frame.to_csv(directories["features"] / "all_model_features_with_soh.csv", index=False, float_format="%.10g")

    feature_sets = {
        "raw_drt_vector": drt_raw_columns,
        "drt_engineered": engineered_columns,
        "ecm_parameters": ecm_model_columns,
        "raw_drt_plus_ecm": drt_raw_columns + ecm_model_columns,
        "combined_with_initial_evolution": engineered_columns + ecm_model_columns + evolution_columns,
    }
    fold_metrics, aggregate, predictions, actual_features = evaluate_feature_sets(
        model_frame, feature_sets, directories["prediction"]
    )
    importance = cross_cell_permutation_importance(
        model_frame, aggregate, actual_features, directories["prediction"]
    )
    plot_model_comparison(aggregate, directories["prediction"] / "model_comparison_rmse.png")
    plot_loco_parity(aggregate, predictions, directories["prediction"] / "loco_parity_best_per_feature_set.png")
    best_feature_set, best_model = plot_best_loco_trends(
        aggregate, predictions, directories["prediction"] / "loco_best_model_soh_trends.png"
    )
    plot_permutation_importance(importance, directories["prediction"] / "best_model_permutation_importance.png")

    configuration = {
        "input_root": str(EIS_ROOT),
        "capacity_file": str(CAPACITY_FILE),
        "frequency_file": str(FREQUENCY_FILE),
        "output_dir": str(args.output_dir),
        "cells": CELL_IDS,
        "n_tau": args.n_tau,
        "lambda_drt": args.lambda_drt,
        "ecm_max_nrmse": args.ecm_max_nrmse,
        "max_curves_per_cell": args.max_curves_per_cell,
        "feature_sets": actual_features,
    }
    (directories["summary"] / "run_configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    elapsed = time.time() - start_time
    write_summary(
        directories["summary"] / "README.md",
        audit,
        drt_quality,
        ecm_parameters,
        aggregate,
        best_feature_set,
        best_model,
        elapsed,
    )
    print(f"Results saved to: {args.output_dir}")
    print(f"Runtime: {elapsed:.1f} s")
    print("Best leave-one-cell-out models by feature set:")
    loco = aggregate[aggregate["Protocol"].eq("leave_one_cell_out")]
    print(
        loco.loc[loco.groupby("FeatureSet")["RMSE_SOH"].idxmin()][
            ["FeatureSet", "Model", "RMSE_pct_point", "MAE_pct_point", "R2"]
        ].sort_values("RMSE_pct_point").to_string(index=False)
    )


if __name__ == "__main__":
    main()

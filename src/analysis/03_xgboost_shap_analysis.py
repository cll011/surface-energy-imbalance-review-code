# -*- coding: utf-8 -*-
"""
Run reorganized Result 3 raster-pixel XGBoost + SHAP experiments.

Output root:
    D:/10_Research/01_Datasets/04_Results/Result3_Figures_optimized/R3_SHAP

Model organization follows the requested manuscript logic:
1. Model 1 background stripping:
   target = T2M anomaly
   features = CO2/ONI/SST/AOD/Snow annual controls + Cloud raster
   excluded = albedo/Rn/energy/water-path variables

2. Model 2 main SHAP contribution model:
   target = Model 1 residual
   features = AlbedoLoss, Rn, SM, LH, SH

2b. Sensitivity SHAP model:
   target = Model 1 residual
   features = AlbedoLoss, Rn, SM, LH, SH, SWabs_MODIS, VPD, Cloud, Snow

3. Model 3 path mechanism model:
   path-specific XGBoost + SHAP equations:
     SWabs_MODIS ~ AlbedoLoss + SWdown + Cloud + Snow
     Rn          ~ AlbedoLoss + SWabs_MODIS + LWdown + Cloud + Snow
     LH          ~ Rn + SM
     SH          ~ Rn + SM
     VPD         ~ SH + LH + SM
     T2M         ~ AlbedoLoss + Rn + LH + SH + SM

For raster variables, the model uses pixel-level anomalies relative to each
pixel's 2001-2010 mean. Annual statistical controls are merged by year and
broadcast to pixels. ONI is intentionally kept as an annual ENSO index and is
not converted to a fake spatial raster.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_paths import get_path  # noqa: E402


ROOT = get_path("result3_root")
LSM_PATH = get_path("lsm_nc")
RASTER_DIR = ROOT / "R3_CommonGrid_Rasters"
TABLE_DIR = ROOT / "R3_Tables"
OUT_DIR = ROOT / "R3_SHAP"
SCRIPT_COPY_DIR = OUT_DIR / "scripts"
MODEL_DIR = OUT_DIR / "models"
TABLE_OUT = OUT_DIR / "tables"
MAP_DIR = OUT_DIR / "spatial_maps"
RESIDUAL_DIR = OUT_DIR / "model1_residual_rasters"

YEARS = list(range(2001, 2025))
BASELINE_YEARS = list(range(2001, 2011))
PERIODS = {
    "Full_2001_2024": YEARS,
    "P1_2001_2014": list(range(2001, 2015)),
    "P2_2015_2019": list(range(2015, 2020)),
    "P3_2020_2024": list(range(2020, 2025)),
}
NODATA = -9999.0
XGB_N_ESTIMATORS = 220
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.06

MODEL1_FEATURES = [
    "CO2_RF_annual_control",
    "CO2_RF_ref2000_annual_control",
    "ONI_annual_control",
    "ONI_Lag1_annual_control",
    "SST_anomaly_annual_control",
    "AOD_annual_control",
    "Snow_annual_control",
    "Cloud",
]

MODEL1_EXCLUDED = [
    "AlbedoLoss",
    "SurfaceAlbedo",
    "Rn",
    "SM",
    "LH",
    "SH",
    "SWdown",
    "LWdown",
    "VPD",
    "SWabs_MODIS",
]

MODEL2_FEATURES = ["AlbedoLoss", "Rn", "SM", "LH", "SH"]
MODEL2B_FEATURES = ["AlbedoLoss", "Rn", "SM", "LH", "SH", "SWabs_MODIS", "VPD", "Cloud", "Snow"]

MODEL3_EQUATIONS = [
    ("M3_01_SWabs_from_albedo_sw_cloud_snow", "SWabs_MODIS", ["AlbedoLoss", "SWdown", "Cloud", "Snow"]),
    ("M3_02_Rn_from_albedo_swabs_lw_cloud_snow", "Rn", ["AlbedoLoss", "SWabs_MODIS", "LWdown", "Cloud", "Snow"]),
    ("M3_03_LH_from_Rn_SM", "LH", ["Rn", "SM"]),
    ("M3_04_SH_from_Rn_SM", "SH", ["Rn", "SM"]),
    ("M3_05_VPD_from_SH_LH_SM", "VPD", ["SH", "LH", "SM"]),
    ("M3_06_T2M_from_albedo_Rn_energy_SM", "T2M", ["AlbedoLoss", "Rn", "LH", "SH", "SM"]),
]

ANNUAL_CONTROL_COLUMNS = {
    "CO2_RF_annual_control",
    "CO2_RF_ref2000_annual_control",
    "ONI_annual_control",
    "ONI_Lag1_annual_control",
    "SST_anomaly_annual_control",
    "AOD_annual_control",
    "Snow_annual_control",
}


def check_dependencies() -> None:
    missing = []
    for name in ["xgboost", "sklearn", "rasterio", "numpy", "pandas"]:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Install the pinned environment from environment.yml."
        )
    if not hasattr(np, "bool"):
        np.bool = np.bool_  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]


def imports():
    check_dependencies()
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from xgboost import DMatrix, XGBRegressor

    return XGBRegressor, DMatrix, train_test_split, r2_score, mean_absolute_error


def ensure_dirs() -> None:
    for folder in [OUT_DIR, SCRIPT_COPY_DIR, MODEL_DIR, TABLE_OUT, MAP_DIR, RESIDUAL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def archive_script() -> None:
    try:
        src = Path(__file__)
        if src.exists():
            shutil.copy2(src, SCRIPT_COPY_DIR / src.name)
    except Exception as exc:
        print(f"[warning] could not archive script copy: {exc}", flush=True)


def phase_from_year(year: int) -> str:
    if 2001 <= int(year) <= 2014:
        return "P1"
    if 2015 <= int(year) <= 2019:
        return "P2"
    if 2020 <= int(year) <= 2024:
        return "P3"
    return "Other"


def raster_path(var: str, year: int) -> Path:
    return RASTER_DIR / var / f"{var}_{year}_R3_common.tif"


def residual_path(year: int) -> Path:
    return RESIDUAL_DIR / "T2M_residual_model1" / f"T2M_residual_model1_{year}_R3_common.tif"


def prediction_path(year: int) -> Path:
    return RESIDUAL_DIR / "T2M_pred_model1" / f"T2M_pred_model1_{year}_R3_common.tif"


def target_anom_path(year: int) -> Path:
    return RESIDUAL_DIR / "T2M_anom" / f"T2M_anom_{year}_R3_common.tif"


def template_profile() -> dict:
    path = raster_path("T2M", 2001)
    if not path.exists():
        raise FileNotFoundError(f"Missing template raster: {path}")
    with rasterio.open(path) as src:
        profile = src.profile.copy()
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate", predictor=3)
    return profile


def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float32").filled(np.nan)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def write_raster(path: Path, arr: np.ndarray, profile: dict, tags: Optional[Dict[str, str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = arr.astype("float32", copy=True)
    out[~np.isfinite(out)] = NODATA
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


def load_land_mask() -> np.ndarray:
    """Sample the archived ERA5-Land mask to the analysis grid (lsm >= 0.5)."""
    template_path = raster_path("T2M", 2001)
    with rasterio.open(template_path) as src:
        if src.crs is None or not src.crs.is_geographic:
            raise ValueError(f"The analysis raster must use a geographic CRS: {src.crs}")
        transform = src.transform
        width, height = src.width, src.height
        template_valid = np.isfinite(
            src.read(1, masked=True).astype("float32").filled(np.nan)
        )

    with xr.open_dataset(LSM_PATH) as dataset:
        variable = "lsm" if "lsm" in dataset.data_vars else list(dataset.data_vars)[0]
        mask = dataset[variable].squeeze(drop=True).load()
    lat_name = next(
        name for name in ("latitude", "lat", "y") if name in mask.coords or name in mask.dims
    )
    lon_name = next(
        name for name in ("longitude", "lon", "x") if name in mask.coords or name in mask.dims
    )
    mask = mask.transpose(lat_name, lon_name)
    longitude = np.asarray(mask[lon_name].values, dtype=float)
    if np.nanmax(longitude) > 180:
        mask = mask.assign_coords(
            {lon_name: ((longitude + 180.0) % 360.0) - 180.0}
        ).sortby(lon_name)
    mask = mask.sortby(lat_name)

    longitudes = transform.c + (np.arange(width) + 0.5) * transform.a
    latitudes = transform.f + (np.arange(height) + 0.5) * transform.e
    sampled = mask.sel(
        {
            lat_name: xr.DataArray(latitudes, dims="y"),
            lon_name: xr.DataArray(((longitudes + 180.0) % 360.0) - 180.0, dims="x"),
        },
        method="nearest",
    ).values
    return template_valid & np.isfinite(sampled) & (sampled >= 0.5)


def load_weights(land_mask: np.ndarray, profile: dict) -> np.ndarray:
    rows = np.arange(profile["height"], dtype="float64")
    transform = profile["transform"]
    lats = transform.f + (rows + 0.5) * transform.e
    weights = np.cos(np.deg2rad(lats))[:, None].astype("float32")
    return np.where(land_mask, weights, np.nan).astype("float32")


def load_annual_controls() -> pd.DataFrame:
    path = TABLE_DIR / "R3_annual_background_controls.csv"
    if not path.exists():
        raise FileNotFoundError(f"Annual control table not found: {path}")
    df = pd.read_csv(path)
    df["Year"] = df["Year"].astype(int)
    missing = [c for c in ANNUAL_CONTROL_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing annual control columns: {missing}")
    return df


def required_raster_variables() -> List[str]:
    vars_ = {"T2M", "Cloud"}
    vars_.update(MODEL2_FEATURES)
    vars_.update(MODEL2B_FEATURES)
    for _eq, target, features in MODEL3_EQUATIONS:
        vars_.add(target)
        vars_.update(features)
    return sorted(vars_)


def check_inputs(controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in required_raster_variables():
        years = [year for year in YEARS if raster_path(var, year).exists()]
        rows.append(
            {
                "Variable": var,
                "Storage": "raster",
                "AvailableYears": len(years),
                "MissingYears": ",".join(map(str, [year for year in YEARS if year not in years])),
                "OK": len(years) == len(YEARS),
            }
        )
    for var in MODEL1_FEATURES:
        if var in ANNUAL_CONTROL_COLUMNS:
            years = controls.loc[controls[var].notna(), "Year"].astype(int).tolist()
            rows.append(
                {
                    "Variable": var,
                    "Storage": "annual_control_table",
                    "AvailableYears": len([y for y in YEARS if y in years]),
                    "MissingYears": ",".join(map(str, [year for year in YEARS if year not in years])),
                    "OK": all(year in years for year in YEARS),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_OUT / "input_variable_coverage.csv", index=False, encoding="utf-8-sig")
    if not bool(df["OK"].all()):
        raise RuntimeError(f"Input coverage check failed. See {TABLE_OUT / 'input_variable_coverage.csv'}")
    return df


def build_baselines(raster_vars: Sequence[str]) -> Dict[str, np.ndarray]:
    baselines: Dict[str, np.ndarray] = {}
    for var in sorted(set(raster_vars)):
        arrays = []
        for year in BASELINE_YEARS:
            path = raster_path(var, year)
            if path.exists():
                arrays.append(read_raster(path))
        if not arrays:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            baselines[var] = np.nanmean(np.stack(arrays, axis=0), axis=0).astype("float32")
    return baselines


def feature_is_control(feature: str) -> bool:
    return feature in ANNUAL_CONTROL_COLUMNS


def get_control_value(controls: pd.DataFrame, year: int, feature: str) -> float:
    row = controls.loc[controls["Year"] == int(year)]
    if row.empty or feature not in row.columns:
        return math.nan
    val = row[feature].iloc[0]
    return float(val) if pd.notna(val) else math.nan


def get_feature_array(
    feature: str,
    year: int,
    land_rows: np.ndarray,
    land_cols: np.ndarray,
    baselines: Dict[str, np.ndarray],
    controls: pd.DataFrame,
    as_anomaly: bool = True,
) -> np.ndarray:
    if feature_is_control(feature):
        value = get_control_value(controls, year, feature)
        return np.full(land_rows.shape[0], value, dtype="float32")
    arr = read_raster(raster_path(feature, year))
    if as_anomaly and feature in baselines:
        arr = arr - baselines[feature]
    return arr[land_rows, land_cols].astype("float32")


def get_target_array(
    target: str,
    year: int,
    land_rows: np.ndarray,
    land_cols: np.ndarray,
    baselines: Dict[str, np.ndarray],
    residual: bool = False,
) -> np.ndarray:
    if residual:
        arr = read_raster(residual_path(year))
        return arr[land_rows, land_cols].astype("float32")
    arr = read_raster(raster_path(target, year))
    if target in baselines:
        arr = arr - baselines[target]
    return arr[land_rows, land_cols].astype("float32")


def make_frame_for_year(
    year: int,
    features: Sequence[str],
    land_rows: np.ndarray,
    land_cols: np.ndarray,
    baselines: Dict[str, np.ndarray],
    controls: pd.DataFrame,
    as_anomaly: bool = True,
) -> pd.DataFrame:
    data = {}
    for feature in features:
        data[feature] = get_feature_array(feature, year, land_rows, land_cols, baselines, controls, as_anomaly=as_anomaly)
    return pd.DataFrame(data)


def finite_rows(X: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
    mask = np.all(np.isfinite(X.to_numpy(dtype="float64")), axis=1)
    if y is not None:
        mask &= np.isfinite(y)
    return mask


def sample_training_data(
    model_label: str,
    target: str,
    features: Sequence[str],
    land_rows: np.ndarray,
    land_cols: np.ndarray,
    baselines: Dict[str, np.ndarray],
    controls: pd.DataFrame,
    max_rows: int,
    seed: int,
    target_is_model1_residual: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    per_year = max(500, int(math.ceil(max_rows / len(YEARS))))
    X_parts = []
    y_parts = []
    meta_parts = []

    for year in YEARS:
        X_year = make_frame_for_year(year, features, land_rows, land_cols, baselines, controls)
        y_year = get_target_array(target, year, land_rows, land_cols, baselines, residual=target_is_model1_residual)
        valid = finite_rows(X_year, y_year)
        idx = np.where(valid)[0]
        if idx.size == 0:
            continue
        if idx.size > per_year:
            idx = rng.choice(idx, size=per_year, replace=False)
        X_parts.append(X_year.iloc[idx].reset_index(drop=True))
        y_parts.append(y_year[idx])
        meta_parts.append(
            pd.DataFrame(
                {
                    "Model": model_label,
                    "Year": year,
                    "Phase": phase_from_year(year),
                    "Row": land_rows[idx],
                    "Col": land_cols[idx],
                }
            )
        )

    if not X_parts:
        raise RuntimeError(f"No finite training data for {model_label}")
    X = pd.concat(X_parts, ignore_index=True)
    y = np.concatenate(y_parts).astype("float32")
    meta = pd.concat(meta_parts, ignore_index=True)
    if len(X) > max_rows:
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]
        meta = meta.iloc[idx].reset_index(drop=True)
    return X, y, meta


@dataclass
class FittedModel:
    label: str
    target: str
    features: List[str]
    model: object
    booster: object
    metrics: Dict[str, object]
    target_is_model1_residual: bool = False


def fit_xgb(
    label: str,
    target: str,
    features: Sequence[str],
    land_rows: np.ndarray,
    land_cols: np.ndarray,
    baselines: Dict[str, np.ndarray],
    controls: pd.DataFrame,
    max_rows: int,
    seed: int,
    n_jobs: int,
    target_is_model1_residual: bool = False,
) -> Tuple[FittedModel, pd.DataFrame]:
    XGBRegressor, _DMatrix, train_test_split, r2_score, mean_absolute_error = imports()
    print(f"[fit] {label}: target={target}; features={', '.join(features)}", flush=True)
    X, y, meta = sample_training_data(
        label,
        target,
        features,
        land_rows,
        land_cols,
        baselines,
        controls,
        max_rows=max_rows,
        seed=seed,
        target_is_model1_residual=target_is_model1_residual,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=n_jobs,
        reg_lambda=1.0,
    )
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    metrics = {
        "Model": label,
        "Target": target,
        "TargetTransform": "Model1 residual" if target_is_model1_residual else "pixel anomaly from 2001-2010 baseline",
        "Features": ",".join(features),
        "XGB_n_estimators": XGB_N_ESTIMATORS,
        "XGB_max_depth": XGB_MAX_DEPTH,
        "XGB_learning_rate": XGB_LEARNING_RATE,
        "N_train": int(len(X_train)),
        "N_test": int(len(X_test)),
        "R2_train": float(r2_score(y_train, pred_train)),
        "R2_test": float(r2_score(y_test, pred_test)),
        "RMSE_train": float(np.sqrt(np.mean((y_train - pred_train) ** 2))),
        "RMSE_test": float(np.sqrt(np.mean((y_test - pred_test) ** 2))),
        "MAE_train": float(mean_absolute_error(y_train, pred_train)),
        "MAE_test": float(mean_absolute_error(y_test, pred_test)),
    }
    model_path = MODEL_DIR / label / f"{label}_xgboost.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    booster = model.get_booster()
    fitted = FittedModel(
        label=label,
        target=target,
        features=list(features),
        model=model,
        booster=booster,
        metrics=metrics,
        target_is_model1_residual=target_is_model1_residual,
    )
    sample = meta.copy()
    sample[target] = y
    for c in X.columns:
        sample[c] = X[c].to_numpy()
    sample.to_csv(TABLE_OUT / f"{label}_training_sample.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    return fitted, sample


def shap_values(fitted: FittedModel, X: pd.DataFrame, approx_contribs: bool) -> np.ndarray:
    """Return XGBoost native TreeSHAP contributions, excluding the bias column."""
    from xgboost import DMatrix

    data = X.loc[:, fitted.features].to_numpy(dtype="float32", copy=False)
    dmat = DMatrix(data, feature_names=fitted.features)
    vals = fitted.booster.predict(dmat, pred_contribs=True, approx_contribs=approx_contribs)
    vals = np.asarray(vals, dtype="float32")
    if vals.ndim == 2 and vals.shape[1] == len(fitted.features) + 1:
        vals = vals[:, :-1]
    return vals


def predict_model1_residuals(
    fitted: FittedModel,
    profile: dict,
    land_mask: np.ndarray,
    land_rows: np.ndarray,
    land_cols: np.ndarray,
    baselines: Dict[str, np.ndarray],
    controls: pd.DataFrame,
) -> None:
    print("[predict] writing Model 1 prediction and residual rasters", flush=True)
    h, w = land_mask.shape
    for year in YEARS:
        X = make_frame_for_year(year, fitted.features, land_rows, land_cols, baselines, controls)
        y = get_target_array("T2M", year, land_rows, land_cols, baselines)
        valid = finite_rows(X, y)
        pred_vals = np.full(land_rows.shape[0], np.nan, dtype="float32")
        if np.any(valid):
            pred_vals[valid] = fitted.model.predict(X.loc[valid, fitted.features]).astype("float32")
        residual_vals = y - pred_vals
        target_map = np.full((h, w), np.nan, dtype="float32")
        pred_map = np.full((h, w), np.nan, dtype="float32")
        residual_map = np.full((h, w), np.nan, dtype="float32")
        target_map[land_rows, land_cols] = y
        pred_map[land_rows, land_cols] = pred_vals
        residual_map[land_rows, land_cols] = residual_vals
        write_raster(
            target_anom_path(year),
            target_map,
            profile,
            {"variable": "T2M_anom", "year": year, "baseline": "2001-2010 pixel mean"},
        )
        write_raster(
            prediction_path(year),
            pred_map,
            profile,
            {"variable": "T2M_pred_model1", "year": year, "model": fitted.label},
        )
        write_raster(
            residual_path(year),
            residual_map,
            profile,
            {"variable": "T2M_residual_model1", "year": year, "model": fitted.label},
        )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return math.nan
    return float(np.nansum(values[valid] * weights[valid]) / np.nansum(weights[valid]))


def compute_spatial_shap(
    fitted: FittedModel,
    profile: dict,
    land_rows: np.ndarray,
    land_cols: np.ndarray,
    land_weights: np.ndarray,
    baselines: Dict[str, np.ndarray],
    controls: pd.DataFrame,
    batch_size: int,
    approx_contribs: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[SHAP spatial] {fitted.label}", flush=True)
    h, w = profile["height"], profile["width"]
    feature_rows = []
    map_rows = []

    phase_period_by_year = {}
    for period, years in PERIODS.items():
        if period == "Full_2001_2024":
            continue
        for year in years:
            phase_period_by_year[int(year)] = period

    accumulators = {}
    for period in PERIODS:
        accumulators[period] = {
            "signed_sum": {feature: np.zeros((h, w), dtype="float64") for feature in fitted.features},
            "abs_sum": {feature: np.zeros((h, w), dtype="float64") for feature in fitted.features},
            "count": np.zeros((h, w), dtype="float32"),
        }

    for year in YEARS:
        phase_period = phase_period_by_year.get(int(year))
        if phase_period is None:
            continue
        print(f"[SHAP spatial] {fitted.label} | year={year} -> Full_2001_2024 + {phase_period}", flush=True)
        X_year = make_frame_for_year(year, fitted.features, land_rows, land_cols, baselines, controls)
        if fitted.target_is_model1_residual:
            y_year = get_target_array(fitted.target, year, land_rows, land_cols, baselines, residual=True)
        else:
            y_year = get_target_array(fitted.target, year, land_rows, land_cols, baselines)
        valid = finite_rows(X_year, y_year)
        valid_idx = np.where(valid)[0]
        if valid_idx.size == 0:
            continue

        for start in range(0, valid_idx.size, batch_size):
            idx = valid_idx[start : start + batch_size]
            X_batch = X_year.iloc[idx]
            vals = shap_values(fitted, X_batch, approx_contribs=approx_contribs)
            rr = land_rows[idx]
            cc = land_cols[idx]
            for period in ("Full_2001_2024", phase_period):
                accum = accumulators[period]
                accum["count"][rr, cc] += 1.0
                for j, feature in enumerate(fitted.features):
                    accum["signed_sum"][feature][rr, cc] += vals[:, j]
                    accum["abs_sum"][feature][rr, cc] += np.abs(vals[:, j])

    for period, years in PERIODS.items():
        print(f"[SHAP spatial] {fitted.label} | {period} | writing maps", flush=True)
        accum = accumulators[period]
        count = accum["count"]
        feature_abs_totals = []
        period_row_start = len(feature_rows)
        for feature in fitted.features:
            signed_map = np.full((h, w), np.nan, dtype="float32")
            abs_map = np.full((h, w), np.nan, dtype="float32")
            ok = count > 0
            signed_map[ok] = (accum["signed_sum"][feature][ok] / count[ok]).astype("float32")
            abs_map[ok] = (accum["abs_sum"][feature][ok] / count[ok]).astype("float32")

            model_map_dir = MAP_DIR / fitted.label / period
            signed_path = model_map_dir / f"{fitted.label}_{period}_{feature}_mean_signed_SHAP.tif"
            abs_path = model_map_dir / f"{fitted.label}_{period}_{feature}_mean_abs_SHAP.tif"
            write_raster(
                signed_path,
                signed_map,
                profile,
                {"model": fitted.label, "period": period, "feature": feature, "metric": "mean_signed_SHAP"},
            )
            write_raster(
                abs_path,
                abs_map,
                profile,
                {"model": fitted.label, "period": period, "feature": feature, "metric": "mean_abs_SHAP"},
            )
            land_abs = abs_map[land_rows, land_cols]
            land_signed = signed_map[land_rows, land_cols]
            mean_abs = weighted_mean(land_abs, land_weights)
            mean_signed = weighted_mean(land_signed, land_weights)
            feature_abs_totals.append(mean_abs)
            feature_rows.append(
                {
                    "Model": fitted.label,
                    "Target": fitted.target,
                    "Period": period,
                    "Feature": feature,
                    "Mean_SHAP": mean_signed,
                    "MeanAbs_SHAP": mean_abs,
                    "ValidPixels": int(np.isfinite(land_abs).sum()),
                    "Years": ",".join(map(str, years)),
                    "FeatureStorage": "annual_control_table" if feature_is_control(feature) else "raster_pixel_anomaly",
                }
            )
            map_rows.extend(
                [
                    {
                        "Model": fitted.label,
                        "Period": period,
                        "Feature": feature,
                        "Metric": "mean_signed_SHAP",
                        "Path": str(signed_path),
                    },
                    {
                        "Model": fitted.label,
                        "Period": period,
                        "Feature": feature,
                        "Metric": "mean_abs_SHAP",
                        "Path": str(abs_path),
                    },
                ]
            )
        total_abs = float(np.nansum(feature_abs_totals))
        if total_abs > 0:
            for row in feature_rows[period_row_start:]:
                row["ShareAbs_SHAP"] = row["MeanAbs_SHAP"] / total_abs
        else:
            for row in feature_rows[period_row_start:]:
                row["ShareAbs_SHAP"] = math.nan
        print(f"[SHAP spatial] {fitted.label} | {period} | maps written", flush=True)

    return pd.DataFrame(feature_rows), pd.DataFrame(map_rows)


def model2_p3_p1_change(feature_stats: pd.DataFrame, model_label: str) -> pd.DataFrame:
    sub = feature_stats[feature_stats["Model"] == model_label].copy()
    p1 = sub[sub["Period"] == "P1_2001_2014"][["Feature", "MeanAbs_SHAP", "ShareAbs_SHAP"]].rename(
        columns={"MeanAbs_SHAP": "P1_MeanAbs_SHAP", "ShareAbs_SHAP": "P1_ShareAbs_SHAP"}
    )
    p3 = sub[sub["Period"] == "P3_2020_2024"][["Feature", "MeanAbs_SHAP", "ShareAbs_SHAP"]].rename(
        columns={"MeanAbs_SHAP": "P3_MeanAbs_SHAP", "ShareAbs_SHAP": "P3_ShareAbs_SHAP"}
    )
    out = p1.merge(p3, on="Feature", how="outer")
    out["P3_minus_P1_MeanAbs_SHAP"] = out["P3_MeanAbs_SHAP"] - out["P1_MeanAbs_SHAP"]
    out["P3_over_P1_MeanAbs_SHAP"] = out["P3_MeanAbs_SHAP"] / out["P1_MeanAbs_SHAP"].replace(0, np.nan)
    out["P3_minus_P1_ShareAbs_SHAP"] = out["P3_ShareAbs_SHAP"] - out["P1_ShareAbs_SHAP"]
    return out.sort_values("P3_minus_P1_MeanAbs_SHAP", ascending=False)


def standardized_beta_for_period(
    target: str,
    features: Sequence[str],
    years: Sequence[int],
    land_rows: np.ndarray,
    land_cols: np.ndarray,
    baselines: Dict[str, np.ndarray],
    controls: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X_parts = []
    y_parts = []
    per_year = max(500, int(math.ceil(max_rows / len(years))))
    for year in years:
        X_year = make_frame_for_year(year, features, land_rows, land_cols, baselines, controls)
        y_year = get_target_array(target, year, land_rows, land_cols, baselines)
        valid = finite_rows(X_year, y_year)
        idx = np.where(valid)[0]
        if idx.size == 0:
            continue
        if idx.size > per_year:
            idx = rng.choice(idx, size=per_year, replace=False)
        X_parts.append(X_year.iloc[idx].reset_index(drop=True))
        y_parts.append(y_year[idx])
    if not X_parts:
        return pd.DataFrame()
    X = pd.concat(X_parts, ignore_index=True)
    y = np.concatenate(y_parts).astype("float64")
    if len(X) < len(features) + 5:
        return pd.DataFrame()
    df = X.copy()
    df[target] = y
    cols = list(features) + [target]
    df = df[np.all(np.isfinite(df[cols].to_numpy(dtype="float64")), axis=1)].copy()
    for col in cols:
        sd = df[col].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return pd.DataFrame()
        df[col] = (df[col] - df[col].mean()) / sd
    Xmat = df[list(features)].to_numpy(dtype="float64")
    yvec = df[target].to_numpy(dtype="float64")
    Xdesign = np.column_stack([np.ones(len(Xmat)), Xmat])
    beta, *_ = np.linalg.lstsq(Xdesign, yvec, rcond=None)
    pred = Xdesign @ beta
    ss_res = float(np.sum((yvec - pred) ** 2))
    ss_tot = float(np.sum((yvec - yvec.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    rows = []
    for feature, b in zip(features, beta[1:]):
        rows.append({"Response": target, "Predictor": feature, "Std_Beta": float(b), "N": int(len(df)), "R2": r2})
    return pd.DataFrame(rows)


def run_all(args: argparse.Namespace) -> None:
    ensure_dirs()
    archive_script()
    check_dependencies()
    controls = load_annual_controls()
    check_inputs(controls)
    profile = template_profile()
    land_mask = load_land_mask()
    weights = load_weights(land_mask, profile)
    land_rows, land_cols = np.where(land_mask)
    land_weights = weights[land_rows, land_cols].astype("float32")

    raster_vars = required_raster_variables()
    baselines = build_baselines(raster_vars)
    baseline_meta = {
        "BaselineYears": BASELINE_YEARS,
        "RasterVariables": sorted(baselines.keys()),
        "Meaning": "Raster features and targets use pixel anomaly relative to 2001-2010 pixel mean.",
    }
    (OUT_DIR / "baseline_anomaly_metadata.json").write_text(json.dumps(baseline_meta, indent=2), encoding="utf-8")

    metrics_rows = []
    all_feature_stats = []
    all_map_manifest = []

    model1, _sample1 = fit_xgb(
        "Model1_background_stripping",
        "T2M",
        MODEL1_FEATURES,
        land_rows,
        land_cols,
        baselines,
        controls,
        args.max_training_rows,
        args.random_seed,
        args.n_jobs,
        target_is_model1_residual=False,
    )
    metrics_rows.append(model1.metrics)
    predict_model1_residuals(model1, profile, land_mask, land_rows, land_cols, baselines, controls)
    stats1, maps1 = compute_spatial_shap(model1, profile, land_rows, land_cols, land_weights, baselines, controls, args.shap_batch_size, args.approx_shap)
    all_feature_stats.append(stats1)
    all_map_manifest.append(maps1)

    model2, _sample2 = fit_xgb(
        "Model2_main_SHAP_core",
        "T2M_residual_model1",
        MODEL2_FEATURES,
        land_rows,
        land_cols,
        baselines,
        controls,
        args.max_training_rows,
        args.random_seed + 20,
        args.n_jobs,
        target_is_model1_residual=True,
    )
    metrics_rows.append(model2.metrics)
    stats2, maps2 = compute_spatial_shap(model2, profile, land_rows, land_cols, land_weights, baselines, controls, args.shap_batch_size, args.approx_shap)
    all_feature_stats.append(stats2)
    all_map_manifest.append(maps2)

    model2b, _sample2b = fit_xgb(
        "Model2b_sensitivity_SHAP",
        "T2M_residual_model1",
        MODEL2B_FEATURES,
        land_rows,
        land_cols,
        baselines,
        controls,
        args.max_training_rows,
        args.random_seed + 21,
        args.n_jobs,
        target_is_model1_residual=True,
    )
    metrics_rows.append(model2b.metrics)
    stats2b, maps2b = compute_spatial_shap(model2b, profile, land_rows, land_cols, land_weights, baselines, controls, args.shap_batch_size, args.approx_shap)
    all_feature_stats.append(stats2b)
    all_map_manifest.append(maps2b)

    path_beta_rows = []
    for i, (eq_label, target, features) in enumerate(MODEL3_EQUATIONS):
        model3, _sample3 = fit_xgb(
            eq_label,
            target,
            features,
            land_rows,
            land_cols,
            baselines,
            controls,
            args.max_training_rows,
            args.random_seed + 100 + i,
            args.n_jobs,
            target_is_model1_residual=False,
        )
        metrics_rows.append(model3.metrics)
        stats3, maps3 = compute_spatial_shap(model3, profile, land_rows, land_cols, land_weights, baselines, controls, args.shap_batch_size, args.approx_shap)
        stats3["PathEquation"] = eq_label
        maps3["PathEquation"] = eq_label
        all_feature_stats.append(stats3)
        all_map_manifest.append(maps3)
        for period, years in PERIODS.items():
            beta_df = standardized_beta_for_period(
                target,
                features,
                years,
                land_rows,
                land_cols,
                baselines,
                controls,
                max_rows=min(args.max_training_rows, 180000),
                seed=args.random_seed + 300 + i,
            )
            if not beta_df.empty:
                beta_df.insert(0, "Model", eq_label)
                beta_df.insert(1, "Period", period)
                path_beta_rows.append(beta_df)

    metrics = pd.DataFrame(metrics_rows)
    feature_stats = pd.concat(all_feature_stats, ignore_index=True)
    map_manifest = pd.concat(all_map_manifest, ignore_index=True)
    path_beta = pd.concat(path_beta_rows, ignore_index=True) if path_beta_rows else pd.DataFrame()

    metrics.to_csv(TABLE_OUT / "model_metrics.csv", index=False, encoding="utf-8-sig")
    feature_stats.to_csv(TABLE_OUT / "shap_feature_contributions_by_period.csv", index=False, encoding="utf-8-sig")
    map_manifest.to_csv(TABLE_OUT / "spatial_shap_map_manifest.csv", index=False, encoding="utf-8-sig")
    path_beta.to_csv(TABLE_OUT / "model3_path_standardized_beta_by_period.csv", index=False, encoding="utf-8-sig")

    comp2 = model2_p3_p1_change(feature_stats, "Model2_main_SHAP_core")
    comp2b = model2_p3_p1_change(feature_stats, "Model2b_sensitivity_SHAP")
    comp2.to_csv(TABLE_OUT / "model2_main_P3_minus_P1_contribution_change.csv", index=False, encoding="utf-8-sig")
    comp2b.to_csv(TABLE_OUT / "model2b_sensitivity_P3_minus_P1_contribution_change.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(TABLE_OUT / "R3_reorganized_XGBoost_SHAP_results.xlsx") as writer:
        metrics.to_excel(writer, sheet_name="model_metrics", index=False)
        feature_stats.to_excel(writer, sheet_name="feature_contributions", index=False)
        comp2.to_excel(writer, sheet_name="model2_P3_minus_P1", index=False)
        comp2b.to_excel(writer, sheet_name="model2b_P3_minus_P1", index=False)
        path_beta.to_excel(writer, sheet_name="model3_path_beta", index=False)
        map_manifest.to_excel(writer, sheet_name="spatial_map_manifest", index=False)

    config = {
        "Root": str(ROOT),
        "Output": str(OUT_DIR),
        "Years": YEARS,
        "Periods": PERIODS,
        "BaselineYears": BASELINE_YEARS,
        "Model1Features": MODEL1_FEATURES,
        "Model1Excluded": MODEL1_EXCLUDED,
        "Model2Features": MODEL2_FEATURES,
        "Model2bFeatures": MODEL2B_FEATURES,
        "Model3Equations": [
            {"Label": label, "Target": target, "Features": features}
            for label, target, features in MODEL3_EQUATIONS
        ],
        "AnnualControlsSource": str(TABLE_DIR / "R3_annual_background_controls.csv"),
        "RasterSource": str(RASTER_DIR),
        "XGBHyperparameters": {
            "n_estimators": XGB_N_ESTIMATORS,
            "max_depth": XGB_MAX_DEPTH,
            "learning_rate": XGB_LEARNING_RATE,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "tree_method": "hist",
        },
        "SHAPBackend": "xgboost.Booster.predict(pred_contribs=True)",
        "ApproxContribs": bool(args.approx_shap),
        "Arguments": vars(args),
    }
    (OUT_DIR / "run_config_reorganized_models.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    report_lines = [
        "Result 3 reorganized Model 1-3 XGBoost + SHAP run",
        "=" * 80,
        "Model 1 strips background controls only; albedo/Rn/energy/path variables are excluded.",
        "Model 2 is the main manuscript SHAP ranking model: AlbedoLoss, Rn, SM, LH, SH.",
        "Model 2b is a sensitivity SHAP model with SWabs_MODIS, VPD, Cloud, and Snow added.",
        "Model 3 is a path-mechanism model; equations are interpreted by path, not as one variable-importance contest.",
        "",
        "All raster variables are pixel anomalies relative to their own 2001-2010 pixel mean.",
        "ONI is read from the annual control table and is not converted to a spatial raster.",
        "",
        f"Training max rows per model: {args.max_training_rows}",
        f"XGBoost trees/depth/learning_rate: {XGB_N_ESTIMATORS}/{XGB_MAX_DEPTH}/{XGB_LEARNING_RATE}",
        f"XGBoost approx_contribs for spatial SHAP: {bool(args.approx_shap)}",
        f"Land pixels: {int(land_mask.sum())}",
        f"Feature contribution rows: {len(feature_stats)}",
        f"Spatial SHAP maps: {len(map_manifest)}",
        "",
        "Main output workbook:",
        str(TABLE_OUT / "R3_reorganized_XGBoost_SHAP_results.xlsx"),
    ]
    (OUT_DIR / "R3_reorganized_SHAP_run_summary.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[done] Reorganized Result 3 SHAP outputs written to: {OUT_DIR}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-training-rows", type=int, default=240000)
    parser.add_argument("--shap-batch-size", type=int, default=200000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument(
        "--approx-shap",
        action="store_true",
        help="Use XGBoost approx_contribs=True for faster full-raster TreeSHAP contribution maps.",
    )
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()

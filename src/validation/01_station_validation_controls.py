# -*- coding: utf-8 -*-
"""
Station-based validation for Result 1.3 variables and mechanism results.

The validation is designed as an external consistency check:
1. compare annual FLUXNET/AmeriFlux observations against Result 3 common-grid
   rasters at station coordinates;
2. compare station-period changes against sampled AlbedoLoss SHAP/contribution
   changes from the completed spatial result maps;
3. export supplementary Nature-style figures and tables.

Outputs are written only under:
    D:/10_Research/01_Datasets/04_Results/Result3_Figures_optimized/R3_SHAP/
    Supplementary_Station_Validation
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import TwoSlopeNorm
from rasterio.warp import transform as rio_transform


ROOT = Path(r"D:\10_Research\01_Datasets\04_Results\Result3_Figures_optimized")
SHAP_DIR = ROOT / "R3_SHAP"
COMMON_GRID = ROOT / "R3_CommonGrid_Rasters"
STATION_RAW = Path(r"D:\10_Research\01_Datasets\01_DataRaw\Station_Data")
STATION_PROCESSED = Path(r"D:\10_Research\01_Datasets\02_DataProcess\02_Fluxnet_Station_Validation")
LEGACY_STATS = Path(r"D:\10_Research\2025_Albedo_Temp\02_Data_Process\05_Statistics")

OUT_DIR = SHAP_DIR / "Supplementary_Station_Validation"
TABLE_OUT = OUT_DIR / "tables"
FIG_OUT = OUT_DIR / "figures"
REPORT_OUT = OUT_DIR / "reports"
SCRIPT_COPY_DIR = SHAP_DIR / "scripts"

YEARS = list(range(2001, 2025))
PERIODS = {
    "P1_2001_2014": range(2001, 2015),
    "P2_2015_2019": range(2015, 2020),
    "P3_2020_2024": range(2020, 2025),
}

VARIABLE_PAIRS = [
    ("SurfaceAlbedo", "SurfaceAlbedo_obs", "R3_SurfaceAlbedo", "unitless"),
    ("Rn", "Rn_obs", "R3_Rn", "W m-2"),
    ("LH", "LH_obs", "R3_LH", "W m-2"),
    ("SH", "SH_obs", "R3_SH", "W m-2"),
    ("T2M", "T2M_obs", "R3_T2M", "deg C"),
    ("VPD", "VPD_obs", "R3_VPD", "hPa or kPa, compare anomalies"),
    ("SWdown", "SWdown_obs", "R3_SWdown", "W m-2"),
    ("LWdown", "LWdown_obs", "R3_LWdown", "W m-2"),
    ("SM", "SM_obs", "R3_SM", "station soil water content vs gridded soil moisture"),
]

PLOT_VARIABLES = ["SurfaceAlbedo", "Rn", "LH", "SH", "T2M", "VPD"]

PALETTE = {
    "albedo": "#B64342",
    "heat": "#D7802A",
    "buffer": "#2E8B9E",
    "soil": "#7C8A3A",
    "neutral_dark": "#3F3F3F",
    "neutral_mid": "#8C8C8C",
    "neutral_light": "#D7D7D7",
    "accent": "#5C6FA7",
}


def ensure_dirs() -> None:
    for folder in [OUT_DIR, TABLE_OUT, FIG_OUT, REPORT_OUT, SCRIPT_COPY_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def archive_script() -> None:
    try:
        shutil.copy2(Path(__file__), SCRIPT_COPY_DIR / Path(__file__).name)
    except Exception:
        pass


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str, size: Tuple[float, float]) -> List[str]:
    fig.set_size_inches(*size)
    out_base = FIG_OUT / stem
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return [str(out_base.with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")


def station_id_from_path(path: Path) -> str:
    m = re.search(r"AMF_([^_]+)_FLUXNET", path.name)
    if m:
        return m.group(1)
    m = re.search(r"AMF_([^_]+)_FLUXNET", path.parent.name)
    if m:
        return m.group(1)
    return path.parent.name


def read_csv_flexible(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, nrows=nrows)


def numeric_series(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            out = pd.to_numeric(df[col], errors="coerce")
            out = out.replace([-9999, -9999.0, -6999, -7777, -8888, 9999], np.nan)
            return out
    return pd.Series(np.nan, index=df.index)


def first_existing_file(folder: Path, patterns: Iterable[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        hits = [p for p in hits if "ERA5_YY" not in p.name]
        if hits:
            return hits[0]
    return None


def load_station_coordinates() -> pd.DataFrame:
    coord_candidates = [
        STATION_PROCESSED / "outputs" / "station_coordinate_inventory_resolved.csv",
        STATION_PROCESSED / "outputs" / "station_coordinate_inventory.csv",
    ]
    coord_frames = []
    for coord_path in coord_candidates:
        if coord_path.exists():
            coord = read_csv_flexible(coord_path)
            cols = ["station_id", "longitude", "latitude", "folder", "note", "coordinate_source_type", "site_name"]
            coord = coord[[c for c in cols if c in coord.columns]].copy()
            coord["coordinate_table_source"] = str(coord_path)
            coord_frames.append(coord)
    if coord_frames:
        coord = pd.concat(coord_frames, ignore_index=True, sort=False)
        coord["has_coordinates"] = coord["longitude"].notna() & coord["latitude"].notna()
        coord = coord.sort_values(["station_id", "has_coordinates"], ascending=[True, False])
        coord = coord.drop_duplicates("station_id", keep="first")
        if coord["has_coordinates"].any():
            return coord.drop(columns=["has_coordinates"])

    rows = []
    for bif in STATION_RAW.glob("*/AMF_*_BIF_*.csv"):
        try:
            df = read_csv_flexible(bif)
        except Exception:
            continue
        sid = station_id_from_path(bif)
        lat = numeric_series(df, ["LOCATION_LAT", "Latitude", "latitude"]).dropna()
        lon = numeric_series(df, ["LOCATION_LONG", "Longitude", "longitude"]).dropna()
        if not lat.empty and not lon.empty:
            rows.append({"station_id": sid, "longitude": lon.iloc[0], "latitude": lat.iloc[0], "folder": bif.parent.name})
    return pd.DataFrame(rows).drop_duplicates("station_id")


def load_station_annual_observations() -> pd.DataFrame:
    coord = load_station_coordinates()
    coord_map = coord.set_index("station_id")[["longitude", "latitude"]].to_dict("index") if not coord.empty else {}

    rows = []
    for folder in sorted(STATION_RAW.glob("AMF_*")):
        if not folder.is_dir():
            continue
        yy_file = first_existing_file(folder, ["*FULLSET_YY*.csv", "*FLUXMET_YY*.csv"])
        if yy_file is None:
            continue
        sid = station_id_from_path(yy_file)
        try:
            df = read_csv_flexible(yy_file)
        except Exception as exc:
            rows.append({"station_id": sid, "source_file": str(yy_file), "read_error": str(exc)})
            continue
        if "TIMESTAMP" not in df.columns:
            continue
        year = pd.to_numeric(df["TIMESTAMP"].astype(str).str.slice(0, 4), errors="coerce").astype("Int64")
        df = df.assign(year=year)
        df = df[df["year"].isin(YEARS)].copy()
        if df.empty:
            continue

        sw_in = numeric_series(df, ["SW_IN_F_MDS", "SW_IN_F", "SW_IN"])
        sw_out = numeric_series(df, ["SW_OUT", "SW_OUT_F"])
        albedo = sw_out / sw_in.where(sw_in > 1)
        albedo = albedo.where((albedo >= 0) & (albedo <= 1))
        sm_cols = [c for c in df.columns if re.match(r"SWC_F_MDS_\d+$", c)]
        if sm_cols:
            sm = df[sm_cols].apply(pd.to_numeric, errors="coerce").replace([-9999, -9999.0], np.nan).mean(axis=1)
        else:
            sm = numeric_series(df, ["SWC_F_MDS", "SWC_F"])

        meta = coord_map.get(sid, {})
        for i, r in df.iterrows():
            item = {
                "station_id": sid,
                "year": int(r["year"]),
                "longitude": meta.get("longitude", np.nan),
                "latitude": meta.get("latitude", np.nan),
                "source_file": str(yy_file),
                "T2M_obs": numeric_series(df.loc[[i]], ["TA_F_MDS", "TA_F"]).iloc[0],
                "Rn_obs": numeric_series(df.loc[[i]], ["NETRAD"]).iloc[0],
                "LH_obs": numeric_series(df.loc[[i]], ["LE_F_MDS", "LE_F"]).iloc[0],
                "SH_obs": numeric_series(df.loc[[i]], ["H_F_MDS", "H_F"]).iloc[0],
                "SWdown_obs": sw_in.loc[i],
                "LWdown_obs": numeric_series(df.loc[[i]], ["LW_IN_F_MDS", "LW_IN_F", "LW_IN"]).iloc[0],
                "VPD_obs": numeric_series(df.loc[[i]], ["VPD_F_MDS", "VPD_F"]).iloc[0],
                "SM_obs": sm.loc[i],
                "SurfaceAlbedo_obs": albedo.loc[i],
            }
            rows.append(item)

    obs = pd.DataFrame(rows)
    if obs.empty:
        return obs

    station_albedo_path = STATION_PROCESSED / "outputs" / "station_albedo_annual_timeseries.csv"
    if station_albedo_path.exists():
        alb = read_csv_flexible(station_albedo_path)
        keep = [c for c in ["station_id", "year", "albedo_ratio_of_sums", "albedo_mean_of_records"] if c in alb.columns]
        if len(keep) >= 3:
            alb = alb[keep].copy()
            obs = obs.merge(alb, on=["station_id", "year"], how="left")
            obs["SurfaceAlbedo_obs"] = obs["SurfaceAlbedo_obs"].fillna(obs.get("albedo_ratio_of_sums"))
            obs["SurfaceAlbedo_obs"] = obs["SurfaceAlbedo_obs"].fillna(obs.get("albedo_mean_of_records"))

    obs["Bowen_obs"] = obs["SH_obs"] / obs["LH_obs"].where(obs["LH_obs"].abs() > 1e-6)
    obs["EF_obs"] = obs["LH_obs"] / (obs["LH_obs"] + obs["SH_obs"]).where((obs["LH_obs"] + obs["SH_obs"]).abs() > 1e-6)
    obs["HeatPartition_obs"] = obs["SH_obs"] - obs["LH_obs"]
    obs = obs.replace([np.inf, -np.inf], np.nan)
    obs = obs.dropna(subset=["longitude", "latitude"], how="any")
    obs.to_csv(TABLE_OUT / "station_annual_observations_from_fluxnet.csv", index=False, encoding="utf-8-sig")
    return obs


def find_raster(variable: str, year: int) -> Optional[Path]:
    folder = COMMON_GRID / variable
    if not folder.exists():
        return None
    patterns = [
        f"{variable}_{year}_R3_common.tif",
        f"{variable}_{year}_*.tif",
        f"*{year}*R3_common.tif",
        f"*{year}*.tif",
    ]
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def sample_raster(path: Path, lon: float, lat: float) -> float:
    try:
        with rasterio.open(path) as src:
            x, y = lon, lat
            if src.crs and str(src.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
                xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
                x, y = xs[0], ys[0]
            val = next(src.sample([(x, y)], masked=True))[0]
            if np.ma.is_masked(val):
                return np.nan
            val = float(val)
            nodata = src.nodata
            if nodata is not None and np.isclose(val, nodata):
                return np.nan
            if not np.isfinite(val) or abs(val) > 1e20:
                return np.nan
            return val
    except Exception:
        return np.nan


def sample_r3_common_grid(obs: pd.DataFrame) -> pd.DataFrame:
    if obs.empty:
        return obs
    out = obs.copy()
    variables = [
        "SurfaceAlbedo",
        "SurfaceAlbedo_GLASS",
        "AlbedoLoss",
        "Rn",
        "LH",
        "SH",
        "SWdown",
        "LWdown",
        "T2M",
        "VPD",
        "SM",
        "SWabs_MODIS",
        "Cloud",
        "Snow",
    ]
    cache: Dict[Tuple[str, int], Optional[Path]] = {}
    for var in variables:
        values = []
        for r in out.itertuples(index=False):
            key = (var, int(r.year))
            if key not in cache:
                cache[key] = find_raster(var, int(r.year))
            path = cache[key]
            values.append(sample_raster(path, float(r.longitude), float(r.latitude)) if path else np.nan)
        out[f"R3_{var}"] = values
    out["R3_Bowen"] = out["R3_SH"] / out["R3_LH"].where(out["R3_LH"].abs() > 1e-6)
    out["R3_EF"] = out["R3_LH"] / (out["R3_LH"] + out["R3_SH"]).where((out["R3_LH"] + out["R3_SH"]).abs() > 1e-6)
    out["R3_HeatPartition"] = out["R3_SH"] - out["R3_LH"]
    out = out.replace([np.inf, -np.inf], np.nan)
    out.to_csv(TABLE_OUT / "station_annual_observations_with_r3_grid_samples.csv", index=False, encoding="utf-8-sig")
    return out


def corr_pair(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 3 or df["a"].nunique() < 2 or df["b"].nunique() < 2:
        return np.nan
    return float(df["a"].corr(df["b"], method=method))


def validation_metrics(df: pd.DataFrame, var_name: str, obs_col: str, grid_col: str, unit: str) -> dict:
    sub = df[["station_id", "year", obs_col, grid_col]].dropna().copy()
    if sub.empty:
        return {
            "Variable": var_name,
            "ObsColumn": obs_col,
            "GridColumn": grid_col,
            "UnitNote": unit,
            "N_station_year": 0,
            "N_sites": 0,
            "Year_min": np.nan,
            "Year_max": np.nan,
        }
    diff = sub[grid_col] - sub[obs_col]
    sub["obs_anom"] = sub[obs_col] - sub.groupby("station_id")[obs_col].transform("mean")
    sub["grid_anom"] = sub[grid_col] - sub.groupby("station_id")[grid_col].transform("mean")
    obs_std = sub.groupby("station_id")[obs_col].transform("std").replace(0, np.nan)
    grid_std = sub.groupby("station_id")[grid_col].transform("std").replace(0, np.nan)
    sub["obs_z"] = (sub[obs_col] - sub.groupby("station_id")[obs_col].transform("mean")) / obs_std
    sub["grid_z"] = (sub[grid_col] - sub.groupby("station_id")[grid_col].transform("mean")) / grid_std
    return {
        "Variable": var_name,
        "ObsColumn": obs_col,
        "GridColumn": grid_col,
        "UnitNote": unit,
        "N_station_year": int(len(sub)),
        "N_sites": int(sub["station_id"].nunique()),
        "Year_min": int(sub["year"].min()),
        "Year_max": int(sub["year"].max()),
        "Pearson_raw": corr_pair(sub[obs_col], sub[grid_col], "pearson"),
        "Spearman_raw": corr_pair(sub[obs_col], sub[grid_col], "spearman"),
        "Bias_grid_minus_obs": float(diff.mean()),
        "RMSE_raw": float(np.sqrt(np.nanmean(diff**2))),
        "MAE_raw": float(diff.abs().mean()),
        "Pearson_station_anomaly": corr_pair(sub["obs_anom"], sub["grid_anom"], "pearson"),
        "Spearman_station_anomaly": corr_pair(sub["obs_anom"], sub["grid_anom"], "spearman"),
        "RMSE_station_anomaly": float(np.sqrt(np.nanmean((sub["grid_anom"] - sub["obs_anom"]) ** 2))),
        "Pearson_station_zscore": corr_pair(sub["obs_z"], sub["grid_z"], "pearson"),
    }


def build_variable_validation(sampled: pd.DataFrame) -> pd.DataFrame:
    rows = [validation_metrics(sampled, *pair) for pair in VARIABLE_PAIRS]

    albedo_site = STATION_PROCESSED / "site_validation_albedo_summary.csv"
    if albedo_site.exists():
        site = read_csv_flexible(albedo_site)
        for _, r in site.iterrows():
            rows.append(
                {
                    "Variable": f"SurfaceAlbedo_{r.get('Product', 'product')}_existing_site_validation",
                    "ObsColumn": "Obs_Albedo",
                    "GridColumn": f"{r.get('Product', 'Product')}_Albedo",
                    "UnitNote": "unitless; existing station validation table",
                    "N_station_year": r.get("N", np.nan),
                    "N_sites": r.get("Sites", np.nan),
                    "Year_min": r.get("Year_min", np.nan),
                    "Year_max": r.get("Year_max", np.nan),
                    "Pearson_raw": r.get("r", np.nan),
                    "RMSE_raw": r.get("RMSE", np.nan),
                    "Bias_grid_minus_obs": r.get("Bias", np.nan),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_OUT / "station_variable_validation_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def period_name(year: int) -> Optional[str]:
    for label, years in PERIODS.items():
        if year in years:
            return label
    return None


def period_means(sampled: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = sampled.copy()
    df["Period"] = df["year"].apply(period_name)
    df = df.dropna(subset=["Period"])
    value_cols = [
        "SurfaceAlbedo_obs",
        "AlbedoLoss_obs",
        "Rn_obs",
        "LH_obs",
        "SH_obs",
        "T2M_obs",
        "VPD_obs",
        "Bowen_obs",
        "EF_obs",
        "HeatPartition_obs",
        "R3_SurfaceAlbedo",
        "R3_AlbedoLoss",
        "R3_Rn",
        "R3_LH",
        "R3_SH",
        "R3_T2M",
        "R3_VPD",
        "R3_Bowen",
        "R3_EF",
        "R3_HeatPartition",
    ]
    df["AlbedoLoss_obs"] = np.nan
    if "SurfaceAlbedo_obs" in df.columns:
        baseline_map = df[df["year"].between(2001, 2010)].groupby("station_id")["SurfaceAlbedo_obs"].mean()
        all_base_map = df.groupby("station_id")["SurfaceAlbedo_obs"].mean()
        base = df["station_id"].map(baseline_map).fillna(df["station_id"].map(all_base_map))
        df["AlbedoLoss_obs"] = -(df["SurfaceAlbedo_obs"] - base)

    agg_cols = [c for c in value_cols if c in df.columns]
    period = (
        df.groupby(["station_id", "longitude", "latitude", "Period"], dropna=False)
        .agg({**{c: "mean" for c in agg_cols}, "year": "count"})
        .rename(columns={"year": "N_years"})
        .reset_index()
    )
    period.to_csv(TABLE_OUT / "station_period_means_obs_and_r3.csv", index=False, encoding="utf-8-sig")

    wide = period.pivot_table(index=["station_id", "longitude", "latitude"], columns="Period", values=agg_cols + ["N_years"])
    wide.columns = [f"{var}_{period_label}" for var, period_label in wide.columns]
    wide = wide.reset_index()
    rows = []
    for _, r in wide.iterrows():
        out = {"station_id": r["station_id"], "longitude": r["longitude"], "latitude": r["latitude"]}
        n1 = r.get("N_years_P1_2001_2014", np.nan)
        n3 = r.get("N_years_P3_2020_2024", np.nan)
        out["N_years_P1"] = n1
        out["N_years_P3"] = n3
        out["Strict_P3_minus_P1_available"] = bool(pd.notna(n1) and pd.notna(n3) and n1 >= 2 and n3 >= 2)
        for col in agg_cols:
            out[f"{col}_P3_minus_P1"] = r.get(f"{col}_P3_2020_2024", np.nan) - r.get(f"{col}_P1_2001_2014", np.nan)
        rows.append(out)
    changes = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    changes.to_csv(TABLE_OUT / "station_period_changes_p3_minus_p1.csv", index=False, encoding="utf-8-sig")
    return period, changes


def map_path_xgb(model: str, period: str, feature: str, signed: bool = False) -> Path:
    metric = "mean_signed_SHAP" if signed else "mean_abs_SHAP"
    return SHAP_DIR / "spatial_maps" / model / period / f"{model}_{period}_{feature}_{metric}.tif"


def map_path_sem(model: str, period: str, feature: str, signed: bool = False) -> Path:
    metric = "mean_signed_contribution" if signed else "mean_abs_contribution"
    return ROOT / "R3_01_Piecewise SEM" / "spatial_contribution_maps" / model / period / f"{model}_{period}_{feature}_{metric}.tif"


def sample_station_period_maps(changes: pd.DataFrame) -> pd.DataFrame:
    if changes.empty:
        return changes
    out = changes.copy()
    map_specs = [
        ("XGB_Model2_AlbedoLoss_abs_SHAP", map_path_xgb("Model2_main_SHAP_core", "P1_2001_2014", "AlbedoLoss", False), map_path_xgb("Model2_main_SHAP_core", "P3_2020_2024", "AlbedoLoss", False)),
        ("XGB_Model2_AlbedoLoss_signed_SHAP", map_path_xgb("Model2_main_SHAP_core", "P1_2001_2014", "AlbedoLoss", True), map_path_xgb("Model2_main_SHAP_core", "P3_2020_2024", "AlbedoLoss", True)),
        ("XGB_M3_T2M_AlbedoLoss_abs_SHAP", map_path_xgb("M3_06_T2M_from_albedo_Rn_energy_SM", "P1_2001_2014", "AlbedoLoss", False), map_path_xgb("M3_06_T2M_from_albedo_Rn_energy_SM", "P3_2020_2024", "AlbedoLoss", False)),
        ("SEM_Model2_AlbedoLoss_abs_contribution", map_path_sem("Model2_main_core_contribution", "P1_2001_2014", "AlbedoLoss", False), map_path_sem("Model2_main_core_contribution", "P3_2020_2024", "AlbedoLoss", False)),
    ]
    for name, p1, p3 in map_specs:
        p1_vals, p3_vals = [], []
        for r in out.itertuples(index=False):
            p1_vals.append(sample_raster(p1, float(r.longitude), float(r.latitude)) if p1.exists() else np.nan)
            p3_vals.append(sample_raster(p3, float(r.longitude), float(r.latitude)) if p3.exists() else np.nan)
        out[f"{name}_P1"] = p1_vals
        out[f"{name}_P3"] = p3_vals
        out[f"{name}_P3_minus_P1"] = out[f"{name}_P3"] - out[f"{name}_P1"]
    out.to_csv(TABLE_OUT / "station_sampled_result_map_changes.csv", index=False, encoding="utf-8-sig")
    return out


def build_mechanism_consistency(sampled_changes: pd.DataFrame) -> pd.DataFrame:
    if sampled_changes.empty:
        out = pd.DataFrame()
        out.to_csv(TABLE_OUT / "station_mechanism_consistency_metrics.csv", index=False, encoding="utf-8-sig")
        return out
    x_cols = [
        "XGB_Model2_AlbedoLoss_abs_SHAP_P3_minus_P1",
        "XGB_Model2_AlbedoLoss_signed_SHAP_P3_minus_P1",
        "XGB_M3_T2M_AlbedoLoss_abs_SHAP_P3_minus_P1",
        "SEM_Model2_AlbedoLoss_abs_contribution_P3_minus_P1",
    ]
    y_cols = [
        "SH_obs_P3_minus_P1",
        "LH_obs_P3_minus_P1",
        "Bowen_obs_P3_minus_P1",
        "EF_obs_P3_minus_P1",
        "T2M_obs_P3_minus_P1",
        "VPD_obs_P3_minus_P1",
        "HeatPartition_obs_P3_minus_P1",
        "R3_SH_P3_minus_P1",
        "R3_LH_P3_minus_P1",
        "R3_Bowen_P3_minus_P1",
        "R3_EF_P3_minus_P1",
        "R3_T2M_P3_minus_P1",
    ]
    rows = []
    for x in x_cols:
        for y in y_cols:
            if x not in sampled_changes.columns or y not in sampled_changes.columns:
                continue
            sub = sampled_changes[["station_id", "Strict_P3_minus_P1_available", x, y]].dropna()
            if sub.empty:
                continue
            strict = sub[sub["Strict_P3_minus_P1_available"].astype(bool)].copy()
            use = strict if len(strict) >= 4 else sub
            x_med = use[x].median()
            high = use[use[x] >= x_med][y].mean() if len(use) else np.nan
            low = use[use[x] < x_med][y].mean() if len(use) else np.nan
            rows.append(
                {
                    "ResultMetric": x,
                    "ObservedOrGridChange": y,
                    "N_sites": int(len(use)),
                    "StrictOnly": bool(len(strict) >= 4),
                    "Pearson": corr_pair(use[x], use[y], "pearson"),
                    "Spearman": corr_pair(use[x], use[y], "spearman"),
                    "HighMinusLow_by_result_metric_median": high - low,
                    "HighGroupMean": high,
                    "LowGroupMean": low,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_OUT / "station_mechanism_consistency_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def add_anomalies(sampled: pd.DataFrame, obs_col: str, grid_col: str) -> pd.DataFrame:
    sub = sampled[["station_id", "year", obs_col, grid_col]].dropna().copy()
    if sub.empty:
        return sub
    sub["obs_anom"] = sub[obs_col] - sub.groupby("station_id")[obs_col].transform("mean")
    sub["grid_anom"] = sub[grid_col] - sub.groupby("station_id")[grid_col].transform("mean")
    return sub


def plot_variable_validation(sampled: pd.DataFrame, metrics: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
    axes = axes.ravel()
    for i, var in enumerate(PLOT_VARIABLES):
        ax = axes[i]
        pair = next((p for p in VARIABLE_PAIRS if p[0] == var), None)
        if pair is None:
            continue
        _, obs_col, grid_col, _ = pair
        sub = add_anomalies(sampled, obs_col, grid_col)
        if not sub.empty:
            ax.scatter(sub["obs_anom"], sub["grid_anom"], s=10, alpha=0.55, color=PALETTE["accent"], lw=0)
            lim = np.nanpercentile(np.abs(pd.concat([sub["obs_anom"], sub["grid_anom"]])), 98)
            if np.isfinite(lim) and lim > 0:
                ax.plot([-lim, lim], [-lim, lim], color="#4A4A4A", lw=0.7, ls="--")
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
            if len(sub) >= 3 and sub["obs_anom"].nunique() > 1:
                coef = np.polyfit(sub["obs_anom"], sub["grid_anom"], 1)
                xx = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
                ax.plot(xx, coef[0] * xx + coef[1], color=PALETTE["albedo"] if var == "SurfaceAlbedo" else PALETTE["heat"], lw=1.1)
        m = metrics[metrics["Variable"].eq(var)]
        r_text = "r=NA"
        n_text = "N=0"
        if not m.empty:
            r_val = m.iloc[0].get("Pearson_station_anomaly", np.nan)
            n_val = m.iloc[0].get("N_station_year", 0)
            site_val = m.iloc[0].get("N_sites", 0)
            r_text = f"r={r_val:.2f}" if pd.notna(r_val) else "r=NA"
            n_text = f"N={int(n_val)}, sites={int(site_val)}"
        ax.text(0.03, 0.95, f"{r_text}\n{n_text}", transform=ax.transAxes, va="top", ha="left", fontsize=6.2)
        ax.set_title(var, fontsize=7.5)
        ax.set_xlabel("Station anomaly")
        ax.set_ylabel("Grid anomaly")
        ax.grid(True, color="#E9E9E9", lw=0.5)
        panel_label(ax, chr(97 + i))
    save_figure(fig, "FigS_R3_station_variable_validation_anomaly_scatter", (7.2, 4.6))


def plot_metric_bar(metrics: pd.DataFrame) -> None:
    apply_style()
    sub = metrics[metrics["Variable"].isin(PLOT_VARIABLES)].copy()
    if sub.empty:
        return
    sub["Pearson_station_anomaly"] = pd.to_numeric(sub["Pearson_station_anomaly"], errors="coerce")
    sub = sub.sort_values("Pearson_station_anomaly", ascending=True)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    colors = [PALETTE["albedo"] if v == "SurfaceAlbedo" else PALETTE["accent"] for v in sub["Variable"]]
    ax.barh(sub["Variable"], sub["Pearson_station_anomaly"], color=colors, height=0.65)
    ax.axvline(0, color="#4A4A4A", lw=0.8)
    xmin = min(-0.05, float(np.nanmin(sub["Pearson_station_anomaly"])) - 0.05)
    xmax = max(0.15, float(np.nanmax(sub["Pearson_station_anomaly"])) + 0.14)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Pearson r of station-wise annual anomalies")
    ax.set_ylabel("")
    for y, (_, r) in enumerate(sub.iterrows()):
        val = r["Pearson_station_anomaly"]
        label = f"N={int(r['N_station_year'])}" if pd.notna(r["N_station_year"]) else ""
        if not np.isfinite(val):
            continue
        x_text = val + 0.02 if val > 0.04 else 0.025
        ax.text(x_text, y, label, va="center", ha="left", fontsize=6)
    ax.grid(True, axis="x", color="#E9E9E9", lw=0.5)
    save_figure(fig, "FigS_R3_station_variable_validation_metric_bar", (3.6, 2.8))


def plot_mechanism_consistency(sampled_changes: pd.DataFrame, consistency: pd.DataFrame) -> None:
    apply_style()
    x = "XGB_Model2_AlbedoLoss_abs_SHAP_P3_minus_P1"
    panels = [
        ("SH_obs_P3_minus_P1", "Sensible heat"),
        ("LH_obs_P3_minus_P1", "Latent heat"),
        ("Bowen_obs_P3_minus_P1", "Bowen ratio"),
        ("EF_obs_P3_minus_P1", "Evaporative fraction"),
        ("T2M_obs_P3_minus_P1", "Air temperature"),
        ("VPD_obs_P3_minus_P1", "VPD"),
    ]
    if sampled_changes.empty or x not in sampled_changes.columns:
        return
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
    axes = axes.ravel()
    for i, (y, title) in enumerate(panels):
        ax = axes[i]
        if y not in sampled_changes.columns:
            ax.axis("off")
            continue
        sub = sampled_changes[["station_id", "Strict_P3_minus_P1_available", x, y]].dropna().copy()
        if sub.empty:
            ax.axis("off")
            continue
        strict = sub[sub["Strict_P3_minus_P1_available"].astype(bool)]
        use = strict if len(strict) >= 4 else sub
        ax.scatter(use[x], use[y], s=22, color=PALETTE["albedo"], alpha=0.75, lw=0)
        if len(use) >= 3 and use[x].nunique() > 1 and use[y].nunique() > 1:
            coef = np.polyfit(use[x], use[y], 1)
            xx = np.linspace(use[x].min(), use[x].max(), 100)
            ax.plot(xx, coef[0] * xx + coef[1], color=PALETTE["neutral_dark"], lw=1.0)
        cm = consistency[(consistency["ResultMetric"].eq(x)) & (consistency["ObservedOrGridChange"].eq(y))]
        if not cm.empty:
            r = cm.iloc[0]["Spearman"]
            n = cm.iloc[0]["N_sites"]
            mode = "strict" if bool(cm.iloc[0]["StrictOnly"]) else "available"
            ax.text(0.03, 0.95, f"rho={r:.2f}\nN={int(n)} ({mode})", transform=ax.transAxes, va="top", ha="left", fontsize=6.2)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("AlbedoLoss SHAP delta")
        ax.set_ylabel("Station P3-P1")
        ax.grid(True, color="#E9E9E9", lw=0.5)
        for _, r in use.iterrows():
            ax.text(r[x], r[y], str(r["station_id"]), fontsize=4.8, alpha=0.7)
        panel_label(ax, chr(97 + i))
    save_figure(fig, "FigS_R3_station_mechanism_consistency_scatter", (7.2, 4.6))


def load_raster_array(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        if nodata is not None:
            arr[np.isclose(arr, nodata)] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        arr[np.abs(arr) > 1e20] = np.nan
        profile = src.profile.copy()
        bounds = src.bounds
        profile["_extent"] = (bounds.left, bounds.right, bounds.bottom, bounds.top)
        return arr, profile


def plot_station_map(sampled_changes: pd.DataFrame) -> None:
    p1 = map_path_xgb("Model2_main_SHAP_core", "P1_2001_2014", "AlbedoLoss", False)
    p3 = map_path_xgb("Model2_main_SHAP_core", "P3_2020_2024", "AlbedoLoss", False)
    if not p1.exists() or not p3.exists():
        return
    a1, prof = load_raster_array(p1)
    a3, _ = load_raster_array(p3)
    delta = a3 - a1
    extent = prof["_extent"]
    vmax = np.nanpercentile(np.abs(delta), 98)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 0.1
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    im = ax.imshow(
        delta,
        extent=extent,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        interpolation="nearest",
    )
    if not sampled_changes.empty:
        strict = sampled_changes[sampled_changes["Strict_P3_minus_P1_available"].astype(bool)].copy()
        loose = sampled_changes[~sampled_changes["Strict_P3_minus_P1_available"].astype(bool)].copy()
        if not loose.empty:
            ax.scatter(loose["longitude"], loose["latitude"], s=16, facecolors="white", edgecolors="#4A4A4A", lw=0.6, label="station, limited P1/P3 years")
        if not strict.empty:
            ax.scatter(
                strict["longitude"],
                strict["latitude"],
                s=30,
                facecolors=PALETTE["albedo"],
                edgecolors="black",
                lw=0.4,
                label="station, strict P1/P3",
            )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Station overlay on AlbedoLoss SHAP P3-P1 map", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("AlbedoLoss mean |SHAP| delta", fontsize=6.5)
    ax.legend(loc="lower left", fontsize=6)
    save_figure(fig, "FigS_R3_station_overlay_on_albedoloss_shap_delta_map", (7.2, 3.2))


def summarize_station_coverage(obs: pd.DataFrame, sampled: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"Metric": "station_directories", "Value": len([p for p in STATION_RAW.glob("AMF_*") if p.is_dir()])},
        {"Metric": "stations_with_annual_flux_observations", "Value": obs["station_id"].nunique() if not obs.empty else 0},
        {"Metric": "station_years_2001_2024", "Value": len(obs) if not obs.empty else 0},
        {"Metric": "stations_with_r3_samples", "Value": sampled["station_id"].nunique() if not sampled.empty else 0},
        {"Metric": "station_years_with_any_r3_sample", "Value": sampled.filter(like="R3_").notna().any(axis=1).sum() if not sampled.empty else 0},
        {"Metric": "stations_with_strict_p3_minus_p1", "Value": int(changes["Strict_P3_minus_P1_available"].sum()) if not changes.empty else 0},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_OUT / "station_validation_coverage_summary.csv", index=False, encoding="utf-8-sig")
    return out


def load_legacy_fluxnet_era5_metrics() -> pd.DataFrame:
    files = sorted(LEGACY_STATS.glob("R1[45]_Fluxnet_ERA5_*.csv"))
    rows = []
    if not files:
        return pd.DataFrame()
    path = files[-1]
    df = read_csv_flexible(path)
    pairs = [
        ("Albedo", "Albedo_Obs", "Albedo_ERA5"),
        ("Temp", "Temp_Obs", "Temp_ERA5"),
        ("NetRad", "NetRad_Obs", "NetRad_ERA5"),
        ("Sensible", "Sensible_Obs", "Sensible_ERA5"),
        ("Latent", "Latent_Obs", "Latent_ERA5"),
    ]
    for var, obs, grid in pairs:
        if obs not in df.columns or grid not in df.columns:
            continue
        rows.append(validation_metrics(df.rename(columns={"Site": "station_id", "Year": "year"}), var, obs, grid, "legacy station validation"))
    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(0, "SourceFile", str(path))
        out.to_csv(TABLE_OUT / "legacy_fluxnet_era5_validation_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def write_report(coverage: pd.DataFrame, metrics: pd.DataFrame, consistency: pd.DataFrame, legacy: pd.DataFrame) -> None:
    def table_text(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except ImportError:
            return "```csv\n" + df.to_csv(index=False) + "```"

    def top_metrics() -> str:
        if metrics.empty:
            return "No station-variable validation metrics were generated."
        cols = ["Variable", "N_station_year", "N_sites", "Pearson_station_anomaly", "RMSE_station_anomaly", "Pearson_raw", "RMSE_raw"]
        return table_text(metrics[[c for c in cols if c in metrics.columns]])

    def consistency_table() -> str:
        if consistency.empty:
            return "No station mechanism-consistency metrics were generated."
        sub = consistency[consistency["ResultMetric"].eq("XGB_Model2_AlbedoLoss_abs_SHAP_P3_minus_P1")].copy()
        if sub.empty:
            sub = consistency.copy()
        cols = ["ObservedOrGridChange", "N_sites", "StrictOnly", "Spearman", "Pearson", "HighMinusLow_by_result_metric_median"]
        return table_text(sub[[c for c in cols if c in sub.columns]])

    legacy_text = ""
    if not legacy.empty:
        cols = ["Variable", "N_station_year", "N_sites", "Pearson_raw", "RMSE_raw", "Bias_grid_minus_obs"]
        legacy_text = table_text(legacy[[c for c in cols if c in legacy.columns]])
    else:
        legacy_text = "No legacy Fluxnet-ERA5 validation table was found."

    report = f"""# Result 1.3 station validation report

## Purpose

This validation uses station data as an independent control group for the variables and mechanism direction used in Result 1.3. It is not a replacement for the global gridded analysis because FLUXNET/AmeriFlux sites are spatially sparse and biome-biased.

## Coverage

{table_text(coverage)}

## Variable validation

The preferred statistics for manuscript text are station-wise annual anomaly correlations because they reduce site representativeness, elevation and biome-offset effects.

{top_metrics()}

## Mechanism consistency

The table below compares sampled AlbedoLoss SHAP P3-P1 changes at station pixels with station-observed P3-P1 changes in energy partitioning and near-surface state.

{consistency_table()}

## Legacy Fluxnet-ERA5 validation

{legacy_text}

## Supplementary figures

- `FigS_R3_station_variable_validation_anomaly_scatter`: station annual anomalies versus Result 3 common-grid annual anomalies.
- `FigS_R3_station_variable_validation_metric_bar`: summary of station-anomaly correlations.
- `FigS_R3_station_mechanism_consistency_scatter`: sampled AlbedoLoss SHAP delta versus station P3-P1 changes.
- `FigS_R3_station_overlay_on_albedoloss_shap_delta_map`: station overlay on global AlbedoLoss SHAP P3-P1 map.

## Interpretation boundary

Use these outputs as site-level reliability and direction-consistency evidence. Strong causal wording should still rely on the designed Model 1-3 workflow and mechanism/path analyses, with station validation reported as an external check.
"""
    (REPORT_OUT / "Result1_3_station_validation_report_zh.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    archive_script()
    obs = load_station_annual_observations()
    sampled = sample_r3_common_grid(obs)
    metrics = build_variable_validation(sampled)
    period, changes = period_means(sampled)
    sampled_changes = sample_station_period_maps(changes)
    consistency = build_mechanism_consistency(sampled_changes)
    coverage = summarize_station_coverage(obs, sampled, sampled_changes)
    legacy = load_legacy_fluxnet_era5_metrics()
    plot_variable_validation(sampled, metrics)
    plot_metric_bar(metrics)
    plot_mechanism_consistency(sampled_changes, consistency)
    plot_station_map(sampled_changes)
    write_report(coverage, metrics, consistency, legacy)
    print("Station validation outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()

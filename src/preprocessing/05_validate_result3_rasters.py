# -*- coding: utf-8 -*-
"""
Workflow 1 - QC Result 3 rasters and SHAP variable availability.

This script checks:
1. Numeric range, unit metadata, sign convention, valid pixels, and missing years
   for generated Result 3 annual rasters.
2. Whether each variable planned for XGBoost/SHAP is available as spatial
   annual rasters, annual statistical controls, or both.

Outputs are written only under:
    D:/10_Research/01_Datasets/04_Results/Result3_Figures_optimized/R3_QC_Checks
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio


R3_ROOT = Path(r"D:\10_Research\01_Datasets\04_Results\Result3_Figures_optimized")
COMMON_DIR = R3_ROOT / "R3_CommonGrid_Rasters"
NATIVE_DIR = R3_ROOT / "R3_Annual_Rasters_NativeExact"
TABLE_DIR = R3_ROOT / "R3_Tables"
META_DIR = R3_ROOT / "R3_Metadata"
OUT_DIR = R3_ROOT / "R3_QC_Checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2001, 2025))
PHASES = {
    "P1": list(range(2001, 2015)),
    "P2": list(range(2015, 2020)),
    "P3": list(range(2020, 2025)),
}

MANIFEST_CSV = R3_ROOT / "R3_annual_raster_manifest.csv"
QC_CSV = R3_ROOT / "R3_annual_raster_qc.csv"
ANNUAL_CONTROLS_CSV = TABLE_DIR / "R3_annual_background_controls.csv"


EXPECTED: Dict[str, Dict[str, object]] = {
    "T2M": {"min": -90, "max": 70, "units": ["degree_C", "degC", "C"], "role": "target", "allow_negative": True, "core": True},
    "SurfaceAlbedo": {"min": 0, "max": 1, "units": ["fraction"], "role": "surface_albedo", "allow_negative": False, "core": True},
    "SurfaceAlbedo_GLASS": {"min": 0, "max": 1, "units": ["fraction"], "role": "albedo_sensitivity", "allow_negative": False, "core": False},
    "AlbedoLoss": {"min": -8, "max": 8, "units": ["standardized", "z"], "role": "surface_albedo_loss", "allow_negative": True, "core": True},
    "Rn": {"min": -300, "max": 600, "units": ["W m-2", "W/m2"], "role": "surface_net_radiation", "allow_negative": True, "core": True},
    "SWdown": {"min": 0, "max": 550, "units": ["W m-2", "W/m2"], "role": "shortwave_input", "allow_negative": False, "core": False},
    "LWdown": {"min": 0, "max": 650, "units": ["W m-2", "W/m2"], "role": "longwave_input", "allow_negative": False, "core": False},
    "SWabs_MODIS": {"min": 0, "max": 550, "units": ["W m-2", "W/m2", "same_as_SWdown"], "role": "absorbed_shortwave", "allow_negative": False, "core": False},
    "LH": {"min": -500, "max": 500, "units": ["W m-2", "W/m2"], "role": "latent_heat", "allow_negative": True, "core": True},
    "SH": {"min": -500, "max": 500, "units": ["W m-2", "W/m2"], "role": "sensible_heat", "allow_negative": True, "core": True},
    "SM": {"min": -0.05, "max": 1.50, "units": ["m3 m-3", "m3/m3"], "role": "soil_moisture", "allow_negative": False, "core": True},
    "VPD": {"min": 0, "max": 20, "units": ["kPa"], "role": "water_limitation", "allow_negative": False, "core": False},
    "Cloud": {"min": 0, "max": 1, "units": ["fraction"], "role": "cloud_control", "allow_negative": False, "core": False},
    "Snow": {"min": 0, "max": 1, "units": ["fraction"], "role": "snow_control", "allow_negative": False, "core": False},
    "AOD": {"min": 0, "max": 5, "units": ["unitless"], "role": "aerosol_control", "allow_negative": False, "core": False},
    "SST": {"min": -20, "max": 20, "units": ["degree_C_anomaly", "degC"], "role": "ocean_background", "allow_negative": True, "core": False},
    "SST_OceanRaw": {"min": -20, "max": 20, "units": ["degree_C_anomaly", "degC"], "role": "ocean_background_raw", "allow_negative": True, "core": False},
    "CO2_CT2025_PBL": {"min": 330, "max": 500, "units": ["ppm", "micromol mol-1"], "role": "co2_spatial_background", "allow_negative": False, "core": False},
    "CO2_RF_CT2025_PBL": {"min": -2, "max": 3, "units": ["W m-2", "W/m2"], "role": "co2_rf_spatial_background", "allow_negative": True, "core": False},
    "area_weight": {"min": 0, "max": 1, "units": ["cos(latitude)"], "role": "area_weight", "allow_negative": False, "core": False},
    "global_land_mask": {"min": 0, "max": 1, "units": ["binary"], "role": "land_mask", "allow_negative": False, "core": False},
}

MODEL_VARIABLES: Dict[str, Dict[str, object]] = {
    "Model1_background_stripping": {
        "purpose": "Strip large-scale/background controls before land mechanism SHAP.",
        "variables": [
            "CO2_Global_ppm_annual_control",
            "CO2_RF_annual_control",
            "CO2_RF_ref2000_annual_control",
            "CO2_CT2025_PBL",
            "CO2_RF_CT2025_PBL",
            "ONI_annual_control",
            "ONI_Lag1_annual_control",
            "ONI_CrossYear_annual_control",
            "SST_anomaly_annual_control",
            "SST",
            "AOD_annual_control",
            "AOD",
            "Snow_annual_control",
            "Snow",
            "Cloud",
        ],
    },
    "Model2_residual_SHAP_core": {
        "purpose": "Core land-surface residual SHAP, aligned with albedo loss -> Rn -> energy partition.",
        "variables": ["AlbedoLoss", "Rn", "SM", "LH", "SH", "T2M"],
    },
    "Model2_sensitivity_SHAP": {
        "purpose": "Sensitivity model including background and diagnostic controls.",
        "variables": ["AlbedoLoss", "Rn", "SM", "LH", "SH", "Cloud", "Snow", "AOD"],
    },
    "Model3_mechanism_chain_path_beta": {
        "purpose": "Mechanism-chain SHAP plus standardized path beta.",
        "variables": [
            "SurfaceAlbedo",
            "AlbedoLoss",
            "SWdown",
            "SWabs_MODIS",
            "LWdown",
            "Rn",
            "SM",
            "LH",
            "SH",
            "VPD",
            "Cloud",
            "Snow",
            "T2M",
        ],
    },
}


def phase_from_year(year: int) -> str:
    year = int(year)
    if 2001 <= year <= 2014:
        return "P1"
    if 2015 <= year <= 2019:
        return "P2"
    if 2020 <= year <= 2024:
        return "P3"
    return "Other"


def common_path(var: str, year: Optional[int] = None) -> Path:
    folder = COMMON_DIR / var
    if year is None:
        return folder / f"{var}_R3_common.tif"
    return folder / f"{var}_{year}_R3_common.tif"


def parse_year_from_name(path: Path, var: str) -> Optional[int]:
    m = re.search(rf"{re.escape(var)}_(\d{{4}})_R3", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{4})", path.name)
    return int(m.group(1)) if m else None


def list_spatial_variables() -> List[str]:
    if not COMMON_DIR.exists():
        return []
    return sorted([p.name for p in COMMON_DIR.iterdir() if p.is_dir()])


def years_available(var: str) -> List[int]:
    folder = COMMON_DIR / var
    if not folder.exists():
        return []
    years = []
    for path in folder.glob("*.tif"):
        y = parse_year_from_name(path, var)
        if y in YEARS:
            years.append(int(y))
    return sorted(set(years))


def read_stats(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {
            "Exists": False,
            "Width": np.nan,
            "Height": np.nan,
            "CRS": "",
            "Transform": "",
            "N": 0,
            "ValidFraction": 0.0,
            "Min": np.nan,
            "P01": np.nan,
            "Mean": np.nan,
            "Median": np.nan,
            "P99": np.nan,
            "Max": np.nan,
            "Std": np.nan,
            "NegativeFraction": np.nan,
            "ZeroFraction": np.nan,
            "Tags": "{}",
        }
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float64").filled(np.nan)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        tags = src.tags()
        profile_bits = {
            "Exists": True,
            "Width": src.width,
            "Height": src.height,
            "CRS": str(src.crs),
            "Transform": str(src.transform),
            "Tags": json.dumps(tags, ensure_ascii=False),
        }
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return {
            **profile_bits,
            "N": 0,
            "ValidFraction": 0.0,
            "Min": np.nan,
            "P01": np.nan,
            "Mean": np.nan,
            "Median": np.nan,
            "P99": np.nan,
            "Max": np.nan,
            "Std": np.nan,
            "NegativeFraction": np.nan,
            "ZeroFraction": np.nan,
        }
    return {
        **profile_bits,
        "N": int(vals.size),
        "ValidFraction": float(vals.size / arr.size),
        "Min": float(np.nanmin(vals)),
        "P01": float(np.nanpercentile(vals, 1)),
        "Mean": float(np.nanmean(vals)),
        "Median": float(np.nanmedian(vals)),
        "P99": float(np.nanpercentile(vals, 99)),
        "Max": float(np.nanmax(vals)),
        "Std": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0,
        "NegativeFraction": float(np.mean(vals < 0)),
        "ZeroFraction": float(np.mean(vals == 0)),
    }


def expected_for(var: str) -> Dict[str, object]:
    return EXPECTED.get(var, {"min": np.nan, "max": np.nan, "units": [], "role": "unknown", "allow_negative": True, "core": False})


def check_issues(var: str, stats: Dict[str, object], manifest_units: str = "") -> Tuple[str, str]:
    exp = expected_for(var)
    issues: List[str] = []
    level = "pass"
    n = float(stats.get("N", 0) or 0)
    vmin = float(stats.get("Min", np.nan)) if pd.notna(stats.get("Min", np.nan)) else np.nan
    vmax = float(stats.get("Max", np.nan)) if pd.notna(stats.get("Max", np.nan)) else np.nan
    mean = float(stats.get("Mean", np.nan)) if pd.notna(stats.get("Mean", np.nan)) else np.nan
    neg = float(stats.get("NegativeFraction", np.nan)) if pd.notna(stats.get("NegativeFraction", np.nan)) else np.nan

    if not stats.get("Exists", False):
        return "missing", "file_missing"
    if n <= 0:
        return "fail", "no_valid_pixels"

    lo = exp.get("min", np.nan)
    hi = exp.get("max", np.nan)
    if pd.notna(lo) and np.isfinite(vmin) and vmin < float(lo):
        issues.append(f"min_below_expected:{vmin:.4g}<{float(lo):.4g}")
    if pd.notna(hi) and np.isfinite(vmax) and vmax > float(hi):
        issues.append(f"max_above_expected:{vmax:.4g}>{float(hi):.4g}")
    if exp.get("allow_negative") is False and np.isfinite(neg) and neg > 0:
        issues.append(f"unexpected_negative_fraction:{neg:.4f}")

    if var == "T2M" and np.isfinite(mean) and mean > 100:
        issues.append("T2M_may_be_Kelvin_not_Celsius")
    if var in {"SurfaceAlbedo", "SurfaceAlbedo_GLASS", "Cloud", "Snow"} and np.isfinite(vmax) and vmax > 1.5:
        issues.append("fraction_variable_may_be_percent_or_unscaled")
    if var == "AOD" and np.isfinite(mean) and (mean < 0.001 or mean > 2.0):
        issues.append("AOD_mean_unusual_check_scale_factor")
    if var == "VPD" and np.isfinite(mean) and mean > 10:
        issues.append("VPD_mean_high_check_hPa_vs_kPa")
    if var in {"Rn", "SWdown", "LWdown", "LH", "SH", "SWabs_MODIS"} and np.isfinite(vmax) and vmax > 1000:
        issues.append("radiation_or_flux_too_large_check_accumulated_units")
    if var == "AlbedoLoss" and np.isfinite(mean) and abs(mean) > 1:
        issues.append("AlbedoLoss_mean_far_from_zero_check_standardization_mask")

    expected_units = [str(u).lower() for u in exp.get("units", [])]
    if manifest_units and expected_units:
        unit_l = str(manifest_units).lower()
        if not any(u in unit_l for u in expected_units):
            issues.append(f"unit_metadata_unexpected:{manifest_units}")

    if issues:
        level = "check"
        if any(x.startswith(("no_valid", "file_missing")) for x in issues):
            level = "fail"
    return level, "; ".join(issues)


def load_manifest_units() -> Dict[str, str]:
    if not MANIFEST_CSV.exists():
        return {}
    df = pd.read_csv(MANIFEST_CSV)
    units = {}
    if "Variable" in df.columns and "Units" in df.columns:
        for var, g in df.groupby("Variable"):
            vals = [str(v) for v in g["Units"].dropna().unique() if str(v).strip()]
            if vals:
                units[str(var)] = vals[0]
    return units


def build_raster_qc() -> pd.DataFrame:
    manifest_units = load_manifest_units()
    rows: List[Dict[str, object]] = []
    variables = sorted(set(list_spatial_variables()) | set(EXPECTED.keys()))
    for var in variables:
        if var in {"area_weight", "global_land_mask"}:
            candidate_paths = [common_path(var, None)]
            if not candidate_paths[0].exists():
                folder = COMMON_DIR / var
                candidate_paths = sorted(folder.glob("*.tif")) if folder.exists() else candidate_paths
            for path in candidate_paths:
                stats = read_stats(path)
                level, issues = check_issues(var, stats, manifest_units.get(var, ""))
                rows.append(
                    {
                        "Variable": var,
                        "Year": "",
                        "Phase": "",
                        "ExpectedRole": expected_for(var).get("role", "unknown"),
                        "ExpectedUnit": ";".join(expected_for(var).get("units", [])),
                        "ManifestUnit": manifest_units.get(var, ""),
                        "QC_Level": level,
                        "Issues": issues,
                        "Path": str(path),
                        **stats,
                    }
                )
            continue

        for year in YEARS:
            path = common_path(var, year)
            stats = read_stats(path)
            level, issues = check_issues(var, stats, manifest_units.get(var, ""))
            rows.append(
                {
                    "Variable": var,
                    "Year": year,
                    "Phase": phase_from_year(year),
                    "ExpectedRole": expected_for(var).get("role", "unknown"),
                    "ExpectedUnit": ";".join(expected_for(var).get("units", [])),
                    "ManifestUnit": manifest_units.get(var, ""),
                    "QC_Level": level,
                    "Issues": issues,
                    "Path": str(path),
                    **stats,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "01_raster_numeric_range_unit_sign_qc.csv", index=False, encoding="utf-8-sig")
    return out


def build_coverage(qc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in sorted(qc["Variable"].dropna().unique()):
        sub = qc[(qc["Variable"] == var) & (qc["Year"].astype(str) != "")]
        available_years = sorted(pd.to_numeric(sub.loc[sub["Exists"] == True, "Year"], errors="coerce").dropna().astype(int).unique())
        missing = [y for y in YEARS if y not in available_years]
        level_counts = sub["QC_Level"].value_counts().to_dict() if not sub.empty else {}
        rows.append(
            {
                "Variable": var,
                "SpatialYearsAvailable": len(available_years),
                "MissingYears": ",".join(map(str, missing)),
                "AllYearsAvailable": len(missing) == 0,
                "QC_Pass_Years": int(level_counts.get("pass", 0)),
                "QC_Check_Years": int(level_counts.get("check", 0)),
                "QC_Fail_Years": int(level_counts.get("fail", 0)),
                "QC_Missing_Years": int(level_counts.get("missing", 0)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "01_raster_year_coverage_summary.csv", index=False, encoding="utf-8-sig")
    return out


def load_annual_controls() -> pd.DataFrame:
    if ANNUAL_CONTROLS_CSV.exists():
        df = pd.read_csv(ANNUAL_CONTROLS_CSV)
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        return df
    return pd.DataFrame({"Year": YEARS})


def variable_storage(var: str, controls: pd.DataFrame) -> Dict[str, object]:
    sp_years = years_available(var)
    in_table = var in controls.columns
    table_years: List[int] = []
    if in_table and "Year" in controls.columns:
        table_years = sorted(
            pd.to_numeric(controls.loc[controls[var].notna(), "Year"], errors="coerce").dropna().astype(int).unique().tolist()
        )
    storage = []
    if sp_years:
        storage.append("spatial_raster")
    if in_table:
        storage.append("annual_statistical_control")
    if not storage:
        storage.append("missing")

    if var.startswith("CO2") or var.startswith("ONI"):
        recommended = "Model 1 background control only; do not use as core land-surface mechanism variable."
    elif var == "SST":
        recommended = "Ocean background control. Spatial raster is ocean-facing; prefer annual control for land-pixel SHAP unless a justified teleconnection design is used."
    elif var in {"AOD", "Snow", "Cloud"}:
        recommended = "Background or sensitivity control; keep separate from the core albedo-Rn-energy pathway."
    elif var in {"AlbedoLoss", "Rn", "SM", "LH", "SH"}:
        recommended = "Core mechanism variable."
    elif var in {"SWdown", "LWdown", "SWabs_MODIS", "VPD"}:
        recommended = "Mechanism-chain/path or sensitivity variable."
    else:
        recommended = ""

    return {
        "StorageType": "+".join(storage),
        "SpatialYearsAvailable": len(sp_years),
        "SpatialMissingYears": ",".join(map(str, [y for y in YEARS if y not in sp_years])),
        "TableYearsAvailable": len(table_years),
        "TableMissingYears": ",".join(map(str, [y for y in YEARS if y not in table_years])) if in_table else "",
        "RecommendedTreatment": recommended,
    }


def build_shap_inventory() -> pd.DataFrame:
    controls = load_annual_controls()
    rows = []
    for model_name, info in MODEL_VARIABLES.items():
        for var in info["variables"]:
            rec = variable_storage(str(var), controls)
            rows.append(
                {
                    "Model": model_name,
                    "Purpose": info["purpose"],
                    "Variable": var,
                    "ExpectedRole": expected_for(str(var)).get("role", "annual_or_unknown"),
                    **rec,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "01_shap_xgboost_variable_storage_inventory.csv", index=False, encoding="utf-8-sig")
    return out


def check_albedo_loss_definition() -> pd.DataFrame:
    meta_path = META_DIR / "R3_AlbedoLoss_standardization.json"
    rows = []
    if not meta_path.exists():
        out = pd.DataFrame([{"Year": "", "QC_Level": "check", "Message": "R3_AlbedoLoss_standardization.json not found."}])
        out.to_csv(OUT_DIR / "01_albedoloss_definition_check.csv", index=False, encoding="utf-8-sig")
        return out
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mu = float(meta.get("mean", np.nan))
    sd = float(meta.get("std", np.nan))
    for year in YEARS:
        alpha_path = common_path("SurfaceAlbedo", year)
        loss_path = common_path("AlbedoLoss", year)
        if not alpha_path.exists() or not loss_path.exists() or not np.isfinite(mu) or not np.isfinite(sd) or sd == 0:
            rows.append({"Year": year, "QC_Level": "missing", "MaxAbsDifference": np.nan, "Message": "Input missing."})
            continue
        with rasterio.open(alpha_path) as src:
            alpha = src.read(1, masked=True).astype("float64").filled(np.nan)
            if src.nodata is not None:
                alpha[alpha == src.nodata] = np.nan
        with rasterio.open(loss_path) as src:
            loss = src.read(1, masked=True).astype("float64").filled(np.nan)
            if src.nodata is not None:
                loss[loss == src.nodata] = np.nan
        expected = -1.0 * ((alpha - mu) / sd)
        diff = np.abs(loss - expected)
        max_abs = float(np.nanmax(diff)) if np.isfinite(diff).any() else np.nan
        mean_abs = float(np.nanmean(diff)) if np.isfinite(diff).any() else np.nan
        rows.append(
            {
                "Year": year,
                "QC_Level": "pass" if np.isfinite(max_abs) and max_abs < 1e-4 else "check",
                "MaxAbsDifference": max_abs,
                "MeanAbsDifference": mean_abs,
                "Definition": "AlbedoLoss = -z(SurfaceAlbedo)",
                "MeanUsed": mu,
                "StdUsed": sd,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "01_albedoloss_definition_check.csv", index=False, encoding="utf-8-sig")
    return out


def write_report(qc: pd.DataFrame, coverage: pd.DataFrame, shap_inventory: pd.DataFrame, albedo_check: pd.DataFrame) -> None:
    problems = qc[qc["QC_Level"].isin(["check", "fail", "missing"])].copy()
    missing_core = shap_inventory[
        (shap_inventory["StorageType"] == "missing")
        & shap_inventory["Variable"].isin(["AlbedoLoss", "Rn", "SM", "LH", "SH", "T2M"])
    ]
    lines = [
        "Result 3 raster and SHAP-variable QC",
        "=" * 80,
        f"Root: {R3_ROOT}",
        f"Years: {YEARS[0]}-{YEARS[-1]}",
        "Phases: P1=2001-2014, P2=2015-2019, P3=2020-2024",
        "",
        f"Raster variables checked: {coverage['Variable'].nunique() if not coverage.empty else 0}",
        f"Raster QC rows with check/fail/missing: {len(problems)}",
        f"Core SHAP variables missing: {len(missing_core)}",
        "",
        "Important interpretation:",
        "- CO2 and ONI are background controls for Model 1, not core albedo mechanism variables.",
        "- If CO2 spatial rasters are available from CarbonTracker, they may be sampled at land pixels as a background field.",
        "- ONI is an ENSO index, not a true global spatial raster. Keep it as an annual control unless a separate Nino3.4 SST-anomaly spatial design is explicitly introduced.",
        "- AOD/Snow/SST spatial products should be official products and documented in the manifest before pixel-scale SHAP.",
        "",
        "Files written:",
        "- 01_raster_numeric_range_unit_sign_qc.csv",
        "- 01_raster_year_coverage_summary.csv",
        "- 01_shap_xgboost_variable_storage_inventory.csv",
        "- 01_albedoloss_definition_check.csv",
        "- 01_result3_raster_and_shap_qc.xlsx",
    ]
    (OUT_DIR / "01_result3_raster_and_shap_qc_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    qc = build_raster_qc()
    coverage = build_coverage(qc)
    shap_inventory = build_shap_inventory()
    albedo_check = check_albedo_loss_definition()

    with pd.ExcelWriter(OUT_DIR / "01_result3_raster_and_shap_qc.xlsx") as writer:
        qc.to_excel(writer, sheet_name="raster_qc", index=False)
        coverage.to_excel(writer, sheet_name="coverage", index=False)
        shap_inventory.to_excel(writer, sheet_name="shap_variable_inventory", index=False)
        albedo_check.to_excel(writer, sheet_name="albedoloss_check", index=False)

    write_report(qc, coverage, shap_inventory, albedo_check)
    print(f"[done] Workflow 1 QC outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

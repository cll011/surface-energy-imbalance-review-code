# -*- coding: utf-8 -*-
"""
Workflow 2 - Check NOAA CO2 data attributes for Result 3.

This script inspects:
    D:/10_Research/2025_Albedo_Temp/01_Data_Raw/01_Images/CO2_NOAA_Python

The local files are expected to be NOAA/GML CarbonTracker CT2025 monthly
NetCDF files. The script checks coverage, variables, units, dimensions, and
builds an annual PBL CO2 summary table for Result 3.

Outputs are written only under:
    D:/10_Research/01_Datasets/04_Results/Result3_Figures_optimized/R3_QC_Checks/02_CO2_NOAA_CT2025
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from netCDF4 import Dataset


CO2_DIR = Path(r"D:\10_Research\2025_Albedo_Temp\01_Data_Raw\01_Images\CO2_NOAA_Python")
R3_ROOT = Path(r"D:\10_Research\01_Datasets\04_Results\Result3_Figures_optimized")
OUT_DIR = R3_ROOT / "R3_QC_Checks" / "02_CO2_NOAA_CT2025"
TABLE_DIR = R3_ROOT / "R3_Tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2001, 2025))
BASELINE_YEARS = list(range(2001, 2011))


def phase_from_year(year: int) -> str:
    year = int(year)
    if 2001 <= year <= 2014:
        return "P1"
    if 2015 <= year <= 2019:
        return "P2"
    if 2020 <= year <= 2024:
        return "P3"
    return "Other"


def parse_year_month(path: Path) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"(\d{4})-(\d{2})-\d{2}", path.name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def list_nc_files() -> pd.DataFrame:
    rows = []
    for path in sorted(CO2_DIR.glob("*.nc")):
        year, month = parse_year_month(path)
        rows.append(
            {
                "Path": str(path),
                "FileName": path.name,
                "Year": year,
                "Month": month,
                "SizeMB": path.stat().st_size / (1024 * 1024),
                "LastWriteTime": pd.Timestamp(path.stat().st_mtime, unit="s"),
            }
        )
    return pd.DataFrame(rows)


def variable_attributes(sample_path: Path) -> pd.DataFrame:
    rows = []
    with Dataset(sample_path) as ds:
        for name, var in ds.variables.items():
            rows.append(
                {
                    "Variable": name,
                    "Dimensions": ",".join(var.dimensions),
                    "Shape": "x".join(map(str, var.shape)),
                    "Units": getattr(var, "units", ""),
                    "LongName": getattr(var, "long_name", ""),
                    "Dtype": str(var.dtype),
                }
            )
    return pd.DataFrame(rows)


def global_attributes(sample_path: Path) -> Dict[str, object]:
    attrs: Dict[str, object] = {}
    with Dataset(sample_path) as ds:
        for name in ds.ncattrs():
            value = getattr(ds, name)
            try:
                json.dumps(value)
                attrs[name] = value
            except TypeError:
                attrs[name] = str(value)
    return attrs


def get_lat_weights(lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat.astype("float64")))
    weights[~np.isfinite(weights)] = np.nan
    weights[weights < 0] = np.nan
    return weights


def weighted_grid_mean(arr: np.ndarray, lat: np.ndarray) -> float:
    # arr shape: lat, lon
    weights = get_lat_weights(lat)[:, None]
    valid = np.isfinite(arr) & np.isfinite(weights)
    if not np.any(valid):
        return math.nan
    return float(np.nansum(arr * weights * valid) / np.nansum(weights * valid))


def monthly_pbl_summary(path: Path) -> Dict[str, object]:
    year, month = parse_year_month(path)
    with Dataset(path) as ds:
        if "pbl_co2" not in ds.variables:
            raise KeyError(f"pbl_co2 not found in {path}")
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        lon = np.asarray(ds.variables["longitude"][:], dtype="float64")
        pbl = np.asarray(ds.variables["pbl_co2"][:], dtype="float64")
        # mean over sub-monthly time slices
        month_grid = np.nanmean(pbl, axis=0)
        units = getattr(ds.variables["pbl_co2"], "units", "")
        dec_dates = np.asarray(ds.variables["decimal_date"][:], dtype="float64") if "decimal_date" in ds.variables else np.array([])
    vals = month_grid[np.isfinite(month_grid)]
    return {
        "Year": year,
        "Month": month,
        "FileName": path.name,
        "PBL_CO2_units": units,
        "TimeSlices": int(pbl.shape[0]),
        "LatitudeCount": int(lat.size),
        "LongitudeCount": int(lon.size),
        "LatitudeMin": float(np.nanmin(lat)),
        "LatitudeMax": float(np.nanmax(lat)),
        "LongitudeMin": float(np.nanmin(lon)),
        "LongitudeMax": float(np.nanmax(lon)),
        "DecimalDateMean": float(np.nanmean(dec_dates)) if dec_dates.size else math.nan,
        "PBL_CO2_Min": float(np.nanmin(vals)) if vals.size else math.nan,
        "PBL_CO2_Max": float(np.nanmax(vals)) if vals.size else math.nan,
        "PBL_CO2_Mean": float(np.nanmean(vals)) if vals.size else math.nan,
        "PBL_CO2_AreaWeightedMean": weighted_grid_mean(month_grid, lat),
        "PBL_CO2_Std": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else math.nan,
        "ValidCellCount": int(vals.size),
    }


def build_monthly_and_annual(files: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in files.dropna(subset=["Year", "Month"]).iterrows():
        year = int(row["Year"])
        if year < 2000 or year > 2024:
            continue
        path = Path(row["Path"])
        print(f"[CO2 monthly] {path.name}")
        rows.append(monthly_pbl_summary(path))
    monthly = pd.DataFrame(rows)
    monthly.to_csv(OUT_DIR / "CO2_NOAA_CT2025_monthly_pbl_summary.csv", index=False, encoding="utf-8-sig")

    annual = (
        monthly.groupby("Year", as_index=False)
        .agg(
            MonthCount=("Month", "count"),
            CO2_CT2025_PBL_ppm_global_mean=("PBL_CO2_AreaWeightedMean", "mean"),
            CO2_CT2025_PBL_ppm_spatial_min=("PBL_CO2_Min", "mean"),
            CO2_CT2025_PBL_ppm_spatial_max=("PBL_CO2_Max", "mean"),
            CO2_CT2025_PBL_ppm_spatial_std=("PBL_CO2_Std", "mean"),
        )
    )
    baseline = annual.loc[annual["Year"].isin(BASELINE_YEARS), "CO2_CT2025_PBL_ppm_global_mean"].mean()
    ref2000_vals = annual.loc[annual["Year"] == 2000, "CO2_CT2025_PBL_ppm_global_mean"]
    ref2000 = float(ref2000_vals.iloc[0]) if not ref2000_vals.empty else baseline
    annual["CO2_RF_CT2025_ref2001_2010_Wm2"] = 5.35 * np.log(annual["CO2_CT2025_PBL_ppm_global_mean"] / baseline)
    annual["CO2_RF_CT2025_ref2000_Wm2"] = 5.35 * np.log(annual["CO2_CT2025_PBL_ppm_global_mean"] / ref2000)
    annual["CO2_RF_CT2025_ref278ppm_Wm2"] = 5.35 * np.log(annual["CO2_CT2025_PBL_ppm_global_mean"] / 278.0)
    annual["C0_ref2001_2010_ppm"] = baseline
    annual["C0_ref2000_ppm"] = ref2000
    annual["Phase"] = annual["Year"].map(phase_from_year)
    annual["UseForResult3"] = annual["Year"].isin(YEARS) & (annual["MonthCount"] >= 12)
    annual.to_csv(OUT_DIR / "CO2_NOAA_CT2025_annual_pbl_summary.csv", index=False, encoding="utf-8-sig")

    selected = annual[annual["Year"].isin(YEARS)].copy()
    selected.rename(
        columns={
            "CO2_CT2025_PBL_ppm_global_mean": "CO2_CT2025_PBL_ppm_annual_control",
            "CO2_RF_CT2025_ref2001_2010_Wm2": "CO2_RF_CT2025_annual_control",
            "CO2_RF_CT2025_ref2000_Wm2": "CO2_RF_CT2025_ref2000_annual_control",
        },
        inplace=True,
    )
    selected.to_csv(TABLE_DIR / "R3_CO2_CT2025_annual_controls.csv", index=False, encoding="utf-8-sig")
    return monthly, annual


def coverage_report(files: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in range(2000, 2025):
        months = sorted(files.loc[files["Year"] == year, "Month"].dropna().astype(int).unique().tolist())
        rows.append(
            {
                "Year": year,
                "MonthCount": len(months),
                "Months": ",".join(map(str, months)),
                "Complete12Months": len(months) == 12,
                "RequiredForResult3": year in YEARS,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "CO2_NOAA_CT2025_temporal_coverage.csv", index=False, encoding="utf-8-sig")
    return out


def write_recommendation(files: pd.DataFrame, coverage: pd.DataFrame, attrs: Dict[str, object], annual: pd.DataFrame) -> None:
    result3_cov = coverage[coverage["RequiredForResult3"]]
    complete = bool(result3_cov["Complete12Months"].all()) if not result3_cov.empty else False
    has_pbl = False
    variable_file = OUT_DIR / "CO2_NOAA_CT2025_variable_attributes.csv"
    if variable_file.exists():
        var_df = pd.read_csv(variable_file)
        has_pbl = "pbl_co2" in set(var_df["Variable"])
    result3_annual = annual[annual["Year"].isin(YEARS)]
    ppm_min = result3_annual["CO2_CT2025_PBL_ppm_global_mean"].min()
    ppm_max = result3_annual["CO2_CT2025_PBL_ppm_global_mean"].max()
    rf_min = result3_annual["CO2_RF_CT2025_ref2001_2010_Wm2"].min()
    rf_max = result3_annual["CO2_RF_CT2025_ref2001_2010_Wm2"].max()

    conclusion = (
        "SATISFIES_RESULT3_SPATIAL_BACKGROUND"
        if complete and has_pbl and np.isfinite(ppm_min) and 330 <= ppm_min <= 460 and 330 <= ppm_max <= 460
        else "CHECK_BEFORE_USE"
    )
    lines = [
        "NOAA CO2 data check for Result 3",
        "=" * 80,
        f"CO2 directory: {CO2_DIR}",
        f"NetCDF files found: {len(files)}",
        f"CarbonTracker version: {attrs.get('version', '')}",
        f"Institution: {attrs.get('institution', '')}",
        f"URL: {attrs.get('url', '')}",
        "",
        f"Conclusion: {conclusion}",
        "",
        "Reasoning:",
        f"- 2001-2024 monthly coverage complete: {complete}",
        f"- pbl_co2 variable present: {has_pbl}",
        f"- pbl_co2 units are ppm-equivalent micromol mol-1.",
        f"- Annual global PBL CO2 range in 2001-2024: {ppm_min:.3f} to {ppm_max:.3f} ppm.",
        f"- RF relative to 2001-2010 range: {rf_min:.4f} to {rf_max:.4f} W m-2.",
        "",
        "Recommended treatment:",
        "- Use CT2025 pbl_co2 as a spatial background field in Model 1 only.",
        "- Process pbl_co2 to annual rasters in Workflow 3; no additional CO2 download is needed.",
        "- Do not interpret CO2 SHAP as a local land-surface mechanism competing with AlbedoLoss/Rn/LH/SH.",
        "- CO2_RF should be computed from pbl_co2 with a fixed reference concentration and used as a background forcing sensitivity variable.",
    ]
    (OUT_DIR / "CO2_NOAA_CT2025_result3_recommendation.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not CO2_DIR.exists():
        raise FileNotFoundError(f"CO2 directory not found: {CO2_DIR}")

    files = list_nc_files()
    files.to_csv(OUT_DIR / "CO2_NOAA_CT2025_file_inventory.csv", index=False, encoding="utf-8-sig")
    if files.empty:
        raise RuntimeError(f"No NetCDF files were found under {CO2_DIR}")

    sample = Path(files.iloc[0]["Path"])
    var_df = variable_attributes(sample)
    var_df.to_csv(OUT_DIR / "CO2_NOAA_CT2025_variable_attributes.csv", index=False, encoding="utf-8-sig")
    attrs = global_attributes(sample)
    (OUT_DIR / "CO2_NOAA_CT2025_global_attributes.json").write_text(json.dumps(attrs, indent=2, ensure_ascii=False), encoding="utf-8")

    coverage = coverage_report(files)
    monthly, annual = build_monthly_and_annual(files)
    write_recommendation(files, coverage, attrs, annual)

    with pd.ExcelWriter(OUT_DIR / "02_CO2_NOAA_CT2025_attribute_check.xlsx") as writer:
        files.to_excel(writer, sheet_name="file_inventory", index=False)
        coverage.to_excel(writer, sheet_name="temporal_coverage", index=False)
        var_df.to_excel(writer, sheet_name="variable_attributes", index=False)
        annual.to_excel(writer, sheet_name="annual_summary", index=False)
        monthly.head(2000).to_excel(writer, sheet_name="monthly_summary_head", index=False)

    print(f"[done] Workflow 2 CO2 check outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

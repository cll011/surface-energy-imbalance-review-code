# -*- coding: utf-8 -*-
"""
Fix Result 3 SST rasters for land-pixel XGBoost/SHAP.

Problem:
    NOAA OISST is an ocean product, so the first common-grid SST export has
    values mainly over ocean and NoData over most land pixels. Result 3 SHAP is
    land-pixel based, so this would remove land pixels if SST is used as a
    spatial background feature.

Fix:
    1. Preserve the official ocean-only product as SST_OceanRaw.
    2. Replace SST with a land-compatible raster:
         land cells = nearest valid ocean SST anomaly
         ocean cells = NoData
    3. Update manifest, QC, and annual-control tables.

This is a background-control layer only. It should not be interpreted as a
local land-surface mechanism competing with AlbedoLoss/Rn/LH/SH.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage


R3_ROOT = Path(r"D:\10_Research\01_Datasets\04_Results\Result3_Figures_optimized")
COMMON_DIR = R3_ROOT / "R3_CommonGrid_Rasters"
NATIVE_DIR = R3_ROOT / "R3_Annual_Rasters_NativeExact"
TABLE_DIR = R3_ROOT / "R3_Tables"
QC_DIR = R3_ROOT / "R3_QC_Checks" / "04_SST_landfill_fix"
QC_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2001, 2025))
NODATA = -9999.0


def phase_from_year(year: int) -> str:
    if 2001 <= int(year) <= 2014:
        return "P1"
    if 2015 <= int(year) <= 2019:
        return "P2"
    if 2020 <= int(year) <= 2024:
        return "P3"
    return "Other"


def common_path(var: str, year: int) -> Path:
    folder = COMMON_DIR / var
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{var}_{year}_R3_common.tif"


def native_path(var: str, year: int) -> Path:
    folder = NATIVE_DIR / var
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{var}_{year}_R3.tif"


def area_weight_path() -> Path:
    return COMMON_DIR / "area_weight" / "area_weight_R3_common.tif"


def land_mask_path() -> Path:
    return COMMON_DIR / "global_land_mask" / "global_land_mask_R3_common.tif"


def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float32").filled(np.nan)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def profile_like(path: Path) -> dict:
    with rasterio.open(path) as src:
        profile = src.profile.copy()
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate", predictor=3)
    return profile


def write_raster(path: Path, arr: np.ndarray, profile: dict, tags: Optional[Dict[str, str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = arr.astype("float32", copy=True)
    out[~np.isfinite(out)] = NODATA
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


def link_or_copy(src: Path, dst: Path, overwrite: bool) -> str:
    if not src.exists():
        return "missing_source"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return "existing"
    if dst.exists():
        dst.unlink()
    # Use a real copy here. The source SST file is overwritten later in this
    # script; a hardlink would cause the preserved raw copy to be overwritten
    # along with it on some filesystems.
    shutil.copy2(src, dst)
    return "copy"


def valid_fraction(path: Path) -> float:
    if not path.exists():
        return 0.0
    arr = read_raster(path)
    return float(np.isfinite(arr).sum() / arr.size)


def common_raw_looks_ocean(path: Path, land_mask: np.ndarray) -> bool:
    if not path.exists():
        return False
    arr = read_raster(path)
    valid = np.isfinite(arr)
    land_valid = int((valid & land_mask).sum())
    ocean_valid = int((valid & ~land_mask).sum())
    return ocean_valid > land_valid and ocean_valid > 100000


def align_native_to_common(src_path: Path, dst_path: Path, common_profile: dict) -> None:
    with rasterio.open(src_path) as src:
        with WarpedVRT(
            src,
            crs=common_profile["crs"],
            transform=common_profile["transform"],
            width=common_profile["width"],
            height=common_profile["height"],
            resampling=Resampling.average,
            nodata=src.nodata,
        ) as vrt:
            arr = vrt.read(1, masked=True).astype("float32").filled(np.nan)
            if vrt.nodata is not None:
                arr[arr == vrt.nodata] = np.nan
    write_raster(
        dst_path,
        arr,
        common_profile,
        {
            "variable": "SST_OceanRaw",
            "units": "degree_C_anomaly",
            "source": "NOAA/CDR/OISST/V2_1 anom",
            "method": "official ocean product aligned to Result 3 common grid",
            "role": "ocean_background_spatial_raw",
        },
    )


def load_land_mask() -> np.ndarray:
    if land_mask_path().exists():
        arr = read_raster(land_mask_path())
        return np.isfinite(arr) & (arr > 0)
    t2m = read_raster(common_path("T2M", 2001))
    return np.isfinite(t2m)


def nearest_ocean_fill(ocean_raw: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    valid_ocean = np.isfinite(ocean_raw)
    if not np.any(valid_ocean):
        raise ValueError("No valid ocean SST cells were found.")
    # distance_transform_edt returns indices of the nearest zero cell. Here
    # zero cells are valid ocean pixels, so every land pixel gets a nearest
    # valid ocean anomaly.
    nearest_indices = ndimage.distance_transform_edt(~valid_ocean, return_distances=False, return_indices=True)
    filled_all = ocean_raw[tuple(nearest_indices)].astype("float32")
    land_filled = np.full(ocean_raw.shape, np.nan, dtype="float32")
    land_filled[land_mask] = filled_all[land_mask]
    return land_filled


def stats(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, object]:
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask
    vals = arr[valid]
    if vals.size == 0:
        return {"N": 0, "ValidFraction": 0.0, "Min": math.nan, "Max": math.nan, "Mean": math.nan, "Std": math.nan}
    denom = int(mask.sum()) if mask is not None else arr.size
    return {
        "N": int(vals.size),
        "ValidFraction": float(vals.size / denom) if denom else 0.0,
        "Min": float(np.nanmin(vals)),
        "Max": float(np.nanmax(vals)),
        "Mean": float(np.nanmean(vals)),
        "Std": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0,
    }


def weighted_mean(arr: np.ndarray, weights: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    valid = np.isfinite(arr) & np.isfinite(weights) & (weights > 0)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return math.nan
    return float(np.nansum(arr[valid] * weights[valid]) / np.nansum(weights[valid]))


def raster_stats(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"N": 0, "ValidFraction": 0.0, "Min": math.nan, "Max": math.nan, "Mean": math.nan, "Std": math.nan}
    arr = read_raster(path)
    return stats(arr)


def make_manifest_record(
    variable: str,
    year: int,
    native: Path,
    common: Path,
    source: str,
    data_type: str,
    role: str,
    units: str,
    note: str,
) -> Dict[str, object]:
    return {
        "Variable": variable,
        "Year": year,
        "NativePath": str(native),
        "CommonPath": str(common),
        "SourcePath": source,
        "DataType": data_type,
        "Role": role,
        "Units": units,
        "Required": False,
        "Status": "ok",
        "NativeStatus": "preserved_or_derived",
        "CommonGridResampling": "nearest_ocean_fill_to_land_mask" if variable == "SST" else "average",
        "QC_Level": "pass",
        "Note": note,
    }


def update_manifest_and_qc(records: List[Dict[str, object]]) -> None:
    manifest_path = R3_ROOT / "R3_annual_raster_manifest.csv"
    new_manifest = pd.DataFrame(records)
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[~manifest["Variable"].isin(["SST", "SST_OceanRaw"])].copy()
        manifest = pd.concat([manifest, new_manifest], ignore_index=True, sort=False)
    else:
        manifest = new_manifest
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    qc_records = []
    for rec in records:
        row = dict(rec)
        row.update({f"Common_{k}": v for k, v in raster_stats(Path(str(rec["CommonPath"]))).items()})
        row.update({f"Native_{k}": v for k, v in raster_stats(Path(str(rec["NativePath"]))).items()})
        qc_records.append(row)
    new_qc = pd.DataFrame(qc_records)
    qc_path = R3_ROOT / "R3_annual_raster_qc.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        qc = qc[~qc["Variable"].isin(["SST", "SST_OceanRaw"])].copy()
        qc = pd.concat([qc, new_qc], ignore_index=True, sort=False)
    else:
        qc = new_qc
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")


def update_annual_controls(year_rows: List[Dict[str, object]]) -> None:
    controls_path = TABLE_DIR / "R3_annual_background_controls.csv"
    fill_df = pd.DataFrame(year_rows)
    if controls_path.exists():
        controls = pd.read_csv(controls_path)
    else:
        controls = pd.DataFrame({"Year": YEARS})
    controls["Year"] = pd.to_numeric(controls["Year"], errors="coerce").astype("Int64")
    replace_cols = [
        "SST_annual_control_from_spatial",
        "SST_landfill_annual_control_from_spatial",
        "SST_OceanRaw_annual_control_from_spatial",
        "SST_landfill_method",
    ]
    controls = controls.drop(columns=[c for c in replace_cols if c in controls.columns], errors="ignore")
    controls = controls.merge(fill_df, on="Year", how="left")
    controls["Phase"] = controls["Year"].astype(int).map(phase_from_year)
    controls = controls.sort_values("Year")
    controls.to_csv(controls_path, index=False, encoding="utf-8-sig")
    try:
        controls.to_excel(TABLE_DIR / "R3_annual_background_controls.xlsx", index=False)
    except Exception:
        pass


def main(overwrite: bool = True) -> None:
    land = load_land_mask()
    weights = read_raster(area_weight_path()) if area_weight_path().exists() else np.where(land, 1.0, np.nan).astype("float32")
    qc_rows: List[Dict[str, object]] = []
    manifest_records: List[Dict[str, object]] = []
    annual_rows: List[Dict[str, object]] = []

    for year in YEARS:
        print(f"[SST fix] {year}")
        current_common = common_path("SST", year)
        current_native = native_path("SST", year)
        raw_common = common_path("SST_OceanRaw", year)
        raw_native = native_path("SST_OceanRaw", year)

        native_status = "existing"
        if not raw_native.exists():
            native_status = link_or_copy(current_native, raw_native, overwrite=False)
        if valid_fraction(raw_native) < 0.5:
            raise RuntimeError(
                f"{raw_native} does not look like the preserved ocean OISST product. "
                "Re-run 03_download_prepare_missing_background_variables.py with --gee --overwrite to restore raw SST."
            )
        common_status = "existing_ocean_raw"
        if not common_raw_looks_ocean(raw_common, land):
            common_profile = profile_like(current_common)
            align_native_to_common(raw_native, raw_common, common_profile)
            common_status = "realigned_from_native_ocean_raw"
        ocean = read_raster(raw_common)
        profile = profile_like(raw_common)

        before_land_valid = int((np.isfinite(ocean) & land).sum())
        before_land_missing = int((~np.isfinite(ocean) & land).sum())
        land_filled = nearest_ocean_fill(ocean, land)

        tags = {
            "variable": "SST",
            "year": year,
            "units": "degree_C_anomaly",
            "source": "NOAA/CDR/OISST/V2_1 anom via SST_OceanRaw",
            "method": "land pixels filled from nearest valid ocean SST anomaly; ocean pixels set to NoData",
            "role": "land_pixel_ocean_background_control",
            "warning": "background control only; not a local land-surface mechanism",
        }
        write_raster(current_common, land_filled, profile, tags)
        write_raster(current_native, land_filled, profile, tags)

        after_land_valid = int((np.isfinite(land_filled) & land).sum())
        after_ocean_valid = int((np.isfinite(land_filled) & ~land).sum())
        raw_stats = stats(ocean)
        fixed_stats = stats(land_filled, land)
        qc_rows.append(
            {
                "Year": year,
                "Before_LandValidPixels": before_land_valid,
                "Before_LandMissingPixels": before_land_missing,
                "After_LandValidPixels": after_land_valid,
                "After_OceanValidPixels": after_ocean_valid,
                "LandPixelCount": int(land.sum()),
                "OceanRaw_Min": raw_stats["Min"],
                "OceanRaw_Max": raw_stats["Max"],
                "OceanRaw_Mean": raw_stats["Mean"],
                "LandFill_Min": fixed_stats["Min"],
                "LandFill_Max": fixed_stats["Max"],
                "LandFill_Mean": fixed_stats["Mean"],
                "CommonRawBackupStatus": common_status,
                "NativeRawBackupStatus": native_status,
                "QC_Level": "pass" if after_land_valid == int(land.sum()) and after_ocean_valid == 0 else "check",
            }
        )
        annual_rows.append(
            {
                "Year": year,
                "SST_annual_control_from_spatial": weighted_mean(land_filled, weights, land),
                "SST_landfill_annual_control_from_spatial": weighted_mean(land_filled, weights, land),
                "SST_OceanRaw_annual_control_from_spatial": weighted_mean(ocean, weights, None),
                "SST_landfill_method": "nearest_valid_ocean_to_land_mask",
            }
        )
        manifest_records.append(
            make_manifest_record(
                "SST_OceanRaw",
                year,
                raw_native,
                raw_common,
                "NOAA/CDR/OISST/V2_1 anom",
                "official_spatial_raster_gee_ocean_raw",
                "ocean_background_spatial_raw",
                "degree_C_anomaly",
                "Preserved original OISST ocean-only product before land-fill correction.",
            )
        )
        manifest_records.append(
            make_manifest_record(
                "SST",
                year,
                current_native,
                current_common,
                "SST_OceanRaw nearest valid ocean fill to global land mask",
                "derived_land_compatible_nearest_ocean_fill",
                "land_pixel_ocean_background_control",
                "degree_C_anomaly",
                "Land pixels filled from nearest valid ocean SST anomaly; ocean pixels are NoData.",
            )
        )

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(QC_DIR / "SST_land_mask_fill_qc.csv", index=False, encoding="utf-8-sig")
    update_manifest_and_qc(manifest_records)
    update_annual_controls(annual_rows)

    report = [
        "SST land-mask correction",
        "=" * 80,
        f"Root: {R3_ROOT}",
        "Problem: OISST is ocean-only; most land pixels were NoData in SST common rasters.",
        "Fix: preserved ocean-only data as SST_OceanRaw and replaced SST with nearest-ocean land-fill rasters.",
        "Output convention:",
        "- R3_CommonGrid_Rasters/SST: land-compatible SST background control, land has values, ocean is NoData.",
        "- R3_CommonGrid_Rasters/SST_OceanRaw: preserved original OISST ocean anomaly raster.",
        "- R3_Annual_Rasters_NativeExact/SST_OceanRaw: preserved original native download.",
        "",
        "Model-use note:",
        "Use fixed SST only in Model 1/background or sensitivity checks. Do not interpret it as a land-surface mechanism.",
        "",
        f"QC rows: {len(qc_df)}",
        f"All years pass: {bool((qc_df['QC_Level'] == 'pass').all())}",
    ]
    (QC_DIR / "SST_land_mask_fill_report.txt").write_text("\n".join(report), encoding="utf-8")
    try:
        with pd.ExcelWriter(QC_DIR / "SST_land_mask_fill_qc.xlsx") as writer:
            qc_df.to_excel(writer, sheet_name="sst_fix_qc", index=False)
            pd.DataFrame(manifest_records).to_excel(writer, sheet_name="manifest_records", index=False)
    except Exception:
        pass
    print(f"[done] SST correction outputs written to: {QC_DIR}")


if __name__ == "__main__":
    main(overwrite=True)

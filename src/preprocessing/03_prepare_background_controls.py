# -*- coding: utf-8 -*-
"""
Workflow 3 - Download or prepare missing background/control variables.

This script prepares spatial/statistical controls needed by Result 3:
1. CO2:
   - Prefer local NOAA/GML CarbonTracker CT2025 files under
     D:/10_Research/2025_Albedo_Temp/01_Data_Raw/01_Images/CO2_NOAA_Python.
   - Process pbl_co2 to annual native and common-grid rasters.
   - Also write CO2 RF annual rasters relative to the 2001-2010 mean.
   - No fake global-constant CO2 rasters are created.
2. ONI:
   - Download official CPC ONI ASCII and write annual statistical controls.
   - ONI is not a spatial raster by definition; no fake global-constant ONI
     rasters are created.
3. AOD, Snow, SST:
   - If missing, use Earth Engine official products to create annual rasters.

Outputs are written only under:
    D:/10_Research/01_Datasets/04_Results/Result3_Figures_optimized

Typical run after Earth Engine authentication:
    python 03_download_prepare_missing_background_variables.py --gee --gee-project 300988058916
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from netCDF4 import Dataset
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT


YEARS = list(range(2001, 2025))
BASELINE_YEARS = list(range(2001, 2011))
NODATA = -9999.0

R3_ROOT = Path(r"D:\10_Research\01_Datasets\04_Results\Result3_Figures_optimized")
COMMON_DIR = R3_ROOT / "R3_CommonGrid_Rasters"
NATIVE_DIR = R3_ROOT / "R3_Annual_Rasters_NativeExact"
TABLE_DIR = R3_ROOT / "R3_Tables"
META_DIR = R3_ROOT / "R3_Metadata"
QC_DIR = R3_ROOT / "R3_QC_Checks"
CO2_DIR = Path(r"D:\10_Research\2025_Albedo_Temp\01_Data_Raw\01_Images\CO2_NOAA_Python")

for folder in [COMMON_DIR, NATIVE_DIR, TABLE_DIR, META_DIR, QC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TEMPLATE_RASTER = COMMON_DIR / "T2M" / "T2M_2001_R3_common.tif"
LAND_MASK_RASTER = COMMON_DIR / "global_land_mask" / "global_land_mask_R3_common.tif"

OFFICIAL_URLS = {
    "NOAA_CO2_GLOBAL_ANNUAL": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.txt",
    "CPC_ONI_ASCII": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
}

GEE_PRODUCTS = {
    "AOD": {
        "collection": "MODIS/061/MOD08_M3",
        "bands": ["AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean", "Aerosol_Optical_Depth_Land_Ocean_Mean_Mean"],
        "scale": 111320,
        "kind": "aod",
        "units": "unitless",
        "role": "aerosol_background_spatial",
        "mask_to_land": True,
        "model_note": "Use as Model 1 background or sensitivity control, not as core mechanism.",
    },
    "Snow": {
        "collection": "MODIS/061/MOD10A1",
        "bands": ["NDSI_Snow_Cover"],
        "scale": 25000,
        "kind": "snow",
        "units": "fraction",
        "role": "snow_background_spatial",
        "mask_to_land": True,
        "model_note": "Use as Model 1 background or sensitivity control.",
    },
    "SST": {
        "collection": "NOAA/CDR/OISST/V2_1",
        "bands": ["anom"],
        "scale": 27830,
        "kind": "sst",
        "units": "degree_C_anomaly",
        "role": "ocean_background_spatial",
        "mask_to_land": False,
        "model_note": "Prefer annual control for land-pixel SHAP; spatial raster is ocean-facing.",
    },
}

MANAGED_RASTER_VARIABLES = ["CO2_CT2025_PBL", "CO2_RF_CT2025_PBL", "AOD", "Snow", "SST"]


def phase_from_year(year: int) -> str:
    year = int(year)
    if 2001 <= year <= 2014:
        return "P1"
    if 2015 <= year <= 2019:
        return "P2"
    if 2020 <= year <= 2024:
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


def template_profile() -> dict:
    if not TEMPLATE_RASTER.exists():
        raise FileNotFoundError(f"Template common-grid raster not found: {TEMPLATE_RASTER}")
    with rasterio.open(TEMPLATE_RASTER) as src:
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


def land_mask(profile: dict) -> np.ndarray:
    if LAND_MASK_RASTER.exists():
        arr = read_raster(LAND_MASK_RASTER)
        return np.isfinite(arr) & (arr > 0)
    t2m = read_raster(TEMPLATE_RASTER)
    return np.isfinite(t2m)


def clean_array(arr: np.ndarray, kind: str) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    arr[(arr <= -3e38) | (arr >= 3e38) | (arr == -9999) | (arr == -32768)] = np.nan
    if kind == "co2_ppm":
        arr[(arr < 300) | (arr > 600)] = np.nan
    elif kind == "co2_rf":
        arr[(arr < -5) | (arr > 10)] = np.nan
    elif kind == "aod":
        arr[(arr < 0) | (arr > 5)] = np.nan
    elif kind == "snow":
        arr[(arr < 0) | (arr > 1)] = np.nan
    elif kind == "sst":
        arr[(arr < -30) | (arr > 30)] = np.nan
    return arr


def align_to_common(
    src_path: Path,
    dst_path: Path,
    profile: dict,
    kind: str,
    mask: Optional[np.ndarray],
    overwrite: bool,
    tags: Dict[str, str],
) -> bool:
    if dst_path.exists() and not overwrite:
        return True
    if not src_path.exists():
        return False
    with rasterio.open(src_path) as src:
        with WarpedVRT(
            src,
            crs=profile["crs"],
            transform=profile["transform"],
            width=profile["width"],
            height=profile["height"],
            resampling=Resampling.average,
            nodata=src.nodata,
        ) as vrt:
            arr = vrt.read(1, masked=True).astype("float32").filled(np.nan)
            if vrt.nodata is not None:
                arr[arr == vrt.nodata] = np.nan
    arr = clean_array(arr, kind)
    if mask is not None:
        arr[~mask] = np.nan
    write_raster(dst_path, arr, profile, tags)
    return True


def raster_stats(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"N": 0, "ValidFraction": 0.0, "Min": math.nan, "Max": math.nan, "Mean": math.nan, "Std": math.nan}
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float64").filled(np.nan)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return {"N": 0, "ValidFraction": 0.0, "Min": math.nan, "Max": math.nan, "Mean": math.nan, "Std": math.nan}
    return {
        "N": int(vals.size),
        "ValidFraction": float(vals.size / arr.size),
        "Min": float(np.nanmin(vals)),
        "Max": float(np.nanmax(vals)),
        "Mean": float(np.nanmean(vals)),
        "Std": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0,
    }


def area_weighted_mean(path: Path, mask: Optional[np.ndarray] = None) -> float:
    if not path.exists():
        return math.nan
    arr = read_raster(path)
    with rasterio.open(path) as src:
        transform = src.transform
        rows = np.arange(src.height, dtype="float64")
        lats = transform.f + (rows + 0.5) * transform.e
    weights = np.cos(np.deg2rad(lats))[:, None]
    valid = np.isfinite(arr) & np.isfinite(weights)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return math.nan
    return float(np.nansum(arr * weights * valid) / np.nansum(weights * valid))


def parse_year_month(path: Path) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"(\d{4})-(\d{2})-\d{2}", path.name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def co2_files_by_year() -> Dict[int, List[Path]]:
    files: Dict[int, List[Path]] = {y: [] for y in YEARS}
    for path in sorted(CO2_DIR.glob("*.nc")):
        year, _month = parse_year_month(path)
        if year in files:
            files[year].append(path)
    return files


def sort_ct_grid(grid: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    lon_order = np.argsort(lon180)
    lat_order = np.argsort(lat)[::-1]
    return grid[np.ix_(lat_order, lon_order)], lat[lat_order], lon180[lon_order]


def ct_native_profile(lat: np.ndarray, lon180: np.ndarray) -> dict:
    dx = float(np.nanmedian(np.diff(np.sort(lon180))))
    lat_sorted = np.sort(lat)
    dy = float(np.nanmedian(np.diff(lat_sorted)))
    west = float(np.nanmin(lon180) - dx / 2.0)
    north = float(np.nanmax(lat) + dy / 2.0)
    return {
        "driver": "GTiff",
        "height": int(lat.size),
        "width": int(lon180.size),
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(west, north, dx, dy),
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
    }


def monthly_ct_grid(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(path) as ds:
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        lon = np.asarray(ds.variables["longitude"][:], dtype="float64")
        pbl = np.asarray(ds.variables["pbl_co2"][:], dtype="float64")
    grid = np.nanmean(pbl, axis=0)
    return sort_ct_grid(grid, lat, lon)


def prepare_co2_ct2025(profile: dict, mask: np.ndarray, overwrite: bool) -> pd.DataFrame:
    files = co2_files_by_year()
    missing_years = [year for year, paths in files.items() if len(paths) < 12]
    if missing_years:
        raise RuntimeError(f"CarbonTracker monthly files are incomplete for years: {missing_years}")

    annual_grids: Dict[int, np.ndarray] = {}
    lat_sorted = None
    lon_sorted = None
    for year in YEARS:
        n_ppm = native_path("CO2_CT2025_PBL", year)
        c_ppm = common_path("CO2_CT2025_PBL", year)
        if n_ppm.exists() and c_ppm.exists() and not overwrite:
            continue
        grids = []
        for path in files[year]:
            grid, lat_sorted, lon_sorted = monthly_ct_grid(path)
            grids.append(grid)
        annual = np.nanmean(np.stack(grids, axis=0), axis=0).astype("float32")
        annual_grids[year] = annual
        native_profile = ct_native_profile(lat_sorted, lon_sorted)
        write_raster(
            n_ppm,
            clean_array(annual, "co2_ppm"),
            native_profile,
            {
                "variable": "CO2_CT2025_PBL",
                "year": year,
                "units": "micromol mol-1 / ppm",
                "source": "NOAA/GML CarbonTracker CT2025 pbl_co2",
            },
        )
        align_to_common(
            n_ppm,
            c_ppm,
            profile,
            "co2_ppm",
            mask,
            overwrite=True,
            tags={
                "variable": "CO2_CT2025_PBL",
                "year": str(year),
                "units": "micromol mol-1 / ppm",
                "source": "NOAA/GML CarbonTracker CT2025 pbl_co2",
                "role": "co2_spatial_background",
            },
        )

    # Read native annual grids if this is a non-overwrite rerun.
    if len(annual_grids) < len(YEARS):
        for year in YEARS:
            n_ppm = native_path("CO2_CT2025_PBL", year)
            if n_ppm.exists() and year not in annual_grids:
                annual_grids[year] = read_raster(n_ppm)

    annual_means = []
    for year in YEARS:
        c_ppm = common_path("CO2_CT2025_PBL", year)
        annual_means.append({"Year": year, "CO2_CT2025_PBL_ppm_annual_control": area_weighted_mean(c_ppm, mask)})
    mean_df = pd.DataFrame(annual_means)
    c0 = float(mean_df.loc[mean_df["Year"].isin(BASELINE_YEARS), "CO2_CT2025_PBL_ppm_annual_control"].mean())

    records = []
    for year in YEARS:
        n_rf = native_path("CO2_RF_CT2025_PBL", year)
        c_rf = common_path("CO2_RF_CT2025_PBL", year)
        if overwrite or not n_rf.exists() or not c_rf.exists():
            n_ppm = native_path("CO2_CT2025_PBL", year)
            with rasterio.open(n_ppm) as src:
                ppm = src.read(1, masked=True).astype("float32").filled(np.nan)
                native_profile = src.profile.copy()
                if src.nodata is not None:
                    ppm[ppm == src.nodata] = np.nan
            rf = 5.35 * np.log(ppm / c0)
            rf[~np.isfinite(ppm)] = np.nan
            write_raster(
                n_rf,
                clean_array(rf, "co2_rf"),
                native_profile,
                {
                    "variable": "CO2_RF_CT2025_PBL",
                    "year": year,
                    "units": "W m-2",
                    "source": "derived from NOAA/GML CarbonTracker CT2025 pbl_co2",
                    "reference": f"global land mean {BASELINE_YEARS[0]}-{BASELINE_YEARS[-1]} C0={c0:.6f} ppm",
                },
            )
            align_to_common(
                n_rf,
                c_rf,
                profile,
                "co2_rf",
                mask,
                overwrite=True,
                tags={
                    "variable": "CO2_RF_CT2025_PBL",
                    "year": str(year),
                    "units": "W m-2",
                    "source": "derived from NOAA/GML CarbonTracker CT2025 pbl_co2",
                    "role": "co2_rf_spatial_background",
                    "reference": f"global land mean {BASELINE_YEARS[0]}-{BASELINE_YEARS[-1]} C0={c0:.6f} ppm",
                },
            )
        records.extend(
            [
                make_manifest_record("CO2_CT2025_PBL", year, native_path("CO2_CT2025_PBL", year), common_path("CO2_CT2025_PBL", year), "NOAA/GML CarbonTracker CT2025 pbl_co2", "official_spatial_raster_local", "co2_spatial_background", "ppm"),
                make_manifest_record("CO2_RF_CT2025_PBL", year, native_path("CO2_RF_CT2025_PBL", year), common_path("CO2_RF_CT2025_PBL", year), "NOAA/GML CarbonTracker CT2025 pbl_co2; RF=5.35*ln(C/C0)", "derived_spatial_raster", "co2_rf_spatial_background", "W m-2"),
            ]
        )

    rf_means = []
    for year in YEARS:
        rf_means.append({"Year": year, "CO2_RF_CT2025_annual_control": area_weighted_mean(common_path("CO2_RF_CT2025_PBL", year), mask)})
    out = mean_df.merge(pd.DataFrame(rf_means), on="Year", how="left")
    out["Phase"] = out["Year"].map(phase_from_year)
    out.to_csv(TABLE_DIR / "R3_CO2_CT2025_spatial_annual_controls.csv", index=False, encoding="utf-8-sig")
    update_manifest_and_qc(records)
    return out


def download_file(url: str, path: Path, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, path)
    return path


def prepare_noaa_global_co2_table(overwrite: bool) -> pd.DataFrame:
    raw = download_file(OFFICIAL_URLS["NOAA_CO2_GLOBAL_ANNUAL"], TABLE_DIR / "noaa_co2_global_annual.txt", overwrite)
    rows = []
    for line in raw.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            rows.append({"Year": int(parts[0]), "CO2_Global_ppm_annual_control": float(parts[1])})
    df = pd.DataFrame(rows)
    df = df[df["Year"].isin(YEARS)].copy()
    c0 = df.loc[df["Year"].isin(BASELINE_YEARS), "CO2_Global_ppm_annual_control"].mean()
    ref2000_series = pd.DataFrame(rows).loc[pd.DataFrame(rows)["Year"] == 2000, "CO2_Global_ppm_annual_control"]
    ref2000 = float(ref2000_series.iloc[0]) if not ref2000_series.empty else float(c0)
    df["CO2_RF_annual_control"] = 5.35 * np.log(df["CO2_Global_ppm_annual_control"] / c0)
    df["CO2_RF_ref2000_annual_control"] = 5.35 * np.log(df["CO2_Global_ppm_annual_control"] / ref2000)
    df["CO2_RF_ref278ppm_annual_control"] = 5.35 * np.log(df["CO2_Global_ppm_annual_control"] / 278.0)
    df["Phase"] = df["Year"].map(phase_from_year)
    df.to_csv(TABLE_DIR / "R3_CO2_NOAA_global_annual_controls.csv", index=False, encoding="utf-8-sig")
    return df


def prepare_oni(overwrite: bool) -> pd.DataFrame:
    raw = download_file(OFFICIAL_URLS["CPC_ONI_ASCII"], TABLE_DIR / "noaa_cpc_oni_ascii.txt", overwrite)
    rows = []
    for line in raw.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[1].isdigit():
            try:
                rows.append({"Season": parts[0], "Year": int(parts[1]), "ONI_annual_control": float(parts[3])})
            except ValueError:
                pass
    monthly_like = pd.DataFrame(rows)
    if monthly_like.empty:
        raise RuntimeError("Failed to parse CPC ONI ASCII data.")
    annual = monthly_like.groupby("Year", as_index=False)["ONI_annual_control"].mean()
    annual["ONI_Lag1_annual_control"] = annual["ONI_annual_control"].shift(1)
    annual = annual[annual["Year"].isin(YEARS)].copy()
    annual["Phase"] = annual["Year"].map(phase_from_year)
    annual.to_csv(TABLE_DIR / "R3_ONI_CPC_annual_controls.csv", index=False, encoding="utf-8-sig")
    return annual


def init_ee(project: Optional[str]):
    import ee

    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    return ee


def choose_band(ee, collection: str, candidates: List[str], year: int) -> str:
    img = ee.ImageCollection(collection).filterDate(f"{year}-01-01", f"{year + 1}-01-01").first()
    names = img.bandNames().getInfo()
    for band in candidates:
        if band in names:
            return band
    raise ValueError(f"No candidate band found for {collection} {year}; available={names[:30]}")


def gee_annual_image(ee, var: str, cfg: Dict[str, object], year: int):
    band = choose_band(ee, str(cfg["collection"]), list(cfg["bands"]), year)
    col = ee.ImageCollection(str(cfg["collection"])).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    if cfg["kind"] == "aod":
        def clean(img):
            b = img.select(band)
            return b.updateMask(b.gte(0).And(b.lt(5000))).multiply(0.001)

        return col.map(clean).mean().rename(var), band
    if cfg["kind"] == "snow":
        def clean(img):
            b = img.select(band)
            return b.updateMask(b.gte(0).And(b.lte(100))).divide(100.0)

        return col.map(clean).mean().rename(var), band
    if cfg["kind"] == "sst":
        return col.select(band).mean().multiply(0.01).rename(var), band
    return col.select(band).mean().rename(var), band


def download_gee_image(ee, image, profile: dict, target: Path, scale: int) -> None:
    bounds = rasterio.transform.array_bounds(profile["height"], profile["width"], profile["transform"])
    region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]], proj="EPSG:4326", geodesic=False)
    url = image.getDownloadURL({"scale": scale, "crs": "EPSG:4326", "region": region, "format": "GEO_TIFF"})
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)


def spatial_complete(var: str) -> bool:
    return all(common_path(var, year).exists() for year in YEARS)


def prepare_gee_spatial(profile: dict, mask: np.ndarray, project: Optional[str], overwrite: bool) -> pd.DataFrame:
    ee = init_ee(project)
    records = []
    annual_controls = pd.DataFrame({"Year": YEARS})
    for var, cfg in GEE_PRODUCTS.items():
        if spatial_complete(var) and not overwrite:
            print(f"[skip] {var} already has 24 common-grid rasters.")
        means = []
        for year in YEARS:
            n_path = native_path(var, year)
            c_path = common_path(var, year)
            band = ""
            if overwrite or not n_path.exists() or not c_path.exists():
                print(f"[GEE] {var} {year}")
                image, band = gee_annual_image(ee, var, cfg, year)
                download_gee_image(ee, image, profile, n_path, int(cfg["scale"]))
                align_to_common(
                    n_path,
                    c_path,
                    profile,
                    str(cfg["kind"]),
                    mask if bool(cfg["mask_to_land"]) else None,
                    overwrite=True,
                    tags={
                        "variable": var,
                        "year": str(year),
                        "units": str(cfg["units"]),
                        "source": str(cfg["collection"]),
                        "source_band": band,
                        "role": str(cfg["role"]),
                    },
                )
            mean_mask = mask if bool(cfg["mask_to_land"]) else None
            means.append({"Year": year, f"{var}_annual_control_from_spatial": area_weighted_mean(c_path, mean_mask)})
            records.append(
                make_manifest_record(
                    var,
                    year,
                    n_path,
                    c_path,
                    str(cfg["collection"]),
                    "official_spatial_raster_gee",
                    str(cfg["role"]),
                    str(cfg["units"]),
                    status="ok" if c_path.exists() else "failed",
                    resampling="average",
                    note=str(cfg["model_note"]),
                )
            )
        annual_controls = annual_controls.merge(pd.DataFrame(means), on="Year", how="left")
    annual_controls["Phase"] = annual_controls["Year"].map(phase_from_year)
    annual_controls.to_csv(TABLE_DIR / "R3_AOD_Snow_SST_spatial_annual_controls.csv", index=False, encoding="utf-8-sig")
    update_manifest_and_qc(records)
    return annual_controls


def make_manifest_record(
    var: str,
    year: int,
    native: Path,
    common: Path,
    source: str,
    dtype: str,
    role: str,
    units: str,
    status: str = "ok",
    resampling: str = "average",
    note: str = "",
) -> Dict[str, object]:
    return {
        "Variable": var,
        "Year": year,
        "NativePath": str(native),
        "CommonPath": str(common),
        "SourcePath": source,
        "DataType": dtype,
        "Role": role,
        "Units": units,
        "Required": False,
        "Status": status,
        "NativeStatus": "local_existing_or_downloaded",
        "CommonGridResampling": resampling,
        "QC_Level": "pass" if status == "ok" else "check",
        "Note": note,
    }


def update_manifest_and_qc(records: List[Dict[str, object]]) -> None:
    if not records:
        return
    new_manifest = pd.DataFrame(records)
    manifest_path = R3_ROOT / "R3_annual_raster_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[~manifest["Variable"].isin(new_manifest["Variable"].unique())].copy()
        manifest = pd.concat([manifest, new_manifest], ignore_index=True, sort=False)
    else:
        manifest = new_manifest
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    qc_records = []
    for rec in records:
        common_stats = raster_stats(Path(str(rec["CommonPath"])))
        native_stats = raster_stats(Path(str(rec["NativePath"])))
        qc = dict(rec)
        qc.update({f"Common_{k}": v for k, v in common_stats.items()})
        qc.update({f"Native_{k}": v for k, v in native_stats.items()})
        qc_records.append(qc)
    new_qc = pd.DataFrame(qc_records)
    qc_path = R3_ROOT / "R3_annual_raster_qc.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        qc = qc[~qc["Variable"].isin(new_qc["Variable"].unique())].copy()
        qc = pd.concat([qc, new_qc], ignore_index=True, sort=False)
    else:
        qc = new_qc
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")


def merge_annual_controls(tables: List[pd.DataFrame]) -> pd.DataFrame:
    out_path = TABLE_DIR / "R3_annual_background_controls.csv"
    if out_path.exists():
        out = pd.read_csv(out_path)
        if "Year" not in out.columns:
            out = pd.DataFrame({"Year": YEARS})
    else:
        out = pd.DataFrame({"Year": YEARS})
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out = out[out["Year"].isin(YEARS)].copy()

    for table in tables:
        if table is None or table.empty or "Year" not in table.columns:
            continue
        table = table.copy()
        table["Year"] = pd.to_numeric(table["Year"], errors="coerce").astype("Int64")
        table = table[table["Year"].isin(YEARS)]
        table = table.drop(columns=["Phase"], errors="ignore")
        replace_cols = [c for c in table.columns if c != "Year" and c in out.columns]
        out = out.drop(columns=replace_cols, errors="ignore")
        out = out.merge(table, on="Year", how="left")
    out["Year"] = out["Year"].astype(int)
    out["Phase"] = out["Year"].map(phase_from_year)
    out = out.sort_values("Year")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    try:
        out.to_excel(TABLE_DIR / "R3_annual_background_controls.xlsx", index=False)
    except Exception:
        pass
    return out


def write_non_spatial_treatment() -> None:
    rows = [
        {
            "Variable": "ONI",
            "SpatialRasterCreated": False,
            "Reason": "ONI is an ENSO index over the Nino3.4 region, not a global spatial field.",
            "Treatment": "Use CPC ONI as annual statistical control in Model 1. Do not create global constant rasters.",
        },
        {
            "Variable": "CO2",
            "SpatialRasterCreated": True,
            "Reason": "NOAA/GML CarbonTracker CT2025 provides gridded pbl_co2, so spatial annual rasters are created.",
            "Treatment": "Use CO2_CT2025_PBL or CO2_RF_CT2025_PBL as Model 1 background controls only.",
        },
        {
            "Variable": "SST",
            "SpatialRasterCreated": True,
            "Reason": "NOAA OISST provides ocean SST anomaly rasters.",
            "Treatment": "Use as ocean/background control; avoid treating ocean-only SST as a land-surface mechanism variable.",
        },
    ]
    pd.DataFrame(rows).to_csv(TABLE_DIR / "R3_non_spatial_variable_treatment.csv", index=False, encoding="utf-8-sig")


def write_summary(
    co2_spatial: pd.DataFrame,
    co2_global: pd.DataFrame,
    oni: pd.DataFrame,
    gee_controls: Optional[pd.DataFrame],
    merged: pd.DataFrame,
) -> None:
    lines = [
        "Workflow 3 background/control variable preparation",
        "=" * 80,
        f"Root: {R3_ROOT}",
        "Phases: P1=2001-2014, P2=2015-2019, P3=2020-2024",
        "",
        "Outputs:",
        "- CO2_CT2025_PBL annual native/common rasters",
        "- CO2_RF_CT2025_PBL annual native/common rasters",
        "- CPC ONI annual control table",
        "- AOD/Snow/SST annual rasters if --gee is used",
        "- Updated R3_annual_background_controls.csv",
        "- Updated R3_annual_raster_manifest.csv and R3_annual_raster_qc.csv",
        "",
        f"CO2 spatial years: {len(co2_spatial) if co2_spatial is not None else 0}",
        f"NOAA global CO2 table years: {len(co2_global) if co2_global is not None else 0}",
        f"ONI years: {len(oni) if oni is not None else 0}",
        f"GEE annual-control rows: {len(gee_controls) if gee_controls is not None else 0}",
        f"Merged annual-control columns: {len(merged.columns) if merged is not None else 0}",
        "",
        "Interpretation note:",
        "CO2 and ONI remain background controls. CO2 can now be sampled spatially from CarbonTracker; ONI remains a scalar climate-mode control.",
    ]
    (QC_DIR / "03_background_variable_preparation_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gee", action="store_true", help="Download/process AOD, Snow, and SST from Google Earth Engine if missing.")
    parser.add_argument("--gee-project", default=None, help="Earth Engine project id, e.g. 300988058916.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated rasters/tables.")
    parser.add_argument("--skip-co2-spatial", action="store_true")
    parser.add_argument("--skip-oni", action="store_true")
    args = parser.parse_args()

    profile = template_profile()
    mask = land_mask(profile)

    co2_spatial = pd.DataFrame()
    if not args.skip_co2_spatial:
        print("[CO2] preparing CarbonTracker CT2025 spatial annual rasters")
        co2_spatial = prepare_co2_ct2025(profile, mask, args.overwrite)

    print("[CO2] preparing NOAA global annual control table")
    co2_global = prepare_noaa_global_co2_table(args.overwrite)

    oni = pd.DataFrame()
    if not args.skip_oni:
        print("[ONI] preparing CPC annual control table")
        oni = prepare_oni(args.overwrite)

    gee_controls = pd.DataFrame()
    if args.gee:
        print("[GEE] preparing AOD/Snow/SST annual spatial rasters")
        gee_controls = prepare_gee_spatial(profile, mask, args.gee_project, args.overwrite)
    else:
        print("[GEE] skipped. Use --gee --gee-project <project_id> to download AOD/Snow/SST.")

    merged = merge_annual_controls([co2_global, co2_spatial, oni, gee_controls])
    write_non_spatial_treatment()
    write_summary(co2_spatial, co2_global, oni, gee_controls, merged)
    print(f"[done] Workflow 3 outputs written under: {R3_ROOT}")


if __name__ == "__main__":
    main()

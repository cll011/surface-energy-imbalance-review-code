# -*- coding: utf-8 -*-
"""
01_prepare_result3_optimized_annual_rasters.py

Prepare Result 3 annual rasters and annual controls.

Main revision in this version
-----------------------------
1. All common-grid modelling rasters are masked by the unified global land
   vector:
       D:\10_Research\2025_Albedo_Temp\01_Data_Raw\03_Shapefiles
       \ne_10m_admin_0_countries.shp

2. T2M input is changed to:
       D:\10_Research\01_Datasets\01_DataRaw\ERA5\Annual_Tif
       \T2M_ERA5_{year}.tif

3. R3_annual_background_controls.csv is always generated. The script first
   tries official downloaded tables, then falls back to local CO2 / sup_csv
   files when available.

4. Manifest Status is standardized:
       Status = "ok" for successfully produced local, GEE and derived rasters.
   DataType records whether the raster is local, GEE or derived.

5. Model 1-3 variable groups are written to:
       R3_Metadata\R3_model_variable_groups.json
       R3_Metadata\R3_model_variable_groups.csv

Scientific focus
----------------
The main Result 3 comparison is designed to highlight:
    AlbedoLoss contribution in P3 > P1,
    and its increase relative to other land-surface variables.

Therefore:
- AlbedoLoss is the core variable.
- Rn, SM, LH and SH are the core land-surface mechanism variables.
- SWdown, LWdown and VPD are exported but are placed in sensitivity or
  pathway-diagnostic groups, not in the main SHAP competition group.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT


# =============================================================================
# 1. Global settings
# =============================================================================

YEARS = list(range(2001, 2025))
NODATA = -9999.0

# Output folders
OUT_DIR = Path(r"D:\10_Research\01_Datasets\04_Results\Result3_Figures_optimized")
NATIVE_OUT = OUT_DIR / "R3_Annual_Rasters_NativeExact"
COMMON_OUT = OUT_DIR / "R3_CommonGrid_Rasters"
TABLE_OUT = OUT_DIR / "R3_Tables"
META_OUT = OUT_DIR / "R3_Metadata"

# Unified global land vector mask
GLOBAL_LAND_VECTOR = Path(
    r"D:\10_Research\2025_Albedo_Temp\01_Data_Raw\03_Shapefiles"
    r"\ne_10m_admin_0_countries.shp"
)

# T2M and ERA5 annual raster directory
ERA5_ANNUAL_DIR = Path(r"D:\10_Research\01_Datasets\01_DataRaw\ERA5\Annual_Tif")
TEMPLATE_RASTER = ERA5_ANNUAL_DIR / "T2M_ERA5_2001.tif"

# Local albedo products
MODIS_ALBEDO_DIR = Path(
    r"D:\10_Research\01_Datasets\02_DataProcess\04_MODIS_BlueSky_Albedo_Global\Annual"
)
GLASS_ALBEDO_DIR = Path(
    r"D:\10_Research\01_Datasets\02_DataProcess\03_SurfaceAlbedo_GLASS"
    r"\blueSky_annual"
)

# Local annual-control fallback folders
LOCAL_CO2_TABLE = Path(r"D:\10_Research\01_Datasets\01_DataRaw\NOAA_CO2\00_CO2_RF_for_SHAP_Annual.csv")
SUP_CSV_DIR_CANDIDATES = [
    Path(r"D:\10_Research\01_Datasets\01_DataRaw\sup_csv"),
    Path(r"D:\10_Research\01_Datasets\04_Results\sup_csv"),
    Path(r"D:\10_Research\2025_Albedo_Temp\02_Data_Process\sup_csv"),
]

# Rasterization option for land mask.
# False uses cell center rule, which is more conservative for global land mask.
LAND_MASK_ALL_TOUCHED = False

# Phase definition for the current manuscript mainline.
# P1: relatively stable stage; P2: transition; P3: warming-surge stage.
PHASE_RULE = {
    "P1": (2001, 2014),
    "P2": (2015, 2019),
    "P3": (2020, 2024),
}


# =============================================================================
# 2. Spatial source definitions
# =============================================================================

@dataclass(frozen=True)
class SpatialSource:
    name: str
    source_dir: Path
    pattern: str
    cleaner: str
    role: str
    units: str
    required: bool = True
    resampling: Resampling = Resampling.average


LOCAL_SPATIAL = [
    # Target
    SpatialSource(
        "T2M",
        ERA5_ANNUAL_DIR,
        "T2M_ERA5_{year}.tif",
        "t2m",
        "target",
        "degree_C",
        required=True,
        resampling=Resampling.average,
    ),

    # Core surface-albedo signal
    SpatialSource(
        "SurfaceAlbedo",
        MODIS_ALBEDO_DIR,
        "MODIS_BlueSky_shortwave_annual_{year}.tif",
        "albedo",
        "surface_albedo_signal_core",
        "fraction",
        required=True,
        resampling=Resampling.average,
    ),

    # Sensitivity surface-albedo product
    SpatialSource(
        "SurfaceAlbedo_GLASS",
        GLASS_ALBEDO_DIR,
        "GLASS_BlueSky_shortwave_annual_{year}_native.tif",
        "albedo",
        "surface_albedo_sensitivity",
        "fraction",
        required=False,
        resampling=Resampling.average,
    ),

    # Core land-surface mechanism variables for the main SHAP comparison
    SpatialSource("Rn", ERA5_ANNUAL_DIR, "Rn_ERA5_{year}.tif", "generic", "core_land_mechanism", "W m-2", required=True),
    SpatialSource("SM", ERA5_ANNUAL_DIR, "SM_ERA5_{year}.tif", "fraction_like", "core_land_mechanism", "m3 m-3", required=True),
    SpatialSource("LH", ERA5_ANNUAL_DIR, "LH_ERA5_{year}.tif", "generic", "core_land_mechanism", "W m-2", required=True),
    SpatialSource("SH", ERA5_ANNUAL_DIR, "SH_ERA5_{year}.tif", "generic", "core_land_mechanism", "W m-2", required=True),

    # Exported but not recommended as main competing variables for SHAP contribution ranking.
    # They can absorb albedo-related variance because they are upstream forcing
    # or downstream atmospheric response.
    SpatialSource("SWdown", ERA5_ANNUAL_DIR, "SWdown_ERA5_{year}.tif", "positive", "sensitivity_radiative_input", "W m-2", required=False),
    SpatialSource("LWdown", ERA5_ANNUAL_DIR, "LWdown_ERA5_{year}.tif", "positive", "background_longwave", "W m-2", required=False),
    SpatialSource("VPD", ERA5_ANNUAL_DIR, "VPD_ERA5_{year}.tif", "positive", "downstream_diagnostic", "kPa", required=False),

    # Controls
    SpatialSource("Cloud", ERA5_ANNUAL_DIR, "Cloud_ERA5_{year}.tif", "fraction", "cloud_control", "fraction", required=False),
]

OFFICIAL_TABLE_URLS = {
    "noaa_co2_global_annual": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.txt",
    "noaa_co2_global_monthly": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.txt",
    "noaa_cpc_oni_ascii": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
}

GEE_PRODUCTS = {
    "AOD": {
        "collection": "MODIS/061/MOD08_M3",
        "bands": [
            "AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean",
            "Aerosol_Optical_Depth_Land_Ocean_Mean_Mean",
        ],
        "scale": 111320,
        "kind": "aod",
        "role": "background_control_optional",
        "units": "unitless",
    },
    "Snow": {
        "collection": "MODIS/061/MOD10A1",
        "bands": ["NDSI_Snow_Cover"],
        "scale": 111320,
        "kind": "snow",
        "role": "snow_control_optional",
        "units": "fraction",
    },
    "SST": {
        "collection": "NOAA/CDR/OISST/V2_1",
        "bands": ["anom"],
        "scale": 111320,
        "kind": "sst",
        "role": "ocean_background_optional",
        "units": "degree_C_anomaly",
    },
}


# =============================================================================
# 3. Model variable grouping metadata
# =============================================================================

MODEL_VARIABLE_GROUPS = {
    "Model_1_background_stripping": {
        "purpose": (
            "Estimate large-scale background warming and produce T2M residuals. "
            "Do not include land-surface mechanism variables here."
        ),
        "target": "T2M",
        "recommended_predictors": [
            "CO2_RF_annual_control",
            "CO2_RF_ref2000_annual_control",
            "ONI_annual_control",
            "ONI_Lag1_annual_control",
            "SST_anomaly_annual_control",
            "AOD_annual_control",
            "Snow_annual_control",
            "Cloud",
        ],
        "excluded_from_main_background_model": [
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
        ],
        "reason": (
            "SWdown, LWdown and VPD are not used in the main background model "
            "because they can absorb albedo-related variance or represent downstream "
            "land-atmosphere responses."
        ),
    },
    "Model_2_main_land_surface_SHAP": {
        "purpose": (
            "Explain the background-controlled T2M residual using land-surface variables. "
            "This is the main contribution comparison used to test whether AlbedoLoss "
            "increases from P1 to P3 more than other land variables."
        ),
        "target": "T2M_residual_from_Model_1",
        "recommended_predictors": [
            "AlbedoLoss",
            "Rn",
            "SM",
            "LH",
            "SH",
        ],
        "main_comparison_variable": "AlbedoLoss",
        "not_in_main_comparison_but_exported": [
            "SWdown",
            "LWdown",
            "VPD",
            "SWabs_MODIS",
            "Cloud",
            "Snow",
            "AOD",
        ],
        "reason": (
            "AlbedoLoss is compared against direct land-energy partitioning variables "
            "rather than against upstream radiative forcing or downstream atmospheric "
            "response variables. This avoids mechanically suppressing the apparent "
            "albedo contribution."
        ),
    },
    "Model_2b_sensitivity_SHAP": {
        "purpose": (
            "Robustness check. Add radiative and downstream diagnostics to test whether "
            "the increasing AlbedoLoss contribution remains visible under stricter controls."
        ),
        "target": "T2M_residual_from_Model_1",
        "recommended_predictors": [
            "AlbedoLoss",
            "Rn",
            "SM",
            "LH",
            "SH",
            "SWabs_MODIS",
            "VPD",
            "Cloud",
            "Snow",
        ],
        "interpretation": (
            "This model is not the primary ranking model. If AlbedoLoss contribution "
            "declines after adding SWabs or VPD, interpret it as pathway mediation, "
            "not failure of the albedo mechanism."
        ),
    },
    "Model_3_path_mechanism": {
        "purpose": (
            "Represent the physical chain rather than a pure variable-importance race."
        ),
        "recommended_path_equations": [
            "SWabs_MODIS ~ AlbedoLoss + SWdown + Cloud + Snow",
            "Rn ~ AlbedoLoss + SWabs_MODIS + LWdown + Cloud + Snow",
            "LH ~ Rn + SM",
            "SH ~ Rn + SM",
            "VPD ~ SH + LH + SM",
            "T2M ~ AlbedoLoss + Rn + LH + SH + SM",
        ],
        "interpretation": (
            "SWdown and LWdown are exogenous radiative inputs; VPD is a downstream "
            "diagnostic. They should not be used to judge whether albedo contribution "
            "is weak or strong in the main SHAP comparison."
        ),
    },
}


# =============================================================================
# 4. Basic path and profile utilities
# =============================================================================

def ensure_dirs() -> None:
    for folder in [OUT_DIR, NATIVE_OUT, COMMON_OUT, TABLE_OUT, META_OUT]:
        folder.mkdir(parents=True, exist_ok=True)


def native_path(var: str, year: int) -> Path:
    folder = NATIVE_OUT / var
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{var}_{year}_R3.tif"


def common_path(var: str, year: Optional[int] = None) -> Path:
    folder = COMMON_OUT / var
    folder.mkdir(parents=True, exist_ok=True)
    if year is None:
        return folder / f"{var}_R3_common.tif"
    return folder / f"{var}_{year}_R3_common.tif"


def profile_template() -> dict:
    if not TEMPLATE_RASTER.exists():
        raise FileNotFoundError(f"Missing template raster: {TEMPLATE_RASTER}")

    with rasterio.open(TEMPLATE_RASTER) as src:
        profile = src.profile.copy()

    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate", predictor=3)
    return profile


def phase_from_year(year: int) -> str:
    y = int(year)
    for phase, (start, end) in PHASE_RULE.items():
        if start <= y <= end:
            return phase
    return "Other"


def read_float(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float32").filled(np.nan)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def clean_array(arr: np.ndarray, cleaner: str) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    arr[(arr <= -3.0e38) | (arr >= 3.0e38) | (arr == -9999) | (arr == -32768)] = np.nan

    if cleaner == "t2m":
        med = np.nanmedian(arr)
        if np.isfinite(med) and med > 100:
            arr = arr - 273.15
        arr[(arr < -90) | (arr > 80)] = np.nan

    elif cleaner == "albedo":
        arr[(arr < 0) | (arr > 1)] = np.nan

    elif cleaner == "fraction":
        med = np.nanmedian(arr)
        if np.isfinite(med) and 1.5 < med <= 100:
            arr = arr / 100.0
        arr[(arr < 0) | (arr > 1)] = np.nan

    elif cleaner == "fraction_like":
        arr[(arr < -0.05) | (arr > 1.5)] = np.nan

    elif cleaner == "positive":
        arr[(arr < 0) | (arr > 1.0e8)] = np.nan

    else:
        arr[(arr < -1.0e10) | (arr > 1.0e10)] = np.nan

    return arr.astype("float32", copy=False)


def write_array(
    path: Path,
    arr: np.ndarray,
    profile: dict,
    tags: Optional[Dict[str, str]] = None,
    mask: Optional[np.ndarray] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = arr.astype("float32", copy=True)

    if mask is not None:
        out[~mask] = np.nan

    out[~np.isfinite(out)] = NODATA

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
        if tags:
            dst.update_tags(**tags)


def same_grid(path: Path, profile: dict) -> bool:
    with rasterio.open(path) as src:
        return (
            src.width == profile["width"]
            and src.height == profile["height"]
            and src.crs == profile["crs"]
            and src.transform.almost_equals(profile["transform"])
        )


def link_or_copy(src: Path, dst: Path, overwrite: bool) -> str:
    """
    Native files are retained only for provenance.
    All modelling rasters are common-grid rasters masked by the global land vector.
    """
    if dst.exists() and not overwrite:
        return "existing"
    if dst.exists():
        try:
            dst.unlink()
        except PermissionError:
            return "locked_existing_not_replaced"
    try:
        os.link(src, dst)
        return "hardlink_native_exact"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_native_exact"


def align_to_common(
    src: Path,
    dst: Path,
    cleaner: str,
    profile: dict,
    resampling: Resampling,
    overwrite: bool,
    tags: Dict[str, str],
    mask: Optional[np.ndarray],
) -> bool:
    if dst.exists() and not overwrite:
        return True
    if not src.exists():
        return False

    if same_grid(src, profile):
        arr = read_float(src)
    else:
        with rasterio.open(src) as src_ds:
            with WarpedVRT(
                src_ds,
                crs=profile["crs"],
                transform=profile["transform"],
                width=profile["width"],
                height=profile["height"],
                resampling=resampling,
                nodata=src_ds.nodata,
            ) as vrt:
                arr = vrt.read(1, masked=True).astype("float32").filled(np.nan)
                if vrt.nodata is not None:
                    arr[arr == vrt.nodata] = np.nan

    arr = clean_array(arr, cleaner)
    write_array(dst, arr, profile, tags=tags, mask=mask)
    return True


# =============================================================================
# 5. Unified global land vector mask
# =============================================================================

def global_land_mask(profile: dict, overwrite: bool = False) -> np.ndarray:
    """
    Rasterize the global land vector to the common grid.

    The mask is used for all common-grid modelling rasters and derived rasters.
    """
    mask_path = common_path("global_land_mask", None)

    if mask_path.exists() and not overwrite:
        arr = read_float(mask_path)
        return np.isfinite(arr) & (arr > 0.5)

    if not GLOBAL_LAND_VECTOR.exists():
        raise FileNotFoundError(f"Missing global land vector: {GLOBAL_LAND_VECTOR}")

    try:
        import geopandas as gpd
    except Exception as exc:
        raise ImportError(
            "geopandas is required to rasterize the global land vector. "
            "Install with: conda install -c conda-forge geopandas"
        ) from exc

    gdf = gpd.read_file(GLOBAL_LAND_VECTOR)
    if gdf.empty:
        raise ValueError(f"Global land vector is empty: {GLOBAL_LAND_VECTOR}")

    if gdf.crs is None:
        # Natural Earth is usually EPSG:4326. If CRS is absent, assume EPSG:4326.
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs(profile["crs"])

    geoms = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if len(geoms) == 0:
        raise ValueError(f"No valid geometries in global land vector: {GLOBAL_LAND_VECTOR}")

    mask = rasterize(
        geoms,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=LAND_MASK_ALL_TOUCHED,
    ).astype(bool)

    write_array(
        mask_path,
        np.where(mask, 1.0, np.nan).astype("float32"),
        profile,
        tags={"variable": "global_land_mask", "source": str(GLOBAL_LAND_VECTOR)},
        mask=None,
    )

    return mask


def write_area_weight(profile: dict, mask: np.ndarray, overwrite: bool) -> dict:
    path = common_path("area_weight", None)
    if overwrite or not path.exists():
        rows = np.arange(profile["height"], dtype="float64")
        transform = profile["transform"]
        lats = transform.f + (rows + 0.5) * transform.e
        weights = np.cos(np.deg2rad(lats))[:, None].astype("float32")
        write_array(path, np.where(mask, weights, np.nan), profile, {"variable": "area_weight"}, mask=mask)

    return {
        "Variable": "area_weight",
        "Year": "",
        "NativePath": "",
        "CommonPath": str(path),
        "SourcePath": "template latitude; global land vector mask",
        "DataType": "derived_common_grid_raster",
        "Role": "area_weight",
        "Units": "cos(latitude)",
        "Required": False,
        "Status": "ok",
        "NativeStatus": "",
        "CommonGridResampling": "",
    }


# =============================================================================
# 6. Official and local annual controls
# =============================================================================

def download_official_tables(overwrite: bool) -> None:
    for name, url in OFFICIAL_TABLE_URLS.items():
        target = TABLE_OUT / f"{name}.txt"
        if target.exists() and not overwrite:
            continue
        print(f"[download table] {url}")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:
            print(f"[warning] failed to download {url}: {exc}")


def parse_noaa_co2_downloaded() -> Optional[pd.DataFrame]:
    path = TABLE_OUT / "noaa_co2_global_annual.txt"
    if not path.exists():
        return None

    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            rows.append({"Year": int(parts[0]), "CO2_Global_ppm_annual_control": float(parts[1])})

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    ref = df.loc[df["Year"].between(2001, 2010), "CO2_Global_ppm_annual_control"].mean()
    if not np.isfinite(ref):
        ref = df.loc[df["Year"] == 2001, "CO2_Global_ppm_annual_control"].mean()

    ref2000_vals = df.loc[df["Year"] == 2000, "CO2_Global_ppm_annual_control"]
    ref2000 = ref2000_vals.iloc[0] if len(ref2000_vals) else ref

    df["CO2_RF_annual_control"] = 5.35 * np.log(df["CO2_Global_ppm_annual_control"] / ref)
    df["CO2_RF_ref2000_annual_control"] = 5.35 * np.log(df["CO2_Global_ppm_annual_control"] / ref2000)

    return df[df["Year"].isin(YEARS)].copy()


def parse_local_co2_table() -> Optional[pd.DataFrame]:
    path = LOCAL_CO2_TABLE
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "Year" not in df.columns:
        return None

    df["Year"] = df["Year"].astype(float).round().astype(int)
    df = df[df["Year"].isin(YEARS)].copy()

    rename = {
        "CO2_Global_ppm": "CO2_Global_ppm_annual_control",
        "CO2_RF_ref2001_2010_Wm2": "CO2_RF_annual_control",
        "CO2_RF_ref2000_Wm2": "CO2_RF_ref2000_annual_control",
        "CO2_RF_ref278ppm_Wm2": "CO2_RF_ref278ppm_annual_control",
    }

    keep = ["Year"]
    ren = {}
    for old, new in rename.items():
        if old in df.columns:
            keep.append(old)
            ren[old] = new

    if len(keep) == 1:
        return None

    return df[keep].rename(columns=ren)


def parse_cpc_oni_downloaded() -> Optional[pd.DataFrame]:
    path = TABLE_OUT / "noaa_cpc_oni_ascii.txt"
    if not path.exists():
        return None

    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[1].isdigit():
            rows.append({"Year": int(parts[1]), "ONI_annual_control": float(parts[3])})

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    annual = df.groupby("Year", as_index=False)["ONI_annual_control"].mean()
    annual["ONI_Lag1_annual_control"] = annual["ONI_annual_control"].shift(1)

    return annual[annual["Year"].isin(YEARS)].copy()


def first_existing_sup_csv(filename: str) -> Optional[Path]:
    for folder in SUP_CSV_DIR_CANDIDATES:
        path = folder / filename
        if path.exists():
            return path
    return None


def annual_controls() -> pd.DataFrame:
    out = pd.DataFrame({"Year": YEARS})
    meta = []

    # CO2: official downloaded first; local NOAA_CO2 processed file as fallback.
    co2 = parse_noaa_co2_downloaded()
    co2_source = str(TABLE_OUT / "noaa_co2_global_annual.txt")
    co2_status = "ok_official_download"

    if co2 is None:
        co2 = parse_local_co2_table()
        co2_source = str(LOCAL_CO2_TABLE)
        co2_status = "ok_local_fallback" if co2 is not None else "missing"

    if co2 is not None:
        out = out.merge(co2, on="Year", how="left")
    meta.append({"Source": "CO2", "Path": co2_source, "Status": co2_status})

    # ONI: official downloaded first; local cross-year ONI fallback later.
    oni = parse_cpc_oni_downloaded()
    if oni is not None:
        out = out.merge(oni, on="Year", how="left")
        meta.append({"Source": "ONI_official", "Path": str(TABLE_OUT / "noaa_cpc_oni_ascii.txt"), "Status": "ok_official_download"})
    else:
        meta.append({"Source": "ONI_official", "Path": str(TABLE_OUT / "noaa_cpc_oni_ascii.txt"), "Status": "missing"})

    # Local supplementary controls
    local_tables = {
        "ONI_cross_year": (
            "R11_oni_processed_annual.csv",
            {
                "ONI_CrossYear_OctSep": "ONI_CrossYear_annual_control",
                "ONI": "ONI_local_annual_control",
                "ONI_Annual": "ONI_local_annual_control",
                "ONI_mean": "ONI_local_annual_control",
            },
        ),
        "SST_SeaIce": (
            "R12_Global_SST_SeaIce_OISST_Annual_2001_2024.csv",
            {
                "Global_SST_anomaly_OISST": "SST_anomaly_annual_control",
                "Global_SST": "SST_anomaly_annual_control",
                "SST": "SST_anomaly_annual_control",
                "SST_anomaly": "SST_anomaly_annual_control",
                "Sea_Ice_OISST": "SeaIce_annual_control",
                "SeaIce": "SeaIce_annual_control",
            },
        ),
        "AOD": (
            "R12_Aerosol_AOD_MOD08_Terra_Annual_2001_2024.csv",
            {
                "Aerosol_AOD_MOD08_Terra_550": "AOD_annual_control",
                "AOD": "AOD_annual_control",
            },
        ),
        "Snow": (
            "R12_Snow_Cover_MODIS_Annual_2001_2024.csv",
            {
                "Snow_Cover_MODIS": "Snow_annual_control",
                "Snow": "Snow_annual_control",
            },
        ),
    }

    for name, (filename, rename) in local_tables.items():
        path = first_existing_sup_csv(filename)
        if path is None:
            meta.append({"Source": name, "Path": filename, "Status": "missing"})
            continue

        df = pd.read_csv(path)
        if "Year" not in df.columns:
            meta.append({"Source": name, "Path": str(path), "Status": "missing_year_column"})
            continue

        df["Year"] = df["Year"].astype(float).round().astype(int)
        keep = ["Year"]
        ren = {}

        for old, new in rename.items():
            if old in df.columns and new not in out.columns:
                keep.append(old)
                ren[old] = new

        if len(keep) > 1:
            out = out.merge(df[keep].rename(columns=ren), on="Year", how="left")
            meta.append({"Source": name, "Path": str(path), "Status": "ok"})
        else:
            meta.append({"Source": name, "Path": str(path), "Status": "no_matching_columns"})

    out["Phase"] = out["Year"].map(phase_from_year)

    controls_path = TABLE_OUT / "R3_annual_background_controls.csv"
    sources_path = TABLE_OUT / "R3_annual_background_controls_sources.csv"

    out.to_csv(controls_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(meta).to_csv(sources_path, index=False, encoding="utf-8-sig")

    return out


# =============================================================================
# 7. Local raster preparation
# =============================================================================

def prepare_local(profile: dict, overwrite: bool, skip_glass: bool, mask: np.ndarray) -> List[dict]:
    records = []

    for src in LOCAL_SPATIAL:
        if skip_glass and src.name == "SurfaceAlbedo_GLASS":
            continue

        for year in YEARS:
            source = src.source_dir / src.pattern.format(year=year)
            n_path = native_path(src.name, year)
            c_path = common_path(src.name, year)

            if source.exists():
                native_status = link_or_copy(source, n_path, overwrite)
                common_ok = align_to_common(
                    source,
                    c_path,
                    src.cleaner,
                    profile,
                    src.resampling,
                    overwrite,
                    {
                        "variable": src.name,
                        "year": str(year),
                        "source": str(source),
                        "role": src.role,
                        "units": src.units,
                        "mask": str(GLOBAL_LAND_VECTOR),
                    },
                    mask=mask,
                )
                status = "ok" if common_ok else "common_failed"
            else:
                native_status = "missing_source"
                status = "missing_required" if src.required else "missing_optional"

            records.append({
                "Variable": src.name,
                "Year": year,
                "NativePath": str(n_path),
                "CommonPath": str(c_path),
                "SourcePath": str(source),
                "DataType": "local_spatial_raster",
                "Role": src.role,
                "Units": src.units,
                "Required": src.required,
                "Status": status,
                "NativeStatus": native_status,
                "CommonGridResampling": src.resampling.name,
            })

    return records


# =============================================================================
# 8. Optional Earth Engine spatial products
# =============================================================================

def init_ee(project: Optional[str]):
    import ee
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    return ee


def choose_band(ee, collection: str, candidates: List[str], year: int) -> str:
    names = (
        ee.ImageCollection(collection)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .first()
        .bandNames()
        .getInfo()
    )
    for band in candidates:
        if band in names:
            return band
    raise ValueError(f"No candidate band found in {collection}; candidates={candidates}; available={names[:20]}")


def gee_image(ee, var: str, cfg: dict, year: int):
    band = choose_band(ee, cfg["collection"], cfg["bands"], year)
    collection = ee.ImageCollection(cfg["collection"]).filterDate(f"{year}-01-01", f"{year + 1}-01-01")

    if cfg["kind"] == "snow":
        def clean(img):
            b = img.select(band)
            return b.updateMask(b.gte(0).And(b.lte(100))).divide(100.0)
        return collection.map(clean).mean().rename(var), band

    if cfg["kind"] == "aod":
        def clean(img):
            b = img.select(band)
            return b.updateMask(b.gte(0)).multiply(0.001)
        return collection.map(clean).mean().rename(var), band

    if cfg["kind"] == "sst":
        return collection.select(band).mean().multiply(0.01).rename(var), band

    return collection.select(band).mean().rename(var), band


def download_gee(ee, image, profile: dict, target: Path, scale: int) -> None:
    bounds = rasterio.transform.array_bounds(profile["height"], profile["width"], profile["transform"])
    region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]], proj="EPSG:4326", geodesic=False)
    url = image.getDownloadURL({"scale": scale, "crs": "EPSG:4326", "region": region, "format": "GEO_TIFF"})
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)


def prepare_gee(profile: dict, overwrite: bool, project: Optional[str], mask: np.ndarray) -> List[dict]:
    records = []
    ee = init_ee(project)

    for var, cfg in GEE_PRODUCTS.items():
        cleaner = "fraction" if var == "Snow" else ("positive" if var == "AOD" else "generic")

        for year in YEARS:
            n_path = native_path(var, year)
            c_path = common_path(var, year)
            band = ""

            try:
                if overwrite or not n_path.exists():
                    image, band = gee_image(ee, var, cfg, year)
                    print(f"[gee] {var} {year} {cfg['collection']}:{band}")
                    download_gee(ee, image, profile, n_path, int(cfg["scale"]))

                ok = align_to_common(
                    n_path,
                    c_path,
                    cleaner,
                    profile,
                    Resampling.average,
                    overwrite,
                    {
                        "variable": var,
                        "year": str(year),
                        "source": cfg["collection"],
                        "source_band": band,
                        "role": cfg["role"],
                        "units": cfg["units"],
                        "mask": str(GLOBAL_LAND_VECTOR),
                    },
                    mask=mask,
                )

                status = "ok" if ok else "gee_failed"

            except Exception as exc:
                print(f"[warning] GEE failed for {var} {year}: {exc}")
                status = "gee_failed"

            records.append({
                "Variable": var,
                "Year": year,
                "NativePath": str(n_path),
                "CommonPath": str(c_path),
                "SourcePath": cfg["collection"],
                "DataType": "official_spatial_raster_gee",
                "Role": cfg["role"],
                "Units": cfg["units"],
                "Required": False,
                "Status": status,
                "NativeStatus": "downloaded_or_existing" if n_path.exists() else "missing",
                "CommonGridResampling": "average",
            })

    return records


# =============================================================================
# 9. Derived variables
# =============================================================================

def derive_albedo_loss(profile: dict, mask: np.ndarray, overwrite: bool) -> List[dict]:
    vals = []

    for year in YEARS:
        p = common_path("SurfaceAlbedo", year)
        if p.exists():
            arr = read_float(p)
            vals.append(arr[mask & np.isfinite(arr)])

    vals = [v for v in vals if v.size]
    if not vals:
        return []

    all_vals = np.concatenate(vals)
    mu = float(np.nanmean(all_vals))
    sd = float(np.nanstd(all_vals, ddof=1))

    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Cannot standardize AlbedoLoss because SurfaceAlbedo std is invalid.")

    (META_OUT / "R3_AlbedoLoss_standardization.json").write_text(
        json.dumps(
            {
                "mean": mu,
                "std": sd,
                "definition": "AlbedoLoss = -z(SurfaceAlbedo)",
                "mask": str(GLOBAL_LAND_VECTOR),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    records = []

    for year in YEARS:
        src = common_path("SurfaceAlbedo", year)
        c_path = common_path("AlbedoLoss", year)
        n_path = native_path("AlbedoLoss", year)

        if src.exists() and (overwrite or not c_path.exists()):
            arr = read_float(src)
            loss = -1.0 * ((arr - mu) / sd)
            loss[~mask | ~np.isfinite(arr)] = np.nan

            write_array(c_path, loss, profile, {"variable": "AlbedoLoss", "year": str(year), "mask": str(GLOBAL_LAND_VECTOR)}, mask=mask)
            write_array(n_path, loss, profile, {"variable": "AlbedoLoss", "year": str(year), "mask": str(GLOBAL_LAND_VECTOR)}, mask=mask)

        records.append({
            "Variable": "AlbedoLoss",
            "Year": year,
            "NativePath": str(n_path),
            "CommonPath": str(c_path),
            "SourcePath": str(src),
            "DataType": "derived_common_grid_raster",
            "Role": "surface_albedo_signal_core",
            "Units": "standardized",
            "Required": True,
            "Status": "ok" if c_path.exists() else "missing_required",
            "NativeStatus": "derived",
            "CommonGridResampling": "",
        })

    return records


def derive_swabs(profile: dict, overwrite: bool, mask: np.ndarray) -> List[dict]:
    """
    Derive absorbed shortwave using MODIS surface albedo and SWdown:
        SWabs_MODIS = (1 - SurfaceAlbedo) * SWdown

    SWabs_MODIS is written for path analysis / sensitivity.
    It should not be used as a main competing variable against AlbedoLoss in
    the primary SHAP ranking because it directly contains SurfaceAlbedo.
    """
    records = []

    for year in YEARS:
        alpha_path = common_path("SurfaceAlbedo", year)
        sw_path = common_path("SWdown", year)
        out = common_path("SWabs_MODIS", year)

        if alpha_path.exists() and sw_path.exists() and (overwrite or not out.exists()):
            alpha = read_float(alpha_path)
            sw = read_float(sw_path)
            arr = (1.0 - alpha) * sw
            arr[~mask | ~np.isfinite(alpha) | ~np.isfinite(sw)] = np.nan
            write_array(out, clean_array(arr, "positive"), profile, {"variable": "SWabs_MODIS", "year": str(year), "mask": str(GLOBAL_LAND_VECTOR)}, mask=mask)

        records.append({
            "Variable": "SWabs_MODIS",
            "Year": year,
            "NativePath": "",
            "CommonPath": str(out),
            "SourcePath": f"{alpha_path}; {sw_path}",
            "DataType": "derived_common_grid_raster",
            "Role": "pathway_sensitivity_radiative_input",
            "Units": "same_as_SWdown",
            "Required": False,
            "Status": "ok" if out.exists() else "missing_optional",
            "NativeStatus": "",
            "CommonGridResampling": "",
        })

    return records


# =============================================================================
# 10. QC and manifest
# =============================================================================

def stats(path: str) -> dict:
    if not path:
        return {"N": 0, "ValidFraction": 0.0, "Min": math.nan, "Max": math.nan, "Mean": math.nan, "Std": math.nan}

    p = Path(path)
    if not p.exists():
        return {"N": 0, "ValidFraction": 0.0, "Min": math.nan, "Max": math.nan, "Mean": math.nan, "Std": math.nan}

    arr = read_float(p)
    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        return {"N": 0, "ValidFraction": 0.0, "Min": math.nan, "Max": math.nan, "Mean": math.nan, "Std": math.nan}

    return {
        "N": int(vals.size),
        "ValidFraction": float(vals.size / arr.size),
        "Min": float(vals.min()),
        "Max": float(vals.max()),
        "Mean": float(vals.mean()),
        "Std": float(vals.std(ddof=1)) if vals.size > 1 else math.nan,
    }


def write_manifest(records: List[dict]) -> None:
    df = pd.DataFrame(records)

    # Add a more explicit QC flag so optional missing files are not confused with failed required files.
    df["QC_Level"] = np.where(df["Status"].eq("ok"), "pass", np.where(df["Required"].astype(str).eq("True"), "required_problem", "optional_not_available"))

    manifest_path = OUT_DIR / "R3_annual_raster_manifest.csv"
    df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    qc = []
    for row in records:
        rec = dict(row)
        rec.update({f"Common_{k}": v for k, v in stats(row.get("CommonPath", "")).items()})
        rec.update({f"Native_{k}": v for k, v in stats(row.get("NativePath", "")).items()})
        qc.append(rec)

    qc_df = pd.DataFrame(qc)
    qc_df["QC_Level"] = np.where(qc_df["Status"].eq("ok"), "pass", np.where(qc_df["Required"].astype(str).eq("True"), "required_problem", "optional_not_available"))
    qc_df.to_csv(OUT_DIR / "R3_annual_raster_qc.csv", index=False, encoding="utf-8-sig")


def write_model_variable_groups() -> None:
    json_path = META_OUT / "R3_model_variable_groups.json"
    json_path.write_text(json.dumps(MODEL_VARIABLE_GROUPS, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for model, cfg in MODEL_VARIABLE_GROUPS.items():
        for key, values in cfg.items():
            if isinstance(values, list):
                for value in values:
                    rows.append({"Model": model, "Field": key, "Variable_or_equation": value})
            else:
                rows.append({"Model": model, "Field": key, "Variable_or_equation": str(values)})

    pd.DataFrame(rows).to_csv(META_OUT / "R3_model_variable_groups.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# 11. Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prefer-gee", action="store_true")
    parser.add_argument("--gee-project", default=None)
    parser.add_argument("--skip-glass", action="store_true")
    parser.add_argument("--download-official-tables", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    profile = profile_template()
    mask = global_land_mask(profile, overwrite=args.overwrite)

    print(f"[out] {OUT_DIR}")
    print(f"[template/common grid] {TEMPLATE_RASTER}")
    print(f"[global land vector] {GLOBAL_LAND_VECTOR}")
    print(f"[common-grid global land pixels] {int(mask.sum())}")

    if args.download_official_tables:
        download_official_tables(args.overwrite)

    controls = annual_controls()
    print(f"[annual controls] {TABLE_OUT / 'R3_annual_background_controls.csv'} rows={len(controls)}")

    records: List[dict] = []
    records.append(write_area_weight(profile, mask, args.overwrite))
    records.extend(prepare_local(profile, args.overwrite, args.skip_glass, mask=mask))

    if args.prefer_gee:
        records.extend(prepare_gee(profile, args.overwrite, args.gee_project, mask=mask))

    records.extend(derive_albedo_loss(profile, mask, args.overwrite))
    records.extend(derive_swabs(profile, args.overwrite, mask=mask))

    write_model_variable_groups()
    write_manifest(records)

    print("[done] Result 3 land-masked common-grid rasters, annual controls, model groups and QC were written.")
    print(f"[manifest] {OUT_DIR / 'R3_annual_raster_manifest.csv'}")
    print(f"[qc] {OUT_DIR / 'R3_annual_raster_qc.csv'}")
    print(f"[annual controls] {TABLE_OUT / 'R3_annual_background_controls.csv'}")
    print(f"[model groups] {META_OUT / 'R3_model_variable_groups.json'}")


if __name__ == "__main__":
    main()

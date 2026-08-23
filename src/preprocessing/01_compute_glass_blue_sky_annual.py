# -*- coding: utf-8 -*-
"""
GLASS monthly blue-sky albedo -> annual products using ERA5-Land LSM.

Purpose
-------
1. Check monthly-file completeness for every year.
2. Aggregate monthly GLASS blue-sky albedo to annual mean using calendar-day weights.
3. KEEP GLASS at its native grid/resolution; do NOT force it to MODIS/CERES/panel resolution.
4. Use one independent, static ERA5-Land land-sea mask as the land-domain definition:
       lsm_1279l4_0.1x0.1.grb_v4_unpack.nc
5. Map the original 0–1 ERA5-Land LSM to the GLASS native grid using nearest-neighbour
   sampling (0.1° -> finer GLASS grid; no artificial new spatial information).
6. Classify land as LSM > 0.5.
7. Apply the SAME derived GLASS-grid mask to every year.
8. Calculate cosine-latitude-area-weighted global land mean.
9. Export annual native GeoTIFFs, masked GeoTIFFs, mask products, CSV summaries and QA logs.

Important scientific choices
----------------------------
- The ERA5-Land NetCDF remains the independent MASTER land definition.
- No MODIS-valid-pixel mask is used.
- No MODIS / CERES / panel target grid is used in this script.
- GLASS is not resampled merely for masking.
- Because the source LSM (0.1°) is coarser than GLASS, the fractional LSM is transferred
  to GLASS with nearest-neighbour sampling, then thresholded at > 0.5.
- For datasets coarser than 0.1° (e.g. ERA5 0.25°, CERES 1°), derive their masks separately
  from the same MASTER LSM, preferably by averaging land fraction to their own grids
  before applying the >0.5 threshold.
- Python 3.8 compatible.
"""

from __future__ import annotations

from pathlib import Path
import calendar
import gc
import re
import warnings

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import reproject


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# Monthly GLASS blue-sky products.
MONTHLY_DIR = Path(
    r"D:\10_Research\01_Datasets\02_DataProcess"
    r"\03_SurfaceAlbedo_GLASS\blueSky_monthly"
)

# Annual output root.
OUT_ROOT = Path(
    r"D:\10_Research\01_Datasets\02_DataProcess"
    r"\03_SurfaceAlbedo_GLASS\blueSky_annual_updated_updated"
)

OUT_NATIVE = OUT_ROOT / "annual_native"
OUT_MASKED = OUT_ROOT / "annual_native_ERA5Land_landmask"
OUT_MASK = OUT_ROOT / "mask_ERA5Land"
OUT_TABLE = OUT_ROOT / "tables"
OUT_LOG = OUT_ROOT / "logs"

for _d in [OUT_NATIVE, OUT_MASKED, OUT_MASK, OUT_TABLE, OUT_LOG]:
    _d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Independent MASTER land-sea mask
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"D:\10_Research\01_Datasets\01_DataRaw\ERA5"
)

ERA5LAND_LSM_FILENAME = "lsm.nc"

# Preferred exact location. If it is not found there, the script will
# automatically search PROJECT_ROOT recursively for the same filename.
ERA5LAND_LSM_NC = PROJECT_ROOT / ERA5LAND_LSM_FILENAME

LAND_THRESHOLD = 0.5

# Outputs derived from the MASTER LSM for the GLASS native grid.
OUT_LSM_FRACTION_GLASS = OUT_MASK / "ERA5Land_LSM_fraction_on_GLASS_native_grid.tif"
OUT_LAND_MASK_GLASS = OUT_MASK / "common_land_mask_GLASS_native_ERA5Land_gt0p5.tif"
OUT_MASK_QA = OUT_MASK / "ERA5Land_mask_GLASS_native_QA.csv"


# -----------------------------------------------------------------------------
# Years
# -----------------------------------------------------------------------------
START_YEAR = 2001
END_YEAR = 2024
AUTO_DETECT_AVAILABLE_YEARS = True


# -----------------------------------------------------------------------------
# Missing-month rule
# -----------------------------------------------------------------------------
REQUIRE_ALL_12_MONTHS = True
MIN_MONTHS_PER_YEAR = 10


# -----------------------------------------------------------------------------
# Data validity
# -----------------------------------------------------------------------------
ALBEDO_MIN = 0.0
ALBEDO_MAX = 1.0
NODATA_VALUE = -9999.0
MASK_NODATA = 255
MIN_VALID_FILE_SIZE_BYTES = 10 * 1024


# -----------------------------------------------------------------------------
# Existing outputs
# -----------------------------------------------------------------------------
SKIP_EXISTING = True
OVERWRITE = False


# =============================================================================
# 2. MONTHLY INVENTORY AND COMPLETENESS
# =============================================================================

MONTHLY_RE = re.compile(
    r"^GLASS_BlueSky_shortwave_monthly_(\d{4})_(\d{2})\.tif$",
    re.IGNORECASE,
)


def scan_monthly_files() -> pd.DataFrame:
    if not MONTHLY_DIR.exists():
        raise FileNotFoundError(
            "\nMONTHLY_DIR does not exist:\n"
            f"{MONTHLY_DIR}"
        )

    rows = []

    for path in sorted(MONTHLY_DIR.rglob("*.tif")):
        m = MONTHLY_RE.match(path.name)
        if not m:
            continue

        year = int(m.group(1))
        month = int(m.group(2))

        if not (1 <= month <= 12):
            continue

        readable = True
        read_error = ""
        width = height = np.nan
        crs = ""
        transform_text = ""
        nodata = np.nan

        try:
            with rasterio.open(path) as src:
                width = src.width
                height = src.height
                crs = str(src.crs)
                transform_text = str(src.transform)
                nodata = src.nodata
        except Exception as exc:
            readable = False
            read_error = f"{type(exc).__name__}: {exc}"

        rows.append({
            "year": year,
            "month": month,
            "path": str(path),
            "size_mb": path.stat().st_size / 1024.0 / 1024.0,
            "readable": readable,
            "read_error": read_error,
            "width": width,
            "height": height,
            "crs": crs,
            "transform": transform_text,
            "nodata": nodata,
        })

    if not rows:
        raise FileNotFoundError(
            "\nNo monthly GLASS GeoTIFFs matching the required filename were found:\n"
            f"{MONTHLY_DIR}\n"
            "Expected: GLASS_BlueSky_shortwave_monthly_YYYY_MM.tif"
        )

    inventory = pd.DataFrame(rows)
    inventory.to_csv(
        OUT_TABLE / "GLASS_BlueSky_monthly_file_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return inventory


def build_monthly_completeness(inventory: pd.DataFrame) -> pd.DataFrame:
    if AUTO_DETECT_AVAILABLE_YEARS:
        years = sorted(
            int(y) for y in inventory["year"].unique()
            if START_YEAR <= int(y) <= END_YEAR
        )
    else:
        years = list(range(START_YEAR, END_YEAR + 1))

    if not years:
        raise RuntimeError(
            f"No monthly GLASS years found between {START_YEAR} and {END_YEAR}."
        )

    rows = []

    for year in years:
        for month in range(1, 13):
            sub = inventory[
                (inventory["year"] == year)
                & (inventory["month"] == month)
            ].copy()

            if len(sub) == 0:
                status = "missing"
                selected_path = ""
            elif len(sub) > 1:
                status = "duplicate"
                selected_path = " | ".join(sub["path"].astype(str).tolist())
            else:
                r = sub.iloc[0]
                selected_path = str(r["path"])

                if not bool(r["readable"]):
                    status = "unreadable"
                elif float(r["size_mb"]) * 1024 * 1024 < MIN_VALID_FILE_SIZE_BYTES:
                    status = "too_small"
                else:
                    status = "ok"

            rows.append({
                "year": year,
                "month": month,
                "month_name": calendar.month_abbr[month],
                "days_in_month": calendar.monthrange(year, month)[1],
                "status": status,
                "selected_path": selected_path,
            })

    completeness = pd.DataFrame(rows)
    completeness.to_csv(
        OUT_TABLE / "GLASS_BlueSky_monthly_completeness_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    year_rows = []

    for year, sub in completeness.groupby("year"):
        ok_months = sub.loc[sub["status"] == "ok", "month"].astype(int).tolist()
        bad = sub.loc[sub["status"] != "ok", ["month", "status"]]

        bad_text = "; ".join(
            f"{int(r.month):02d}:{r.status}"
            for r in bad.itertuples(index=False)
        )

        year_rows.append({
            "year": int(year),
            "ok_months": len(ok_months),
            "missing_or_invalid_months": bad_text,
            "complete_12_months": len(ok_months) == 12,
            "calendar_days_if_complete": (
                366 if calendar.isleap(int(year)) else 365
            ),
        })

    year_summary = pd.DataFrame(year_rows)
    year_summary.to_csv(
        OUT_TABLE / "GLASS_BlueSky_monthly_completeness_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("STEP 0 — MONTHLY COMPLETENESS CHECK")
    print("=" * 100)

    for r in year_summary.itertuples(index=False):
        if r.complete_12_months:
            print(f"{r.year}: OK — 12/12 months")
        else:
            print(
                f"{r.year}: INCOMPLETE — {r.ok_months}/12 months | "
                f"{r.missing_or_invalid_months}"
            )

    return completeness


def file_for_year_month(
    completeness: pd.DataFrame,
    year: int,
    month: int,
):
    sub = completeness[
        (completeness["year"] == year)
        & (completeness["month"] == month)
    ]

    if len(sub) != 1:
        return None

    row = sub.iloc[0]

    if row["status"] != "ok":
        return None

    return Path(row["selected_path"])


# =============================================================================
# 3. RASTER HELPERS
# =============================================================================

def read_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        arr[arr < -1e20] = np.nan

    return arr, profile


def clean_albedo(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    arr[(arr < ALBEDO_MIN) | (arr > ALBEDO_MAX)] = np.nan
    return arr


def normalized_output_profile(profile: dict) -> dict:
    p = profile.copy()
    p.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=NODATA_VALUE,
        compress="lzw",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    return p


def save_float_raster(arr: np.ndarray, profile: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = normalized_output_profile(profile)

    out = arr.astype("float32", copy=True)
    out[~np.isfinite(out)] = NODATA_VALUE

    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(out, 1)


def save_binary_mask(mask: np.ndarray, profile: dict, out_path: Path):
    p = profile.copy()
    p.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=MASK_NODATA,
        compress="lzw",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    out = np.full(mask.shape, MASK_NODATA, dtype="uint8")
    out[mask == 0] = 0
    out[mask == 1] = 1

    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(out, 1)
        dst.update_tags(
            source="ERA5-Land land-sea mask",
            source_file=str(resolve_era5land_lsm_path()),
            land_rule=f"LSM > {LAND_THRESHOLD}",
            land_value="1",
            water_value="0",
            nodata_value=str(MASK_NODATA),
        )


def same_grid(profile_a: dict, profile_b: dict) -> bool:
    try:
        same_crs = profile_a.get("crs") == profile_b.get("crs")
        same_shape = (
            int(profile_a["height"]) == int(profile_b["height"])
            and int(profile_a["width"]) == int(profile_b["width"])
        )

        ta = profile_a["transform"]
        tb = profile_b["transform"]

        same_transform = all(
            abs(float(a) - float(b)) < 1e-10
            for a, b in zip(ta, tb)
        )

        return same_crs and same_shape and same_transform

    except Exception:
        return False


def pixel_size(profile: dict):
    transform = profile["transform"]
    return abs(float(transform.a)), abs(float(transform.e))


# =============================================================================
# 4. MONTHLY -> ANNUAL, CALENDAR-DAY WEIGHTED
# =============================================================================

def aggregate_one_year_native(
    year: int,
    completeness: pd.DataFrame,
):
    out_tif = OUT_NATIVE / (
        f"GLASS_BlueSky_shortwave_annual_{year}_native.tif"
    )

    year_check = completeness[completeness["year"] == year]

    ok_months = year_check.loc[
        year_check["status"] == "ok", "month"
    ].astype(int).tolist()

    bad_rows = year_check[year_check["status"] != "ok"]

    bad_text = "; ".join(
        f"{int(r.month):02d}:{r.status}"
        for r in bad_rows.itertuples(index=False)
    )

    if REQUIRE_ALL_12_MONTHS and len(ok_months) != 12:
        return None, None, {
            "year": year,
            "status": "skip_incomplete_year",
            "available_months": len(ok_months),
            "missing_or_invalid_months": bad_text,
            "available_calendar_days": int(sum(
                calendar.monthrange(year, m)[1] for m in ok_months
            )),
            "expected_calendar_days": 366 if calendar.isleap(year) else 365,
            "native_output": "",
        }

    if not REQUIRE_ALL_12_MONTHS and len(ok_months) < MIN_MONTHS_PER_YEAR:
        return None, None, {
            "year": year,
            "status": "skip_too_few_months",
            "available_months": len(ok_months),
            "missing_or_invalid_months": bad_text,
            "available_calendar_days": int(sum(
                calendar.monthrange(year, m)[1] for m in ok_months
            )),
            "expected_calendar_days": 366 if calendar.isleap(year) else 365,
            "native_output": "",
        }

    if (
        SKIP_EXISTING
        and not OVERWRITE
        and out_tif.exists()
        and out_tif.stat().st_size >= MIN_VALID_FILE_SIZE_BYTES
    ):
        arr, profile = read_raster(out_tif)

        return arr, profile, {
            "year": year,
            "status": "skip_existing_native_annual",
            "available_months": len(ok_months),
            "missing_or_invalid_months": bad_text,
            "available_calendar_days": int(sum(
                calendar.monthrange(year, m)[1] for m in ok_months
            )),
            "expected_calendar_days": 366 if calendar.isleap(year) else 365,
            "native_output": str(out_tif),
        }

    weighted_sum = None
    valid_day_sum = None
    valid_month_count = None
    ref_profile = None

    for month in ok_months:
        path = file_for_year_month(completeness, year, month)

        if path is None:
            continue

        arr, profile = read_raster(path)
        arr = clean_albedo(arr)

        if ref_profile is None:
            ref_profile = profile.copy()
            weighted_sum = np.zeros(arr.shape, dtype="float64")
            valid_day_sum = np.zeros(arr.shape, dtype="float32")
            valid_month_count = np.zeros(arr.shape, dtype="uint8")
        else:
            if not same_grid(profile, ref_profile):
                raise ValueError(
                    f"Monthly native-grid mismatch in {path.name}: "
                    f"{arr.shape} / {profile['transform']}."
                )

        days = float(calendar.monthrange(year, month)[1])
        valid = np.isfinite(arr)

        weighted_sum[valid] += arr[valid].astype("float64") * days
        valid_day_sum[valid] += days
        valid_month_count[valid] += 1

        del arr, valid
        gc.collect()

    if ref_profile is None:
        return None, None, {
            "year": year,
            "status": "no_valid_monthly_rasters",
            "available_months": 0,
            "missing_or_invalid_months": bad_text,
            "available_calendar_days": 0,
            "expected_calendar_days": 366 if calendar.isleap(year) else 365,
            "native_output": "",
        }

    annual = np.full(weighted_sum.shape, np.nan, dtype="float32")
    valid_annual = valid_day_sum > 0

    annual[valid_annual] = (
        weighted_sum[valid_annual] / valid_day_sum[valid_annual]
    ).astype("float32")

    annual = clean_albedo(annual)
    save_float_raster(annual, ref_profile, out_tif)

    valid_days_values = valid_day_sum[valid_annual]
    valid_month_values = valid_month_count[valid_annual]

    info = {
        "year": year,
        "status": "ok_native_annual",
        "available_months": len(ok_months),
        "missing_or_invalid_months": bad_text,
        "available_calendar_days": int(sum(
            calendar.monthrange(year, m)[1] for m in ok_months
        )),
        "expected_calendar_days": 366 if calendar.isleap(year) else 365,
        "pixel_valid_days_median": (
            float(np.nanmedian(valid_days_values))
            if valid_days_values.size else np.nan
        ),
        "pixel_valid_months_median": (
            float(np.nanmedian(valid_month_values))
            if valid_month_values.size else np.nan
        ),
        "native_valid_pixels": int(np.isfinite(annual).sum()),
        "native_output": str(out_tif),
    }

    del weighted_sum, valid_day_sum, valid_month_count, valid_annual
    gc.collect()

    return annual, ref_profile, info



# =============================================================================
# 5. ERA5-LAND MASTER LSM
# =============================================================================

def resolve_era5land_lsm_path() -> Path:
    """
    Resolve the actual ERA5-Land LSM NetCDF path robustly.

    Search order
    ------------
    1. Exact configured path:
       PROJECT_ROOT / ERA5LAND_LSM_FILENAME
    2. Recursive exact-filename search under PROJECT_ROOT
    3. Recursive broad search for *lsm*.nc under PROJECT_ROOT

    The function prints the resolved path so the path used in the analysis
    is explicit and reproducible.
    """
    exact = ERA5LAND_LSM_NC

    print("\n" + "=" * 100)
    print("ERA5-LAND LSM PATH CHECK")
    print("=" * 100)
    print("Configured path:", exact)
    print("Configured path exists:", exact.exists())
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("PROJECT_ROOT exists:", PROJECT_ROOT.exists())

    if exact.exists() and exact.is_file():
        resolved = exact.resolve()
        print("Resolved LSM:", resolved)
        return resolved

    if not PROJECT_ROOT.exists():
        parent = PROJECT_ROOT.parent
        msg = [
            "PROJECT_ROOT does not exist:",
            str(PROJECT_ROOT),
            "",
            "Parent directory:",
            str(parent),
            f"Parent exists: {parent.exists()}",
        ]

        if parent.exists():
            try:
                dirs = sorted([p.name for p in parent.iterdir() if p.is_dir()])
                msg.append("")
                msg.append("Available sibling directories:")
                msg.extend(f"  {d}" for d in dirs[:50])
            except Exception:
                pass

        raise FileNotFoundError("\n".join(msg))

    # Exact filename, anywhere under project root.
    exact_hits = sorted(
        p for p in PROJECT_ROOT.rglob(ERA5LAND_LSM_FILENAME)
        if p.is_file()
    )

    if exact_hits:
        print(
            f"[INFO] Configured location was not found, but {len(exact_hits)} "
            "matching file(s) were found recursively:"
        )
        for p in exact_hits:
            print("  ", p)

        # Prefer the shallowest path; if tied, lexical order is deterministic.
        resolved = sorted(
            exact_hits,
            key=lambda p: (len(p.parts), str(p).lower())
        )[0].resolve()

        print("Resolved LSM:", resolved)
        return resolved

    # Broader fallback in case the downloaded filename differs slightly.
    broad_hits = sorted(
        p for p in PROJECT_ROOT.rglob("*.nc")
        if p.is_file() and "lsm" in p.name.lower()
    )

    if broad_hits:
        print(
            f"[WARNING] Exact filename was not found. "
            f"Found {len(broad_hits)} NetCDF file(s) containing 'lsm':"
        )
        for p in broad_hits[:50]:
            print("  ", p)

        if len(broad_hits) == 1:
            resolved = broad_hits[0].resolve()
            print("Resolved LSM by broad fallback:", resolved)
            return resolved

        raise FileNotFoundError(
            "Multiple possible LSM NetCDF files were found. "
            "Set ERA5LAND_LSM_NC to the intended file explicitly."
        )

    # Provide a compact directory inventory for diagnosis.
    nc_files = sorted(
        p for p in PROJECT_ROOT.rglob("*.nc")
        if p.is_file()
    )

    msg = [
        "ERA5-Land LSM NetCDF could not be found.",
        "",
        "Configured path:",
        str(exact),
        "",
        "Recursive search root:",
        str(PROJECT_ROOT),
        "",
        "Expected filename:",
        ERA5LAND_LSM_FILENAME,
    ]

    if nc_files:
        msg.append("")
        msg.append("Other NetCDF files found under PROJECT_ROOT:")
        msg.extend(f"  {p}" for p in nc_files[:50])
    else:
        msg.append("")
        msg.append("No .nc files were found anywhere under PROJECT_ROOT.")

    raise FileNotFoundError("\n".join(msg))



def identify_lsm_variable(ds: xr.Dataset) -> str:
    for name in ["lsm", "land_sea_mask", "land-sea-mask"]:
        if name in ds.data_vars:
            return name

    candidates = []

    for name in ds.data_vars:
        low = name.lower()

        if "lsm" in low or ("land" in low and "mask" in low):
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(
        "Cannot uniquely identify ERA5-Land LSM variable.\n"
        f"Available variables: {list(ds.data_vars)}"
    )


def identify_coord_name(da: xr.DataArray, kind: str) -> str:
    if kind == "lat":
        exact = ["lat", "latitude"]
        clue = "lat"
    else:
        exact = ["lon", "longitude"]
        clue = "lon"

    for name in exact:
        if name in da.coords:
            return name

    for name in da.coords:
        if clue in name.lower():
            return name

    for name in da.dims:
        if clue in name.lower():
            return name

    raise KeyError(
        f"Cannot identify {kind} coordinate.\n"
        f"coords={list(da.coords)}, dims={list(da.dims)}"
    )


def load_era5land_lsm():
    """
    Read the ERA5-Land 0–1 fractional land-sea mask robustly.

    The source NetCDF stores the regular 0.1° coordinates as float32:
        longitude: 0.0, 0.1, ..., 359.9
        latitude : 90.0, 89.9, ..., -90.0

    Because float32 decimal tenths are not exact in binary, differences such as
    0.099976–0.100006° may occur numerically even though the intended grid is
    perfectly regular. Therefore the nominal resolution is reconstructed from
    the coordinate span and number of grid points, then rounded to 6 decimals.

    The LSM DATA VALUES are never altered; only the coordinate labels used for
    indexing/QA are normalized to their intended regular grid.
    """
    lsm_path = resolve_era5land_lsm_path()

    # Explicit netcdf4 avoids the h5py/h5netcdf HDF5 mismatch seen in py38.
    ds = xr.open_dataset(lsm_path, engine="netcdf4")

    var_name = identify_lsm_variable(ds)
    da = ds[var_name].squeeze(drop=True)

    lat_name = identify_coord_name(da, "lat")
    lon_name = identify_coord_name(da, "lon")

    rename = {}
    if lat_name != "lat":
        rename[lat_name] = "lat"
    if lon_name != "lon":
        rename[lon_name] = "lon"
    if rename:
        da = da.rename(rename)

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(
            "ERA5-Land LSM must contain lat/lon dimensions after squeeze.\n"
            f"Current dims: {da.dims}"
        )

    da = da.transpose("lat", "lon")

    lat_raw = np.asarray(da["lat"].values, dtype="float64")
    lon_raw = np.asarray(da["lon"].values, dtype="float64")
    arr = np.asarray(da.values, dtype="float32")

    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < 0.0) | (arr > 1.0)] = np.nan

    if len(lat_raw) < 2 or len(lon_raw) < 2:
        raise ValueError("ERA5-Land coordinate vectors are too short.")

    # ---------------------------------------------------------------------
    # 1. Confirm monotonic source coordinates.
    # ---------------------------------------------------------------------
    raw_lon_diff = np.diff(lon_raw)
    raw_lat_diff = np.diff(lat_raw)

    if not (np.all(raw_lon_diff > 0) or np.all(raw_lon_diff < 0)):
        raise ValueError("ERA5-Land longitude is not monotonic.")

    if not (np.all(raw_lat_diff > 0) or np.all(raw_lat_diff < 0)):
        raise ValueError("ERA5-Land latitude is not monotonic.")

    # ---------------------------------------------------------------------
    # 2. Reconstruct the NOMINAL regular-grid spacing from endpoints/count.
    #
    # This is much more robust than using median(diff), because accumulating
    # a tiny float32 error over 3600 longitude points can falsely imply a
    # 360.0108° domain instead of the intended 360.0° periodic grid.
    # ---------------------------------------------------------------------
    xres_from_span = abs(
        (float(lon_raw[-1]) - float(lon_raw[0]))
        / float(len(lon_raw) - 1)
    )
    yres_from_span = abs(
        (float(lat_raw[-1]) - float(lat_raw[0]))
        / float(len(lat_raw) - 1)
    )

    # ERA5-Land coordinates are nominally decimal-degree coordinates.
    # Six decimals is far finer than the source 0.1° resolution.
    xres = round(xres_from_span, 6)
    yres = round(yres_from_span, 6)

    if xres <= 0 or yres <= 0:
        raise ValueError(
            f"Invalid reconstructed ERA5-Land resolution: {xres}, {yres}"
        )

    # ---------------------------------------------------------------------
    # 3. Build normalized nominal coordinate labels WITHOUT modifying LSM.
    # ---------------------------------------------------------------------
    lon_start = round(float(lon_raw[0]), 6)
    lat_start = round(float(lat_raw[0]), 6)

    lon_sign = 1.0 if lon_raw[-1] > lon_raw[0] else -1.0
    lat_sign = 1.0 if lat_raw[-1] > lat_raw[0] else -1.0

    lon = (
        lon_start
        + lon_sign * np.arange(len(lon_raw), dtype="float64") * xres
    )
    lat = (
        lat_start
        + lat_sign * np.arange(len(lat_raw), dtype="float64") * yres
    )

    # Compare actual stored float32 coordinates with the intended regular grid.
    lon_coord_dev = float(np.nanmax(np.abs(lon_raw - lon)))
    lat_coord_dev = float(np.nanmax(np.abs(lat_raw - lat)))

    # 5e-5° is only 0.05% of a 0.1° cell and safely covers float32 rounding.
    coord_tol = max(5e-5, 5e-4 * max(xres, yres))

    print("\nERA5-Land coordinate QA:")
    print(
        f"  stored longitude: {lon_raw[0]:.9f} ... {lon_raw[-1]:.9f}"
    )
    print(
        f"  nominal longitude: {lon[0]:.6f} ... {lon[-1]:.6f} | "
        f"dx={xres:.6f}° | max stored-vs-nominal deviation="
        f"{lon_coord_dev:.10g}°"
    )
    print(
        f"  stored latitude : {lat_raw[0]:.9f} ... {lat_raw[-1]:.9f}"
    )
    print(
        f"  nominal latitude : {lat[0]:.6f} ... {lat[-1]:.6f} | "
        f"dy={yres:.6f}° | max stored-vs-nominal deviation="
        f"{lat_coord_dev:.10g}°"
    )
    print(f"  coordinate tolerance={coord_tol:.10g}°")

    if lon_coord_dev > coord_tol:
        raise ValueError(
            "ERA5-Land longitude coordinates deviate too much from the "
            f"reconstructed regular grid: {lon_coord_dev} > {coord_tol}."
        )

    if lat_coord_dev > coord_tol:
        raise ValueError(
            "ERA5-Land latitude coordinates deviate too much from the "
            f"reconstructed regular grid: {lat_coord_dev} > {coord_tol}."
        )

    # ---------------------------------------------------------------------
    # 4. Global-domain QA using NOMINAL spacing, not noisy float32 diffs.
    # ---------------------------------------------------------------------
    nominal_lon_coverage = xres * len(lon)

    # For this ERA5-Land auxiliary file we expect a periodic 360° longitude
    # domain: 3600 points × 0.1° = 360°.
    if abs(nominal_lon_coverage - 360.0) > 1e-6:
        raise ValueError(
            "ERA5-Land longitude grid does not appear to be global/periodic "
            "after nominal-coordinate reconstruction: "
            f"N={len(lon)}, dx={xres}, N*dx={nominal_lon_coverage}."
        )

    # Latitude grid includes both poles: 1801 points at 0.1° from +90 to -90.
    nominal_lat_span = abs(float(lat[-1]) - float(lat[0]))
    if abs(nominal_lat_span - 180.0) > 1e-6:
        raise ValueError(
            "ERA5-Land latitude grid does not span -90..90 after "
            f"reconstruction: span={nominal_lat_span}."
        )

    meta = {
        "source_file": str(lsm_path),
        "variable": var_name,
        "height": arr.shape[0],
        "width": arr.shape[1],
        "source_xres_deg": xres,
        "source_yres_deg": yres,
        "source_min": float(np.nanmin(arr)),
        "source_max": float(np.nanmax(arr)),
        "source_mean": float(np.nanmean(arr)),
        "lat_min": float(np.nanmin(lat)),
        "lat_max": float(np.nanmax(lat)),
        "lon_min": float(np.nanmin(lon)),
        "lon_max": float(np.nanmax(lon)),
        "longitude_convention": "0_to_360_periodic",
        "coordinate_normalization": "nominal grid reconstructed from endpoints/count",
        "longitude_max_stored_vs_nominal_deviation_deg": lon_coord_dev,
        "latitude_max_stored_vs_nominal_deviation_deg": lat_coord_dev,
    }

    ds.close()

    return arr, lat, lon, meta


def derive_era5land_mask_on_glass_grid(glass_profile: dict):
    """
    Transfer the ERA5-Land 0–1 LSM to the GLASS native grid using an exact
    nearest-coordinate lookup.

    Notes
    -----
    load_era5land_lsm() returns FOUR objects:
        src_lsm, src_lat, src_lon, src_meta

    The ERA5-Land auxiliary file is a global grid-point field:
        lon = 0.0 ... 359.9 (periodic)
        lat = 90.0 ... -90.0

    GLASS is kept on its native 0.05° grid. Each GLASS pixel centre is mapped
    to the nearest ERA5-Land 0.1° grid point; longitude is treated cyclically.
    The land/water threshold is then:
        land = LSM > 0.5
    """
    # -------------------------------------------------------------------------
    # Reuse an already-derived mask only when BOTH files exist and share the
    # current GLASS native grid.
    # -------------------------------------------------------------------------
    if (
        SKIP_EXISTING
        and not OVERWRITE
        and OUT_LSM_FRACTION_GLASS.exists()
        and OUT_LAND_MASK_GLASS.exists()
    ):
        fraction, f_profile = read_raster(OUT_LSM_FRACTION_GLASS)

        with rasterio.open(OUT_LAND_MASK_GLASS) as src:
            mask_raw = src.read(1)
            m_profile = src.profile.copy()

        if (
            same_grid(f_profile, glass_profile)
            and same_grid(m_profile, glass_profile)
        ):
            land_mask = mask_raw == 1

            print(
                "[INFO] Reusing existing ERA5-Land mask on GLASS native grid."
            )
            return fraction, land_mask

        print(
            "[INFO] Existing ERA5-Land derived mask uses a different grid; "
            "rebuilding it."
        )

    # -------------------------------------------------------------------------
    # IMPORTANT FIX:
    # load_era5land_lsm() returns four values, not three.
    # -------------------------------------------------------------------------
    src_lsm, src_lat, src_lon, src_meta = load_era5land_lsm()

    print("\nERA5-Land MASTER LSM")
    print("Source:", src_meta["source_file"])
    print(
        "Source shape:",
        src_meta["height"],
        "x",
        src_meta["width"],
    )
    print(
        "Source resolution:",
        src_meta["source_xres_deg"],
        "x",
        src_meta["source_yres_deg"],
        "degrees",
    )
    print(
        "Source fraction range:",
        src_meta["source_min"],
        "to",
        src_meta["source_max"],
    )

    # -------------------------------------------------------------------------
    # GLASS pixel-centre coordinates
    # -------------------------------------------------------------------------
    transform = glass_profile["transform"]

    if abs(float(transform.b)) > 1e-12 or abs(float(transform.d)) > 1e-12:
        raise ValueError(
            "GLASS transform contains rotation/shear. "
            "This script expects a north-up regular lon-lat grid."
        )

    nrows = int(glass_profile["height"])
    ncols = int(glass_profile["width"])

    target_lat = (
        float(transform.f)
        + (np.arange(nrows, dtype="float64") + 0.5)
        * float(transform.e)
    )

    target_lon = (
        float(transform.c)
        + (np.arange(ncols, dtype="float64") + 0.5)
        * float(transform.a)
    )

    # -------------------------------------------------------------------------
    # Nearest source latitude indices
    # -------------------------------------------------------------------------
    lat_step = float(src_meta["source_yres_deg"])
    lon_step = float(src_meta["source_xres_deg"])

    if src_lat[0] > src_lat[-1]:
        # Source latitude descending: +90 -> -90.
        lat_idx = np.rint(
            (float(src_lat[0]) - target_lat) / lat_step
        ).astype("int64")
    else:
        lat_idx = np.rint(
            (target_lat - float(src_lat[0])) / lat_step
        ).astype("int64")

    lat_idx = np.clip(lat_idx, 0, len(src_lat) - 1)

    # -------------------------------------------------------------------------
    # Nearest source longitude indices with 360° periodic wrapping
    # -------------------------------------------------------------------------
    if src_lon[0] < src_lon[-1]:
        lon0 = float(src_lon[0])

        # Convert GLASS -180..180 pixel centres to the ERA5-Land periodic
        # coordinate domain that begins at lon0 (normally 0°).
        target_lon_wrapped = (
            (target_lon - lon0) % 360.0
        ) + lon0

        lon_idx = np.rint(
            (target_lon_wrapped - lon0) / lon_step
        ).astype("int64")

        # Example: a point nearest 360.0° is physically the same as 0.0°.
        lon_idx = np.mod(lon_idx, len(src_lon))

    else:
        raise ValueError(
            "Descending ERA5-Land longitude is not expected for this file."
        )

    # -------------------------------------------------------------------------
    # Direct nearest-neighbour coordinate lookup.
    #
    # This preserves the original 0–1 ERA5-Land fraction values and does not
    # create artificial sub-0.1° spatial information.
    # -------------------------------------------------------------------------
    fraction_glass = src_lsm[np.ix_(lat_idx, lon_idx)].astype(
        "float32",
        copy=True,
    )

    finite = np.isfinite(fraction_glass)

    if not np.any(finite):
        raise RuntimeError(
            "ERA5-Land LSM -> GLASS mapping produced zero finite pixels."
        )

    fraction_glass[finite] = np.clip(
        fraction_glass[finite],
        0.0,
        1.0,
    )

    land_mask = finite & (fraction_glass > LAND_THRESHOLD)

    # -------------------------------------------------------------------------
    # Save fractional LSM and binary mask on the unchanged GLASS native grid.
    # -------------------------------------------------------------------------
    save_float_raster(
        fraction_glass,
        glass_profile,
        OUT_LSM_FRACTION_GLASS,
    )

    mask01 = np.full(
        fraction_glass.shape,
        MASK_NODATA,
        dtype="uint8",
    )
    mask01[finite & (fraction_glass <= LAND_THRESHOLD)] = 0
    mask01[land_mask] = 1

    save_binary_mask(
        mask01,
        glass_profile,
        OUT_LAND_MASK_GLASS,
    )

    # -------------------------------------------------------------------------
    # QA
    # -------------------------------------------------------------------------
    valid_cells = int(np.count_nonzero(finite))
    land_cells = int(np.count_nonzero(land_mask))
    water_cells = int(
        np.count_nonzero(
            finite & (fraction_glass <= LAND_THRESHOLD)
        )
    )
    nodata_cells = int(fraction_glass.size - valid_cells)

    qa = {
        "master_lsm_source": str(src_meta["source_file"]),
        "master_lsm_variable": src_meta["variable"],
        "master_resolution_deg": (
            f'{src_meta["source_xres_deg"]} x '
            f'{src_meta["source_yres_deg"]}'
        ),
        "master_longitude_convention": src_meta[
            "longitude_convention"
        ],
        "derived_grid": "GLASS_native",
        "derived_width": glass_profile["width"],
        "derived_height": glass_profile["height"],
        "derived_crs": str(glass_profile["crs"]),
        "derived_transform": str(glass_profile["transform"]),
        "derived_xres_deg": pixel_size(glass_profile)[0],
        "derived_yres_deg": pixel_size(glass_profile)[1],
        "resampling": "nearest_coordinate_lookup",
        "periodic_longitude": True,
        "land_rule": f"LSM > {LAND_THRESHOLD}",
        "valid_cells": valid_cells,
        "land_cells": land_cells,
        "water_cells": water_cells,
        "nodata_cells": nodata_cells,
        "land_percent_of_valid_cells": (
            100.0 * land_cells / valid_cells
            if valid_cells > 0
            else np.nan
        ),
        "fraction_output": str(OUT_LSM_FRACTION_GLASS),
        "binary_mask_output": str(OUT_LAND_MASK_GLASS),
    }

    pd.DataFrame([qa]).to_csv(
        OUT_MASK_QA,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDerived ERA5-Land mask on GLASS native grid:")
    print(
        "Mapping: nearest-coordinate lookup with cyclic longitude"
    )
    print("Fraction raster:", OUT_LSM_FRACTION_GLASS)
    print("Binary mask:", OUT_LAND_MASK_GLASS)
    print("Land cells:", f"{land_cells:,}")
    print("Water cells:", f"{water_cells:,}")
    print("NoData cells:", f"{nodata_cells:,}")

    print(
        "Dateline mapping QA:",
        f"first GLASS col -> source lon index {int(lon_idx[0])}; "
        f"last GLASS col -> source lon index {int(lon_idx[-1])}"
    )

    # Optional coordinate-index QA.
    print(
        "Latitude mapping QA:",
        f"first GLASS row -> source lat index {int(lat_idx[0])}; "
        f"last GLASS row -> source lat index {int(lat_idx[-1])}"
    )

    del src_lsm
    gc.collect()

    return fraction_glass, land_mask


# =============================================================================
# 6. AREA-WEIGHTED GLOBAL LAND STATISTICS
# =============================================================================

def require_geographic_grid(profile: dict):
    crs = profile.get("crs", None)

    if crs is None:
        raise ValueError(
            "GLASS grid CRS is missing; latitude weighting cannot be calculated."
        )

    if hasattr(crs, "is_geographic") and not crs.is_geographic:
        raise ValueError(
            f"GLASS CRS is projected ({crs}). "
            "cos(latitude) weighting requires a geographic lon-lat grid."
        )


def row_center_latitudes(profile: dict) -> np.ndarray:
    transform = profile["transform"]
    rows = np.arange(profile["height"], dtype="float64")

    return np.array(
        [(transform * (0.5, r + 0.5))[1] for r in rows],
        dtype="float64",
    )


def latitude_area_weighted_mean(
    arr: np.ndarray,
    profile: dict,
    land_mask: np.ndarray,
):
    require_geographic_grid(profile)

    if arr.shape != land_mask.shape:
        raise ValueError(
            f"Array/mask shape mismatch: {arr.shape} vs {land_mask.shape}"
        )

    lats = row_center_latitudes(profile)
    row_weights = np.cos(np.deg2rad(lats)).astype("float64")

    valid = np.isfinite(arr) & land_mask

    if not np.any(valid):
        return np.nan, 0, np.nan

    weighted_sum = 0.0
    total_weight = 0.0

    for r in range(arr.shape[0]):
        vr = valid[r]

        if not np.any(vr):
            continue

        w = float(row_weights[r])

        if not np.isfinite(w) or w <= 0:
            continue

        vals = arr[r, vr].astype("float64")

        weighted_sum += float(np.sum(vals) * w)
        total_weight += float(vals.size * w)

    mean_value = (
        weighted_sum / total_weight
        if total_weight > 0
        else np.nan
    )

    land_cells = int(np.count_nonzero(land_mask))
    valid_cells = int(np.count_nonzero(valid))

    valid_fraction = (
        100.0 * valid_cells / land_cells
        if land_cells > 0
        else np.nan
    )

    return float(mean_value), valid_cells, float(valid_fraction)


def unweighted_land_mean(
    arr: np.ndarray,
    land_mask: np.ndarray,
):
    valid = np.isfinite(arr) & land_mask

    if not np.any(valid):
        return np.nan, 0

    return float(np.nanmean(arr[valid])), int(np.count_nonzero(valid))


# =============================================================================
# 7. MAIN
# =============================================================================

def process_all_years():
    print("\n" + "#" * 100)
    print("GLASS MONTHLY -> ANNUAL BLUE-SKY ALBEDO")
    print("LAND DOMAIN: ERA5-Land static LSM")
    print("#" * 100)

    print("MONTHLY_DIR:", MONTHLY_DIR)
    print("OUT_ROOT:", OUT_ROOT)
    print("ERA5LAND_LSM_NC configured:", ERA5LAND_LSM_NC)
    print("LAND_THRESHOLD:", LAND_THRESHOLD)
    print("GLASS grid policy: KEEP NATIVE GRID / NO MODIS RESAMPLING")

    # Resolve the LSM path BEFORE monthly/annual processing so path errors fail fast.
    resolved_lsm = resolve_era5land_lsm_path()
    print("ERA5LAND_LSM_NC resolved:", resolved_lsm)

    # -------------------------------------------------------------------------
    # STEP 0: monthly input audit
    # -------------------------------------------------------------------------
    inventory = scan_monthly_files()
    completeness = build_monthly_completeness(inventory)

    years = sorted(
        int(y) for y in completeness["year"].unique()
        if START_YEAR <= int(y) <= END_YEAR
    )

    fatal = completeness[
        completeness["status"].isin(
            ["duplicate", "unreadable", "too_small"]
        )
    ]

    if not fatal.empty:
        print("\n[FATAL] Invalid monthly inputs:")
        print(
            fatal[
                ["year", "month", "status", "selected_path"]
            ].to_string(index=False)
        )

        raise RuntimeError(
            "Fix duplicate/unreadable/too-small monthly files before processing."
        )

    # -------------------------------------------------------------------------
    # Establish the GLASS native grid from the first processable year/month.
    # This grid is NOT changed.
    # -------------------------------------------------------------------------
    first_profile = None

    for year in years:
        year_rows = completeness[
            (completeness["year"] == year)
            & (completeness["status"] == "ok")
        ]

        if not year_rows.empty:
            p = Path(year_rows.iloc[0]["selected_path"])
            _, first_profile = read_raster(p)
            break

    if first_profile is None:
        raise RuntimeError("Could not determine GLASS native grid.")

    print("\n" + "=" * 100)
    print("GLASS NATIVE GRID")
    print("=" * 100)
    print(
        "Shape:",
        first_profile["height"],
        "x",
        first_profile["width"],
    )
    print("CRS:", first_profile["crs"])
    print("Transform:", first_profile["transform"])
    print("Pixel size:", pixel_size(first_profile))

    # -------------------------------------------------------------------------
    # Derive one fixed GLASS-grid mask from the independent ERA5-Land MASTER.
    # -------------------------------------------------------------------------
    _, land_mask = derive_era5land_mask_on_glass_grid(first_profile)

    annual_rows = []
    processing_rows = []

    for year in years:
        print("\n" + "=" * 100)
        print(f"YEAR {year}")
        print("=" * 100)

        try:
            # 1. Day-weighted annual mean on unchanged GLASS native grid.
            annual_native, native_profile, info = aggregate_one_year_native(
                year,
                completeness,
            )

            processing_rows.append(info)

            if annual_native is None:
                print(
                    f"[SKIP] {year}: {info['status']} | "
                    f"{info['missing_or_invalid_months']}"
                )
                continue

            if not same_grid(native_profile, first_profile):
                raise ValueError(
                    f"{year} annual GLASS grid differs from the fixed GLASS grid. "
                    "No automatic GLASS resampling is allowed in this script."
                )

            # 2. Apply the SAME ERA5-Land-derived land mask.
            out_masked = OUT_MASKED / (
                f"GLASS_BlueSky_shortwave_annual_{year}_"
                f"native_ERA5Land_landmask.tif"
            )

            if (
                SKIP_EXISTING
                and not OVERWRITE
                and out_masked.exists()
                and out_masked.stat().st_size >= MIN_VALID_FILE_SIZE_BYTES
            ):
                annual_land, masked_profile = read_raster(out_masked)

                if not same_grid(masked_profile, native_profile):
                    print(
                        "[INFO] Existing masked raster grid differs; rebuilding."
                    )
                    annual_land = None
                else:
                    print("Reuse existing masked annual:", out_masked.name)
            else:
                annual_land = None

            if annual_land is None:
                annual_land = annual_native.copy()
                annual_land[~land_mask] = np.nan

                save_float_raster(
                    annual_land,
                    native_profile,
                    out_masked,
                )

            # 3. Global land statistics on the same fixed land definition.
            weighted_mean, weighted_n, valid_land_fraction = (
                latitude_area_weighted_mean(
                    annual_land,
                    native_profile,
                    land_mask,
                )
            )

            unweighted_mean, unweighted_n = unweighted_land_mean(
                annual_land,
                land_mask,
            )

            annual_rows.append({
                "year": year,
                "status": info["status"],
                "available_months": info["available_months"],
                "missing_or_invalid_months": info[
                    "missing_or_invalid_months"
                ],
                "available_calendar_days": info[
                    "available_calendar_days"
                ],
                "expected_calendar_days": info[
                    "expected_calendar_days"
                ],
                "analysis_grid": "GLASS_native",
                "land_mask_master_source": str(resolved_lsm),
                "land_mask_rule": f"ERA5-Land LSM > {LAND_THRESHOLD}",
                "land_mask_resampling_to_GLASS": "nearest_coordinate_lookup_periodic_lon",
                "derived_land_mask_tif": str(OUT_LAND_MASK_GLASS),
                "annual_native_tif": info["native_output"],
                "annual_native_landmask_tif": str(out_masked),
                "area_weighted_global_land_mean": weighted_mean,
                "area_weighted_valid_land_pixels": weighted_n,
                "valid_fraction_of_land_mask_percent": valid_land_fraction,
                "unweighted_global_land_mean": unweighted_mean,
                "unweighted_valid_land_pixels": unweighted_n,
            })

            pd.DataFrame(annual_rows).to_csv(
                OUT_TABLE / (
                    "GLASS_BlueSky_annual_global_land_mean_"
                    "ERA5LandMask.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )

            pd.DataFrame(processing_rows).to_csv(
                OUT_LOG / "GLASS_BlueSky_annual_processing_log.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print(
                f"{year} DONE | "
                f"weighted land mean={weighted_mean:.6f} | "
                f"valid land pixels={weighted_n:,} | "
                f"coverage={valid_land_fraction:.2f}%"
            )

            del annual_native, annual_land
            gc.collect()

        except Exception as exc:
            print(
                f"[ERROR] {year}: "
                f"{type(exc).__name__}: {exc}"
            )

            annual_rows.append({
                "year": year,
                "status": f"error: {type(exc).__name__}: {exc}",
                "available_months": np.nan,
                "missing_or_invalid_months": "",
                "available_calendar_days": np.nan,
                "expected_calendar_days": (
                    366 if calendar.isleap(year) else 365
                ),
                "analysis_grid": "GLASS_native",
                "land_mask_master_source": str(resolved_lsm),
                "land_mask_rule": f"ERA5-Land LSM > {LAND_THRESHOLD}",
                "land_mask_resampling_to_GLASS": "nearest_coordinate_lookup_periodic_lon",
                "derived_land_mask_tif": str(OUT_LAND_MASK_GLASS),
                "annual_native_tif": "",
                "annual_native_landmask_tif": "",
                "area_weighted_global_land_mean": np.nan,
                "area_weighted_valid_land_pixels": np.nan,
                "valid_fraction_of_land_mask_percent": np.nan,
                "unweighted_global_land_mean": np.nan,
                "unweighted_valid_land_pixels": np.nan,
            })

    pd.DataFrame(annual_rows).to_csv(
        OUT_TABLE / (
            "GLASS_BlueSky_annual_global_land_mean_"
            "ERA5LandMask.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(processing_rows).to_csv(
        OUT_LOG / "GLASS_BlueSky_annual_processing_log.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "#" * 100)
    print("FINISHED")
    print("#" * 100)
    print("Annual native rasters:", OUT_NATIVE)
    print("Annual ERA5-Land-masked native rasters:", OUT_MASKED)
    print("ERA5-Land fraction on GLASS grid:", OUT_LSM_FRACTION_GLASS)
    print("Common GLASS land mask:", OUT_LAND_MASK_GLASS)
    print("Mask QA:", OUT_MASK_QA)
    print(
        "Annual statistics:",
        OUT_TABLE / (
            "GLASS_BlueSky_annual_global_land_mean_"
            "ERA5LandMask.csv"
        ),
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    process_all_years()

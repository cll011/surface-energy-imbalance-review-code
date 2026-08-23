import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import statsmodels.api as sm

warnings.filterwarnings("ignore")

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_paths import get_path  # noqa: E402

# ==========================================================
# 1. Configuration
# ==========================================================
START_YEAR = 2001
END_YEAR = 2024
YEARS = list(range(START_YEAR, END_YEAR + 1))
LAND_THRESHOLD = 0.5
EPS = 1e-12

ERA5_DIR = get_path("era5_annual")
ALBEDO_DIR = get_path("glass_annual_landmask")
LSM_PATH = get_path("lsm_nc")
OUTPUT_DIR = PROJECT_ROOT / "results" / "pathway"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exact file names; no wildcard guessing.
FILE_BUILDERS = {
    "AirTemp": lambda year: ERA5_DIR / f"T2M_ERA5_{year}.tif",
    "SurfaceAlbedo": lambda year: ALBEDO_DIR / f"GLASS_BlueSky_shortwave_annual_{year}_native_ERA5Land_landmask.tif",
    "LH": lambda year: ERA5_DIR / f"LH_ERA5_{year}.tif",
    "SH": lambda year: ERA5_DIR / f"SH_ERA5_{year}.tif",
    "Rn": lambda year: ERA5_DIR / f"Rn_ERA5_{year}.tif",
    "SM": lambda year: ERA5_DIR / f"SM_ERA5_{year}.tif",
    "VPD": lambda year: ERA5_DIR / f"VPD_ERA5_{year}.tif",
}

# IMPORTANT: Do not infer flux sign from mean values.
# Keep all multipliers at +1 unless the metadata/provenance of the TIFFs
# explicitly confirms that a sign conversion is required.
SIGN_MULTIPLIER = {
    "AirTemp": 1.0,
    "SurfaceAlbedo": 1.0,
    "LH": 1.0,
    "SH": 1.0,
    "Rn": 1.0,
    "SM": 1.0,
    "VPD": 1.0,
}

# The formal pathway analysis is fitted only to the complete 2001-2024 record.
PERIODS = [
    ("Full Record", 2001, 2024),
]

# Path structure after removing Precip and TCC.
PATH_MODELS = [
    ("Rn", ["SurfaceAlbedo"]),
    ("LH", ["SM", "Rn"]),
    ("SH", ["Rn"]),
    ("AirTemp", ["SH", "LH"]),
    ("VPD", ["AirTemp"]),
    ("SM", ["VPD", "Rn"]),
]


# ==========================================================
# 2. Land-mask handling
# ==========================================================
def _find_coord_name(da, candidates):
    for name in candidates:
        if name in da.coords:
            return name
    for name in candidates:
        if name in da.dims:
            return name
    return None


def load_landmask(mask_path):
    """Load lsm.nc as a 2-D lat-lon DataArray without assuming a variable by filename."""
    if not mask_path.exists():
        raise FileNotFoundError(f"Land-mask file not found: {mask_path}")

    ds = xr.open_dataset(mask_path)

    if "lsm" in ds.data_vars:
        var_name = "lsm"
    else:
        candidates = [name for name, da in ds.data_vars.items() if da.ndim >= 2]
        if len(candidates) != 1:
            ds.close()
            raise ValueError(
                "Cannot identify the land-mask variable uniquely in lsm.nc. "
                f"Available data variables: {list(ds.data_vars)}"
            )
        var_name = candidates[0]

    da = ds[var_name].squeeze(drop=True).load()
    ds.close()

    lat_name = _find_coord_name(da, ["latitude", "lat", "y"])
    lon_name = _find_coord_name(da, ["longitude", "lon", "x"])
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Cannot identify latitude/longitude coordinates in {mask_path}. "
            f"dims={da.dims}, coords={list(da.coords)}"
        )

    extra_dims = [d for d in da.dims if d not in (lat_name, lon_name)]
    if extra_dims:
        raise ValueError(
            "Land-mask variable is not purely 2-D after squeeze. "
            f"Remaining dimensions: {extra_dims}"
        )

    da = da.transpose(lat_name, lon_name)

    # Normalize the mask longitude coordinate to [-180, 180) only when needed.
    lon = np.asarray(da[lon_name].values, dtype=float)
    if np.nanmax(lon) > 180.0:
        lon_norm = ((lon + 180.0) % 360.0) - 180.0
        da = da.assign_coords({lon_name: lon_norm}).sortby(lon_name)

    # Sorting is explicit so nearest-neighbour indexing is deterministic.
    da = da.sortby(lat_name)

    vmin = float(np.nanmin(da.values))
    vmax = float(np.nanmax(da.values))
    print(
        f"[LSM] variable={var_name}, dims={da.shape}, range=[{vmin:.4f}, {vmax:.4f}], "
        f"threshold={LAND_THRESHOLD}"
    )
    return da, lat_name, lon_name


def raster_centers(src):
    """Return 1-D longitude and latitude vectors for a north-up geographic raster."""
    if src.crs is None or not src.crs.is_geographic:
        raise ValueError(f"Raster must use a geographic CRS. File: {src.name}, CRS={src.crs}")

    t = src.transform
    if abs(t.b) > EPS or abs(t.d) > EPS:
        raise ValueError(f"Rotated rasters are not supported: {src.name}")

    xs = t.c + (np.arange(src.width) + 0.5) * t.a
    ys = t.f + (np.arange(src.height) + 0.5) * t.e
    return xs.astype(float), ys.astype(float)


def mask_for_raster(src, lsm_da, lat_name, lon_name, cache):
    """Nearest-neighbour sampling of the same lsm.nc onto each raster grid."""
    key = (
        src.width,
        src.height,
        tuple(round(v, 12) for v in tuple(src.transform)),
        src.crs.to_string() if src.crs else None,
    )
    if key in cache:
        return cache[key]

    xs, ys = raster_centers(src)
    xs_for_mask = ((xs + 180.0) % 360.0) - 180.0

    sampled = lsm_da.sel(
        {
            lat_name: xr.DataArray(ys, dims="y"),
            lon_name: xr.DataArray(xs_for_mask, dims="x"),
        },
        method="nearest",
    ).values

    land = np.isfinite(sampled) & (sampled >= LAND_THRESHOLD)
    cache[key] = land
    return land


# ==========================================================
# 3. Annual global land mean extraction
# ==========================================================
def area_weighted_land_mean(tif_path, lsm_da, lat_name, lon_name, mask_cache):
    if not tif_path.exists():
        raise FileNotFoundError(f"Missing input TIFF: {tif_path}")

    with rasterio.open(tif_path) as src:
        arr = src.read(1, masked=True).astype("float64").filled(np.nan)
        arr[~np.isfinite(arr)] = np.nan

        land_mask = mask_for_raster(src, lsm_da, lat_name, lon_name, mask_cache)
        if land_mask.shape != arr.shape:
            raise ValueError(
                f"Mask/raster shape mismatch for {tif_path}: mask={land_mask.shape}, raster={arr.shape}"
            )

        _, lats = raster_centers(src)
        row_weights = np.cos(np.deg2rad(lats))
        row_weights = np.clip(row_weights, 0.0, None)

        valid = land_mask & np.isfinite(arr)
        if not np.any(valid):
            raise ValueError(f"No valid land pixels after masking: {tif_path}")

        weights_2d = np.broadcast_to(row_weights[:, None], arr.shape)
        denominator = np.sum(weights_2d[valid])
        numerator = np.sum(arr[valid] * weights_2d[valid])

        if denominator <= 0:
            raise ValueError(f"Invalid area-weight denominator for {tif_path}")

        return numerator / denominator, int(valid.sum())


def build_annual_dataframe():
    print("⏳ Extracting annual area-weighted global land means...")
    lsm_da, lat_name, lon_name = load_landmask(LSM_PATH)
    mask_cache = {}

    records = []
    pixel_records = []

    for year in YEARS:
        row = {"Year": year}
        prow = {"Year": year}

        for var, builder in FILE_BUILDERS.items():
            tif_path = builder(year)
            mean_value, n_valid = area_weighted_land_mean(
                tif_path, lsm_da, lat_name, lon_name, mask_cache
            )
            row[var] = mean_value * SIGN_MULTIPLIER[var]
            prow[var] = n_valid

        records.append(row)
        pixel_records.append(prow)
        print(f"  ✓ {year}")

    df = pd.DataFrame(records).set_index("Year")
    n_pixels = pd.DataFrame(pixel_records).set_index("Year")

    # Fail rather than interpolate/fill missing values.
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"NaN remains in annual global means for variables: {bad}")

    return df, n_pixels


# ==========================================================
# 4. Standardized path analysis
# ==========================================================
def zscore_within_period(df_period):
    out = pd.DataFrame(index=df_period.index)
    for col in df_period.columns:
        x = df_period[col].astype(float)
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd <= EPS:
            raise ValueError(f"Variable '{col}' has zero/invalid variance in this period.")
        out[col] = (x - x.mean()) / sd
    return out


def significance_label(p):
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fit_path_model(df_std, y_col, x_cols, period_name, start_year, end_year):
    model_df = df_std[[y_col] + x_cols].dropna()
    n = len(model_df)
    k = len(x_cols)

    if n <= k + 1:
        raise ValueError(
            f"Insufficient observations for {y_col} ~ {' + '.join(x_cols)}: n={n}, predictors={k}."
        )

    X = sm.add_constant(model_df[x_cols], has_constant="add")
    model = sm.OLS(model_df[y_col], X).fit()

    print(
        f"\n[Model] {y_col} ~ {' + '.join(x_cols)} | "
        f"n={n}, R²={model.rsquared:.3f}, Adj.R²={model.rsquared_adj:.3f}"
    )

    rows = []
    conf = model.conf_int(alpha=0.05)
    for x in x_cols:
        beta = float(model.params[x])
        p = float(model.pvalues[x])
        se = float(model.bse[x])
        ci_low = float(conf.loc[x, 0])
        ci_high = float(conf.loc[x, 1])
        sig = significance_label(p)

        print(
            f"    β ({x:<15} -> {y_col:<10}) = {beta:>8.3f}  "
            f"SE={se:.3f}, p={p:.4g} ({sig})"
        )

        rows.append(
            {
                "Period": period_name,
                "StartYear": start_year,
                "EndYear": end_year,
                "N": n,
                "Outcome": y_col,
                "Predictor": x,
                "Beta_std": beta,
                "SE": se,
                "CI95_low": ci_low,
                "CI95_high": ci_high,
                "p_value": p,
                "Significance": sig,
                "R2": float(model.rsquared),
                "Adj_R2": float(model.rsquared_adj),
                "AIC": float(model.aic),
                "BIC": float(model.bic),
                "Df_resid": float(model.df_resid),
            }
        )

    return rows


def run_path_analysis_for_period(df_master, start_year, end_year, period_name):
    print("\n" + "=" * 72)
    print(f"📊 Period: {period_name} ({start_year}-{end_year})")
    print("=" * 72)

    df_period = df_master.loc[start_year:end_year].copy()
    expected_n = end_year - start_year + 1
    if len(df_period) != expected_n:
        raise ValueError(
            f"Period {period_name} expected {expected_n} years but found {len(df_period)}."
        )

    if len(df_period) <= 5:
        print(
            f"⚠ Small-sample warning: n={len(df_period)}. Standardized coefficients can be computed, "
            "but p-values, confidence intervals and model ranking are highly unstable."
        )

    df_std = zscore_within_period(df_period)

    all_rows = []
    for y_col, x_cols in PATH_MODELS:
        all_rows.extend(
            fit_path_model(
                df_std=df_std,
                y_col=y_col,
                x_cols=x_cols,
                period_name=period_name,
                start_year=start_year,
                end_year=end_year,
            )
        )
    return all_rows


# ==========================================================
# 5. Main
# ==========================================================
def main():
    df, n_pixels = build_annual_dataframe()

    annual_csv = OUTPUT_DIR / "annual_global_land_means_2001_2024.csv"
    pixels_csv = OUTPUT_DIR / "annual_valid_land_pixel_counts_2001_2024.csv"
    df.to_csv(annual_csv, encoding="utf-8-sig")
    n_pixels.to_csv(pixels_csv, encoding="utf-8-sig")

    print("\nAnnual global land means:")
    print(df.round(6))

    # Basic QC only; no automatic sign changes.
    print("\nVariable ranges for sign/unit QC:")
    qc = pd.DataFrame(
        {
            "min": df.min(),
            "mean": df.mean(),
            "max": df.max(),
            "std": df.std(ddof=1),
        }
    )
    print(qc.round(6))
    qc.to_csv(OUTPUT_DIR / "annual_series_QC.csv", encoding="utf-8-sig")

    results = []
    for period_name, start_year, end_year in PERIODS:
        results.extend(
            run_path_analysis_for_period(df, start_year, end_year, period_name)
        )

    results_df = pd.DataFrame(results)
    results_csv = OUTPUT_DIR / "path_analysis_standardized_results.csv"
    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("✅ Completed")
    print(f"Annual means: {annual_csv}")
    print(f"Path results: {results_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()

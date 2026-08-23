# -*- coding: utf-8 -*-
"""
Rebuild updated Fig. 2 time-series diagnostics from annual GLASS blue-sky albedo
and ERA5 2-m air temperature (T2M), 2001–2024.

Inputs
------
1) Updated annual GLASS blue-sky albedo:
   D:/10_Research/01_Datasets/02_DataProcess/
   03_SurfaceAlbedo_GLASS/blueSky_annual_updated

   The script searches recursively for files containing:
       GLASS_BlueSky_shortwave_annual_{year}*.tif

   If several annual rasters exist for one year, preference is:
       aligned + landmask > panelgrid/modisgrid > landmask > native

2) ERA5 annual T2M:
   D:/10_Research/01_Datasets/01_DataRaw/ERA5/Annual_Tif
       T2M_ERA5_{year}.tif

Main outputs
------------
A. Fig2a_T2M_GLASS_annual_timeseries_quadratic
   Annual global-land T2M and GLASS albedo + independent quadratic fits.

B. Fig2b_T2M_and_GLASS_own_changepoints_reference_style
   Independent changepoint segmentation for T2M and GLASS albedo.
   Changepoints are re-estimated from the updated series; no old breakpoint
   years are hard-coded.

C. Fig2c_first_order_rate_second_order_trend
   First differences:
       dT2M/dt
       -d(GLASS albedo)/dt
   standardized independently as z scores, with quadratic fits.

Scientific / processing choices
-------------------------------
- All T2M rasters are aligned to the GLASS analysis grid before global means.
- One fixed land mask is used for all years.
- If a common mask exists under the updated GLASS directory, it is used.
  Otherwise the script falls back to the finite footprint of the reference
  GLASS annual raster and prints a warning.
- Global land means use cos(latitude) weighting on a geographic lon-lat grid.
- ERA5 T2M is converted from K to °C automatically when values indicate Kelvin.
- Changepoint default: BIC selection among 0, 1 and 2 breakpoints with a
  minimum segment length. To force exactly two breakpoints, set
  CHANGEPOINT_MODE = "force_2".
- No old 2006/2012/2015/2020 breakpoint values are inserted manually.

Python
------
Compatible with Python 3.8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import gc
import math
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_paths import get_path  # noqa: E402


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

START_YEAR = 2001
END_YEAR = 2024
YEARS = list(range(START_YEAR, END_YEAR + 1))

ALBEDO_ROOT = get_path("glass_annual")
T2M_ROOT = get_path("era5_annual")
OUT_ROOT = PROJECT_ROOT / "results" / "fig2_temporal"

# Requested deliverables: independent Fig. 2a and Fig. 2b panels for direct
# manuscript replacement.  The output does not assemble a multi-panel figure.
OUT_FIG = PROJECT_ROOT / "results" / "main"
OUT_TABLE = OUT_ROOT / "source_data"
OUT_LOG = OUT_ROOT / "logs"

for _d in [OUT_FIG, OUT_TABLE, OUT_LOG]:
    _d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Fixed common land mask.
# None = auto search under ALBEDO_ROOT:
#   **/common_land_mask_on_target_grid.tif
# If absent, finite footprint of the reference GLASS raster is used.
# -----------------------------------------------------------------------------
LAND_MASK_RASTER = None

# Preferred reference year for fixed analysis geometry.
REFERENCE_YEAR = 2020

# GLASS valid range
ALBEDO_MIN = 0.0
ALBEDO_MAX = 1.0

# T2M valid range after conversion to Celsius.
T2M_MIN_C = -80.0
T2M_MAX_C = 60.0

# Require at least this fraction of fixed common-land mask cells in each year.
MIN_SPATIAL_COVERAGE = 0.80

# -----------------------------------------------------------------------------
# Changepoint model
# -----------------------------------------------------------------------------
# "bic_up_to_2" = scientifically preferred: choose 0, 1 or 2 breaks by BIC.
# "force_2"     = always return exactly 2 breaks, for strict visual comparison.
CHANGEPOINT_MODE = "bic_up_to_2"
MIN_SEGMENT_YEARS = 4

# -----------------------------------------------------------------------------
# Figure settings
# -----------------------------------------------------------------------------
DPI = 600
FIG_FORMATS = ("png", "pdf", "svg")

# The manuscript Fig. 2 is a five-panel composition.  This script updates only
# panels a and b from the revised annual GLASS and ERA5 series; panels c-e are
# retained exactly as supplied in the current manuscript figure.
MANUSCRIPT_FIG_DIR = PROJECT_ROOT / "reference_outputs" / "main"
MANUSCRIPT_FIG_REFERENCE = (
    MANUSCRIPT_FIG_DIR
    / "Fig2_surface_albedo_decline_land_warming_embedded_regional.png"
)
MANUSCRIPT_FIG_UPDATED_STEM = (
    "Fig2_surface_albedo_decline_land_warming_redrawn_updated"
)
MANUSCRIPT_FIG_DPI = 1200
MANUSCRIPT_FIG_SIZE_MM = (183.0, 168.0)

# Period guides used only as manuscript period context, NOT as estimated breaks.
P1_END = 2014
P2_END = 2019

# Keep figures clean and Nature-like.
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =============================================================================
# 2. FILE DISCOVERY
# =============================================================================

def _annual_albedo_score(path: Path) -> Tuple[int, int, str]:
    """
    Larger score = preferred when multiple files exist for the same year.
    """
    name = path.name.lower()
    score = 0

    if "aligned" in name:
        score += 50
    if "landmask" in name or "land_mask" in name:
        score += 40
    if "panelgrid" in name or "panel_grid" in name:
        score += 30
    if "modisgrid" in name or "modis_grid" in name:
        score += 20
    if "native" in name:
        score -= 10

    return score, -len(str(path)), str(path)


def find_albedo_file(year: int) -> Path:
    if not ALBEDO_ROOT.exists():
        raise FileNotFoundError(f"ALBEDO_ROOT does not exist:\n{ALBEDO_ROOT}")

    exact = list(
        ALBEDO_ROOT.rglob(
            f"GLASS_BlueSky_shortwave_annual_{year}*.tif"
        )
    )

    exact = [p for p in exact if p.is_file()]

    if not exact:
        # Broader fallback for legacy naming.
        broad = []
        for p in ALBEDO_ROOT.rglob("*.tif"):
            low = p.name.lower()
            if (
                str(year) in low
                and "glass" in low
                and "bluesky" in low
                and "annual" in low
            ):
                broad.append(p)
        exact = broad

    if not exact:
        raise FileNotFoundError(
            f"No updated GLASS annual raster found for {year} under:\n"
            f"{ALBEDO_ROOT}"
        )

    exact = sorted(exact, key=_annual_albedo_score, reverse=True)
    return exact[0]


def find_t2m_file(year: int) -> Path:
    candidates = [
        T2M_ROOT / f"T2M_ERA5_{year}.tif",
        T2M_ROOT / f"T2M_ERA5_{year}.TIF",
    ]

    for p in candidates:
        if p.exists():
            return p

    hits = sorted(T2M_ROOT.glob(f"*T2M*ERA5*{year}*.tif"))
    if not hits:
        hits = sorted(T2M_ROOT.glob(f"*T2M*{year}*.tif"))

    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"No ERA5 T2M raster found for {year} under:\n{T2M_ROOT}"
    )


def discover_common_mask() -> Optional[Path]:
    if LAND_MASK_RASTER is not None:
        p = Path(LAND_MASK_RASTER)
        if not p.exists():
            raise FileNotFoundError(f"LAND_MASK_RASTER not found:\n{p}")
        return p

    patterns = [
        "common_land_mask_on_target_grid.tif",
        "*common*land*mask*.tif",
    ]

    for pattern in patterns:
        hits = sorted(ALBEDO_ROOT.rglob(pattern))
        if hits:
            return hits[0]

    return None


# =============================================================================
# 3. RASTER HELPERS
# =============================================================================

def read_raster(path: Path) -> Tuple[np.ndarray, Dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        arr[arr < -1e20] = np.nan

    return arr, profile


def same_grid(a: Dict, b: Dict) -> bool:
    try:
        if a["width"] != b["width"] or a["height"] != b["height"]:
            return False
        if a.get("crs") != b.get("crs"):
            return False

        ta = a["transform"]
        tb = b["transform"]

        return all(
            abs(float(x) - float(y)) < 1e-10
            for x, y in zip(ta, tb)
        )
    except Exception:
        return False


def reproject_to_profile(
    src_arr: np.ndarray,
    src_profile: Dict,
    dst_profile: Dict,
    resampling: Resampling,
) -> np.ndarray:
    if same_grid(src_profile, dst_profile):
        return src_arr.astype("float32", copy=True)

    dst = np.full(
        (dst_profile["height"], dst_profile["width"]),
        np.nan,
        dtype="float32",
    )

    reproject(
        source=src_arr.astype("float32"),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling,
    )

    return dst


def normalize_albedo(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    arr[(arr < ALBEDO_MIN) | (arr > ALBEDO_MAX)] = np.nan
    return arr


def normalize_t2m_to_celsius(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    arr = arr.astype("float32", copy=False)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, "unknown_empty"

    median = float(np.nanmedian(finite))

    if median > 100.0:
        arr = arr - np.float32(273.15)
        unit_method = "Kelvin_to_Celsius"
    else:
        unit_method = "already_Celsius"

    arr[(arr < T2M_MIN_C) | (arr > T2M_MAX_C)] = np.nan
    return arr, unit_method


def require_geographic_grid(profile: Dict):
    crs = profile.get("crs")

    if crs is None:
        raise ValueError("Reference grid CRS is missing.")

    if hasattr(crs, "is_geographic") and not crs.is_geographic:
        raise ValueError(
            f"Reference grid is projected ({crs}). "
            "This script uses cos(latitude) weights and therefore requires "
            "a geographic lon-lat analysis grid."
        )


def row_latitudes(profile: Dict) -> np.ndarray:
    transform = profile["transform"]
    rows = np.arange(profile["height"], dtype="float64")
    return np.asarray(
        [(transform * (0.5, r + 0.5))[1] for r in rows],
        dtype="float64",
    )


def latitude_weighted_mean(
    arr: np.ndarray,
    profile: Dict,
    fixed_land_mask: np.ndarray,
) -> Tuple[float, int, float]:
    require_geographic_grid(profile)

    if arr.shape != fixed_land_mask.shape:
        raise ValueError(
            f"Array/mask shape mismatch: {arr.shape} vs {fixed_land_mask.shape}"
        )

    valid = np.isfinite(arr) & fixed_land_mask

    land_n = int(np.count_nonzero(fixed_land_mask))
    valid_n = int(np.count_nonzero(valid))

    if valid_n == 0:
        return np.nan, 0, 0.0

    lats = row_latitudes(profile)
    row_w = np.cos(np.deg2rad(lats))

    numerator = 0.0
    denominator = 0.0

    for r in range(arr.shape[0]):
        vr = valid[r]
        n = int(np.count_nonzero(vr))
        if n == 0:
            continue

        w = float(row_w[r])
        vals = arr[r, vr].astype("float64")

        numerator += float(np.sum(vals) * w)
        denominator += float(n * w)

    mean_value = numerator / denominator
    coverage = valid_n / land_n if land_n > 0 else np.nan

    return float(mean_value), valid_n, float(coverage)


# =============================================================================
# 4. BUILD FIXED ANALYSIS GRID + MASK
# =============================================================================

def get_reference_albedo() -> Tuple[Path, np.ndarray, Dict]:
    if REFERENCE_YEAR not in YEARS:
        raise ValueError("REFERENCE_YEAR must fall inside requested YEARS.")

    p = find_albedo_file(REFERENCE_YEAR)
    arr, profile = read_raster(p)
    arr = normalize_albedo(arr)

    require_geographic_grid(profile)

    return p, arr, profile


def build_fixed_land_mask(
    reference_arr: np.ndarray,
    reference_profile: Dict,
) -> Tuple[np.ndarray, str]:
    mask_path = discover_common_mask()

    if mask_path is None:
        warnings.warn(
            "\nNo common_land_mask_on_target_grid.tif was found under "
            "ALBEDO_ROOT. Falling back to the finite footprint of the "
            f"{REFERENCE_YEAR} updated GLASS raster. For final manuscript "
            "analysis, a dedicated fixed common land/water mask is preferable."
        )
        mask = np.isfinite(reference_arr)
        return mask, (
            f"finite_footprint_of_{REFERENCE_YEAR}_GLASS:"
            f"{find_albedo_file(REFERENCE_YEAR)}"
        )

    mask_arr, mask_profile = read_raster(mask_path)

    if not same_grid(mask_profile, reference_profile):
        binary = np.where(np.isfinite(mask_arr) & (mask_arr > 0.5), 1.0, 0.0)
        mask_arr = reproject_to_profile(
            binary.astype("float32"),
            mask_profile,
            reference_profile,
            Resampling.nearest,
        )

    mask = np.isfinite(mask_arr) & (mask_arr > 0.5)

    if np.count_nonzero(mask) == 0:
        raise RuntimeError(f"Common mask contains no valid land cells:\n{mask_path}")

    return mask, str(mask_path)


# =============================================================================
# 5. GLOBAL ANNUAL SERIES
# =============================================================================

def build_global_annual_series() -> pd.DataFrame:
    ref_path, ref_arr, ref_profile = get_reference_albedo()
    fixed_mask, mask_source = build_fixed_land_mask(ref_arr, ref_profile)

    print("\n" + "=" * 100)
    print("REFERENCE GRID / COMMON MASK")
    print("=" * 100)
    print("Reference GLASS raster:", ref_path)
    print("Grid:", ref_profile["width"], "x", ref_profile["height"])
    print("CRS:", ref_profile["crs"])
    print("Mask source:", mask_source)
    print("Common land cells:", int(np.count_nonzero(fixed_mask)))

    rows = []

    for year in YEARS:
        print(f"\nProcessing {year} ...")

        albedo_path = find_albedo_file(year)
        t2m_path = find_t2m_file(year)

        albedo, albedo_profile = read_raster(albedo_path)
        albedo = normalize_albedo(albedo)

        # All updated GLASS annual products are expected to share the final grid.
        if not same_grid(albedo_profile, ref_profile):
            print(
                "  GLASS grid differs from reference; reprojecting to common grid."
            )
            albedo = reproject_to_profile(
                albedo,
                albedo_profile,
                ref_profile,
                Resampling.average,
            )
            albedo = normalize_albedo(albedo)

        t2m, t2m_profile = read_raster(t2m_path)
        t2m, unit_method = normalize_t2m_to_celsius(t2m)

        if not same_grid(t2m_profile, ref_profile):
            t2m = reproject_to_profile(
                t2m,
                t2m_profile,
                ref_profile,
                Resampling.bilinear,
            )
            t2m, _ = normalize_t2m_to_celsius(t2m)

        albedo_mean, albedo_n, albedo_cov = latitude_weighted_mean(
            albedo,
            ref_profile,
            fixed_mask,
        )
        t2m_mean, t2m_n, t2m_cov = latitude_weighted_mean(
            t2m,
            ref_profile,
            fixed_mask,
        )

        if albedo_cov < MIN_SPATIAL_COVERAGE:
            raise RuntimeError(
                f"{year} GLASS coverage is only {albedo_cov:.1%}, below "
                f"MIN_SPATIAL_COVERAGE={MIN_SPATIAL_COVERAGE:.0%}."
            )

        if t2m_cov < MIN_SPATIAL_COVERAGE:
            raise RuntimeError(
                f"{year} T2M coverage is only {t2m_cov:.1%}, below "
                f"MIN_SPATIAL_COVERAGE={MIN_SPATIAL_COVERAGE:.0%}."
            )

        rows.append({
            "Year": year,
            "T2M_C": t2m_mean,
            "GLASS_Albedo": albedo_mean,
            "T2M_valid_land_pixels": t2m_n,
            "GLASS_valid_land_pixels": albedo_n,
            "T2M_common_land_coverage": t2m_cov,
            "GLASS_common_land_coverage": albedo_cov,
            "T2M_unit_handling": unit_method,
            "T2M_file": str(t2m_path),
            "GLASS_file": str(albedo_path),
            "common_mask_source": mask_source,
            "reference_grid": str(ref_path),
        })

        print(
            f"  T2M={t2m_mean:.4f} °C | "
            f"GLASS={albedo_mean:.6f} | "
            f"coverage T2M/GLASS={t2m_cov:.1%}/{albedo_cov:.1%}"
        )

        del albedo, t2m
        gc.collect()

    df = pd.DataFrame(rows)
    out_csv = OUT_TABLE / "Fig2_updated_global_land_annual_series.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return df


# =============================================================================
# 6. FITTING HELPERS
# =============================================================================

def quadratic_fit(
    years: Sequence[float],
    values: Sequence[float],
) -> Dict[str, object]:
    x = np.asarray(years, dtype="float64")
    y = np.asarray(values, dtype="float64")

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    # Center year for numerical stability.
    x0 = float(np.mean(x))
    xc = x - x0

    coeff = np.polyfit(xc, y, 2)
    pred = np.polyval(coeff, xc)

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 400)
    y_dense = np.polyval(coeff, x_dense - x0)

    return {
        "center_year": x0,
        "coef_quadratic": float(coeff[0]),
        "coef_linear": float(coeff[1]),
        "intercept": float(coeff[2]),
        "r2": r2,
        "x_dense": x_dense,
        "y_dense": y_dense,
    }


def zscore(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype="float64")
    out = np.full(a.shape, np.nan, dtype="float64")

    valid = np.isfinite(a)
    if np.count_nonzero(valid) < 2:
        return out

    mu = float(np.mean(a[valid]))
    sd = float(np.std(a[valid], ddof=1))

    if sd == 0:
        out[valid] = 0.0
    else:
        out[valid] = (a[valid] - mu) / sd

    return out


# =============================================================================
# 7. CHANGEPOINT SEGMENTATION
# =============================================================================

def segment_sse(y: np.ndarray, bounds: List[Tuple[int, int]]) -> float:
    sse = 0.0
    for start, end in bounds:
        seg = y[start:end]
        if seg.size == 0:
            return np.inf
        mu = float(np.mean(seg))
        sse += float(np.sum((seg - mu) ** 2))
    return sse


def candidate_model(
    years: np.ndarray,
    y: np.ndarray,
    breaks_idx: List[int],
) -> Dict[str, object]:
    n = len(y)

    starts = [0] + breaks_idx
    ends = breaks_idx + [n]
    bounds = list(zip(starts, ends))

    sse = segment_sse(y, bounds)
    k = len(breaks_idx)

    # Count means + break locations as fitted parameters.
    p = (k + 1) + k
    mse = max(sse / n, np.finfo(float).tiny)
    bic = n * np.log(mse) + p * np.log(n)

    means = [float(np.mean(y[s:e])) for s, e in bounds]
    break_years = [int(years[i]) for i in breaks_idx]

    return {
        "n_breaks": k,
        "break_indices": breaks_idx,
        "break_years": break_years,
        "bounds": bounds,
        "segment_means": means,
        "sse": float(sse),
        "bic": float(bic),
    }


def fit_changepoints(
    years: Sequence[int],
    values: Sequence[float],
    min_segment: int = 4,
    mode: str = "bic_up_to_2",
) -> Tuple[Dict[str, object], pd.DataFrame]:
    x = np.asarray(years, dtype=int)
    y = np.asarray(values, dtype="float64")

    valid = np.isfinite(y)
    x = x[valid]
    y = y[valid]

    n = len(y)

    if n < 2 * min_segment:
        raise ValueError("Too few valid years for changepoint analysis.")

    models = []

    # 0 breaks
    models.append(candidate_model(x, y, []))

    # 1 break
    best1 = None
    for i in range(min_segment, n - min_segment + 1):
        m = candidate_model(x, y, [i])
        if best1 is None or m["sse"] < best1["sse"]:
            best1 = m
    if best1 is not None:
        models.append(best1)

    # 2 breaks
    best2 = None
    if n >= 3 * min_segment:
        for i in range(min_segment, n - 2 * min_segment + 1):
            for j in range(i + min_segment, n - min_segment + 1):
                m = candidate_model(x, y, [i, j])
                if best2 is None or m["sse"] < best2["sse"]:
                    best2 = m
        if best2 is not None:
            models.append(best2)

    model_table = pd.DataFrame([
        {
            "n_breaks": m["n_breaks"],
            "break_years": ",".join(map(str, m["break_years"])),
            "SSE": m["sse"],
            "BIC": m["bic"],
        }
        for m in models
    ])

    if mode == "force_2":
        selected = [m for m in models if m["n_breaks"] == 2]
        if not selected:
            raise RuntimeError("Cannot fit two breakpoints with current settings.")
        best = selected[0]
    elif mode == "bic_up_to_2":
        best = min(models, key=lambda m: m["bic"])
    else:
        raise ValueError(
            'CHANGEPOINT_MODE must be "bic_up_to_2" or "force_2".'
        )

    return best, model_table


# =============================================================================
# 8. FIGURE UTILITIES
# =============================================================================

def panel_label(ax, label: str):
    ax.text(
        -0.08, 1.04, label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(axis="y", linewidth=0.45, alpha=0.25)


def save_figure(fig, stem: str):
    for fmt in FIG_FORMATS:
        out = OUT_FIG / f"{stem}.{fmt}"
        if fmt == "png":
            fig.savefig(out, dpi=DPI)
        else:
            fig.savefig(out)


def add_period_guides(ax, y_top_fraction=0.97):
    ax.axvline(2015, linestyle="--", linewidth=0.8, alpha=0.55)
    ax.axvline(2020, linestyle="--", linewidth=0.8, alpha=0.55)

    ylim = ax.get_ylim()
    y = ylim[0] + y_top_fraction * (ylim[1] - ylim[0])

    ax.text(2008, y, "Stable", ha="center", va="top")
    ax.text(2017, y, "Transition", ha="center", va="top")
    ax.text(2022, y, "Warming", ha="center", va="top")


def _draw_manuscript_time_series_panel(
    fig: plt.Figure,
    spec,
    df: pd.DataFrame,
) -> None:
    """Redraw Fig. 2a using the original manuscript panel geometry."""
    years = df["Year"].to_numpy(dtype=float)
    temp = df["T2M_C"].to_numpy(dtype=float)
    albedo = df["GLASS_Albedo"].to_numpy(dtype=float)

    fit_t = quadratic_fit(years, temp)
    fit_a = quadratic_fit(years, albedo)

    t2m_colour = "#B33A3A"
    albedo_colour = "#6BAED6"
    break_colour = "#6F737A"

    ax_t = fig.add_subplot(spec, zorder=2)
    ax_a = ax_t.twinx()
    ax_t.set_facecolor("white")
    ax_a.set_facecolor("none")

    for ax in (ax_t, ax_a):
        ax.axvspan(2015, 2019.98, color="#D8A24A", alpha=0.10, lw=0, zorder=0)
        ax.axvspan(2020, 2024.4, color="#C24B4B", alpha=0.08, lw=0, zorder=0)
        ax.axvline(2015, color=break_colour, lw=0.75, ls=(0, (3.2, 2.2)), zorder=1)
        ax.axvline(2020, color=break_colour, lw=0.75, ls=(0, (3.2, 2.2)), zorder=1)

    ax_t.grid(axis="y", color="#E5E7EB", linewidth=0.55)
    ax_a.grid(False)

    t_line, = ax_t.plot(
        years, temp, color=t2m_colour, lw=1.05, marker="^", ms=3.2,
        markerfacecolor=t2m_colour, markeredgecolor=t2m_colour, alpha=0.88,
        zorder=4, label="ERA5 T2M",
    )
    t_fit, = ax_t.plot(
        fit_t["x_dense"], fit_t["y_dense"], color=t2m_colour, lw=2.0,
        zorder=5, label="ERA5 T2M quadratic",
    )
    a_line, = ax_a.plot(
        years, albedo, color=albedo_colour, lw=1.05, marker="o", ms=3.1,
        markerfacecolor=albedo_colour, markeredgecolor=albedo_colour, alpha=0.80,
        zorder=4, label="GLASS albedo",
    )
    a_fit, = ax_a.plot(
        fit_a["x_dense"], fit_a["y_dense"], color=albedo_colour, lw=2.0,
        alpha=0.95, zorder=5, label="GLASS albedo quadratic",
    )

    ax_t.set_ylabel("ERA5 T2M (°C)", color=t2m_colour, fontweight="bold", labelpad=1)
    ax_a.set_ylabel("GLASS albedo", color=albedo_colour, fontweight="bold", labelpad=1)
    ax_t.tick_params(axis="y", colors=t2m_colour)
    ax_a.tick_params(axis="y", colors=albedo_colour)
    ax_a.spines["right"].set_visible(True)
    ax_a.spines["right"].set_color(albedo_colour)
    ax_t.spines["top"].set_visible(False)
    ax_a.spines["top"].set_visible(False)
    ax_t.set_title("T2M and GLASS albedo", pad=3)
    panel_label(ax_t, "a")
    ax_t.set_xlabel("Year")
    ax_t.set_xlim(2000.5, 2024.5)
    ax_t.set_xticks([2001, 2006, 2011, 2016, 2021, 2024])
    ax_t.set_ylim(np.nanmin(temp) - 0.15, np.nanmax(temp) + 0.20)
    ax_a.set_ylim(np.nanmin(albedo) - 0.00055, np.nanmax(albedo) + 0.00055)
    ax_t.legend(
        handles=[t_line, t_fit, a_line, a_fit],
        labels=[h.get_label() for h in [t_line, t_fit, a_line, a_fit]],
        loc="upper center", bbox_to_anchor=(0.54, 0.995), ncol=2,
        frameon=False, handlelength=1.55, columnspacing=0.9, borderpad=0.1,
        labelspacing=0.25, fontsize=5.4,
    )


def _draw_manuscript_breakpoint_panel(
    fig: plt.Figure,
    spec,
    df: pd.DataFrame,
    temp_model: Dict[str, object],
    albedo_model: Dict[str, object],
) -> None:
    """Redraw Fig. 2b from the BIC-selected updated changepoint models."""
    years = df["Year"].to_numpy(dtype=int)
    temp = df["T2M_C"].to_numpy(dtype=float)
    albedo = df["GLASS_Albedo"].to_numpy(dtype=float)

    t2m_colour = "#B33A3A"
    albedo_colour = "#2C7FB8"
    break_colour = "#676B72"

    sub = spec.subgridspec(2, 1, hspace=0.08)
    ax_t = fig.add_subplot(sub[0, 0], zorder=2)
    ax_a = fig.add_subplot(sub[1, 0], sharex=ax_t, zorder=2)

    def draw_segments(
        ax: plt.Axes,
        values: np.ndarray,
        model: Dict[str, object],
        colour: str,
        ylabel: str,
        decimals: int,
    ) -> None:
        ax.set_facecolor("white")
        ax.plot(years, values, color=colour, lw=0.75, marker="o", ms=1.8, alpha=0.65)
        for breakpoint in model["break_years"]:
            ax.axvline(breakpoint, color=break_colour, lw=0.65, ls=(0, (3, 2)))
        for (start, end), mean in zip(model["bounds"], model["segment_means"]):
            x0 = years[start]
            x1 = years[end - 1]
            ax.hlines(mean, x0, x1, color=colour, lw=2.0)
            ax.text(
                (x0 + x1) / 2.0, mean, f"{mean:.{decimals}f}",
                color=colour, ha="center", va="bottom", fontsize=5.4,
                fontweight="bold",
            )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    draw_segments(ax_t, temp, temp_model, t2m_colour, "T2M (°C)", 2)
    draw_segments(ax_a, albedo, albedo_model, albedo_colour, "GLASS albedo", 3)
    ax_t.set_title("Breakpoint segmentation and period means", pad=3)
    panel_label(ax_t, "b")
    plt.setp(ax_t.get_xticklabels(), visible=False)
    ax_a.set_xlabel("Year")
    ax_a.set_xlim(2000.5, 2024.5)
    ax_a.set_xticks([2001, 2006, 2011, 2016, 2021, 2024])

    def break_text(label: str, model: Dict[str, object]) -> str:
        years_text = ", ".join(str(v) for v in model["break_years"])
        return f"{label} BIC break{'s' if len(model['break_years']) != 1 else ''}: {years_text or 'none'}"

    ax_t.text(0.02, 0.88, break_text("T2M", temp_model), transform=ax_t.transAxes, fontsize=5.5)
    ax_a.text(0.02, 0.12, break_text("Albedo", albedo_model), transform=ax_a.transAxes, fontsize=5.5)


def plot_manuscript_fig2_ab_updated(df: pd.DataFrame) -> None:
    """Write the manuscript Fig. 2 with updated a/b and unchanged c-e panels."""
    if not MANUSCRIPT_FIG_REFERENCE.exists():
        raise FileNotFoundError(
            "Current manuscript Fig. 2 reference PNG is required to retain "
            f"unchanged panels c-e:\n{MANUSCRIPT_FIG_REFERENCE}"
        )

    temp_model, _ = fit_changepoints(
        df["Year"].to_numpy(dtype=int), df["T2M_C"].to_numpy(dtype=float),
        min_segment=MIN_SEGMENT_YEARS, mode=CHANGEPOINT_MODE,
    )
    albedo_model, _ = fit_changepoints(
        df["Year"].to_numpy(dtype=int), df["GLASS_Albedo"].to_numpy(dtype=float),
        min_segment=MIN_SEGMENT_YEARS, mode=CHANGEPOINT_MODE,
    )

    reference = plt.imread(MANUSCRIPT_FIG_REFERENCE)
    width_mm, height_mm = MANUSCRIPT_FIG_SIZE_MM
    expected_size = (
        round(width_mm / 25.4 * MANUSCRIPT_FIG_DPI),
        round(height_mm / 25.4 * MANUSCRIPT_FIG_DPI),
    )
    if (
        abs(reference.shape[1] - expected_size[0]) > 1
        or abs(reference.shape[0] - expected_size[1]) > 1
    ):
        raise RuntimeError(
            "The reference figure dimensions do not match the locked manuscript "
            f"canvas: got {reference.shape[1]}x{reference.shape[0]}, expected "
            f"{expected_size[0]}x{expected_size[1]}."
        )

    # Lock the render canvas to the actual supplied manuscript PNG.  The
    # nominal 183 x 168 mm export can differ by one pixel after TIFF/PNG DPI
    # rounding, whereas an exact pixel match preserves panels c-e unchanged.
    fig = plt.figure(
        figsize=(
            reference.shape[1] / MANUSCRIPT_FIG_DPI,
            reference.shape[0] / MANUSCRIPT_FIG_DPI,
        ),
        dpi=MANUSCRIPT_FIG_DPI,
    )
    # Matplotlib's figure-image coordinates originate at the lower edge of the
    # canvas.  Using ``origin='lower'`` preserves the top-to-bottom ordering of
    # the supplied manuscript PNG when it is placed at ``yo=0``.
    fig.figimage(reference, xo=0, yo=0, origin="lower", zorder=-10)

    # Mask only the first-row panels before redrawing them. This preserves the
    # original panel c-e pixels and their manually checked map aspect ratios.
    mask_axis = fig.add_axes([0.0, 0.715, 1.0, 0.285], zorder=0)
    mask_axis.set_facecolor("white")
    mask_axis.set_axis_off()

    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.04, 1.38, 1.05], width_ratios=[1.0, 1.0],
        left=0.105, right=0.940, top=0.965, bottom=0.070,
        hspace=0.31, wspace=0.31,
    )
    _draw_manuscript_time_series_panel(fig, grid[0, 0], df)
    _draw_manuscript_breakpoint_panel(fig, grid[0, 1], df, temp_model, albedo_model)

    for fmt in ("png", "pdf"):
        output = MANUSCRIPT_FIG_DIR / f"{MANUSCRIPT_FIG_UPDATED_STEM}.{fmt}"
        fig.savefig(output, dpi=MANUSCRIPT_FIG_DPI if fmt == "png" else 600)

    plt.close(fig)


# =============================================================================
# 9. FIGURE A — ANNUAL T2M + GLASS WITH QUADRATIC FITS
# =============================================================================

def plot_fig2a(df: pd.DataFrame):
    years = df["Year"].to_numpy()
    temp = df["T2M_C"].to_numpy()
    alb = df["GLASS_Albedo"].to_numpy()

    fit_t = quadratic_fit(years, temp)
    fit_a = quadratic_fit(years, alb)

    fig, ax1 = plt.subplots(figsize=(7.2, 3.25))
    ax2 = ax1.twinx()

    # Data series.
    line_t, = ax1.plot(
        years, temp,
        marker="^",
        markersize=4.2,
        linewidth=1.0,
        label="ERA5 T2M",
    )
    fit_line_t, = ax1.plot(
        fit_t["x_dense"],
        fit_t["y_dense"],
        linewidth=2.0,
        label="ERA5 T2M quadratic",
    )

    line_a, = ax2.plot(
        years, alb,
        marker="o",
        markersize=3.8,
        linewidth=1.0,
        label="GLASS albedo",
    )
    fit_line_a, = ax2.plot(
        fit_a["x_dense"],
        fit_a["y_dense"],
        linewidth=2.0,
        label="GLASS albedo quadratic",
    )

    # Manuscript period guides, not estimated changepoints.
    ax1.axvline(2015, linestyle="--", linewidth=0.8, alpha=0.55)
    ax1.axvline(2020, linestyle="--", linewidth=0.8, alpha=0.55)

    ax1.set_xlim(2000.4, 2024.6)
    ax1.set_xticks([2001, 2005, 2010, 2015, 2020, 2024])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("ERA5 T2M (°C)")
    ax2.set_ylabel("GLASS surface albedo")

    clean_axis(ax1)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(direction="out")

    # Labels for manuscript periods.
    ymin, ymax = ax1.get_ylim()
    yy = ymax - 0.03 * (ymax - ymin)
    ax1.text(2008, yy, "Stable", ha="center", va="top")
    ax1.text(2017, yy, "Transition", ha="center", va="top")
    ax1.text(2022, yy, "Warming", ha="center", va="top")

    handles = [line_t, fit_line_t, line_a, fit_line_a]
    labels = [h.get_label() for h in handles]
    ax1.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        ncol=2,
        columnspacing=1.0,
        handlelength=2.1,
    )

    panel_label(ax1, "a")

    fig.tight_layout()
    save_figure(
        fig,
        "Fig2a_T2M_GLASS_annual_timeseries_quadratic_updated",
    )
    plt.close(fig)

    fit_table = pd.DataFrame([
        {
            "Variable": "T2M",
            "center_year": fit_t["center_year"],
            "quadratic_coef": fit_t["coef_quadratic"],
            "linear_coef": fit_t["coef_linear"],
            "intercept": fit_t["intercept"],
            "R2": fit_t["r2"],
        },
        {
            "Variable": "GLASS_Albedo",
            "center_year": fit_a["center_year"],
            "quadratic_coef": fit_a["coef_quadratic"],
            "linear_coef": fit_a["coef_linear"],
            "intercept": fit_a["intercept"],
            "R2": fit_a["r2"],
        },
    ])

    fit_table.to_csv(
        OUT_TABLE / "Fig2a_quadratic_fit_summary_updated.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 10. FIGURE B — INDEPENDENT CHANGEPOINTS
# =============================================================================

def plot_segment_means(
    ax,
    years: np.ndarray,
    model: Dict[str, object],
    label_prefix: str,
):
    bounds = model["bounds"]
    means = model["segment_means"]

    for (start, end), mean in zip(bounds, means):
        x0 = years[start]
        x1 = years[end - 1]
        ax.hlines(
            mean,
            x0,
            x1,
            linewidth=2.2,
        )
        ax.text(
            (x0 + x1) / 2.0,
            mean,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )

    for b in model["break_years"]:
        ax.axvline(
            b,
            linestyle="--",
            linewidth=0.8,
            alpha=0.65,
        )


def plot_fig2b(df: pd.DataFrame):
    years = df["Year"].to_numpy(dtype=int)
    temp = df["T2M_C"].to_numpy(dtype=float)
    alb = df["GLASS_Albedo"].to_numpy(dtype=float)

    temp_model, temp_models = fit_changepoints(
        years,
        temp,
        min_segment=MIN_SEGMENT_YEARS,
        mode=CHANGEPOINT_MODE,
    )
    alb_model, alb_models = fit_changepoints(
        years,
        alb,
        min_segment=MIN_SEGMENT_YEARS,
        mode=CHANGEPOINT_MODE,
    )

    fig, axes = plt.subplots(
        2, 1,
        figsize=(7.2, 4.5),
        sharex=True,
        gridspec_kw={"hspace": 0.10},
    )

    ax_t, ax_a = axes

    ax_t.plot(
        years,
        temp,
        marker="o",
        markersize=3.5,
        linewidth=0.9,
    )
    plot_segment_means(ax_t, years, temp_model, "T2M")

    ax_a.plot(
        years,
        alb,
        marker="o",
        markersize=3.5,
        linewidth=0.9,
    )
    plot_segment_means(ax_a, years, alb_model, "Albedo")

    ax_t.set_ylabel("ERA5 T2M (°C)")
    ax_a.set_ylabel("GLASS surface albedo")
    ax_a.set_xlabel("Year")

    ax_a.set_xlim(2000.4, 2024.6)
    ax_a.set_xticks([2001, 2005, 2010, 2015, 2020, 2024])

    clean_axis(ax_t)
    clean_axis(ax_a)

    t_breaks = ", ".join(map(str, temp_model["break_years"]))
    a_breaks = ", ".join(map(str, alb_model["break_years"]))

    if not t_breaks:
        t_breaks = "none selected"
    if not a_breaks:
        a_breaks = "none selected"

    ax_t.text(
        0.01, 0.94,
        f"T2M breaks: {t_breaks}",
        transform=ax_t.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    ax_a.text(
        0.01, 0.94,
        f"GLASS albedo breaks: {a_breaks}",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )

    panel_label(ax_t, "b")

    fig.tight_layout()
    save_figure(
        fig,
        "Fig2b_T2M_and_GLASS_own_changepoints_reference_style_updated",
    )
    plt.close(fig)

    # Export candidate-model evidence.
    temp_models = temp_models.copy()
    temp_models.insert(0, "Variable", "T2M")

    alb_models = alb_models.copy()
    alb_models.insert(0, "Variable", "GLASS_Albedo")

    pd.concat([temp_models, alb_models], ignore_index=True).to_csv(
        OUT_TABLE / "Fig2b_changepoint_candidate_models_updated.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rows = []

    for variable, model in [
        ("T2M", temp_model),
        ("GLASS_Albedo", alb_model),
    ]:
        row = {
            "Variable": variable,
            "SelectionMode": CHANGEPOINT_MODE,
            "N_breaks": model["n_breaks"],
            "Break_1": (
                model["break_years"][0]
                if len(model["break_years"]) >= 1 else np.nan
            ),
            "Break_2": (
                model["break_years"][1]
                if len(model["break_years"]) >= 2 else np.nan
            ),
            "SSE": model["sse"],
            "BIC": model["bic"],
        }

        for i, ((start, end), mean) in enumerate(
            zip(model["bounds"], model["segment_means"]),
            start=1,
        ):
            row[f"P{i}_start"] = int(years[start])
            row[f"P{i}_end"] = int(years[end - 1])
            row[f"P{i}_mean"] = float(mean)

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        OUT_TABLE / "Fig2b_changepoint_segment_summary_updated.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 11. FIGURE C — FIRST-ORDER RATES + QUADRATIC TREND
# =============================================================================

def plot_fig2c(df: pd.DataFrame):
    years_full = df["Year"].to_numpy(dtype=int)
    temp = df["T2M_C"].to_numpy(dtype=float)
    alb = df["GLASS_Albedo"].to_numpy(dtype=float)

    # Year-to-year first differences correspond to years 2002–2024.
    rate_years = years_full[1:]
    d_temp = np.diff(temp)
    neg_d_alb = -np.diff(alb)

    d_temp_z = zscore(d_temp)
    neg_d_alb_z = zscore(neg_d_alb)

    fit_t = quadratic_fit(rate_years, d_temp_z)
    fit_a = quadratic_fit(rate_years, neg_d_alb_z)

    same_positive = (
        np.isfinite(d_temp_z)
        & np.isfinite(neg_d_alb_z)
        & (d_temp_z > 0)
        & (neg_d_alb_z > 0)
    )

    n_valid = int(np.count_nonzero(
        np.isfinite(d_temp_z) & np.isfinite(neg_d_alb_z)
    ))
    n_same = int(np.count_nonzero(same_positive))
    frac = n_same / n_valid if n_valid else np.nan

    fig, ax = plt.subplots(figsize=(7.2, 4.1))

    line_a, = ax.plot(
        rate_years,
        neg_d_alb_z,
        marker="^",
        markersize=4.2,
        linestyle=":",
        linewidth=1.1,
        alpha=0.65,
        label="-d(GLASS albedo)/dt",
    )
    fit_line_a, = ax.plot(
        fit_a["x_dense"],
        fit_a["y_dense"],
        linewidth=2.3,
        label="-d(GLASS albedo)/dt quadratic",
    )

    line_t, = ax.plot(
        rate_years,
        d_temp_z,
        marker="D",
        markersize=3.8,
        linestyle=":",
        linewidth=1.1,
        alpha=0.65,
        label="d(T2M)/dt",
    )
    fit_line_t, = ax.plot(
        fit_t["x_dense"],
        fit_t["y_dense"],
        linewidth=2.3,
        label="d(T2M)/dt quadratic",
    )

    ax.axhline(0, linestyle=":", linewidth=0.9, alpha=0.7)

    # Period boundaries are manuscript context only.
    ax.axvline(2015, linestyle="--", linewidth=0.9, alpha=0.65)
    ax.axvline(2020, linestyle="--", linewidth=0.9, alpha=0.65)

    ax.set_xlim(2001.2, 2024.8)
    ax.set_xticks([2002, 2005, 2010, 2015, 2020, 2024])
    ax.set_xlabel("Year")
    ax.set_ylabel("First-order rate (z score)")

    clean_axis(ax)

    ymin, ymax = ax.get_ylim()
    yy = ymax - 0.03 * (ymax - ymin)

    ax.text(2008, yy, "Stable", ha="center", va="top")
    ax.text(2017, yy, "Transition", ha="center", va="top")
    ax.text(2022, yy, "Warming", ha="center", va="top")

    ax.text(
        0.01,
        0.035,
        (
            "dT2M > 0 and -d(GLASS albedo)/dt > 0: "
            f"{n_same}/{n_valid} years ({100.0 * frac:.1f}%)"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
    )

    # Keep only the two observed-rate definitions in the compact legend;
    # thick continuous curves are visually identifiable as quadratic fits.
    ax.legend(
        handles=[line_a, line_t],
        labels=["-d(GLASS albedo)/dt", "d(T2M)/dt"],
        loc="upper center",
        bbox_to_anchor=(0.65, 1.16),
        frameon=False,
        ncol=2,
        handlelength=3.2,
    )

    panel_label(ax, "c")

    fig.tight_layout()
    save_figure(
        fig,
        "Fig2c_first_order_rate_second_order_trend_updated",
    )
    plt.close(fig)

    rate_df = pd.DataFrame({
        "Year": rate_years,
        "dT2M_C_per_year": d_temp,
        "neg_dGLASS_Albedo_per_year": neg_d_alb,
        "dT2M_z": d_temp_z,
        "neg_dGLASS_Albedo_z": neg_d_alb_z,
        "both_positive_z": same_positive.astype(int),
    })

    rate_df.to_csv(
        OUT_TABLE / "Fig2c_first_order_rate_second_order_trend_updated.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([
        {
            "Variable": "dT2M_z",
            "center_year": fit_t["center_year"],
            "quadratic_coef": fit_t["coef_quadratic"],
            "linear_coef": fit_t["coef_linear"],
            "intercept": fit_t["intercept"],
            "R2": fit_t["r2"],
        },
        {
            "Variable": "neg_dGLASS_Albedo_z",
            "center_year": fit_a["center_year"],
            "quadratic_coef": fit_a["coef_quadratic"],
            "linear_coef": fit_a["coef_linear"],
            "intercept": fit_a["intercept"],
            "R2": fit_a["r2"],
        },
    ]).to_csv(
        OUT_TABLE / "Fig2c_quadratic_fit_summary_updated.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 12. OPTIONAL COMBINED FIGURE A–C
# =============================================================================

def make_combined_abc_preview():
    """
    The three publication panels are intentionally saved separately above.
    This function is not used automatically because combining rasterized files
    would reduce vector editability. Assemble the SVG/PDF panels in Illustrator,
    PowerPoint or the manuscript layout after scientific QC.
    """
    pass


# =============================================================================
# 13. MAIN
# =============================================================================

def main():
    print("\n" + "#" * 100)
    print("UPDATED FIG. 2 — GLASS SURFACE ALBEDO AND ERA5 T2M")
    print("#" * 100)
    print("ALBEDO_ROOT:", ALBEDO_ROOT)
    print("T2M_ROOT:", T2M_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("Years:", START_YEAR, "-", END_YEAR)
    print("Changepoint mode:", CHANGEPOINT_MODE)

    # -------------------------------------------------------------------------
    # File audit before calculations.
    # -------------------------------------------------------------------------
    audit_rows = []

    for year in YEARS:
        a = find_albedo_file(year)
        t = find_t2m_file(year)

        audit_rows.append({
            "Year": year,
            "GLASS_exists": a.exists(),
            "GLASS_file": str(a),
            "T2M_exists": t.exists(),
            "T2M_file": str(t),
        })

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(
        OUT_LOG / "Fig2_updated_input_file_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nInput file audit: OK — all 2001–2024 source files located.")

    # -------------------------------------------------------------------------
    # Global annual means.
    # -------------------------------------------------------------------------
    df = build_global_annual_series()

    if len(df) != len(YEARS):
        raise RuntimeError(
            f"Expected {len(YEARS)} annual records, obtained {len(df)}."
        )

    # -------------------------------------------------------------------------
    # Figures.
    # -------------------------------------------------------------------------
    plot_fig2a(df)
    print("Saved Fig. 2a")

    plot_fig2b(df)
    print("Saved Fig. 2b")

    print("Fig. 2c was not redrawn: this run updates Fig. 2a and Fig. 2b only.")

    print("\n" + "#" * 100)
    print("FINISHED")
    print("#" * 100)
    print("Figures:", OUT_FIG)
    print("Source data:", OUT_TABLE)
    print("Logs:", OUT_LOG)
    print(
        "\nImportant: changepoints were re-estimated from the UPDATED annual "
        "series and were not copied from previous figures."
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()

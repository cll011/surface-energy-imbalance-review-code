# -*- coding: utf-8 -*-
"""Shared data, style and export helpers for the revised supplementary figures."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapefile
from matplotlib.colors import TwoSlopeNorm
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from shapely.geometry import LineString, mapping, shape
from shapely.ops import transform as shapely_transform


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "source_data"
SOURCE_ROOT = PROJECT_ROOT / "results" / "fig2_source"
PATH_ROOT = DATA_ROOT / "pathway"
SITE_CSV = DATA_ROOT / "validation" / "FigS4_site_validation_input_latest_rasters.csv"
CONTROLS_CSV = DATA_ROOT / "controls" / "R3_annual_background_controls.csv"
SHAPE_ROOT = PROJECT_ROOT / "data" / "regions"
SOURCE_DATA = DATA_ROOT / "supplementary"
SOURCE_DATA.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "supplementary"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

YEARS = np.arange(2001, 2025)
PERIODS = {
    "P1": np.arange(2001, 2015),
    "P2": np.arange(2015, 2020),
    "P3": np.arange(2020, 2025),
}
REGIONS = [
    ("Plateau", "Plateau", "ROI_Plateau.shp"),
    ("I Boreal Arctic", "Boreal-Arctic", "Zone_I_Boreal_Arctic.shp"),
    ("II MidLat Arid", "Mid-latitude arid", "Zone_II_MidLat_Arid.shp"),
    ("SiberianTaiga", "Siberian taiga", "ROI_SiberianTaiga.shp"),
    ("Greenland", "Greenland", "ROI_Greenland.shp"),
    ("Sahelian", "Sahel", "ROI_Sahelian.shp"),
    ("Amazon", "Amazon", "ROI_Amazon.shp"),
    ("III Tropical South", "Tropical/Southern", "Zone_III_Tropical_South.shp"),
]

COLORS = {
    "t2m": "#B6403A",
    "albedo": "#2C75A8",
    "rn": "#D18B2C",
    "sh": "#B6403A",
    "lh": "#2A8C82",
    "vpd": "#7C5B8C",
    "sm": "#557A46",
    "neutral": "#596168",
    "light": "#DCE3E7",
    "grid": "#E5E9EC",
    "black": "#262626",
}
PERIOD_COLORS = {"P1": "#78A7C8", "P2": "#D8A653", "P3": "#C45B58"}


C = None


def require_common_grid_module():
    raise RuntimeError(
        "This operation rebuilds source data from the full raster archive. "
        "Run the corresponding script under src/analysis first."
    )


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style_axis(ax, grid: bool = False) -> None:
    ax.tick_params(length=2.4, width=0.7, direction="out")
    if grid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)


def panel_label(ax, label: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.5, fontweight="bold", ha="left", va="bottom")


def save_figure(fig, stem: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg", "tiff"):
        path = OUTPUT_ROOT / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.035}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
    plt.close(fig)


def write_qa(stem: str, details: dict) -> None:
    files = {}
    for suffix in ("png", "pdf", "svg", "tiff"):
        path = OUTPUT_ROOT / f"{stem}.{suffix}"
        files[suffix] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
    payload = {"figure": stem, "files": files, **details}
    (OUTPUT_ROOT / f"{stem}_QA.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def zscore(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x, ddof=0)


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        meta = {"transform": src.transform, "crs": src.crs, "bounds": src.bounds}
    return arr, meta


def _grid_centres(transform, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    lon = transform.c + (np.arange(width) + 0.5) * transform.a
    lat = transform.f + (np.arange(height) + 0.5) * transform.e
    return lon, lat


def add_map(ax, array: np.ndarray, *, cmap, norm=None, vmin=None, vmax=None, label: str, ticks=None, colorbar: bool = True):
    # Rectilinear display preserves the native 0.25-degree cells and avoids a
    # Cartopy/GEOS dateline failure in the local Python stack.
    im = ax.imshow(
        np.ma.masked_invalid(array),
        extent=(-180, 180, -90, 90),
        origin="upper",
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
        aspect="auto",
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks([-120, 0, 120])
    ax.set_yticks([-40, 0, 40, 80])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    add_world_outline(ax)
    if colorbar:
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.025, ticks=ticks)
        cb.outline.set_linewidth(0.45)
        cb.set_label(label, labelpad=1.5)
        cb.ax.tick_params(length=2, width=0.5)
    return im


@lru_cache(maxsize=1)
def _world_outline_segments() -> tuple[np.ndarray, ...]:
    """Read the local dissolved-land boundary once and split dateline crossings."""
    path = SHAPE_ROOT / "ne_10m_global_disolve.shp"
    reader = shapefile.Reader(str(path))
    segments = []
    for geometry in reader.shapes():
        points = np.asarray(geometry.points, dtype=float)
        boundaries = list(geometry.parts) + [len(points)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            part = points[start:stop]
            if len(part) < 2:
                continue
            jumps = np.where(np.abs(np.diff(part[:, 0])) > 180)[0] + 1
            for segment in np.split(part, jumps):
                if len(segment) >= 2:
                    simplified = LineString(segment).simplify(0.15, preserve_topology=False)
                    coordinates = np.asarray(simplified.coords, dtype=float)
                    if len(coordinates) >= 2:
                        segments.append(coordinates)
    return tuple(segments)


def add_world_outline(ax) -> None:
    """Add a restrained geographic reference boundary without obscuring raster values."""
    for segment in _world_outline_segments():
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color="#555E63",
            linewidth=0.20,
            alpha=0.58,
            zorder=4,
        )


def robust_symmetric_limit(array: np.ndarray, percentile: float = 98.0, floor: float = 1e-6) -> float:
    values = np.abs(array[np.isfinite(array)])
    return max(float(np.percentile(values, percentile)), floor)


def robust_limits(array: np.ndarray, low: float = 2, high: float = 98) -> tuple[float, float]:
    values = array[np.isfinite(array)]
    return float(np.percentile(values, low)), float(np.percentile(values, high))


def _read_region_geometries(path: Path, target_crs: CRS) -> list[dict]:
    reader = shapefile.Reader(str(path))
    geometries = [shape(item.__geo_interface__) for item in reader.shapes()]
    geometries = [item for item in geometries if not item.is_empty]
    source_crs = target_crs
    prj = path.with_suffix(".prj")
    if prj.exists():
        text = prj.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            source_crs = CRS.from_wkt(text)
    if source_crs != target_crs:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        geometries = [shapely_transform(transformer.transform, item) for item in geometries]
    return [mapping(item) for item in geometries]


@lru_cache(maxsize=16)
def region_mask(region_key: str) -> np.ndarray:
    matches = [item for item in REGIONS if item[0] == region_key]
    if len(matches) != 1:
        raise KeyError(region_key)
    path = SHAPE_ROOT / matches[0][2]
    if C is None:
        require_common_grid_module()
    profile = C.target_profile()
    target_crs = CRS.from_user_input(profile["crs"])
    return geometry_mask(
        _read_region_geometries(path, target_crs),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        invert=True,
        all_touched=False,
    )


def ensure_s2_spatial_data() -> dict[str, Path]:
    paths = {
        "t2m_mean": SOURCE_DATA / "FigS2_1_T2M_mean_2001_2024.tif",
        "albedo_mean": SOURCE_DATA / "FigS2_1_GLASS_albedo_mean_2001_2024.tif",
        "t2m_slope": SOURCE_DATA / "FigS2_1_T2M_slope_2001_2024.tif",
        "albedo_slope": SOURCE_DATA / "FigS2_1_GLASS_albedo_slope_2001_2024.tif",
        "zonal": SOURCE_DATA / "FigS2_1_zonal_profiles.csv",
    }
    if all(path.exists() for path in paths.values()):
        return paths

    if C is None:
        require_common_grid_module()

    t2m = np.stack([C.load_analysis_variable("T2M", int(year)) for year in YEARS]).astype("float32")
    albedo = np.stack([C.load_analysis_variable("SurfaceAlbedo_GLASS", int(year)) for year in YEARS]).astype("float32")
    t2m_mean = np.nanmean(t2m, axis=0)
    albedo_mean = np.nanmean(albedo, axis=0)
    x = YEARS.astype(float) - YEARS.mean()
    denom = float(np.sum(x * x))
    t2m_slope = np.nansum(x[:, None, None] * (t2m - np.nanmean(t2m, axis=0)), axis=0) / denom
    albedo_slope = np.nansum(x[:, None, None] * (albedo - np.nanmean(albedo, axis=0)), axis=0) / denom
    valid = C.common_land_mask() & np.all(np.isfinite(t2m), axis=0) & np.all(np.isfinite(albedo), axis=0)
    for arr in (t2m_mean, albedo_mean, t2m_slope, albedo_slope):
        arr[~valid] = np.nan
    C.write_raster(paths["t2m_mean"], t2m_mean, {"units": "degree_C"})
    C.write_raster(paths["albedo_mean"], albedo_mean, {"units": "fraction"})
    C.write_raster(paths["t2m_slope"], t2m_slope, {"units": "degree_C_per_year"})
    C.write_raster(paths["albedo_slope"], albedo_slope, {"units": "fraction_per_year"})
    profile = C.target_profile()
    _lon, lat = _grid_centres(profile["transform"], profile["height"], profile["width"])
    zonal = pd.DataFrame(
        {
            "Latitude": lat,
            "T2M_mean_C": np.nanmean(t2m_mean, axis=1),
            "GLASS_albedo_mean": np.nanmean(albedo_mean, axis=1),
            "T2M_slope_C_per_year": np.nanmean(t2m_slope, axis=1),
            "GLASS_albedo_slope_per_year": np.nanmean(albedo_slope, axis=1),
        }
    )
    zonal.to_csv(paths["zonal"], index=False, encoding="utf-8-sig")
    return paths


def ensure_regional_t2m_albedo_timeseries() -> Path:
    path = SOURCE_DATA / "FigS2_2_regional_T2M_GLASS_annual_means_and_slopes.csv"
    if path.exists():
        return path
    if C is None:
        require_common_grid_module()
    rows = []
    arrays = {
        int(year): (
            C.load_analysis_variable("T2M", int(year)),
            C.load_analysis_variable("SurfaceAlbedo_GLASS", int(year)),
        )
        for year in YEARS
    }
    weights = C.latitude_weights()
    for key, label, _filename in REGIONS:
        mask = region_mask(key) & C.common_land_mask()
        for year in YEARS:
            t2m, albedo = arrays[int(year)]
            valid = mask & np.isfinite(t2m) & np.isfinite(albedo)
            w = weights[valid].astype(float)
            rows.append(
                {
                    "Region": key,
                    "DisplayRegion": label,
                    "Year": int(year),
                    "T2M_C": float(np.average(t2m[valid], weights=w)),
                    "GLASS_Albedo": float(np.average(albedo[valid], weights=w)),
                    "ValidPixels": int(valid.sum()),
                }
            )
    frame = pd.DataFrame(rows)
    frame["T2M_z"] = frame.groupby("Region")["T2M_C"].transform(lambda x: zscore(x))
    frame["GLASS_z"] = frame.groupby("Region")["GLASS_Albedo"].transform(lambda x: zscore(x))
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def make_manifest() -> None:
    rows = []
    for path in sorted(OUTPUT_ROOT.iterdir()):
        if path.is_file() and path.suffix.lower() in {".py", ".png", ".pdf", ".svg", ".tiff", ".csv", ".json", ".md", ".txt"}:
            rows.append({"File": path.name, "Bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "Output_Manifest.csv", index=False, encoding="utf-8-sig")


set_style()

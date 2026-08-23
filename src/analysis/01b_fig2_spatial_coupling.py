# -*- coding: utf-8 -*-
"""Calculate the final GLASS-albedo/T2M spatial coupling products for Fig. 2.

The script uses annual common-grid rasters for 2001-2024 and independently
samples the same ERA5-Land land-sea mask onto the analysis grid. It writes the
pixel-wise temporal correlation, linear trends, directional classes and the
eight prespecified regional summaries used by the main and supplementary
figures.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
import shapefile
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_paths import get_path  # noqa: E402


YEARS = np.arange(2001, 2025)
LAND_THRESHOLD = 0.5
NODATA_FLOAT = -9999.0
NODATA_CLASS = 0
DISPLAY_SAMPLE_PER_REGION = 20000
RANDOM_SEED = 20260818

R3_GRID = get_path("r3_common_grid")
LSM_PATH = get_path("lsm_nc")
REGION_ROOT = PROJECT_ROOT / "data" / "regions"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "fig2_source"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

REGIONS: Sequence[Tuple[str, str]] = (
    ("Plateau", "ROI_Plateau.shp"),
    ("I Boreal Arctic", "Zone_I_Boreal_Arctic.shp"),
    ("II MidLat Arid", "Zone_II_MidLat_Arid.shp"),
    ("SiberianTaiga", "ROI_SiberianTaiga.shp"),
    ("Greenland", "ROI_Greenland.shp"),
    ("Sahelian", "ROI_Sahelian.shp"),
    ("Amazon", "ROI_Amazon.shp"),
    ("III Tropical South", "Zone_III_Tropical_South.shp"),
)


def raster_path(variable: str, year: int) -> Path:
    path = R3_GRID / variable / f"{variable}_{year}_R3_common.tif"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_stack(variable: str) -> Tuple[np.ndarray, dict]:
    arrays: List[np.ndarray] = []
    profile = None
    for year in YEARS:
        with rasterio.open(raster_path(variable, int(year))) as src:
            array = src.read(1, masked=True).astype("float32").filled(np.nan)
            if src.nodata is not None:
                array[array == src.nodata] = np.nan
            if profile is None:
                profile = src.profile.copy()
            elif src.shape != arrays[0].shape or src.transform != profile["transform"]:
                raise ValueError(f"Grid mismatch for {variable}, {year}")
            arrays.append(array)
    if profile is None:
        raise RuntimeError(f"No rasters loaded for {variable}")
    return np.stack(arrays), profile


def _coord_name(data_array, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in data_array.coords or name in data_array.dims:
            return name
    raise ValueError(f"Coordinate not found among {candidates}")


def common_land_mask(profile: dict) -> np.ndarray:
    with xr.open_dataset(LSM_PATH) as dataset:
        variable = "lsm" if "lsm" in dataset.data_vars else list(dataset.data_vars)[0]
        mask = dataset[variable].squeeze(drop=True).load()
    lat_name = _coord_name(mask, ("latitude", "lat", "y"))
    lon_name = _coord_name(mask, ("longitude", "lon", "x"))
    mask = mask.transpose(lat_name, lon_name)
    longitude = np.asarray(mask[lon_name].values, dtype=float)
    if np.nanmax(longitude) > 180:
        mask = mask.assign_coords(
            {lon_name: ((longitude + 180.0) % 360.0) - 180.0}
        ).sortby(lon_name)
    mask = mask.sortby(lat_name)

    transform = profile["transform"]
    width = profile["width"]
    height = profile["height"]
    lon = transform.c + (np.arange(width) + 0.5) * transform.a
    lat = transform.f + (np.arange(height) + 0.5) * transform.e
    sampled = mask.sel(
        {
            lat_name: xr.DataArray(lat, dims="y"),
            lon_name: xr.DataArray(((lon + 180) % 360) - 180, dims="x"),
        },
        method="nearest",
    ).values
    return np.isfinite(sampled) & (sampled >= LAND_THRESHOLD)


def temporal_statistics(
    albedo: np.ndarray, temperature: np.ndarray, land: np.ndarray
) -> Dict[str, np.ndarray]:
    valid = (
        land
        & np.all(np.isfinite(albedo), axis=0)
        & np.all(np.isfinite(temperature), axis=0)
    )
    albedo_mean = np.mean(albedo, axis=0)
    temperature_mean = np.mean(temperature, axis=0)
    albedo_anomaly = albedo - albedo_mean
    temperature_anomaly = temperature - temperature_mean
    numerator = np.sum(albedo_anomaly * temperature_anomaly, axis=0)
    denominator = np.sqrt(
        np.sum(albedo_anomaly**2, axis=0)
        * np.sum(temperature_anomaly**2, axis=0)
    )
    correlation = np.full(valid.shape, np.nan, dtype="float32")
    valid_corr = valid & np.isfinite(denominator) & (denominator > 0)
    correlation[valid_corr] = (numerator[valid_corr] / denominator[valid_corr]).astype(
        "float32"
    )

    centred_year = YEARS.astype(float) - YEARS.mean()
    denominator_year = float(np.sum(centred_year**2))
    albedo_slope = np.sum(
        centred_year[:, None, None] * albedo_anomaly, axis=0
    ) / denominator_year
    temperature_slope = np.sum(
        centred_year[:, None, None] * temperature_anomaly, axis=0
    ) / denominator_year
    albedo_slope[~valid] = np.nan
    temperature_slope[~valid] = np.nan

    classes = np.zeros(valid.shape, dtype="uint8")
    darkening_warming = valid & (albedo_slope < 0) & (temperature_slope > 0)
    classes[darkening_warming & (correlation < 0)] = 1
    classes[darkening_warming & (correlation >= 0)] = 2
    classes[valid & (temperature_slope > 0) & (albedo_slope >= 0)] = 3
    classes[valid & (classes == 0)] = 4
    return {
        "correlation": correlation,
        "albedo_slope": albedo_slope.astype("float32"),
        "temperature_slope": temperature_slope.astype("float32"),
        "classes": classes,
        "valid": valid,
    }


def write_raster(path: Path, array: np.ndarray, profile: dict, categorical: bool) -> None:
    output_profile = profile.copy()
    if categorical:
        output_profile.update(dtype="uint8", nodata=NODATA_CLASS, compress="deflate")
        output = array.astype("uint8")
    else:
        output_profile.update(
            dtype="float32", nodata=NODATA_FLOAT, compress="deflate", predictor=3
        )
        output = array.astype("float32", copy=True)
        output[~np.isfinite(output)] = NODATA_FLOAT
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(output, 1)


def _region_geometries(path: Path, target_crs: CRS) -> List[dict]:
    reader = shapefile.Reader(str(path))
    geometries = [shape(item.__geo_interface__) for item in reader.shapes()]
    prj = path.with_suffix(".prj")
    source_crs = target_crs
    if prj.exists():
        text = prj.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            source_crs = CRS.from_wkt(text)
    if source_crs != target_crs:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        geometries = [
            shapely_transform(transformer.transform, geometry)
            for geometry in geometries
        ]
    return [mapping(geometry) for geometry in geometries if not geometry.is_empty]


def regional_summaries(
    correlation: np.ndarray, profile: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_crs = CRS.from_user_input(profile["crs"])
    rng = np.random.default_rng(RANDOM_SEED)
    rows: List[dict] = []
    samples: List[pd.DataFrame] = []
    for order, (region, filename) in enumerate(REGIONS):
        inside = geometry_mask(
            _region_geometries(REGION_ROOT / filename, target_crs),
            out_shape=correlation.shape,
            transform=profile["transform"],
            invert=True,
            all_touched=False,
        )
        values = correlation[inside & np.isfinite(correlation)]
        if values.size == 0:
            raise ValueError(f"No valid pixels in {region}")
        rows.append(
            {
                "Order": order,
                "Region": region,
                "N": int(values.size),
                "Mean": float(np.mean(values)),
                "Median": float(np.median(values)),
                "P10": float(np.percentile(values, 10)),
                "Q25": float(np.percentile(values, 25)),
                "Q75": float(np.percentile(values, 75)),
                "P90": float(np.percentile(values, 90)),
                "Negative_fraction_percent": float(np.mean(values < 0) * 100),
            }
        )
        count = min(DISPLAY_SAMPLE_PER_REGION, values.size)
        index = rng.choice(values.size, size=count, replace=False)
        samples.append(pd.DataFrame({"Region": region, "r": values[index]}))
    return pd.DataFrame(rows), pd.concat(samples, ignore_index=True)


def main() -> None:
    albedo, profile = read_stack("SurfaceAlbedo_GLASS")
    temperature, temperature_profile = read_stack("T2M")
    if temperature_profile["transform"] != profile["transform"]:
        raise ValueError("GLASS and T2M rasters are not grid aligned")
    if np.nanmedian(temperature) > 100:
        temperature = temperature - 273.15

    land = common_land_mask(profile)
    result = temporal_statistics(albedo, temperature, land)
    write_raster(
        OUTPUT_ROOT / "Fig2c_spatial_raw_corr_GLASS_albedo_T2M_lsm_updated.tif",
        result["correlation"],
        profile,
        categorical=False,
    )
    write_raster(
        OUTPUT_ROOT / "Fig2d_directional_classes_GLASS_T2M_lsm_updated.tif",
        result["classes"],
        profile,
        categorical=True,
    )
    write_raster(
        OUTPUT_ROOT / "Fig2_GLASS_albedo_slope_2001_2024.tif",
        result["albedo_slope"],
        profile,
        categorical=False,
    )
    write_raster(
        OUTPUT_ROOT / "Fig2_T2M_slope_2001_2024.tif",
        result["temperature_slope"],
        profile,
        categorical=False,
    )

    summary, sample = regional_summaries(result["correlation"], profile)
    summary.to_csv(
        OUTPUT_ROOT / "Fig2e_regional_correlation_summary_lsm_updated.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sample.to_csv(
        OUTPUT_ROOT / "Fig2e_regional_correlation_display_sample_lsm_updated.csv.gz",
        index=False,
        compression="gzip",
    )

    valid_corr = np.isfinite(result["correlation"])
    valid_class = result["classes"] > 0
    audit = pd.DataFrame(
        {
            "Metric": ["negative_r_percent", "r_below_minus_0_3_percent", "C1_percent", "C3_percent"],
            "Value": [
                np.mean(result["correlation"][valid_corr] < 0) * 100,
                np.mean(result["correlation"][valid_corr] < -0.3) * 100,
                np.mean(result["classes"][valid_class] == 1) * 100,
                np.mean(result["classes"][valid_class] == 3) * 100,
            ],
        }
    )
    audit.to_csv(OUTPUT_ROOT / "Fig2_spatial_summary.csv", index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()


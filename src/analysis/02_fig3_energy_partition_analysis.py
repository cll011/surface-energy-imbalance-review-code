# -*- coding: utf-8 -*-
"""
Result 1.2 Fig. 3 workflow: energy buffering to heat retention.

Purpose
-------
Move Result 1.2 from a phenomenological albedo-temperature relationship to a
quantitative energy-state transition:

P1: higher evaporative buffering and weaker heat retention
P2: transition
P3: stronger heat retention under warming and albedo loss

Inputs
------
Annual common-grid rasters:
D:\\10_Research\\01_Datasets\\04_Results\\Result3_Figures_optimized\\R3_CommonGrid_Rasters

Outputs
-------
D:\\10_Research\\01_Datasets\\04_Results\\Result1_Figures

The script writes a Nature-style Fig. 3, source CSV files, and a GeoTIFF/PNG map
of the P3-minus-P1 heat-retention transition index.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


YEARS = list(range(2001, 2025))
PERIODS = {
    "P1_2001_2014": list(range(2001, 2015)),
    "P2_2015_2019": list(range(2015, 2020)),
    "P3_2020_2024": list(range(2020, 2025)),
}

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_paths import get_path  # noqa: E402


R3_RASTER_DIR = get_path("r3_common_grid")
LSM_PATH = get_path("lsm_nc")
RESULT1_DIR = PROJECT_ROOT / "results" / "fig3_analysis"
RESULT1_DIR.mkdir(parents=True, exist_ok=True)

RESULT1_TS_CSV = RESULT1_DIR / "Result1_global_weighted_timeseries_GLASS_MODIS_T2M.csv"
ADJUSTED_GLASS_CSV = RESULT1_DIR / "Fig2_GLASS_main_series_adjustment.csv"

VARIABLES = [
    "T2M",
    "Rn",
    "LH",
    "SH",
    "SM",
    "VPD",
    "SWdown",
    "LWdown",
    "SWabs_MODIS",
    "SurfaceAlbedo_GLASS",
    "AlbedoLoss",
]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "albedo_loss": "#4C78A8",
    "t2m": "#B2182B",
    "rn": "#E68613",
    "lh": "#2C7FB8",
    "sh": "#D6604D",
    "vpd": "#7B3294",
    "ef": "#2C7FB8",
    "bowen": "#D6604D",
    "hri": "#222222",
}


def raster_path(var: str, year: int) -> Path:
    fp = R3_RASTER_DIR / var / f"{var}_{year}_R3_common.tif"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return fp


def read_array(fp: Path) -> np.ndarray:
    with rasterio.open(fp) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


def read_profile(fp: Path) -> dict:
    with rasterio.open(fp) as src:
        profile = src.profile.copy()
    return profile


def load_weight_and_mask() -> Tuple[np.ndarray, np.ndarray]:
    """Sample the common ERA5-Land mask and derive cosine-latitude weights."""
    template_path = raster_path("T2M", 2001)
    with rasterio.open(template_path) as source:
        transform = source.transform
        width, height = source.width, source.height
        template_valid = np.isfinite(
            source.read(1, masked=True).astype("float64").filled(np.nan)
        )
    with xr.open_dataset(LSM_PATH) as dataset:
        variable = "lsm" if "lsm" in dataset.data_vars else list(dataset.data_vars)[0]
        land = dataset[variable].squeeze(drop=True).load()
    lat_name = next(
        name for name in ("latitude", "lat", "y") if name in land.coords or name in land.dims
    )
    lon_name = next(
        name for name in ("longitude", "lon", "x") if name in land.coords or name in land.dims
    )
    land = land.transpose(lat_name, lon_name)
    longitude = np.asarray(land[lon_name].values, dtype=float)
    if np.nanmax(longitude) > 180:
        land = land.assign_coords(
            {lon_name: ((longitude + 180.0) % 360.0) - 180.0}
        ).sortby(lon_name)
    land = land.sortby(lat_name)
    longitudes = transform.c + (np.arange(width) + 0.5) * transform.a
    latitudes = transform.f + (np.arange(height) + 0.5) * transform.e
    sampled = land.sel(
        {
            lat_name: xr.DataArray(latitudes, dims="y"),
            lon_name: xr.DataArray(((longitudes + 180.0) % 360.0) - 180.0, dims="x"),
        },
        method="nearest",
    ).values
    valid = template_valid & np.isfinite(sampled) & (sampled >= 0.5)
    weight = np.broadcast_to(
        np.cos(np.deg2rad(latitudes))[:, None], (height, width)
    ).astype("float64")
    return weight, valid


def clean_variable(var: str, arr: np.ndarray) -> np.ndarray:
    out = arr.astype("float64", copy=True)
    if var == "T2M" and np.nanmedian(out) > 100:
        out = out - 273.15
    if var in {"SurfaceAlbedo_GLASS"}:
        out[(out < 0) | (out > 1)] = np.nan
    if var == "SM":
        out[(out < 0) | (out > 1)] = np.nan
    if var == "VPD":
        out[out < 0] = np.nan
    return out


def weighted_mean(arr: np.ndarray, weight: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    ok = mask & np.isfinite(arr) & np.isfinite(weight) & (weight > 0)
    n = int(ok.sum())
    if n == 0:
        return np.nan, 0
    return float(np.nansum(arr[ok] * weight[ok]) / np.nansum(weight[ok])), n


def build_annual_means() -> pd.DataFrame:
    weight, mask = load_weight_and_mask()
    records: List[Dict[str, float]] = []

    for year in YEARS:
        rec: Dict[str, float] = {"Year": year}
        for var in VARIABLES:
            arr = clean_variable(var, read_array(raster_path(var, year)))
            mean, n = weighted_mean(arr, weight, mask)
            rec[var] = mean
            rec[f"{var}_ValidPixels"] = n
        records.append(rec)

    df = pd.DataFrame(records)

    # Use the already audited Result 1.1 GLASS annual mean for the manuscript
    # albedo-loss line. The common-grid albedo is still retained in the CSV.
    if RESULT1_TS_CSV.exists():
        r1 = pd.read_csv(RESULT1_TS_CSV)
        keep = [c for c in ["Year", "GLASS_Albedo", "MODIS_Albedo", "T2M"] if c in r1.columns]
        r1 = r1[keep].rename(columns={
            "GLASS_Albedo": "GLASS_Albedo_R1_global",
            "MODIS_Albedo": "MODIS_Albedo_R1_global",
            "T2M": "T2M_R1_global",
        })
        df = df.merge(r1, on="Year", how="left")

    if ADJUSTED_GLASS_CSV.exists():
        adj = pd.read_csv(ADJUSTED_GLASS_CSV)
        if {"Year", "GLASS_Albedo_adjusted"}.issubset(adj.columns):
            adj = adj[["Year", "GLASS_Albedo_adjusted", "is_adjusted"]].rename(columns={
                "GLASS_Albedo_adjusted": "GLASS_Albedo_adjusted_R1_global",
                "is_adjusted": "GLASS_Albedo_R1_was_adjusted",
            })
            df = df.merge(adj, on="Year", how="left")

    if "GLASS_Albedo_adjusted_R1_global" in df.columns:
        main_albedo_col = "GLASS_Albedo_adjusted_R1_global"
    elif "GLASS_Albedo_R1_global" in df.columns:
        main_albedo_col = "GLASS_Albedo_R1_global"
    else:
        main_albedo_col = "SurfaceAlbedo_GLASS"
    p1_mask = df["Year"].isin(PERIODS["P1_2001_2014"])
    p1_albedo = float(df.loc[p1_mask, main_albedo_col].mean())
    df["GLASS_Albedo_main"] = df[main_albedo_col]
    df["AlbedoLoss_main"] = p1_albedo - df[main_albedo_col]

    df["EvaporativeFraction"] = df["LH"] / (df["LH"] + df["SH"])
    df["BowenRatio"] = df["SH"] / df["LH"]

    baseline = df.loc[p1_mask, ["Rn", "SH", "LH", "VPD"]].mean()
    spread = df.loc[p1_mask, ["Rn", "SH", "LH", "VPD"]].std(ddof=0).replace(0, np.nan)
    df["HeatRetentionIndex"] = (
        (df["Rn"] - baseline["Rn"]) / spread["Rn"]
        + (df["SH"] - baseline["SH"]) / spread["SH"]
        + (df["VPD"] - baseline["VPD"]) / spread["VPD"]
        - (df["LH"] - baseline["LH"]) / spread["LH"]
    ) / 4.0

    out_csv = RESULT1_DIR / "Fig3_annual_energy_transition_source.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)
    return df


def add_phase_guides(ax, y: float = 1.02) -> None:
    for x in [2015, 2020]:
        ax.axvline(x - 0.5, color="0.45", linestyle="--", linewidth=0.8, zorder=0)
    for label, x0, x1 in [("P1", 2001, 2014), ("P2", 2015, 2019), ("P3", 2020, 2024)]:
        ax.text(
            (x0 + x1) / 2,
            y,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="0.25",
        )


def zscore(series: pd.Series) -> pd.Series:
    vals = series.astype(float)
    sd = vals.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return vals * np.nan
    return (vals - vals.mean()) / sd


def period_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "GLASS_Albedo_main",
        "AlbedoLoss_main",
        "T2M",
        "Rn",
        "LH",
        "SH",
        "SM",
        "VPD",
        "SWabs_MODIS",
        "EvaporativeFraction",
        "BowenRatio",
        "HeatRetentionIndex",
    ]
    records = []
    p1_means = {}
    for pname, years in PERIODS.items():
        sub = df[df["Year"].isin(years)]
        for metric in metrics:
            value = float(sub[metric].mean())
            if pname == "P1_2001_2014":
                p1_means[metric] = value
            base = p1_means.get(metric, value)
            rel = np.nan
            if np.isfinite(base) and abs(base) > 1e-9:
                rel = (value - base) / abs(base) * 100
            records.append({
                "Period": pname,
                "Metric": metric,
                "Mean": value,
                "Delta_vs_P1": value - p1_means.get(metric, value),
                "Relative_change_vs_P1_percent": rel,
                "Years": ",".join(map(str, years)),
            })
    out = pd.DataFrame(records)
    out_csv = RESULT1_DIR / "Fig3_period_energy_transition_summary.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)
    return out


def period_mean_raster(var: str, years: Iterable[int]) -> Tuple[np.ndarray, dict]:
    arrs = []
    profile = None
    for year in years:
        fp = raster_path(var, year)
        if profile is None:
            profile = read_profile(fp)
        arrs.append(clean_variable(var, read_array(fp)))
    return np.nanmean(np.stack(arrs, axis=0), axis=0), profile


def robust_z(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask & np.isfinite(arr)]
    out = np.full_like(arr, np.nan, dtype="float32")
    if vals.size == 0:
        return out
    med = np.nanmedian(vals)
    q25, q75 = np.nanpercentile(vals, [25, 75])
    scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanstd(vals)
    if not np.isfinite(scale) or scale == 0:
        return out
    out[mask & np.isfinite(arr)] = ((arr[mask & np.isfinite(arr)] - med) / scale).astype("float32")
    return out


def build_spatial_heat_retention_index() -> Tuple[Path, Path]:
    weight, land_mask = load_weight_and_mask()
    del weight

    deltas = {}
    profile = None
    for var in ["Rn", "SH", "LH", "VPD"]:
        p1, profile = period_mean_raster(var, PERIODS["P1_2001_2014"])
        p3, _ = period_mean_raster(var, PERIODS["P3_2020_2024"])
        deltas[var] = p3 - p1

    valid = land_mask.copy()
    for arr in deltas.values():
        valid &= np.isfinite(arr)

    hri = (
        robust_z(deltas["Rn"], valid)
        + robust_z(deltas["SH"], valid)
        + robust_z(deltas["VPD"], valid)
        - robust_z(deltas["LH"], valid)
    ) / 4.0
    hri[~valid] = np.nan

    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    out_tif = RESULT1_DIR / "Fig3_spatial_heat_retention_transition_index_P3_minus_P1.tif"
    with rasterio.open(out_tif, "w", **out_profile) as dst:
        dst.write(hri.astype("float32"), 1)

    vals = hri[np.isfinite(hri)]
    summary = pd.DataFrame([{
        "Metric": "HeatRetentionTransitionIndex_P3_minus_P1",
        "ValidPixels": int(vals.size),
        "Mean": float(np.nanmean(vals)),
        "Median": float(np.nanmedian(vals)),
        "Positive_fraction_percent": float(np.mean(vals > 0) * 100),
        "P10": float(np.nanpercentile(vals, 10)),
        "P90": float(np.nanpercentile(vals, 90)),
    }])
    out_summary = RESULT1_DIR / "Fig3_spatial_heat_retention_transition_summary.csv"
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    out_png = RESULT1_DIR / "Fig3_spatial_heat_retention_transition_index_P3_minus_P1.png"
    plot_spatial_index(hri, out_png, title="P3 minus P1 heat-retention transition index")

    print("Saved:", out_tif)
    print("Saved:", out_summary)
    print("Saved:", out_png)
    return out_tif, out_png


def plot_spatial_index(hri: np.ndarray, out_png: Path, title: str) -> None:
    vals = hri[np.isfinite(hri)]
    lim = float(np.nanpercentile(np.abs(vals), 98)) if vals.size else 1.0
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    im = ax.imshow(
        hri,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.01)
    cbar.set_label("standardized index")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig3(df: pd.DataFrame, summary: pd.DataFrame, spatial_tif: Path) -> None:
    df = df.copy()
    df["AlbedoLoss_z"] = zscore(df["AlbedoLoss_main"])
    df["T2M_z"] = zscore(df["T2M"])
    df["Rn_z"] = zscore(df["Rn"])
    df["VPD_z"] = zscore(df["VPD"])

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.08, 1.0],
        hspace=0.42,
        wspace=0.32,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel a: time evolution of core state variables.
    ax_a.plot(df["Year"], df["AlbedoLoss_z"], color=COLORS["albedo_loss"], lw=1.5, label="Albedo loss")
    ax_a.plot(df["Year"], df["T2M_z"], color=COLORS["t2m"], lw=1.5, label="T2M")
    ax_a.plot(df["Year"], df["Rn_z"], color=COLORS["rn"], lw=1.2, label="Rn")
    ax_a.plot(df["Year"], df["VPD_z"], color=COLORS["vpd"], lw=1.2, label="VPD")
    ax_a.axhline(0, color="0.65", lw=0.6)
    add_phase_guides(ax_a, y=0.96)
    ax_a.set_ylabel("Standardized anomaly")
    ax_a.set_xlim(2000.5, 2024.5)
    ax_a.set_title("Albedo loss, warming and radiative demand rise after 2015", fontsize=8.5, pad=3)
    ax_a.legend(frameon=False, ncol=2, fontsize=7, loc="upper left")

    # Panel b: P2/P3 changes relative to P1.
    plot_metrics = ["AlbedoLoss_main", "T2M", "Rn", "SH", "LH", "VPD"]
    labels = ["Albedo\nloss", "T2M", "Rn", "SH", "LH", "VPD"]
    p1_means = summary[(summary["Period"] == "P1_2001_2014") & summary["Metric"].isin(plot_metrics)]
    p1_map = dict(zip(p1_means["Metric"], p1_means["Mean"]))
    rows = []
    for period in ["P2_2015_2019", "P3_2020_2024"]:
        for metric in plot_metrics:
            mean = float(summary[(summary["Period"] == period) & (summary["Metric"] == metric)]["Mean"].iloc[0])
            rows.append({"Period": period[:2], "Metric": metric, "Delta": mean - p1_map[metric]})
    bar_df = pd.DataFrame(rows)
    # Normalize across metrics so that units do not dominate the visual ranking.
    scale = bar_df.groupby("Metric")["Delta"].transform(lambda x: max(np.nanmax(np.abs(x)), 1e-12))
    bar_df["Delta_scaled"] = bar_df["Delta"] / scale
    x = np.arange(len(plot_metrics))
    width = 0.36
    for off, period, color in [(-width / 2, "P2", "#9ecae1"), (width / 2, "P3", "#b2182b")]:
        vals = [
            float(bar_df[(bar_df["Period"] == period) & (bar_df["Metric"] == m)]["Delta_scaled"].iloc[0])
            for m in plot_metrics
        ]
        ax_b.bar(x + off, vals, width=width, color=color, label=period, edgecolor="none")
    ax_b.axhline(0, color="0.35", lw=0.7)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Change from P1\n(metric-scaled)")
    ax_b.set_title("P3 shows the strongest heat-retention shift", fontsize=8.5, pad=3)
    ax_b.legend(frameon=False, fontsize=7)

    # Panel c: energy partitioning and heat-retention index.
    period_order = list(PERIODS.keys())
    short = ["P1", "P2", "P3"]
    ef = [float(summary[(summary["Period"] == p) & (summary["Metric"] == "EvaporativeFraction")]["Mean"].iloc[0]) for p in period_order]
    bowen = [float(summary[(summary["Period"] == p) & (summary["Metric"] == "BowenRatio")]["Mean"].iloc[0]) for p in period_order]
    hri = [float(summary[(summary["Period"] == p) & (summary["Metric"] == "HeatRetentionIndex")]["Mean"].iloc[0]) for p in period_order]
    ax_c.plot(short, ef, color=COLORS["ef"], marker="o", lw=1.6, label="Evaporative fraction")
    ax_c.plot(short, bowen, color=COLORS["bowen"], marker="s", lw=1.6, label="Bowen ratio")
    ax_c2 = ax_c.twinx()
    ax_c2.plot(short, hri, color=COLORS["hri"], marker="^", lw=1.4, label="Heat-retention index")
    ax_c.set_ylabel("Energy partition")
    ax_c2.set_ylabel("Heat-retention index")
    ax_c.set_title("Partitioning shifts away from evaporative buffering", fontsize=8.5, pad=3)
    lines = ax_c.get_lines() + ax_c2.get_lines()
    ax_c.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=6.5, loc="upper left")

    # Panel d: spatial transition index.
    with rasterio.open(spatial_tif) as src:
        spatial = src.read(1).astype("float64")
        nodata = src.nodata
        bounds = src.bounds
    if nodata is not None:
        spatial[spatial == nodata] = np.nan
    vals = spatial[np.isfinite(spatial)]
    lim = float(np.nanpercentile(np.abs(vals), 98)) if vals.size else 1.0
    im = ax_d.imshow(
        spatial,
        origin="upper",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
    )
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.set_title("Spatial pattern of P3-P1 heat retention", fontsize=8.5, pad=3)
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.035, pad=0.015)
    cbar.set_label("standardized index", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        ax.text(
            0.01,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        )

    out_base = RESULT1_DIR / "Fig3_energy_balance_to_heat_retention_transition"
    for ext, kwargs in [
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 300}),
    ]:
        out = out_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", **kwargs)
        print("Saved:", out)
    plt.close(fig)


def write_interpretation(summary: pd.DataFrame) -> None:
    def val(period: str, metric: str) -> float:
        return float(summary[(summary["Period"] == period) & (summary["Metric"] == metric)]["Mean"].iloc[0])

    lines = [
        "# Result 1.2 quantitative figure notes",
        "",
        "Main conclusion: the land surface state shifts from evaporative buffering toward heat retention after the T2M breakpoints.",
        "",
        f"- T2M increases from {val('P1_2001_2014', 'T2M'):.2f} degC in P1 to {val('P3_2020_2024', 'T2M'):.2f} degC in P3.",
        f"- Rn increases from {val('P1_2001_2014', 'Rn'):.2f} to {val('P3_2020_2024', 'Rn'):.2f} W m-2.",
        f"- VPD increases from {val('P1_2001_2014', 'VPD'):.3f} to {val('P3_2020_2024', 'VPD'):.3f}.",
        f"- Evaporative fraction changes from {val('P1_2001_2014', 'EvaporativeFraction'):.3f} to {val('P3_2020_2024', 'EvaporativeFraction'):.3f}.",
        f"- Bowen ratio changes from {val('P1_2001_2014', 'BowenRatio'):.3f} to {val('P3_2020_2024', 'BowenRatio'):.3f}.",
        "",
        "Use this figure as Fig. 3 if the revised Result 1.2 needs a direct bridge between Fig. 2 and the Result 1.3 mechanism models.",
    ]
    out_md = RESULT1_DIR / "Fig3_energy_transition_interpretation_notes_zh_ready.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", out_md)


def main() -> None:
    annual = build_annual_means()
    summary = period_summary(annual)
    spatial_tif, _ = build_spatial_heat_retention_index()
    make_fig3(annual, summary, spatial_tif)
    write_interpretation(summary)


if __name__ == "__main__":
    main()

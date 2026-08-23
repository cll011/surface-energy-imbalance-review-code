# -*- coding: utf-8 -*-
"""
Replace Fig. 3c with regional pixel-level results derived from the Fig. 3d map.

Scientific role
---------------
The revised panel quantifies the regional heterogeneity visible in Fig. 3d.
It uses the same P3-minus-P1 heat-retention transition-index raster and the
eight region boundaries used by Fig. 2e. Positive values indicate a stronger
heat-retention state in P3 than in P1; negative values indicate a weaker state.

Outputs
-------
All outputs are written to:
D:\\10_Research\\2025_Albedo_Temp\\06_文档写作\\07_第二次定稿\\code

The script writes:
1. A revised four-panel Fig. 3 in PNG, TIFF, SVG and PDF.
2. A standalone revised Fig. 3c in PNG, TIFF, SVG and PDF.
3. Regional summary statistics and the deterministic pixel sample shown.
4. A suggested revised legend/Results paragraph for manuscript editing.

Dependencies
------------
numpy, pandas, matplotlib, rasterio, pyshp, pyproj, shapely
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapefile
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform


# ---------------------------------------------------------------------------
# Paths and analysis definitions
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT1_ROOT = PROJECT_ROOT / "data" / "source_data" / "fig3"
SHAPE_ROOT = PROJECT_ROOT / "data" / "regions"
OUT_ROOT = PROJECT_ROOT / "results" / "main"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ANNUAL_CSV = RESULT1_ROOT / "Fig3_annual_energy_transition_source.csv"
PERIOD_CSV = RESULT1_ROOT / "Fig3_period_energy_transition_summary.csv"
SPATIAL_TIF = (
    RESULT1_ROOT
    / "Fig3_spatial_heat_retention_transition_index_P3_minus_P1.tif"
)

REGIONS: Sequence[Tuple[str, str, str]] = (
    ("Plateau", "Plateau", "ROI_Plateau.shp"),
    ("I Boreal Arctic", "Boreal-Arctic", "Zone_I_Boreal_Arctic.shp"),
    ("II MidLat Arid", "Mid-latitude arid", "Zone_II_MidLat_Arid.shp"),
    ("SiberianTaiga", "Siberian taiga", "ROI_SiberianTaiga.shp"),
    ("Greenland", "Greenland", "ROI_Greenland.shp"),
    ("Sahelian", "Sahel", "ROI_Sahelian.shp"),
    ("Amazon", "Amazon", "ROI_Amazon.shp"),
    (
        "III Tropical South",
        "Tropical/Southern",
        "Zone_III_Tropical_South.shp",
    ),
)

PANEL_XLIM = (-4.0, 4.0)
MAX_DISPLAY_POINTS = 450
RNG_SEED = 20260727


# ---------------------------------------------------------------------------
# Nature-style figure settings
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

COLORS: Dict[str, str] = {
    "p1": "#5B8DB8",
    "p2": "#D6A85A",
    "p3": "#B84A4A",
    "negative": "#4C88B5",
    "positive": "#C85A54",
    "neutral": "#4B4F52",
    "grid": "#E2E7EA",
    "interval": "#AEB8BE",
    "light": "#F4F6F7",
}


def period_label(year: int) -> str:
    if year <= 2014:
        return "P1"
    if year <= 2019:
        return "P2"
    return "P3"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom",
    )


def validate_inputs() -> None:
    required = [ANNUAL_CSV, PERIOD_CSV, SPATIAL_TIF]
    required.extend(SHAPE_ROOT / region[2] for region in REGIONS)
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required input files are missing:\n{joined}")


def read_shapefile_geometries(
    shp_path: Path,
    target_crs: CRS,
) -> List[dict]:
    """Read geometries with pyshp and reproject them when required."""
    reader = shapefile.Reader(str(shp_path))
    geometries = [shape(item.__geo_interface__) for item in reader.shapes()]
    geometries = [geom for geom in geometries if not geom.is_empty]
    if not geometries:
        raise ValueError(f"No valid geometries in {shp_path}")

    prj_path = shp_path.with_suffix(".prj")
    source_crs = target_crs
    if prj_path.exists():
        prj_text = prj_path.read_text(encoding="utf-8", errors="ignore").strip()
        if prj_text:
            source_crs = CRS.from_wkt(prj_text)

    if source_crs != target_crs:
        transformer = Transformer.from_crs(
            source_crs,
            target_crs,
            always_xy=True,
        )
        geometries = [
            shapely_transform(transformer.transform, geom)
            for geom in geometries
        ]

    return [mapping(geom) for geom in geometries]


def extract_regional_values() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract all valid transition-index pixels in the Fig. 2e regions.

    Statistics are pixel based, matching the valid-pixel language used in the
    manuscript and the pixel distributions in the previous Fig. 3c. The
    displayed scatter is a deterministic sample; summary statistics use all
    valid pixels.
    """
    rng = np.random.default_rng(RNG_SEED)
    summary_records: List[dict] = []
    sample_records: List[pd.DataFrame] = []

    with rasterio.open(SPATIAL_TIF) as src:
        raster_crs = CRS.from_user_input(src.crs)
        spatial = src.read(1, masked=True).filled(np.nan).astype("float64")

        for order, (source_label, display_label, shp_name) in enumerate(REGIONS):
            shp_path = SHAPE_ROOT / shp_name
            geometries = read_shapefile_geometries(shp_path, raster_crs)
            inside = geometry_mask(
                geometries,
                transform=src.transform,
                out_shape=(src.height, src.width),
                invert=True,
                all_touched=False,
            )
            values = spatial[inside]
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(
                    f"No valid transition-index pixels in region {source_label}"
                )

            summary_records.append(
                {
                    "Order": order,
                    "Region": source_label,
                    "DisplayRegion": display_label,
                    "Shapefile": shp_name,
                    "ValidPixels": int(values.size),
                    "Mean": float(np.mean(values)),
                    "Median": float(np.median(values)),
                    "P10": float(np.percentile(values, 10)),
                    "Q25": float(np.percentile(values, 25)),
                    "Q75": float(np.percentile(values, 75)),
                    "P90": float(np.percentile(values, 90)),
                    "PositivePixels": int(np.count_nonzero(values > 0)),
                    "PositiveFractionPercent": float(np.mean(values > 0) * 100),
                }
            )

            sample_n = min(MAX_DISPLAY_POINTS, values.size)
            sample_index = rng.choice(values.size, sample_n, replace=False)
            sample_values = values[sample_index]
            sample_records.append(
                pd.DataFrame(
                    {
                        "Order": order,
                        "Region": source_label,
                        "DisplayRegion": display_label,
                        "TransitionIndex": sample_values,
                    }
                )
            )

    summary = pd.DataFrame(summary_records).sort_values("Order")
    sample = pd.concat(sample_records, ignore_index=True)
    return summary, sample


def draw_regional_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    sample: pd.DataFrame,
    add_panel_letter: bool = True,
    show_key: bool = False,
) -> None:
    """
    Draw a compact regional scatter-and-interval panel.

    Faint points are the deterministic pixel sample. Thin and thick horizontal
    segments are P10-P90 and Q25-Q75. Circles mark medians and diamonds mark
    means. Text on the right gives the percentage of all valid pixels > 0.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    y_positions = np.arange(len(summary))[::-1]
    display_to_y = {
        row.DisplayRegion: y
        for row, y in zip(summary.itertuples(index=False), y_positions)
    }

    ax.axvspan(0, PANEL_XLIM[1], color="#FBEDEC", zorder=0)
    ax.axvspan(PANEL_XLIM[0], 0, color="#EEF5FA", zorder=0)
    ax.axvline(0, color="#565B5E", linewidth=0.8, linestyle="--", zorder=2)

    for row in summary.itertuples(index=False):
        y = display_to_y[row.DisplayRegion]
        sub = sample.loc[
            sample["DisplayRegion"] == row.DisplayRegion,
            "TransitionIndex",
        ].to_numpy()
        displayed = np.clip(sub, PANEL_XLIM[0], PANEL_XLIM[1])
        jitter = rng.normal(0, 0.065, displayed.size)
        point_colors = np.where(
            sub > 0,
            COLORS["positive"],
            COLORS["negative"],
        )
        ax.scatter(
            displayed,
            y + jitter,
            s=4,
            c=point_colors,
            alpha=0.16,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )

        p10 = np.clip(row.P10, *PANEL_XLIM)
        p90 = np.clip(row.P90, *PANEL_XLIM)
        q25 = np.clip(row.Q25, *PANEL_XLIM)
        q75 = np.clip(row.Q75, *PANEL_XLIM)
        median = np.clip(row.Median, *PANEL_XLIM)
        mean = np.clip(row.Mean, *PANEL_XLIM)
        mean_color = (
            COLORS["positive"] if row.Mean > 0 else COLORS["negative"]
        )

        ax.plot(
            [p10, p90],
            [y, y],
            color=COLORS["interval"],
            linewidth=0.9,
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            [q25, q75],
            [y, y],
            color=COLORS["neutral"],
            linewidth=3.2,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            median,
            y,
            s=18,
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.8,
            zorder=5,
        )
        ax.scatter(
            mean,
            y,
            s=23,
            marker="D",
            facecolor=mean_color,
            edgecolor="white",
            linewidth=0.5,
            zorder=6,
        )
        ax.text(
            PANEL_XLIM[1] - 0.06,
            y + 0.20,
            f"{row.PositiveFractionPercent:.0f}% > 0",
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#4B4F52",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.68,
                "pad": 0.35,
            },
        )

    ax.set_xlim(PANEL_XLIM)
    ax.set_ylim(-0.6, len(summary) - 0.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["DisplayRegion"], fontsize=6.6)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("P3-minus-P1 heat-retention transition index", fontsize=7.2)
    ax.set_title(
        "Regional heat-retention transitions",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)

    if show_key:
        ax.text(
            0.0,
            1.015,
            (
                "Thin line, P10-P90; thick line, Q25-Q75; "
                "circle, median; diamond, mean"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.0,
            color="#4B4F52",
        )
    if add_panel_letter:
        panel_label(ax, "c")


def read_spatial() -> Tuple[np.ma.MaskedArray, rasterio.coords.BoundingBox]:
    with rasterio.open(SPATIAL_TIF) as src:
        array = src.read(1, masked=True).astype("float32")
        bounds = src.bounds
    return array, bounds


def draw_panel_a(ax: plt.Axes, annual: pd.DataFrame) -> None:
    annual = annual.copy()
    annual["Period"] = annual["Year"].map(period_label)
    for column in [
        "AlbedoLoss_main",
        "HeatRetentionIndex",
        "VPD",
    ]:
        standard_deviation = annual[column].std(ddof=0)
        annual[column + "_z"] = (
            annual[column] - annual[column].mean()
        ) / standard_deviation

    period_colors = {
        "P1": COLORS["p1"],
        "P2": COLORS["p2"],
        "P3": COLORS["p3"],
    }
    for period_name in ["P1", "P2", "P3"]:
        subset = annual.loc[annual["Period"] == period_name]
        sizes = 35 + 42 * (
            subset["VPD_z"] - annual["VPD_z"].min()
        ) / (annual["VPD_z"].max() - annual["VPD_z"].min())
        ax.scatter(
            subset["AlbedoLoss_main_z"],
            subset["HeatRetentionIndex_z"],
            s=sizes,
            color=period_colors[period_name],
            alpha=0.70,
            edgecolor="white",
            linewidth=0.5,
            label=period_name,
        )

    centroids = (
        annual.groupby("Period")[
            ["AlbedoLoss_main_z", "HeatRetentionIndex_z"]
        ]
        .mean()
        .loc[["P1", "P2", "P3"]]
    )
    ax.plot(
        centroids["AlbedoLoss_main_z"],
        centroids["HeatRetentionIndex_z"],
        color="#242424",
        linewidth=1.3,
        zorder=3,
    )
    ax.annotate(
        "",
        xy=centroids.iloc[2].values,
        xytext=centroids.iloc[0].values,
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#242424"},
    )
    ax.axhline(0, color="#C9D1D5", linewidth=0.8)
    ax.axvline(0, color="#C9D1D5", linewidth=0.8)
    ax.set_xlabel("Albedo-loss anomaly (z)")
    ax.set_ylabel("Heat-retention index (z)")
    ax.set_title(
        "Annual state trajectory",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    ax.legend(title="Period", fontsize=7, title_fontsize=7, loc="upper left")
    panel_label(ax, "a")


def draw_panel_b(ax: plt.Axes, annual: pd.DataFrame) -> None:
    annual = annual.copy()
    annual["Period"] = annual["Year"].map(period_label)
    metrics = [
        "T2M",
        "Rn",
        "SWabs_MODIS",
        "VPD",
        "SH",
        "LH",
        "EvaporativeFraction",
    ]
    labels = ["T2M", "Rn", "SWabs", "VPD", "SH", "LH", "EF"]
    metric_colors = [
        "#333333",
        COLORS["p3"],
        "#C87B42",
        "#9B4E8B",
        "#D18428",
        "#4A9B8E",
        COLORS["negative"],
    ]
    standard_deviation = annual[metrics].std(ddof=0)
    p1_mean = annual.loc[annual["Period"] == "P1", metrics].mean()
    p2_mean = annual.loc[annual["Period"] == "P2", metrics].mean()
    p3_mean = annual.loc[annual["Period"] == "P3", metrics].mean()
    p2_z = ((p2_mean - p1_mean) / standard_deviation).values
    p3_z = ((p3_mean - p1_mean) / standard_deviation).values

    y_positions = np.arange(len(metrics))[::-1]
    for index, (y, p2_value, p3_value, color) in enumerate(
        zip(y_positions, p2_z, p3_z, metric_colors)
    ):
        ax.plot(
            [p2_value, p3_value],
            [y, y],
            color="#B8C0C6",
            linewidth=1.3,
            zorder=1,
        )
        ax.scatter(
            p2_value,
            y,
            s=32,
            color=COLORS["p2"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        p3_color = COLORS["negative"] if labels[index] == "EF" else color
        ax.scatter(
            p3_value,
            y,
            s=40,
            color=p3_color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )

    ax.axvline(0, color="#5F6468", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Period mean change from P1 (standard deviations)")
    ax.set_title(
        "Energy-pathway perturbation",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["p2"],
            markeredgecolor="white",
            markersize=5,
            label="P2",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["p3"],
            markeredgecolor="white",
            markersize=5,
            label="P3",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=7,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    ax.grid(axis="x", color="#E6EAED", linewidth=0.6)
    panel_label(ax, "b")


def draw_panel_d(
    ax: plt.Axes,
    spatial: np.ma.MaskedArray,
    bounds: rasterio.coords.BoundingBox,
    figure: plt.Figure,
) -> None:
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    image = ax.imshow(
        np.ma.masked_invalid(spatial),
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks([-120, 0, 120])
    ax.set_yticks([-40, 0, 40, 80])
    ax.tick_params(labelsize=7, length=2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Where heat retention strengthened",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    color_axis = inset_axes(
        ax,
        width="3%",
        height="72%",
        loc="center right",
        borderpad=1.2,
    )
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("Index", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)
    panel_label(ax, "d")


def save_publication_figure(
    figure: plt.Figure,
    stem: Path,
    dpi: int = 600,
) -> None:
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=dpi,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def make_standalone_panel(
    summary: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(figsize=(7.1, 4.15))
    draw_regional_panel(
        axis,
        summary,
        sample,
        add_panel_letter=False,
        show_key=False,
    )
    figure.text(
        0.01,
        0.01,
        (
            "Thin lines, P10-P90; thick lines, Q25-Q75; circles, medians; "
            "diamonds, means.\nPoints are a deterministic sample for display; "
            "intervals and percentages use all valid pixels. Values are "
            "clipped at +/-4 for display only."
        ),
        fontsize=6.2,
        color="#4D4D4D",
    )
    figure.subplots_adjust(left=0.19, right=0.98, top=0.90, bottom=0.20)
    save_publication_figure(
        figure,
        OUT_ROOT / "Fig3c_regional_heat_retention_transition",
    )
    plt.close(figure)


def make_full_figure(
    summary: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    annual = pd.read_csv(ANNUAL_CSV)
    _ = pd.read_csv(PERIOD_CSV)  # Explicit source-presence check.
    spatial, bounds = read_spatial()

    figure = plt.figure(figsize=(7.2, 6.2), constrained_layout=False)
    grid = GridSpec(
        2,
        2,
        figure=figure,
        width_ratios=[1.06, 1.15],
        height_ratios=[1.0, 1.07],
        hspace=0.36,
        wspace=0.34,
    )

    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])

    draw_panel_a(axis_a, annual)
    draw_panel_b(axis_b, annual)
    draw_regional_panel(axis_c, summary, sample, show_key=False)
    draw_panel_d(axis_d, spatial, bounds, figure)

    figure.suptitle(
        (
            "Land energy partitioning shifts toward heat retention "
            "after the warming breakpoints"
        ),
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.01,
        (
            "P1=2001-2014, P2=2015-2019 and P3=2020-2024. "
            "Panel b uses standard deviations of annual global land means. "
            "Panel c uses the Fig. 2e regions and the Fig. 3d transition index."
        ),
        fontsize=6.3,
        color="#4D4D4D",
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.91,
        bottom=0.085,
    )

    save_publication_figure(
        figure,
        OUT_ROOT / "Fig3_energy_partition_regional_revised",
    )
    plt.close(figure)


def write_manuscript_text(summary: pd.DataFrame) -> None:
    lookup = summary.set_index("Region")
    amazon = lookup.loc["Amazon"]
    sahel = lookup.loc["Sahelian"]
    plateau = lookup.loc["Plateau"]
    boreal = lookup.loc["I Boreal Arctic"]
    siberian = lookup.loc["SiberianTaiga"]
    greenland = lookup.loc["Greenland"]

    legend = (
        "Fig. 3 | Albedo loss increases absorbed energy as evaporative "
        "buffering weakens. a, Annual albedo-loss anomaly versus the "
        "heat-retention index; centroids show P1-P3. b, Period changes from "
        "P1 in standard deviations of annual global land means. c, Regional "
        "pixel distributions of the P3-minus-P1 heat-retention transition "
        "index for the eight regions used in Fig. 2e. Faint points show a "
        "deterministic display sample; thin and thick intervals show P10-P90 "
        "and Q25-Q75, circles mark medians, diamonds mark means, and labels "
        "give the percentage of all valid pixels above zero. d, Spatial "
        "distribution of the same transition index. Rn, net radiation; "
        "SWabs, absorbed shortwave radiation; EF, evaporative fraction; LH "
        "and SH, latent and sensible heat fluxes; VPD, vapour-pressure deficit."
    )

    paragraph = (
        "The transition to heat retention was regionally heterogeneous. "
        f"The Amazon showed the strongest positive shift (mean "
        f"{amazon.Mean:.2f}; {amazon.PositiveFractionPercent:.1f}% of valid "
        f"pixels above zero), followed by the Sahel (mean {sahel.Mean:.2f}; "
        f"{sahel.PositiveFractionPercent:.1f}%) and the Plateau (mean "
        f"{plateau.Mean:.2f}; {plateau.PositiveFractionPercent:.1f}%). "
        f"By contrast, the Boreal-Arctic and Siberian-taiga regions retained "
        f"negative means ({boreal.Mean:.2f} and {siberian.Mean:.2f}, "
        f"respectively), whereas Greenland remained close to neutral "
        f"(mean {greenland.Mean:.2f}; "
        f"{greenland.PositiveFractionPercent:.1f}% positive; Fig. 3c,d). "
        "This contrast complements Fig. 2e: strong inverse annual "
        "albedo-temperature coupling at high latitudes did not necessarily "
        "coincide with a positive shift in the heat-retention index, whereas "
        "several tropical, dryland and high-elevation regions showed a larger "
        "energy-partition response."
    )

    note = (
        "Suggested revised legend\n"
        "========================\n"
        f"{legend}\n\n"
        "Suggested replacement for the third Results paragraph\n"
        "=====================================================\n"
        f"{paragraph}\n\n"
        "Statistical boundary\n"
        "====================\n"
        "Regional distributions are unweighted valid-pixel summaries, matching "
        "the pixel-level Fig. 3d raster and the manuscript's percentage-of-"
        "pixels wording. They are not global land-area-weighted means. The "
        "display is clipped at +/-4, but all reported statistics use the "
        "unclipped values."
    )
    (OUT_ROOT / "Fig3c_revised_caption_and_results_text.txt").write_text(
        note,
        encoding="utf-8",
    )


def main() -> None:
    validate_inputs()
    summary, sample = extract_regional_values()

    summary.to_csv(
        OUT_ROOT / "Fig3c_regional_heat_retention_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sample.to_csv(
        OUT_ROOT / "Fig3c_regional_heat_retention_display_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_standalone_panel(summary, sample)
    make_full_figure(summary, sample)
    write_manuscript_text(summary)

    print(summary.to_string(index=False))
    print(f"Outputs written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()

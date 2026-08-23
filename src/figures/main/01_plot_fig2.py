# -*- coding: utf-8 -*-
"""Reproduce Fig. 2 from the final temporal and spatial source products.

Run ``src/analysis/01b_fig2_spatial_coupling.py`` once before this script to
create the pixel-wise correlation, directional classes and regional summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapefile
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPORAL_ROOT = PROJECT_ROOT / "data" / "source_data" / "fig2"
SPATIAL_ROOT = PROJECT_ROOT / "results" / "fig2_source"
REGION_ROOT = PROJECT_ROOT / "data" / "regions"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "main"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

ANNUAL_CSV = TEMPORAL_ROOT / "Fig2_updated_global_land_annual_series.csv"
SEGMENT_CSV = TEMPORAL_ROOT / "Fig2b_changepoint_segment_summary_updated.csv"
CORRELATION_TIF = (
    SPATIAL_ROOT / "Fig2c_spatial_raw_corr_GLASS_albedo_T2M_lsm_updated.tif"
)
CLASS_TIF = SPATIAL_ROOT / "Fig2d_directional_classes_GLASS_T2M_lsm_updated.tif"
REGIONAL_CSV = SPATIAL_ROOT / "Fig2e_regional_correlation_summary_lsm_updated.csv"
SPATIAL_SUMMARY_CSV = SPATIAL_ROOT / "Fig2_spatial_summary.csv"
WORLD_SHP = REGION_ROOT / "ne_10m_global_disolve.shp"

T2M_COLOR = "#B9473F"
ALBEDO_COLOR = "#4D95C6"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "axes.linewidth": 0.65,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def mm_to_inches(width: float, height: float) -> Tuple[float, float]:
    return width / 25.4, height / 25.4


def panel_label(ax: plt.Axes, label: str, x: float = -0.09) -> None:
    ax.text(
        x,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def validate_inputs() -> None:
    required = [
        ANNUAL_CSV,
        SEGMENT_CSV,
        CORRELATION_TIF,
        CLASS_TIF,
        REGIONAL_CSV,
        SPATIAL_SUMMARY_CSV,
        WORLD_SHP,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Fig. 2 input(s). Run the spatial analysis first:\n"
            + "\n".join(missing)
        )


def _world_outline(ax: plt.Axes) -> None:
    reader = shapefile.Reader(str(WORLD_SHP))
    for item in reader.shapes():
        points = np.asarray(item.points)
        parts = list(item.parts) + [len(points)]
        for start, end in zip(parts[:-1], parts[1:]):
            segment = points[start:end]
            if segment.size:
                ax.plot(segment[:, 0], segment[:, 1], color="#50575C", lw=0.22)


def draw_panel_a(ax: plt.Axes, annual: pd.DataFrame) -> None:
    years = annual["Year"].to_numpy(dtype=float)
    temperature = annual["T2M_C"].to_numpy(dtype=float)
    albedo = annual["GLASS_Albedo"].to_numpy(dtype=float)
    centre = years.mean()
    x = years - centre
    dense_year = np.linspace(years.min(), years.max(), 300)
    dense_x = dense_year - centre
    fit_temperature = np.polyval(np.polyfit(x, temperature, 2), dense_x)
    fit_albedo = np.polyval(np.polyfit(x, albedo, 2), dense_x)

    ax_albedo = ax.twinx()
    ax.plot(years, temperature, "^-", color=T2M_COLOR, lw=0.8, ms=3.0, alpha=0.9)
    ax.plot(dense_year, fit_temperature, color=T2M_COLOR, lw=2.0)
    ax_albedo.plot(years, albedo, "o-", color=ALBEDO_COLOR, lw=0.8, ms=2.8, alpha=0.75)
    ax_albedo.plot(dense_year, fit_albedo, color=ALBEDO_COLOR, lw=2.0)
    ax.axvline(2015, color="#555B60", ls=(0, (3, 2)), lw=0.8)
    ax.axvline(2020, color="#8C9297", ls=(0, (2, 2)), lw=0.7)
    ax.set_xlim(2000.5, 2024.5)
    ax.set_xticks([2001, 2006, 2011, 2015, 2020, 2024])
    ax.set_ylabel("T2M (°C)", color=T2M_COLOR)
    ax_albedo.set_ylabel("GLASS surface albedo", color=ALBEDO_COLOR)
    ax.tick_params(axis="y", colors=T2M_COLOR)
    ax_albedo.tick_params(axis="y", colors=ALBEDO_COLOR)
    ax.grid(axis="y", color="#E3E7EA", lw=0.5)
    ax.set_title("Annual temperature and surface albedo", loc="left", fontweight="bold")
    handles = [
        Line2D([0], [0], color=T2M_COLOR, marker="^", label="T2M"),
        Line2D([0], [0], color=ALBEDO_COLOR, marker="o", label="GLASS albedo"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=2)
    panel_label(ax, "a", x=-0.12)


def draw_segment_axis(
    ax: plt.Axes,
    annual: pd.DataFrame,
    segment: pd.Series,
    variable: str,
    color: str,
    ylabel: str,
) -> None:
    years = annual["Year"].to_numpy(dtype=int)
    values = annual[variable].to_numpy(dtype=float)
    ax.plot(years, values, "o-", color=color, lw=0.8, ms=2.7, alpha=0.75)
    ax.hlines(
        float(segment.P1_mean),
        int(segment.P1_start),
        int(segment.P1_end) + 0.8,
        color=color,
        lw=2.1,
    )
    ax.hlines(
        float(segment.P2_mean),
        int(segment.P2_start) - 0.2,
        int(segment.P2_end),
        color=color,
        lw=2.1,
    )
    ax.axvline(2015, color="#555B60", ls=(0, (3, 2)), lw=0.8)
    ax.axvline(2020, color="#8C9297", ls=(0, (2, 2)), lw=0.7)
    ax.set_xlim(2000.5, 2024.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E3E7EA", lw=0.5)


def draw_panel_b(ax_top: plt.Axes, ax_bottom: plt.Axes, annual: pd.DataFrame) -> None:
    segments = pd.read_csv(SEGMENT_CSV).set_index("Variable")
    draw_segment_axis(
        ax_top,
        annual,
        segments.loc["T2M"],
        "T2M_C",
        T2M_COLOR,
        "T2M (°C)",
    )
    draw_segment_axis(
        ax_bottom,
        annual,
        segments.loc["GLASS_Albedo"],
        "GLASS_Albedo",
        ALBEDO_COLOR,
        "GLASS albedo",
    )
    ax_top.set_xticklabels([])
    ax_bottom.set_xticks([2001, 2006, 2011, 2015, 2020, 2024])
    ax_bottom.set_xlabel("Year")
    ax_top.set_title(
        "BIC-selected breakpoint and recent-period boundary",
        loc="left",
        fontweight="bold",
    )
    ax_top.text(
        0.02,
        0.90,
        "BIC breakpoint: 2015",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
    )
    ax_bottom.text(
        0.98,
        0.08,
        "2020: predefined P3 boundary",
        transform=ax_bottom.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
        color="#626970",
    )
    panel_label(ax_top, "b", x=-0.12)


def draw_raster_panel(
    ax: plt.Axes, path: Path, categorical: bool, title: str
) -> None:
    with rasterio.open(path) as source:
        array = source.read(1, masked=True).filled(np.nan)
        bounds = source.bounds
    if categorical:
        cmap = ListedColormap(["#C5524B", "#E5A23A", "#7992AD", "#D5D8DA"])
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    else:
        cmap = "RdBu_r"
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    image = ax.imshow(
        array,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )
    _world_outline(ax)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks([-120, 0, 120])
    ax.set_yticks([-40, 0, 40, 80])
    ax.set_title(title, loc="left", fontweight="bold")
    if categorical:
        handles = [
            Patch(color="#C5524B", label="C1: darkening + warming, r < 0"),
            Patch(color="#E5A23A", label="C2: darkening + warming, r ≥ 0"),
            Patch(color="#7992AD", label="C3: warming without darkening"),
            Patch(color="#D5D8DA", label="C4: other"),
        ]
        ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=5.3, ncol=2)
    else:
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.03, pad=0.015)
        colorbar.set_label("r(GLASS albedo, T2M)")


def draw_panel_e(ax: plt.Axes, regional: pd.DataFrame) -> None:
    data = regional.sort_values("Order")
    positions = np.arange(len(data))[::-1]
    ax.axvline(0, color="#50575C", lw=0.8)
    for y, row in zip(positions, data.itertuples(index=False)):
        ax.plot([row.P10, row.P90], [y, y], color="#AAB3BA", lw=0.9)
        ax.plot([row.Q25, row.Q75], [y, y], color="#6C879D", lw=3.0)
        ax.scatter(row.Median, y, s=17, facecolor="white", edgecolor="#30363A", zorder=3)
        ax.text(
            0.98,
            y,
            f"{row.Negative_fraction_percent:.0f}%",
            ha="right",
            va="center",
            fontsize=5.8,
        )
    ax.set_xlim(-1, 1)
    ax.set_yticks(positions)
    ax.set_yticklabels(data["Region"])
    ax.set_xlabel("r(GLASS albedo, T2M)")
    ax.set_title("Regional coupling distributions", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E3E7EA", lw=0.5)
    ax.tick_params(axis="y", length=0)
    ax.text(
        0.99,
        1.01,
        "negative pixels",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
    )
    panel_label(ax, "e", x=-0.055)


def main() -> None:
    validate_inputs()
    annual = pd.read_csv(ANNUAL_CSV)
    regional = pd.read_csv(REGIONAL_CSV)
    spatial_summary = pd.read_csv(SPATIAL_SUMMARY_CSV)

    fig = plt.figure(figsize=mm_to_inches(183, 178))
    grid = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[0.95, 1.0, 0.78],
        left=0.08,
        right=0.98,
        top=0.97,
        bottom=0.07,
        hspace=0.45,
        wspace=0.32,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    b_subgrid = grid[0, 1].subgridspec(2, 1, hspace=0.08)
    ax_b_top = fig.add_subplot(b_subgrid[0, 0])
    ax_b_bottom = fig.add_subplot(b_subgrid[1, 0])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_e = fig.add_subplot(grid[2, :])

    draw_panel_a(ax_a, annual)
    draw_panel_b(ax_b_top, ax_b_bottom, annual)
    draw_raster_panel(ax_c, CORRELATION_TIF, False, "Pixel-wise coupling")
    panel_label(ax_c, "c", x=-0.12)
    draw_raster_panel(ax_d, CLASS_TIF, True, "Directional classification")
    panel_label(ax_d, "d", x=-0.12)
    draw_panel_e(ax_e, regional)

    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if extension == "png" else {}
        fig.savefig(
            OUTPUT_ROOT / f"Fig2_surface_albedo_temperature_reproduced.{extension}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)
    spatial_summary.to_csv(
        OUTPUT_ROOT / "Fig2_reproduction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()

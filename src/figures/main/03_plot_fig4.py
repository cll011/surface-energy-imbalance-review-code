# -*- coding: utf-8 -*-
"""Reproduce the final four-panel Fig. 4 from archived source data.

Panels show (a) P1/P3 fitted XGBoost-SHAP contributions, (b) the pathway
diagram used in the manuscript, (c) regional AlbedoLoss contribution gains,
and (d) the global spatial AlbedoLoss contribution gain. SHAP values are
fitted predictor contributions and are not causal effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

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
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "data" / "source_data" / "fig4"
REGION_ROOT = PROJECT_ROOT / "data" / "regions"
ASSET_ROOT = PROJECT_ROOT / "assets"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "main"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

PERIOD_CSV = SOURCE_ROOT / "fig4_period_contributions_final.csv"
GAIN_CSV = SOURCE_ROOT / "fig4_contribution_gain_final.csv"
REGIONAL_CSV = SOURCE_ROOT / "fig4_regional_contributions_final.csv"
SPATIAL_TIF = SOURCE_ROOT / "fig4_albedo_loss_shap_gain_final.tif"
SPATIAL_SUMMARY_CSV = SOURCE_ROOT / "fig4_spatial_gain_summary_final.csv"
MODEL_METRICS_CSV = SOURCE_ROOT / "xgboost_model_metrics_final.csv"
MECHANISM_PNG = ASSET_ROOT / "fig4b_mechanism_path.png"
WORLD_SHP = REGION_ROOT / "ne_10m_global_disolve.shp"

FEATURES: Sequence[str] = ("AlbedoLoss", "LH", "SH", "SM", "Rn")
FEATURE_LABELS = {
    "AlbedoLoss": "Albedo loss",
    "LH": "Latent heat",
    "SH": "Sensible heat",
    "SM": "Soil moisture",
    "Rn": "Net radiation",
}
FEATURE_COLORS = {
    "AlbedoLoss": "#B94848",
    "LH": "#2F8E9E",
    "SH": "#D9842B",
    "SM": "#758A3A",
    "Rn": "#586EA8",
}

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


def panel_label(ax: plt.Axes, label: str, x: float = -0.08) -> None:
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
        PERIOD_CSV,
        GAIN_CSV,
        REGIONAL_CSV,
        SPATIAL_TIF,
        SPATIAL_SUMMARY_CSV,
        MODEL_METRICS_CSV,
        MECHANISM_PNG,
        WORLD_SHP,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Fig. 4 inputs:\n" + "\n".join(missing))


def contribution_panel(
    ax: plt.Axes, contributions: pd.DataFrame, gains: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    p1 = contributions.loc[
        contributions["Period"].eq("P1_2001_2014")
    ].set_index("Feature")
    p3 = contributions.loc[
        contributions["Period"].eq("P3_2020_2024")
    ].set_index("Feature")
    gain = gains.set_index("Feature")
    positions = np.arange(len(FEATURES))[::-1]
    values: Dict[str, Dict[str, float]] = {}

    for y, feature in zip(positions, FEATURES):
        value_p1 = float(p1.loc[feature, "MeanAbs_SHAP"])
        value_p3 = float(p3.loc[feature, "MeanAbs_SHAP"])
        delta = float(gain.loc[feature, "P3_minus_P1_MeanAbs_SHAP"])
        values[feature] = {"P1": value_p1, "P3": value_p3, "gain": delta}
        ax.plot([value_p1, value_p3], [y, y], color="#B6BDC3", lw=1.5)
        ax.scatter(
            value_p1,
            y,
            s=28,
            facecolor="white",
            edgecolor="#858D93",
            linewidth=0.8,
            zorder=3,
        )
        ax.scatter(
            value_p3,
            y,
            s=34,
            facecolor=FEATURE_COLORS[feature],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        ax.text(
            0.158,
            y,
            f"{delta:+.3f}",
            ha="right",
            va="center",
            color=FEATURE_COLORS[feature],
            fontweight="bold" if feature == "AlbedoLoss" else "normal",
            fontsize=6.0,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([FEATURE_LABELS[item] for item in FEATURES])
    ax.set_xlim(0.02, 0.165)
    ax.set_xticks([0.04, 0.08, 0.12, 0.16])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Fitted land-surface contributions", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E2E7EA", linewidth=0.55)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#858D93",
            label="P1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#B94848",
            markeredgecolor="white",
            label="P3",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2)
    panel_label(ax, "a", x=-0.13)
    return values


def mechanism_panel(ax: plt.Axes) -> None:
    with Image.open(MECHANISM_PNG) as image:
        mechanism = np.asarray(image.convert("RGB"))
    ax.imshow(mechanism, interpolation="lanczos")
    ax.set_axis_off()
    panel_label(ax, "b", x=-0.03)


def regional_panel(ax: plt.Axes, regional: pd.DataFrame) -> None:
    data = regional.loc[regional["Feature"].eq("AlbedoLoss")].copy()
    order = [
        "Plateau",
        "Boreal-Arctic",
        "Mid-latitude arid",
        "Siberian taiga",
        "Greenland",
        "Sahel",
        "Amazon",
        "Tropical/Southern",
    ]
    data["DisplayRegion"] = pd.Categorical(
        data["DisplayRegion"], categories=order, ordered=True
    )
    data = data.sort_values("DisplayRegion")
    positions = np.arange(len(data))[::-1]

    for y, row in zip(positions, data.itertuples(index=False)):
        ax.plot(
            [row.P1_MeanAbs_SHAP, row.P3_MeanAbs_SHAP],
            [y, y],
            color="#B7BEC4",
            lw=1.35,
        )
        ax.scatter(
            row.P1_MeanAbs_SHAP,
            y,
            s=20,
            facecolor="white",
            edgecolor="#858D93",
            linewidth=0.7,
            zorder=3,
        )
        ax.scatter(
            row.P3_MeanAbs_SHAP,
            y,
            s=27,
            facecolor="#B94848",
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        ax.text(
            0.243,
            y,
            f"{row.PositiveFractionPercent:.1f}%",
            ha="right",
            va="center",
            fontsize=5.7,
            color="#5F666C",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([str(item) for item in data["DisplayRegion"]])
    ax.set_xlim(0.02, 0.25)
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_xlabel("Albedo-loss mean |SHAP value|")
    ax.set_title("Regional contribution change", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E2E7EA", linewidth=0.55)
    ax.tick_params(axis="y", length=0)
    ax.text(
        0.99,
        1.01,
        "pixels with P3−P1 gain > 0",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color="#5F666C",
    )
    panel_label(ax, "c", x=-0.10)


def _world_outline(ax: plt.Axes) -> None:
    reader = shapefile.Reader(str(WORLD_SHP))
    for item in reader.shapes():
        points = np.asarray(item.points)
        parts = list(item.parts) + [len(points)]
        for start, end in zip(parts[:-1], parts[1:]):
            segment = points[start:end]
            if segment.size:
                ax.plot(segment[:, 0], segment[:, 1], color="#545B60", lw=0.25)


def spatial_panel(ax: plt.Axes) -> Dict[str, float]:
    with rasterio.open(SPATIAL_TIF) as source:
        array = source.read(1, masked=True).astype("float64").filled(np.nan)
        bounds = source.bounds
    finite = array[np.isfinite(array)]
    limit = max(float(np.nanpercentile(np.abs(finite), 98)), 0.05)
    image = ax.imshow(
        array,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
        rasterized=True,
    )
    _world_outline(ax)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks([-120, 0, 120])
    ax.set_yticks([-40, 0, 40, 80])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial AlbedoLoss contribution gain", loc="left", fontweight="bold")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label("P3−P1 mean |SHAP value|")
    panel_label(ax, "d", x=-0.055)
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "positive_percent": float(np.mean(finite > 0) * 100),
    }


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_ROOT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(
        OUTPUT_ROOT / f"{stem}.tif",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    validate_inputs()
    contributions = pd.read_csv(PERIOD_CSV)
    gains = pd.read_csv(GAIN_CSV)
    regional = pd.read_csv(REGIONAL_CSV)
    spatial_summary = pd.read_csv(SPATIAL_SUMMARY_CSV)
    metrics = pd.read_csv(MODEL_METRICS_CSV)

    fig = plt.figure(figsize=mm_to_inches(183, 175))
    grid = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[0.9, 1.1],
        height_ratios=[0.85, 1.15],
        left=0.08,
        right=0.98,
        top=0.97,
        bottom=0.07,
        hspace=0.32,
        wspace=0.30,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    values = contribution_panel(ax_a, contributions, gains)
    mechanism_panel(ax_b)
    regional_panel(ax_c, regional)
    map_summary = spatial_panel(ax_d)
    save_figure(fig, "Fig4_albedo_loss_contribution_feedback_shift_reproduced")
    plt.close(fig)

    final_albedo = spatial_summary.loc[
        spatial_summary["Feature"].eq("AlbedoLoss")
    ].iloc[0]
    background_r2 = float(
        metrics.loc[
            metrics["Model"].eq("Model1_background_stripping"), "R2_test"
        ].iloc[0]
    )
    contribution_r2 = float(
        metrics.loc[
            metrics["Model"].eq("Model2_main_SHAP_core"), "R2_test"
        ].iloc[0]
    )
    report = pd.DataFrame(
        [
            {
                "AlbedoLoss_P1": values["AlbedoLoss"]["P1"],
                "AlbedoLoss_P3": values["AlbedoLoss"]["P3"],
                "AlbedoLoss_gain": values["AlbedoLoss"]["gain"],
                "SH_gain": values["SH"]["gain"],
                "spatial_positive_percent": final_albedo.PositiveFractionPercent,
                "spatial_unweighted_mean": final_albedo.UnweightedMean,
                "spatial_valid_pixels": int(final_albedo.N),
                "background_test_R2": background_r2,
                "contribution_test_R2": contribution_r2,
                "map_check_positive_percent": map_summary["positive_percent"],
            }
        ]
    )
    report.to_csv(
        OUTPUT_ROOT / "Fig4_reproduction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Fig. S7-1: spatial P3-minus-P1 gains in fitted SHAP contributions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import figure_common as F


STEM = "Fig_S7_1_SHAP_spatial_gain"
FEATURES = ["AlbedoLoss", "Rn", "SM", "LH", "SH"]
LABELS = {
    "AlbedoLoss": "Albedo loss",
    "Rn": "Net radiation",
    "SM": "Soil moisture",
    "LH": "Latent heat",
    "SH": "Sensible heat",
}


def add_manuscript_map(ax, array, norm, *, show_x, show_y):
    """Use the rectilinear projection and 2:1 map aspect of the main figures."""
    image = ax.imshow(
        np.ma.masked_invalid(array),
        extent=(-180, 180, -90, 90),
        origin="upper",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        rasterized=True,
        aspect="equal",
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    F.add_world_outline(ax)
    ax.set_xticks([-120, 0, 120])
    ax.set_yticks([-60, 0, 60])
    if show_x:
        ax.set_xlabel("Longitude")
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_ylabel("Latitude")
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=2.4, width=0.7, direction="out")
    return image


def main() -> None:
    arrays = {}
    rows = []
    for feature in FEATURES:
        gain, _ = F.read_raster(
            F.SOURCE_DATA / f"FigS7_1_{feature}_mean_abs_SHAP_gain_P3_minus_P1.tif"
        )
        arrays[feature] = gain
        values = gain[np.isfinite(gain)]
        rows.append(
            {
                "Feature": feature,
                "DisplayFeature": LABELS[feature],
                "N": int(values.size),
                "UnweightedMean": float(np.mean(values)),
                "Median": float(np.median(values)),
                "PositiveFractionPercent": float(np.mean(values > 0) * 100),
            }
        )

    summary = pd.DataFrame(rows).set_index("Feature").reindex(FEATURES).reset_index()
    common_limit = max(F.robust_symmetric_limit(array, 98) for array in arrays.values())
    common_norm = TwoSlopeNorm(vmin=-common_limit, vcenter=0, vmax=common_limit)

    fig = plt.figure(figsize=(7.2, 5.85))
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 1.0, 1.0, 0.075],
        hspace=0.42,
        wspace=0.25,
    )

    shared_image = None
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    for index, (feature, position) in enumerate(zip(FEATURES, positions)):
        ax = fig.add_subplot(grid[position])
        shared_image = add_manuscript_map(
            ax,
            arrays[feature],
            common_norm,
            show_x=position[0] == 2,
            show_y=position[1] == 0,
        )
        fraction = summary.loc[
            summary["Feature"] == feature, "PositiveFractionPercent"
        ].iloc[0]
        ax.set_title(f"{LABELS[feature]}  ({fraction:.1f}% > 0)")
        F.panel_label(ax, chr(97 + index), x=-0.09, y=1.02)

    ax = fig.add_subplot(grid[2, 1])
    frame = summary.iloc[::-1]
    y = np.arange(len(frame))
    colors = [
        F.COLORS["t2m"] if feature == "AlbedoLoss" else F.COLORS["neutral"]
        for feature in frame["Feature"]
    ]
    ax.barh(y, frame["PositiveFractionPercent"], color=colors, height=0.58)
    ax.axvline(50, color="#858B8F", linestyle="--", linewidth=0.7)
    ax.set_yticks(y, frame["DisplayFeature"])
    ax.set_xlim(0, 100)
    ax.set(xlabel="Pixels with gain > 0 (%)", title="Spatial prevalence")
    for yi, value in zip(y, frame["PositiveFractionPercent"]):
        ax.text(value + 1.5, yi, f"{value:.1f}%", ha="left", va="center", fontsize=5.8)
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "f", x=-0.16, y=1.02)

    if shared_image is not None:
        colorbar_axis = fig.add_subplot(grid[3, :])
        colorbar = fig.colorbar(
            shared_image,
            cax=colorbar_axis,
            orientation="horizontal",
            ticks=[-common_limit, 0, common_limit],
        )
        colorbar.set_label("P3 - P1 mean(|SHAP|)")
        colorbar.outline.set_linewidth(0.45)

    fig.suptitle(
        "Spatial distribution of fitted land-surface contribution gains",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=10.2,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.925, bottom=0.075)
    F.save_figure(fig, STEM)
    summary.to_csv(
        F.SOURCE_DATA / "FigS7_1_spatial_gain_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    F.write_qa(
        STEM,
        {
            "features": FEATURES,
            "projection": "rectilinear geographic longitude-latitude",
            "map_aspect": "2:1",
            "common_colour_limit": common_limit,
            "AlbedoLoss_positive_pixels_percent": 80.1,
            "interpretation": "fitted predictor contribution, not causal attribution",
        },
    )


if __name__ == "__main__":
    main()

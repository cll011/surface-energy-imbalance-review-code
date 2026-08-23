# -*- coding: utf-8 -*-
"""Fig. S3: global coupling context and regional pixel-wise violin distributions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

import figure_common as F


STEM = "Fig_S3_spatial_coupling_regional_violin"


def main() -> None:
    corr_path = F.SOURCE_ROOT / "Fig2c_spatial_raw_corr_GLASS_albedo_T2M_lsm_updated.tif"
    class_path = F.SOURCE_ROOT / "Fig2d_directional_classes_GLASS_T2M_lsm_updated.tif"
    sample_path = F.SOURCE_ROOT / "Fig2e_regional_correlation_display_sample_lsm_updated.csv.gz"
    summary_path = F.SOURCE_ROOT / "Fig2e_regional_correlation_summary_lsm_updated.csv"
    corr, _ = F.read_raster(corr_path)
    classes, _ = F.read_raster(class_path)
    sample = pd.read_csv(sample_path)
    summary = pd.read_csv(summary_path)
    display = {key: label for key, label, _ in F.REGIONS}
    sample["DisplayRegion"] = sample["Region"].map(display)
    order = [label for _key, label, _ in F.REGIONS]

    fig = plt.figure(figsize=(7.2, 5.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35])
    ax = fig.add_subplot(gs[0, 0])
    F.add_map(ax, corr, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), label="r(GLASS albedo, T2M)", ticks=[-1, -0.5, 0, 0.5, 1])
    ax.set_title("Pixel-wise temporal correlation")
    F.panel_label(ax, "a", x=-0.04)

    ax = fig.add_subplot(gs[0, 1])
    class_colors = ["#C6534A", "#E2A64A", "#718AA7", "#D6D9DB"]
    cmap = ListedColormap(class_colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    F.add_map(ax, classes, cmap=cmap, norm=norm, label="Directional class", ticks=[1, 2, 3, 4], colorbar=False)
    ax.set_title("Trend-direction classes")
    handles = [
        Patch(color=class_colors[0], label="C1: darkening + warming, r < 0"),
        Patch(color=class_colors[1], label="C2: darkening + warming, r ≥ 0"),
        Patch(color=class_colors[2], label="C3: warming without darkening"),
        Patch(color=class_colors[3], label="C4: other"),
    ]
    legend = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=5.0, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.88)
    F.panel_label(ax, "b", x=-0.04)

    ax = fig.add_subplot(gs[1, :])
    palette = ["#6F9FC2"] * 7 + ["#D69470"]
    sns.violinplot(
        data=sample,
        y="DisplayRegion",
        x="r",
        hue="DisplayRegion",
        order=order,
        hue_order=order,
        orient="h",
        cut=0,
        inner="quartile",
        linewidth=0.65,
        density_norm="width",
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.axvline(0, color="#52585C", linewidth=0.8, linestyle="--")
    ax.set(xlim=(-1, 1), xlabel="Pixel-wise r(GLASS albedo, T2M)", ylabel="", title="Regional coupling distributions")
    lookup = summary.set_index("Region")
    for yi, key in enumerate([key for key, _label, _ in F.REGIONS]):
        ax.text(1.015, yi, f"{lookup.loc[key, 'Negative_fraction_percent']:.0f}% < 0", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.7, color="#4B5053")
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "c", x=-0.055)

    fig.suptitle("Spatially widespread but regionally heterogeneous albedo–temperature coupling", x=0.01, ha="left", fontsize=10.2, fontweight="bold")
    F.save_figure(fig, STEM)
    sample.to_csv(F.SOURCE_DATA / "FigS3_regional_correlation_violin_source.csv.gz", index=False, compression="gzip")
    summary.to_csv(F.SOURCE_DATA / "FigS3_regional_correlation_summary.csv", index=False, encoding="utf-8-sig")
    F.write_qa(STEM, {"violin_input": "pixel-wise correlations sampled for display", "negative_global_percent": 63.7, "interpretation": "heterogeneous spatial association"})


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Fig. S5: distribution, map and regional intervals of the heat-retention transition."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import figure_common as F


STEM = "Fig_S5_heat_retention_spatial"


def main() -> None:
    fig3_root = F.DATA_ROOT / "fig3"
    raster_path = fig3_root / "Fig3_spatial_heat_retention_transition_index_P3_minus_P1.tif"
    summary_path = fig3_root / "Fig3_spatial_heat_retention_transition_summary.csv"
    regional_path = F.SOURCE_DATA / "FigS5_regional_heat_retention_summary.csv"
    array, _ = F.read_raster(raster_path)
    summary = pd.read_csv(summary_path).iloc[0]
    regional = pd.read_csv(regional_path).sort_values("Order")
    values = array[np.isfinite(array)]

    fig = plt.figure(figsize=(7.2, 5.15), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.35])
    ax = fig.add_subplot(gs[0, 0])
    shown = values[(values >= -4) & (values <= 4)]
    bins = np.linspace(-4, 4, 65)
    ax.hist(shown[shown < 0], bins=bins, density=True, color="#74A4C3", alpha=0.80)
    ax.hist(shown[shown >= 0], bins=bins, density=True, color="#C86B64", alpha=0.72)
    ax.axvline(float(summary["Mean"]), color=F.COLORS["t2m"], linewidth=1.2, label=f"mean = {summary['Mean']:.2f}")
    ax.axvline(float(summary["Median"]), color=F.COLORS["albedo"], linewidth=1.1, linestyle="--", label=f"median = {summary['Median']:.2f}")
    ax.set(xlabel="P3 − P1 heat-retention transition index", ylabel="Pixel density", title="Positively skewed global distribution")
    ax.legend()
    F.style_axis(ax)
    F.panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    lim = max(4.0, F.robust_symmetric_limit(array, 97))
    F.add_map(ax, array, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), label="Heat-retention transition index", ticks=[-lim, 0, lim])
    ax.set_title("Where relative heat retention strengthened")
    F.panel_label(ax, "b", x=-0.04)

    ax = fig.add_subplot(gs[1, :])
    frame = regional.sort_values("Order", ascending=False).reset_index(drop=True)
    y = np.arange(len(frame))
    ax.hlines(y, frame["P10"], frame["P90"], color="#B0B8BD", linewidth=1.0)
    ax.hlines(y, frame["Q25"], frame["Q75"], color="#596168", linewidth=4.0)
    ax.scatter(frame["Median"], y, s=24, facecolor="white", edgecolor="#262626", linewidth=0.75, zorder=3)
    ax.axvline(0, color="#666C70", linestyle="--", linewidth=0.75)
    ax.set_yticks(y, frame["DisplayRegion"])
    ax.set(xlabel="P3 − P1 heat-retention transition index", title="Regional median, interquartile and 10th–90th percentile ranges")
    xmin, xmax = ax.get_xlim()
    for yi, fraction in zip(y, frame["PositiveFractionPercent"]):
        ax.text(xmax, yi, f"  {fraction:.0f}% > 0", va="center", ha="left", fontsize=5.7)
    ax.set_xlim(xmin, xmax + 0.18 * (xmax - xmin))
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "c", x=-0.055)

    fig.suptitle("Regional strengthening of relative surface heat retention", x=0.01, ha="left", fontsize=10.2, fontweight="bold")
    F.save_figure(fig, STEM)
    regional.to_csv(F.SOURCE_DATA / "FigS5_regional_heat_retention_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"TransitionIndex": shown}).to_csv(F.SOURCE_DATA / "FigS5_displayed_heat_retention_distribution.csv.gz", index=False, compression="gzip")
    F.write_qa(STEM, {"mean": 0.187, "median": -0.017, "positive_pixels_percent": 48.46, "interpretation": "relative surface-energy partitioning diagnostic, not subsurface heat storage"})


if __name__ == "__main__":
    main()

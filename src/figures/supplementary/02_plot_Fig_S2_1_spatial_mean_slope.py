# -*- coding: utf-8 -*-
"""Fig. S2-1: spatial means and 2001-2024 linear slopes of T2M and GLASS albedo."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import figure_common as F


STEM = "Fig_S2_1_spatial_mean_slope"


def main() -> None:
    paths = F.ensure_s2_spatial_data()
    tmean, _ = F.read_raster(paths["t2m_mean"])
    amean, _ = F.read_raster(paths["albedo_mean"])
    tslope, _ = F.read_raster(paths["t2m_slope"])
    aslope, _ = F.read_raster(paths["albedo_slope"])
    zonal = pd.read_csv(paths["zonal"])

    fig = plt.figure(figsize=(7.2, 5.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.56])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    lo, hi = F.robust_limits(tmean)
    F.add_map(axes[0], tmean, cmap="coolwarm", vmin=lo, vmax=hi, label="Mean T2M (°C)")
    axes[0].set_title("Mean land temperature")
    F.panel_label(axes[0], "a", x=-0.04)

    lo, hi = F.robust_limits(amean)
    F.add_map(axes[1], amean, cmap="YlGnBu_r", vmin=lo, vmax=hi, label="Mean GLASS albedo")
    axes[1].set_title("Mean surface albedo")
    F.panel_label(axes[1], "b", x=-0.04)

    lim = F.robust_symmetric_limit(tslope)
    F.add_map(axes[2], tslope, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), label=r"T2M slope (°C yr$^{-1}$)", ticks=[-lim, 0, lim])
    axes[2].set_title("Temperature slope, 2001–2024")
    F.panel_label(axes[2], "c", x=-0.04)

    lim = F.robust_symmetric_limit(aslope)
    F.add_map(axes[3], aslope, cmap="BrBG", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), label=r"Albedo slope (yr$^{-1}$)", ticks=[-lim, 0, lim])
    axes[3].set_title("Surface-albedo slope, 2001–2024")
    F.panel_label(axes[3], "d", x=-0.04)

    ax = fig.add_subplot(gs[0, 2])
    ax2 = ax.twiny()
    ax.plot(zonal["T2M_mean_C"], zonal["Latitude"], color=F.COLORS["t2m"], linewidth=1.1)
    ax2.plot(zonal["GLASS_albedo_mean"], zonal["Latitude"], color=F.COLORS["albedo"], linewidth=1.1)
    ax.set(xlabel="T2M (°C)", ylabel="Latitude", ylim=(-60, 85), title="Zonal means")
    ax2.set_xlabel("GLASS albedo", color=F.COLORS["albedo"], labelpad=2)
    ax.tick_params(axis="x", colors=F.COLORS["t2m"])
    ax2.tick_params(axis="x", colors=F.COLORS["albedo"], length=2)
    F.style_axis(ax)
    F.panel_label(ax, "e", x=-0.18)

    ax = fig.add_subplot(gs[1, 2])
    ax2 = ax.twiny()
    ax.axvline(0, color="#8A8F92", linewidth=0.65, linestyle="--")
    ax2.axvline(0, color="#8A8F92", linewidth=0.65, linestyle="--")
    ax.plot(zonal["T2M_slope_C_per_year"], zonal["Latitude"], color=F.COLORS["t2m"], linewidth=1.1)
    ax2.plot(zonal["GLASS_albedo_slope_per_year"], zonal["Latitude"], color=F.COLORS["albedo"], linewidth=1.1)
    ax.set(xlabel="T2M slope", ylabel="Latitude", ylim=(-60, 85), title="Zonal slopes")
    ax2.set_xlabel("Albedo slope", color=F.COLORS["albedo"], labelpad=2)
    ax.tick_params(axis="x", colors=F.COLORS["t2m"])
    ax2.tick_params(axis="x", colors=F.COLORS["albedo"], length=2)
    F.style_axis(ax)
    F.panel_label(ax, "f", x=-0.18)

    fig.suptitle("Spatial state and linear change of land temperature and GLASS albedo", x=0.01, ha="left", fontsize=10.2, fontweight="bold")
    F.save_figure(fig, STEM)
    summary = []
    for name, arr in [("T2M_mean_C", tmean), ("GLASS_albedo_mean", amean), ("T2M_slope_C_per_year", tslope), ("GLASS_albedo_slope_per_year", aslope)]:
        values = arr[np.isfinite(arr)]
        summary.append({"Variable": name, "N": values.size, "Mean": np.mean(values), "Median": np.median(values), "P02": np.percentile(values, 2), "P98": np.percentile(values, 98)})
    pd.DataFrame(summary).to_csv(F.SOURCE_DATA / "FigS2_1_spatial_summary.csv", index=False, encoding="utf-8-sig")
    F.write_qa(STEM, {"period": "2001-2024", "grid": "0.25 degree", "land_mask": "lsm >= 0.5", "primary_albedo": "GLASS"})


if __name__ == "__main__":
    main()

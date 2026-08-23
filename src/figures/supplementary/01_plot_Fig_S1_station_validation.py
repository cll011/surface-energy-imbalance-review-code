# -*- coding: utf-8 -*-
"""Fig. S1: independent site-level consistency check for GLASS and MODIS albedo."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_common as F


STEM = "Fig_S1_station_validation"


def main() -> None:
    data = pd.read_csv(F.SITE_CSV)
    products = [("GLASS_Albedo", "GLASS", F.COLORS["albedo"]), ("MODIS_Albedo", "MODIS", F.COLORS["lh"])]
    metrics = []
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)

    ax = axes[0]
    all_values = data[["Obs_Albedo", "GLASS_Albedo", "MODIS_Albedo"]].to_numpy(float)
    lo, hi = float(np.nanmin(all_values)) - 0.008, float(np.nanmax(all_values)) + 0.008
    ax.plot([lo, hi], [lo, hi], color="#777777", linestyle="--", linewidth=0.8)
    for column, label, color in products:
        valid = data[["Obs_Albedo", column]].dropna()
        obs, pred = valid.iloc[:, 0].to_numpy(), valid.iloc[:, 1].to_numpy()
        r = float(np.corrcoef(obs, pred)[0, 1])
        rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
        bias = float(np.mean(pred - obs))
        metrics.append({"Product": label, "n": len(valid), "r": r, "RMSE": rmse, "Bias": bias})
        ax.scatter(obs, pred, s=18, color=color, alpha=0.82, edgecolor="white", linewidth=0.35, label=label)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Tower-observed albedo", ylabel="Raster albedo", title="Site-year agreement")
    ax.legend(loc="upper left")
    F.style_axis(ax)
    F.panel_label(ax, "a")

    ax = axes[1]
    residual_rows = []
    rng = np.random.default_rng(20260811)
    residual_sets = []
    for xpos, (column, label, color) in enumerate(products):
        residual = (data[column] - data["Obs_Albedo"]).dropna().to_numpy()
        residual_sets.append(residual)
        residual_rows.extend({"Product": label, "Residual": float(v)} for v in residual)
        ax.scatter(xpos + rng.normal(0, 0.035, residual.size), residual, s=7, color=color, alpha=0.68, edgecolor="none")
    box = ax.boxplot(residual_sets, positions=[0, 1], widths=0.50, patch_artist=True, showfliers=False)
    for patch, (_column, _label, color) in zip(box["boxes"], products):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.24)
    for median in box["medians"]:
        median.set_color(F.COLORS["black"])
        median.set_linewidth(1.2)
    ax.axhline(0, color="#777777", linewidth=0.75, linestyle="--")
    ax.set_xticks([0, 1], ["GLASS", "MODIS"])
    ax.set(ylabel="Raster - observation", title="Residual distribution")
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "b")

    ax = axes[2]
    ax.plot(data["Year"], data["Obs_Albedo"], color=F.COLORS["black"], marker="o", ms=2.7, linewidth=1.0, label="Observed")
    for column, label, color in products:
        ax.plot(data["Year"], data[column], color=color, marker="o", ms=2.3, linewidth=0.9, label=label)
    ax.set(xlabel="Year", ylabel="Annual albedo", title="CA-Mer annual record")
    ax.legend(loc="best")
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "c")

    fig.suptitle("Tower-based evaluation of annual surface-albedo products", x=0.01, ha="left", fontsize=10.2, fontweight="bold")
    F.save_figure(fig, STEM)
    pd.DataFrame(metrics).to_csv(F.SOURCE_DATA / "FigS1_validation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(residual_rows).to_csv(F.SOURCE_DATA / "FigS1_residuals.csv", index=False, encoding="utf-8-sig")
    data.to_csv(F.SOURCE_DATA / "FigS1_site_year_data.csv", index=False, encoding="utf-8-sig")
    F.write_qa(STEM, {"site": "CA-Mer", "n_years": int(data["Year"].nunique()), "interpretation": "single-site consistency check"})


if __name__ == "__main__":
    main()

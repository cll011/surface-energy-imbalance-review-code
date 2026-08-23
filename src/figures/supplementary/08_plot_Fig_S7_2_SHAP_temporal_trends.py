# -*- coding: utf-8 -*-
"""Fig. S7-2: regional and period-wise fitted SHAP contributions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

import figure_common as F


STEM = "Fig_S7_2_SHAP_temporal_trends"
FEATURES = ["AlbedoLoss", "Rn", "SM", "LH", "SH"]
LABELS = {
    "AlbedoLoss": "Albedo loss",
    "Rn": "Net radiation",
    "SM": "Soil moisture",
    "LH": "Latent heat",
    "SH": "Sensible heat",
}
FEATURE_COLORS = {
    "AlbedoLoss": F.COLORS["t2m"],
    "Rn": "#647F9A",
    "SM": F.COLORS["sm"],
    "LH": F.COLORS["lh"],
    "SH": F.COLORS["rn"],
}
PERIOD_ORDER = ["P1_2001_2014", "P2_2015_2019", "P3_2020_2024"]
PERIOD_LABELS = ["P1\n2001-2014", "P2\n2015-2019", "P3\n2020-2024"]


def weighted_mean(array, valid, weights):
    return float(np.average(array[valid].astype(float), weights=weights[valid].astype(float)))


def main() -> None:
    period = pd.read_csv(F.SOURCE_DATA / "FigS7_2_period_contributions.csv")
    gain = (
        pd.read_csv(F.SOURCE_DATA / "FigS7_2_P3_minus_P1_contribution_gain.csv")
        .set_index("Feature")
        .reindex(FEATURES)
        .reset_index()
    )
    data = period.loc[
        period["Period"].isin(PERIOD_ORDER) & period["Feature"].isin(FEATURES)
    ].copy()
    data["Period"] = pd.Categorical(data["Period"], PERIOD_ORDER, ordered=True)
    data = data.sort_values(["Feature", "Period"])
    mean_pivot = (
        data.pivot(index="Feature", columns="Period", values="MeanAbs_SHAP")
        .reindex(FEATURES)
        .reindex(columns=PERIOD_ORDER)
    )
    regional = pd.read_csv(F.SOURCE_DATA / "FigS7_2_regional_multivariable_contributions.csv")
    regional_albedo = regional.loc[regional["Feature"] == "AlbedoLoss"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    period_x = np.arange(3)

    ax = axes[0, 0]
    region_order = [item[0] for item in F.REGIONS]
    display_order = [item[1] for item in F.REGIONS]
    regional_albedo = regional_albedo.set_index("Region").reindex(region_order).reset_index()
    y = np.arange(len(regional_albedo))[::-1]
    p1_values = regional_albedo["P1_MeanAbs_SHAP"].to_numpy(float)
    p3_values = regional_albedo["P3_MeanAbs_SHAP"].to_numpy(float)
    for yi, p1_value, p3_value in zip(y, p1_values, p3_values):
        ax.plot([p1_value, p3_value], [yi, yi], color="#AEB7BD", linewidth=1.0, zorder=1)
    ax.scatter(
        p1_values,
        y,
        s=22,
        facecolor="white",
        edgecolor=F.PERIOD_COLORS["P1"],
        linewidth=1.0,
        label="P1",
        zorder=3,
    )
    ax.scatter(
        p3_values,
        y,
        s=25,
        marker="D",
        facecolor=F.PERIOD_COLORS["P3"],
        edgecolor="white",
        linewidth=0.5,
        label="P3",
        zorder=4,
    )
    ax.set_yticks(y, display_order)
    ax.set(xlabel="Regional mean |SHAP| for albedo loss", title="Regional P1-P3 comparison")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="x", color=F.COLORS["grid"], linewidth=0.55, zorder=0)
    F.style_axis(ax, grid=False)
    F.panel_label(ax, "a", x=-0.18)

    ax = axes[0, 1]
    regional_gain = regional.pivot(
        index="Region", columns="Feature", values="P3_minus_P1_MeanAbs_SHAP"
    ).reindex(index=region_order, columns=FEATURES)
    gain_matrix = regional_gain.to_numpy(float)
    regional_limit = max(float(np.nanpercentile(np.abs(gain_matrix), 98)), 1e-6)
    image = ax.imshow(
        gain_matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-regional_limit, vcenter=0, vmax=regional_limit),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(FEATURES)), ["Albedo\nloss", "Rn", "SM", "LH", "SH"])
    ax.set_yticks(np.arange(len(display_order)), display_order)
    for row in range(gain_matrix.shape[0]):
        for column in range(gain_matrix.shape[1]):
            value = gain_matrix[row, column]
            text_color = "white" if abs(value) > 0.62 * regional_limit else F.COLORS["black"]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=5.3, color=text_color)
    colorbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("P3 - P1 mean(|SHAP|)")
    colorbar.outline.set_linewidth(0.45)
    ax.set_title("Regional multivariable contribution gain")
    F.panel_label(ax, "b", x=-0.20)

    ax = axes[1, 0]
    matrix = mean_pivot.to_numpy(float)
    image = ax.imshow(
        matrix,
        cmap="YlOrRd",
        norm=Normalize(vmin=0, vmax=float(np.nanmax(matrix))),
        aspect="auto",
    )
    ax.set_xticks(period_x, PERIOD_LABELS)
    ax.set_yticks(np.arange(len(FEATURES)), [LABELS[feature] for feature in FEATURES])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if value > 0.62 * np.nanmax(matrix) else F.COLORS["black"]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=6.0, color=text_color)
    ax.add_patch(
        plt.Rectangle(
            (-0.49, -0.49),
            2.98,
            0.98,
            fill=False,
            edgecolor=F.COLORS["t2m"],
            linewidth=1.4,
        )
    )
    colorbar = plt.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
    colorbar.set_label("Mean(|SHAP|)")
    colorbar.outline.set_linewidth(0.45)
    ax.set_title("Global period contribution matrix")
    F.panel_label(ax, "c", x=-0.22)

    ax = axes[1, 1]
    frame = gain.iloc[::-1].copy()
    y = np.arange(len(frame))
    values = frame["P3_minus_P1_MeanAbs_SHAP"].to_numpy(float)
    colors = [FEATURE_COLORS[feature] for feature in frame["Feature"]]
    ax.hlines(y, 0, values, color=colors, linewidth=1.4)
    ax.scatter(values, y, s=34, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color="#777D80", linewidth=0.7)
    ax.set_yticks(y, [LABELS[feature] for feature in frame["Feature"]])
    ax.set(xlabel="P3 - P1 mean(|SHAP|)", title="Global contribution increase")
    for yi, value in zip(y, values):
        ax.text(value + 0.0012, yi, f"{value:.3f}", ha="left", va="center", fontsize=5.8)
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "d", x=-0.20)

    fig.suptitle(
        "Regional and period-wise changes in fitted land-surface contributions",
        x=0.01,
        ha="left",
        fontsize=10.2,
        fontweight="bold",
    )
    F.save_figure(fig, STEM)
    data.to_csv(
        F.SOURCE_DATA / "FigS7_2_period_contributions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gain.to_csv(
        F.SOURCE_DATA / "FigS7_2_P3_minus_P1_contribution_gain.csv",
        index=False,
        encoding="utf-8-sig",
    )
    regional.to_csv(
        F.SOURCE_DATA / "FigS7_2_regional_multivariable_contributions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    F.write_qa(
        STEM,
        {
            "periods": PERIOD_ORDER,
            "regions": len(F.REGIONS),
            "AlbedoLoss_P1": 0.084,
            "AlbedoLoss_P3": 0.133,
            "AlbedoLoss_gain": 0.049,
            "interpretation": "fitted predictor contribution, not causal attribution",
        },
    )


if __name__ == "__main__":
    main()

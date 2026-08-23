# -*- coding: utf-8 -*-
"""Fig. S6: large-scale background controls and independent-test model diagnostics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import figure_common as F


STEM = "Fig_S6_background_controls"


def _period(year: int) -> str:
    return "P1" if year <= 2014 else ("P2" if year <= 2019 else "P3")


def main() -> None:
    data = pd.read_csv(F.SOURCE_DATA / "FigS6_standardized_background_controls.csv")
    metrics = pd.read_csv(F.SOURCE_DATA / "FigS6_model_metrics.csv")
    variables = {
        "CO2 RF": "CO2_RF_ref2000_annual_control",
        "SST anomaly": "SST_landfill_annual_control_from_spatial",
        "ONI": "ONI_annual_control",
        "ONI lag 1": "ONI_Lag1_annual_control",
        "AOD": "AOD_annual_control_from_spatial",
        "Snow": "Snow_annual_control_from_spatial",
    }
    for label, column in variables.items():
        if label + "_z" not in data.columns:
            data[label + "_z"] = F.zscore(data[column])
    if "T2M_z" not in data.columns:
        data["T2M_z"] = F.zscore(data["T2M_C"])
    if "Period" not in data.columns:
        data["Period"] = data["Year"].map(_period)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    ax = axes[0, 0]
    for column, label, color in [("T2M_z", "T2M", F.COLORS["black"]), ("CO2 RF_z", "CO2 radiative forcing", F.COLORS["t2m"]), ("SST anomaly_z", "SST anomaly", F.COLORS["rn"])]:
        ax.plot(data["Year"], data[column], color=color, linewidth=1.05, label=label)
    ax.axhline(0, color="#858B8F", linewidth=0.6)
    ax.axvline(2015, color="#686E72", linestyle="--", linewidth=0.7)
    ax.axvline(2020, color="#686E72", linestyle=":", linewidth=0.7)
    ax.set(xlabel="Year", ylabel="Standardized anomaly (z)", title="Greenhouse and ocean context")
    ax.legend()
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "a")

    ax = axes[0, 1]
    for column, label, color in [("ONI_z", "ONI", F.COLORS["albedo"]), ("ONI lag 1_z", "ONI lag 1", "#86ABC4"), ("AOD_z", "AOD", F.COLORS["vpd"]), ("Snow_z", "Snow", F.COLORS["lh"])]:
        ax.plot(data["Year"], data[column], color=color, linewidth=0.95, label=label)
    ax.axhline(0, color="#858B8F", linewidth=0.6)
    ax.axvline(2015, color="#686E72", linestyle="--", linewidth=0.7)
    ax.axvline(2020, color="#686E72", linestyle=":", linewidth=0.7)
    ax.set(xlabel="Year", ylabel="Standardized anomaly (z)", title="ENSO, aerosol and snow controls")
    ax.legend(ncol=2)
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "b")

    heat = data.groupby("Period", observed=True)[[label + "_z" for label in variables]].mean().reindex(["P1", "P2", "P3"])
    ax = axes[1, 0]
    im = ax.imshow(heat.T.to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5), aspect="auto")
    ax.set_xticks(np.arange(3), ["P1", "P2", "P3"])
    ax.set_yticks(np.arange(len(variables)), list(variables))
    ax.set_title("Period means of background controls")
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    cb.set_label("Mean z")
    cb.outline.set_linewidth(0.45)
    F.panel_label(ax, "c")

    ax = axes[1, 1]
    labels = ["Background\ncontrols", "Land-surface\ncontribution model"]
    x = np.arange(len(metrics))
    bars = ax.bar(x, metrics["R2_test"], color=[F.COLORS["neutral"], F.COLORS["t2m"]], width=0.58)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.25)
    ax.set(ylabel="Independent-test R²", title="Predictive diagnostics")
    for bar, row in zip(bars, metrics.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"R² = {row.R2_test:.3f}\nRMSE = {row.RMSE_test:.3f}", ha="center", va="bottom", fontsize=6.1)
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "d")

    fig.suptitle("Background-control inputs and predictive model performance", x=0.01, ha="left", fontsize=10.2, fontweight="bold")
    F.save_figure(fig, STEM)
    data.to_csv(F.SOURCE_DATA / "FigS6_standardized_background_controls.csv", index=False, encoding="utf-8-sig")
    heat.reset_index().to_csv(F.SOURCE_DATA / "FigS6_period_background_means.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(F.SOURCE_DATA / "FigS6_model_metrics.csv", index=False, encoding="utf-8-sig")
    F.write_qa(STEM, {"background_model_test_R2": 0.167, "contribution_model_test_R2": 0.201, "interpretation": "predictive diagnostics, not causal removal"})


if __name__ == "__main__":
    main()

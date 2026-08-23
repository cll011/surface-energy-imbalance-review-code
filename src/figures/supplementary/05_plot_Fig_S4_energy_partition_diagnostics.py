# -*- coding: utf-8 -*-
"""Fig. S4: annual and period diagnostics of surface-energy partitioning."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figure_common as F


STEM = "Fig_S4_energy_partition_diagnostics"


def _period(year: int) -> str:
    return "P1" if year <= 2014 else ("P2" if year <= 2019 else "P3")


def main() -> None:
    annual_path = F.SOURCE_DATA / "FigS4_annual_energy_diagnostics.csv"
    annual = pd.read_csv(annual_path)
    annual["Period"] = annual["Year"].map(_period)
    for column in ["T2M", "Rn", "SH", "LH", "VPD", "EvaporativeFraction", "AlbedoLoss_main"]:
        annual[column + "_z"] = F.zscore(annual[column])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    ax = axes[0, 0]
    for column, label, color in [("Rn_z", "Rn", F.COLORS["rn"]), ("SH_z", "SH", F.COLORS["sh"]), ("LH_z", "LH", F.COLORS["lh"])]:
        ax.plot(annual["Year"], annual[column], color=color, linewidth=1.0, marker="o", ms=2.1, label=label)
    ax.axhline(0, color="#858B8F", linewidth=0.65)
    ax.axvline(2015, color="#686E72", linestyle="--", linewidth=0.7)
    ax.axvline(2020, color="#686E72", linestyle=":", linewidth=0.7)
    ax.set(xlabel="Year", ylabel="Annual anomaly (z)", title="Radiative input and turbulent fluxes")
    ax.legend(ncol=3)
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "a")

    ax = axes[0, 1]
    for column, label, color in [("T2M_z", "T2M", F.COLORS["t2m"]), ("VPD_z", "VPD", F.COLORS["vpd"]), ("EvaporativeFraction_z", "EF", F.COLORS["albedo"])]:
        ax.plot(annual["Year"], annual[column], color=color, linewidth=1.0, marker="o", ms=2.1, label=label)
    ax.axhline(0, color="#858B8F", linewidth=0.65)
    ax.axvline(2015, color="#686E72", linestyle="--", linewidth=0.7)
    ax.axvline(2020, color="#686E72", linestyle=":", linewidth=0.7)
    ax.set(xlabel="Year", ylabel="Annual anomaly (z)", title="Thermal and evaporative response")
    ax.legend(ncol=3)
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "b")

    period_means = annual.groupby("Period", observed=True).mean(numeric_only=True).reindex(["P1", "P2", "P3"])
    physical_rows = []
    for period in ["P2", "P3"]:
        for metric in ["Rn", "SH", "LH"]:
            physical_rows.append({"Period": period, "Metric": metric, "Delta_vs_P1": period_means.loc[period, metric] - period_means.loc["P1", metric]})
    physical = pd.DataFrame(physical_rows)
    ax = axes[1, 0]
    x = np.arange(3)
    width = 0.32
    for offset, period in [(-width / 2, "P2"), (width / 2, "P3")]:
        values = physical.loc[physical["Period"] == period].set_index("Metric").reindex(["Rn", "SH", "LH"])["Delta_vs_P1"]
        ax.bar(x + offset, values, width, color=F.PERIOD_COLORS[period], label=period)
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set_xticks(x, ["Rn", "SH", "LH"])
    ax.set(ylabel=r"Change from P1 (W m$^{-2}$)", title="Physical-unit flux changes")
    ax.legend(ncol=2)
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "c")

    metrics = ["T2M", "Rn", "SH", "LH", "VPD", "EvaporativeFraction"]
    labels = ["T2M", "Rn", "SH", "LH", "VPD", "EF"]
    standard_rows = []
    for period in ["P2", "P3"]:
        for metric in metrics:
            delta = period_means.loc[period, metric] - period_means.loc["P1", metric]
            standard_rows.append({"Period": period, "Metric": metric, "Delta_SD": delta / annual[metric].std(ddof=0)})
    standard = pd.DataFrame(standard_rows)
    ax = axes[1, 1]
    y = np.arange(len(metrics))
    for period, marker, color, offset in [("P2", "o", F.PERIOD_COLORS["P2"], -0.10), ("P3", "s", F.PERIOD_COLORS["P3"], 0.10)]:
        values = standard.loc[standard["Period"] == period].set_index("Metric").reindex(metrics)["Delta_SD"]
        ax.scatter(values, y + offset, s=26, marker=marker, color=color, label=period, zorder=3)
        ax.hlines(y + offset, 0, values, color=color, linewidth=0.9, alpha=0.72)
    ax.axvline(0, color="#777777", linewidth=0.7, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlabel="Period mean change from P1 (SD)", title="Comparable period perturbations")
    ax.legend(ncol=2)
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "d")

    fig.suptitle("Diagnostics of the shift from evaporative dissipation towards sensible heating", x=0.01, ha="left", fontsize=10.2, fontweight="bold")
    F.save_figure(fig, STEM)
    annual.to_csv(F.SOURCE_DATA / "FigS4_annual_energy_diagnostics.csv", index=False, encoding="utf-8-sig")
    physical.to_csv(F.SOURCE_DATA / "FigS4_physical_period_changes.csv", index=False, encoding="utf-8-sig")
    standard.to_csv(F.SOURCE_DATA / "FigS4_standardized_period_changes.csv", index=False, encoding="utf-8-sig")
    F.write_qa(STEM, {"periods": "P1=2001-2014; P2=2015-2019; P3=2020-2024", "P3_minus_P1_Rn_W_m2": 0.731, "P3_minus_P1_SH_W_m2": 0.406, "P3_minus_P1_LH_W_m2": 0.223})


if __name__ == "__main__":
    main()

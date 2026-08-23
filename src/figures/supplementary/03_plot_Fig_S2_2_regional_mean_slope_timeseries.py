# -*- coding: utf-8 -*-
"""Fig. S2-2: regional T2M and GLASS annual means with full-record slopes."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figure_common as F


STEM = "Fig_S2_2_regional_mean_slope_timeseries"


def main() -> None:
    source = F.ensure_regional_t2m_albedo_timeseries()
    data = pd.read_csv(source)
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 7.15), sharex=True, sharey=True, constrained_layout=True)
    slope_rows = []
    for index, ((key, label, _filename), ax) in enumerate(zip(F.REGIONS, axes.flat)):
        frame = data.loc[data["Region"] == key].sort_values("Year")
        x = frame["Year"].to_numpy(float)
        tx = frame["T2M_z"].to_numpy(float)
        axx = frame["GLASS_z"].to_numpy(float)
        tcoef = np.polyfit(x, tx, 1)
        acoef = np.polyfit(x, axx, 1)
        t_raw = np.polyfit(x, frame["T2M_C"], 1)[0] * 10
        a_raw = np.polyfit(x, frame["GLASS_Albedo"], 1)[0] * 1000 * 10
        slope_rows.append({"Region": key, "DisplayRegion": label, "T2M_C_per_decade": t_raw, "GLASS_albedo_x1e3_per_decade": a_raw})

        ax.axvspan(2015, 2019.99, color=F.PERIOD_COLORS["P2"], alpha=0.09, linewidth=0)
        ax.axvspan(2020, 2024.6, color=F.PERIOD_COLORS["P3"], alpha=0.09, linewidth=0)
        ax.plot(x, tx, color=F.COLORS["t2m"], marker="o", ms=2.0, linewidth=0.65, alpha=0.58)
        ax.plot(x, axx, color=F.COLORS["albedo"], marker="s", ms=1.8, linewidth=0.65, alpha=0.58)
        ax.plot(x, np.polyval(tcoef, x), color=F.COLORS["t2m"], linewidth=1.45)
        ax.plot(x, np.polyval(acoef, x), color=F.COLORS["albedo"], linewidth=1.45)
        ax.axhline(0, color="#90979B", linewidth=0.55)
        ax.axvline(2015, color="#686E72", linestyle="--", linewidth=0.65)
        ax.axvline(2020, color="#686E72", linestyle=":", linewidth=0.7)
        ax.set_title(label, pad=3)
        ax.text(0.02, 0.04, f"T2M {t_raw:+.2f} °C decade$^{{-1}}$\nAlbedo {a_raw:+.2f} ×10$^{{-3}}$ decade$^{{-1}}$", transform=ax.transAxes, fontsize=5.5, ha="left", va="bottom", color="#3A3E41")
        F.style_axis(ax, grid=True)
        F.panel_label(ax, chr(97 + index), x=-0.07, y=1.02)

    for ax in axes[-1, :]:
        ax.set_xlabel("Year")
    for ax in axes[:, 0]:
        ax.set_ylabel("Regional annual mean (z)")
    handles = [
        Line2D([0], [0], color=F.COLORS["t2m"], lw=1.5, label="T2M"),
        Line2D([0], [0], color=F.COLORS["albedo"], lw=1.5, label="GLASS albedo"),
        Line2D([0], [0], color="#686E72", lw=0.8, ls="--", label="2015 BIC breakpoint"),
        Line2D([0], [0], color="#686E72", lw=0.8, ls=":", label="2020 predefined P3 boundary"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.985), ncol=4, frameon=False)
    fig.suptitle("Regional evolution of land temperature and GLASS albedo", x=0.01, ha="left", fontsize=10.2, fontweight="bold")
    F.save_figure(fig, STEM)
    pd.DataFrame(slope_rows).to_csv(F.SOURCE_DATA / "FigS2_2_regional_full_record_slopes.csv", index=False, encoding="utf-8-sig")
    F.write_qa(STEM, {"regions": len(F.REGIONS), "period": "2001-2024", "slope_definition": "ordinary least-squares linear slope over full record"})


if __name__ == "__main__":
    main()

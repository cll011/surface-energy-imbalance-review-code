# -*- coding: utf-8 -*-
"""Fig. S8: full-record standardized pathway analysis (2001-2024, n=24)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import figure_common as F


STEM = "Fig_S8_full_record_standardized_pathway"


def _p_text(p: float) -> str:
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"


def _node(ax, xy, text, color="#F4F6F7"):
    x, y = xy
    patch = FancyBboxPatch((x - 0.48, y - 0.18), 0.96, 0.36, boxstyle="round,pad=0.03,rounding_size=0.04", facecolor=color, edgecolor="#6E767B", linewidth=0.75)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=6.4)


def _edge(ax, start, end, beta, p, rad=0.0):
    color = F.COLORS["t2m"] if beta >= 0 else F.COLORS["albedo"]
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.7 + 1.2 * abs(beta), color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=18, shrinkB=18, alpha=0.90)
    ax.add_patch(arrow)
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    ax.text(mx, my + 0.10 + 0.25 * rad, f"β = {beta:.3f}\n{_p_text(p)}", ha="center", va="center", fontsize=5.0, color=color, bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.4))


def main() -> None:
    source = F.PATH_ROOT / "path_analysis_standardized_results.csv"
    data = pd.read_csv(source)
    data = data.loc[data["Period"] == "Full Record"].copy()
    data["Path"] = data["Predictor"] + " → " + data["Outcome"]
    data["PathLabel"] = data["Path"].replace({"SurfaceAlbedo → Rn": "Surface albedo → Rn", "SH → AirTemp": "SH → T2M", "LH → AirTemp": "LH → T2M", "AirTemp → VPD": "T2M → VPD"})

    fig = plt.figure(figsize=(7.2, 6.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0], width_ratios=[1.55, 1.0])
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(-0.6, 7.0)
    ax.set_ylim(-0.55, 2.05)
    ax.axis("off")
    coords = {"SurfaceAlbedo": (0.1, 1.15), "Rn": (1.65, 1.15), "SH": (3.25, 1.58), "LH": (3.25, 0.58), "AirTemp": (4.85, 1.15), "VPD": (6.35, 1.15), "SM": (1.65, -0.05)}
    labels = {"SurfaceAlbedo": "Surface\nalbedo", "Rn": "Net radiation\n(Rn)", "SH": "Sensible heat\n(SH)", "LH": "Latent heat\n(LH)", "AirTemp": "Land temperature\n(T2M)", "VPD": "Vapour-pressure\ndeficit", "SM": "Soil moisture\n(SM)"}
    for key, xy in coords.items():
        _node(ax, xy, labels[key], "#F6F8F9" if key not in {"SurfaceAlbedo", "AirTemp"} else ("#EDF4F8" if key == "SurfaceAlbedo" else "#FAEFED"))
    for row in data.itertuples():
        rad = 0.0
        if row.Predictor == "VPD" and row.Outcome == "SM":
            rad = -0.42
        if row.Predictor == "Rn" and row.Outcome == "SM":
            rad = 0.18
        _edge(ax, coords[row.Predictor], coords[row.Outcome], row.Beta_std, row.p_value, rad)
    ax.text(0.01, 0.97, "Linked standardized OLS regressions", transform=ax.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold")
    ax.text(0.99, 0.03, "Blue: negative association   Red: positive association", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="#4D5356")
    F.panel_label(ax, "a", x=-0.01, y=1.01)

    ax = fig.add_subplot(gs[1, 0])
    frame = data.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(frame))
    beta = frame["Beta_std"].to_numpy(float)
    low = frame["CI95_low"].to_numpy(float)
    high = frame["CI95_high"].to_numpy(float)
    colors = [F.COLORS["t2m"] if value >= 0 else F.COLORS["albedo"] for value in beta]
    for yi, value, lo, hi, color in zip(y, beta, low, high, colors):
        ax.hlines(yi, lo, hi, color=color, linewidth=1.1)
        ax.scatter(value, yi, s=24, color=color, edgecolor="white", linewidth=0.45, zorder=3)
    ax.axvline(0, color="#6D7377", linestyle="--", linewidth=0.75)
    ax.set_yticks(y, frame["PathLabel"])
    ax.set(xlabel="Standardized coefficient β (95% CI)", title="All prespecified paths")
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "b", x=-0.25)

    ax = fig.add_subplot(gs[1, 1])
    models = data.drop_duplicates("Outcome")[["Outcome", "R2", "Adj_R2"]].copy()
    order = ["Rn", "LH", "SH", "AirTemp", "VPD", "SM"]
    models = models.set_index("Outcome").reindex(order).reset_index()
    labels2 = ["Rn", "LH", "SH", "T2M", "VPD", "SM"]
    y = np.arange(len(models))
    ax.scatter(models["R2"], y - 0.08, s=26, color=F.COLORS["neutral"], label="R²")
    ax.scatter(models["Adj_R2"], y + 0.08, s=26, facecolor="white", edgecolor=F.COLORS["neutral"], linewidth=0.9, label="Adjusted R²")
    ax.hlines(y, models["Adj_R2"], models["R2"], color="#B2B9BD", linewidth=0.8)
    ax.set_yticks(y, labels2)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="Explained variance", title="Equation-level fit")
    ax.legend(loc="lower right")
    F.style_axis(ax, grid=True)
    F.panel_label(ax, "c", x=-0.16)

    fig.suptitle("Full-record standardized pathway relationships, 2001–2024", x=0.01, ha="left", fontsize=10.2, fontweight="bold")
    F.save_figure(fig, STEM)
    data.to_csv(F.SOURCE_DATA / "FigS8_full_record_path_coefficients.csv", index=False, encoding="utf-8-sig")
    models.to_csv(F.SOURCE_DATA / "FigS8_full_record_model_fit.csv", index=False, encoding="utf-8-sig")
    F.write_qa(STEM, {"period": "2001-2024", "n": 24, "model": "linked standardized ordinary-least-squares regressions", "interpretation": "statistical associations, not experimental causal effects"})


if __name__ == "__main__":
    main()


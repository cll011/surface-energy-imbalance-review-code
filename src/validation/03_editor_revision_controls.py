"""Reproduce the editor-requested albedo controls and sensitivity figures.

This single entry point covers four additions made during editorial revision:

1. verification of the published ALLUMs annual-series reproduction;
2. formal-estimand and strict-common-domain albedo trend comparisons;
3. validation of four albedo estimates against flux-site observations; and
4. GLASS-versus-MCD43A3 sensitivity of the fitted AlbedoLoss contribution.

The script uses only compact, anonymized source tables included in the review
archive. The upstream raster construction remains documented in the original
analysis scripts and data inventory.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "source_data" / "editor_revision"
DEFAULT_OUTPUT = ROOT / "results" / "editor_revision"

COLORS = {
    "ALLUMs snow-free": "#23678E",
    "ALLUMs all-land": "#68A0BF",
    "GLASS all-land": "#C94F52",
    "MCD43A3 all-land": "#DF902B",
    "MODIS all-land": "#DF902B",
    "Flux sites": "#333333",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def percent_anomaly(values: pd.Series) -> pd.Series:
    baseline = float(values.iloc[0])
    if not np.isfinite(baseline) or baseline == 0:
        raise ValueError("The 2001 baseline must be finite and non-zero.")
    return 100.0 * (values.astype(float) - baseline) / baseline


def fit_trend(years: pd.Series, values: pd.Series) -> dict[str, float]:
    x = years.to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    fit = stats.linregress(x, y)
    dof = len(x) - 2
    tcrit = stats.t.ppf(0.975, dof)
    return {
        "n": int(len(x)),
        "slope_percentage_points_per_year": float(fit.slope),
        "intercept": float(fit.intercept),
        "ci95_low": float(fit.slope - tcrit * fit.stderr),
        "ci95_high": float(fit.slope + tcrit * fit.stderr),
        "p_value": float(fit.pvalue),
        "r_squared": float(fit.rvalue**2),
        "endpoint_change_percent": float(y[-1] - y[0]),
    }


def fitted_line_and_ci(years: pd.Series, values: pd.Series):
    x = years.to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = y - fitted
    dof = len(x) - 2
    mse = np.sum(residual**2) / dof
    sxx = np.sum((x - x.mean()) ** 2)
    se = np.sqrt(mse * (1.0 / len(x) + (x - x.mean()) ** 2 / sxx))
    half_width = stats.t.ppf(0.975, dof) * se
    return fitted, fitted - half_width, fitted + half_width


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(width=1.0, length=4)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)


def draw_series(ax, years, values, label, color) -> None:
    ax.plot(
        years,
        values,
        color=color,
        linewidth=1.2,
        marker="o",
        markersize=4.4,
        markeredgecolor="white",
        markeredgewidth=0.7,
        alpha=0.58,
    )
    fitted, lower, upper = fitted_line_and_ci(years, values)
    ax.plot(years, fitted, color=color, linewidth=2.5, label=label)
    ax.fill_between(years, lower, upper, color=color, alpha=0.11, linewidth=0)


def trend_comparison(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    formal = pd.read_csv(input_dir / "ALLUMs_GLASS_anomaly_source_data_2001_2020.csv")
    common = pd.read_csv(input_dir / "harmonized_common_grid_series_2001_2020.csv")

    formal_series = {
        "ALLUMs snow-free": formal["ALLUMs_snow_free_percent_anomaly"],
        "ALLUMs all-land": formal["ALLUMs_all_land_percent_anomaly"],
        "GLASS all-land": formal["GLASS_all_land_percent_anomaly"],
    }
    common_series = {
        "ALLUMs snow-free, direct grid": percent_anomaly(
            common["ALLUMs_direct_snow_free_common"]
        ),
        "ALLUMs all-land, direct grid": percent_anomaly(
            common["ALLUMs_direct_all_land_common"]
        ),
        "GLASS, ERA5 blue-sky": percent_anomaly(common["GLASS_ERA5_common"]),
        "MCD43A3, ERA5 blue-sky": percent_anomaly(common["MODIS_ERA5_common"]),
    }
    color_lookup = {
        "ALLUMs snow-free": COLORS["ALLUMs snow-free"],
        "ALLUMs all-land": "#5C927E",
        "GLASS all-land": COLORS["GLASS all-land"],
        "ALLUMs snow-free, direct grid": COLORS["ALLUMs snow-free"],
        "ALLUMs all-land, direct grid": "#5C927E",
        "GLASS, ERA5 blue-sky": COLORS["GLASS all-land"],
        "MCD43A3, ERA5 blue-sky": COLORS["MCD43A3 all-land"],
    }

    rows = []
    for domain, years, series_map in (
        ("formal estimand", formal["Year"], formal_series),
        ("strict common domain", common["Year"], common_series),
    ):
        for name, values in series_map.items():
            rows.append({"domain": domain, "series": name, **fit_trend(years, values)})
    trends = pd.DataFrame(rows)
    trends.to_csv(output_dir / "Supplementary_Table_S11_trend_statistics.csv", index=False)

    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    for name, values in formal_series.items():
        draw_series(axes[0], formal["Year"], values, name, color_lookup[name])
    axes[0].set_title("Published estimands", loc="left", fontweight="bold", fontsize=10)
    for name, values in common_series.items():
        draw_series(axes[1], common["Year"], values, name, color_lookup[name])
    axes[1].set_title("Strict common land domain", loc="left", fontweight="bold", fontsize=10)
    for index, ax in enumerate(axes):
        style_axes(ax)
        ax.axhline(0, color="#777777", linewidth=0.9, linestyle="--")
        ax.set_xlim(2000.6, 2020.4)
        ax.set_xticks([2001, 2005, 2010, 2015, 2020])
        ax.set_xlabel("Year")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        ax.text(-0.13, 1.02, chr(ord("a") + index), transform=ax.transAxes,
                fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Albedo anomaly relative to 2001 (%)")
    fig.tight_layout(w_pad=2.0)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"Supplementary_Fig_S11_albedo_trend_controls.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return trends


def station_validation(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(input_dir / "station_validation_long_four_products_2001_2020.csv")
    products = ["ALLUMs snow-free", "ALLUMs all-land", "GLASS all-land", "MODIS all-land"]
    metrics = []
    for product in products:
        subset = data.loc[data["Product"] == product].copy()
        error = subset["Error"].to_numpy(dtype=float)
        metrics.append(
            {
                "Product": product,
                "N_station_years": len(subset),
                "N_sites": subset["station_id"].nunique(),
                "Bias": float(np.mean(error)),
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "Pearson_raw": float(stats.pearsonr(subset["station_albedo"], subset["ProductAlbedo"])[0]),
                "Pearson_station_anomaly": float(stats.pearsonr(subset["ObservedAnomaly"], subset["ProductAnomaly"])[0]),
            }
        )
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "Supplementary_Table_S12_site_validation_metrics.csv", index=False)

    observed = (
        data.drop_duplicates(["station_id", "year"])
        .groupby("year", as_index=False)["ObservedAnomaly"]
        .mean()
    )
    annual = data.groupby(["Product", "year"], as_index=False)["ProductAnomaly"].mean()

    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    axes[0].plot(observed["year"], observed["ObservedAnomaly"], color=COLORS["Flux sites"],
                 linewidth=2.2, label="Flux sites")
    for product in products:
        subset = annual.loc[annual["Product"] == product]
        axes[0].plot(subset["year"], subset["ProductAnomaly"], linewidth=1.35,
                     color=COLORS.get(product, "#888888"), label=product)
    axes[0].axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axes[0].set_title("Mean annual anomalies across available sites", loc="left",
                      fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Albedo anomaly")
    axes[0].legend(frameon=False, fontsize=7.3, ncol=2, loc="upper right")
    style_axes(axes[0])

    errors = [data.loc[data["Product"] == product, "Error"].to_numpy() for product in products]
    violin = axes[1].violinplot(errors, showmeans=False, showmedians=True, showextrema=True)
    for body, product in zip(violin["bodies"], products):
        body.set_facecolor(COLORS[product])
        body.set_edgecolor("none")
        body.set_alpha(0.68)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        violin[key].set_color("#555555")
        violin[key].set_linewidth(0.9)
    axes[1].axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(range(1, len(products) + 1))
    axes[1].set_xticklabels(products, rotation=20, ha="right")
    axes[1].set_ylabel("Product minus site albedo")
    axes[1].set_title("Error distributions", loc="left", fontweight="bold", fontsize=10)
    upper = max(np.nanmax(values) for values in errors)
    axes[1].text(0.02, 0.98, "RMSE", transform=axes[1].transAxes,
                 ha="left", va="top", fontsize=7.3, color="#444444")
    for index, row in metrics_df.iterrows():
        axes[1].text(index + 1, upper + 0.006, f"{row['RMSE']:.4f}",
                     ha="center", va="bottom", fontsize=7.3)
    axes[1].set_ylim(min(np.nanmin(values) for values in errors) - 0.01, upper + 0.03)
    style_axes(axes[1])
    for index, ax in enumerate(axes):
        ax.text(-0.13, 1.02, chr(ord("a") + index), transform=ax.transAxes,
                fontweight="bold", fontsize=11)
    fig.tight_layout(w_pad=2.1)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"Supplementary_Fig_S12_four_product_site_validation.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return metrics_df


def shap_product_sensitivity(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(input_dir / "AlbedoLoss_SHAP_contribution_by_period.csv")
    period_order = ["P1_2001_2014", "P2_2015_2019", "P3_2020_2024"]
    labels = ["P1", "P2", "P3"]
    rows = []
    for product in ("GLASS", "MCD43A3"):
        subset = data.loc[data["AlbedoProduct"] == product].set_index("Period").loc[period_order]
        p1 = float(subset.iloc[0]["AreaWeightedMeanAbsSHAP"])
        p3 = float(subset.iloc[-1]["AreaWeightedMeanAbsSHAP"])
        rows.append(
            {
                "AlbedoProduct": product,
                "P1_mean_abs_SHAP": p1,
                "P2_mean_abs_SHAP": float(subset.iloc[1]["AreaWeightedMeanAbsSHAP"]),
                "P3_mean_abs_SHAP": p3,
                "P3_minus_P1": p3 - p1,
                "P3_to_P1_ratio": p3 / p1,
                "Percent_increase": 100.0 * (p3 - p1) / p1,
                "Valid_common_pixels": int(subset.iloc[0]["ValidCommonPixels"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "Supplementary_Table_S13_AlbedoLoss_product_sensitivity.csv", index=False)

    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, ax = plt.subplots(figsize=(4.3, 3.3))
    for product, color in (("GLASS", COLORS["GLASS all-land"]),
                           ("MCD43A3", COLORS["MCD43A3 all-land"])):
        subset = data.loc[data["AlbedoProduct"] == product].set_index("Period").loc[period_order]
        values = subset["AreaWeightedMeanAbsSHAP"].to_numpy(dtype=float)
        ax.plot(labels, values, marker="o", markersize=6, linewidth=2.0,
                color=color, label=product)
        row = summary.loc[summary["AlbedoProduct"] == product].iloc[0]
        ax.text(2.06, values[-1], f"+{row['Percent_increase']:.1f}%",
                color=color, va="center", fontsize=8)
    ax.set_xlim(-0.15, 2.55)
    ax.set_ylabel("Mean |SHAP| for AlbedoLoss")
    ax.set_xlabel("Study period")
    ax.set_title("Albedo-product sensitivity of fitted contribution", loc="left",
                 fontweight="bold", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"Supplementary_Fig_S13_AlbedoLoss_product_sensitivity.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return summary


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "ALLUMs_GLASS_anomaly_source_data_2001_2020.csv",
        "harmonized_common_grid_series_2001_2020.csv",
        "station_validation_long_four_products_2001_2020.csv",
        "AlbedoLoss_SHAP_contribution_by_period.csv",
        "albedo_method_comparison.csv",
        "allums_reproduction_rmse.csv",
    ]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing editor-revision source files: {missing}")

    trends = trend_comparison(input_dir, output_dir)
    sites = station_validation(input_dir, output_dir)
    sensitivity = shap_product_sensitivity(input_dir, output_dir)
    shutil.copy2(
        input_dir / "albedo_method_comparison.csv",
        output_dir / "Supplementary_Table_S10_albedo_method_comparison.csv",
    )
    shutil.copy2(
        input_dir / "allums_reproduction_rmse.csv",
        output_dir / "ALLUMs_reproduction_RMSE.csv",
    )

    reproduction = pd.read_csv(input_dir / "allums_reproduction_rmse.csv")
    report = {
        "allums_reproduction": reproduction.to_dict(orient="records"),
        "formal_trends": trends.loc[trends["domain"] == "formal estimand"].to_dict(orient="records"),
        "common_domain_trends": trends.loc[trends["domain"] == "strict common domain"].to_dict(orient="records"),
        "site_validation": sites.to_dict(orient="records"),
        "albedo_product_sensitivity": sensitivity.to_dict(orient="records"),
    }
    (output_dir / "editor_revision_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Editor-revision outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

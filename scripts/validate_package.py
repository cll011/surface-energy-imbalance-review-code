"""Validate archive structure, frozen results, syntax and anonymity."""

from __future__ import annotations

import csv
import math
import py_compile
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yml", ".yaml"}
SKIP_PARTS = {"__pycache__", ".git"}
REQUIRED = [
    "README.md",
    "ANONYMOUS_PEER_REVIEW.md",
    "CODE_AVAILABILITY.md",
    "DOI_RELEASE_PLAN.md",
    "data/DATA_INVENTORY.csv",
    "docs/PIPELINE.md",
    "docs/FIGURE_CODE_MAP.csv",
    "src/analysis/01_fig2_temporal_analysis.py",
    "src/analysis/01b_fig2_spatial_coupling.py",
    "src/analysis/02_fig3_energy_partition_analysis.py",
    "src/analysis/03_xgboost_shap_analysis.py",
    "src/analysis/04_pathway_analysis_full_record.py",
    "src/validation/03_editor_revision_controls.py",
    "src/figures/main/01_plot_fig2.py",
    "src/figures/main/02_plot_fig3.py",
    "src/figures/main/03_plot_fig4.py",
    "data/source_data/fig4/fig4_contribution_gain_final.csv",
    "data/source_data/fig4/fig4_spatial_gain_summary_final.csv",
    "data/source_data/fig4/xgboost_model_metrics_final.csv",
    "data/source_data/pathway/path_analysis_standardized_results.csv",
    "data/source_data/editor_revision/ALLUMs_GLASS_anomaly_source_data_2001_2020.csv",
    "data/source_data/editor_revision/harmonized_common_grid_series_2001_2020.csv",
    "data/source_data/editor_revision/station_validation_long_four_products_2001_2020.csv",
    "data/source_data/editor_revision/AlbedoLoss_SHAP_contribution_by_period.csv",
    "data/source_data/editor_revision/albedo_method_comparison.csv",
    "data/source_data/editor_revision/allums_reproduction_rmse.csv",
]


def read_csv(relative: str):
    with (PROJECT_ROOT / relative).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual: float, expected: float, tolerance: float = 5e-4) -> bool:
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def validate_required(errors):
    for relative in REQUIRED:
        if not (PROJECT_ROOT / relative).exists():
            errors.append(f"Missing required file: {relative}")


def validate_syntax(errors):
    for path in PROJECT_ROOT.joinpath("src").rglob("*.py"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"Syntax error: {path.relative_to(PROJECT_ROOT)}: {exc}")


def validate_frozen_results(errors):
    gains = {row["Feature"]: row for row in read_csv(
        "data/source_data/fig4/fig4_contribution_gain_final.csv"
    )}
    albedo = gains["AlbedoLoss"]
    expected = {
        "P1_MeanAbs_SHAP": 0.0840404,
        "P3_MeanAbs_SHAP": 0.1328320,
        "P3_minus_P1_MeanAbs_SHAP": 0.0487916,
    }
    for field, value in expected.items():
        if not close(float(albedo[field]), value):
            errors.append(f"Unexpected AlbedoLoss {field}: {albedo[field]}")
    if not close(float(gains["SH"]["P3_minus_P1_MeanAbs_SHAP"]), 0.031844):
        errors.append("Unexpected SH contribution gain")

    spatial = {row["Feature"]: row for row in read_csv(
        "data/source_data/fig4/fig4_spatial_gain_summary_final.csv"
    )}["AlbedoLoss"]
    if int(float(spatial["N"])) != 335413:
        errors.append("Unexpected final SHAP pixel count")
    if not close(float(spatial["PositiveFractionPercent"]), 80.1105, 0.01):
        errors.append("Unexpected positive AlbedoLoss spatial-gain fraction")
    if not close(float(spatial["UnweightedMean"]), 0.0633393):
        errors.append("Unexpected unweighted AlbedoLoss spatial gain")

    metrics = {row["Model"]: row for row in read_csv(
        "data/source_data/fig4/xgboost_model_metrics_final.csv"
    )}
    if not close(float(metrics["Model1_background_stripping"]["R2_test"]), 0.166716):
        errors.append("Unexpected background-model test R2")
    if not close(float(metrics["Model2_main_SHAP_core"]["R2_test"]), 0.200534):
        errors.append("Unexpected contribution-model test R2")

    pathway = read_csv(
        "data/source_data/pathway/path_analysis_standardized_results.csv"
    )
    if len(pathway) != 9 or {row["Period"] for row in pathway} != {"Full Record"}:
        errors.append("Pathway table must contain exactly nine Full Record paths")
    keyed = {(row["Predictor"], row["Outcome"]): row for row in pathway}
    pathway_expected = {
        ("SurfaceAlbedo", "Rn"): -0.453140,
        ("Rn", "SH"): 0.586030,
        ("SH", "AirTemp"): 0.933072,
        ("AirTemp", "VPD"): 0.931001,
        ("VPD", "SM"): -0.736349,
    }
    for key, value in pathway_expected.items():
        if key not in keyed or not close(float(keyed[key]["Beta_std"]), value):
            errors.append(f"Unexpected pathway coefficient: {key}")

    reproduction = {row["Series"]: row for row in read_csv(
        "data/source_data/editor_revision/allums_reproduction_rmse.csv"
    )}
    if not close(float(reproduction["ALLUMs all-land"]["RMSE"]), 6.61e-6, 1e-8):
        errors.append("Unexpected ALLUMs all-land reproduction RMSE")
    if not close(float(reproduction["ALLUMs snow-free"]["RMSE"]), 6.97e-6, 1e-8):
        errors.append("Unexpected ALLUMs snow-free reproduction RMSE")

    sensitivity = read_csv(
        "data/source_data/editor_revision/AlbedoLoss_SHAP_contribution_by_period.csv"
    )
    keyed_sensitivity = {(row["AlbedoProduct"], row["Period"]): row for row in sensitivity}
    expected_sensitivity = {
        ("GLASS", "P1_2001_2014"): 0.0842356,
        ("GLASS", "P3_2020_2024"): 0.1312368,
        ("MCD43A3", "P1_2001_2014"): 0.1058740,
        ("MCD43A3", "P3_2020_2024"): 0.1368347,
    }
    for key, value in expected_sensitivity.items():
        if key not in keyed_sensitivity or not close(
            float(keyed_sensitivity[key]["AreaWeightedMeanAbsSHAP"]), value
        ):
            errors.append(f"Unexpected editor-revision sensitivity value: {key}")


def validate_anonymity(errors):
    email_pattern = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        if re.search(r"C:[\\/]Users[\\/]", text, re.I):
            errors.append(f"User-profile path found in {relative.as_posix()}")
        if email_pattern.search(text):
            errors.append(f"E-mail address found in {relative.as_posix()}")


def main() -> None:
    errors = []
    validate_required(errors)
    validate_syntax(errors)
    validate_frozen_results(errors)
    validate_anonymity(errors)
    if errors:
        print("PACKAGE VALIDATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("PACKAGE VALIDATION PASSED")
    print("- required files present")
    print("- Python sources compile")
    print("- frozen Fig. 4 and pathway values match")
    print("- no user-profile paths or e-mail addresses detected")


if __name__ == "__main__":
    main()

"""Regenerate figures that rely only on compact source data in this archive."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "src/figures/main/02_plot_fig3.py",
    "src/figures/main/03_plot_fig4.py",
    "src/figures/supplementary/01_plot_Fig_S1_station_validation.py",
    "src/figures/supplementary/02_plot_Fig_S2_1_spatial_mean_slope.py",
    "src/figures/supplementary/03_plot_Fig_S2_2_regional_mean_slope_timeseries.py",
    "src/figures/supplementary/05_plot_Fig_S4_energy_partition_diagnostics.py",
    "src/figures/supplementary/06_plot_Fig_S5_heat_retention_spatial.py",
    "src/figures/supplementary/07_plot_Fig_S6_background_controls.py",
    "src/figures/supplementary/08_plot_Fig_S7_1_SHAP_spatial_gain.py",
    "src/figures/supplementary/08_plot_Fig_S7_2_SHAP_temporal_trends.py",
    "src/figures/supplementary/09_plot_Fig_S8_full_record_pathway.py",
    "src/validation/03_editor_revision_controls.py",
]


def main() -> None:
    for relative in SCRIPTS:
        print(f"Running {relative}", flush=True)
        subprocess.run([sys.executable, str(PROJECT_ROOT / relative)], check=True)


if __name__ == "__main__":
    main()

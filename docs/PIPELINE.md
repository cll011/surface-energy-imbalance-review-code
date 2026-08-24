# Reproducible analysis pipeline

## Evidence flow

```text
Public satellite/reanalysis products
        |
        v
GLASS blue-sky annual processing + ERA5-Land lsm >= 0.5
        |
        v
Annual harmonized common-grid rasters and background-control table
        |
        +----------------------+-----------------------+
        |                      |                       |
        v                      v                       v
Fig. 2 coupling         Fig. 3 partitioning     Background XGBoost
temporal + spatial      Rn / SH / LH / EF       residual temperature
        |                      |                       |
        |                      |                       v
        |                      |                 Core XGBoost-SHAP
        |                      |                 AlbedoLoss/Rn/SM/LH/SH
        |                      |                       |
        +----------------------+-----------------------+
                               |
                               v
                 Full-record standardized pathway analysis
                               |
                               v
                 Main and supplementary figure generation
```

## Stage 0. Configure external paths

Copy `config/paths.example.json` to `config/paths.local.json` and replace placeholders. The local file is excluded from Git and the public DOI archive.

## Stage 1. Construct annual GLASS blue-sky albedo

Script: `src/preprocessing/01_compute_glass_blue_sky_annual.py`

The script checks monthly completeness, forms calendar-day-weighted annual means, applies the ERA5-Land `lsm > 0.5` mask mapped to the native GLASS grid, and exports annual rasters and area-weighted global-land summaries.

## Stage 2. Harmonize annual predictors

Scripts:

- `src/preprocessing/02_prepare_result3_common_grid.py`
- `src/preprocessing/03_prepare_background_controls.py`
- `src/preprocessing/04_fix_sst_land_mask.py`
- `src/preprocessing/05_validate_result3_rasters.py`

The output is the annual common-grid raster collection and annual scalar control table. ONI remains a scalar annual control. The validation script checks coverage, units, signs, valid ranges and required years.

## Stage 3. Fig. 2 temporal and spatial coupling

Scripts:

- `src/analysis/01_fig2_temporal_analysis.py`
- `src/analysis/01b_fig2_spatial_coupling.py`
- `src/figures/main/01_plot_fig2.py`

The temporal script calculates area-weighted global-land annual series, quadratic display curves and BIC change-point candidates. The spatial script calculates pixel-wise Pearson correlation, linear trends, directional classes and regional summaries. The plotting script composes the temporal and spatial evidence.

The BIC-selected single breakpoint is 2015.

## Stage 4. Fig. 3 surface-energy partitioning

Scripts:

- `src/analysis/02_fig3_energy_partition_analysis.py`
- `src/figures/main/02_plot_fig3.py`

The analysis calculates annual and period means of Rn, SH, LH, EF, VPD and T2M, then derives the diagnostic heat-retention transition index. It is a relative surface-energy-partitioning diagnostic and not a direct measurement of subsurface heat storage.

## Stage 5. Fig. 4 background control and XGBoost-SHAP

Scripts:

- `src/analysis/03_xgboost_shap_analysis.py`
- `src/figures/main/03_plot_fig4.py`

The background model uses the specified large-scale controls. The contribution model uses AlbedoLoss, Rn, SM, LH and SH to predict residual temperature. The final analysis applies the same `lsm >= 0.5` domain to every raster variable.

## Stage 6. Full-record pathway analysis

Script: `src/analysis/04_pathway_analysis_full_record.py`

The formal analysis uses annual cosine-latitude-weighted global-land means for 2001-2024 (`n=24`). Variables are standardized over the full record. Six linked OLS regressions, each with an intercept, provide standardized coefficients, SE, 95% CI, P, R2 and adjusted R2.

## Stage 7. Validation and supplementary figures

Validation scripts are under `src/validation/`. Supplementary plotting scripts and their shared helper are under `src/figures/supplementary/`. Final numbering is mapped in `docs/FIGURE_CODE_MAP.csv`.

The editor-requested additions are consolidated in
`src/validation/03_editor_revision_controls.py`. Using compact anonymized
tables in `data/source_data/editor_revision/`, this script (i) records the
ALLUMs all-land and snow-free reproduction RMSE values, (ii) compares formal
product estimands and a strict common 1-degree land domain, (iii) validates
ALLUMs, GLASS and MCD43A3 estimates against flux-site observations, and (iv)
tests whether the P1-to-P3 fitted AlbedoLoss contribution increase is retained
when GLASS is replaced by MCD43A3. It regenerates Supplementary Figs. S11-S13
and Supplementary Tables S7-S9 used in the revision.

## Stage 8. Package validation

```powershell
python scripts/build_manifest.py
python scripts/validate_package.py
```

The validator checks file presence, Python syntax, final frozen metrics, source-data traceability and anonymous-review text leakage.

# Unified data and figure provenance

This document consolidates the project directories recorded during manuscript development with the earlier data-list file. `data/DATA_INVENTORY.csv` is the machine-readable authority; this page explains how those entries feed the submitted figures.

## Active data roots

| Data layer | Active local analysis root | Use in the manuscript | Included in this archive |
|---|---|---|---|
| GLASS blue-sky albedo | `D:\10_Research\01_Datasets\02_DataProcess\03_SurfaceAlbedo_GLASS\blueSky_annual_updated_updated` | Primary surface-albedo product, AlbedoLoss and validation | Annual summaries and selected figure source data |
| MODIS blue-sky albedo | `D:\10_Research\01_Datasets\02_DataProcess\04_MODIS_BlueSky_Albedo_Global\Annual` | Cross-product and station validation only | Validation source tables |
| ERA5 annual fields | `D:\10_Research\01_Datasets\01_DataRaw\ERA5\Annual_Tif` | T2M, Rn, LH, SH, VPD, Cloud, SWdown, LWdown and supporting variables | Compact summaries; raw archive remains external |
| ERA5-Land mask | `D:\10_Research\01_Datasets\01_DataRaw\lsm.nc` | Common analytical land domain, `lsm >= 0.5` | Derived common-grid mask and checksum equivalence record |
| FLUXNET/ONEFlux | `D:\10_Research\01_Datasets\02_DataProcess\02_Fluxnet_Station_Validation` | Independent albedo/site consistency check | Validation source tables |
| Background controls | `D:\10_Research\01_Datasets\04_Results\Result3_Figures_optimized\R3_Tables` | CO2 RF, ONI, lagged ONI, SST anomaly, AOD and Snow | Annual control table |
| Common-grid rasters | `D:\10_Research\01_Datasets\04_Results\Result3_Figures_optimized\R3_CommonGrid_Rasters` | Pixel-wise Fig. 2-4 analysis | Selected derived rasters only |
| Regional polygons | `D:\10_Research\2025_Albedo_Temp\02_Data_Process\04_ShapeFiles` | Eight regional summaries in main and supplementary figures | `data/regions/` |

The full satellite and reanalysis archives are not redistributed. The reviewer can point `config/paths.local.json` to equivalent copies obtained from the cited providers.

## Fig. 2 processing chain

1. `src/analysis/01_fig2_temporal_analysis.py` reads annual GLASS and T2M rasters, applies one common land mask and cosine-latitude weighting, and produces annual means, quadratic display fits and BIC change-point candidates.
2. `src/analysis/01b_fig2_spatial_coupling.py` calculates pixel-wise Pearson correlation, 2001-2024 trends, C1-C4 directional classes and the eight regional distributions using the same `lsm >= 0.5` domain.
3. `src/figures/main/01_plot_fig2.py` assembles the temporal, change-point, spatial-correlation, directional-class and regional panels.

## Fig. 3 processing chain

1. `src/analysis/02_fig3_energy_partition_analysis.py` constructs annual and period summaries for albedo, T2M, Rn, SH, LH, VPD and EF and calculates the diagnostic heat-retention transition index.
2. `src/figures/main/02_plot_fig3.py` calculates regional distributions from the archived transition raster and produces the final panel layout.
3. The reproducible compact inputs are in `data/source_data/fig3/`.

## Fig. 4 processing chain

1. `src/analysis/03_xgboost_shap_analysis.py` fits the background model, calculates residual temperature, fits the five-variable contribution model and exports TreeSHAP summaries and spatial gains on the common mask.
2. `src/analysis/04_pathway_analysis_full_record.py` fits the six linked standardized OLS regressions to 24 annual area-weighted global-land means. Only full-record results are retained for formal inference.
3. `src/figures/main/03_plot_fig4.py` reads the frozen current outputs in `data/source_data/fig4/` and the full-record pathway files in `data/source_data/pathway/`.
4. The active SHAP result has 335,413 valid pixels, AlbedoLoss mean absolute SHAP of 0.084 in P1 and 0.133 in P3, and a positive spatial gain in 80.1% of valid pixels.

## Supplementary figures

Supplementary plotting scripts are stored individually in `src/figures/supplementary/`. Final numbering, code, source files and reference PNGs are linked in `docs/FIGURE_CODE_MAP.csv`. `scripts/run_archived_figures.py` reruns every supplementary figure that depends only on the compact source data included here.

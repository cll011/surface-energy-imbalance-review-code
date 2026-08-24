# Data inventory and compact source data

`DATA_INVENTORY.csv` is the authoritative mapping between manuscript variables, original providers, local analysis directories and archive contents. Large third-party rasters are not copied into this review archive. Compact source tables and selected derived rasters are grouped by figure in `source_data/`.

## Source-data groups

- `fig2/`: annual global-land series, quadratic-fit summaries, BIC change-point candidates and input audit.
- `fig3/`: annual and period energy-partitioning tables plus the P3-minus-P1 heat-retention diagnostic raster.
- `fig4/`: final common-mask model metrics, contribution tables, regional summaries and AlbedoLoss SHAP-gain raster.
- `pathway/`: annual global-land means, quality-control table and full standardized OLS output.
- `supplementary/`: final source tables and selected rasters for Supplementary Figs. S1-S10.
- `validation/`: site-validation and product-comparison tables.
- `controls/`: annual background-control table, common land mask and area weights.

The `fig4/` directory contains the final `lsm >= 0.5` source data.


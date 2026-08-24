# Anonymous analytical and plotting code archive

This archive supports anonymous peer review of the manuscript **Surface energy imbalance amplifies recent global warming**. It contains the analytical and plotting code used for Figs. 2-4 and Supplementary Figs. S1-S13, compact figure source data, environment specifications, data-path documentation and integrity checks.

## Frozen study design

- Study period: 2001-2024.
- P1 (stable period): 2001-2014.
- P2 (transition period): 2015-2019.
- P3 (warming period): 2020-2024.
- Analytical land domain: ERA5-Land land-sea mask with `lsm >= 0.5`.
- Area weighting: cosine of cell-centre latitude.
- Primary albedo product: GLASS blue-sky shortwave albedo.
- MODIS albedo: cross-product and site validation.
- ONI: annual scalar control.

## Archive structure

```text
Project/
|-- README.md
|-- ANONYMOUS_PEER_REVIEW.md
|-- CODE_AVAILABILITY.md
|-- DOI_RELEASE_PLAN.md
|-- AUTHOR_ACTION_REQUIRED.md
|-- config/paths.example.json
|-- data/
|   |-- DATA_INVENTORY.csv
|   |-- source_data/
|   `-- regions/
|-- docs/
|   |-- PIPELINE.md
|   |-- DATA_AND_FIGURE_PROVENANCE.md
|   |-- FIGURE_CODE_MAP.csv
|   |-- CODE_MANIFEST.csv
|   `-- LEGACY_DATA_LIST_NORMALIZED.md
|-- src/
|   |-- preprocessing/
|   |-- analysis/
|   |-- validation/
|   `-- figures/
|-- scripts/
|-- reference_outputs/
|-- environment.yml
|-- requirements.txt
|-- MANIFEST.csv
`-- SHA256SUMS.txt
```

## Reproduction levels

### 1. Source-data review and figure regeneration

The compact files in `data/source_data/` contain the final source tables and selected analysis rasters underlying the submitted figures. They are included so reviewers can inspect reported values without downloading the full third-party raster archive. Reference PNGs are in `reference_outputs/`.

```powershell
conda env create -f environment.yml
conda activate albedo-review
python scripts/validate_package.py
python scripts/run_archived_figures.py
```

The editor-requested reproduction, product-control, site-validation and
albedo-product-sensitivity analyses are consolidated in one portable entry
point:

```powershell
python src/validation/03_editor_revision_controls.py
```

This command regenerates Supplementary Figs. S11-S13 and their associated
tables from `data/source_data/editor_revision/`. The archived ALLUMs
reproduction RMSE values compare the independently reconstructed annual
all-land and snow-free series with the corresponding official published
series; they are not a comparison between all-land and snow-free albedo.

### 2. Full analysis from external gridded products

Copy `config/paths.example.json` to `config/paths.local.json`, replace each placeholder with the local location of the corresponding dataset, and run the stages listed in `docs/PIPELINE.md`. 

## Code-to-figure traceability

`docs/FIGURE_CODE_MAP.csv` maps every main and supplementary figure to its analysis code, plotting code, compact source data and reference output. The mapping distinguishes analytical calculation from later panel layout. The conceptual Fig. 1 was assembled as an editable illustration and is not a statistical output; no numerical analysis code is claimed for it.

## Final-version safeguards

The archive uses the final common-mask XGBoost-SHAP results: AlbedoLoss mean absolute SHAP increases from 0.084 in P1 to 0.133 in P3, and 80.1% of valid land pixels have a positive P3-minus-P1 gain. 

## Availability and planned DOI release

If the paper is published, a versioned public release containing the code, environment files, compact source data, file manifest and checksums will be deposited in a DOI-minting repository.

# Anonymous peer-review handoff

## What is supplied

1. All retained preprocessing, analytical, validation and plotting scripts used for the submitted evidence chain.
2. Final compact source tables and selected derived rasters for Figs. 2-4 and Supplementary Figs. S1-S13.
3. A figure-to-code map, data inventory, environment specifications, checksums and a package validator.
4. Reference PNGs for visual comparison.

## What is not duplicated

Large third-party satellite, reanalysis and climate-index archives are not redistributed. Their original products remain available from the providers identified in the manuscript and Supplementary Information. `data/DATA_INVENTORY.csv` records every external input and its expected local layout.

## Reviewer checks

```powershell
conda env create -f environment.yml
conda activate albedo-review
python scripts/validate_package.py
python src/validation/03_editor_revision_controls.py
```

The second command reproduces the editor-requested ALLUMs comparison,
harmonized GLASS/MCD43A3 controls, four-product site validation and
albedo-product contribution sensitivity from one consolidated script.

The validation report distinguishes three states:

- `verified`: included source data and package structure were checked locally;
- `inspectable`: complete code is present, but execution requires external raw datasets;
- `manual asset`: an editable illustration or mechanism panel was assembled outside the numerical workflow.

## Anonymity

This archive does not contain author names, affiliations, e-mail addresses or repository account identifiers. Local paths in retained provenance comments identify only generic drive-level project folders. A local configuration file is not included.

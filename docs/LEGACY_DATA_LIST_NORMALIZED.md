# Consolidation of the earlier data list

The earlier file `数据清单整理_Manuscript_地表能量失衡加剧近期全球升温.txt` correctly identified the main raw and processed data roots, but it mixed current inputs, intermediate products and superseded outputs. The consolidated treatment is:

| Earlier entry | Final treatment |
|---|---|
| `blueSky_annual` | Superseded by `blueSky_annual_updated_updated`; the latter is the active GLASS annual root. |
| MODIS annual blue-sky albedo | Retained for cross-product/site validation only. |
| ERA5 `Annual_Tif` | Retained as the active meteorological and flux input. |
| FLUXNET station directory | Retained as independent validation input. |
| Regional shapefiles | Retained and copied to `data/regions/`. |
| NOAA CO2 annual table | Retained only as a scalar background-control source. Uniform pseudo-spatial CO2 rasters remain rejected. |
| `R3_Annual_Rasters_NativeExact` | Retained locally as intermediate provenance; not required in the compact review archive. |
| `R3_CommonGrid_Rasters` | Active pixel-analysis input for Fig. 3 and Fig. 4. |
| `R3_Tables` | Active source for annual scalar controls, including ONI. |
| historical `R3_SHAP` tables | Superseded where they contain the earlier 342,123-pixel result. Final common-mask source data are the 335,413-pixel files in this archive. |
| `Result1_Figures` | Active source for mapped Fig. 3 and validation products; historical MODIS-labelled Fig. 2 outputs are not treated as final GLASS results. |

The authoritative machine-readable version is `data/DATA_INVENTORY.csv`.


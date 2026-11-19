# Longitudinal Social Vulnerability Index for Texas Metro Areas (2009 to 2024)

A reproducible Python pipeline that pulls American Community Survey (ACS) 5-year data from the U.S. Census Bureau API, harmonizes four survey periods to a common census tract geography, and builds a temporally comparable Social Vulnerability Index (SVI) with labeled spatial outputs for ArcGIS Pro.

This code supports my doctoral dissertation at Texas State University:
**Spatiotemporal Social Vulnerability in the Austin-San Antonio Metropolitan Corridor: Persistence, Change, and Hazard Congruence, 2009-2024.**

## Study area

- San Antonio-New Braunfels MSA (8 counties) and Austin-Round Rock-Georgetown MSA (5 counties), Texas
- 807 census tracts on 2010 tract boundaries
- Four ACS 5-year vintages: 2005-2009, 2010-2014, 2015-2019, 2020-2024

## What the pipeline does

| Step | Script | Purpose |
|---|---|---|
| 1 | `src/01_census_acs5_pull.py` | Pulls ACS detailed and profile tables for every tract in the 13 counties, batching requests under the Census API's 50-variable limit, and downloads matching TIGER/Line tract boundaries |
| 2 to 5 | *(to be added)* | Boundary crosswalks (2000 and 2020 tracts to 2010), CPI adjustment, and stacking into a long panel |
| 6 | `src/06_build_svi_index.py` | Computes 12 indicators, pooled z-scores, composite and theme scores, uncertainty flags, a sensitivity test, and labeled GeoPackages |

## Method highlights

- **12 indicators in 4 themes:** socioeconomic status, household composition, minority status and language, housing and transportation.
- **Pooled z-score standardization** across all tract-year observations, so every vintage sits on the same scale and scores can be compared over time (following Tate 2012 and Spielman and Folch 2015).
- **Year-aware variable codes:** some ACS profile variables change position between vintages, so codes are looked up per year.
- **Fail-fast validation:** the build stops with a clear error if any expected ACS column is missing, instead of silently producing a wrong indicator.
- **Missing-data rules:** a composite score requires at least 10 of 12 valid indicators, with theme-level thresholds as well.
- **Uncertainty flag:** tracts where key estimates have a coefficient of variation above 40% are flagged as high uncertainty.
- **Sensitivity analysis:** compares the composite with and without unemployment for the 2014 to 2024 vintages and reports Pearson and Spearman correlations.
- **Within-year percentile ranks** are also provided for CDC-style cross-sectional comparison.

## Outputs

- `output/svi_scores_all_years.csv`: scored panel with indicators, z-scores, composite, themes, flags, and metadata
- `output/svi_sensitivity_unemployment.csv`: sensitivity test results
- `output/svi_field_reference.csv`: plain-language definition of every output field
- `output/spatial/svi_{YEAR}_labeled.gpkg`: one labeled layer per vintage
- `output/spatial/svi_all_years_labeled_time_enabled.gpkg`: one time-enabled layer for animation in ArcGIS Pro

## How to run

```bash
pip install -r requirements.txt

# Get a free key at https://api.census.gov/data/key_signup.html
export CENSUS_API_KEY="your_key_here"

# Pull each vintage
ACS_YEAR=2009 python src/01_census_acs5_pull.py
ACS_YEAR=2014 python src/01_census_acs5_pull.py
ACS_YEAR=2019 python src/01_census_acs5_pull.py
ACS_YEAR=2024 python src/01_census_acs5_pull.py

# After the crosswalk and stacking steps
python src/06_build_svi_index.py
```

On Windows PowerShell, use `$env:CENSUS_API_KEY="your_key_here"` and `$env:ACS_YEAR="2009"`.

## Tools

Python (pandas, NumPy, GeoPandas, requests), U.S. Census Bureau API, TIGER/Line shapefiles, ArcGIS Pro.

## Author

**Mogahid Hussein, Ph.D., GISP**
Ph.D. in Geography (Geographic Information Science), Texas State University

## License

Code is released under the MIT License. ACS and TIGER/Line data are public domain products of the U.S. Census Bureau.

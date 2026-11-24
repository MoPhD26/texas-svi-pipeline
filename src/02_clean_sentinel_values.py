"""
02_clean_sentinel_values.py
─────────────────────────────────────────────────────────────────
Replaces Census Bureau sentinel suppression codes with NaN across
all four vintage CSVs. Must be run BEFORE any mathematical
transformations (CPI adjustment, crosswalk) to prevent sentinel
values from being propagated or disguised by downstream operations.

Census Bureau sentinel codes replaced:
   666666666  — primary suppression (small sample / not applicable)
  -666666666  — same, negative variant
   222222222  — special MOE code paired with suppressed estimates
  -222222222  — same, negative variant

Inputs:  output/acs5_{YEAR}_msa_tracts.csv  (all four vintages, from step 01)
Outputs: output/acs5_{YEAR}_msa_tracts.csv  (overwrites in place)
         output/sentinel_cleaning_log.csv   (full audit trail)
         output/spatial/msa_tracts_acs5_{YEAR}_clean.gpkg

Run order: step 02, after the pull and before CPI adjustment.
"""

import os

import numpy as np
import pandas as pd

INPUT_DIR = "output"
YEARS     = [2009, 2014, 2019, 2024]

# Census Bureau suppression and special codes
SENTINEL_VALUES = [666666666, -666666666, 222222222, -222222222]

META_COLS = [
    "YEAR5", "GEOID10", "msa", "county_fips", "county_name",
    "tract_code", "NAME", "state", "county", "tract",
    "data_retrieved_date"
]


def replace_sentinels(df, year, meta_cols=META_COLS, sentinels=SENTINEL_VALUES, verbose=True):
    """
    Replace sentinel codes with NaN in every non-metadata column.

    Returns the cleaned DataFrame and a list of log rows, one per
    (variable, sentinel) combination found.
    """
    df = df.copy()
    log_rows = []

    for col in df.columns:
        if col in meta_cols:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        for sentinel in sentinels:
            mask  = numeric == sentinel
            count = int(mask.sum())

            if count > 0:
                df[col] = df[col].where(~mask, other=np.nan)

                affected_geoids = df[mask]["GEOID10"].tolist()
                affected_pop    = (
                    pd.to_numeric(df[mask]["B01001_001E"], errors="coerce").tolist()
                    if "B01001_001E" in df.columns else []
                )
                pop_list = [int(p) if not pd.isna(p) else "NA" for p in affected_pop]

                if verbose:
                    print(f"    {col} = {sentinel}: "
                          f"{count} tract(s) → NaN | "
                          f"GEOIDs: {affected_geoids} | "
                          f"Pop: {pop_list}")

                log_rows.append({
                    "vintage_year":    year,
                    "variable":        col,
                    "sentinel_value":  sentinel,
                    "tracts_affected": count,
                    "affected_geoids": str(affected_geoids),
                    "tract_pop":       str(pop_list),
                    "action":          "replaced with NaN",
                })

    return df, log_rows


def regenerate_spatial():
    """Join cleaned CSV attributes back onto the boundary GeoPackages."""
    import geopandas as gpd

    spatial_dir = os.path.join(INPUT_DIR, "spatial")

    # 2009 uses 2000 boundaries, 2014/2019 use 2010, 2024 uses 2020
    boundary_gpkg = {yr: os.path.join(spatial_dir, f"msa_tracts_{yr}.gpkg") for yr in YEARS}

    # GEOID column name varies by boundary vintage
    geoid_cols = {
        2009: "CTIDFP00",   # Census 2000 boundaries
        2014: "GEOID",      # Census 2010 boundaries
        2019: "GEOID",      # Census 2010 boundaries
        2024: "GEOID",      # Census 2020 boundaries
    }

    for year in YEARS:
        csv_path  = os.path.join(INPUT_DIR, f"acs5_{year}_msa_tracts.csv")
        gpkg_path = boundary_gpkg[year]
        out_path  = os.path.join(spatial_dir, f"msa_tracts_acs5_{year}_clean.gpkg")

        if not os.path.exists(csv_path):
            print(f"  ⚠  CSV not found, skipping {year}")
            continue
        if not os.path.exists(gpkg_path):
            print(f"  ⚠  Boundary GPKG not found, skipping {year}")
            continue

        gdf = gpd.read_file(gpkg_path)
        df  = pd.read_csv(csv_path, dtype={"GEOID10": str})

        geoid_col = geoid_cols[year]
        if geoid_col not in gdf.columns:
            candidates = [c for c in gdf.columns
                          if "GEOID" in c.upper() or "CTIDFP" in c.upper()]
            geoid_col = candidates[0] if candidates else None

        if geoid_col is None:
            print(f"  ⚠  Could not detect GEOID column for {year}, skipping")
            continue

        gdf_geo = gdf[[geoid_col, "geometry"]].copy()
        gdf_joined = gdf_geo.merge(df, left_on=geoid_col, right_on="GEOID10", how="left")
        gdf_joined.to_file(out_path, driver="GPKG")
        print(f"  ✓ {year}: {len(gdf_joined)} tracts → {out_path}")


def main():
    print("Sentinel Value Cleaner")
    print("=" * 60)

    all_logs = []
    for year in YEARS:
        path = os.path.join(INPUT_DIR, f"acs5_{year}_msa_tracts.csv")
        if not os.path.exists(path):
            print(f"\n  ⚠  File not found, skipping: {path}")
            continue

        df = pd.read_csv(path, dtype={"GEOID10": str})
        print(f"\n  {year} — {len(df)} tracts × {len(df.columns)} columns")

        df, log_rows = replace_sentinels(df, year)
        df.to_csv(path, index=False)
        all_logs.extend(log_rows)

        total = sum(r["tracts_affected"] for r in log_rows)
        print("    No sentinel values found ✓" if total == 0
              else f"    ✓ {total} total replacements — file saved")

    log_path = os.path.join(INPUT_DIR, "sentinel_cleaning_log.csv")
    pd.DataFrame(all_logs).to_csv(log_path, index=False)

    print(f"\n{'=' * 60}")
    print("Regenerating spatial files from cleaned CSVs...")
    print("=" * 60)
    try:
        regenerate_spatial()
        print("\nUse *_clean.gpkg files for all spatial analysis.")
    except ImportError:
        print("  geopandas not available — spatial regeneration skipped.")

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"  Variable and sentinel combinations replaced: {len(all_logs)}")
    print(f"  Audit log saved → {log_path}")
    print("\nNext step: python src/03_adjust_income_cpi.py")


if __name__ == "__main__":
    main()

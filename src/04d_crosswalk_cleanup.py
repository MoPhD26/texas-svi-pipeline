"""
04d_crosswalk_cleanup.py
─────────────────────────────────────────────────────────────────
Post-crosswalk cleanup for both crosswalked vintages (2009 and 2024).
Combines what were two scripts during the research, remove_ghost_tracts.py
and final_crosswalk_cleanup.py, into one step with the same logic.

1. Remove ghost tracts from both crosswalks.
   Ghost tracts have county FIPS codes outside the 13-county study area.
   They arise when source tracts along county edges overlap 2010 tracts
   in neighboring counties. The areal crosswalk then moves a tiny
   population sliver into those out-of-area 2010 tracts, creating
   spurious low-population rows.
     2009: 4 ghost tracts in Blanco (053), Frio (163), and McMullen (311)
           counties, each with a fractional population under 200
     2024: 3 ghost tracts (FIPS 331 and 385)

2. Restore per capita income for Wilson County tracts in the 2009
   crosswalk directly from the CPI-adjusted 2009 ACS source file.
   Assumption: the affected Wilson County tracts carry the same GEOID in
   the 2009 ACS source and on 2010 boundaries, so the source value applies
   directly to the 2010 tract.

3. Add CPI documentation columns to both crosswalked files.

4. Regenerate the crosswalked spatial files on 2010 boundaries.

Validation: each vintage must end with exactly 807 tracts, no duplicate
GEOIDs, and every tract inside the 13 study counties. The files are only
saved if those checks pass.

Inputs:  output/acs5_2009_msa_tracts_cw2010.csv, output/acs5_2024_msa_tracts_cw2010.csv
         output/acs5_2009_msa_tracts.csv, output/acs5_2024_msa_tracts.csv
         output/spatial/msa_tracts_2019.gpkg  (2010 tract boundaries)
Outputs: the two crosswalked CSVs (overwritten) and
         output/spatial/msa_tracts_acs5_{2009,2024}_cw2010.gpkg

Run order: step 04d, after the Wilson County steps 04a to 04c.
"""

from pathlib import Path

import pandas as pd

INPUT_DIR   = Path("output")
SPATIAL_DIR = INPUT_DIR / "spatial"
EXPECTED_TRACTS = 807
WILSON_COUNTY_PREFIX = "48493"

# 13 study area county FIPS codes
MSA_COUNTY_FIPS = {
    "013", "019", "029", "091", "187", "259", "325", "493",  # San Antonio-New Braunfels MSA
    "021", "055", "209", "453", "491",                       # Austin-Round Rock-Georgetown MSA
}

CPI_COLS = ["B19301_cpi_base_year", "B19301_cpi_vintage_year",
            "B19301_cpi_factor", "B19301_cpi_source"]
CPI = {2009: 214.537, 2014: 236.736, 2019: 255.657, 2024: 313.689}


def county_fips_from_geoid(geoids):
    """County FIPS is characters 3 to 5 of an 11-digit tract GEOID (SSCCCTTTTTT)."""
    return geoids.astype(str).str[2:5]


def split_ghost_tracts(df, valid_fips=MSA_COUNTY_FIPS):
    """Return (tracts inside the study area, ghost tracts outside it)."""
    inside = county_fips_from_geoid(df["GEOID10"]).isin(valid_fips)
    return df[inside].reset_index(drop=True), df[~inside].reset_index(drop=True)


def restore_income(cw, src, county_prefix=WILSON_COUNTY_PREFIX):
    """
    Fill null per capita income (estimate and MOE) for tracts in one county
    using the source file. Returns (updated copy, number of tracts restored).
    """
    cw = cw.copy()
    income = pd.to_numeric(cw["B19301_001E"], errors="coerce")
    nulls = cw[income.isna() & cw["GEOID10"].str.startswith(county_prefix)]

    fix = src[src["GEOID10"].isin(nulls["GEOID10"])][["GEOID10", "B19301_001E", "B19301_001M"]]
    for _, row in fix.iterrows():
        mask = cw["GEOID10"] == row["GEOID10"]
        cw.loc[mask, "B19301_001E"] = row["B19301_001E"]
        cw.loc[mask, "B19301_001M"] = row["B19301_001M"]
    return cw, len(fix)


def add_cpi_columns(df, year, src_df=None):
    """Add CPI documentation columns if they are missing."""
    df = df.copy()
    defaults = {
        "B19301_cpi_base_year":    2019,
        "B19301_cpi_vintage_year": year,
        "B19301_cpi_factor":       round(CPI[2019] / CPI[year], 6),
        "B19301_cpi_source":       "BLS CUUR0000SA0 Annual Average",
    }
    for col in CPI_COLS:
        if col in df.columns:
            continue
        if src_df is not None and col in src_df.columns and src_df[col].notna().any():
            df[col] = src_df[col].dropna().iloc[0]
        else:
            df[col] = defaults[col]
    return df


def validate(df, label, expected=EXPECTED_TRACTS):
    """Return a list of problems. An empty list means the file is good."""
    problems = []
    if len(df) != expected:
        problems.append(f"{label}: {len(df)} tracts, expected {expected}")
    dups = int(df["GEOID10"].duplicated().sum())
    if dups:
        problems.append(f"{label}: {dups} duplicate GEOIDs")
    if not county_fips_from_geoid(df["GEOID10"]).isin(MSA_COUNTY_FIPS).all():
        problems.append(f"{label}: tracts outside the study area remain")
    return problems


def main():
    import geopandas as gpd

    print("=" * 60)
    print("Crosswalk Cleanup (2009 and 2024)")
    print("=" * 60)

    read = lambda name: pd.read_csv(INPUT_DIR / name, dtype={"GEOID10": str})
    cw09, cw24 = read("acs5_2009_msa_tracts_cw2010.csv"), read("acs5_2024_msa_tracts_cw2010.csv")
    src09, src24 = read("acs5_2009_msa_tracts.csv"), read("acs5_2024_msa_tracts.csv")

    # ── 1. Ghost tracts ──────────────────────────────────────────────────────
    print("\n[1] Removing ghost tracts outside the 13 study counties")
    cleaned = {}
    for year, df in [(2009, cw09), (2024, cw24)]:
        kept, ghosts = split_ghost_tracts(df)
        print(f"  {year}: {len(ghosts)} ghost tracts removed")
        for _, r in ghosts.iterrows():
            pop = pd.to_numeric(r["B01001_001E"], errors="coerce")
            print(f"    {r['GEOID10']} | county FIPS {r['GEOID10'][2:5]} | pop={pop:.2f}")
        cleaned[year] = kept

    # ── 2. Wilson County income ──────────────────────────────────────────────
    print("\n[2] Restoring Wilson County per capita income in 2009")
    cleaned[2009], n = restore_income(cleaned[2009], src09)
    print(f"  Restored {n} tracts from the CPI-adjusted 2009 source")
    nulls = pd.to_numeric(cleaned[2009]["B19301_001E"], errors="coerce").isna().sum()
    print(f"  Income nulls remaining: {nulls} (expected 2, zero-population Bexar tracts)")

    # ── 3. CPI documentation ─────────────────────────────────────────────────
    print("\n[3] Adding CPI documentation columns")
    cleaned[2009] = add_cpi_columns(cleaned[2009], 2009, src09)
    cleaned[2024] = add_cpi_columns(cleaned[2024], 2024, src24)

    # ── Validate before saving ───────────────────────────────────────────────
    problems = validate(cleaned[2009], "2009") + validate(cleaned[2024], "2024")
    if problems:
        print("\n  ⚠ NOT SAVED. Validation failed:")
        for p in problems:
            print(f"    {p}")
        raise SystemExit(1)

    print("\n[Saving]")
    for year in (2009, 2024):
        df = cleaned[year].sort_values(["county_fips", "tract_code"]).reset_index(drop=True)
        cleaned[year] = df
        out = INPUT_DIR / f"acs5_{year}_msa_tracts_cw2010.csv"
        df.to_csv(out, index=False)
        pop = pd.to_numeric(df["B01001_001E"], errors="coerce").sum()
        print(f"  ✓ {year}: {len(df)} tracts, population {pop:,.0f} → {out}")

    # ── 4. Spatial files on 2010 boundaries ──────────────────────────────────
    print("\n[4] Regenerating spatial files")
    gdf_2010 = gpd.read_file(SPATIAL_DIR / "msa_tracts_2019.gpkg")
    geoid_col = next((c for c in gdf_2010.columns if c in ("GEOID10", "GEOID")), None)
    if geoid_col is None:
        raise RuntimeError("Cannot find a GEOID column in msa_tracts_2019.gpkg")
    for year in (2009, 2024):
        joined = gdf_2010[[geoid_col, "geometry"]].merge(
            cleaned[year], left_on=geoid_col, right_on="GEOID10", how="left"
        )
        out = SPATIAL_DIR / f"msa_tracts_acs5_{year}_cw2010.gpkg"
        joined.to_file(out, driver="GPKG")
        print(f"  ✓ {year} → {out} ({len(joined)} tracts)")

    print("\nNext step: python src/05_stack_longitudinal.py")


if __name__ == "__main__":
    main()

"""
05_stack_longitudinal.py
─────────────────────────────────────────────────────────────────
Stacks all four ACS5 vintages into a single longitudinal panel
dataset on a common Census 2010 tract geography.

Input files (all on 2010 boundaries):
  acs5_2009_msa_tracts_cw2010.csv   (crosswalked from 2000)
  acs5_2014_msa_tracts.csv          (native 2010 boundaries)
  acs5_2019_msa_tracts.csv          (native 2010 boundaries)
  acs5_2024_msa_tracts_cw2010.csv   (crosswalked from 2020)

Output:
  acs_tract_2010std_all_years_long.csv  (807 tracts × 4 vintages = 3,228 rows)

Schema decisions:
  - DP02 limited English renamed to canonical DP02_ltd_english_PE/PM
    across all vintages (vintage-specific codes documented in a column)
  - B23025 unemployment NaN for 2009 (not available at tract level)
  - B19301 income in constant 2019 dollars (CPI-adjusted in step 03)
  - All sentinel values already cleaned in step 02
  - Column order: identifiers → counts → MOEs → rates → metadata

Run order: step 05, after the crosswalk steps 04 to 04d.
"""

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DIR = Path("output")
YEARS     = [2009, 2014, 2019, 2024]
EXPECTED_TRACTS = 807

FILE_MAP = {
    2009: INPUT_DIR / "acs5_2009_msa_tracts_cw2010.csv",
    2014: INPUT_DIR / "acs5_2014_msa_tracts.csv",
    2019: INPUT_DIR / "acs5_2019_msa_tracts.csv",
    2024: INPUT_DIR / "acs5_2024_msa_tracts_cw2010.csv",
}

# Vintage-specific DP02 codes → canonical column name
DP02_SOURCE_CODES = {
    2009: "DP02_0113",
    2014: "DP02_0113",
    2019: "DP02_0114",
    2024: "DP02_0115",
}
DP02_CANONICAL_PE = "DP02_ltd_english_PE"
DP02_CANONICAL_PM = "DP02_ltd_english_PM"

CPI_FACTORS = {
    2009: round(255.657 / 214.537, 6),
    2014: round(255.657 / 236.736, 6),
    2019: round(255.657 / 255.657, 6),
    2024: round(255.657 / 313.689, 6),
}

BOUNDARY_VINTAGE = {2009: 2000, 2014: 2010, 2019: 2010, 2024: 2020}

# Pipeline-internal columns not needed in the analysis file
DROP_COLS = [
    "NAME", "state", "county", "tract",
    "B19301_cpi_vintage_year",  # redundant with YEAR5
]

ID_COLS = ["YEAR5", "GEOID10", "msa", "county_fips", "county_name", "tract_code"]

COUNT_E = [
    "B17001_001E", "B17001_002E",
    "B15002_001E",
] + [f"B15002_{i:03d}E" for i in list(range(3, 11)) + list(range(20, 28))] + [
    "B01001_001E",
] + [f"B01001_{i:03d}E" for i in list(range(3, 7)) + list(range(20, 26))
                              + list(range(27, 31)) + list(range(44, 50))] + [
    "B11005_001E", "B11005_006E", "B11005_007E",
    "B03002_001E", "B03002_003E",
    "B25024_001E", "B25024_007E", "B25024_008E", "B25024_009E", "B25024_010E",
    "B25014_001E", "B25014_005E", "B25014_006E", "B25014_007E",
    "B25014_011E", "B25014_012E", "B25014_013E",
    "B25044_001E", "B25044_003E", "B25044_010E",
    "B23025_003E", "B23025_005E",
    "B19301_001E",
]
COUNT_M = [c.replace("E", "M") for c in COUNT_E]
RATE_COLS = [DP02_CANONICAL_PE, DP02_CANONICAL_PM]
META_COLS = [
    "boundary_vintage", "crosswalked", "crosswalk_note",
    "DP02_source_code",
    "B19301_cpi_base_year", "B19301_cpi_factor", "B19301_cpi_source",
    "data_retrieved_date",
]


def standardize_vintage(df, year, verbose=True):
    """Bring one vintage onto the common panel schema."""
    df = df.copy()

    # Rename DP02 to canonical name
    dp02_code = DP02_SOURCE_CODES[year]
    pe_old, pm_old = f"{dp02_code}PE", f"{dp02_code}PM"
    if pe_old in df.columns:
        df = df.rename(columns={pe_old: DP02_CANONICAL_PE, pm_old: DP02_CANONICAL_PM})
        if verbose:
            print(f"  DP02: {pe_old} → {DP02_CANONICAL_PE} ✓")
    else:
        if verbose:
            print(f"  ⚠ {pe_old} not found — adding as NaN")
        df[DP02_CANONICAL_PE] = np.nan
        df[DP02_CANONICAL_PM] = np.nan
    df["DP02_source_code"] = dp02_code + "PE"

    # Ensure B23025 columns exist (NaN for 2009)
    for col in ["B23025_003E", "B23025_003M", "B23025_005E", "B23025_005M"]:
        if col not in df.columns:
            df[col] = np.nan
            if year != 2009 and verbose:
                print(f"  ⚠ {col} unexpectedly missing for {year}")

    # Ensure CPI columns exist
    if "B19301_cpi_base_year" not in df.columns:
        df["B19301_cpi_base_year"] = 2019
        df["B19301_cpi_factor"]    = CPI_FACTORS[year]
        df["B19301_cpi_source"]    = "BLS CUUR0000SA0 Annual Average"

    df["boundary_vintage"] = BOUNDARY_VINTAGE[year]
    df["crosswalked"]      = year in (2009, 2024)

    if "crosswalk_note" not in df.columns:
        df["crosswalk_note"] = "No crosswalk — native 2010 boundaries"

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    if "data_retrieved_date" not in df.columns:
        df["data_retrieved_date"] = "2026-04-21"  # date of the original pulls

    return df


def stack(frames):
    """Stack standardized vintages, order the columns, and sort by tract then year."""
    long_df = pd.concat(frames, ignore_index=True, sort=False)
    ordered = ID_COLS + COUNT_E + COUNT_M + RATE_COLS + META_COLS
    final_cols = [c for c in ordered if c in long_df.columns]
    final_cols += [c for c in long_df.columns if c not in final_cols]
    return long_df[final_cols].sort_values(["GEOID10", "YEAR5"]).reset_index(drop=True)


def validate(long_df, expected_tracts=EXPECTED_TRACTS):
    """Return a list of problems with the stacked panel. Empty means it passed."""
    problems = []
    for yr, cnt in long_df.groupby("YEAR5").size().items():
        if cnt != expected_tracts:
            problems.append(f"{yr}: {cnt} tracts, expected {expected_tracts}")
    if len(long_df) != expected_tracts * len(YEARS):
        problems.append(f"{len(long_df)} rows, expected {expected_tracts * len(YEARS)}")
    if long_df.duplicated(subset=["GEOID10", "YEAR5"]).any():
        problems.append("duplicate GEOID10 + YEAR5 rows")
    u09 = pd.to_numeric(long_df.loc[long_df["YEAR5"] == 2009, "B23025_003E"], errors="coerce")
    if u09.notna().any():
        problems.append("2009 has unemployment values, expected none")
    return problems


def main():
    print("=" * 60)
    print("Stack Longitudinal Panel Dataset")
    print("=" * 60)
    print(f"\n  Vintages: {YEARS}")
    print("  Geography: Census 2010 tract boundaries (all vintages)")
    print("  Income: constant 2019 dollars (BLS CPI-U CUUR0000SA0)")

    frames = []
    for year in YEARS:
        print(f"\n{'─' * 60}\n  Loading {year} ...")
        df = pd.read_csv(FILE_MAP[year], dtype={"GEOID10": str, "county_fips": str, "tract_code": str})
        print(f"  Raw shape: {df.shape[0]} tracts × {df.shape[1]} columns")
        df = standardize_vintage(df, year)
        print(f"  Final shape: {df.shape[0]} tracts × {df.shape[1]} columns")
        frames.append(df)

    long_df = stack(frames)

    print(f"\n{'=' * 60}\nValidation\n{'=' * 60}")
    problems = validate(long_df)
    for p in problems:
        print(f"  ⚠ {p}")
    if not problems:
        print(f"  ✓ {len(long_df)} rows, {EXPECTED_TRACTS} tracts in every vintage, no duplicates")

    b23025_2014 = pd.to_numeric(long_df.loc[long_df["YEAR5"] == 2014, "B23025_003E"], errors="coerce")
    print(f"  2014 B23025 has values: {'✓' if b23025_2014.notna().sum() > 800 else '⚠'}")
    dp02 = pd.to_numeric(long_df[DP02_CANONICAL_PE], errors="coerce")
    print(f"  {DP02_CANONICAL_PE}: non-null={dp02.notna().sum()} / {len(long_df)}")

    # Income should sit roughly between $25k and $45k in 2019 dollars
    print("\n  Mean per capita income by vintage (2019 dollars):")
    for yr, mean_inc in long_df.groupby("YEAR5")["B19301_001E"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").mean()).items():
        print(f"    {yr}: ${mean_inc:,.0f} {'✓' if 25000 < mean_inc < 45000 else '⚠'}")

    out_path = INPUT_DIR / "acs_tract_2010std_all_years_long.csv"
    long_df.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved → {out_path}  ({len(long_df)} rows × {len(long_df.columns)} columns)")
    print("\nNext step: python src/06_build_svi_index.py")


if __name__ == "__main__":
    main()

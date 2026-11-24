"""
03_adjust_income_cpi.py
─────────────────────────────────────────────────────────────────
Adjusts B19301_001E (per capita income) and its MOE (B19301_001M)
to constant 2019 dollars using BLS CPI-U annual averages across
all four ACS5 vintage CSVs.

Must be run AFTER 02_clean_sentinel_values.py and BEFORE the crosswalk,
so a suppression code is never multiplied into a plausible looking number.

Why 2019 as base year:
  - Pre-pandemic baseline, which avoids COVID-19 economic distortion
  - Middle of the time series, which keeps adjustments small in both directions
  - Matches the CDC SVI 2020 release reference period

CPI-U source: BLS Series CUUR0000SA0, Annual Average (1982-84 = 100)
  Verify at: https://data.bls.gov/timeseries/CUUR0000SA0
  Verified against the BLS data file on 2026-04-21.
  2009: 214.537
  2014: 236.736
  2019: 255.657  ← base year (factor = 1.000)
  2024: 313.689  (an earlier draft used 314.175, the June monthly value,
                  not the annual average)

Adjustment formula:
  adjusted_value = reported_value × (CPI_2019 / CPI_vintage_year)

Applied to:
  B19301_001E  — per capita income estimate
  B19301_001M  — per capita income margin of error

Inputs:  output/acs5_{YEAR}_msa_tracts.csv  (sentinel-cleaned)
Outputs: output/acs5_{YEAR}_msa_tracts.csv  (overwrites with adjusted values)
         output/spatial/msa_tracts_acs5_{YEAR}_clean.gpkg  (regenerated)
         output/cpi_adjustment_log.csv      (audit trail)

Run order: step 03, after sentinel cleaning and before the crosswalk.
"""

import os

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIR   = "output"
SPATIAL_DIR = os.path.join(INPUT_DIR, "spatial")
YEARS       = [2009, 2014, 2019, 2024]
BASE_YEAR   = 2019

# BLS CPI-U Series CUUR0000SA0, Annual Average
CPI = {
    2009: 214.537,
    2014: 236.736,
    2019: 255.657,
    2024: 313.689,
}
CPI_SOURCE = "BLS CUUR0000SA0 Annual Average"

INCOME_E = "B19301_001E"
INCOME_M = "B19301_001M"

# GEOID column name varies by boundary vintage
GEOID_COL = {
    2009: "CTIDFP00",
    2014: "GEOID",
    2019: "GEOID",
    2024: "GEOID",
}


def cpi_factor(year, base_year=BASE_YEAR, cpi=CPI):
    """Multiplier that converts dollars of `year` into dollars of `base_year`."""
    return cpi[base_year] / cpi[year]


def adjust_income(df, year, base_year=BASE_YEAR):
    """
    Return a copy of df with income estimate and MOE in base-year dollars,
    plus a summary dict for the audit log. Null (suppressed) values stay null.
    """
    df = df.copy()
    factor = cpi_factor(year, base_year)
    summary = {"factor": factor, "n_adjusted": 0, "n_null": 0,
               "mean_before": float("nan"), "mean_after": float("nan")}

    if INCOME_E in df.columns:
        income = pd.to_numeric(df[INCOME_E], errors="coerce")
        non_null = income.notna()
        summary["n_adjusted"] = int(non_null.sum())
        summary["n_null"] = int(income.isna().sum())
        summary["mean_before"] = income[non_null].mean()
        df.loc[non_null, INCOME_E] = income[non_null] * factor
        summary["mean_after"] = pd.to_numeric(df[INCOME_E], errors="coerce")[non_null].mean()

    if INCOME_M in df.columns:
        moe = pd.to_numeric(df[INCOME_M], errors="coerce")
        non_null = moe.notna()
        df.loc[non_null, INCOME_M] = moe[non_null] * factor

    # Documentation columns travel with the data
    df["B19301_cpi_base_year"]    = base_year
    df["B19301_cpi_vintage_year"] = year
    df["B19301_cpi_factor"]       = round(factor, 6)
    df["B19301_cpi_source"]       = CPI_SOURCE
    return df, summary


def regenerate_spatial():
    import geopandas as gpd

    for year in YEARS:
        csv_path  = os.path.join(INPUT_DIR, f"acs5_{year}_msa_tracts.csv")
        gpkg_path = os.path.join(SPATIAL_DIR, f"msa_tracts_{year}.gpkg")
        out_path  = os.path.join(SPATIAL_DIR, f"msa_tracts_acs5_{year}_clean.gpkg")

        if not os.path.exists(csv_path) or not os.path.exists(gpkg_path):
            print(f"  ⚠  Missing files for {year}, skipping")
            continue

        gdf = gpd.read_file(gpkg_path)
        df  = pd.read_csv(csv_path, dtype={"GEOID10": str})
        geoid_col = GEOID_COL[year]

        if geoid_col not in gdf.columns:
            candidates = [c for c in gdf.columns
                          if "GEOID" in c.upper() or "CTIDFP" in c.upper()]
            geoid_col = candidates[0] if candidates else None

        if geoid_col is None:
            print(f"  ⚠  GEOID column not found for {year}, skipping")
            continue

        joined = gdf[[geoid_col, "geometry"]].merge(
            df, left_on=geoid_col, right_on="GEOID10", how="left"
        )
        joined.to_file(out_path, driver="GPKG")
        print(f"  ✓ {year}: {len(joined)} tracts → {out_path}")


def main():
    print("CPI Income Adjustment — Per Capita Income to 2019 Dollars")
    print("=" * 60)
    print(f"  Base year:  {BASE_YEAR}")
    print(f"  Variable:   {INCOME_E} and {INCOME_M}")
    print(f"  CPI source: {CPI_SOURCE}\n")

    log_rows = []
    for year in YEARS:
        csv_path = os.path.join(INPUT_DIR, f"acs5_{year}_msa_tracts.csv")
        if not os.path.exists(csv_path):
            print(f"  ⚠  Not found, skipping: {csv_path}")
            continue

        df = pd.read_csv(csv_path, dtype={"GEOID10": str})
        df, s = adjust_income(df, year)

        print(f"  {year}")
        print(f"    CPI-U {year}: {CPI[year]}   CPI-U {BASE_YEAR}: {CPI[BASE_YEAR]}")
        print(f"    Adjustment factor:   {s['factor']:.6f}")
        print(f"    {INCOME_E}: {s['n_adjusted']} tracts adjusted "
              f"({s['n_null']} null/suppressed skipped)")
        print(f"      Mean before: ${s['mean_before']:,.0f} ({year}$)")
        print(f"      Mean after:  ${s['mean_after']:,.0f} (2019$)")

        df.to_csv(csv_path, index=False)
        print(f"    ✓ Saved → {csv_path}\n")

        log_rows.append({
            "vintage_year":        year,
            "base_year":           BASE_YEAR,
            "cpi_vintage":         CPI[year],
            "cpi_base":            CPI[BASE_YEAR],
            "adjustment_factor":   round(s["factor"], 6),
            "tracts_adjusted":     s["n_adjusted"],
            "tracts_skipped_null": s["n_null"],
            "mean_income_before":  round(s["mean_before"], 2),
            "mean_income_after":   round(s["mean_after"], 2),
            "variables_adjusted":  f"{INCOME_E}, {INCOME_M}",
            "cpi_source":          CPI_SOURCE,
        })

    log_path = os.path.join(INPUT_DIR, "cpi_adjustment_log.csv")
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"  ✓ Audit log saved → {log_path}")

    print(f"\n{'=' * 60}")
    print("Regenerating clean spatial files with CPI-adjusted income...")
    print("=" * 60)
    try:
        regenerate_spatial()
    except ImportError:
        print("  geopandas not available — spatial regeneration skipped.")

    print("\nNext step: python src/04_crosswalk_to_2010.py")


if __name__ == "__main__":
    main()

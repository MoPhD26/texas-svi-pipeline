"""
07_verify_lisa_exports.py
─────────────────────────────────────────────────────────────────
Checks the four Local Moran's I (LISA) exports from ArcGIS Pro before
they are used to build the trajectory typology.

LISA is run in ArcGIS Pro with Cluster and Outlier Analysis (Anselin
Local Moran's I) on SVI_SCORE in output/spatial/svi_{YEAR}_labeled.gpkg:
  - Conceptualization: contiguity, edges and corners (first-order Queen)
  - 999 permutations
  - False discovery rate (FDR) correction
Export each result table to output/LISA_results_{YEAR}.csv.

This script reports row counts and the cluster type (COType) distribution
for each year so a missing or mislabeled export is caught early.

Run order: step 07, after the four LISA runs in ArcGIS Pro.
"""

from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("output")
YEARS = [2009, 2014, 2019, 2024]


def cotype_distribution(df):
    """Count cluster types, treating blanks as NS (not significant)."""
    cotype_col = next((c for c in df.columns if "COTYPE" in c.upper()), None)
    if cotype_col is None:
        return None
    return df[cotype_col].fillna("NS").replace("", "NS").value_counts()


def main():
    print("LISA Export Verification")
    print("=" * 50)

    for year in YEARS:
        fpath = OUTPUT_DIR / f"LISA_results_{year}.csv"
        if not fpath.exists():
            print(f"\n{year}: ⚠ File not found — {fpath}")
            continue

        df = pd.read_csv(fpath)
        print(f"\n{year}:")
        print(f"  Rows:    {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        counts = cotype_distribution(df)
        if counts is None:
            print("  ⚠ COType column not found")
            continue
        print("  COType distribution:")
        for k, v in counts.items():
            print(f"    {k:>4}: {v:>4} tracts ({v / len(df) * 100:.1f}%)")

    print("\nNext step: python src/08_build_trajectory_typology.py")


if __name__ == "__main__":
    main()

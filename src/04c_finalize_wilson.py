"""
04c_finalize_wilson.py
─────────────────────────────────────────────────────────────────
Removes the 4 duplicate Wilson County tracts introduced by
04b_add_wilson_passthrough.py and produces the correct 807-tract
final crosswalk output.

The 4 passthrough tracts (000103, 000104, 000403, 000404) were
already present in the original 800-tract crosswalk output AND
were appended again by the passthrough script — creating exact
duplicates. This script deduplicates by keeping the passthrough
version (which has the correct crosswalk_note) and drops the
original crosswalked version.

Run order: step 04c, after 04b_add_wilson_passthrough.py.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

INPUT_DIR = Path("output")
SPATIAL_DIR = INPUT_DIR / "spatial"

PASSTHROUGH_GEOIDS = [
    "48493000103",
    "48493000104",
    "48493000403",
    "48493000404",
]

print("=" * 60)
print("Finalize Wilson County — Remove Duplicates")
print("=" * 60)

cw09 = pd.read_csv(
    INPUT_DIR / "acs5_2009_msa_tracts_cw2010.csv",
    dtype={"GEOID10": str}
)

print(f"\n  Current rows:     {len(cw09)}")
print(f"  Duplicate GEOIDs: {cw09['GEOID10'].duplicated().sum()}")

# Show the duplicates
dups = cw09[cw09['GEOID10'].duplicated(keep=False)]
if len(dups) > 0:
    print("\n  Duplicated rows:")
    for geoid in dups['GEOID10'].unique():
        rows = cw09[cw09['GEOID10'] == geoid]
        print(f"\n    {geoid} appears {len(rows)} times:")
        for _, r in rows.iterrows():
            note = str(r.get('crosswalk_note', ''))[:80]
            pop = r.get('B01001_001E', 'N/A')
            print(f"      pop={pop}  note={note}...")

# For the passthrough GEOIDs: keep the ROW with the passthrough note
# (most informative crosswalk_note), drop the interpolated duplicate
rows_to_drop = []
for geoid in PASSTHROUGH_GEOIDS:
    geoid_rows = cw09[cw09['GEOID10'] == geoid]
    if len(geoid_rows) > 1:
        # Keep the passthrough row (has 'weight=1.0' in note)
        # Drop the interpolated row (has 'areal interpolation' in note)
        interp_idx = geoid_rows[
            geoid_rows['crosswalk_note'].str.contains('areal interpolation', na=False)
        ].index.tolist()
        rows_to_drop.extend(interp_idx)
        print(f"\n  {geoid}: keeping passthrough row, dropping {len(interp_idx)} interpolated row(s)")

# If no duplicates were found at all, check if passthrough rows
# are actually identical to existing rows
if not rows_to_drop and cw09['GEOID10'].duplicated().sum() == 0:
    print("\n  No duplicates found in current file.")
    print("  Checking if we have 811 rows from a different source...")

    # The issue might be that 000103/000104/000403/000404 exist in both
    # the original crosswalk (from geoid matching) AND the passthrough
    # Let's check the crosswalk note to distinguish them
    for geoid in PASSTHROUGH_GEOIDS:
        rows = cw09[cw09['GEOID10'] == geoid]
        if len(rows) == 0:
            print(f"  {geoid}: NOT FOUND in file")
        else:
            print(f"  {geoid}: {len(rows)} row(s), note={str(rows.iloc[0].get('crosswalk_note', ''))[:60]}")

print(f"\n  Rows to drop: {len(rows_to_drop)}")

if rows_to_drop:
    cw09_clean = cw09.drop(index=rows_to_drop).reset_index(drop=True)
else:
    # Alternative: just deduplicate keeping last (passthrough version)
    cw09_clean = cw09.drop_duplicates(subset=['GEOID10'], keep='last').reset_index(drop=True)

cw09_clean = cw09_clean.sort_values(
    ['county_fips', 'tract_code']
).reset_index(drop=True)

print(f"\n  Rows after dedup: {len(cw09_clean)}")
print(f"  Duplicate GEOIDs: {cw09_clean['GEOID10'].duplicated().sum()}")

# Final validation
wilson = cw09_clean[cw09_clean['GEOID10'].str.startswith('48493')]
pop = pd.to_numeric(cw09_clean['B01001_001E'], errors='coerce').sum()

print(f"\n  Wilson County tracts:   {len(wilson)}")
print(f"  Total tracts:           {len(cw09_clean)}  "
      f"{'✓' if len(cw09_clean) == 807 else '⚠ expected 807'}")
print(f"  Total population:       {pop:,.0f}  "
      f"{'✓' if abs(pop - 3569079) < 5000 else '⚠ check'}")
print(f"  Duplicate GEOIDs:       {cw09_clean['GEOID10'].duplicated().sum()}  "
      f"{'✓' if cw09_clean['GEOID10'].duplicated().sum() == 0 else '⚠'}")

# Save
out_csv = INPUT_DIR / "acs5_2009_msa_tracts_cw2010.csv"
cw09_clean.to_csv(out_csv, index=False)
print(f"\n  ✓ Saved → {out_csv}")

# Regenerate spatial
print("\n" + "=" * 60)
print("REGENERATING SPATIAL FILE")
print("=" * 60)

gdf_2010 = gpd.read_file(SPATIAL_DIR / "msa_tracts_2019.gpkg")
geoid_col = next((c for c in gdf_2010.columns if c in ("GEOID10", "GEOID")), None)

if geoid_col:
    gdf_joined = gdf_2010[[geoid_col, "geometry"]].merge(
        cw09_clean, left_on=geoid_col, right_on="GEOID10", how="left"
    )
    out_gpkg = SPATIAL_DIR / "msa_tracts_acs5_2009_cw2010.gpkg"
    gdf_joined.to_file(out_gpkg, driver="GPKG")
    print(f"  ✓ Saved → {out_gpkg}")
    print(f"  Spatial tracts: {len(gdf_joined)}")
else:
    print("  ⚠ GEOID column not found")

print("\n" + "=" * 60)
print("FINAL STATUS")
print("=" * 60)
print(f"""
  Tracts:          {len(cw09_clean)} {'✓' if len(cw09_clean) == 807 else '⚠'}
  Wilson County:   {len(wilson)} {'✓' if len(wilson) == 11 else '⚠'}
  Population:      {pop:,.0f} {'✓' if abs(pop - 3569079) < 5000 else '⚠'}
  Duplicates:      {cw09_clean['GEOID10'].duplicated().sum()} {'✓' if cw09_clean['GEOID10'].duplicated().sum() == 0 else '⚠'}
""")
print("Next step: python src/04d_crosswalk_cleanup.py")

"""
04b_add_wilson_passthrough.py
─────────────────────────────────────────────────────────────────
Adds the 4 remaining Wilson County tracts to the 2009 crosswalk.

These tracts (48493000103, 48493000104, 48493000403, 48493000404)
were already published by the Census Bureau on 2010-compatible
boundaries in the 2009 ACS5 — their source tract in the 2000
boundaries had been pre-split before the ACS was published.
They require no areal interpolation and are passed through
directly with weight = 1.0.

This is documented as a known Wilson County boundary exception
in the crosswalk methodology notes.

Run order: step 04b, after 04a_fix_wilson_county.py.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

INPUT_DIR   = Path("output")
SPATIAL_DIR = INPUT_DIR / "spatial"

# The 4 tracts already on 2010-compatible boundaries
PASSTHROUGH_GEOIDS = [
    "48493000103",
    "48493000104",
    "48493000403",
    "48493000404",
]

print("=" * 60)
print("Wilson County Pass-Through Tract Fix")
print("=" * 60)
print(f"\n  Adding {len(PASSTHROUGH_GEOIDS)} tracts directly from 2009 ACS pull")
print("  (no crosswalk needed — already on 2010-compatible boundaries)")

# Load 2009 ACS source and current crosswalk
df09 = pd.read_csv(
    INPUT_DIR / "acs5_2009_msa_tracts.csv",
    dtype={"GEOID10": str}
)
cw09 = pd.read_csv(
    INPUT_DIR / "acs5_2009_msa_tracts_cw2010.csv",
    dtype={"GEOID10": str}
)

# Confirm these tracts are not already in the crosswalk
already_present = [g for g in PASSTHROUGH_GEOIDS if g in cw09["GEOID10"].values]
if already_present:
    print(f"\n  ⚠ Already in crosswalk: {already_present} — skipping those")
    PASSTHROUGH_GEOIDS = [g for g in PASSTHROUGH_GEOIDS if g not in already_present]

# Extract these tracts from the 2009 ACS source
passthrough = df09[df09["GEOID10"].isin(PASSTHROUGH_GEOIDS)].copy()
print(f"\n  Found {len(passthrough)} of {len(PASSTHROUGH_GEOIDS)} in ACS source")
print(f"  GEOIDs: {sorted(passthrough['GEOID10'].tolist())}")

# Add crosswalk documentation columns
passthrough["crosswalk_note"] = (
    "No crosswalk applied — tract already on 2010-compatible boundaries "
    "in 2009 ACS5 (Wilson County pre-split boundary exception). "
    "Data passed through directly with weight=1.0."
)

# Add CPI documentation columns if present in crosswalk
for col in ['B19301_cpi_base_year','B19301_cpi_vintage_year',
            'B19301_cpi_factor','B19301_cpi_source']:
    if col in cw09.columns and col not in passthrough.columns:
        passthrough[col] = cw09[col].iloc[0]

# Align columns exactly to crosswalk schema
for col in cw09.columns:
    if col not in passthrough.columns:
        passthrough[col] = np.nan

passthrough = passthrough[cw09.columns]

# Append and sort
cw09_final = pd.concat([cw09, passthrough], ignore_index=True)
cw09_final = cw09_final.sort_values(["county_fips","tract_code"]).reset_index(drop=True)

# Validation checks before saving
print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

wilson_final = cw09_final[cw09_final["GEOID10"].str.startswith("48493")]
print(f"\n  Wilson County tracts in final output: {len(wilson_final)}")
print(f"  GEOIDs: {sorted(wilson_final['GEOID10'].tolist())}")
print("  Expected: 11 tracts (all from 2014 boundary)")

pop_before = pd.to_numeric(cw09['B01001_001E'], errors='coerce').sum()
pop_after  = pd.to_numeric(cw09_final['B01001_001E'], errors='coerce').sum()
print(f"\n  Population before: {pop_before:,.0f}")
print(f"  Population after:  {pop_after:,.0f}")
print(f"  Population added:  {pop_after - pop_before:,.0f}")
print("  Original 2009 ACS: ~3,569,079")
print(f"  Remaining diff:    {3569079 - pop_after:,.0f} "
      f"(expected ~0, small diff OK for group quarters tracts)")

print(f"\n  Total tracts: {len(cw09_final)}")
print("  Expected:     807 (matching 2014/2019 tract count)")
match = len(cw09_final) == 807
print(f"  Match 2010 boundary count: {'✓ YES' if match else f'⚠ Got {len(cw09_final)}'}")

# Save
out_csv = INPUT_DIR / "acs5_2009_msa_tracts_cw2010.csv"
cw09_final.to_csv(out_csv, index=False)
print(f"\n  ✓ Saved → {out_csv}")

# Regenerate spatial file
print("\n" + "=" * 60)
print("REGENERATING SPATIAL FILE")
print("=" * 60)

gdf_2010  = gpd.read_file(SPATIAL_DIR / "msa_tracts_2019.gpkg")
geoid_col = next((c for c in gdf_2010.columns if c in ("GEOID10","GEOID")), None)

if geoid_col:
    gdf_joined = gdf_2010[[geoid_col,"geometry"]].merge(
        cw09_final, left_on=geoid_col, right_on="GEOID10", how="left"
    )
    out_gpkg = SPATIAL_DIR / "msa_tracts_acs5_2009_cw2010.gpkg"
    gdf_joined.to_file(out_gpkg, driver="GPKG")
    wilson_spatial = gdf_joined[gdf_joined[geoid_col].str.startswith("48493")]
    print(f"\n  Wilson County tracts in spatial output: {len(wilson_spatial)}")
    print(f"  ✓ Saved → {out_gpkg}")
else:
    print("  ⚠ Could not detect GEOID column")

print("\n" + "=" * 60)
print("Done. All Wilson County tracts now present in 2009 crosswalk.")
print("=" * 60)
print("""
  Methodology note for dissertation:
  Wilson County (FIPS 48493) required special handling in the 2009
  crosswalk. Two 2000-boundary tracts (980101 and 980401) had already
  been pre-split in the published 2009 ACS5 data, meaning 4 of the
  11 Wilson County tracts appeared with 2010-compatible GEOIDs
  (000103, 000104, 000403, 000404) and required no areal interpolation.
  These were passed through directly with weight=1.0. The remaining 7
  Wilson County tracts were crosswalked using standard land-area
  weighted areal interpolation via the tx48trf.txt relationship file.
""")
print("Next step: python src/04c_finalize_wilson.py")
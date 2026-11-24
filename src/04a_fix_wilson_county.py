"""
04a_fix_wilson_county.py
─────────────────────────────────────────────────────────────────
Investigates and fixes the Wilson County (FIPS 48493) GEOID
mismatch between the tx48trf.txt relationship file and the
2009 ACS pull.

The tx48trf.txt file records Wilson County 2000 tract GEOIDs
with '98xx' tract codes (e.g. 48493980101) while the Census
ACS API returns them with '00xx' codes (e.g. 48493000102).
This mismatch causes all Wilson County tracts to be dropped
during the 2009 crosswalk.

This script:
1. Diagnoses the exact GEOID mapping
2. Applies the correction to the 2009 crosswalked CSV
3. Regenerates the 2009 crosswalked spatial file

Wilson County sequence (run in order, directly after step 04):
  04a_fix_wilson_county.py       remap the 98xx relationship file GEOIDs and crosswalk Wilson
  04b_add_wilson_passthrough.py  add the 4 tracts already on 2010-compatible boundaries
  04c_finalize_wilson.py         remove the duplicates the pass-through step created
Then 04d_crosswalk_cleanup.py removes ghost tracts and validates all 807 tracts.

Diagnostic used to find the problem: tools/check_wilson.py
"""

import io
import requests
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

INPUT_DIR   = Path("output")
SPATIAL_DIR = INPUT_DIR / "spatial"

# ── Step 1: Download relationship file and inspect Wilson County rows ─────────
print("=" * 60)
print("STEP 1 — Inspect Wilson County in tx48trf.txt")
print("=" * 60)

resp = requests.get(
    "https://www2.census.gov/geo/docs/maps-data/data/rel/trf_txt/tx48trf.txt",
    timeout=180
)
TRF_COLS = [
    "STATEFP00","COUNTYFP00","TRACTCE00","GEOID00","POP00","HU00",
    "PART00","AREA00","AREALAND00","STATEFP10","COUNTYFP10","TRACTCE10",
    "GEOID10","POP10","HU10","PART10","AREA10","AREALAND10",
    "AREAPT","AREALANDPT","POPPCT00","HUPCT00","POPPCT10","HUPCT10",
    "AREAPCT00","AREALANDPCT00","AREAPCT10","AREALANDPCT10",
    "AREAPTPCT","AREALANDPTPCT",
]
rel = pd.read_csv(
    io.StringIO(resp.text), sep=",",
    header=None, names=TRF_COLS, dtype=str
)

wilson_rel = rel[rel["GEOID00"].str.startswith("48493")].copy()
print(f"\n  Wilson County rows in relationship file: {len(wilson_rel)}")
print(f"\n  {wilson_rel[['GEOID00','GEOID10','AREALAND00','AREALANDPT']].to_string()}")

# ── Step 2: Load 2009 ACS and find Wilson County GEOIDs ──────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Wilson County GEOIDs in 2009 ACS pull")
print("=" * 60)

df09 = pd.read_csv(
    INPUT_DIR / "acs5_2009_msa_tracts.csv",
    dtype={"GEOID10": str}
)
wilson_acs = df09[df09["GEOID10"].str.startswith("48493")]
print(f"\n  Wilson County tracts in 2009 ACS: {len(wilson_acs)}")
print(f"  GEOIDs: {sorted(wilson_acs['GEOID10'].tolist())}")

# ── Step 3: Build the GEOID mapping ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — GEOID mapping: relationship file → ACS")
print("=" * 60)

# The relationship file has GEOID00 like 48493980101
# The ACS has GEOID10 like 48493000102
# Pattern: strip county prefix (48493), compare last 4 digits of tract code
wilson_rel["tract_last4_rel"] = wilson_rel["GEOID00"].str[-4:]
wilson_acs["tract_last4_acs"] = wilson_acs["GEOID10"].str[-4:]

print("\n  Relationship file tract suffixes:", sorted(wilson_rel["tract_last4_rel"].unique()))
print("  ACS 2009 tract suffixes:         ", sorted(wilson_acs["tract_last4_acs"].unique()))

# Build explicit mapping: rel GEOID00 → ACS GEOID
mapping = []
for _, rel_row in wilson_rel.iterrows():
    rel_suffix = rel_row["GEOID00"][-4:]
    # Find matching ACS tract by last 4 digits
    acs_match = wilson_acs[wilson_acs["tract_last4_acs"] == rel_suffix]
    if len(acs_match) == 1:
        mapping.append({
            "geoid_rel": rel_row["GEOID00"],
            "geoid_acs": acs_match.iloc[0]["GEOID10"],
            "geoid10":   rel_row["GEOID10"],
            "aland00":   rel_row["AREALAND00"],
            "alandpt":   rel_row["AREALANDPT"],
        })
    else:
        print(f"  ⚠ No unique match for {rel_row['GEOID00']} (last4={rel_suffix}): "
              f"found {len(acs_match)} ACS tracts")

print(f"\n  Mapped {len(mapping)} of {len(wilson_rel)} relationship rows")
for m in mapping:
    print(f"    {m['geoid_rel']} → ACS {m['geoid_acs']} → 2010 {m['geoid10']}")

# ── Step 4: Build corrected relationship table for Wilson County ──────────────
print("\n" + "=" * 60)
print("STEP 4 — Build corrected relationship records")
print("=" * 60)

map_df = pd.DataFrame(mapping)
map_df["aland00"] = pd.to_numeric(map_df["aland00"], errors="coerce").fillna(0)
map_df["alandpt"] = pd.to_numeric(map_df["alandpt"], errors="coerce").fillna(0)
map_df["weight"]  = np.where(
    map_df["aland00"] > 0,
    map_df["alandpt"] / map_df["aland00"],
    0.0
)
# Rename to match crosswalk function schema
map_df = map_df.rename(columns={
    "geoid_acs": "geoid20",
    "geoid10":   "geoid10",
    "aland00":   "aland_20",
    "alandpt":   "aland_part",
})

# Weight sum check
wt_sums = map_df.groupby("geoid20")["weight"].sum()
print("\n  Weight sum check (should be ≈ 1.0):")
for geoid, wt in wt_sums.items():
    status = "✓" if abs(wt - 1.0) < 0.02 else "⚠"
    print(f"    {geoid}: {wt:.4f} {status}")

# ── Step 5: Crosswalk Wilson County tracts ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Crosswalk Wilson County tracts")
print("=" * 60)

# Get variables to crosswalk from the existing 2009 crosswalked file
cw09_existing = pd.read_csv(
    INPUT_DIR / "acs5_2009_msa_tracts_cw2010.csv",
    dtype={"GEOID10": str}
)

# Identify count E and M columns
meta_cols = ['YEAR5','GEOID10','msa','county_fips','county_name','tract_code',
             'crosswalk_note','B19301_cpi_base_year','B19301_cpi_vintage_year',
             'B19301_cpi_factor','B19301_cpi_source']
count_e = [c for c in cw09_existing.columns
           if c.endswith('E') and c not in meta_cols and not c.startswith('DP')]
count_m = [c.replace('E','M') for c in count_e]
count_m = [c for c in count_m if c in cw09_existing.columns]
rate_cols = ['B19301_001E', 'B19301_001M']
pct_cols  = ['DP02_0113PE', 'DP02_0113PM']

wilson_source = df09[df09["GEOID10"].str.startswith("48493")].copy()
print(f"\n  Wilson source tracts: {len(wilson_source)}")

# Count variables — area weighted sum
def xwalk_counts(src, rel, e_cols, m_cols):
    present_e = [c for c in e_cols if c in src.columns]
    present_m = [c for c in m_cols if c in src.columns]
    merged = rel.merge(
        src[["GEOID10"] + present_e + present_m].rename(columns={"GEOID10":"geoid20"}),
        on="geoid20", how="inner"
    )
    for e, m in zip(present_e, present_m):
        merged[e] = pd.to_numeric(merged[e], errors="coerce").fillna(0) * merged["weight"]
        merged[m] = pd.to_numeric(merged[m], errors="coerce").fillna(0) * merged["weight"]
    return (
        merged.groupby("geoid10")
        .agg({**{e:"sum" for e in present_e},
              **{m: lambda x: np.sqrt((x**2).sum()) for m in present_m}})
        .reset_index().rename(columns={"geoid10":"GEOID10"})
    )

# Rate variables — population weighted average
def xwalk_rate(src, rel, r_cols):
    needed = [c for c in ["GEOID10","B01001_001E"] + r_cols if c in src.columns]
    merged = rel.merge(
        src[needed].rename(columns={"GEOID10":"geoid20"}),
        on="geoid20", how="inner"
    )
    merged["_pw"] = pd.to_numeric(merged["B01001_001E"], errors="coerce").fillna(0) * merged["weight"]
    rows = []
    for g, grp in merged.groupby("geoid10"):
        tpw = grp["_pw"].sum()
        row = {"GEOID10": g}
        for c in [x for x in r_cols if x in grp.columns]:
            row[c] = (pd.to_numeric(grp[c], errors="coerce").fillna(0) * grp["_pw"]).sum() / tpw if tpw > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

# Pct variables — population weighted average
def xwalk_pct(src, rel, pe_cols, pm_cols):
    needed = [c for c in ["GEOID10","B01001_001E"] + pe_cols + pm_cols if c in src.columns]
    merged = rel.merge(
        src[needed].rename(columns={"GEOID10":"geoid20"}),
        on="geoid20", how="inner"
    )
    merged["_pw"] = pd.to_numeric(merged["B01001_001E"], errors="coerce").fillna(0) * merged["weight"]
    rows = []
    for g, grp in merged.groupby("geoid10"):
        tpw = grp["_pw"].sum()
        row = {"GEOID10": g}
        for pe, pm in zip([c for c in pe_cols if c in grp.columns],
                          [c for c in pm_cols if c in grp.columns]):
            row[pe] = (pd.to_numeric(grp[pe], errors="coerce").fillna(0) * grp["_pw"]).sum() / tpw if tpw > 0 else np.nan
            row[pm] = np.sqrt(((pd.to_numeric(grp[pm], errors="coerce").fillna(0) * grp["_pw"] / tpw) ** 2).sum()) if tpw > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

wilson_counts = xwalk_counts(wilson_source, map_df, count_e, count_m)
wilson_rates  = xwalk_rate(wilson_source, map_df, rate_cols)
wilson_pct    = xwalk_pct(wilson_source, map_df, ['DP02_0113PE'], ['DP02_0113PM'])

# Get MSA metadata for Wilson County target tracts
meta_cols_keep = ['msa','county_fips','county_name']
best_idx = map_df.groupby("geoid10")["aland_part"].idxmax()
wilson_meta = map_df.loc[best_idx, ["geoid10"]].rename(columns={"geoid10":"GEOID10"}).copy()
# Carry forward metadata from source
wilson_meta["msa"]          = "San Antonio-New Braunfels MSA"
wilson_meta["county_fips"]  = "493"
wilson_meta["county_name"]  = "Wilson County"

wilson_new = (
    wilson_meta
    .merge(wilson_counts, on="GEOID10", how="left")
    .merge(wilson_rates,  on="GEOID10", how="left")
    .merge(wilson_pct,    on="GEOID10", how="left")
)
wilson_new["YEAR5"]      = 2009
wilson_new["tract_code"] = wilson_new["GEOID10"].str[-6:]
wilson_new["crosswalk_note"] = cw09_existing["crosswalk_note"].iloc[0]

# Add CPI columns
for col in ['B19301_cpi_base_year','B19301_cpi_vintage_year',
            'B19301_cpi_factor','B19301_cpi_source']:
    if col in cw09_existing.columns:
        wilson_new[col] = cw09_existing[col].iloc[0]

print(f"  Wilson County target tracts produced: {len(wilson_new)}")
print(f"  GEOIDs: {sorted(wilson_new['GEOID10'].tolist())}")

# ── Step 6: Append Wilson County to existing crosswalk ───────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Append Wilson County and save updated crosswalk")
print("=" * 60)

# Reorder columns to match existing crosswalk
for col in cw09_existing.columns:
    if col not in wilson_new.columns:
        wilson_new[col] = np.nan

wilson_new = wilson_new[cw09_existing.columns]

cw09_updated = pd.concat([cw09_existing, wilson_new], ignore_index=True)
cw09_updated = cw09_updated.sort_values(["county_fips","tract_code"]).reset_index(drop=True)

out_csv = INPUT_DIR / "acs5_2009_msa_tracts_cw2010.csv"
cw09_updated.to_csv(out_csv, index=False)
print(f"\n  Tracts before: {len(cw09_existing)}")
print(f"  Tracts after:  {len(cw09_updated)}")
print(f"  ✓ Saved → {out_csv}")

# Population check
pop_before = pd.to_numeric(cw09_existing['B01001_001E'], errors='coerce').sum()
pop_after  = pd.to_numeric(cw09_updated['B01001_001E'], errors='coerce').sum()
print(f"\n  Population before: {pop_before:,.0f}")
print(f"  Population after:  {pop_after:,.0f}")
print(f"  Population added:  {pop_after - pop_before:,.0f}")

# ── Step 7: Regenerate spatial file ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Regenerate 2009 crosswalked spatial file")
print("=" * 60)

gdf_2010  = gpd.read_file(SPATIAL_DIR / "msa_tracts_2019.gpkg")
geoid_col = next((c for c in gdf_2010.columns if c in ("GEOID10","GEOID")), None)

if geoid_col:
    gdf_joined = gdf_2010[[geoid_col,"geometry"]].merge(
        cw09_updated, left_on=geoid_col, right_on="GEOID10", how="left"
    )
    out_gpkg = SPATIAL_DIR / "msa_tracts_acs5_2009_cw2010.gpkg"
    gdf_joined.to_file(out_gpkg, driver="GPKG")
    joined_wilson = gdf_joined[gdf_joined[geoid_col].str.startswith("48493")]
    print(f"  Wilson County tracts in spatial output: {len(joined_wilson)}")
    print(f"  ✓ Spatial file regenerated → {out_gpkg}")
else:
    print("  ⚠ Could not detect GEOID column")

print("\n" + "=" * 60)
print("Done. Wilson County patch applied successfully.")
print("=" * 60)
print("\nNext step: python src/04b_add_wilson_passthrough.py")
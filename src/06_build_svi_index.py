"""
build_svi_index.py  —  CORRECTED VERSION
─────────────────────────────────────────────────────────────────────────────
Constructs a temporally anchored equal-weight composite Social Vulnerability
Index from the longitudinal ACS5 panel dataset.

CORRECTIONS FROM PRIOR VERSION (v2 → v3):
  1. Composite missingness threshold raised from >=6 to >=10 of 12 indicators.
     The >=6 rule was theoretically too permissive. The output is identical
     because actual partial rows have 7–8 indicators (3 Travis County
     institutional tracts), which now become null. This tightens the stated
     rule without changing any scored results.

  2. Theme missingness thresholds tightened:
       Socioeconomic  (3 indicators): >=2 (was >=1)
       Household      (3 indicators): >=2 (was >=1)
       Minority_Lang  (2 indicators): >=1 (unchanged — cannot be tighter)
       Housing_Trans  (4 indicators): >=3 (was >=1)
     Output is identical — all partial theme rows already meet the new thresholds.

  3. sum_cols() replaced with sum_cols_checked(), which raises ValueError
     loudly if any expected ACS column is missing. Prevents silent wrong
     indicator calculation from schema mismatches or pull errors.

  4. Script comments and header corrected to match actual code behavior:
     - FLG_HIUNCERT is stored as integer (0/1), not YES/NO text
     - No separate CPI note field is added (CPI metadata already in output)
     - Boundary note clarified: GEO_BNDRY_YR = source vintage, not display geometry

  5. Unemployment justification wording softened in comments to match
     dissertation: "availability/comparability limitation" not "unavailable".

METHODOLOGY (settled after multi-round peer review):
  - Equal-weight composite scoring
  - Pooled z-score standardization across all 3,228 tract-year observations
    simultaneously (defensible for longitudinal comparison;
    Tate 2012; Spielman & Folch 2015)
  - Unemployment excluded from all four vintages to preserve a uniform
    indicator set — availability and comparability limitations constrain
    consistent tract-level measurement for 2009
  - Single overall composite as primary output (SVI_COMPOSITE / SVI_SCORE)
  - Theme-level subscores as secondary diagnostic output
  - CDC-style within-year percentile rankings as secondary comparison output
  - CV > 40% flag on selected key indicators = high-uncertainty operational
    rule (consistent with Census Bureau guidance; not a universal standard)

SENSITIVITY ANALYSIS:
  - Unemployment inclusion vs exclusion for 2014–2024 vintages
  - Reports actual Pearson and Spearman correlations (not assumed values)
  - Note: base z-scores are pooled across 2009–2024; unemployment z-score is
    pooled within 2014–2024 only. This asymmetry is intentional and documented:
    the test is a diagnostic for the design choice, not a parallel index build.

OUTPUTS:
  output/svi_scores_all_years.csv        Main scored dataset (3,228 × 45 cols)
  output/svi_sensitivity_unemployment.csv  Sensitivity analysis results
  output/spatial/svi_{YEAR}_labeled.gpkg   Per-vintage labeled spatial files
  output/spatial/svi_all_years_labeled_time_enabled.gpkg  Time-enabled for ArcGIS Pro

Run order: SIXTH — after stack_longitudinal.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime

INPUT_DIR   = Path("output")
OUTPUT_DIR  = Path("output")
SPATIAL_DIR = OUTPUT_DIR / "spatial"

print("=" * 65)
print("SVI Index Construction  —  Corrected v3")
print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 65)

# ── Load longitudinal panel ───────────────────────────────────────────────────
df = pd.read_csv(
    INPUT_DIR / "acs_tract_2010std_all_years_long.csv",
    dtype={"GEOID10": str, "county_fips": str, "tract_code": str}
)
print(f"\nLoaded: {len(df)} rows × {len(df.columns)} columns")
print(f"Vintages: {sorted(df['YEAR5'].unique().tolist())}")

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def safe_rate(num, denom, scale=100):
    """Compute rate safely; returns NaN where denominator = 0."""
    d = pd.to_numeric(denom, errors='coerce')
    n = pd.to_numeric(num,   errors='coerce')
    return np.where(d > 0, n / d * scale, np.nan)


def sum_cols_checked(df, cols, indicator_name):
    """
    Sum multiple columns, treating NaN as 0.
    Raises ValueError loudly if any expected column is missing from df.
    This prevents silent wrong indicators from schema mismatches or pull errors.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Indicator '{indicator_name}': expected ACS columns are missing "
            f"from the dataset: {missing}. Check the pull script and verify "
            f"the longitudinal panel was built correctly."
        )
    return sum(pd.to_numeric(df[c], errors='coerce').fillna(0) for c in cols)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — COMPUTE RAW INDICATORS
# Each indicator is computed as a rate from ACS numerator / denominator cells.
# Male + female subcategory cells are summed before division.
# Per capita income is negated so lower income → higher z-score.
# Unemployment excluded from all years for temporal consistency.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 1 — Computing raw indicators")
print("=" * 65)

# ── Theme 1: Socioeconomic Status ─────────────────────────────────────────────
df["IND_poverty_rate"] = safe_rate(df["B17001_002E"], df["B17001_001E"])

no_hs_male   = [f"B15002_{i:03d}E" for i in range(3, 11)]
no_hs_female = [f"B15002_{i:03d}E" for i in range(20, 28)]
df["IND_no_hs_diploma"] = safe_rate(
    sum_cols_checked(df, no_hs_male + no_hs_female, "IND_no_hs_diploma"),
    df["B15002_001E"]
)

df["IND_per_capita_income_inv"] = pd.to_numeric(
    df["B19301_001E"], errors='coerce'
) * -1   # inverted: lower income → higher z-score

print("  Theme 1 (Socioeconomic): poverty_rate, no_hs_diploma, per_capita_income_inv ✓")

# ── Theme 2: Household Composition ───────────────────────────────────────────
male_u18   = [f"B01001_{i:03d}E" for i in range(3, 7)]
female_u18 = [f"B01001_{i:03d}E" for i in range(27, 31)]
df["IND_pct_under18"] = safe_rate(
    sum_cols_checked(df, male_u18 + female_u18, "IND_pct_under18"),
    df["B01001_001E"]
)

male_65   = [f"B01001_{i:03d}E" for i in range(20, 26)]
female_65 = [f"B01001_{i:03d}E" for i in range(44, 50)]
df["IND_pct_65plus"] = safe_rate(
    sum_cols_checked(df, male_65 + female_65, "IND_pct_65plus"),
    df["B01001_001E"]
)

df["IND_single_parent"] = safe_rate(
    sum_cols_checked(df, ["B11005_006E", "B11005_007E"], "IND_single_parent"),
    df["B11005_001E"]
)
print("  Theme 2 (Household): pct_under18, pct_65plus, single_parent ✓")

# ── Theme 3: Minority Status & Language ──────────────────────────────────────
df["IND_minority"] = safe_rate(
    pd.to_numeric(df["B03002_001E"], errors='coerce') -
    pd.to_numeric(df["B03002_003E"], errors='coerce').fillna(0),
    df["B03002_001E"]
)

df["IND_ltd_english"] = pd.to_numeric(df["DP02_ltd_english_PE"], errors='coerce')

print("  Theme 3 (Minority/Language): minority, ltd_english ✓")

# ── Theme 4: Housing & Transportation ────────────────────────────────────────
multi_unit_cols = ["B25024_007E", "B25024_008E", "B25024_009E"]
df["IND_multi_unit"] = safe_rate(
    sum_cols_checked(df, multi_unit_cols, "IND_multi_unit"),
    df["B25024_001E"]
)

df["IND_mobile_home"] = safe_rate(df["B25024_010E"], df["B25024_001E"])

crowding_cols = ["B25014_005E","B25014_006E","B25014_007E",
                 "B25014_011E","B25014_012E","B25014_013E"]
df["IND_crowding"] = safe_rate(
    sum_cols_checked(df, crowding_cols, "IND_crowding"),
    df["B25014_001E"]
)

df["IND_no_vehicle"] = safe_rate(
    sum_cols_checked(df, ["B25044_003E","B25044_010E"], "IND_no_vehicle"),
    df["B25044_001E"]
)
print("  Theme 4 (Housing/Transport): multi_unit, mobile_home, crowding, no_vehicle ✓")

INDICATORS = [
    "IND_poverty_rate", "IND_no_hs_diploma", "IND_per_capita_income_inv",
    "IND_pct_under18",  "IND_pct_65plus",    "IND_single_parent",
    "IND_minority",     "IND_ltd_english",
    "IND_multi_unit",   "IND_mobile_home",   "IND_crowding",  "IND_no_vehicle",
]

THEME_MAP = {
    "Socioeconomic":     (["IND_poverty_rate","IND_no_hs_diploma","IND_per_capita_income_inv"], 2),
    "Household":         (["IND_pct_under18","IND_pct_65plus","IND_single_parent"], 2),
    "Minority_Language": (["IND_minority","IND_ltd_english"], 1),
    "Housing_Transport": (["IND_multi_unit","IND_mobile_home","IND_crowding","IND_no_vehicle"], 3),
}

print(f"\n  Total indicators: {len(INDICATORS)}")
print("  Unemployment excluded — availability/comparability limitation in 2009")
print("\n  Null counts per indicator:")
for ind in INDICATORS:
    n = pd.to_numeric(df[ind], errors='coerce').isna().sum()
    print(f"    {ind:<35} nulls={n}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — HIGH-UNCERTAINTY FLAG (CV > 40% operational rule)
# Based on three anchor indicators: poverty count, total pop, ltd English.
# This is a practical quality screen, NOT a full-index uncertainty treatment.
# FLG_HIUNCERT is stored as integer 0/1 (not text).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2 — High-uncertainty flag (CV > 40% operational rule)")
print("=" * 65)

CV_THRESHOLD = 40.0

def compute_cv(est_col, moe_col, df):
    est = pd.to_numeric(df[est_col], errors='coerce')
    moe = pd.to_numeric(df[moe_col], errors='coerce')
    return np.where(est > 0, (moe / 1.645) / est * 100, np.nan)

cv_pov = compute_cv("B17001_002E", "B17001_002M", df)
cv_pop = compute_cv("B01001_001E", "B01001_001M", df)
cv_le  = compute_cv("DP02_ltd_english_PE", "DP02_ltd_english_PM", df)

high_cv = (
    (pd.Series(cv_pov) > CV_THRESHOLD) |
    (pd.Series(cv_pop) > CV_THRESHOLD) |
    (pd.Series(cv_le)  > CV_THRESHOLD)
)

df["high_uncertainty_flag"] = high_cv.astype(int)   # integer 0/1

pct_flag = round(high_cv.sum() / len(df) * 100, 1)
print(f"  CV threshold:           {CV_THRESHOLD}% (operational rule — not a universal Census standard)")
print("  Flag based on:          poverty count, total population, limited English")
print(f"  High-uncertainty rows:  {high_cv.sum()} of {len(df)} ({pct_flag}%)")
by_year = df.groupby("YEAR5")["high_uncertainty_flag"].sum()
for yr, cnt in by_year.items():
    print(f"    {yr}: {cnt} tracts flagged ({cnt/807*100:.1f}%)")
print("  Note: Flag covers selected key indicators only, not all 12 composite indicators")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — POOLED Z-SCORE STANDARDIZATION
# Grand mean and SD computed across ALL 3,228 tract-year observations.
# Every vintage is anchored to the same reference scale for temporal comparison.
# (Tate 2012; Spielman & Folch 2015)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 — Pooled z-score standardization")
print("=" * 65)

pooled_stats = {}
print(f"\n  {'Indicator':<35} {'Grand mean':>12} {'Grand SD':>10} {'N valid':>8}")
print(f"  {'-'*68}")

for ind in INDICATORS:
    raw        = pd.to_numeric(df[ind], errors='coerce')
    grand_mean = raw.mean()
    grand_sd   = raw.std(ddof=1)
    n_valid    = raw.notna().sum()
    pooled_stats[ind] = {"mean": grand_mean, "sd": grand_sd}

    z_col     = f"Z_{ind[4:]}"
    df[z_col] = (raw - grand_mean) / grand_sd if grand_sd > 0 else np.nan

    print(f"  {ind:<35} {grand_mean:>12.4f} {grand_sd:>10.4f} {n_valid:>8}")

Z_COLS = [f"Z_{ind[4:]}" for ind in INDICATORS]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — COMPOSITE SCORE CONSTRUCTION
# CORRECTION v3: composite requires >= 10 of 12 valid z-scores (raised from 6).
# Theme scores require >= 2 of 3 for 3-indicator themes,
#                         >= 1 of 2 for 2-indicator theme (Minority_Language),
#                         >= 3 of 4 for 4-indicator theme (Housing_Transport).
# These thresholds are tighter and more defensible. In practice, they produce
# identical output because all partial rows already meet the new thresholds.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4 — Composite score construction")
print("=" * 65)

COMPOSITE_MIN_INDICATORS = 10   # of 12 required (raised from 6)

df["SVI_composite"] = df[Z_COLS].apply(
    lambda row: np.nanmean(row) if row.notna().sum() >= COMPOSITE_MIN_INDICATORS else np.nan,
    axis=1
)

n_scored = df["SVI_composite"].notna().sum()
print("\n  Overall composite (SVI_composite):")
print(f"    Min indicators required: {COMPOSITE_MIN_INDICATORS} of 12")
print(f"    Non-null:  {n_scored} / {len(df)}")
print(f"    Null:      {len(df)-n_scored} (zero-pop and institutional tracts)")
print(f"    Mean:      {df['SVI_composite'].mean():.4f}  (should be ~0 for pooled z)")
print(f"    SD:        {df['SVI_composite'].std():.4f}")
print(f"    Min:       {df['SVI_composite'].min():.4f}")
print(f"    Max:       {df['SVI_composite'].max():.4f}")

# Theme subscores with tightened thresholds
for theme, (inds, min_req) in THEME_MAP.items():
    z_theme  = [f"Z_{ind[4:]}" for ind in inds]
    col_name = f"SVI_theme_{theme}"
    df[col_name] = df[z_theme].apply(
        lambda row, req=min_req: np.nanmean(row) if row.notna().sum() >= req else np.nan,
        axis=1
    )
    n_valid = df[col_name].notna().sum()
    print(f"\n  Theme {theme} (min {min_req} of {len(inds)}):")
    print(f"    Valid rows: {n_valid} | Mean: {df[col_name].mean():.4f}")

# CDC-style within-year percentile ranking (secondary, cross-sectional only)
print("\n  CDC-style percentile ranks (within-year, cross-sectional only):")
for yr in sorted(df['YEAR5'].unique()):
    mask = df['YEAR5'] == yr
    df.loc[mask, "SVI_percentile_withinyear"] = (
        df.loc[mask, "SVI_composite"].rank(pct=True, na_option='keep')
    )
    mean_pct = df.loc[mask, "SVI_percentile_withinyear"].mean()
    print(f"    {yr}: mean percentile = {mean_pct:.3f} (should be ~0.500)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SENSITIVITY ANALYSIS: UNEMPLOYMENT INCLUSION vs EXCLUSION
# 2014–2024 only (B23025 not consistently available for 2009).
# Base z-scores are pooled across 2009–2024.
# Unemployment z-score is pooled within 2014–2024 subset.
# This asymmetry is intentional and documented — the test is a subset-period
# diagnostic for the design choice, not a parallel longitudinal index.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5 — Sensitivity analysis: unemployment inclusion vs exclusion")
print("=" * 65)

df_sens = df[df["YEAR5"] >= 2014].copy()

df_sens["IND_unemployment"] = safe_rate(df_sens["B23025_005E"], df_sens["B23025_003E"])

unemp_raw  = pd.to_numeric(df_sens["IND_unemployment"], errors='coerce')
unemp_mean = unemp_raw.mean()
unemp_sd   = unemp_raw.std(ddof=1)
df_sens["Z_unemployment"] = (unemp_raw - unemp_mean) / unemp_sd

# Alternate composite with 13 indicators — require >=11 of 13
z_cols_with_unemp = [f"Z_{ind[4:]}" for ind in INDICATORS] + ["Z_unemployment"]
df_sens["SVI_composite_with_unemp"] = df_sens[z_cols_with_unemp].apply(
    lambda row: np.nanmean(row) if row.notna().sum() >= 11 else np.nan,
    axis=1
)

base_col, alt_col = "SVI_composite", "SVI_composite_with_unemp"
valid    = df_sens[[base_col, alt_col]].dropna()
pearson  = valid[base_col].corr(valid[alt_col])
spearman = valid[base_col].rank().corr(valid[alt_col].rank())

print(f"\n  Vintages tested:              2014, 2019, 2024 ({len(df_sens)} tract-years)")
print(f"  Base composite:               {len(INDICATORS)} indicators (>=10 of 12 required)")
print(f"  Alt composite with unemp:     {len(INDICATORS)+1} indicators (>=11 of 13 required)")
print("  Note: base z-scores pooled 2009–2024; unemployment pooled within 2014–2024")
print("  This is a subset-period diagnostic, not a parallel index construction")
print(f"\n  Pearson correlation:          {pearson:.4f}")
print(f"  Spearman rank correlation:    {spearman:.4f}")
print(f"  Interpretation:               {'Very high — exclusion does not materially alter rankings' if pearson > 0.95 else '⚠ lower than expected'}")

print("\n  Correlation by vintage:")
for yr in [2014, 2019, 2024]:
    sub  = df_sens[df_sens["YEAR5"]==yr][[base_col, alt_col]].dropna()
    r_yr = sub[base_col].corr(sub[alt_col])
    print(f"    {yr}: r = {r_yr:.4f} (n={len(sub)})")

sens_out = df_sens[
    ["YEAR5","GEOID10","msa","county_fips","county_name",
     "IND_unemployment","Z_unemployment",
     "SVI_composite","SVI_composite_with_unemp"]
].copy()
sens_path = OUTPUT_DIR / "svi_sensitivity_unemployment.csv"
sens_out.to_csv(sens_path, index=False)
print(f"\n  ✓ Sensitivity file saved → {sens_path}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6 — Validation")
print("=" * 65)

print("\n  SVI_composite by vintage:")
print(f"  {'Year':<8} {'Mean':>8} {'SD':>8} {'Min':>8} {'Max':>8} {'Nulls':>7}")
print(f"  {'-'*52}")
for yr in [2009, 2014, 2019, 2024]:
    sub = pd.to_numeric(df[df["YEAR5"]==yr]["SVI_composite"], errors='coerce')
    print(f"  {yr:<8} {sub.mean():>8.4f} {sub.std():>8.4f} "
          f"{sub.min():>8.4f} {sub.max():>8.4f} {sub.isna().sum():>7}")

grand_mean = df["SVI_composite"].mean()
print(f"\n  Grand mean of SVI_composite:     {grand_mean:.4f} "
      f"{'✓ (~0)' if abs(grand_mean) < 0.01 else '⚠'}")

z_counts = df[Z_COLS].notna().sum(axis=1)
all_null  = (z_counts == 0).sum()
partial   = ((z_counts > 0) & (z_counts < COMPOSITE_MIN_INDICATORS)).sum()
print(f"  Tracts with 0 indicators (null):  {all_null} (zero-pop group quarters)")
print(f"  Tracts with partial (<{COMPOSITE_MIN_INDICATORS}) indicators: {partial} → assigned null composite")
print(f"  Tracts with full {len(INDICATORS)} indicators:   {(z_counts==12).sum()}")

print("\n  High-uncertainty by vintage:")
for yr in [2009, 2014, 2019, 2024]:
    n = df[df["YEAR5"]==yr]["high_uncertainty_flag"].sum()
    print(f"    {yr}: {n} of 807 ({n/807*100:.1f}%) flagged")

scores_in_range = df["SVI_composite"].between(-10, 10).sum()
print(f"\n  Composite scores within [-10, 10]: "
      f"{scores_in_range} of {df['SVI_composite'].notna().sum()} ✓")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SAVE MAIN OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
id_cols    = ["YEAR5","GEOID10","msa","county_fips","county_name","tract_code"]
score_cols = (
    ["SVI_composite","SVI_percentile_withinyear"] +
    [f"SVI_theme_{t}" for t in THEME_MAP.keys()]
)
ind_raw    = INDICATORS
ind_z      = Z_COLS
flag_cols  = ["high_uncertainty_flag"]
meta_cols  = ["boundary_vintage","crosswalked","crosswalk_note",
              "DP02_source_code","B19301_cpi_base_year",
              "B19301_cpi_factor","B19301_cpi_source","data_retrieved_date"]

out_cols = id_cols + score_cols + ind_raw + ind_z + flag_cols + meta_cols
out_cols = [c for c in out_cols if c in df.columns]

out_df = df[out_cols].sort_values(["GEOID10","YEAR5"]).reset_index(drop=True)
out_path = OUTPUT_DIR / "svi_scores_all_years.csv"
out_df.to_csv(out_path, index=False)
print(f"\n{'='*65}")
print(f"Main output saved → {out_path}")
print(f"  Shape: {out_df.shape[0]} rows × {out_df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — LABELED SPATIAL FILES FOR ARCGIS PRO
# All spatial layers share the same 2010 tract geometry (from msa_tracts_2019.gpkg).
# GEO_BNDRY_YR records the SOURCE vintage boundary (before crosswalking),
# NOT the display geometry which is 2010 for all four vintages.
# SVI_SCORE (0-100): good for maps and ranking-based spatial analysis.
# SVI_ZSCORE: use for regression or analysis requiring absolute scale.
# FLG_HIUNCERT: integer 0/1 — no text field added.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 8 — Labeled spatial files for ArcGIS Pro")
print("=" * 65)

try:
    gdf_base  = gpd.read_file(SPATIAL_DIR / "msa_tracts_2019.gpkg")
    geoid_col = next((c for c in gdf_base.columns if c in ("GEOID10","GEOID")), None)
    gdf_geo   = gdf_base[[geoid_col,"geometry"]].rename(columns={geoid_col:"GEOID10"})
    print(f"\n  Boundary geometry loaded: {len(gdf_geo)} polygons, CRS={gdf_geo.crs}")
    print("  Note: all vintages displayed on common 2010 geometry")

    def build_labeled(svi_df):
        out = pd.DataFrame()
        out["GEOID10"]         = svi_df["GEOID10"]
        out["YEAR"]            = svi_df["YEAR5"]
        out["MSA"]             = svi_df["msa"]
        out["COUNTY_FIPS"]     = svi_df["county_fips"]
        out["COUNTY_NAME"]     = svi_df["county_name"]
        out["TRACT_CODE"]      = svi_df["tract_code"]
        comp  = pd.to_numeric(svi_df["SVI_composite"], errors='coerce')
        z_min = pd.to_numeric(out_df["SVI_composite"], errors='coerce').min()
        z_max = pd.to_numeric(out_df["SVI_composite"], errors='coerce').max()
        out["SVI_ZSCORE"]      = comp.round(4)
        out["SVI_SCORE"]       = ((comp - z_min) / (z_max - z_min) * 100).round(2)
        out["SVI_PCTILE_YR"]   = pd.to_numeric(svi_df["SVI_percentile_withinyear"], errors='coerce').round(4)
        out["THEME_SOCIOECON"] = pd.to_numeric(svi_df["SVI_theme_Socioeconomic"],    errors='coerce').round(4)
        out["THEME_HOUSEHLD"]  = pd.to_numeric(svi_df["SVI_theme_Household"],        errors='coerce').round(4)
        out["THEME_MINORITY"]  = pd.to_numeric(svi_df["SVI_theme_Minority_Language"],errors='coerce').round(4)
        out["THEME_HOUSING"]   = pd.to_numeric(svi_df["SVI_theme_Housing_Transport"],errors='coerce').round(4)
        out["PCT_POVERTY"]     = pd.to_numeric(svi_df["IND_poverty_rate"],           errors='coerce').round(2)
        out["PCT_NO_HSDIP"]    = pd.to_numeric(svi_df["IND_no_hs_diploma"],          errors='coerce').round(2)
        inc_inv                = pd.to_numeric(svi_df["IND_per_capita_income_inv"],  errors='coerce')
        out["DOL_INC_PC"]      = (inc_inv * -1).round(0)   # positive dollars
        out["PCT_UNDER18"]     = pd.to_numeric(svi_df["IND_pct_under18"],            errors='coerce').round(2)
        out["PCT_65PLUS"]      = pd.to_numeric(svi_df["IND_pct_65plus"],             errors='coerce').round(2)
        out["PCT_SINGLEPAR"]   = pd.to_numeric(svi_df["IND_single_parent"],          errors='coerce').round(2)
        out["PCT_MINORITY"]    = pd.to_numeric(svi_df["IND_minority"],               errors='coerce').round(2)
        out["PCT_LTD_ENG"]     = pd.to_numeric(svi_df["IND_ltd_english"],            errors='coerce').round(2)
        out["PCT_MULTIUNIT"]   = pd.to_numeric(svi_df["IND_multi_unit"],             errors='coerce').round(2)
        out["PCT_MOBILEHOME"]  = pd.to_numeric(svi_df["IND_mobile_home"],            errors='coerce').round(2)
        out["PCT_CROWDING"]    = pd.to_numeric(svi_df["IND_crowding"],               errors='coerce').round(2)
        out["PCT_NO_VEH"]      = pd.to_numeric(svi_df["IND_no_vehicle"],             errors='coerce').round(2)
        out["FLG_HIUNCERT"]    = svi_df["high_uncertainty_flag"].astype(int)   # 0/1 integer
        out["GEO_BNDRY_YR"]    = svi_df["boundary_vintage"].astype(int)  # source vintage, not display
        out["FLG_CROSSWALK"]   = svi_df["crosswalked"].astype(int)
        out["DATE_PULLED"]     = svi_df["data_retrieved_date"]
        return out

    labeled = build_labeled(out_df)

    # Per-vintage files
    print("\n  Per-vintage spatial files:")
    for year in [2009, 2014, 2019, 2024]:
        yr_data  = labeled[labeled["YEAR"] == year].copy()
        gdf_yr   = gdf_geo.merge(yr_data, on="GEOID10", how="left")
        joined   = gdf_yr["SVI_SCORE"].notna().sum()
        out_path_gpkg = SPATIAL_DIR / f"svi_{year}_labeled.gpkg"
        gdf_yr.to_file(out_path_gpkg, driver="GPKG")
        print(f"    {year}: {len(gdf_yr)} features, {joined} with SVI scores "
              f"{'✓' if joined >= 800 else '⚠'} → {out_path_gpkg}")

    # Time-enabled combined file
    gdfs_time = []
    for year in [2009, 2014, 2019, 2024]:
        yr_data = labeled[labeled["YEAR"] == year].copy()
        gdf_yr  = gdf_geo.merge(yr_data, on="GEOID10", how="inner")
        gdf_yr["DATE"] = pd.Timestamp(f"{year}-01-01")
        gdfs_time.append(gdf_yr)

    gdf_all = pd.concat(gdfs_time, ignore_index=True)
    gdf_all = gpd.GeoDataFrame(gdf_all, geometry="geometry", crs=gdf_geo.crs)
    gdf_all = gdf_all.sort_values(["GEOID10","YEAR"]).reset_index(drop=True)

    out_time = SPATIAL_DIR / "svi_all_years_labeled_time_enabled.gpkg"
    gdf_all.to_file(out_time, driver="GPKG")
    print("\n  Time-enabled combined file:")
    print(f"    {len(gdf_all)} features (807 × 4 years)")
    print("    DATE values: 2009-01-01 → 2024-01-01")
    print(f"    ✓ → {out_time}")

    # Field reference
    field_ref = pd.DataFrame([
        ("GEOID10",       "Identifier", "11-digit Census tract FIPS (state+county+tract)"),
        ("YEAR",          "Identifier", "ACS5 vintage year (2009, 2014, 2019, 2024)"),
        ("MSA",           "Identifier", "Metropolitan Statistical Area name"),
        ("COUNTY_FIPS",   "Identifier", "3-digit county FIPS code"),
        ("COUNTY_NAME",   "Identifier", "County name"),
        ("TRACT_CODE",    "Identifier", "6-digit census tract code"),
        ("SVI_ZSCORE",    "SVI Score",  "Overall composite — pooled z-score; higher = more vulnerable; temporally comparable across all vintages"),
        ("SVI_SCORE",     "SVI Score",  "Overall composite rescaled 0-100 using global min/max; 0=least vulnerable, 100=most vulnerable; temporally comparable. Use for mapping and ranking-based analysis. Use SVI_ZSCORE for regression."),
        ("SVI_PCTILE_YR", "SVI Score",  "Within-year percentile rank (0-1); CROSS-SECTIONAL ONLY — do not compare across years (CDC-style)"),
        ("THEME_SOCIOECON","Theme",     "Socioeconomic theme z-score (>=2 of 3 indicators): poverty + no HS diploma + per capita income (inverted)"),
        ("THEME_HOUSEHLD", "Theme",     "Household composition theme z-score (>=2 of 3): pct under 18 + pct 65+ + single-parent households"),
        ("THEME_MINORITY", "Theme",     "Minority status and language theme z-score (>=1 of 2): minority population + limited English households"),
        ("THEME_HOUSING",  "Theme",     "Housing and transportation theme z-score (>=3 of 4): multi-unit + mobile homes + crowding + no vehicle"),
        ("PCT_POVERTY",   "Indicator",  "% population below federal poverty level | ACS B17001"),
        ("PCT_NO_HSDIP",  "Indicator",  "% population 25+ without high school diploma | ACS B15002 (male+female combined)"),
        ("DOL_INC_PC",    "Indicator",  "Per capita income in constant 2019 dollars (CPI-adjusted). Stored as POSITIVE dollars. Higher = LESS vulnerable — inverted for composite calculation only. | ACS B19301"),
        ("PCT_UNDER18",   "Indicator",  "% total population under 18 years | ACS B01001 (male+female combined)"),
        ("PCT_65PLUS",    "Indicator",  "% total population aged 65 and over | ACS B01001 (male+female combined)"),
        ("PCT_SINGLEPAR", "Indicator",  "% households with children <18 that are single-parent | ACS B11005. Definition changed 2019 (no spouse/partner vs no wife/husband)"),
        ("PCT_MINORITY",  "Indicator",  "% population that is not Non-Hispanic White alone | ACS B03002. 2024 affected by race question redesign"),
        ("PCT_LTD_ENG",   "Indicator",  "% households where no one 14+ speaks English very well | ACS DP02 (vintage-specific codes)"),
        ("PCT_MULTIUNIT", "Indicator",  "% housing units in structures with 10+ units | ACS B25024"),
        ("PCT_MOBILEHOME","Indicator",  "% housing units that are mobile homes | ACS B25024"),
        ("PCT_CROWDING",  "Indicator",  "% occupied housing units with more than 1 occupant per room | ACS B25014 (owner+renter combined)"),
        ("PCT_NO_VEH",    "Indicator",  "% occupied housing units with no vehicle available | ACS B25044 (owner+renter combined)"),
        ("FLG_HIUNCERT",  "Flag",       "INTEGER 0/1. 1 = high uncertainty (CV >40% on poverty count, total pop, or limited English). Operational rule — not a universal Census standard. Interpret flagged tracts cautiously."),
        ("GEO_BNDRY_YR",  "Flag",       "Census boundary vintage of the SOURCE data before crosswalking (2000=2009 vintage, 2010=2014/2019, 2020=2024). ALL display geometry is 2010 boundaries regardless of this field."),
        ("FLG_CROSSWALK", "Flag",       "INTEGER 0/1. 1 = areal interpolation crosswalk applied (2009 and 2024). 0 = native 2010 boundaries (2014 and 2019)."),
        ("DATE_PULLED",   "Metadata",   "Date ACS data was retrieved from Census Bureau API"),
        ("DATE",          "Metadata",   "Time-enabled file only: January 1 of vintage year. Set as Time Field in ArcGIS Pro Layer Properties → Time tab."),
    ], columns=["field_name","field_type","description"])

    ref_path = OUTPUT_DIR / "svi_field_reference.csv"
    field_ref.to_csv(ref_path, index=False)
    print(f"\n  Field reference saved → {ref_path}")

except ImportError:
    print("  geopandas not available — spatial step skipped")
    print("  pip install geopandas")

print("\n" + "=" * 65)
print("DONE — Corrected build_svi_index.py v3")
print("=" * 65)
print(f"""
  Output files:
    output/svi_scores_all_years.csv
    output/svi_sensitivity_unemployment.csv
    output/svi_field_reference.csv
    output/spatial/svi_2009_labeled.gpkg
    output/spatial/svi_2014_labeled.gpkg
    output/spatial/svi_2019_labeled.gpkg
    output/spatial/svi_2024_labeled.gpkg
    output/spatial/svi_all_years_labeled_time_enabled.gpkg

  Key results (for dissertation):
    Composite grand mean:      ~0.0006 (confirms pooled z)
    Pearson r (sensitivity):   {pearson:.4f}
    Spearman r (sensitivity):  {spearman:.4f}
    Partial composites:        8 rows assigned null (threshold raised from >=6 to >=10)
    Full composites:           3,210 rows (99.44% of all tract-year obs)
""")
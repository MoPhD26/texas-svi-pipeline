"""
================================================================================
RQ3: Hazard-Vulnerability Co-location Analysis
================================================================================
Research Question 3: To what extent are persistently high-vulnerability tracts
spatially co-located with elevated modeled natural-hazard burden, and does that
spatial congruence differ between the Austin and San Antonio metropolitan areas?

Author: Mogahid Hussein
Institution: Texas State University, Department of Geography
Created: 2026-04-29
Pipeline order: step 10, after the trajectory typology (07 to 09)

================================================================================
INPUTS
================================================================================
1. FEMA NRI tract-level CSV (December 2025 v1.20)
   - Tract-level data from https://hazards.fema.gov/nri/data-resources
   - Either national OR Texas-only state CSV; both work
   - Boundary vintage: 2021 TIGER/Line (= Census 2020 tract boundaries)

2. Project files (already produced by earlier pipeline scripts):
   - trajectory_typology.csv       (script 8 output)
   - quintile_assignments.csv      (script 9 output)
   - svi_scores_all_years.csv      (script 6 output)
   - trajectory_map.gpkg           (script 7 output, optional, for spatial join)

3. Crosswalk file (for 2020->2010 boundary harmonization):
   - tab20_tract20_tract10_natl.txt (Census Bureau, optional but recommended)
   - URL: https://www2.census.gov/geo/docs/maps-data/data/rel2020/tract/

================================================================================
OUTPUTS
================================================================================
Tabular (saved to OUTPUT_DIR):
  - rq3_master_panel_RQ3.csv           Master 807-tract analytical panel
  - rq3_eal_by_trajectory_RQ3.csv      Per-class EAL summary statistics
  - rq3_kruskal_wallis_RQ3.csv         Kruskal-Wallis test results
  - rq3_dunns_posthoc_RQ3.csv          Dunn's post-hoc with BH-FDR
  - rq3_quintile_overlap_combined_RQ3.csv  5x5 contingency, all tracts
  - rq3_quintile_overlap_austin_RQ3.csv    5x5 contingency, Austin MSA
  - rq3_quintile_overlap_sanantonio_RQ3.csv 5x5 contingency, San Antonio MSA
  - rq3_chi_square_RQ3.csv             Chi-square test results
  - rq3_msa_comparison_RQ3.csv         Austin vs San Antonio comparison
  - rq3_geoda_input_RQ3.csv            Input ready for bivariate Lee's L

Spatial (saved to SPATIAL_OUTPUT_DIR, if geometry source available):
  - rq3_master_panel_RQ3.gpkg          Tract polygons + all RQ3 attributes
  - rq3_persistent_high_eal_RQ3.gpkg   Subset: Persistently_High tracts only

================================================================================
HOW TO RUN (PyCharm or terminal)
================================================================================
1. Run from the repository root, after steps 01 to 09.
2. Download the FEMA NRI tract CSV to data/nri/NRI_Table_CensusTracts_Texas.csv,
   or set NRI_CSV to another location.
3. Download tab20_tract20_tract10_natl.txt to data/crosswalk/, or set CROSSWALK_FILE.
4. python src/10_hazard_congruence.py
5. All outputs are saved to output/ with an "_RQ3" suffix.

Required Python packages: pandas, numpy, scipy, geopandas (optional, for .gpkg)
Install: pip install pandas numpy scipy geopandas

================================================================================
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# ============================================================================
# CONFIG — EDIT THESE PATHS FOR YOUR LOCAL ENVIRONMENT
# ============================================================================

# Paths are relative to the repository root. Override with environment variables.
import os
PROJECT_ROOT = Path(os.environ.get("SVI_PROJECT_ROOT", "."))
OUTPUT_DIR = PROJECT_ROOT / "output"                  # tabular outputs
SPATIAL_OUTPUT_DIR = PROJECT_ROOT / "output" / "spatial"  # GeoPackage outputs

# Inputs from the existing pipeline (already exist)
TRAJECTORY_CSV = OUTPUT_DIR / "trajectory_typology.csv"
QUINTILE_CSV = OUTPUT_DIR / "quintile_assignments.csv"
SVI_SCORES_CSV = OUTPUT_DIR / "svi_scores_all_years.csv"
TRAJECTORY_GPKG = SPATIAL_OUTPUT_DIR / "trajectory_map.gpkg"   # used for geometry source

# NEW input — FEMA NRI CSV (place anywhere; update this path)
# Use the Texas state CSV (smaller, simpler) OR the national CSV — script handles both
NRI_CSV = Path(os.environ.get("NRI_CSV", PROJECT_ROOT / "data" / "nri" / "NRI_Table_CensusTracts_Texas.csv"))

# Crosswalk file — Census 2020 -> Census 2010 tract relationships
# Optional but recommended. If not present, script falls back to direct GEOID match.
CROSSWALK_FILE = Path(os.environ.get("CROSSWALK_FILE", PROJECT_ROOT / "data" / "crosswalk" / "tab20_tract20_tract10_natl.txt"))

# ============================================================================
# Constants — do not edit
# ============================================================================

# 13 OMB-defined MSA counties (verified against OMB Bulletin 23-01, July 2023)
# NOTE: OMB renamed Austin MSA from "...-Georgetown" to "...-San Marcos" in 2023.
# We keep the Georgetown naming consistent with how data was pulled across vintages.
MSA_COUNTIES = {
    # Austin-Round Rock-Georgetown MSA — 5 counties
    "48021": "Austin-Round Rock-Georgetown MSA",   # Bastrop
    "48055": "Austin-Round Rock-Georgetown MSA",   # Caldwell
    "48209": "Austin-Round Rock-Georgetown MSA",   # Hays
    "48453": "Austin-Round Rock-Georgetown MSA",   # Travis
    "48491": "Austin-Round Rock-Georgetown MSA",   # Williamson
    # San Antonio-New Braunfels MSA — 8 counties
    "48013": "San Antonio-New Braunfels MSA",      # Atascosa
    "48019": "San Antonio-New Braunfels MSA",      # Bandera
    "48029": "San Antonio-New Braunfels MSA",      # Bexar
    "48091": "San Antonio-New Braunfels MSA",      # Comal
    "48187": "San Antonio-New Braunfels MSA",      # Guadalupe
    "48259": "San Antonio-New Braunfels MSA",      # Kendall
    "48325": "San Antonio-New Braunfels MSA",      # Medina
    "48493": "San Antonio-New Braunfels MSA",      # Wilson
}


# ============================================================================
# STEP 1 — Load and filter NRI to study area
# ============================================================================
def load_and_filter_nri(nri_path):
    """Load FEMA NRI tract-level CSV, filter to 13 MSA counties.

    NRI v1.20 columns of primary interest:
      TRACTFIPS    11-digit Census 2020 tract GEOID (string with leading zeros)
      STCOFIPS     5-digit county FIPS (state+county); some files use COUNTYFIPS
      EAL_VALT     Composite Expected Annual Loss, all 18 hazards (USD)
      EAL_SCORE    Composite EAL national percentile score (0-100)
      EAL_VALB     EAL building component
      EAL_VALP     EAL population component (statistical valuation, USD)
      EAL_VALA     EAL agriculture component
      RISK_VALUE   Composite Risk Index value (EAL × SoVI / Resilience)
      RISK_SCORE   Composite Risk Index national percentile score
    """
    print("[1/9] Loading FEMA NRI tract data...")
    print(f"      Source: {nri_path}")

    # Read with all FIPS columns as strings to preserve leading zeros
    fips_cols = {"TRACTFIPS": str, "COUNTYFIPS": str,
                 "STATEFIPS": str, "STCOFIPS": str}
    nri = pd.read_csv(nri_path, dtype=fips_cols, low_memory=False)
    print(f"      Total tracts in NRI file: {len(nri):,}")

    # NRI files have used different names across versions for the county FIPS column.
    # Normalize to STCOFIPS (the v1.20 standard).
    if "STCOFIPS" not in nri.columns and "COUNTYFIPS" in nri.columns:
        nri = nri.rename(columns={"COUNTYFIPS": "STCOFIPS"})

    # Pad to expected widths
    nri["STCOFIPS"] = nri["STCOFIPS"].astype(str).str.zfill(5)
    nri["TRACTFIPS"] = nri["TRACTFIPS"].astype(str).str.zfill(11)

    # Filter to study area
    msa_nri = nri[nri["STCOFIPS"].isin(MSA_COUNTIES.keys())].copy()
    msa_nri["MSA"] = msa_nri["STCOFIPS"].map(MSA_COUNTIES)
    msa_nri["GEOID20"] = msa_nri["TRACTFIPS"]

    print(f"      Filtered to 13 MSA counties: {len(msa_nri):,} tracts (2020 boundaries)")
    austin_n = (msa_nri["MSA"] == "Austin-Round Rock-Georgetown MSA").sum()
    sa_n = (msa_nri["MSA"] == "San Antonio-New Braunfels MSA").sum()
    print(f"      Austin MSA: {austin_n} | San Antonio MSA: {sa_n}")

    if "EAL_VALT" not in msa_nri.columns:
        sample = list(msa_nri.columns)[:30]
        raise ValueError(f"EAL_VALT column not found. First 30 columns: {sample}")

    n_null = msa_nri["EAL_VALT"].isna().sum()
    n_zero = (msa_nri["EAL_VALT"].fillna(-1) == 0).sum()
    n_pos = (msa_nri["EAL_VALT"].fillna(-1) > 0).sum()
    print(f"      EAL_VALT: {n_pos} positive, {n_zero} zero, {n_null} NaN")
    return msa_nri


# ============================================================================
# STEP 2 — Crosswalk NRI from 2020 to 2010 tract boundaries
# ============================================================================
def crosswalk_nri_to_2010(msa_nri, crosswalk_path):
    """Apply 2020->2010 crosswalk via area-weighted aggregation.

    EAL_VALT is a dollar-value variable (annual loss summed across 18 hazards).
    For dollar values associated with area-distributed economic activity,
    area-weighted aggregation is appropriate (same convention used for population
    counts in the dissertation's existing 2009 and 2024 ACS crosswalks).

    weight_i = AREALAND_PART_i / AREALAND_TRACT_20_i
    EAL_2010_j = sum over all 2020 tracts i intersecting 2010 tract j of (EAL_i * weight_i)
    """
    print("[2/9] Applying 2020->2010 crosswalk to NRI EAL...")

    if not Path(crosswalk_path).exists():
        print(f"      ⚠️  Crosswalk file not found at: {crosswalk_path}")
        print("      Falling back to direct GEOID matching (assumes tract continuity).")
        print("      Roughly 95% of tracts are stable across the 2010->2020 redistricting,")
        print("      but ~5% will have boundary changes the fallback cannot capture.")
        return crosswalk_fallback(msa_nri)

    print(f"      Loading crosswalk: {crosswalk_path}")
    cw = pd.read_csv(crosswalk_path, sep="|",
                     dtype={"GEOID_TRACT_20": str, "GEOID_TRACT_10": str})
    cw = cw[cw["GEOID_TRACT_20"].str.startswith("48")].copy()  # Texas only
    cw["weight"] = cw["AREALAND_PART"] / cw["AREALAND_TRACT_20"]

    # Merge NRI to crosswalk on the 2020 GEOID
    merged = msa_nri.merge(
        cw[["GEOID_TRACT_20", "GEOID_TRACT_10", "weight"]],
        left_on="GEOID20", right_on="GEOID_TRACT_20",
        how="inner"
    )
    print(f"      Crosswalk relationships found: {len(merged):,}")

    # Compute weighted EAL contribution per 2020-2010 part
    merged["EAL_VALT_w"] = merged["EAL_VALT"].fillna(0) * merged["weight"]

    # Aggregate to 2010 boundaries
    cw_eal = merged.groupby("GEOID_TRACT_10").agg(
        EAL_VALT_2010=("EAL_VALT_w", "sum"),
        n_source_tracts_2020=("GEOID_TRACT_20", "nunique"),
    ).reset_index().rename(columns={"GEOID_TRACT_10": "GEOID10"})

    cw_eal["GEOID10"] = cw_eal["GEOID10"].astype(str).str.zfill(11)
    print(f"      Crosswalk produced {len(cw_eal)} 2010-vintage tract aggregations")
    return cw_eal


def crosswalk_fallback(msa_nri):
    """Fallback: direct GEOID match. Assumes 2020 GEOID == 2010 GEOID,
    which is true for ~95% of tracts but introduces error for the rest."""
    out = msa_nri[["GEOID20", "EAL_VALT"]].copy()
    out = out.rename(columns={"GEOID20": "GEOID10",
                              "EAL_VALT": "EAL_VALT_2010"})
    out["n_source_tracts_2020"] = 1
    out["GEOID10"] = out["GEOID10"].astype(str).str.zfill(11)
    return out[["GEOID10", "EAL_VALT_2010", "n_source_tracts_2020"]]


# ============================================================================
# STEP 3 — Build the master analytical panel
# ============================================================================
def build_master_panel(eal_2010, traj_path, quint_path):
    """Join EAL to trajectory typology and quintile assignments.
    Apply human-readable column names suitable for ArcGIS field display."""
    print("[3/9] Building master analytical panel...")

    traj = pd.read_csv(traj_path, dtype={"GEOID10": str})
    quint = pd.read_csv(quint_path, dtype={"GEOID10": str})

    # Pad GEOIDs to 11 digits
    traj["GEOID10"] = traj["GEOID10"].str.zfill(11)
    quint["GEOID10"] = quint["GEOID10"].str.zfill(11)
    eal_2010["GEOID10"] = eal_2010["GEOID10"].astype(str).str.zfill(11)

    # Trajectory file is the canonical source for MSA / county attribution
    panel = traj[["GEOID10", "MSA", "COUNTY_NAME", "Trajectory", "LISA_Sequence",
                  "SVI_2009", "SVI_2014", "SVI_2019", "SVI_2024"]].copy()

    # Bring in quintile fields
    quint_cols = ["GEOID10", "Q_2009", "Q_2014", "Q_2019", "Q_2024",
                  "MOBILITY_TYPE", "Q5_PERSIST", "Q5_ESCAPE"]
    quint_cols = [c for c in quint_cols if c in quint.columns]
    panel = panel.merge(quint[quint_cols], on="GEOID10", how="left")

    # Bring in EAL
    panel = panel.merge(eal_2010, on="GEOID10", how="left")

    # Add EAL quintile (within study area, for the chi-square contingency test)
    panel["EAL_quintile"] = pd.qcut(
        panel["EAL_VALT_2010"], 5,
        labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype("Float64").astype(float)

    # Log-transformed EAL for parametric / mapping use
    panel["EAL_log"] = np.log1p(panel["EAL_VALT_2010"].clip(lower=0))

    # Add a binary persistent-high flag for simpler mapping
    panel["IS_PERSIST_HIGH"] = (panel["Trajectory"] == "Persistently_High").astype(int)

    # ArcGIS-friendly column renames (10-char field name compatibility for older shapefile use,
    # though we save .gpkg by default so longer names are fine).
    rename_map = {
        "EAL_VALT_2010": "EAL_USD",                       # raw dollar value
        "EAL_log": "EAL_USD_LOG",                          # log-scaled for mapping
        "EAL_quintile": "EAL_QUINT",                       # 1=lowest, 5=highest
        "n_source_tracts_2020": "N_SRC_TRACTS",            # # of 2020 tracts contributing
        "Q_2009": "SVI_QUINT_2009",
        "Q_2014": "SVI_QUINT_2014",
        "Q_2019": "SVI_QUINT_2019",
        "Q_2024": "SVI_QUINT_2024",
        "MOBILITY_TYPE": "MOBILITY",
        "Q5_PERSIST": "SVI_Q5_PERSIST",
        "Q5_ESCAPE": "SVI_Q5_ESCAPE",
        "Trajectory": "TRAJECTORY",
        "LISA_Sequence": "LISA_SEQ",
        "COUNTY_NAME": "COUNTY",
    }
    panel = panel.rename(columns={k: v for k, v in rename_map.items()
                                   if k in panel.columns})

    print(f"      Panel size: {len(panel)} tracts")
    print(f"      With EAL data: {panel['EAL_USD'].notna().sum()}")
    print(f"      With Trajectory: {panel['TRAJECTORY'].notna().sum()}")
    print(f"      With SVI_QUINT_2024: {panel['SVI_QUINT_2024'].notna().sum()}")

    # Force GEOID10 to a zero-padded 11-char string (ArcGIS-safe for joins)
    panel["GEOID10"] = panel["GEOID10"].astype(str).str.zfill(11)
    return panel


# ============================================================================
# STEP 4 — Kruskal-Wallis: EAL across trajectory classes
# ============================================================================
def kruskal_wallis_trajectory(panel, output_dir):
    """Test whether EAL distributions differ across LISA trajectory classes."""
    print("[4/9] Kruskal-Wallis test: EAL across trajectory classes...")
    from scipy.stats import kruskal

    df = panel.dropna(subset=["EAL_USD", "TRAJECTORY"]).copy()
    df = df[~df["TRAJECTORY"].isin(["Incomplete_Data"])]

    # Per-class summary statistics
    class_stats = df.groupby("TRAJECTORY").agg(
        n_tracts=("EAL_USD", "size"),
        median_EAL_USD=("EAL_USD", "median"),
        mean_EAL_USD=("EAL_USD", "mean"),
        sd_EAL_USD=("EAL_USD", "std"),
        min_EAL_USD=("EAL_USD", "min"),
        max_EAL_USD=("EAL_USD", "max"),
    ).reset_index().sort_values("median_EAL_USD", ascending=False)
    class_stats.to_csv(output_dir / "rq3_eal_by_trajectory_RQ3.csv", index=False)

    print("      Per-class EAL summary (sorted by median):")
    print(class_stats.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    # Run Kruskal-Wallis at each scope
    rows = []
    for label, sub in [("Combined", df),
                       ("Austin", df[df["MSA"] == "Austin-Round Rock-Georgetown MSA"]),
                       ("San Antonio", df[df["MSA"] == "San Antonio-New Braunfels MSA"])]:
        groups = [sub[sub["TRAJECTORY"] == c]["EAL_USD"].values
                  for c in sub["TRAJECTORY"].unique()
                  if (sub["TRAJECTORY"] == c).sum() >= 3]
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        rows.append({
            "scope": label,
            "n_tracts": len(sub),
            "n_groups": len(groups),
            "H_statistic": H,
            "df": len(groups) - 1,
            "p_value": p,
            "significant_005": p < 0.05,
        })
        print(f"      {label}: H={H:.3f}, p={p:.4e}, df={len(groups)-1}")

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "rq3_kruskal_wallis_RQ3.csv", index=False)
    return out, df


# ============================================================================
# STEP 5 — Dunn's post-hoc with BH-FDR correction
# ============================================================================
def dunn_posthoc(df, output_dir):
    """Pairwise comparisons between trajectory classes, BH-FDR adjusted."""
    print("[5/9] Dunn's post-hoc with Benjamini-Hochberg FDR correction...")

    # Try scikit-posthocs first (standard library); fall back to manual if absent
    try:
        import scikit_posthocs as sp
        dunn = sp.posthoc_dunn(df, val_col="EAL_USD",
                                group_col="TRAJECTORY", p_adjust="fdr_bh")
        # Reshape upper triangle to long
        long = (dunn.where(np.triu(np.ones(dunn.shape), k=1).astype(bool))
                    .stack().reset_index()
                    .rename(columns={"level_0": "class_a",
                                     "level_1": "class_b",
                                     0: "p_adj"}))
        long["significant_005"] = long["p_adj"] < 0.05
        long = long.sort_values("p_adj").reset_index(drop=True)
    except ImportError:
        print("      scikit-posthocs not installed; using manual Dunn implementation.")
        long = manual_dunn(df)

    long.to_csv(output_dir / "rq3_dunns_posthoc_RQ3.csv", index=False)
    print(f"      Pairwise comparisons: {len(long)}; "
          f"significant after BH-FDR: {long['significant_005'].sum()}")
    return long


def manual_dunn(df):
    """Manual Dunn's test with BH-FDR adjustment (no scikit-posthocs dependency)."""
    from scipy.stats import rankdata, norm
    from itertools import combinations

    classes = sorted(df["TRAJECTORY"].unique())
    df = df.copy()
    df["rank"] = rankdata(df["EAL_USD"])
    N = len(df)

    R = df.groupby("TRAJECTORY")["rank"].agg(["mean", "size"]).rename(
        columns={"mean": "Rbar", "size": "n"})

    # Tie correction
    _, tie_counts = np.unique(df["rank"], return_counts=True)
    tie_term = sum(t**3 - t for t in tie_counts) / (12 * (N - 1))

    rows = []
    for a, b in combinations(classes, 2):
        Ra, na = R.loc[a, "Rbar"], R.loc[a, "n"]
        Rb, nb = R.loc[b, "Rbar"], R.loc[b, "n"]
        se = np.sqrt(((N * (N + 1) / 12) - tie_term) * (1/na + 1/nb))
        z = (Ra - Rb) / se
        p = 2 * (1 - norm.cdf(abs(z)))
        rows.append({"class_a": a, "class_b": b, "z": z, "p_raw": p,
                     "Rbar_a": Ra, "Rbar_b": Rb, "n_a": na, "n_b": nb})

    out = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m = len(out)
    # Benjamini-Hochberg adjustment
    out["p_adj"] = (out["p_raw"] * m / (out.index + 1)).clip(upper=1)
    out["p_adj"] = out["p_adj"].iloc[::-1].cummin().iloc[::-1]
    out["significant_005"] = out["p_adj"] < 0.05
    return out


# ============================================================================
# STEP 6 — SVI quintile × EAL quintile chi-square
# ============================================================================
def quintile_overlap_chi_square(panel, output_dir):
    """5x5 contingency: SVI quintile (2024) × EAL quintile."""
    print("[6/9] Chi-square: SVI quintile × EAL quintile...")
    from scipy.stats import chi2_contingency

    df = panel.dropna(subset=["SVI_QUINT_2024", "EAL_QUINT"]).copy()

    rows = []
    for label, sub, slug in [
        ("Combined", df, "combined"),
        ("Austin", df[df["MSA"] == "Austin-Round Rock-Georgetown MSA"], "austin"),
        ("San Antonio", df[df["MSA"] == "San Antonio-New Braunfels MSA"], "sanantonio"),
    ]:
        if len(sub) < 25:
            continue
        ct = pd.crosstab(sub["SVI_QUINT_2024"].astype(int),
                         sub["EAL_QUINT"].astype(int),
                         margins=False)
        chi2, p, dof, _ = chi2_contingency(ct)
        n = ct.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if min(ct.shape) > 1 else np.nan

        rows.append({
            "scope": label, "n_tracts": n, "chi2": chi2, "df": dof,
            "p_value": p, "cramers_v": cramers_v,
            "significant_005": p < 0.05
        })

        # Save the cross-tab itself with descriptive index/column names
        ct.index.name = "SVI_quintile_2024"
        ct.columns.name = "EAL_quintile"
        ct.to_csv(output_dir / f"rq3_quintile_overlap_{slug}_RQ3.csv")

        print(f"      {label} (n={n}): χ²({dof})={chi2:.2f}, "
              f"p={p:.4e}, Cramer's V={cramers_v:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "rq3_chi_square_RQ3.csv", index=False)
    return out


# ============================================================================
# STEP 7 — Austin vs San Antonio comparison
# ============================================================================
def msa_comparison(panel, output_dir):
    """Direct Austin vs San Antonio comparison of EAL distribution.
    Tests both (a) all tracts and (b) Persistently_High tracts only."""
    print("[7/9] Mann-Whitney U: Austin vs San Antonio...")
    from scipy.stats import mannwhitneyu

    df = panel.dropna(subset=["EAL_USD", "TRAJECTORY"]).copy()

    rows = []

    # Comparison 1: All tracts
    austin = df[df["MSA"] == "Austin-Round Rock-Georgetown MSA"]["EAL_USD"]
    sa = df[df["MSA"] == "San Antonio-New Braunfels MSA"]["EAL_USD"]
    if len(austin) >= 3 and len(sa) >= 3:
        U, p = mannwhitneyu(austin, sa, alternative="two-sided")
        rows.append({
            "comparison": "All tracts: Austin vs San Antonio",
            "n_austin": len(austin),
            "n_san_antonio": len(sa),
            "median_EAL_austin": austin.median(),
            "median_EAL_san_antonio": sa.median(),
            "mann_whitney_U": U,
            "p_value": p,
            "significant_005": p < 0.05,
        })

    # Comparison 2: Persistently_High tracts only — the key RQ3 question
    ph = df[df["TRAJECTORY"] == "Persistently_High"]
    austin_ph = ph[ph["MSA"] == "Austin-Round Rock-Georgetown MSA"]["EAL_USD"]
    sa_ph = ph[ph["MSA"] == "San Antonio-New Braunfels MSA"]["EAL_USD"]
    if len(austin_ph) >= 3 and len(sa_ph) >= 3:
        U2, p2 = mannwhitneyu(austin_ph, sa_ph, alternative="two-sided")
        rows.append({
            "comparison": "Persistently_High: Austin vs San Antonio",
            "n_austin": len(austin_ph),
            "n_san_antonio": len(sa_ph),
            "median_EAL_austin": austin_ph.median(),
            "median_EAL_san_antonio": sa_ph.median(),
            "mann_whitney_U": U2,
            "p_value": p2,
            "significant_005": p2 < 0.05,
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "rq3_msa_comparison_RQ3.csv", index=False)
    print(out.to_string(index=False))
    return out


# ============================================================================
# STEP 8 — Export GeoDa input for bivariate Lee's L
# ============================================================================
def export_geoda_input(panel, output_dir):
    """Tabular file ready to join to a 2010-tract shapefile in GeoDa or ArcGIS Pro
    for bivariate Local Moran's I (Lee's L) analysis."""
    print("[8/9] Exporting GeoDa input...")
    keep_cols = ["GEOID10", "MSA", "COUNTY", "TRAJECTORY",
                 "SVI_2024", "EAL_USD", "EAL_USD_LOG", "EAL_QUINT",
                 "SVI_QUINT_2024", "SVI_Q5_PERSIST", "IS_PERSIST_HIGH"]
    keep_cols = [c for c in keep_cols if c in panel.columns]
    panel[keep_cols].to_csv(output_dir / "rq3_geoda_input_RQ3.csv", index=False)
    print(f"      Saved: rq3_geoda_input_RQ3.csv "
          f"({len(panel)} tracts, {len(keep_cols)} columns)")


# ============================================================================
# STEP 9 — Spatial outputs (GeoPackages for ArcGIS Pro)
# ============================================================================
def export_spatial_outputs(panel, geometry_source, spatial_dir):
    """Join the master panel to tract geometry and write two GeoPackages:
    (1) full panel for mapping, (2) Persistently_High subset for focused analysis.

    geometry_source : Path to a GeoPackage or shapefile containing
                      tract polygons keyed by GEOID10.
                      e.g., the dissertation's existing trajectory_map.gpkg.
    """
    print("[9/9] Building spatial outputs (GeoPackages for ArcGIS Pro)...")

    try:
        import geopandas as gpd
    except ImportError:
        print("      ⚠️  geopandas not installed. Spatial outputs skipped.")
        print("      Install with: pip install geopandas")
        print("      The CSV outputs are still usable — just join them to your")
        print("      existing trajectory_map.gpkg in ArcGIS Pro on GEOID10.")
        return

    if not Path(geometry_source).exists():
        print(f"      ⚠️  Geometry source not found at: {geometry_source}")
        print("      Spatial outputs skipped. Use the CSV with ArcGIS join-by-attribute.")
        return

    print(f"      Reading geometry from: {geometry_source}")
    geom = gpd.read_file(geometry_source)

    # Identify the GEOID column (varies by source)
    geoid_candidates = ["GEOID10", "GEOID", "GEOID20", "TRACTCE10", "FIPS"]
    geoid_col = None
    for c in geoid_candidates:
        if c in geom.columns:
            geoid_col = c
            break
    if geoid_col is None:
        print(f"      ⚠️  No recognizable GEOID column in geometry. "
              f"Found: {list(geom.columns)[:10]}")
        return
    print(f"      Geometry GEOID column: {geoid_col}")

    # Standardize geometry GEOID to 11-char string
    geom["GEOID10"] = geom[geoid_col].astype(str).str.zfill(11)

    # Keep only the geometry + GEOID; drop everything else from the geometry source
    # to avoid duplicate columns when joining
    geom_min = geom[["GEOID10", "geometry"]].copy()

    # Join the panel
    spatial_panel = geom_min.merge(panel, on="GEOID10", how="inner")
    print(f"      Joined panel + geometry: {len(spatial_panel)} tracts")

    # Output 1: Full panel
    out1 = spatial_dir / "rq3_master_panel_RQ3.gpkg"
    spatial_panel.to_file(out1, driver="GPKG")
    print(f"      Saved: {out1}")

    # Output 2: Persistently_High subset only
    ph_only = spatial_panel[spatial_panel["TRAJECTORY"] == "Persistently_High"].copy()
    if len(ph_only) > 0:
        out2 = spatial_dir / "rq3_persistent_high_eal_RQ3.gpkg"
        ph_only.to_file(out2, driver="GPKG")
        print(f"      Saved: {out2} ({len(ph_only)} Persistently_High tracts)")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 72)
    print("RQ3: Hazard-Vulnerability Co-location Analysis")
    print("=" * 72)

    # Make sure output dirs exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPATIAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    for label, p in [("NRI CSV", NRI_CSV),
                     ("Trajectory typology", TRAJECTORY_CSV),
                     ("Quintile assignments", QUINTILE_CSV)]:
        if not Path(p).exists():
            print(f"❌ {label} not found at: {p}")
            print("   Edit the CONFIG block at the top of this script and rerun.")
            sys.exit(1)

    # Pipeline
    msa_nri = load_and_filter_nri(NRI_CSV)
    eal_2010 = crosswalk_nri_to_2010(msa_nri, CROSSWALK_FILE)
    panel = build_master_panel(eal_2010, TRAJECTORY_CSV, QUINTILE_CSV)

    # Save the master panel
    panel.to_csv(OUTPUT_DIR / "rq3_master_panel_RQ3.csv", index=False)
    print("      Saved: rq3_master_panel_RQ3.csv")

    # Statistical tests
    _, df_for_dunn = kruskal_wallis_trajectory(panel, OUTPUT_DIR)
    dunn_posthoc(df_for_dunn, OUTPUT_DIR)
    quintile_overlap_chi_square(panel, OUTPUT_DIR)
    msa_comparison(panel, OUTPUT_DIR)

    # GeoDa input + spatial outputs
    export_geoda_input(panel, OUTPUT_DIR)
    export_spatial_outputs(panel, TRAJECTORY_GPKG, SPATIAL_OUTPUT_DIR)

    print("\n" + "=" * 72)
    print("RQ3 ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"\nTabular outputs in:  {OUTPUT_DIR}")
    print(f"Spatial outputs in:  {SPATIAL_OUTPUT_DIR}")
    print("\nFiles produced:")
    for f in sorted(OUTPUT_DIR.glob("*RQ3*")):
        print(f"  {f.name}")
    for f in sorted(SPATIAL_OUTPUT_DIR.glob("*RQ3*")):
        print(f"  spatial/{f.name}")


if __name__ == "__main__":
    main()

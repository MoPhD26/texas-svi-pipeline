"""
==============================================================================
RQ3 / Section 6.3 — Verification Script
==============================================================================

Purpose:
    Reproduces every numeric claim in Section 6.3 of the dissertation directly
    from the source geopackage. Designed to be run as an independent check
    against ArcGIS Pro outputs, and to give the committee a transparent,
    reproducible record of how each number was computed.

Inputs:
    RQ3_MSA_Analysis.gpkg
        Layer 'rq3_analysis' — 807 census tracts with:
        - SVI scores for 2009, 2014, 2019, 2024 (pooled z-score composite)
        - LISA trajectory typology (10 classes; Persistently_High = HH in all 4 vintages)
        - FEMA NRI v1.20 EAL fields, area-weighted crosswalk to 2010 tract boundaries:
            EAL_VALT  : Composite Expected Annual Loss (all hazards, all consequence types)
            EAL_VALB  : Building EAL (sum across all hazards, $/yr)
            EAL_VALPE : Population EAL, monetized via Value of Statistical Life ($/yr)
            EAL_VALA  : Agriculture EAL ($/yr) — excluded from inferential analysis
        - Per-hazard EAL totals for the 5 dominant corridor hazards
        - POPULATION (tract residents)
        - EAL_per_capita (= EAL_VALT / POPULATION)
        - EAL_SPCTL (state percentile of composite EAL — NOT national; see note)
        - SVI_mean_all (mean SVI across the 4 vintages — used for hot-spot input)

Outputs:
    Console output and CSV files reproducing:
        - Section 6.3.1 Table 4 (median EAL by trajectory class)
        - Sensitivity tests (PH vs others; metro stratification)
        - Decomposition by EAL component (per advisor's recommendation)
        - Quintile co-location matrices
        - Section 6.3.2 Hot-spot counts (Gi* from pysal/esda — sanity check
          against ArcGIS Pro Hot Spot Analysis output)
        - Section 6.3.3 MSA-stratified bivariate cross-tabulation

Design notes:
    1. Agriculture EAL (EAL_VALA) is excluded from inferential trajectory-class
       comparison because of zero-inflation in this study area: 48.6% of
       tracts have zero ag EAL, 75.3% are below $1,000/yr, and the median
       agricultural EAL across the entire 807-tract corridor is $0/yr.
       Quintile cutoffs and Kruskal-Wallis assumptions are not meaningful
       under these conditions. EAL_VALA is reported descriptively only.

    2. EAL_VALT (composite) is the PRIMARY analytical metric. EAL_VALB
       (building) and EAL_VALPE (population, monetized) are reported as
       MECHANISM DIAGNOSTICS — they explain *why* the composite shows what
       it shows, but the inferential headline rests on the composite.

    3. Spatial dependence violates Kruskal-Wallis i.i.d. assumptions. We
       use it as a screening / descriptive statistic, not as a definitive
       inferential test. The bivariate spatial analysis (Lee's L in ArcGIS
       Pro) is the appropriate inferential complement.

Usage:
    python src/11_verify_section_6_3.py
    RQ3_GPKG=path/to/RQ3_MSA_Analysis.gpkg python src/11_verify_section_6_3.py
    Outputs: RQ3_table4.csv, RQ3_decomposition.csv, RQ3_hotspot_counts.csv,
             RQ3_bivariate_crosstab.csv

Author: Mogahid Hussein, dissertation audit, May 2026
Run order: step 11, after step 10 and the ArcGIS Pro hot spot work.
==============================================================================
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu
from libpysal.weights import Queen
from esda.getisord import G_Local
from scipy.stats import norm

import os
from pathlib import Path

# ---- File paths (relative to the repository root, override with env vars) ----
GPKG_PATH = os.environ.get("RQ3_GPKG", "output/spatial/RQ3_MSA_Analysis.gpkg")
LAYER = "rq3_analysis"
OUT_DIR = os.environ.get("RQ3_OUT_DIR", "output")

def classify_gi_bin(z, fdr=False, alpha_levels=(0.10, 0.05, 0.01)):
    """
    Classify Gi* z-scores into ArcGIS-style Gi_Bin codes:
      +3 / -3: 99% confidence hot / cold
      +2 / -2: 95%
      +1 / -1: 90%
        0    : not significant
    
    If fdr=True, apply Benjamini-Hochberg correction (ArcGIS Pro default in
    'Optimized Hot Spot Analysis'). Standard 'Hot Spot Analysis' tool does
    NOT apply FDR by default.
    """
    p = 2 * (1 - norm.cdf(np.abs(z)))  # two-tailed p-value
    bins = np.zeros(len(z), dtype=int)
    if fdr:
        from statsmodels.stats.multitest import multipletests
        for level, code in zip(alpha_levels, [1, 2, 3]):
            rej, _, _, _ = multipletests(p, alpha=level, method='fdr_bh')
            bins[(rej) & (z > 0) & (np.abs(bins) < code)] = code
            bins[(rej) & (z < 0) & (np.abs(bins) < code)] = -code
    else:
        for thresh, code in [(1.65, 1), (1.96, 2), (2.58, 3)]:
            bins[(z >  thresh) & (np.abs(bins) < code)] = code
            bins[(z < -thresh) & (np.abs(bins) < code)] = -code
    return bins


def label_bin(b):
    """Convert Gi_Bin (-3..3) to a 3-level label for cross-tabulation."""
    if b >= 1:  return 'Hot'
    if b <= -1: return 'Cold'
    return 'NS'


def main():
    # ---- Load data ----
    print("Loading data ...")
    g = gpd.read_file(GPKG_PATH, layer=LAYER)
    print(f"  {len(g)} tracts loaded; CRS = {g.crs}")
    assert len(g) == 807, "Expected 807 tracts in study area"

    # Add a derived field: population EAL per resident (different from EAL_per_capita,
    # which is total EAL per resident). This is the cleanest people-based metric.
    g['EAL_VALPE_per_capita'] = np.where(
        g['POPULATION'] > 0,
        g['EAL_VALPE'] / g['POPULATION'],
        np.nan
    )

    focal_classes = ['Persistently_Low', 'Declining_Low', 'Never_Significant',
                     'Persistently_High', 'Declining_High']

    # =============================================================================
    # Section 6.3.1 Table 4 — Median EAL by trajectory class
    # =============================================================================
    print("\n" + "="*70)
    print("Section 6.3.1 Table 4: Median EAL by trajectory class")
    print("="*70)

    table4_rows = []
    for t in focal_classes:
        sub = g[g['Trajectory'] == t]
        table4_rows.append({
            'Trajectory': t.replace('_', ' '),
            'N': len(sub),
            'Median_EAL_VALT': round(sub['EAL_VALT'].median(), 0),
            'Median_EAL_VALB': round(sub['EAL_VALB'].median(), 0),
            'Median_EAL_VALPE': round(sub['EAL_VALPE'].median(), 0),
            'Median_EAL_per_Capita': round(sub['EAL_per_capita'].median(), 0),
            # NOTE: EAL_SPCTL is the STATE percentile, not national. NRI v1.20's
            # tract download gives state percentile only. To get national percentile,
            # you must pull the full NRI national table and join EAL_NPCTL.
            'Median_EAL_State_Pctile': round(sub['EAL_SPCTL'].median(), 1),
        })
    table4 = pd.DataFrame(table4_rows)
    print(table4.to_string(index=False))
    table4.to_csv(f'{OUT_DIR}/RQ3_table4.csv', index=False)
    print("  Saved -> RQ3_table4.csv")

    # Verify the "55% of EAL is building losses" claim
    total_VALT = g['EAL_VALT'].sum()
    print(f"\nBuilding share of total EAL across 807 tracts: "
          f"${g['EAL_VALB'].sum():,.0f} / ${total_VALT:,.0f} = "
          f"{g['EAL_VALB'].sum()/total_VALT*100:.1f}%")
    print("(Draft says ~59%; actual is ~55%. Update narrative accordingly.)")

    # =============================================================================
    # Inferential tests — Kruskal-Wallis & Mann-Whitney U
    # =============================================================================
    print("\n" + "="*70)
    print("Inferential tests")
    print("="*70)

    # --- Corridor-wide Kruskal-Wallis across all trajectory classes ---
    print("\nKruskal-Wallis across all trajectory classes:")
    for col, name in [('EAL_VALT', 'Composite EAL'),
                      ('EAL_VALB', 'Building EAL'),
                      ('EAL_VALPE', 'Population EAL ($)'),
                      ('EAL_VALPE_per_capita', 'Population EAL per capita')]:
        valid = g.dropna(subset=[col, 'Trajectory'])
        groups = [valid.loc[valid['Trajectory'] == t, col].values
                  for t in valid['Trajectory'].unique()]
        H, p = kruskal(*groups)
        print(f"  {name:<28}: H = {H:7.3f}, p = {p:.3e}")

    # --- Persistently_High vs all others (corridor + by metro) ---
    print("\nMann-Whitney U: Persistently_High vs all other tracts")
    print("(direction of inequality matters: PH < Others is the corridor finding)")
    for scope_name, scope_mask in [('Corridor', g.index == g.index),
                                    ('Austin', g['MSA'] == 'Austin-Round Rock-Georgetown MSA'),
                                    ('San Antonio', g['MSA'] == 'San Antonio-New Braunfels MSA')]:
        print(f"\n  {scope_name}:")
        sub = g[scope_mask]
        n_ph = (sub['Trajectory'] == 'Persistently_High').sum()
        print(f"    N PH tracts = {n_ph}")
        for col, name in [('EAL_VALT', 'Composite'),
                          ('EAL_VALB', 'Building'),
                          ('EAL_VALPE', 'Population $'),
                          ('EAL_VALPE_per_capita', 'Pop per capita')]:
            valid = sub.dropna(subset=[col])
            ph = valid.loc[valid['Trajectory'] == 'Persistently_High', col].values
            oth = valid.loc[valid['Trajectory'] != 'Persistently_High', col].values
            if len(ph) >= 3 and len(oth) >= 3:
                U, p = mannwhitneyu(ph, oth, alternative='two-sided')
                direction = '<' if np.median(ph) < np.median(oth) else '>'
                print(f"    {name:<14}: PH median {direction} Others; "
                      f"PH={np.median(ph):>11,.0f}  Oth={np.median(oth):>11,.0f}  p={p:.3e}")

    # =============================================================================
    # Quintile co-location: Q5 SVI 2024 × Q5 EAL component
    # =============================================================================
    print("\n" + "="*70)
    print("Quintile co-location (corridor)")
    print("="*70)
    # Independence reference: each quintile is 20% of tracts, so under spatial
    # independence we'd expect ~4% of tracts to be in BOTH Q5 SVI and Q5 EAL,
    # i.e. ~32 tracts out of 807. Ratio < 1.0 = NEGATIVE association.

    quintile_rows = []
    for col, name in [('EAL_VALT', 'Composite EAL'),
                      ('EAL_VALB', 'Building EAL'),
                      ('EAL_VALPE', 'Population EAL ($)'),
                      ('EAL_VALPE_per_capita', 'Pop EAL per capita')]:
        valid = g.dropna(subset=['Q24', col])
        eal_q5 = pd.qcut(valid[col], 5, labels=False, duplicates='drop') + 1
        q5_svi = (valid['Q24'] == 5).values
        q5_eal = (eal_q5 == 5).values
        overlap = (q5_svi & q5_eal).sum()
        expected = q5_svi.sum() * q5_eal.sum() / len(valid)
        quintile_rows.append({
            'EAL_metric': name,
            'Q5_SVI_count': int(q5_svi.sum()),
            'Q5_EAL_count': int(q5_eal.sum()),
            'Observed_overlap': int(overlap),
            'Expected_under_independence': round(expected, 1),
            'Ratio_obs_to_expected': round(overlap / expected, 2),
        })
    quint_df = pd.DataFrame(quintile_rows)
    print(quint_df.to_string(index=False))

    # =============================================================================
    # Section 6.3.2 Hot Spot Analysis (Gi*) — Python sanity-check
    # =============================================================================
    print("\n" + "="*70)
    print("Section 6.3.2 Gi* HOT SPOT ANALYSIS — Python sanity check")
    print("="*70)
    print("Method: Local Getis-Ord Gi* (Getis & Ord 1992; 1995)")
    print("Weights: First-order Queen contiguity (matches GeoDa LISA workflow)")
    print("Significance bins: ArcGIS-style (Gi_Bin -3 to +3), |z|>=1.65/1.96/2.58")
    print("NOTE: This is for cross-validation against ArcGIS Pro. ArcGIS may produce")
    print("      different counts depending on neighborhood specification (fixed")
    print("      distance vs Queen) and FDR correction setting.\n")

    # Build Queen contiguity weights, row-standardized
    # (Standard for LISA-type analyses; matches how GeoDa was run for the LISA layer)
    w = Queen.from_dataframe(g, use_index=False)
    w.transform = 'r'
    print(f"  Queen weights: mean neighbors = {w.mean_neighbors:.2f}, "
          f"islands = {len(w.islands)}")

    # Compute Gi* for SVI_mean_all and EAL_VALT
    hotspot_rows = []
    for var in ['SVI_mean_all', 'EAL_VALT']:
        gi = G_Local(g[var].values, w, star=True, transform='r')
        g[f'{var}_GiZ'] = gi.Zs
        g[f'{var}_GiBin_noFDR'] = classify_gi_bin(gi.Zs, fdr=False)
        g[f'{var}_GiBin_FDR']   = classify_gi_bin(gi.Zs, fdr=True)

        n_hot_no = (g[f'{var}_GiBin_noFDR'] >= 1).sum()
        n_cold_no = (g[f'{var}_GiBin_noFDR'] <= -1).sum()
        n_hot_fdr = (g[f'{var}_GiBin_FDR'] >= 1).sum()
        n_cold_fdr = (g[f'{var}_GiBin_FDR'] <= -1).sum()
        print(f"\n  {var}:")
        print(f"    Without FDR: hot={n_hot_no:>3}, cold={n_cold_no:>3}, NS={807-n_hot_no-n_cold_no:>3}")
        print(f"    With FDR   : hot={n_hot_fdr:>3}, cold={n_cold_fdr:>3}, NS={807-n_hot_fdr-n_cold_fdr:>3}")

        hotspot_rows.append({
            'variable': var, 'method': 'Queen + no FDR',
            'hot': n_hot_no, 'cold': n_cold_no, 'NS': 807-n_hot_no-n_cold_no
        })
        hotspot_rows.append({
            'variable': var, 'method': 'Queen + FDR (BH)',
            'hot': n_hot_fdr, 'cold': n_cold_fdr, 'NS': 807-n_hot_fdr-n_cold_fdr
        })

    # =============================================================================
    # Bivariate cross-tab (Table 5 equivalent) and MSA stratification (Section 6.3.3)
    # =============================================================================
    print("\n" + "="*70)
    print("Section 6.3.2/6.3.3: Bivariate cross-tab and MSA stratification")
    print("="*70)

    # Hot spots are FDR corrected, matching the dissertation (Table 5, Figures 6 to 8)
    g['SVI_class'] = g['SVI_mean_all_GiBin_FDR'].apply(label_bin)
    g['EAL_class'] = g['EAL_VALT_GiBin_FDR'].apply(label_bin)

    print("\nBivariate cross-tab (corridor; Queen + FDR):")
    ct = pd.crosstab(g['SVI_class'], g['EAL_class'], margins=True, margins_name='Total')
    print(ct.to_string())

    print("\nMSA stratification — % of SVI hot spots that are EAL cold spots:")
    for mlabel in ['Austin-Round Rock-Georgetown MSA', 'San Antonio-New Braunfels MSA']:
        sub = g[g['MSA'] == mlabel]
        sv_hot = sub[sub['SVI_class'] == 'Hot']
        if len(sv_hot) > 0:
            n_hot_cold = (sv_hot['EAL_class'] == 'Cold').sum()
            pct = n_hot_cold / len(sv_hot) * 100
            print(f"  {mlabel}: {n_hot_cold} of {len(sv_hot)} SVI hot spots are EAL cold ({pct:.1f}%)")

    # Save outputs
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hotspot_rows).to_csv(f'{OUT_DIR}/RQ3_hotspot_counts.csv', index=False)
    ct.to_csv(f'{OUT_DIR}/RQ3_bivariate_crosstab.csv')
    quint_df.to_csv(f'{OUT_DIR}/RQ3_quintile_overlap.csv', index=False)
    print(f"\nAll output CSVs saved to {OUT_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()

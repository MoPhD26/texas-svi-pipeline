"""
09_build_transition_matrix.py
─────────────────────────────────────────────────────────────────
Assigns every tract-year to a pooled SVI quintile and measures how tracts
moved between quintiles over time.

Quintile breaks are computed once from all tract-years pooled together
(Tate 2012, Spielman and Folch 2015), so Q5 means the same level of
vulnerability in 2009 and in 2024.

Transition matrices are built for three scopes (corridor, Austin MSA,
San Antonio MSA) and four periods (2009-2014, 2014-2019, 2019-2024, and
2009-2024). Each row shows where tracts that started in a quintile ended up:
the diagonal is persistence, cells to the left are improvement, cells to the
right are worsening.

Inputs:  output/trajectory_typology.csv     (step 08)
         output/spatial/trajectory_map.gpkg (step 08, for geometry)
Outputs: output/quintile_assignments.csv    tract-level quintiles and mobility type
         output/transition_summary.csv      persistence, improvement, worsening rates
         output/transition_matrices.xlsx    12 matrices (3 scopes × 4 periods)
         output/spatial/quintile_map.gpkg and quintile_time_enabled.gpkg

Run order: step 09, after 08_build_trajectory_typology.py.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR  = Path("output")
SPATIAL_DIR = OUTPUT_DIR / "spatial"
YEARS       = [2009, 2014, 2019, 2024]
PERIODS     = [(2009,2014),(2014,2019),(2019,2024),(2009,2024)]
PERIOD_LABELS = {
    (2009,2014): "2009→2014 (Post-Recession)",
    (2014,2019): "2014→2019 (Recovery/Expansion)",
    (2019,2024): "2019→2024 (Post-Pandemic)",
    (2009,2024): "2009→2024 (Full 15-Year Period)",
}
QUINTILE_LABELS = {1:"Q1 Least Vulnerable",2:"Q2 Low-Moderate",
                   3:"Q3 Moderate",4:"Q4 High-Moderate",5:"Q5 Most Vulnerable"}
QUINTILE_COLORS = {1:"#1A9641",2:"#A6D96A",3:"#FFFFBF",4:"#FDAE61",5:"#D7191C"}
MOBILITY_COLORS = {"Persistent_Q5":"#D7191C","Persistent_Q1":"#1A9641",
                   "Persistent_Mid":"#CCCCCC","Improved":"#74C476",
                   "Worsened":"#F46D43","Mixed":"#9ECAE1","No_Data":"#FFFFFF"}
scope_map = {
    "combined":    ("Combined Study Area", None),
    "austin":      ("Austin-Round Rock-Georgetown MSA","Austin"),
    "san_antonio": ("San Antonio-New Braunfels MSA","San Antonio"),
}

def assign_pooled_quintiles(svi_long):
    """Quintiles from the pooled distribution of all tract-years (1 = least vulnerable)."""
    out = svi_long.copy()
    out["Quintile"] = pd.qcut(out["SVI"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    return out


def mobility_type(row):
    q09, q24 = row.get("Q_2009"), row.get("Q_2024")
    if pd.isna(q09) or pd.isna(q24): return "No_Data"
    q09, q24 = int(q09), int(q24)
    if q09 == q24:
        if q09 == 5: return "Persistent_Q5"
        if q09 == 1: return "Persistent_Q1"
        return "Persistent_Mid"
    changes = []
    for y1,y2 in [(2009,2014),(2014,2019),(2019,2024)]:
        v1,v2 = row.get(f"Q_{y1}"), row.get(f"Q_{y2}")
        if pd.notna(v1) and pd.notna(v2): changes.append(int(v2)-int(v1))
    improved = sum(1 for c in changes if c < 0)
    worsened = sum(1 for c in changes if c > 0)
    if improved > 0 and worsened > 0: return "Mixed"
    if improved > 0: return "Improved"
    if worsened > 0: return "Worsened"
    return "Persistent_Mid"


def build_matrix(df, y1, y2):
    sub = df[[f"Q_{y1}",f"Q_{y2}"]].dropna()
    counts = pd.crosstab(sub[f"Q_{y1}"], sub[f"Q_{y2}"])
    for q in [1,2,3,4,5]:
        if q not in counts.index:   counts.loc[q] = 0
        if q not in counts.columns: counts[q]     = 0
    counts = counts.sort_index()[sorted(counts.columns)]
    pct    = counts.div(counts.sum(axis=1),axis=0).multiply(100).round(1)
    return counts, pct

def compute_metrics(counts_df, y1, y2, scope_label):
    total = counts_df.values.sum()
    m = {"period":f"{y1}→{y2}","scope":scope_label,"n_tracts":total}
    diag = 0
    for q in [1,2,3,4,5]:
        rt = int(counts_df.loc[q].sum()) if q in counts_df.index else 0
        m[f"Q{q}_n"] = rt
        if rt == 0:
            m[f"Q{q}_persist_pct"] = np.nan
            m[f"Q{q}_improve_pct"] = np.nan
            m[f"Q{q}_worsen_pct"]  = np.nan
            continue
        stay    = counts_df.loc[q,q] if q in counts_df.columns else 0
        improve = sum(counts_df.loc[q,c] for c in [1,2,3,4,5] if c<q and c in counts_df.columns)
        worsen  = sum(counts_df.loc[q,c] for c in [1,2,3,4,5] if c>q and c in counts_df.columns)
        m[f"Q{q}_persist_pct"] = stay/rt*100
        m[f"Q{q}_improve_pct"] = improve/rt*100
        m[f"Q{q}_worsen_pct"]  = worsen/rt*100
        diag += stay
    m["overall_persist_pct"] = diag/total*100 if total>0 else np.nan
    return m


def main():
    print("="*65)
    print(f"Transition Matrix Construction v2  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*65)

    # ── STEP 1: Load and assign pooled quintiles ──────────────────────
    print("\nSTEP 1 — Loading data and assigning pooled quintiles")
    traj = pd.read_csv(OUTPUT_DIR/"trajectory_typology.csv", dtype={"GEOID10":str})
    print(f"  Loaded: {len(traj)} tracts")

    svi_long = []
    for yr in YEARS:
        tmp = traj[["GEOID10","MSA",f"SVI_{yr}"]].copy()
        tmp.columns = ["GEOID10","MSA","SVI"]
        tmp["YEAR"] = yr
        svi_long.append(tmp)
    svi_long = pd.concat(svi_long, ignore_index=True).dropna(subset=["SVI"])

    svi_long = assign_pooled_quintiles(svi_long)

    print("\n  Pooled quintile breakpoints (SVI_composite):")
    for q in [1,2,3,4,5]:
        sub = svi_long[svi_long["Quintile"]==q]["SVI"]
        print(f"    Q{q}: {sub.min():>7.3f} to {sub.max():>7.3f}  n={len(sub)}")

    quintile_wide = svi_long.pivot_table(
        index=["GEOID10","MSA"], columns="YEAR", values="Quintile"
    ).reset_index()
    quintile_wide.columns.name = None
    quintile_wide.columns = ["GEOID10","MSA"] + [f"Q_{y}" for y in YEARS]
    quintile_wide = quintile_wide.merge(
        traj[["GEOID10","Trajectory","LISA_Sequence"]], on="GEOID10", how="left"
    )

    # Derived fields
    quintile_wide["Q_CHANGE"] = (
        pd.to_numeric(quintile_wide["Q_2024"], errors='coerce') -
        pd.to_numeric(quintile_wide["Q_2009"], errors='coerce')
    )

    quintile_wide["MOBILITY_TYPE"]  = quintile_wide.apply(mobility_type, axis=1)
    quintile_wide["MOBILITY_COLOR"] = quintile_wide["MOBILITY_TYPE"].map(MOBILITY_COLORS)
    quintile_wide["Q5_ESCAPE"]      = ((quintile_wide["Q_2009"]==5)&(quintile_wide["Q_2024"]!=5)).astype(int)
    quintile_wide["Q5_PERSIST"]     = ((quintile_wide["Q_2009"]==5)&(quintile_wide["Q_2024"]==5)).astype(int)
    for yr in YEARS:
        quintile_wide[f"Q_{yr}_COLOR"] = pd.to_numeric(
            quintile_wide[f"Q_{yr}"],errors='coerce').map(QUINTILE_COLORS)

    quintile_wide.to_csv(OUTPUT_DIR/"quintile_assignments.csv", index=False)
    print(f"  ✓ quintile_assignments.csv  ({len(quintile_wide)} tracts)")

    # ── STEP 2: Build matrices ────────────────────────────────────────
    print("\nSTEP 2 — Building transition matrices")

    all_matrices, all_metrics = {}, []
    for scope,(scope_label,msa_filter) in scope_map.items():
        df_s = quintile_wide[quintile_wide["MSA"].str.contains(msa_filter,na=False)] \
               if msa_filter else quintile_wide
        for (y1,y2) in PERIODS:
            counts,pct = build_matrix(df_s,y1,y2)
            all_matrices[(scope,y1,y2)] = (counts,pct)
            m = compute_metrics(counts,y1,y2,scope_label)
            all_metrics.append(m)
            print(f"\n  {scope_label} | {y1}→{y2}")
            print(f"    Overall persistence: {m['overall_persist_pct']:.1f}%  "
                  f"Q5 persist: {m.get('Q5_persist_pct',float('nan')):.1f}%  "
                  f"Q5 improve: {m.get('Q5_improve_pct',float('nan')):.1f}%")

    # ── STEP 3: Print primary matrix ──────────────────────────────────
    print("\nSTEP 3 — Primary matrix: 2009→2024, Combined")
    c_counts,c_pct = all_matrices[("combined",2009,2024)]
    print("\n  Row % | Q1=least vulnerable  Q5=most vulnerable\n")
    print(f"  {'From':<14} " + "".join(f"{'Q'+str(q):>9}" for q in [1,2,3,4,5]) + "  RowN")
    print(f"  {'-'*63}")
    for qf in [1,2,3,4,5]:
        rn   = int(c_counts.loc[qf].sum()) if qf in c_counts.index else 0
        vals = [c_pct.loc[qf,qt] if (qf in c_pct.index and qt in c_pct.columns) else 0.0
                for qt in [1,2,3,4,5]]
        lbl  = QUINTILE_LABELS[qf][:13]
        print(f"  {lbl:<14} " + "".join(f"{v:>8.1f}%" for v in vals) + f"  {rn:>5}")

    # ── STEP 4: Spatial outputs ───────────────────────────────────────
    print("\nSTEP 4 — Creating spatial files for ArcGIS Pro")
    try:
        import geopandas as gpd
        gdf_base = gpd.read_file(SPATIAL_DIR/"trajectory_map.gpkg")
        if "GEOID10" not in gdf_base.columns:
            geoid_col = next(c for c in gdf_base.columns if "GEOID" in c.upper())
            gdf_base  = gdf_base.rename(columns={geoid_col:"GEOID10"})
        print(f"  Base geometry: {len(gdf_base)} polygons  CRS={gdf_base.crs}")

        # Static quintile map
        q_cols = (["GEOID10"] +
                  [f"Q_{y}" for y in YEARS] +
                  [f"Q_{y}_COLOR" for y in YEARS] +
                  ["Q_CHANGE","MOBILITY_TYPE","MOBILITY_COLOR",
                   "Q5_ESCAPE","Q5_PERSIST","Trajectory","LISA_Sequence"])
        q_cols = [c for c in q_cols if c in quintile_wide.columns]
        gdf_q  = gdf_base.merge(quintile_wide[q_cols], on="GEOID10", how="left")
        out_static = SPATIAL_DIR/"quintile_map.gpkg"
        gdf_q.to_file(out_static, driver="GPKG")
        print(f"  ✓ quintile_map.gpkg  ({len(gdf_q)} features, {len(gdf_q.columns)} fields)")
        print("    Fields: Q_2009 Q_2014 Q_2019 Q_2024 | Q_CHANGE | MOBILITY_TYPE")
        print("    Q5_ESCAPE=1 → was most vulnerable in 2009, improved by 2024")
        print("    Q5_PERSIST=1 → was most vulnerable in 2009, still is in 2024")

        # Time-enabled file
        gdfs_time = []
        for yr in YEARS:
            yr_data = quintile_wide[["GEOID10","MSA","Trajectory","MOBILITY_TYPE",
                                      "Q5_ESCAPE","Q5_PERSIST","Q_CHANGE",
                                      f"Q_{yr}",f"Q_{yr}_COLOR"]].copy()
            yr_data = yr_data.rename(columns={f"Q_{yr}":"QUINTILE",
                                               f"Q_{yr}_COLOR":"Q_COLOR"})
            yr_data["YEAR"] = yr
            yr_data["DATE"] = pd.Timestamp(f"{yr}-01-01")
            if f"SVI_{yr}" in traj.columns:
                yr_data["SVI"] = traj.set_index("GEOID10")[f"SVI_{yr}"].reindex(
                    yr_data["GEOID10"].values).values
            gdf_yr = gdf_base[["GEOID10","geometry"]].merge(
                yr_data, on="GEOID10", how="inner")
            gdfs_time.append(gdf_yr)

        gdf_all = gpd.GeoDataFrame(
            pd.concat(gdfs_time, ignore_index=True),
            geometry="geometry", crs=gdf_base.crs
        ).sort_values(["GEOID10","YEAR"]).reset_index(drop=True)
        out_time = SPATIAL_DIR/"quintile_time_enabled.gpkg"
        gdf_all.to_file(out_time, driver="GPKG")
        print(f"  ✓ quintile_time_enabled.gpkg  ({len(gdf_all)} features = 807 × 4 years)")
        print("    Time Field: DATE  |  Symbolize by: QUINTILE (1-5)")

    except Exception as e:
        print(f"  ⚠ Spatial error: {e}")
        import traceback
        traceback.print_exc()

    # ── STEP 5: Save tabular outputs ──────────────────────────────────
    print("\nSTEP 5 — Saving tabular outputs")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR/"transition_summary.csv", index=False)
    print("  ✓ transition_summary.csv")

    try:
        import openpyxl  # noqa: F401
        scope_short  = {"combined":"Combined","austin":"Austin","san_antonio":"SanAntonio"}
        period_short = {(2009,2014):"2009_2014",(2014,2019):"2014_2019",
                        (2019,2024):"2019_2024",(2009,2024):"2009_2024"}
        with pd.ExcelWriter(OUTPUT_DIR/"transition_matrices.xlsx", engine="openpyxl") as writer:
            pd.DataFrame([
                ["Transition Matrix Workbook — Spatiotemporal SVI"],
                ["Q1=least vulnerable (bottom 20%)  Q5=most vulnerable (top 20%)"],
                ["Pooled quintiles across all 4 vintages (Tate 2012; Spielman & Folch 2015)"],
                ["Row % shown — diagonal = persistence, off-diagonal = mobility"],
            ]).to_excel(writer, sheet_name="README", index=False, header=False)
            metrics_df.to_excel(writer, sheet_name="Summary_Metrics", index=False)
            for scope in scope_short:
                for (y1,y2) in PERIODS:
                    counts,pct = all_matrices[(scope,y1,y2)]
                    sheet = f"{scope_short[scope]}_{period_short[(y1,y2)]}"
                    sl    = scope_map[scope][0]
                    pd.DataFrame([[f"{sl} | {PERIOD_LABELS[(y1,y2)]} | Row %"]]).to_excel(
                        writer,sheet_name=sheet,startrow=0,index=False,header=False)
                    pct_out = pct.copy()
                    pct_out.index   = [f"From Q{i}" for i in pct_out.index]
                    pct_out.columns = [f"To Q{i}"   for i in pct_out.columns]
                    pct_out.to_excel(writer,sheet_name=sheet,startrow=2)
                    pd.DataFrame([[f"{sl} | {PERIOD_LABELS[(y1,y2)]} | Raw Counts"]]).to_excel(
                        writer,sheet_name=sheet,startrow=10,index=False,header=False)
                    c_out = counts.copy()
                    c_out.index   = [f"From Q{i}" for i in c_out.index]
                    c_out.columns = [f"To Q{i}"   for i in c_out.columns]
                    c_out.to_excel(writer,sheet_name=sheet,startrow=12)
        print("  ✓ transition_matrices.xlsx")
    except ImportError:
        print("  ⚠ openpyxl missing — pip install openpyxl")

    print(f"""
    {'='*65}
    ALL OUTPUTS SAVED
    {'='*65}

    TABULAR:
      output/transition_matrices.xlsx      12 matrices (3 scopes × 4 periods)
      output/transition_summary.csv        Key metrics
      output/quintile_assignments.csv      Tract-level quintile data

    SPATIAL:
      output/spatial/quintile_map.gpkg
        Recommended maps:
        1. Symbolize Q_2009 — baseline vulnerability quintiles
        2. Symbolize Q_2024 — current vulnerability quintiles
        3. Symbolize MOBILITY_TYPE — who improved/worsened/persisted
        4. Definition Query Q5_ESCAPE=1 — highlight escape tracts
        5. Definition Query Q5_PERSIST=1 — highlight lock-in tracts

      output/spatial/quintile_time_enabled.gpkg
        Layer Properties → Time → Time Field: DATE
        Symbolize QUINTILE (1-5) with red→green diverging ramp
        Time slider: 2009 → 2014 → 2019 → 2024
    {'='*65}
    """)
    print("Next step: python src/10_hazard_congruence.py")


if __name__ == "__main__":
    main()

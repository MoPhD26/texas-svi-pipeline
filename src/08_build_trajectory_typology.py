"""
08_build_trajectory_typology.py
─────────────────────────────────────────────────────────────────
Builds the spatiotemporal vulnerability trajectory typology from four
vintages of Anselin Local Moran's I (LISA) results.

Each tract's cluster type (HH, LL, HL, LH, or NS) in 2009, 2014, 2019,
and 2024 is concatenated into a sequence such as HH-HH-HH-HH, and the
sequence is assigned to one of ten trajectory classes. The rules are
applied in this order, identically for every tract:

  1. Incomplete Data     any period missing (island tract dropped by LISA)
  2. Never Significant   NS in all four periods
  3. Persistently High   HH in 2009, in 2024, and in at least 3 of 4 periods
  4. Declining High      HH in 2009 but not in 2024
  5. Emerging High       HH in 2019 and 2024 but not in 2009
  6. Persistently Low    as 3, with LL
  7. Declining Low       as 4, with LL
  8. Emerging Low        as 5, with LL
  9. Recurrent Outlier   HL or LH in at least two periods
 10. Transient           every other sequence with some significance

Inputs (from ArcGIS Pro LISA exports, see step 07):
  output/LISA_results_{YEAR}.csv
  output/spatial/svi_{YEAR}_labeled.gpkg  (to recover GEOID10)
  output/svi_scores_all_years.csv         (SVI scores)

Outputs:
  output/trajectory_typology.csv         Full tract-level results
  output/trajectory_summary.csv          Counts by MSA
  output/spatial/trajectory_map.gpkg     Spatial file for ArcGIS Pro

Run order: step 08, after 07_verify_lisa_exports.py.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("output")
SPATIAL_DIR = OUTPUT_DIR / "spatial"
YEARS = [2009, 2014, 2019, 2024]

TRAJ_ORDER = [
    "Persistently_High", "Declining_High", "Emerging_High",
    "Persistently_Low", "Declining_Low", "Emerging_Low",
    "Recurrent_Outlier", "Transient", "Never_Significant", "Incomplete_Data",
]

TRAJ_DESCRIPTIONS = {
    "Persistently_High": "HH in 2009 and 2024, and in 3 or 4 periods",
    "Declining_High":    "HH in 2009, not in 2024",
    "Emerging_High":     "HH in 2019 and 2024, not in 2009",
    "Persistently_Low":  "LL in 2009 and 2024, and in 3 or 4 periods",
    "Declining_Low":     "LL in 2009, not in 2024",
    "Emerging_Low":      "LL in 2019 and 2024, not in 2009",
    "Recurrent_Outlier": "HL or LH in 2 or more periods",
    "Transient":         "Some significance, no sustained pattern",
    "Never_Significant": "NS in all 4 periods",
    "Incomplete_Data":   "Missing at least one period (island tract)",
}

# Colors for ArcGIS Pro symbology (diverging, colorblind-safe)
TRAJ_COLORS = {
    "Persistently_High": "#D7191C",
    "Declining_High":    "#FDAE61",
    "Emerging_High":     "#F46D43",
    "Persistently_Low":  "#1A9641",
    "Declining_Low":     "#A8DDB5",
    "Emerging_Low":      "#74C476",
    "Recurrent_Outlier": "#762A83",
    "Transient":         "#E0E0E0",
    "Never_Significant": "#CCCCCC",
    "Incomplete_Data":   "#FFFFFF",
}


def _cluster_rule(types, cluster, name):
    """Persistent, declining, or emerging rule for one cluster type (HH or LL)."""
    n = types.count(cluster)
    if types[0] == cluster and types[3] == cluster and n >= 3:
        return f"Persistently_{name}"
    if types[0] == cluster and types[3] != cluster:
        return f"Declining_{name}"
    if types[2] == cluster and types[3] == cluster and types[0] != cluster:
        return f"Emerging_{name}"
    return None


def classify_sequence(types):
    """Assign one of the ten trajectory classes to a four-period cluster sequence."""
    types = list(types)
    if len(types) != 4:
        raise ValueError("A trajectory needs exactly four periods")
    if "NO_DATA" in types:
        return "Incomplete_Data"
    if all(t == "NS" for t in types):
        return "Never_Significant"
    for cluster, name in (("HH", "High"), ("LL", "Low")):
        label = _cluster_rule(types, cluster, name)
        if label:
            return label
    if sum(t in ("HL", "LH") for t in types) >= 2:
        return "Recurrent_Outlier"
    return "Transient"


def read_gpkg_table(path, columns):
    """Read selected columns from the first table in a GeoPackage without GDAL."""
    con = sqlite3.connect(str(path))
    try:
        tname = pd.read_sql("SELECT table_name FROM gpkg_contents", con)["table_name"].iloc[0]
        return pd.read_sql(f"SELECT {', '.join(columns)} FROM '{tname}'", con)
    finally:
        con.close()


def load_lisa(year):
    """Load one LISA export and recover GEOID10 through SOURCE_ID = fid."""
    lisa_path = OUTPUT_DIR / f"LISA_results_{year}.csv"
    if not lisa_path.exists():
        print(f"  {year}: ⚠ File not found — {lisa_path}")
        return None

    ref = read_gpkg_table(SPATIAL_DIR / f"svi_{year}_labeled.gpkg", ["fid", "GEOID10"])
    lisa = pd.read_csv(lisa_path)

    src_col = next((c for c in lisa.columns if c.upper() in ("SOURCE_ID", "OBJECTID")), None)
    cotype_col = next((c for c in lisa.columns if "COTYPE" in c.upper()), None)
    if src_col is None or cotype_col is None:
        print(f"  {year}: ⚠ SOURCE_ID or COType column missing. Columns: {list(lisa.columns)}")
        return None

    lisa["_fid"] = pd.to_numeric(lisa[src_col], errors="coerce")
    merged = lisa.merge(ref, left_on="_fid", right_on="fid", how="left")
    merged["cotype"] = merged[cotype_col].fillna("").astype(str).str.strip().replace("", "NS")

    clean = merged[["GEOID10", "cotype"]].rename(columns={"cotype": f"COType_{year}"})
    counts = clean[f"COType_{year}"].value_counts()
    missing = clean["GEOID10"].isna().sum()
    print(f"  {year}: {len(clean)} rows | HH={counts.get('HH', 0)} LL={counts.get('LL', 0)} "
          f"NS={counts.get('NS', 0)} | Missing GEOID10={missing} {'✓' if missing == 0 else '⚠'}")
    return clean


def main():
    print("=" * 65)
    print("Trajectory Typology Construction")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    # ── STEP 1: LISA results with GEOID10 ────────────────────────────────────
    print("\nSTEP 1 — Recovering GEOID10 from LISA exports")
    lisa = {yr: load_lisa(yr) for yr in YEARS}
    if any(v is None for v in lisa.values()):
        print("\n⚠ Not all four LISA files could be read. Check file names and rerun.")
        sys.exit(1)

    # ── STEP 2: Master tract table ───────────────────────────────────────────
    print("\nSTEP 2 — Building master tract table")
    master = read_gpkg_table(SPATIAL_DIR / "svi_2009_labeled.gpkg", ["GEOID10", "COUNTY_NAME", "MSA"])
    print(f"  Master list: {len(master)} tracts")
    for year in YEARS:
        master = master.merge(lisa[year], on="GEOID10", how="left")
        master[f"COType_{year}"] = master[f"COType_{year}"].fillna("NO_DATA")
        nd = (master[f"COType_{year}"] == "NO_DATA").sum()
        print(f"    {year}: {nd} tracts with NO_DATA (geometric islands)")

    svi = pd.read_csv(OUTPUT_DIR / "svi_scores_all_years.csv", dtype={"GEOID10": str})
    svi_wide = svi.pivot_table(index="GEOID10", columns="YEAR5", values="SVI_composite").reset_index()
    svi_wide.columns = ["GEOID10"] + [f"SVI_{y}" for y in svi_wide.columns[1:]]
    master = master.merge(svi_wide, on="GEOID10", how="left")

    # ── STEP 3: Trajectory labels ────────────────────────────────────────────
    print("\nSTEP 3 — Assigning trajectory labels")
    seq_cols = [f"COType_{y}" for y in YEARS]
    master["Trajectory"] = master[seq_cols].apply(lambda r: classify_sequence(r.tolist()), axis=1)
    master["LISA_Sequence"] = master[seq_cols].agg("-".join, axis=1)
    master["Traj_Color"] = master["Trajectory"].map(TRAJ_COLORS)

    counts = master["Trajectory"].value_counts()
    print(f"\n  {'Trajectory':<20} {'N':>5} {'%':>7}  Rule")
    print(f"  {'-' * 75}")
    for traj in TRAJ_ORDER:
        n = counts.get(traj, 0)
        print(f"  {traj:<20} {n:>5} {n / len(master) * 100:>6.1f}%  {TRAJ_DESCRIPTIONS[traj]}")

    print("\n  By MSA:")
    print(pd.crosstab(master["Trajectory"], master["MSA"]).reindex(TRAJ_ORDER).fillna(0).astype(int).to_string())

    # ── STEP 4: Mean SVI by trajectory ───────────────────────────────────────
    print("\nSTEP 4 — Mean SVI by trajectory")
    print(f"  {'Trajectory':<20} {'SVI_2009':>9} {'SVI_2024':>9} {'Change':>8}")
    for traj in TRAJ_ORDER:
        sub = master[master["Trajectory"] == traj]
        if len(sub) < 2:
            continue
        s09, s24 = sub["SVI_2009"].mean(), sub["SVI_2024"].mean()
        if pd.notna(s09) and pd.notna(s24):
            print(f"  {traj:<20} {s09:>9.2f} {s24:>9.2f} {s24 - s09:>+8.2f}")

    # ── STEP 5: Save outputs ─────────────────────────────────────────────────
    print("\nSTEP 5 — Saving outputs")
    master.to_csv(OUTPUT_DIR / "trajectory_typology.csv", index=False)
    print(f"  ✓ trajectory_typology.csv  ({len(master)} rows)")

    summary = (master.groupby(["MSA", "Trajectory"]).size().rename("N_tracts").reset_index())
    summary["Pct_of_MSA"] = (summary["N_tracts"] /
                             summary.groupby("MSA")["N_tracts"].transform("sum") * 100).round(2)
    summary["Color"] = summary["Trajectory"].map(TRAJ_COLORS)
    summary.to_csv(OUTPUT_DIR / "trajectory_summary.csv", index=False)
    print("  ✓ trajectory_summary.csv")

    try:
        import geopandas as gpd

        gdf_base = gpd.read_file(SPATIAL_DIR / "msa_tracts_2019.gpkg")
        geoid_col = next(c for c in gdf_base.columns if c in ("GEOID10", "GEOID"))
        gdf_geo = gdf_base[[geoid_col, "geometry"]].rename(columns={geoid_col: "GEOID10"})
        spatial_cols = (["GEOID10", "COUNTY_NAME", "MSA", "Trajectory", "LISA_Sequence", "Traj_Color"]
                        + seq_cols + [f"SVI_{y}" for y in YEARS])
        gdf = gdf_geo.merge(master[[c for c in spatial_cols if c in master.columns]], on="GEOID10", how="left")
        gdf.to_file(SPATIAL_DIR / "trajectory_map.gpkg", driver="GPKG")
        print(f"  ✓ trajectory_map.gpkg  ({len(gdf)} features)")
        print("    In ArcGIS Pro, symbolize by Trajectory (Unique Values). Traj_Color holds the hex colors.")
    except Exception as e:
        print(f"  ⚠ Spatial file skipped: {e}")

    print("\nNext step: python src/09_build_transition_matrix.py")


if __name__ == "__main__":
    main()

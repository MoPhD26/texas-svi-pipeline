"""
05b_export_panel_geopackages.py
─────────────────────────────────────────────────────────────────
Creates spatial files from the longitudinal panel dataset for
use in ArcGIS Pro.

Produces two spatial outputs:

1. Per-vintage GeoPackages (one per year, 2010 boundaries)
   For standard vintage-by-vintage mapping and analysis.
   Files: spatial/acs5_svi_2009_2010bdry.gpkg
          spatial/acs5_svi_2014_2010bdry.gpkg
          spatial/acs5_svi_2019_2010bdry.gpkg
          spatial/acs5_svi_2024_2010bdry.gpkg

2. Time-enabled combined GeoPackage (all vintages stacked)
   For ArcGIS Pro time slider and spatiotemporal analysis.
   Each feature has a DATE field (Jan 1 of the vintage year)
   enabling the ArcGIS Pro time slider to work correctly.
   File: spatial/acs5_svi_all_years_time_enabled.gpkg

Requirements: pip install geopandas pandas numpy

Run order: step 05b, optional, after 05_stack_longitudinal.py.
These are raw panel layers. Step 06 produces the scored, labeled layers.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

INPUT_DIR   = Path("output")
SPATIAL_DIR = INPUT_DIR / "spatial"
YEARS       = [2009, 2014, 2019, 2024]

# ── Boundary GeoPackage for each vintage (all use 2010 boundaries) ────────────
# 2014 and 2019 are native 2010 boundaries — use either one as the base
# 2009 and 2024 are crosswalked TO 2010 — use the same base geometry
BOUNDARY_GPKG = SPATIAL_DIR / "msa_tracts_2019.gpkg"   # 2010 boundaries

# ── Columns to exclude from spatial files ────────────────────────────────────
# These are pipeline metadata columns not needed for spatial analysis
EXCLUDE_COLS = [
    "NAME", "state", "county", "tract",
    "B19301_cpi_vintage_year",
]

print("=" * 65)
print("Create Longitudinal Spatial Files for ArcGIS Pro")
print("=" * 65)

# ── Load longitudinal panel ───────────────────────────────────────────────────
print("\nLoading longitudinal panel...")
long_df = pd.read_csv(
    INPUT_DIR / "acs_tract_2010std_all_years_long.csv",
    dtype={"GEOID10": str, "county_fips": str, "tract_code": str}
)
print(f"  {len(long_df)} rows × {len(long_df.columns)} columns")

# ── Load 2010 boundary geometry ───────────────────────────────────────────────
print("\nLoading 2010 boundary geometry...")
gdf_base  = gpd.read_file(BOUNDARY_GPKG)
geoid_col = next((c for c in gdf_base.columns if c in ("GEOID10","GEOID")), None)
if geoid_col is None:
    raise RuntimeError("Cannot detect GEOID column in boundary file")

# Keep only geometry and GEOID — all attributes come from the panel
gdf_geo = gdf_base[[geoid_col, "geometry"]].copy()
gdf_geo = gdf_geo.rename(columns={geoid_col: "GEOID10"})
print(f"  {len(gdf_geo)} boundary polygons loaded")
print(f"  CRS: {gdf_geo.crs}")

# ── OUTPUT 1: Per-vintage GeoPackages ────────────────────────────────────────
print(f"\n{'='*65}")
print("Output 1 — Per-Vintage GeoPackages (2010 boundaries)")
print(f"{'='*65}")

for year in YEARS:
    yr_data = long_df[long_df["YEAR5"] == year].copy()
    drop    = [c for c in EXCLUDE_COLS if c in yr_data.columns]
    yr_data = yr_data.drop(columns=drop)

    # Join attributes onto geometry
    gdf_yr  = gdf_geo.merge(yr_data, on="GEOID10", how="left")

    # Count joined tracts
    joined  = gdf_yr["YEAR5"].notna().sum()
    missing = gdf_yr["YEAR5"].isna().sum()

    out_path = SPATIAL_DIR / f"acs5_svi_{year}_2010bdry.gpkg"
    gdf_yr.to_file(out_path, driver="GPKG")
    print(f"\n  {year}:")
    print(f"    Features:  {len(gdf_yr)}")
    print(f"    Joined:    {joined} {'✓' if joined==807 else '⚠'}")
    print(f"    Unjoined:  {missing} {'✓' if missing==0 else '⚠ tracts without data'}")
    print(f"    CRS:       {gdf_yr.crs}")
    print(f"    ✓ Saved → {out_path}")

# ── OUTPUT 2: Time-enabled combined GeoPackage ────────────────────────────────
print(f"\n{'='*65}")
print("Output 2 — Time-Enabled Combined GeoPackage (ArcGIS Pro)")
print(f"{'='*65}")
print("""
  This file stacks all four vintages with a DATE field set to
  January 1 of each vintage year. In ArcGIS Pro:
  1. Add the layer
  2. Right-click → Properties → Time tab
  3. Set Layer Time: Each feature has a single time field
  4. Time Field: DATE
  5. Enable time on the map and use the time slider
""")

# Build combined spatial dataset
gdfs_time = []
for year in YEARS:
    yr_data = long_df[long_df["YEAR5"] == year].copy()
    drop    = [c for c in EXCLUDE_COLS if c in yr_data.columns]
    yr_data = yr_data.drop(columns=drop)

    gdf_yr = gdf_geo.merge(yr_data, on="GEOID10", how="inner")

    # Add DATE field — ArcGIS Pro time slider uses this
    # Set to January 1 of the vintage year
    gdf_yr["DATE"] = pd.Timestamp(f"{year}-01-01")

    gdfs_time.append(gdf_yr)
    print(f"  {year}: {len(gdf_yr)} features with DATE={year}-01-01")

gdf_time_all = pd.concat(gdfs_time, ignore_index=True)
gdf_time_all = gpd.GeoDataFrame(gdf_time_all, geometry="geometry", crs=gdf_geo.crs)

# Sort by GEOID10 then DATE for clean layer structure
gdf_time_all = gdf_time_all.sort_values(["GEOID10","YEAR5"]).reset_index(drop=True)

out_time = SPATIAL_DIR / "acs5_svi_all_years_time_enabled.gpkg"
gdf_time_all.to_file(out_time, driver="GPKG")

print("\n  Combined time-enabled file:")
print(f"    Total features:  {len(gdf_time_all)} ({807} tracts × 4 years)")
print(f"    Columns:         {len(gdf_time_all.columns)}")
print(f"    CRS:             {gdf_time_all.crs}")
print(f"    DATE range:      {gdf_time_all['DATE'].min()} → {gdf_time_all['DATE'].max()}")
print(f"    ✓ Saved → {out_time}")

# ── ArcGIS Pro setup instructions ────────────────────────────────────────────
print(f"\n{'='*65}")
print("HOW TO USE IN ARCGIS PRO")
print(f"{'='*65}")
print("""
  OPTION A — Per-Vintage Files (recommended for most analysis):
  ─────────────────────────────────────────────────────────────
  1. Add each acs5_svi_{YEAR}_2010bdry.gpkg as a layer
  2. Symbolize by any variable (e.g. B17001_002E / B17001_001E
     for poverty rate)
  3. Use Definition Query to filter by msa if needed

  OPTION B — Time-Enabled File (for time slider animation):
  ──────────────────────────────────────────────────────────
  1. Add acs5_svi_all_years_time_enabled.gpkg to your map
  2. In the Contents pane, right-click the layer → Properties
  3. Click the Time tab
  4. Layer Time: Each feature has a single time field
  5. Time Field: DATE
  6. Click OK
  7. On the Map tab → Time group → Time Slider button
  8. Use the slider to move through 2009, 2014, 2019, 2024

  CRS NOTE:
  ─────────────────────────────────────────────────────────────
  The files are in the CRS of the 2019 TIGER/Line shapefile
  (typically NAD83 geographic, EPSG:4269). If your ArcGIS Pro
  project uses a projected CRS (e.g. State Plane Texas),
  ArcGIS Pro will reproject on the fly — or you can project
  the files in advance using Project tool (Data Management).

  RECOMMENDED PROJECTED CRS for Texas analysis:
  EPSG:3083 — NAD83 / Texas Centric Albers Equal Area
  This preserves area relationships for vulnerability mapping.
""")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"{'='*65}")
print("Done. Spatial output files:")
print(f"{'='*65}")
for year in YEARS:
    p = SPATIAL_DIR / f"acs5_svi_{year}_2010bdry.gpkg"
    print(f"  {p}")
print(f"  {out_time}")
print("\nNext step: python src/06_build_svi_index.py")
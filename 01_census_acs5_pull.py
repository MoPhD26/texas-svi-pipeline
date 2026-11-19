#!/usr/bin/env python3
"""
census_acs5_pull.py
────────────────────────────────────────────────────────────────────────────
Pull ACS5 census-tract-level data + TIGER/Line tract shapefiles for the
San Antonio–New Braunfels MSA and Austin–Round Rock–Georgetown MSA (Texas).

Geography
---------
All tracts within the MSA-defining counties are pulled.
GEOID10 is the standard 11-digit tract identifier:
    state(2) + county(3) + tract(6)   e.g. "48029000100"

Year note
---------
YEAR=2009 → ACS5 2005–2009  (first-ever ACS5 release; published Dec 2010)
YEAR=2010 → ACS5 2006–2010
Set YEAR below to the desired vintage.  Default = 2009.

Requirements
------------
    pip install requests pandas geopandas
"""

import io
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from datetime import date

import geopandas as gpd
import pandas as pd
import requests

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  –  edit these values
# ══════════════════════════════════════════════════════════════════════════════

# Census API key: get a free key at https://api.census.gov/data/key_signup.html
CENSUS_API_KEY: str = os.environ.get("CENSUS_API_KEY", "")  # set via environment variable, never commit a key

# ACS5 vintage year sent to the Census API
#   2009 → 2005–2009 estimates
#   2014 → 2010–2014 estimates
#   2019 → 2015–2019 estimates
#   2024 → 2020–2024 estimates
YEAR: int = int(os.environ.get("ACS_YEAR", "2024"))  # 2009, 2014, 2019, or 2024
 
# Texas state FIPS
STATE_FIPS: str = "48"
 
# MSA definitions: {msa_label: {county_fips: county_name}}
# Tracts within ALL listed counties are pulled for each MSA.
MSAS: Dict[str, Dict[str, str]] = {
    "San Antonio-New Braunfels MSA": {
        "013": "Atascosa County",
        "019": "Bandera County",
        "029": "Bexar County",
        "091": "Comal County",
        "187": "Guadalupe County",
        "259": "Kendall County",
        "325": "Medina County",
        "493": "Wilson County",
    },
    "Austin-Round Rock-Georgetown MSA": {
        "021": "Bastrop County",
        "055": "Caldwell County",
        "209": "Hays County",
        "453": "Travis County",
        "491": "Williamson County",
    },
}
 
# Output directory (created if absent)
OUTPUT_DIR = Path("output")
 
# ══════════════════════════════════════════════════════════════════════════════
# VARIABLE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
 
def _ev(prefix: str, nums) -> List[str]:
    """
    Expand a Census variable range into E/M pairs.
    e.g. _ev("B01001", range(3, 7))
         → ["B01001_003E","B01001_003M","B01001_004E","B01001_004M",...]
    """
    return [f"{prefix}_{n:03d}{s}" for n in nums for s in ("E", "M")]
 
 
# ── ACS5 Detailed Tables (endpoint: /acs/acs5) ────────────────────────────
 
DETAIL_GROUPS: Dict[str, List[str]] = {
 
    # Poverty (persons below poverty level) ─────────────────────────────────
    # B17001_001 = total pop for whom poverty status determined (denom)
    # B17001_002 = below poverty level (numer)
    "poverty": [
        "B17001_001E", "B17001_001M",
        "B17001_002E", "B17001_002M",
    ],
 
    # Per capita income ───────────────────────────────────────────────────────
    "per_capita_income": [
        "B19301_001E", "B19301_001M",
    ],
 
    # No HS diploma (population 25+) ─────────────────────────────────────────
    # B15002_001  = total pop 25+  (denom)
    # B15002_003–010 = male, less than 9th grade → some HS, no diploma
    # B15002_020–027 = female equivalents
    "no_hs_diploma": (
        ["B15002_001E", "B15002_001M"]
        + _ev("B15002", range(3,  11))   # male: 003–010
        + _ev("B15002", range(20, 28))   # female: 020–027
    ),
 
    # Age 65+ ─────────────────────────────────────────────────────────────────
    # B01001_001  = total pop (denom; also used for Under 18)
    # B01001_020–025 = male 65–67 … 85+
    # B01001_044–049 = female equivalents
    "age_65plus": (
        ["B01001_001E", "B01001_001M"]
        + _ev("B01001", range(20, 26))   # male 65+:   020–025
        + _ev("B01001", range(44, 50))   # female 65+: 044–049
    ),
 
    # Under 18 ────────────────────────────────────────────────────────────────
    # Denom (B01001_001E/M) shared with age_65plus; deduped in flat list below
    # B01001_003–006 = male <5 … 15–17
    # B01001_027–030 = female equivalents
    "under_18": (
        ["B01001_001E", "B01001_001M"]
        + _ev("B01001", range(3,  7))    # male <18:   003–006
        + _ev("B01001", range(27, 31))   # female <18: 027–030
    ),
 
    # Single-parent households (with children <18) ───────────────────────────
    # B11005_001 = total households w/ own children <18 (denom)
    # B11005_006 = male householder, no spouse present
    # B11005_007 = female householder, no spouse present
    "single_parent": [
        "B11005_001E", "B11005_001M",
        "B11005_006E", "B11005_006M",
        "B11005_007E", "B11005_007M",
    ],
 
    # Minority (Total − Non-Hispanic White alone) ────────────────────────────
    # B03002_001 = total pop (denom)
    # B03002_003 = non-Hispanic White alone (subtract to get minority numer)
    "minority": [
        "B03002_001E", "B03002_001M",
        "B03002_003E", "B03002_003M",
    ],
 
    # Multi-unit housing (10+ units) & mobile homes ──────────────────────────
    # B25024_001 = total housing units (denom; shared between multi-unit & mobile)
    # B25024_007 = 10–19 units
    # B25024_008 = 20–49 units
    # B25024_009 = 50+ units
    # B25024_010 = mobile homes
    "housing_type": [
        "B25024_001E", "B25024_001M",
        "B25024_007E", "B25024_007M",
        "B25024_008E", "B25024_008M",
        "B25024_009E", "B25024_009M",
        "B25024_010E", "B25024_010M",
    ],
 
    # Crowding (>1.0 occupants per room) ─────────────────────────────────────
    # B25014_001 = total occupied units (denom)
    # B25014_005–007 = owner-occ: 1.01–1.50, 1.51–2.00, 2.01+
    # B25014_011–013 = renter-occ: same bands
    "crowding": [
        "B25014_001E", "B25014_001M",
        "B25014_005E", "B25014_005M",
        "B25014_006E", "B25014_006M",
        "B25014_007E", "B25014_007M",
        "B25014_011E", "B25014_011M",
        "B25014_012E", "B25014_012M",
        "B25014_013E", "B25014_013M",
    ],
 
    # No vehicle available ────────────────────────────────────────────────────
    # B25044_001 = total occupied units (denom)
    # B25044_003 = owner-occupied, no vehicle
    # B25044_010 = renter-occupied, no vehicle
    "no_vehicle": [
        "B25044_001E", "B25044_001M",
        "B25044_003E", "B25044_003M",
        "B25044_010E", "B25044_010M",
    ],

    # Unemployment ────────────────────────────────────────────────────────────
    # B23025 (simple 7-row table) available 2014+ only.
    # B23025_003E = civilian labor force (denominator)
    # B23025_005E = unemployed (numerator)
    # Rate = B23025_005E / B23025_003E * 100
    #
    # NOTE: B23025 does not exist in the 2009 ACS5 vintage (HTTP 404).
    # For 2009, unemployment is pulled as DP03_0009PE from the profile
    # endpoint at the county level only — not available at tract level.
    # Unemployment is therefore treated as tract-level missing for 2009
    # and excluded from the 2009 SVI calculation. Documented as a known
    # limitation of the first ACS5 release.
    "unemployment": (
        ["B23025_003E", "B23025_003M",
         "B23025_005E", "B23025_005M"]
        if YEAR >= 2014 else []
    ),
}
 
# Flatten + deduplicate while preserving insertion order
_seen: set = set()
ALL_DETAIL_VARS: List[str] = []
for _group_vars in DETAIL_GROUPS.values():
    for _v in _group_vars:
        if _v not in _seen:
            ALL_DETAIL_VARS.append(_v)
            _seen.add(_v)
 
 
# ── ACS5 Data Profile Tables (endpoint: /acs/acs5/profile) ───────────────────
# PE = percent estimate, PM = percent margin of error
# DP02 limited English variable shifts row position across ACS vintages.
# Verified against Census API variable manifests for each vintage year.

_DP02_CODE: Dict[int, str] = {
    2009: "DP02_0113",
    2014: "DP02_0113",
    2019: "DP02_0114",
    2024: "DP02_0115",
}
_dp02 = _DP02_CODE[YEAR]

PROFILE_VARS: List[str] = [

    f"{_dp02}PE",  f"{_dp02}PM",        # limited English (%) — vintage-specific
]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# API HELPERS
# ══════════════════════════════════════════════════════════════════════════════
 
def _batched(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
 
 
def fetch_acs5_tracts(
    variables: List[str],
    year: int,
    state: str,
    counties: List[str],
    api_key: str,
    profile: bool = False,
    batch_size: int = 49,
) -> pd.DataFrame:
    """
    Fetch ACS5 variables for all census tracts within the specified counties.
 
    Uses the Census API geography syntax:
        for=tract:*  in=state:{state} county:{county1,county2,...}
 
    Automatically batches requests to respect the Census API's 50-variable
    limit, then merges the batch results into a single DataFrame.
 
    Parameters
    ----------
    variables   : list of Census variable codes (e.g. ["B17001_001E", ...])
    year        : ACS5 vintage year (e.g. 2009)
    state       : 2-digit state FIPS (e.g. "48" for Texas)
    counties    : list of 3-digit county FIPS strings to include
    api_key     : your Census API key
    profile     : True → use /acs/acs5/profile endpoint (DP tables)
    batch_size  : max variables per request (Census limit = 50; use 49 for safety)
 
    Returns
    -------
    pd.DataFrame with one row per tract and all variables as columns.
    Geo columns: NAME, state, county, tract.
    Numeric variable columns are cast to float.
    """
    endpoint = f"https://api.census.gov/data/{year}/acs/acs5"
    if profile:
        endpoint += "/profile"
 
    county_str = ",".join(counties)
    batch_dfs: List[pd.DataFrame] = []
 
    for i, batch in enumerate(_batched(variables, batch_size), start=1):
        params = {
            "get": "NAME," + ",".join(batch),
            "for": "tract:*",
            "in":  f"state:{state} county:{county_str}",
            "key": api_key,
        }
        print(f"  → request {i}: {len(batch)} variables … ", end="", flush=True)
        resp = requests.get(endpoint, params=params, timeout=30)
 
        if resp.status_code != 200:
            raise RuntimeError(
                f"Census API error {resp.status_code}: {resp.text[:300]}"
            )
 
        rows = resp.json()
        df = pd.DataFrame(rows[1:], columns=rows[0])
        batch_dfs.append(df)
        print(f"{len(df)} tracts returned")
        time.sleep(0.35)   # polite rate-limiting
 
    # ── Merge batches on geographic identity columns ──────────────────────
    geo_cols = ["NAME", "state", "county", "tract"]
    result = batch_dfs[0]
    for df in batch_dfs[1:]:
        merge_on = [c for c in geo_cols if c in result.columns and c in df.columns]
        # Only keep new variable columns from right side (avoid duplicating NAME etc.)
        right_keep = merge_on + [c for c in df.columns if c not in result.columns]
        result = result.merge(df[right_keep], on=merge_on, how="outer")
 
    # ── Cast variable columns to numeric ─────────────────────────────────
    for col in result.columns:
        if col not in geo_cols:
            result[col] = pd.to_numeric(result[col], errors="coerce")
 
    return result
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TIGER/LINE TRACT SHAPEFILE DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
 
def _download_zip(url: str, dest_dir: Path, timeout: int = 300,
                  max_retries: int = 3) -> None:
    """Download a zip file from url and extract into dest_dir.
    Retries up to max_retries times on connection errors."""
    import time
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Download attempt {attempt}/{max_retries}...")
            resp = requests.get(url, timeout=timeout, stream=True)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"TIGER download error {resp.status_code}: {resp.text[:300]}\n"
                    f"URL tried: {url}"
                )
            # Read content in chunks to avoid timeout on large files
            content = b""
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                content += chunk
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                z.extractall(dest_dir)
            return  # success — exit function
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            if attempt < max_retries:
                wait = attempt * 5
                print(f"    Connection error — retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"TIGER download failed after {max_retries} attempts.\n"
                    f"URL: {url}\nError: {e}"
                )
 
 
def download_tiger_tracts(
    year: int,
    state_fips: str,
    out_dir: Path,
) -> gpd.GeoDataFrame:
    """
    Download TIGER/Line census-tract shapefiles and return a GeoDataFrame.

    URL format by vintage
    ─────────────────────
    2009    Census 2000 boundary tracts from TIGER2010 companion file.
            …/TIGER2010/TRACT/2000/tl_2010_{state_fips}_tract00.zip

    2010    Single state file in year-named subdirectory; tract10 suffix.
            …/TIGER2010/TRACT/2010/tl_2010_{state_fips}_tract10.zip

    2011+   Single state file; no year suffix.
            …/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip
    """
    tiger_dir = out_dir / f"tiger_tracts_{year}"
    tiger_dir.mkdir(parents=True, exist_ok=True)

    # ── Build the correct URL by vintage ─────────────────────────────────
    if year <= 2009:
        # The 2005-2009 ACS uses Census 2000 tract definitions. TIGER2009
        # does not host a usable tract file. The TIGER2010 release published
        # the 2000 boundaries as a companion file (tract00) alongside tract10.
        url = (
            "https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2000/"
            f"tl_2010_{state_fips}_tract00.zip"
        )
        print(f"  → {url}  [TIGER2010 Census-2000-boundary tracts]")
    elif year == 2010:
        url = (
            f"https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/"
            f"tl_2010_{state_fips}_tract10.zip"
        )
        print(f"  → {url}")
    else:
        url = (
            f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
            f"tl_{year}_{state_fips}_tract.zip"
        )
        print(f"  → {url}")

    # ── Download, extract, and load ───────────────────────────────────────
    # This block now runs for ALL vintages — no duplication needed
    _download_zip(url, tiger_dir)

    shp_files = list(tiger_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in {tiger_dir}")

    gdf = gpd.read_file(shp_files[0])
    print(f"  Loaded: {shp_files[0].name}  ({len(gdf)} tract features)")
    return gdf
 
 
def _detect_geoid_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """
    Identify the 11-digit tract GEOID column from a TIGER/Line GeoDataFrame.
 
    Column name varies by vintage:
      2009  → CTIDFP00  (Census Tract ID FIPS, 2000 definitions)
      2010  → GEOID10
      2011+ → GEOID
    """
    for candidate in ("GEOID10", "GEOID", "CTIDFP00", "GEOID00"):
        if candidate in gdf.columns:
            return candidate
    # broader fallback
    candidates = [c for c in gdf.columns if c.upper().startswith("GEOID")]
    return candidates[0] if candidates else None
 
 
def _detect_county_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """
    Identify the county FIPS column in a TIGER/Line tract GeoDataFrame.
 
    Column name varies by vintage:
      2009  → COUNTYFP00
      2010  → COUNTYFP10
      2011+ → COUNTYFP
    """
    for candidate in ("COUNTYFP", "COUNTYFP10", "COUNTYFP00"):
        if candidate in gdf.columns:
            return candidate
    candidates = [c for c in gdf.columns if "COUNTYFP" in c.upper()]
    return candidates[0] if candidates else None
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
 
def main() -> None:
    # ── Setup ────────────────────────────────────────────────────────────────
    if not CENSUS_API_KEY or CENSUS_API_KEY.strip() in ("", "YOUR_API_KEY", "YOUR_API_KEY_HERE"):
        raise ValueError(
            "Set CENSUS_API_KEY in the script or as an environment variable."
        )
 
    spatial_dir = OUTPUT_DIR / "spatial"
    spatial_dir.mkdir(parents=True, exist_ok=True)
 
    all_counties: List[str] = [
        fips for msa in MSAS.values() for fips in msa
    ]
    # {county_fips: msa_label}
    county_to_msa: Dict[str, str] = {
        fips: msa_name
        for msa_name, counties in MSAS.items()
        for fips in counties
    }
    # {county_fips: county_name}
    county_to_name: Dict[str, str] = {
        fips: name
        for counties in MSAS.values()
        for fips, name in counties.items()
    }
 
    # ── 1. ACS5 Detailed Tables ───────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"Step 1 – ACS5 Detailed Tables  (year={YEAR}, geography=tract)")
    print(f"  {len(ALL_DETAIL_VARS)} unique variables across {len(DETAIL_GROUPS)} topic groups")
    print(f"  {len(all_counties)} counties  ·  all tracts within  ·  batch size = 49")
    print("═"*60)
    df_detail = fetch_acs5_tracts(
        variables=ALL_DETAIL_VARS,
        year=YEAR,
        state=STATE_FIPS,
        counties=all_counties,
        api_key=CENSUS_API_KEY,
        profile=False,
    )
    print(f"  ✓ Detailed table shape: {df_detail.shape}")
 
    # ── 2. ACS5 Data Profile Tables ───────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"Step 2 – ACS5 Data Profile Tables  (year={YEAR}, geography=tract)")
    print(f"  {len(PROFILE_VARS)} variables  ·  unemployment rate + limited English")
    print("═"*60)
    df_profile = fetch_acs5_tracts(
        variables=PROFILE_VARS,
        year=YEAR,
        state=STATE_FIPS,
        counties=all_counties,
        api_key=CENSUS_API_KEY,
        profile=True,
    )
    print(f"  ✓ Profile table shape: {df_profile.shape}")
 
    # ── 3. Merge detailed + profile ───────────────────────────────────────
    df = df_detail.merge(
        df_profile.drop(columns=["NAME"], errors="ignore"),
        on=["state", "county", "tract"],
        how="left",
    )
 
    # Build 11-digit GEOID10: state(2) + county(3) + tract(6)
    df.insert(0, "YEAR5",       YEAR)
    df.insert(1, "GEOID10",     df["state"].str.zfill(2)
                                + df["county"].str.zfill(3)
                                + df["tract"].str.zfill(6))
    df.insert(2, "msa",         df["county"].map(county_to_msa))
    df.insert(3, "county_fips", df["county"])
    df.insert(4, "county_name", df["county"].map(county_to_name))
    df.insert(5, "tract_code",  df["tract"])
 
    # Sort for readability
    df = df.sort_values(["msa", "county_fips", "tract_code"]).reset_index(drop=True)

    df["data_retrieved_date"] = str(date.today())  # documents pull date
    out_csv = OUTPUT_DIR / f"acs5_{YEAR}_msa_tracts.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  ✓ Saved tabular data → {out_csv}")
    print(f"     {len(df)} tracts  ×  {len(df.columns)} columns")
 
    # ── 4. TIGER/Line Tract Shapefile ─────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"Step 3 – TIGER/Line tract shapefile  (year={YEAR}, state={STATE_FIPS})")
    print("═"*60)
    gdf_state = download_tiger_tracts(YEAR, STATE_FIPS, spatial_dir)
 
    # Filter to MSA counties only
    county_col = _detect_county_col(gdf_state)
    geoid_col  = _detect_geoid_col(gdf_state)
 
    if county_col:
        gdf_msa = gdf_state[gdf_state[county_col].isin(all_counties)].copy()
        gdf_msa["msa"]         = gdf_msa[county_col].map(county_to_msa)
        gdf_msa["county_name"] = gdf_msa[county_col].map(county_to_name)
        print(f"  Filtered to {len(gdf_msa)} MSA tracts  "
              f"(county col: {county_col!r}, geoid col: {geoid_col!r})")
    else:
        print("  Warning: could not detect county FIPS column – keeping full state file")
        gdf_msa = gdf_state.copy()
 
    out_gpkg = spatial_dir / f"msa_tracts_{YEAR}.gpkg"
    gdf_msa.to_file(out_gpkg, driver="GPKG")
    print(f"  ✓ Saved tract shapefile → {out_gpkg}")
 
    # ── 5. Spatial join: merge ACS attributes onto geometries ─────────────
    if geoid_col:
        acs_cols = (
            ["GEOID10", "YEAR5", "msa", "county_fips", "county_name", "tract_code"]
            + [c for c in df.columns if c not in (
                "state", "county", "tract", "NAME",
                "GEOID10", "YEAR5", "msa", "county_fips", "county_name", "tract_code",
            )]
        )
        gdf_joined = gdf_msa.merge(
            df[acs_cols],
            left_on=geoid_col,
            right_on="GEOID10",
            how="left",
        )
        out_joined = spatial_dir / f"msa_tracts_acs5_{YEAR}.gpkg"
        gdf_joined.to_file(out_joined, driver="GPKG")
        print(f"  ✓ Saved joined spatial+ACS → {out_joined}")
    else:
        print("  Warning: could not detect GEOID column – spatial join skipped")
        out_joined = None
 
    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("Done!  Output files:")
    print(f"  {out_csv}")
    print(f"  {out_gpkg}")
    if out_joined:
        print(f"  {out_joined}")
    print("═"*60)
 
 
if __name__ == "__main__":
    main()
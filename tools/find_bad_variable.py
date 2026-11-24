"""
Identifies which variable in batch 2 is causing the Census API 500 error
for the 2009 vintage by testing each variable individually.

Written while debugging the 2009 pull: a batched request failed with HTTP 500,
so this script tests suspect variables one at a time against a single county.

Usage: CENSUS_API_KEY=your_key python tools/find_bad_variable.py
"""
import os

import requests
import time

key = os.environ.get("CENSUS_API_KEY", "")
if not key:
    raise SystemExit("Set the CENSUS_API_KEY environment variable first.")
year = 2009
state = "48"
county = "029"  # Bexar County — small test geography

# These are the variables in batch 2 (positions 49-97 of ALL_DETAIL_VARS)
# Paste your full ALL_DETAIL_VARS list here after running the script once
# For now we test the most likely suspects — B15002 female education cells
# and B01001 age cells which are in this range

suspects = [
    # B15002 female education (020-027)
    "B15002_020E", "B15002_021E", "B15002_022E", "B15002_023E",
    "B15002_024E", "B15002_025E", "B15002_026E", "B15002_027E",
    # B01001 age/sex cells
    "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E",
    "B01001_024E", "B01001_025E", "B01001_027E", "B01001_028E",
    "B01001_029E", "B01001_030E", "B01001_044E", "B01001_045E",
    "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E",
    # B11005 single parent
    "B11005_001E", "B11005_006E", "B11005_007E",
    # B03002 race/ethnicity
    "B03002_001E", "B03002_003E",
    # B25024 housing type
    "B25024_001E", "B25024_007E", "B25024_008E", "B25024_009E", "B25024_010E",
]

print(f"Testing {len(suspects)} variables individually against 2009 API...\n")

bad_vars = []
for var in suspects:
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": f"NAME,{var}",
        "for": "tract:*",
        "in": f"state:{state} county:{county}",
        "key": key,
    }
    r = requests.get(url, params=params, timeout=15)
    status = "✓ OK   " if r.status_code == 200 else f"✗ {r.status_code}"
    print(f"  {status}  {var}")
    if r.status_code != 200:
        bad_vars.append((var, r.status_code))
    time.sleep(0.2)

print(f"\n{'='*50}")
if bad_vars:
    print("PROBLEM VARIABLES FOUND:")
    for var, code in bad_vars:
        print(f"  {var}: HTTP {code}")
else:
    print("All tested variables returned OK.")
    print("Problem may be in the batch combination — check variable order.")
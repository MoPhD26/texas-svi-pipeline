"""
Checks which unemployment variables exist in each ACS 5-year vintage.

This diagnostic found that B23025 (employment status) is not available at
the tract level in the 2005-2009 ACS, which led to unemployment being
excluded from the index and tested separately in a sensitivity analysis.

Usage: CENSUS_API_KEY=your_key python tools/check_unemployment_vars.py
"""
import os

import requests

key = os.environ.get("CENSUS_API_KEY", "")
if not key:
    raise SystemExit("Set the CENSUS_API_KEY environment variable first.")
years = [2009, 2014, 2019, 2024]

# Candidate unemployment variables to test across vintages
candidates = [
    "B23025_003E",  # civilian labor force — later vintages
    "B23025_005E",  # unemployed — later vintages
    "B23001_003E",  # sex by age by employment status — older vintages
    "C23002_006E",  # collapsed employment status
]

for year in years:
    print(f"\n{'='*50}")
    print(f"YEAR: {year}")
    print(f"{'='*50}")
    for var in candidates:
        url = f"https://api.census.gov/data/{year}/acs/acs5/variables/{var}.json"
        try:
            r = requests.get(url, params={"key": key}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                label = data.get("label", "no label")
                print(f"  FOUND     {var}: {label}")
            else:
                print(f"  NOT FOUND {var}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  ERROR     {var}: {e}")
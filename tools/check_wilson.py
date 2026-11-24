"""
Lists Wilson County (FIPS 48493) rows in the Census 2000 to 2010 tract
relationship file. This diagnostic revealed that the relationship file uses
98xx tract codes while the ACS API returns 00xx codes, which silently dropped
every Wilson County tract from the first 2009 crosswalk. Fixed in src/04a.
"""
import requests, io, pandas as pd

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
rel = pd.read_csv(io.StringIO(resp.text), sep=",", header=None,
                  names=TRF_COLS, dtype=str)

wilson = rel[rel["GEOID00"].str.startswith("48493")]
print(f"Wilson County rows in tx48trf.txt: {len(wilson)}")
print(wilson[["GEOID00","GEOID10","AREALAND00","AREALANDPT"]].head(20))
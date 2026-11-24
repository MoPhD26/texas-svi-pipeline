import math

import numpy as np
import pandas as pd
import pytest


def make_tracts():
    return pd.DataFrame({
        "GEOID10":     ["48029110100", "48029110200", "48453000101"],
        "county_name": ["Bexar", "Bexar", "Travis"],
        "B01001_001E": [4200, 0, 3100],
        "B19301_001E": [31000, -666666666, 45000],
        "B19301_001M": [2100, -222222222, 3300],
        "B17001_002E": [600, 666666666, 210],
    })


def test_every_sentinel_code_becomes_nan(sentinel):
    df, log = sentinel.replace_sentinels(make_tracts(), 2009, verbose=False)
    assert pd.isna(df.loc[1, "B19301_001E"])
    assert pd.isna(df.loc[1, "B19301_001M"])
    assert pd.isna(df.loc[1, "B17001_002E"])
    assert {r["sentinel_value"] for r in log} == {-666666666, -222222222, 666666666}


def test_real_values_and_metadata_are_untouched(sentinel):
    df, _ = sentinel.replace_sentinels(make_tracts(), 2009, verbose=False)
    assert df.loc[0, "B19301_001E"] == 31000
    assert df.loc[2, "B17001_002E"] == 210
    assert list(df["GEOID10"]) == list(make_tracts()["GEOID10"])


def test_log_records_which_tracts_were_affected(sentinel):
    _, log = sentinel.replace_sentinels(make_tracts(), 2009, verbose=False)
    income = next(r for r in log if r["variable"] == "B19301_001E")
    assert income["tracts_affected"] == 1
    assert "48029110200" in income["affected_geoids"]


def test_clean_file_produces_empty_log(sentinel):
    df = make_tracts().iloc[[0, 2]]
    _, log = sentinel.replace_sentinels(df, 2019, verbose=False)
    assert log == []


def test_cpi_factors_match_bls_annual_averages(cpi):
    assert cpi.cpi_factor(2019) == 1.0
    assert cpi.cpi_factor(2009) == pytest.approx(255.657 / 214.537)
    assert cpi.cpi_factor(2024) == pytest.approx(255.657 / 313.689)
    assert cpi.cpi_factor(2009) > 1 > cpi.cpi_factor(2024)


def test_cpi_uses_annual_average_not_june_value(cpi):
    # An earlier draft used 314.175, the June 2024 monthly index.
    assert cpi.CPI[2024] == 313.689


def test_income_and_moe_scale_by_the_same_factor(cpi):
    df = pd.DataFrame({"GEOID10": ["a", "b"], "B19301_001E": [30000, np.nan], "B19301_001M": [1500, np.nan]})
    out, summary = cpi.adjust_income(df, 2024)
    f = 255.657 / 313.689
    assert out.loc[0, "B19301_001E"] == pytest.approx(30000 * f)
    assert out.loc[0, "B19301_001M"] == pytest.approx(1500 * f)
    assert math.isnan(out.loc[1, "B19301_001E"])
    assert summary["n_adjusted"] == 1 and summary["n_null"] == 1


def test_cpi_documentation_travels_with_the_data(cpi):
    df = pd.DataFrame({"GEOID10": ["a"], "B19301_001E": [100.0], "B19301_001M": [10.0]})
    out, _ = cpi.adjust_income(df, 2014)
    assert out.loc[0, "B19301_cpi_base_year"] == 2019
    assert out.loc[0, "B19301_cpi_vintage_year"] == 2014
    assert out.loc[0, "B19301_cpi_source"] == "BLS CUUR0000SA0 Annual Average"

"""
Crosswalk math on a tiny synthetic example.

Source tract S1 splits 60/40 by land area into 2010 tracts T1 and T2.
Source tract S2 sits entirely inside T2.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rel():
    return pd.DataFrame({
        "geoid20":    ["S1", "S1", "S2"],
        "geoid10":    ["T1", "T2", "T2"],
        "aland_part": [60.0, 40.0, 50.0],
        "aland_20":   [100.0, 100.0, 50.0],
        "aland_10":   [60.0, 90.0, 90.0],
        "weight":     [0.6, 0.4, 1.0],
    })


@pytest.fixture
def source():
    return pd.DataFrame({
        "GEOID10":     ["S1", "S2"],
        "msa":         ["Austin", "Austin"],
        "county_fips": ["453", "453"],
        "county_name": ["Travis", "Travis"],
        "B01001_001E": [1000, 500],
        "B01001_001M": [100, 80],
        "B19301_001E": [30000, 60000],
        "B19301_001M": [3000, 4000],
    })


def by_tract(df):
    return df.set_index("GEOID10")


def test_counts_are_split_by_land_area_and_totals_are_preserved(crosswalk, rel, source):
    out = by_tract(crosswalk.crosswalk_counts(source, rel, ["B01001_001E"], ["B01001_001M"]))
    assert out.loc["T1", "B01001_001E"] == pytest.approx(600)
    assert out.loc["T2", "B01001_001E"] == pytest.approx(400 + 500)
    assert out["B01001_001E"].sum() == pytest.approx(source["B01001_001E"].sum())


def test_moes_combine_by_root_sum_of_squares(crosswalk, rel, source):
    out = by_tract(crosswalk.crosswalk_counts(source, rel, ["B01001_001E"], ["B01001_001M"]))
    assert out.loc["T1", "B01001_001M"] == pytest.approx(60)
    assert out.loc["T2", "B01001_001M"] == pytest.approx(np.sqrt(40**2 + 80**2))


def test_variables_missing_from_a_vintage_are_skipped(crosswalk, rel, source):
    # B23025 unemployment does not exist in the 2009 ACS at tract level
    out = crosswalk.crosswalk_counts(source, rel, ["B01001_001E", "B23025_005E"], ["B01001_001M", "B23025_005M"])
    assert "B23025_005E" not in out.columns
    assert "B01001_001E" in out.columns


def test_income_is_population_weighted_not_summed(crosswalk, rel, source):
    out = by_tract(crosswalk.crosswalk_rates(source, rel, "B01001_001E", ["B19301_001E"], ["B19301_001M"]))
    assert out.loc["T1", "B19301_001E"] == pytest.approx(30000)
    # T2 gets 400 people at $30,000 and 500 people at $60,000
    assert out.loc["T2", "B19301_001E"] == pytest.approx((400 * 30000 + 500 * 60000) / 900)


def test_income_is_null_when_no_population_contributes(crosswalk):
    rel = pd.DataFrame({"geoid20": ["S"], "geoid10": ["T"], "aland_part": [1.0],
                        "aland_20": [1.0], "aland_10": [1.0], "weight": [1.0]})
    src = pd.DataFrame({"GEOID10": ["S"], "B01001_001E": [0], "B19301_001E": [50000], "B19301_001M": [100]})
    out = crosswalk.crosswalk_rates(src, rel, "B01001_001E", ["B19301_001E"], ["B19301_001M"])
    assert pd.isna(out.loc[0, "B19301_001E"])


def test_metadata_comes_from_the_largest_contributing_tract(crosswalk, rel):
    src = pd.DataFrame({"GEOID10": ["S1", "S2"], "msa": ["A", "B"],
                        "county_fips": ["453", "209"], "county_name": ["Travis", "Hays"]})
    meta = by_tract(crosswalk.build_meta_lookup(src, rel))
    assert meta.loc["T2", "county_name"] == "Hays"   # 50 land units beat 40


def test_weight_check_flags_bad_relationship_rows(crosswalk, rel, capsys):
    crosswalk.weight_sum_check(rel, "good")
    assert "all source tract weight sums" in capsys.readouterr().out
    bad = rel.copy()
    bad.loc[0, "weight"] = 0.5
    crosswalk.weight_sum_check(bad, "bad")
    assert "S1: weight_sum = 0.9000" in capsys.readouterr().out

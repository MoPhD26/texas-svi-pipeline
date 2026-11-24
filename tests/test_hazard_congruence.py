import numpy as np
import pandas as pd
import pytest


def test_nri_is_filtered_to_the_13_counties_with_fips_padding(hazard, tmp_path):
    f = tmp_path / "nri.csv"
    pd.DataFrame({
        "TRACTFIPS": ["48029110100", "48453000101", "48201100000"],
        "STCOFIPS": ["48029", "48453", "48201"],   # 48201 is Harris County, outside the study area
        "EAL_VALT": [1000.0, 2500.0, 9999.0],
    }).to_csv(f, index=False)
    out = hazard.load_and_filter_nri(f)
    assert list(out["GEOID20"]) == ["48029110100", "48453000101"]
    assert list(out["MSA"]) == ["San Antonio-New Braunfels MSA", "Austin-Round Rock-Georgetown MSA"]


def test_nri_accepts_the_older_countyfips_column_name(hazard, tmp_path):
    f = tmp_path / "nri.csv"
    pd.DataFrame({"TRACTFIPS": ["48029110100"], "COUNTYFIPS": ["48029"], "EAL_VALT": [5.0]}).to_csv(f, index=False)
    assert len(hazard.load_and_filter_nri(f)) == 1


def test_eal_is_area_weighted_onto_2010_tracts(hazard, tmp_path):
    xw = tmp_path / "xw.txt"
    pd.DataFrame({
        "GEOID_TRACT_20": ["48029000101", "48029000102", "48029000200"],
        "GEOID_TRACT_10": ["48029000100", "48029000100", "48029000200"],
        "AREALAND_PART":  [100, 300, 50],
        "AREALAND_TRACT_20": [100, 300, 100],
    }).to_csv(xw, sep="|", index=False)
    nri = pd.DataFrame({"GEOID20": ["48029000101", "48029000102", "48029000200"],
                        "EAL_VALT": [1000.0, 3000.0, 800.0]})
    out = hazard.crosswalk_nri_to_2010(nri, xw).set_index("GEOID10")
    assert out.loc["48029000100", "EAL_VALT_2010"] == pytest.approx(4000)    # two 2020 tracts merge
    assert out.loc["48029000100", "n_source_tracts_2020"] == 2
    assert out.loc["48029000200", "EAL_VALT_2010"] == pytest.approx(400)     # half the land area


def test_missing_crosswalk_falls_back_to_direct_geoid_match(hazard, tmp_path):
    nri = pd.DataFrame({"GEOID20": ["48029000100"], "EAL_VALT": [123.0]})
    out = hazard.crosswalk_nri_to_2010(nri, tmp_path / "missing.txt")
    assert out.loc[0, "GEOID10"] == "48029000100" and out.loc[0, "EAL_VALT_2010"] == 123.0


def test_manual_dunn_adjusts_p_values_with_benjamini_hochberg(hazard):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "TRAJECTORY": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        "EAL_USD": np.concatenate([rng.normal(100, 10, 30), rng.normal(100, 10, 30), rng.normal(160, 10, 30)]),
    })
    out = hazard.manual_dunn(df)
    assert len(out) == 3
    assert (out["p_adj"] >= out["p_raw"]).all()
    assert out["p_adj"].is_monotonic_increasing
    sig = set(map(frozenset, out.loc[out["significant_005"], ["class_a", "class_b"]].values))
    assert {frozenset({"A", "C"}), frozenset({"B", "C"})} <= sig
    assert frozenset({"A", "B"}) not in sig


def test_gi_bins_follow_arcgis_confidence_levels(verify):
    z = np.array([0.5, 1.7, 2.0, 2.6, -1.7, -2.0, -2.6])
    assert list(verify.classify_gi_bin(z)) == [0, 1, 2, 3, -1, -2, -3]
    assert [verify.label_bin(b) for b in (3, 1, 0, -2)] == ["Hot", "Hot", "NS", "Cold"]


def test_fdr_correction_never_finds_more_hot_spots(verify):
    z = np.random.default_rng(1).normal(0, 1.5, 807)
    raw = verify.classify_gi_bin(z, fdr=False)
    fdr = verify.classify_gi_bin(z, fdr=True)
    assert (np.abs(fdr) > 0).sum() <= (np.abs(raw) > 0).sum()

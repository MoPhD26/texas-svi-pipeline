import numpy as np
import pandas as pd


def vintage(year, n=3, dp02="DP02_0113", unemployment=True):
    df = pd.DataFrame({
        "YEAR5": year,
        "GEOID10": [f"48029{i:06d}" for i in range(n)],
        "msa": "San Antonio-New Braunfels MSA",
        "county_fips": "029", "county_name": "Bexar",
        "tract_code": [f"{i:06d}" for i in range(n)],
        "NAME": "x", "state": "48",
        "B01001_001E": 1000, "B19301_001E": 32000.0,
        f"{dp02}PE": 7.5, f"{dp02}PM": 1.2,
    })
    if unemployment:
        df["B23025_003E"] = 500
        df["B23025_005E"] = 25
    return df


def test_dp02_codes_are_renamed_to_one_canonical_column(stack):
    out = stack.standardize_vintage(vintage(2019, dp02="DP02_0114"), 2019, verbose=False)
    assert out.loc[0, "DP02_ltd_english_PE"] == 7.5
    assert out.loc[0, "DP02_source_code"] == "DP02_0114PE"
    assert "DP02_0114PE" not in out.columns


def test_each_vintage_uses_its_own_dp02_code(stack):
    assert stack.DP02_SOURCE_CODES == {2009: "DP02_0113", 2014: "DP02_0113", 2019: "DP02_0114", 2024: "DP02_0115"}


def test_2009_gets_unemployment_columns_as_nan(stack):
    out = stack.standardize_vintage(vintage(2009, unemployment=False), 2009, verbose=False)
    assert out["B23025_003E"].isna().all()
    assert out["B23025_005M"].isna().all()


def test_boundary_metadata_and_dropped_columns(stack):
    out09 = stack.standardize_vintage(vintage(2009), 2009, verbose=False)
    out14 = stack.standardize_vintage(vintage(2014), 2014, verbose=False)
    assert out09.loc[0, "boundary_vintage"] == 2000 and bool(out09.loc[0, "crosswalked"])
    assert out14.loc[0, "boundary_vintage"] == 2010 and not bool(out14.loc[0, "crosswalked"])
    assert "NAME" not in out14.columns and "state" not in out14.columns
    assert out14.loc[0, "crosswalk_note"].startswith("No crosswalk")


def full_panel(stack, n=807):
    frames = []
    for yr, code in stack.DP02_SOURCE_CODES.items():
        frames.append(stack.standardize_vintage(
            vintage(yr, n=n, dp02=code, unemployment=yr != 2009), yr, verbose=False))
    return stack.stack(frames)


def test_stacked_panel_passes_validation(stack):
    panel = full_panel(stack)
    assert len(panel) == 3228
    assert stack.validate(panel) == []
    assert list(panel.columns[:6]) == stack.ID_COLS
    assert panel.iloc[0]["YEAR5"] == 2009 and panel.iloc[1]["YEAR5"] == 2014


def test_validation_catches_missing_tracts_and_duplicates(stack):
    panel = full_panel(stack)
    assert any("expected 807" in p for p in stack.validate(panel.iloc[1:]))
    dup = pd.concat([panel, panel.iloc[[0]]])
    assert any("duplicate" in p for p in stack.validate(dup))
    bad = panel.copy()
    bad.loc[bad["YEAR5"] == 2009, "B23025_003E"] = 1
    assert any("2009 has unemployment" in p for p in stack.validate(bad))
    assert np.isnan(panel.loc[panel["YEAR5"] == 2009, "B23025_003E"]).all()

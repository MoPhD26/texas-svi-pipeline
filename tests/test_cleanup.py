import pandas as pd


def tracts(geoids, **cols):
    return pd.DataFrame({"GEOID10": geoids, **cols})


def test_ghost_tracts_outside_the_13_counties_are_removed(cleanup):
    df = tracts(["48029110100", "48453000101", "48053960100", "48163950100"],
                B01001_001E=[4000, 3000, 12.5, 180.2])
    kept, ghosts = cleanup.split_ghost_tracts(df)
    assert list(kept["GEOID10"]) == ["48029110100", "48453000101"]
    assert set(ghosts["GEOID10"].str[2:5]) == {"053", "163"}


def test_study_area_has_13_counties(cleanup):
    assert len(cleanup.MSA_COUNTY_FIPS) == 13


def test_wilson_income_is_restored_only_where_null(cleanup):
    cw = tracts(["48493000101", "48493000102", "48029110100"],
                B19301_001E=[None, 28000, None], B19301_001M=[None, 900, None])
    src = tracts(["48493000101", "48493000102", "48029110100"],
                 B19301_001E=[31000, 99999, 55555], B19301_001M=[1200, 1, 1])
    out, n = cleanup.restore_income(cw, src)
    assert n == 1
    assert out.loc[0, "B19301_001E"] == 31000
    assert out.loc[1, "B19301_001E"] == 28000           # existing value kept
    assert pd.isna(out.loc[2, "B19301_001E"])           # other counties untouched


def test_cpi_columns_are_added_once(cleanup):
    out = cleanup.add_cpi_columns(tracts(["48029110100"]), 2024)
    assert out.loc[0, "B19301_cpi_factor"] == round(255.657 / 313.689, 6)
    again = cleanup.add_cpi_columns(out.assign(B19301_cpi_factor=1.23), 2024)
    assert again.loc[0, "B19301_cpi_factor"] == 1.23


def test_validation_catches_count_duplicates_and_ghosts(cleanup):
    good = tracts([f"48029{i:06d}" for i in range(807)])
    assert cleanup.validate(good, "x") == []
    dup = pd.concat([good.iloc[:806], good.iloc[[0]]])
    assert any("duplicate" in p for p in cleanup.validate(dup, "x"))
    ghost = good.copy()
    ghost.loc[0, "GEOID10"] = "48053960100"
    assert any("outside" in p for p in cleanup.validate(ghost, "x"))
    assert any("expected 807" in p for p in cleanup.validate(good.iloc[:800], "x"))

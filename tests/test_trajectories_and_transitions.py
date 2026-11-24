import pandas as pd
import pytest


def seq(s):
    return s.split("-")


@pytest.mark.parametrize("sequence, expected", [
    ("HH-HH-HH-HH", "Persistently_High"),
    ("HH-NS-HH-HH", "Persistently_High"),      # 3 of 4, both endpoints HH
    ("HH-HH-HH-NS", "Declining_High"),
    ("HH-NS-NS-HH", "Transient"),              # endpoints only, no persistence
    ("NS-NS-HH-HH", "Emerging_High"),
    ("NS-HH-NS-HH", "Transient"),              # 2019 must also be HH to be emerging
    ("LL-LL-LL-LL", "Persistently_Low"),
    ("LL-LL-NS-NS", "Declining_Low"),
    ("NS-NS-LL-LL", "Emerging_Low"),
    ("NS-NS-LH-LH", "Recurrent_Outlier"),      # the dissertation's own example
    ("HL-NS-HL-NS", "Recurrent_Outlier"),
    ("NS-HH-NS-NS", "Transient"),
    ("NS-NS-NS-LL", "Transient"),
    ("NS-NS-NS-NS", "Never_Significant"),
    ("HH-NO_DATA-HH-HH", "Incomplete_Data"),
])
def test_trajectory_rules_match_the_dissertation(trajectory, sequence, expected):
    assert trajectory.classify_sequence(seq(sequence)) == expected


def test_high_rules_are_checked_before_low_rules(trajectory):
    # HH in 2009, LL in 2024 is a declining high tract first
    assert trajectory.classify_sequence(seq("HH-NS-NS-LL")) == "Declining_High"


def test_there_are_exactly_ten_classes(trajectory):
    assert len(trajectory.TRAJ_ORDER) == 10
    assert set(trajectory.TRAJ_ORDER) == set(trajectory.TRAJ_COLORS)


def test_every_possible_sequence_gets_a_known_class(trajectory):
    from itertools import product
    types = ["HH", "LL", "HL", "LH", "NS", "NO_DATA"]
    labels = {trajectory.classify_sequence(list(s)) for s in product(types, repeat=4)}
    assert labels == set(trajectory.TRAJ_ORDER)


def test_blank_cluster_types_count_as_not_significant(lisa_check):
    df = pd.DataFrame({"SOURCE_ID": [1, 2, 3, 4], "LISA_COType": ["HH", None, "", "LL"]})
    counts = lisa_check.cotype_distribution(df)
    assert counts["NS"] == 2 and counts["HH"] == 1 and counts["LL"] == 1


def test_pooled_quintiles_split_all_tract_years_evenly(transition):
    svi = pd.DataFrame({"SVI": range(100)})
    out = transition.assign_pooled_quintiles(svi)
    assert out["Quintile"].value_counts().tolist() == [20] * 5
    assert out.loc[0, "Quintile"] == 1 and out.loc[99, "Quintile"] == 5


@pytest.mark.parametrize("quintiles, expected", [
    ((5, 5, 5, 5), "Persistent_Q5"),
    ((1, 2, 1, 1), "Persistent_Q1"),
    ((3, 3, 3, 3), "Persistent_Mid"),
    ((5, 4, 3, 2), "Improved"),
    ((2, 3, 3, 4), "Worsened"),
    ((3, 4, 2, 4), "Mixed"),
    ((None, 3, 3, 3), "No_Data"),
])
def test_mobility_type(transition, quintiles, expected):
    row = {f"Q_{y}": q for y, q in zip([2009, 2014, 2019, 2024], quintiles)}
    assert transition.mobility_type(pd.Series(row)) == expected


def test_transition_matrix_and_persistence_rates(transition):
    df = pd.DataFrame({
        "Q_2009": [5, 5, 5, 5, 1, 1],
        "Q_2024": [5, 5, 5, 3, 1, 2],
    })
    counts, pct = transition.build_matrix(df, 2009, 2024)
    assert counts.shape == (5, 5)                       # all quintiles present even if empty
    assert counts.loc[5, 5] == 3 and pct.loc[5, 5] == 75.0
    m = transition.compute_metrics(counts, 2009, 2024, "Test")
    assert m["Q5_persist_pct"] == 75.0
    assert m["Q5_improve_pct"] == 25.0
    assert m["Q1_worsen_pct"] == 50.0
    assert m["overall_persist_pct"] == pytest.approx(4 / 6 * 100)

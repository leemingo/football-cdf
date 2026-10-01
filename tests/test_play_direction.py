"""`home_team_side[i]` is period `i + 1`, whatever order `match_periods` is in.

The provider writes `home_team_side` in fixed period order and gives it no
period numbers. `match_periods` is not always in that order: 8 of 612 K League
files list it as [2, 1]. Pairing the two arrays by list position therefore swaps
the halves' attacking directions in exactly those files, which survives the
home-relative normalisation and leaves every snapshot rotated 180 degrees.

The convention was checked against raw tracking, not assumed: in all eight files
the home keeper's median x in period 1 agrees with `home_team_side[0]`.
"""
from __future__ import annotations

import pandas as pd
import pytest

from football_cdf.skillcorner_preprocessing import SkillcornerDataPreprocessor as P


def _meta(periods, sides):
    return {"match_periods": [{"period": p, "name": f"P{p}"} for p in periods],
            "home_team_side": sides}


def test_ascending_periods_map_in_order():
    m = _meta([1, 2], ["left_to_right", "right_to_left"])
    assert P._home_team_side_by_period(m) == {1: "left_to_right", 2: "right_to_left"}


def test_descending_match_periods_do_not_reorder_the_sides():
    # The shape of the eight affected K League files: `match_periods` is [2, 1]
    # but `home_team_side` is still [period 1, period 2].
    m = _meta([2, 1], ["left_to_right", "right_to_left"])
    assert P._home_team_side_by_period(m) == {1: "left_to_right", 2: "right_to_left"}


def test_match_periods_order_is_irrelevant_to_the_mapping():
    sides = ["left_to_right", "right_to_left"]
    assert (P._home_team_side_by_period(_meta([1, 2], sides))
            == P._home_team_side_by_period(_meta([2, 1], sides))
            == {1: "left_to_right", 2: "right_to_left"})


def test_play_direction_ignores_match_periods_order():
    direction = P._build_play_direction(_meta([2, 1], ["left_to_right", "right_to_left"]))
    assert direction["first_half"] == "leftright"
    assert direction["second_half"] == "rightleft"


def test_period_details_carry_the_matching_side():
    rows = P._extract_period_details(_meta([2, 1], ["left_to_right", "right_to_left"]))
    assert rows["home_team_side_period_1"] == "left_to_right"
    assert rows["home_team_side_period_2"] == "right_to_left"


@pytest.mark.parametrize("meta", [
    {"match_periods": [], "home_team_side": []},
    {"match_periods": [{"period": 1}], "home_team_side": []},
    {"match_periods": [{"period": None}], "home_team_side": ["left_to_right"]},
])
def test_missing_or_partial_metadata_does_not_raise(meta):
    assert isinstance(P._home_team_side_by_period(meta), dict)
    assert isinstance(P._build_play_direction(meta), dict)


def test_side_is_missing_rather_than_guessed_when_the_arrays_disagree():
    # Two periods but one side: the unpaired period gets no direction rather
    # than borrowing its neighbour's.
    rows = P._extract_period_details(_meta([1, 2], ["left_to_right"]))
    assert rows["home_team_side_period_1"] == "left_to_right"
    assert pd.isna(rows["home_team_side_period_2"])

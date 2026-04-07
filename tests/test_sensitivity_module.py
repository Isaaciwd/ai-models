import numpy as np

from ai_models.sensitivity import parse_target_area
from ai_models.sensitivity import target_slug


def test_parse_target_area_with_string():
    assert parse_target_area("60,200,10,280") == (60.0, 200.0, 10.0, 280.0)


def test_parse_target_area_with_sequence():
    assert parse_target_area([50, 230, 30, 245]) == (50.0, 230.0, 30.0, 245.0)


def test_parse_target_area_wraps_longitudes():
    north, west, south, east = parse_target_area("50,-130,30,-115")
    assert np.isclose(north, 50.0)
    assert np.isclose(south, 30.0)
    assert np.isclose(west, 230.0)
    assert np.isclose(east, 245.0)


def test_target_slug_sanitizes():
    assert target_slug("West Coast R850", "fallback") == "west-coast-r850"

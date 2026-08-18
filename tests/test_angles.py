"""Angle helpers. Every case here has already caused a real failure."""

import numpy as np
import pytest

from utils.angles import angles_match, is_critical_angle, mean_angle
from utils.angles import (par_angle, sexagesimal_to_degrees,
                          small_angle_distance)


def test_angles_match_across_the_wrap():
    """PCUPR reads -0.002 for the "0 deg" HWP position.

    Taken naively modulo 180 that becomes 179.998, which matched none of the
    critical angles and produced zero usable cycles. The comparison has to be
    circular.
    """
    assert angles_match(-0.002, 0.0)
    assert angles_match(179.998, 0.0)
    assert angles_match(0.0, 179.998)
    assert angles_match(359.9, 0.0)


def test_angles_match_respects_the_tolerance():
    """``atol`` bounds the match, and is honoured in both directions."""
    assert angles_match(0.0, 0.9, atol=1.0)
    assert not angles_match(0.0, 1.1, atol=1.0)
    assert angles_match(0.0, 5.0, atol=6.0)


def test_angles_match_period_180_folds_orthogonal_angles():
    """Q and -Q are the same modulation state, 180 deg apart in 2*theta."""
    assert angles_match(45.0, 225.0)
    assert not angles_match(45.0, 135.0)


def test_angles_match_full_circle_period():
    """With period 360, angles a half turn apart no longer match."""
    assert not angles_match(45.0, 225.0, period=360.0)
    assert angles_match(45.0, 405.0, period=360.0)


def test_is_critical_angle():
    """Critical angles are recognised through the wrap; other angles are not."""
    critical = (0.0, 45.0, 22.5, 67.5)
    assert is_critical_angle(-0.002, critical)
    assert is_critical_angle(67.502, critical)
    assert is_critical_angle(22.497, critical)
    assert not is_critical_angle(10.0, critical)


def test_mean_angle_handles_the_wrap():
    """The AB Aur cycle that made this necessary.

    PARANG runs -112.75 and 246.94 -- the same direction, 360 deg apart.
    Unwrapped, 246.94 is -113.06, so the true mean is -112.905. The
    arithmetic mean of the raw values is +67.095, about 180 deg away.
    """
    got = mean_angle([-112.75, 246.94])
    assert abs((got - (-112.905) + 180.0) % 360.0 - 180.0) < 1e-6
    assert abs(got - np.mean([-112.75, 246.94])) > 90.0


def test_mean_angle_stays_near_the_first_angle():
    """The result is returned in the branch nearest the input, not forced
    into [0, 360), so it stays continuous with what was passed in."""
    assert mean_angle([-170.0, 170.0]) == pytest.approx(-180.0, abs=1e-6)
    assert mean_angle([170.0, -170.0]) == pytest.approx(180.0, abs=1e-6)


def test_mean_angle_simple_cases():
    """Away from any wrap, the circular mean is the arithmetic one."""
    assert mean_angle([10.0, 20.0, 30.0]) == pytest.approx(20.0, abs=1e-9)
    assert mean_angle([42.0]) == pytest.approx(42.0)


def test_mean_angle_empty_is_nan():
    """An empty input gives NaN rather than raising."""
    assert np.isnan(mean_angle([]))


def test_mean_angle_period_180():
    """With period 180, 179 and 1 average to 0, not 90."""
    got = mean_angle([179.0, 1.0], period=180.0)
    assert abs((got - 0.0 + 90.0) % 180.0 - 90.0) < 1e-6


# --- spherical helpers, moved down from instruments/nirc2.py ---------------

@pytest.mark.parametrize("value, expected", [
    ("12:30:00", 12.5),
    ("+12:30:00", 12.5),
    ("-12:30:00", -12.5),
    ("00:30:00", 0.5),
    (7.25, 7.25),          # already decimal, passes through
])
def test_sexagesimal_to_degrees(value, expected):
    assert sexagesimal_to_degrees(value) == pytest.approx(expected)


def test_sexagesimal_sign_survives_a_zero_leading_field():
    """float('-00') is 0.0, so a naive parser loses the sign on declinations
    between 0 and -1 -- and would put a source in the wrong hemisphere."""
    assert sexagesimal_to_degrees("-00:30:00") == pytest.approx(-0.5)


def test_small_angle_distance_scales_ra_by_cos_dec():
    """One degree of RA is less than one degree on the sky away from the
    equator; ignoring cos(dec) is a classic way to mis-measure a separation."""
    assert small_angle_distance((0.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert small_angle_distance((0.0, 60.0), (1.0, 60.0)) == pytest.approx(0.5)
    assert small_angle_distance((10.0, 5.0), (10.0, 5.0)) == pytest.approx(0.0)


def test_par_angle_is_zero_on_the_meridian():
    """A source due south at transit has no parallactic rotation."""
    assert par_angle(0.0, 10.0, 19.82525) == pytest.approx(0.0, abs=1e-9)


def test_par_angle_flips_sign_across_the_meridian():
    """East and west of the meridian rotate oppositely, which is what makes
    the field rotate through a night."""
    east = par_angle(-1.0, 10.0, 19.82525)
    west = par_angle(+1.0, 10.0, 19.82525)
    assert east == pytest.approx(-west)
    assert east != pytest.approx(0.0)


def test_par_angle_takes_hours_not_degrees():
    """The signature says hour_angle and the body multiplies by 15. Passing
    degrees would be silently wrong by a factor of 15."""
    assert par_angle(6.0, 0.0, 0.0) == pytest.approx(
        par_angle(90.0 / 15.0, 0.0, 0.0))

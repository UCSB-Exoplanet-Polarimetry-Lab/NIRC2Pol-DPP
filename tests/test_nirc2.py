"""NIRC2-specific header interpretation."""

import numpy as np
import pytest

from instruments.nirc2 import (PLATE_SCALE, band_of, calculate_north_angle,
                               check_background_choice)
from instruments.nirc2 import NIRC2PolarimetryData


def test_band_of_prefers_fwiname():
    assert band_of({"FWINAME": "Kp", "FILTER": "Kp + Wollaston"}) == "Kp"


def test_band_of_falls_back_to_the_filter_first_token():
    """FILTER is the combination of both wheels, so the band is its head."""
    assert band_of({"FILTER": "Kp + Wollaston"}) == "Kp"
    assert band_of({"FILTER": "H2O_ice + Wollaston"}) == "H2O_ice"


def test_band_of_strips_whitespace():
    assert band_of({"FWINAME": "  Lp "}) == "Lp"


@pytest.mark.parametrize("slitname,expected", [
    ("corona150", 150 / 2 / 1000 / PLATE_SCALE),
    ("corona400", 400 / 2 / 1000 / PLATE_SCALE),
    ("corona1000", 1000 / 2 / 1000 / PLATE_SCALE),
])
def test_occulting_radius_from_slitname(slitname, expected):
    """coronaNNN names a *diameter* in milliarcsec, so corona150 is 7.5 px."""
    got = NIRC2PolarimetryData().occulting_radius({"SLITNAME": slitname})
    assert got == pytest.approx(expected)


def test_occulting_radius_corona150_is_about_seven_and_a_half_pixels():
    got = NIRC2PolarimetryData().occulting_radius({"SLITNAME": "corona150"})
    assert got == pytest.approx(7.54, abs=0.01)


@pytest.mark.parametrize("slitname", ["none", "", "clear", "coronaXYZ"])
def test_occulting_radius_none_when_unocculted_or_unparseable(slitname):
    assert NIRC2PolarimetryData().occulting_radius(
        {"SLITNAME": slitname}) is None


def test_check_background_choice_warns_on_annulus_at_lprime():
    """An annulus subtracts one scalar, which is only right for a flat
    background. The L' thermal pedestal is both huge and structured, so it
    needs dither pairs or at least a chosen box."""
    assert check_background_choice("Lp", "annulus") is False
    assert check_background_choice("Lp", "dither") is True
    assert check_background_choice("Lp", "mean_box") is True


def test_check_background_choice_accepts_annulus_in_the_near_infrared():
    assert check_background_choice("Kp", "annulus") is True
    assert check_background_choice("H", "annulus") is True


def test_check_background_choice_passes_unknown_bands_and_none():
    assert check_background_choice("H2O_ice", "annulus") is True
    assert check_background_choice("Lp", None) is True


# --- north angle -------------------------------------------------------
#
# Property-based on purpose. Pinning a computed value would lock in whatever
# the model does today, including any error in it; these assert the
# relationships the model claims instead.

def _pa_header(mode, rotposn=90.0, parang=30.0, instangl=0.7):
    return {"ROTMODE": mode, "ROTPOSN": rotposn, "INSTANGL": instangl,
            "PARANG": parang}


def test_position_angle_mode_ignores_the_parallactic_angle():
    """The rotator holds the field, so PARANG must not enter.

    This is why the Sgr A* sequence came out constant to 0.000 deg across an
    hour of parallactic rotation.
    """
    a, _ = calculate_north_angle(_pa_header("position angle", parang=10.0))
    b, _ = calculate_north_angle(_pa_header("position angle", parang=170.0))
    assert a == pytest.approx(b)


def test_position_angle_mode_tracks_the_rotator_one_for_one():
    a, _ = calculate_north_angle(_pa_header("position angle", rotposn=90.0))
    b, _ = calculate_north_angle(_pa_header("position angle", rotposn=100.0))
    assert b - a == pytest.approx(10.0)


def test_position_angle_mode_offsets_by_the_instrument_angle():
    a, _ = calculate_north_angle(_pa_header("position angle", instangl=0.0))
    b, _ = calculate_north_angle(_pa_header("position angle", instangl=5.0))
    assert a - b == pytest.approx(5.0)


def test_stationary_mode_is_nan():
    angle, _ = calculate_north_angle(_pa_header("stationary"))
    assert np.isnan(angle)


def test_unknown_rotator_mode_raises():
    with pytest.raises(ValueError, match="Unknown rotator mode"):
        calculate_north_angle(_pa_header("nonsense"))

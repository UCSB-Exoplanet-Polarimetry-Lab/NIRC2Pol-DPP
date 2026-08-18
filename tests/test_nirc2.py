"""NIRC2-specific header interpretation."""

import logging

import numpy as np

from utils import Frame
import pytest

from instruments.nirc2 import (PLATE_SCALE, band_of, calculate_north_angle,
                               check_background_choice)
from instruments.nirc2 import NIRC2PolarimetryData


def test_band_of_prefers_fwiname():
    """FWINAME wins when present, since FILTER holds both wheels."""
    assert band_of({"FWINAME": "Kp", "FILTER": "Kp + Wollaston"}) == "Kp"


def test_band_of_falls_back_to_the_filter_first_token():
    """FILTER is the combination of both wheels, so the band is its head."""
    assert band_of({"FILTER": "Kp + Wollaston"}) == "Kp"
    assert band_of({"FILTER": "H2O_ice + Wollaston"}) == "H2O_ice"


def test_band_of_strips_whitespace():
    """Padded header values are trimmed."""
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
    """The documented 7.5 px figure for corona150, checked numerically."""
    got = NIRC2PolarimetryData().occulting_radius({"SLITNAME": "corona150"})
    assert got == pytest.approx(7.54, abs=0.01)


@pytest.mark.parametrize("slitname", ["none", "", "clear", "coronaXYZ"])
def test_occulting_radius_none_when_unocculted_or_unparseable(slitname):
    """Unocculted or unrecognised slit names give None, not a guess."""
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
    """An annulus is the recommended choice in JHK."""
    assert check_background_choice("Kp", "annulus") is True
    assert check_background_choice("H", "annulus") is True


def test_check_background_choice_passes_unknown_bands_and_none():
    """Unknown bands and an unset method pass, rather than warning blindly."""
    assert check_background_choice("H2O_ice", "annulus") is True
    assert check_background_choice("Lp", None) is True


# --- north angle -------------------------------------------------------
#
# Property-based on purpose. Pinning a computed value would lock in whatever
# the model does today, including any error in it; these assert the
# relationships the model claims instead.

def _pa_header(mode, rotposn=90.0, parang=30.0, instangl=0.7):
    """A header for the north-angle model.

    Returns
    -------
    dict
        Header with the rotator mode, position, instrument angle and PARANG.
    """
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
    """In position angle mode the north angle follows ROTPOSN exactly."""
    a, _ = calculate_north_angle(_pa_header("position angle", rotposn=90.0))
    b, _ = calculate_north_angle(_pa_header("position angle", rotposn=100.0))
    assert b - a == pytest.approx(10.0)


def test_position_angle_mode_offsets_by_the_instrument_angle():
    """INSTANGL enters with the opposite sign to ROTPOSN."""
    a, _ = calculate_north_angle(_pa_header("position angle", instangl=0.0))
    b, _ = calculate_north_angle(_pa_header("position angle", instangl=5.0))
    assert a - b == pytest.approx(5.0)


def test_stationary_mode_is_nan():
    """Stationary mode has no defined north angle, so it returns NaN."""
    angle, _ = calculate_north_angle(_pa_header("stationary"))
    assert np.isnan(angle)


def test_unknown_rotator_mode_raises():
    """An unrecognised rotator mode raises rather than guessing."""
    with pytest.raises(ValueError, match="Unknown rotator mode"):
        calculate_north_angle(_pa_header("nonsense"))


# --- the uncalibrated fast axis must not pass silently ---------------------

def _rot_header():
    return Frame(np.zeros((4, 4)),
                 {"PARANG": 10.0, "EL": 45.0, "ROTPDEST": 0.0}).header


def test_unspecified_fast_axis_offset_warns(caplog):
    """Reducing at theta_off = 0 leaves Q/U in the instrument frame, so the
    polarization angles mean nothing. Saying so is the whole point."""
    NIRC2PolarimetryData.reset_warnings()
    with caplog.at_level(logging.WARNING):
        NIRC2PolarimetryData().qu_rotation_angle(_rot_header())
    assert any("uncalibrated" in r.getMessage() for r in caplog.records)


def test_an_explicit_offset_does_not_warn(caplog):
    """A measured value is not a missed calibration."""
    NIRC2PolarimetryData.reset_warnings()
    with caplog.at_level(logging.WARNING):
        NIRC2PolarimetryData().qu_rotation_angle(_rot_header(), -13.0)
    assert not any("uncalibrated" in r.getMessage() for r in caplog.records)


def test_a_deliberate_zero_does_not_warn(caplog):
    """polarimetry.fast_axis evaluates the rotation at zero offset before
    scanning, which is a request rather than an omission."""
    NIRC2PolarimetryData.reset_warnings()
    with caplog.at_level(logging.WARNING):
        NIRC2PolarimetryData().qu_rotation_angle(_rot_header(), 0.0)
    assert not any("uncalibrated" in r.getMessage() for r in caplog.records)


def test_the_cube_builders_default_to_unspecified_not_zero():
    """These defaults were 0.0, which is indistinguishable from a deliberate
    zero by the time it reaches qu_rotation_angle -- so the warning could
    never fire on the ordinary path, which is the only path most people use.
    """
    import inspect

    from polarimetry.stokes import build_stokes_cube, build_stokes_cubes

    for func in (build_stokes_cube, build_stokes_cubes):
        default = inspect.signature(func).parameters["fast_axis_offset"].default
        assert default is None, (
            f"{func.__name__} defaults to {default!r}; a literal zero here "
            "silences the uncalibrated-offset warning for every caller who "
            "does not pass one")


# --- the band/background check belongs to the instrument -------------------

def test_a_poor_background_choice_for_the_band_warns(caplog):
    NIRC2PolarimetryData.reset_warnings()

    class Annulus(NIRC2PolarimetryData):
        background_method = "annulus"
        background_annulus = (150, 200)

    header = Frame(np.zeros((4, 4)), {"FWINAME": "Lp"}).header
    with caplog.at_level(logging.WARNING):
        Annulus().check_background_choice(header)
    assert any("not recommended" in r.getMessage() for r in caplog.records)


def test_the_background_warning_is_resettable(caplog):
    """It used to be keyed on instrument._bkg_checked, which did not match
    the _warned_* naming reset_warnings looks for, so this one flag alone
    could never be cleared."""
    class Annulus(NIRC2PolarimetryData):
        background_method = "annulus"
        background_annulus = (150, 200)

    header = Frame(np.zeros((4, 4)), {"FWINAME": "Lp"}).header
    Annulus.reset_warnings()
    with caplog.at_level(logging.WARNING):
        Annulus().check_background_choice(header)
        Annulus().check_background_choice(header)
    first = len([r for r in caplog.records if "not recommended" in r.getMessage()])
    assert first == 1, "should warn once per class"

    caplog.clear()
    Annulus.reset_warnings()
    with caplog.at_level(logging.WARNING):
        Annulus().check_background_choice(header)
    assert any("not recommended" in r.getMessage() for r in caplog.records)


def test_an_instrument_without_band_knowledge_stays_quiet(caplog):
    """The caller used to wrap this in except Exception: pass, so a
    non-NIRC2 instrument got no check and nothing said so. Now the base
    class simply has nothing to say, which is a different thing."""
    from instruments.base import PolarimetryData

    class Other(PolarimetryData):
        name = "other"
        plate_scale = 0.1

        def gain(self, header):
            return 1.0

        def saturation_limit(self, header):
            return 1e9

        def bad_pixel_mask(self):
            return None

        def sort_frames(self, filenames, **kwargs):
            return {}

        def north_angle(self, header):
            return 0.0

        def split_beams(self, frame):
            return None

        def qu_rotation_angle(self, header, fast_axis_offset=None):
            return 0.0

    header = Frame(np.zeros((4, 4)), {"FWINAME": "Lp"}).header
    with caplog.at_level(logging.WARNING):
        Other().check_background_choice(header)      # must not raise
    assert not caplog.records

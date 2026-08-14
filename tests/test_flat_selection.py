"""Flat selection policy: which flat is allowed to calibrate which frame.

This is policy, not implementation detail — the rules were specified rather
than derived, and getting them wrong produces a reduction that looks fine.
"""

import numpy as np
import pytest

from reduction.calibrate import find_closest_flat
from reduction.masters import flat_sort_key, required_flat_type_for
from utils.frame import Frame


def _flat(filtername="Kp + Wollaston", band="Kp", flattype="LAMP", n=64,
          nframes=3, polflat=False):
    """A master flat with the given filter, band and type.

    Returns
    -------
    Frame
        The synthetic flat, carrying the header keywords the matcher reads.
    """
    return Frame(np.ones((n, n)),
                 {"FILTER": filtername, "FWINAME": band, "NAXIS1": n,
                  "NAXIS2": n, "FLATTYPE": flattype, "NFRAMES": nframes,
                  "POLFLAT": polflat})


def _science(filtername="Kp + Wollaston", band="Kp", n=64):
    """A science frame needing a flat.

    Returns
    -------
    Frame
        The synthetic frame.
    """
    return Frame(np.ones((n, n)),
                 {"FILTER": filtername, "FWINAME": band, "NAXIS1": n,
                  "NAXIS2": n, "FILENAME": "sci.fits"})


# --- which type each band requires -------------------------------------

@pytest.mark.parametrize("band,expected", [
    ("Lp", "SKY"), ("L", "SKY"), ("Ms", "SKY"), ("M", "SKY"),
    ("J", "LAMP"), ("H", "LAMP"), ("K", "LAMP"), ("Kp", "LAMP"),
    ("Ks", "LAMP"),
])
def test_required_flat_type_by_band(band, expected):
    """Sky flats in the thermal infrared, where the dome lamp is swamped by
    thermal background; lamp flats in the near infrared."""
    assert required_flat_type_for(band) == expected


def test_required_flat_type_override_wins():
    """Some observers want skies in JHK too."""
    assert required_flat_type_for("Kp", override="SKY") == "SKY"
    assert required_flat_type_for("Kp", override="sky") == "SKY"


# --- the requirement is enforced, not merely preferred -----------------

def test_wrong_flat_type_raises():
    """An L' frame offered only a lamp flat refuses to reduce."""
    with pytest.raises(ValueError, match="requires a SKY flat"):
        find_closest_flat(_science("Lp + Wollaston", "Lp"),
                          [_flat("Lp + Wollaston", "Lp", "LAMP")])


def test_right_flat_type_is_accepted():
    """An L' frame takes a sky flat without complaint."""
    _, got = find_closest_flat(_science("Lp + Wollaston", "Lp"),
                               [_flat("Lp + Wollaston", "Lp", "SKY")])
    assert got is not None and got["FLATTYPE"] == "SKY"


def test_override_downgrades_the_error_and_is_recorded():
    """The override proceeds and records FLATMISM, so it stays auditable."""
    _, got = find_closest_flat(_science("Lp + Wollaston", "Lp"),
                               [_flat("Lp + Wollaston", "Lp", "LAMP")],
                               allow_flat_type_mismatch=True)
    assert got is not None
    assert got["FLATMISM"] is True, "the substitution must be recorded"


def test_explicit_required_type_overrides_the_band_rule():
    """An explicit required type wins over the band default."""
    _, got = find_closest_flat(_science("Lp + Wollaston", "Lp"),
                               [_flat("Lp + Wollaston", "Lp", "LAMP")],
                               required_flat_type="LAMP")
    assert got is not None


# --- the filter must match; size and exposure are not the same ---------

def test_filter_must_match():
    """A flat in the wrong filter is never used, whatever else matches."""
    _, got = find_closest_flat(_science("Kp + Wollaston", "Kp"),
                               [_flat("H + Wollaston", "H", "LAMP")])
    assert got is None, "a flat in the wrong filter describes the wrong "\
                        "throughput and must not be used"


def test_exposure_settings_are_ignored():
    """A flat is normalized, so ITIME and COADDS are irrelevant."""
    flat = _flat()
    flat["ITIME"], flat["COADDS"] = 30.0, 1
    sci = _science()
    sci["ITIME"], sci["COADDS"] = 0.45, 45
    _, got = find_closest_flat(sci, [flat])
    assert got is not None


def test_larger_flat_is_trimmed():
    """A full-frame flat is cropped to a subarray frame and marked FLATTRIM."""
    _, got = find_closest_flat(_science(n=32), [_flat(n=64)])
    assert got is not None
    assert got.shape == (32, 32)
    assert got["FLATTRIM"] is True


def test_smaller_flat_is_refused():
    """A flat smaller than the frame cannot cover it, so it is refused."""
    _, got = find_closest_flat(_science(n=64), [_flat(n=32)])
    assert got is None, "a flat smaller than the frame cannot calibrate it"


def test_flat_exceptions_substitute_a_filter():
    """Narrowband filters with no flats of their own borrow a broadband one."""
    _, got = find_closest_flat(
        _science("H2O_ice + Wollaston", "H2O_ice"),
        [_flat("Kp + Wollaston", "Kp", "LAMP")],
        exceptions={"H2O_ice": ("Kp",)}, required_flat_type="LAMP")
    assert got is not None and "Kp" in got["FILTER"]


# --- ordering among valid flats ----------------------------------------

def test_flat_sort_order():
    """Polarimetric first, then dark-subtracted, then the required type,
    then whichever was built from the most frames."""
    pol = _flat(polflat=True, nframes=3)
    nodark = _flat(flattype="LAMP+NODARK", nframes=9)
    wrong_type = _flat(flattype="SKY", nframes=9)
    few = _flat(nframes=3)
    many = _flat(nframes=9)

    order = sorted([nodark, wrong_type, few, many, pol], key=flat_sort_key)
    assert order[0] is pol
    assert order[1] is many, "more frames wins among equals"
    assert order[2] is few
    assert order[3] is wrong_type, "the non-required real type ranks above "\
                                   "a darkless one"
    assert order[4] is nodark

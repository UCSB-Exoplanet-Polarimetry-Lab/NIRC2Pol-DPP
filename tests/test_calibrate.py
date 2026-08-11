"""Bad pixel handling and dark matching."""

import numpy as np
import pytest

from reduction.calibrate import (find_closest_dark, interpolate_bad_pixels,
                                 local_interpolate_bad_pixels,
                                 local_median_replace_bad_pixels)
from utils.frame import Frame
from utils.imutils import plus_mask


def _smooth_field(n=32):
    yy, xx = np.mgrid[:n, :n]
    return 100.0 + 0.5 * xx + 0.25 * yy


def test_interpolate_replaces_an_isolated_hot_pixel():
    data = _smooth_field()
    expected = data[16, 16]
    data[16, 16] = 1e6
    mask = np.zeros(data.shape, dtype=bool)
    mask[16, 16] = True

    local_interpolate_bad_pixels(data, mask, 9)
    assert data[16, 16] == pytest.approx(expected, rel=1e-3)


def test_interpolate_handles_a_cluster_without_raising():
    """A clustered defect used to make the RBF interpolator singular.

    Growing the static bad-pixel mask by a dilation to cover the Cygnus A
    detector defect raised "Singular matrix ... does not have full column
    rank", which is why that defect is blanked to NaN rather than dilated.
    Clustered masks must at least not explode.
    """
    data = _smooth_field()
    mask = np.zeros(data.shape, dtype=bool)
    mask[14:19, 14:19] = True          # a 5x5 block
    data[mask] = 1e6

    local_interpolate_bad_pixels(data, mask, 11)
    assert np.all(np.isfinite(data))
    assert data[16, 16] < 1e5, "the cluster centre was left untouched"


def test_median_replacement_is_an_alternative_route():
    data = _smooth_field()
    expected = data[10, 10]
    data[10, 10] = -9999.0
    mask = np.zeros(data.shape, dtype=bool)
    mask[10, 10] = True

    local_median_replace_bad_pixels(data, mask, 5)
    assert data[10, 10] == pytest.approx(expected, rel=5e-3)


def test_interpolate_bad_pixels_whole_frame():
    data = _smooth_field(16)
    expected = data[8, 8]
    data[8, 8] = np.nan
    mask = ~np.isfinite(data)
    interpolate_bad_pixels(data, mask)
    assert np.isfinite(data[8, 8])
    assert data[8, 8] == pytest.approx(expected, rel=1e-2)


def test_plus_mask_grows_along_the_cross():
    """Saturated pixels bleed along detector rows and columns, so their mask
    is grown into a plus rather than a square."""
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    grown = plus_mask(mask, radius=1)
    assert grown[3, 3] and grown[2, 3] and grown[4, 3]
    assert grown[3, 2] and grown[3, 4]
    assert not grown[2, 2], "corners are not part of a plus"

    # The default must produce exactly the same plus. Asserting merely that
    # it grew is not enough: scipy reads iterations=0 as "dilate until
    # nothing changes", so a default of 0 fills the entire frame -- which is
    # bigger, and catastrophically wrong for a saturated-pixel mask.
    default = plus_mask(mask)
    assert default.sum() == 5, "the default must be a one-step plus"
    assert not default[0, 0], "a runaway dilation would reach the corners"


def _dark(itime, coadds=1, n=16):
    return Frame(np.zeros((n, n)),
                 {"NAXIS1": n, "NAXIS2": n, "ITIME": itime, "COADDS": coadds,
                  "SAMPMODE": 3, "READS": 1})


def test_find_closest_dark_matches_exposure():
    frame = _dark(30.0)
    darks = [_dark(10.0), _dark(30.0), _dark(60.0)]
    _, got = find_closest_dark(frame, darks)
    assert got is not None and got["ITIME"] == 30.0


def test_find_closest_dark_crops_an_oversized_dark():
    """A full-frame dark can calibrate a subarray exposure.

    The match relaxes step by step down to ITIME alone, and at that point the
    dark is cropped to the frame — the same trimming rule flats use, so one
    set of full-frame darks serves every subarray of the night.
    """
    _, got = find_closest_dark(_dark(30.0, n=16), [_dark(30.0, n=32)])
    assert got is not None
    assert got.shape == (16, 16)


def test_find_closest_dark_returns_none_with_no_darks():
    _, got = find_closest_dark(_dark(30.0), [])
    assert got is None

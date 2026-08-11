"""Centering.

The first test here is a regression test for a real bug: when
``fit_2d_gaussian`` was moved out of ``utils/fitting.py`` into
``reduction/registration.py``, the ``scipy.optimize.least_squares`` import
and the Gaussian model function were both left behind. Every call raised
``NameError``, ``find_center_smooth`` swallowed it in a bare
``except Exception`` and fell back to the integer smoothed peak, and all
registration silently lost its subpixel precision. Nothing failed loudly.
"""

import numpy as np
import pytest

from reduction.registration import (find_center, find_center_gaussian,
                                    find_center_smooth, fit_2d_gaussian)


def gaussian_star(n=128, cy=63.4, cx=64.7, sigma=3.0, amp=1000.0, bg=50.0):
    yy, xx = np.mgrid[:n, :n]
    return amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2)
                          / (2 * sigma ** 2))) + bg


@pytest.mark.parametrize("cy,cx", [(63.4, 64.7), (40.15, 90.85), (64.5, 64.5)])
def test_smooth_peak_is_subpixel(cy, cx):
    """Integer-only answers mean the Gaussian refinement is broken."""
    got = find_center_smooth(gaussian_star(cy=cy, cx=cx))
    assert got[0] == pytest.approx(cy, abs=0.01)
    assert got[1] == pytest.approx(cx, abs=0.01)


def test_fit_2d_gaussian_recovers_parameters():
    img = gaussian_star(cy=63.4, cx=64.7, sigma=3.0, amp=1000.0, bg=50.0)
    amp, x0, y0, offset = fit_2d_gaussian(img, [900.0, 64, 63, 40.0],
                                          fixed_sigma=3.0)
    assert amp == pytest.approx(1000.0, rel=1e-3)
    assert x0 == pytest.approx(64.7, abs=0.01)
    assert y0 == pytest.approx(63.4, abs=0.01)
    assert offset == pytest.approx(50.0, abs=0.5)


def test_gaussian_method_is_subpixel():
    got = find_center(gaussian_star(cy=63.4, cx=64.7), method="gaussian")
    assert got[0] == pytest.approx(63.4, abs=0.05)
    assert got[1] == pytest.approx(64.7, abs=0.05)


def test_gaussian_survives_small_cutout():
    """A frame smaller than the background radius used to raise.

    The background is the median outside a fixed radius of the seed; on a
    64x64 stamp that selection is empty, np.median returns NaN, and the NaN
    reached least_squares as an initial guess.
    """
    got = find_center_gaussian(gaussian_star(n=64, cy=31.4, cx=32.7))
    assert got[0] == pytest.approx(31.4, abs=0.1)
    assert got[1] == pytest.approx(32.7, abs=0.1)


def test_gaussian_on_all_nan_raises_clearly():
    with pytest.raises(ValueError, match="no finite pixels"):
        find_center_gaussian(np.full((64, 64), np.nan))


def test_hot_pixel_does_not_capture_smooth_peak():
    """The median filter exists to stop a single hot pixel winning."""
    img = gaussian_star(cy=63.4, cx=64.7)
    img[10, 10] = 1e6
    got = find_center_smooth(img)
    assert got[0] == pytest.approx(63.4, abs=0.05)
    assert got[1] == pytest.approx(64.7, abs=0.05)

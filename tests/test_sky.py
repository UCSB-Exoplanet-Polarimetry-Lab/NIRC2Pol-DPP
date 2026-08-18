"""Background subtraction, and lamp flat construction."""

import numpy as np
import pytest

from reduction.masters import make_lamp_flats, make_master_darks
from reduction.sky import (subtract_annulus_background, subtract_background,
                           subtract_dither_pairs, subtract_mean_background)
from utils.frame import Frame


def _star_on_pedestal(n=64, pedestal=500.0, amp=1000.0):
    """A Gaussian star on a flat pedestal. Its peak lands half a pixel off
        the grid centre, so tests compare against the input rather than ``amp``.

    Parameters
    ----------
    n : int, optional
        Frame size.
    pedestal : float, optional
        Flat background level.
    amp : float, optional
        Star amplitude.

    Returns
    -------
    img, r : ndarray
        The image and its radius grid.
    """
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2)
    return amp * np.exp(-r ** 2 / (2 * 3.0 ** 2)) + pedestal, r


def test_subtract_mean_background_removes_a_known_pedestal():
    """The pedestal measured in a source-free box is removed; the star stays."""
    img, r = _star_on_pedestal()
    pedestal = 500.0
    out = subtract_mean_background(img, box=(0, 12, 0, 12))   # star-free corner
    assert np.nanmean(out[0:12, 0:12]) == pytest.approx(0.0, abs=1e-9)
    assert out.max() == pytest.approx(img.max() - pedestal, abs=1.0), \
        "the star must survive"


def test_subtract_mean_background_defaults_to_the_whole_image():
    """With no box given the whole image sets the level."""
    img = np.full((8, 8), 3.0)
    assert np.allclose(subtract_mean_background(img), 0.0)


def test_subtract_mean_background_does_not_mutate_its_input():
    """The caller's array is left alone; a new one is returned."""
    img = np.full((8, 8), 3.0)
    subtract_mean_background(img)
    assert np.allclose(img, 3.0), "must return a new array"


def test_subtract_mean_background_handles_a_stack():
    """Each plane of a stack gets its own background."""
    stack = np.stack([np.full((8, 8), 2.0), np.full((8, 8), 5.0)])
    out = subtract_mean_background(stack)
    assert np.allclose(out, 0.0), "each plane gets its own background"


def test_subtract_annulus_background_removes_the_pedestal():
    """The annulus median is removed and the source survives."""
    img, r = _star_on_pedestal()
    out = subtract_annulus_background(img, 20, 30, center=(31.5, 31.5))
    annulus = (r >= 20) & (r <= 30)
    assert np.nanmedian(out[annulus]) == pytest.approx(0.0, abs=1e-9)
    assert out.max() == pytest.approx(img.max() - 500.0, abs=1.0)


def test_subtract_annulus_background_uses_the_annulus_not_the_whole_frame():
    """The annulus has to be what is measured.

    On a flat pedestal the annulus median and the global median agree, so
    this uses a background rising with radius and a *small* annulus. An
    annulus at 20-30 px would not do: on a 64 px frame its median radius is
    almost exactly the frame's, so both estimators return the same number
    and the distinction stays invisible.
    """
    n = 64
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2)
    img = 100.0 + 8.0 * r                      # rises outward
    annulus = (r >= 5) & (r <= 10)

    # the two estimators must actually disagree, or this proves nothing
    assert abs(np.nanmedian(img) - np.nanmedian(img[annulus])) > 100.0

    out = subtract_annulus_background(img, 5, 10, center=(31.5, 31.5))
    assert np.nanmedian(out[annulus]) == pytest.approx(0.0, abs=1e-9)


def test_subtract_annulus_background_finds_the_star_itself():
    """With no centre given it locates the source by quantile peak."""
    img, _ = _star_on_pedestal()
    out = subtract_annulus_background(img, 20, 30)
    assert abs(np.nanmedian(out)) < 1.0


def test_subtract_dither_pairs_cancels_a_common_pedestal():
    """A pedestal common to both dither positions cancels exactly."""
    pedestal = np.random.default_rng(0).normal(800, 5, size=(16, 16))
    a = Frame(pedestal + 10.0, {"FILENAME": "a.fits"})
    b = Frame(pedestal.copy(), {"FILENAME": "b.fits"})
    out = subtract_dither_pairs([a], [b])
    assert len(out) == 1
    assert np.allclose(out[0].data, 10.0)
    assert out[0]["DITHSUB"] == "b.fits"


def test_subtract_dither_pairs_pairs_up_to_the_shorter_list():
    """Mismatched lists pair up to the shorter one rather than raising."""
    frames = [Frame(np.zeros((4, 4)), {"FILENAME": f"{i}.fits"})
              for i in range(3)]
    assert len(subtract_dither_pairs(frames, frames[:2])) == 2


def _lamp_frames(n=5, level=5000.0, itime=30.0):
    """Lamp-on dome flats at a uniform level.

    Returns
    -------
    list of Frame
        The synthetic flats.
    """
    return [Frame(np.full((16, 16), level),
                  {"FILTER": "Kp + Wollaston", "FWINAME": "Kp", "NAXIS1": 16,
                   "NAXIS2": 16, "ITIME": itime, "COADDS": 1, "SAMPMODE": 3,
                   "READS": 1})
            for _ in range(n)]


def _dark_frames(n=5, level=100.0, itime=30.0):
    """Darks matching the lamp flats' exposure settings.

    Returns
    -------
    list of Frame
        The synthetic darks.
    """
    return [Frame(np.full((16, 16), level),
                  {"FILTER": "Kp + Wollaston", "FWINAME": "Kp", "NAXIS1": 16,
                   "NAXIS2": 16, "ITIME": itime, "COADDS": 1, "SAMPMODE": 3,
                   "READS": 1})
            for _ in range(n)]


def test_make_lamp_flats_subtracts_the_dark_and_tags_lamp():
    """The branch this project actually uses: lamp-on frames plus darks.

    The tag matters beyond bookkeeping — find_closest_flat refuses a flat
    whose type is not the one the band requires, so a lamp flat that came out
    tagged anything else would be rejected for JHK data.
    """
    darks, _ = make_master_darks(_dark_frames())
    flats, _ = make_lamp_flats(_lamp_frames(), darks)
    assert flats, "no lamp flat was built"
    flat = list(flats.values())[0]
    assert flat["FLATTYPE"] == "LAMP"
    # (5000 - 100) normalized by its own median
    assert np.allclose(flat.data, 1.0)


def test_make_lamp_flats_tags_nodark_when_no_dark_matches():
    """flat_sort_key ranks NODARK below dark-subtracted flats, which only
    works if the tag is actually set."""
    flats, _ = make_lamp_flats(_lamp_frames(), [])
    flat = list(flats.values())[0]
    assert flat["FLATTYPE"] == "LAMP+NODARK"


def test_make_lamp_flats_needs_enough_frames():
    """Too few frames yields no master rather than a noisy one."""
    assert make_lamp_flats(_lamp_frames(n=2), [])[0] == {}


# --- the dispatch, now beside the subtractions it dispatches to ------------

def test_none_returns_the_input_untouched():
    """How a caller says the omission is deliberate."""
    data = np.arange(16, dtype=float).reshape(4, 4)
    assert subtract_background(data, None) is data


def test_dither_is_a_no_op_per_beam():
    """Dither pairs are differenced at frame level by subtract_dither_pairs,
    before the beams are cut out, so there is nothing left to do here."""
    data = np.ones((4, 4))
    assert subtract_background(data, "dither") is data


def test_mean_box_and_annulus_reach_the_right_subtraction():
    data = np.ones((20, 20)) * 5.0
    boxed = subtract_background(data, "mean_box", box=(0, 20, 0, 20))
    assert np.allclose(boxed, 0.0), "a flat pedestal should subtract to zero"

    ringed = subtract_background(data, "annulus", annulus=(4, 8))
    assert np.allclose(ringed, 0.0)


@pytest.mark.parametrize("method, kwargs", [
    ("mean_box", {}),
    ("annulus", {}),
])
def test_a_method_without_its_parameters_refuses(method, kwargs):
    """Silently skipping the subtraction would leave the pedestal in the
    data with nothing to show for it."""
    with pytest.raises(ValueError, match=method):
        subtract_background(np.ones((8, 8)), method, **kwargs)


def test_an_unknown_method_refuses():
    with pytest.raises(ValueError, match="Unknown background method"):
        subtract_background(np.ones((8, 8)), "sky_hook")

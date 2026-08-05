"""Sky / dither / background subtraction, as its own pipeline stage.

Runs on pre-processed (dark-subtracted, flat-divided) frames. Two options:

Options (SPIE Sec. 3.2):

- :func:`subtract_annulus_background` — annulus median around the star,
  usually sufficient for J / H / Kp without dedicated sky frames.
- :func:`subtract_dither_pairs` — pairwise subtraction between dither
  positions, needed in L' where the thermal background is rapidly varying
  and spatially structured.
- :func:`subtract_sky_frames` — subtract a matched master sky frame
  (built from sky flats).
- :func:`subtract_mean_background` — subtract the mean level of an empty
  box region, per beam.
"""

from __future__ import annotations

import logging

import numpy as np

from utils.imutils import argquantile, make_annulus_mask
from .calibrate import find_closest_sky

log = logging.getLogger(__name__)


def subtract_annulus_background(data, r_in, r_out, center=None):
    """Subtract the median level of an annulus around the star (located by
    quantile peak if ``center = (cy, cx)`` is not given). Works on a 2D
    image or each plane of a stack. Returns a new array."""
    data = np.asarray(data, dtype=float).copy()

    def _sub(img):
        c = center if center is not None else argquantile(img, 0.9999)
        annulus = make_annulus_mask(img.shape, r_in, r_out, center=c)
        img -= np.nanmedian(img[annulus])

    if data.ndim == 2:
        _sub(data)
    else:
        for plane in data.reshape(-1, *data.shape[-2:]):
            _sub(plane)
    return data


def subtract_dither_pairs(frames_a, frames_b):
    """Pairwise dither subtraction: subtract each frame at dither position B
    from its counterpart at position A (matched by order). Returns new
    frames from position A with the B background removed."""
    if len(frames_a) != len(frames_b):
        log.warning("Dither position lists differ in length (%d vs %d); "
                    "pairing up to the shorter one",
                    len(frames_a), len(frames_b))

    out = []
    for fa, fb in zip(frames_a, frames_b):
        result = fa.copy()
        result.data = fa.data - fb.data
        result["DITHSUB"] = fb.get("FILENAME", "")
        out.append(result)
    return out


def subtract_sky_frames(frames, master_skies):
    """Subtract the best-matching master sky from each frame (in place on
    copies). Frames with no matching sky are passed through unchanged, with
    the SKYSUB header keyword recording what happened."""
    out = []
    for frame in frames:
        result = frame.copy()
        result["SKYSUB"] = False
        _, matched_sky = find_closest_sky(frame, master_skies)
        if matched_sky is not None:
            result.data -= matched_sky.data
            result["SKYSUB"] = True
        out.append(result)
    return out


def subtract_mean_background(data, box=None):
    """Subtract the mean of a background box from an image (or from each
    plane of a stack of images).

    ``box`` is ``(ylow, yhigh, xlow, xhigh)``; defaults to the whole image.
    Returns a new array.
    """
    data = np.asarray(data, dtype=float).copy()

    def _sub(img):
        if box is None:
            img -= np.nanmean(img)
        else:
            ylow, yhigh, xlow, xhigh = box
            img -= np.nanmean(img[ylow:yhigh, xlow:xhigh])

    if data.ndim == 2:
        _sub(data)
    else:
        for plane in data.reshape(-1, *data.shape[-2:]):
            _sub(plane)
    return data

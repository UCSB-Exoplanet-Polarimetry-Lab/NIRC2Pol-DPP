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
        image or each plane of a stack. Returns a new array.

    Parameters
    ----------
    data : array_like
        2D image or a stack; each plane is treated separately.
    r_in, r_out : float
        Annulus radii in pixels.
    center : tuple of float, optional
        ``(cy, cx)``; located by quantile peak when not given.

    Returns
    -------
    ndarray
        A new array with the annulus median removed. The input is unchanged.

    Notes
    -----
    Subtracts one scalar, so it assumes the background is flat across the
    frame. That holds in JHK but not at L' or M, where the thermal pedestal is
    large and structured -- see
    ``instruments.nirc2.check_background_choice``.
    """
    data = np.asarray(data, dtype=float).copy()

    def _sub(img):
        """Subtract this plane's background in place.

        Subtract this plane background in place.
        """
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
        frames from position A with the B background removed.

    Parameters
    ----------
    frames_a, frames_b : list of Frame
        Frames at the two dither positions, paired by order.

    Returns
    -------
    list of Frame
        New frames from position A with B subtracted, each recording the
        subtracted file in ``DITHSUB``. Warns and pairs up to the shorter list
        if the lengths differ.

    Notes
    -----
    The cleanest background removal at L' and M: the pedestal is measured
    through the same optics moments apart, so its structure cancels rather
    than being approximated by a single number.
    """
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
        the SKYSUB header keyword recording what happened.

    Parameters
    ----------
    frames : iterable of Frame
        Science frames.
    master_skies : list of Frame
        Candidate master skies.

    Returns
    -------
    list of Frame
        Copies with the best-matching sky subtracted where one was found.
        ``SKYSUB`` records what happened for each frame, so a frame that was
        passed through unchanged is identifiable afterwards.
    """
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

    Parameters
    ----------
    data : array_like
        2D image or a stack; each plane is treated separately.
    box : tuple of int, optional
        ``(ylow, yhigh, xlow, xhigh)`` region to average; the whole image by
        default.

    Returns
    -------
    ndarray
        A new array with the mean removed. The input is unchanged.
    """
    data = np.asarray(data, dtype=float).copy()

    def _sub(img):
        """Subtract this plane background in place."""
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

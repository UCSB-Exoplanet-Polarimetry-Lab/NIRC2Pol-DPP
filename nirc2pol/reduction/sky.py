"""Sky / dither / background-offset subtraction.

Three paths corresponding to common modes:
- :func:`subtract_sky_flat` — pre-built Sky Flats (optional for JHK).
- :func:`subtract_dither_sky` — build a sky frame from dithered exposures
  and subtract it.
- :func:`subtract_box_mean_background` — subtract the mean of a fixed
  detector box from each beam. Used in the working AB Aur reduction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nirc2pol.instruments.base import PolarimetryData


def subtract_sky_flat(dataset: PolarimetryData, sky_flat_path: Path) -> None:
    """Subtract a matching Sky Flat from every science frame.

    Updates ``dataset.data`` and ``dataset.variance`` in place. Matching key
    is the filter (and camera) — raises :class:`NoMatchingCalibrationError`
    if none is found.
    """
    raise NotImplementedError


def subtract_dither_sky(dataset: PolarimetryData) -> None:
    """Construct a sky frame from the dithered science frames and subtract it.

    Median-combine the science frames with the source masked, then subtract
    that median from each input frame.
    """
    raise NotImplementedError


def subtract_box_mean_background(
    dataset: PolarimetryData,
    box: tuple[int, int, int, int] = (25, 350, 50, 400),
) -> None:
    """Subtract the mean of a fixed detector box from each beam.

    Translation of the notebook's ``sub_mean_bkg``. The box is given in the
    chopped per-beam coordinate system as ``(ylow, yhigh, xlow, xhigh)``;
    a separate mean is measured and subtracted per (frame, beam).
    """
    ylow, yhigh, xlow, xhigh = box
    box_means = np.mean(dataset.data[..., ylow:yhigh, xlow:xhigh], axis=(-2, -1))
    dataset.data -= box_means[..., None, None]
    # TODO: propagate variance using the variance of the box estimate.

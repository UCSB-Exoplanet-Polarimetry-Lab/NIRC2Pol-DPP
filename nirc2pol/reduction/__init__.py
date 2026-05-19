"""Pre-polarimetry reduction steps (NumPy): calibration files, dark/flat, sky, registration."""
from __future__ import annotations

from nirc2pol.reduction.dark_flat import (
    divide_flat,
    interpolate_bad_pixels,
    mask_bad_pixels,
    subtract_dark,
)
from nirc2pol.reduction.exposure import normalize_by_exposure
from nirc2pol.reduction.registration import register_frames
from nirc2pol.reduction.sky import subtract_sky

__all__ = [
    "subtract_dark",
    "divide_flat",
    "interpolate_bad_pixels",
    "mask_bad_pixels",
    "subtract_sky",
    "register_frames",
    "normalize_by_exposure",
]

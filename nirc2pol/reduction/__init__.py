"""Pre-polarimetry reduction (NumPy): calibration files, dark/flat, sky, registration."""
from __future__ import annotations

from nirc2pol.reduction.dark_flat import (
    divide_flat,
    interpolate_bad_pixels,
    mask_bad_pixels,
    subtract_dark,
)
from nirc2pol.reduction.registration import register_frames
from nirc2pol.reduction.sky import (
    subtract_box_mean_background,
    subtract_dither_sky,
    subtract_sky_flat,
)

__all__ = [
    "subtract_dark",
    "divide_flat",
    "interpolate_bad_pixels",
    "mask_bad_pixels",
    "subtract_sky_flat",
    "subtract_dither_sky",
    "subtract_box_mean_background",
    "register_frames",
]

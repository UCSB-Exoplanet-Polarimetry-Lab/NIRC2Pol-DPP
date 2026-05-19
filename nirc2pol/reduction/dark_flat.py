"""Apply master darks/flats to science frames; propagate variance.

Reference: AIR.jl ``mass_reduce.jl``. Matches each frame by (ITIME, COADDS,
readout mode) and raises :class:`NoMatchingCalibrationError` rather than
silently substituting.
"""
from __future__ import annotations

from pathlib import Path

from nirc2pol.instruments.base import PolarimetryData


def subtract_dark(dataset: PolarimetryData, dark_library_path: Path) -> None:
    """Subtract the matching master dark from every frame.

    Updates ``dataset.data`` and ``dataset.variance`` in place. Raises
    :class:`NoMatchingCalibrationError` if any frame lacks a matching dark.
    """
    raise NotImplementedError


def divide_flat(dataset: PolarimetryData, flat_library_path: Path) -> None:
    """Divide each frame by the matching master flat; propagate variance."""
    raise NotImplementedError


def interpolate_bad_pixels(dataset: PolarimetryData, mask_path: Path) -> None:
    """Replace bad pixels with interpolated neighbour values."""
    raise NotImplementedError


def mask_bad_pixels(dataset: PolarimetryData, mask_path: Path) -> None:
    """Set bad pixels to NaN and inflate their variance to infinity."""
    raise NotImplementedError

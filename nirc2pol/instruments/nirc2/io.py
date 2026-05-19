"""NIRC2 FITS reading — raw files → data, variance, headers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_nirc2_files(filelist: list[Path]) -> dict[str, Any]:
    """Read a list of NIRC2 FITS files into a dict of observational arrays.

    Parameters
    ----------
    filelist
        Paths to NIRC2 FITS frames.

    Returns
    -------
    dict
        Keys match the attribute names on :class:`PolarimetryData`:
        ``data``, ``variance``, ``parangs``, ``hwp_angles``, ``elevations``,
        ``rotator_angles``, ``wavelengths``, ``exposure_times``, ``coadds``,
        ``filenames``, ``prihdrs``, ``exthdrs``, ``wcs``, ``pixel_scale``.
    """
    raise NotImplementedError


def read_single_frame(path: Path) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """Read one FITS file and return ``(data, variance, prihdr, exthdr)``.

    Initial variance is derived from the detector noise model (read noise,
    gain, Poisson term).
    """
    raise NotImplementedError


def estimate_initial_variance(data: np.ndarray, gain: float, read_noise: float) -> np.ndarray:
    """Initial per-pixel variance from a Poisson + Gaussian noise model."""
    raise NotImplementedError

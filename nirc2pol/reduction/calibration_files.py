"""Build master calibration frames (darks, flats, bad-pixel masks).

Reference: AIR.jl ``make_masters.jl``. Sorts raw frames by type using
normalized headers from ``instruments/nirc2/headers.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def build_master_dark(
    dark_frames: np.ndarray,
    itime_s: float,
    coadds: int,
    readout_mode: str,
    sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sigma-clipped stack of darks with matching (ITIME, COADDS, readout mode).

    Returns
    -------
    master, master_variance, header_metadata
        ``master`` is the median frame; ``master_variance`` is the per-pixel
        variance from sigma-clipped statistics; ``header_metadata`` records
        the grouping keys for matching downstream.
    """
    raise NotImplementedError


def build_master_flat(
    flat_frames: np.ndarray,
    master_dark: np.ndarray,
    filter_name: str,
    camera: str,
    sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Dark-subtract, normalize, sigma-clip a stack of flats."""
    raise NotImplementedError


def build_bad_pixel_mask(
    master_dark_variance: np.ndarray,
    master_flat: np.ndarray,
    hot_sigma: float = 5.0,
    dead_threshold: float = 0.5,
) -> np.ndarray:
    """Boolean mask: hot pixels from dark variance, dead pixels from flat response."""
    raise NotImplementedError


def write_master(path: Path, master: np.ndarray, variance: np.ndarray, metadata: dict[str, Any]) -> None:
    """Write a master frame plus provenance to a FITS file."""
    raise NotImplementedError

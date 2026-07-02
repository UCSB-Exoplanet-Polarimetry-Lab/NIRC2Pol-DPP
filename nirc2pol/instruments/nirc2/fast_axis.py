"""NIRC2 HWP fast-axis calibration.

The HWP fast-axis offset is measured from dedicated calibration flats taken
through the polarimetry optics. The output is a single scalar (deg) that
parametrizes the HWP Mueller matrix downstream.

Inputs: Darks, Flats, Fast Axis Calibration Flats.
Output: Fast Axis Value (deg).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def calibrate_fast_axis(
    fac_flat_paths: list[Path],
    master_dark: np.ndarray,
    master_flat: np.ndarray,
) -> tuple[float, float]:
    """Estimate the HWP fast-axis angle from a set of calibration flats.

    Parameters
    ----------
    fac_flat_paths
        Paths to the Fast Axis Calibration Flat FITS files.
    master_dark, master_flat
        Master calibration frames matching the FAC flats' exposure settings.

    Returns
    -------
    fast_axis_deg, fast_axis_uncertainty_deg
        Best-fit fast-axis offset (deg) and 1-sigma uncertainty.
    """
    raise NotImplementedError


def fit_fast_axis_from_modulation(intensity_per_hwp: np.ndarray, hwp_angles_deg: np.ndarray) -> tuple[float, float]:
    """Fit a sinusoid to ``I(HWP)`` to extract the fast-axis phase offset."""
    raise NotImplementedError

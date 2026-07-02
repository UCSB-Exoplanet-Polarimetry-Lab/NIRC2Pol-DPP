"""Polarimetric efficiency as a function of wavelength."""
from __future__ import annotations

import numpy as np

from nirc2pol.instruments.base import PolarimetryData


def polarimetric_efficiency(dataset: PolarimetryData) -> np.ndarray:
    """Fractional polarization recovery vs wavelength.

    Returns
    -------
    np.ndarray, shape (N_wavelengths,)
        Efficiency in ``[0, 1]`` (1 = perfect recovery).
    """
    raise NotImplementedError

"""Derived polarimetric products: polarized intensity, DoLP, AoLP.

All three are computed from the median Stokes cube
(``dataset.output["median_stokes_cube"]``).
"""
from __future__ import annotations

import numpy as np

from nirc2pol.instruments.base import PolarimetryData


def compute_polarized_intensity(dataset: PolarimetryData) -> None:
    """PI = sqrt(Q^2 + U^2). Writes ``dataset.output["polarized_intensity"]``."""
    _, q, u = dataset.output["median_stokes_cube"]
    dataset.output["polarized_intensity"] = np.sqrt(q * q + u * u)


def compute_degree_of_linear_polarization(dataset: PolarimetryData) -> None:
    """DoLP = sqrt(Q^2 + U^2) / I, NaN-safe near zero intensity.

    Writes ``dataset.output["dolp"]``.
    """
    i, q, u = dataset.output["median_stokes_cube"]
    pi = np.sqrt(q * q + u * u)
    dataset.output["dolp"] = np.where(np.abs(i) > 0, pi / i, np.nan)


def compute_angle_of_linear_polarization(dataset: PolarimetryData) -> None:
    """AoLP = 0.5 * atan2(U, Q), deg east of north.

    Writes ``dataset.output["aolp"]``. See :doc:`/ARCHITECTURE` for the
    sign convention.
    """
    _, q, u = dataset.output["median_stokes_cube"]
    dataset.output["aolp"] = 0.5 * np.degrees(np.arctan2(u, q))

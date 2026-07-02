"""Polarimetry algorithms (JAX): double differencing, Stokes cubes, Mueller interface, derived products."""
from __future__ import annotations

from nirc2pol.polarimetry.derived_products import (
    compute_angle_of_linear_polarization,
    compute_degree_of_linear_polarization,
    compute_polarized_intensity,
)
from nirc2pol.polarimetry.double_diff import double_difference
from nirc2pol.polarimetry.mueller import correct_instrumental_polarization
from nirc2pol.polarimetry.radial_stokes import compute_radial_stokes
from nirc2pol.polarimetry.stokes import (
    compute_median_stokes_cube,
    compute_stokes_cubes_per_hwp_cycle,
)

__all__ = [
    "double_difference",
    "correct_instrumental_polarization",
    "compute_stokes_cubes_per_hwp_cycle",
    "compute_median_stokes_cube",
    "compute_polarized_intensity",
    "compute_degree_of_linear_polarization",
    "compute_angle_of_linear_polarization",
    "compute_radial_stokes",
]

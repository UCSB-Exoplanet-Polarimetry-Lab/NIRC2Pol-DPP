"""Radial Stokes (Qφ, Uφ) — the pipeline's PDI data product.

The stellar PSF is unpolarized, so it drops out of Qφ naturally. This is
the pipeline's PSF-subtraction equivalent for polarized signal — no KLIP,
no ADI.

Reference: Schmid et al. (2006), "Limb polarization of Uranus and Neptune".
"""
from __future__ import annotations

import jax.numpy as jnp

from nirc2pol.instruments.base import PolarimetryData


def compute_radial_stokes(dataset: PolarimetryData) -> None:
    """Compute (Qφ, Uφ) from (Q, U) and the star center.

    Requires ``dataset.output["star_center"]`` (populated by
    :func:`register_frames`). Writes
    ``dataset.output["radial_stokes_cube"]`` of shape
    ``(N_sets, 2, ny, nx)`` — first axis is ``(Qφ, Uφ)``.

    Sign convention: Qφ > 0 for azimuthal polarization (disk-like),
    Qφ < 0 for radial polarization.
    """
    raise NotImplementedError


def radial_stokes_single(q: jnp.ndarray, u: jnp.ndarray, center_xy: tuple[float, float]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """(Qφ, Uφ) for a single (Q, U) pair around the given center."""
    raise NotImplementedError

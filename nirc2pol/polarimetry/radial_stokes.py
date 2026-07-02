"""Radial Stokes (Qφ, Uφ) — the pipeline's PDI data product.

The stellar PSF is unpolarized, so it drops out of Qφ naturally. This is
the pipeline's PSF-subtraction equivalent for polarized signal — no KLIP,
no ADI.

Reference: Schmid et al. (2006), "Limb polarization of Uranus and Neptune".
"""
from __future__ import annotations

import numpy as np

from nirc2pol.instruments.base import PolarimetryData


def compute_radial_stokes(dataset: PolarimetryData) -> None:
    """Compute (Qφ, Uφ) from the median Stokes cube and the star center.

    Requires ``dataset.output["star_center"]`` (populated by
    :func:`register_frames`) and ``dataset.output["median_stokes_cube"]``.
    Writes ``dataset.output["radial_stokes_cube"]`` of shape ``(2, ny, nx)`` —
    first axis is ``(Qφ, Uφ)``.

    Sign convention: Qφ > 0 for azimuthal polarization (disk-like),
    Qφ < 0 for radial polarization.
    """
    median = dataset.output["median_stokes_cube"]  # (3, ny, nx) = (I, Q, U)
    q, u = median[1], median[2]
    cx, cy = dataset.output["star_center"][0]
    qphi, uphi = radial_stokes_single(q, u, (cx, cy))
    dataset.output["radial_stokes_cube"] = np.stack([qphi, uphi])


def radial_stokes_single(
    q: np.ndarray,
    u: np.ndarray,
    center_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """(Qφ, Uφ) for a single (Q, U) pair around the given center.

    Uses ``φ = atan2(y - cy, x - cx)`` and the standard Schmid (2006)
    rotation:
    ``Qφ = Q cos(2φ) + U sin(2φ)``, ``Uφ = -Q sin(2φ) + U cos(2φ)``.
    """
    cx, cy = center_xy
    ys, xs = np.indices(q.shape)
    phi = np.arctan2(ys - cy, xs - cx)
    c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)
    qphi = q * c2 + u * s2
    uphi = -q * s2 + u * c2
    return qphi, uphi

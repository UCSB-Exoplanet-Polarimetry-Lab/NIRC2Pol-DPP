"""Mueller-matrix model interface.

Applies per-frame Mueller inversion and exposes the model parameters to
calibration fitters. The science layer never imports a concrete instrument —
everything goes through ``dataset.get_mueller_matrix_sequence()`` etc.

In the current AB Aur port, Mueller demodulation is folded into
:func:`nirc2pol.polarimetry.stokes.compute_stokes_cubes_per_hwp_cycle`
rather than being applied as a separate prior step on raw data. The two
functions below are kept as stubs for future use (e.g. JAX-autodiff
fitting in :mod:`nirc2pol.polarimetry.mueller_fit`).
"""
from __future__ import annotations

import numpy as np

from nirc2pol.instruments.base import PolarimetryData


def correct_instrumental_polarization(dataset: PolarimetryData) -> None:
    """Apply per-frame Mueller inversion to ``dataset.data`` (not used by AB Aur port)."""
    raise NotImplementedError


def propagate_variance(variance: np.ndarray, mueller_inverse: np.ndarray) -> np.ndarray:
    """Propagate per-pixel variance through ``M^-1``.

    For ``y = M^-1 x`` with diagonal pixel variance ``Var(x)``,
    ``Var(y) = (M^-1) diag(Var(x)) (M^-1)^T``.
    """
    raise NotImplementedError

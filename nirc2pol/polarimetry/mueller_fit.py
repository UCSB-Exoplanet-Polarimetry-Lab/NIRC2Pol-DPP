"""Fit free Mueller-model parameters via the abstract dataset hook (JAX autodiff)."""
from __future__ import annotations

from collections.abc import Callable

from nirc2pol.instruments.base import PolarimetryData


def fit_mueller_parameters(
    dataset: PolarimetryData,
    expected_stokes: dict[str, float],
    free_param_names: list[str],
    loss_fn: Callable[..., float] | None = None,
    max_iter: int = 200,
) -> dict[str, float]:
    """Fit named Mueller free parameters to match expected Stokes for a standard.

    Calls ``dataset.set_mueller_parameters`` / ``get_mueller_matrix_sequence``
    only — never imports a concrete instrument.
    """
    raise NotImplementedError

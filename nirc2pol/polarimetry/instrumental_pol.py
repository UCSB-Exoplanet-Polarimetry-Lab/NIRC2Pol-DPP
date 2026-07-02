"""Measure residual instrumental polarization from unpolarized standards."""
from __future__ import annotations

from nirc2pol.instruments.base import PolarimetryData


def measure_instrumental_pol(dataset: PolarimetryData) -> dict[str, float]:
    """Fit the residual (Q/I, U/I) of an unpolarized standard.

    Diagnostic only — does not modify dataset Mueller parameters.
    """
    raise NotImplementedError

"""HWP angle → Stokes weights, plus PA conventions.

This module is the **single source of truth** for mapping a commanded HWP
angle to its Q/U Stokes weights. All other modules use these functions —
never re-derive the convention inline.
"""
from __future__ import annotations


def hwp_to_stokes_weight(
    hwp_angle_deg: float,
    parang_deg: float,
    rotator_angle_deg: float,
) -> dict[str, float]:
    """Map commanded HWP angle to Q/U Stokes weights.

    Parameters
    ----------
    hwp_angle_deg
        Commanded HWP angle, deg. Positive = counterclockwise into the beam.
    parang_deg
        Parallactic angle, deg.
    rotator_angle_deg
        Instrument rotator angle, deg.

    Returns
    -------
    dict[str, float]
        ``{"q_weight": ..., "u_weight": ...}`` for the linear combination
        ``Q = Σ q_weight_i · I_i``, ``U = Σ u_weight_i · I_i`` over a HWP set.

    Notes
    -----
    Must be validated against a polarized standard of known PA before any
    science run. Cite the specific NIRC2 documentation in the implementation
    docstring.
    """
    raise NotImplementedError


def sky_pa_to_detector_pa(sky_pa_deg: float, parang_deg: float, rotator_angle_deg: float) -> float:
    """Convert sky PA (deg E of N) to detector PA, deg."""
    raise NotImplementedError


def detector_pa_to_sky_pa(detector_pa_deg: float, parang_deg: float, rotator_angle_deg: float) -> float:
    """Convert detector PA to sky PA (deg E of N)."""
    raise NotImplementedError

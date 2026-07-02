"""NIRC2 Mueller-model math.

Simplified ``working-ab-aur2`` model: a single Stokes-rotation by

    angle_deg = -2*PA + 2*EL + 2*ROT + 4*offset

acting on (Q, U). Higher-fidelity decomposition into telescope (M1+M2+M3) ·
AO bench · HWP retarder is a future upgrade — keep this signature so the
upgrade is drop-in.

References: Wiktorowicz & Matthews (2008); van Holstein et al. (2020).
"""
from __future__ import annotations

import numpy as np


def full_system_mueller(
    parang_deg: float,
    elevation_deg: float,
    rotator_angle_deg: float,
    hwp_angle_deg: float,
    wavelength_m: float,
    retardance_rad: float,
    fast_axis_offset_deg: float,
) -> np.ndarray:
    """(4, 4) Mueller matrix for the full optical path of one frame.

    Returns the forward Mueller ``M`` such that ``S_inst = M S_sky``. The
    inverse (used by :func:`correct_instrumental_polarization`) maps
    measured Stokes back to sky Stokes.
    """
    angle_rad = np.deg2rad(
        -2.0 * parang_deg
        + 2.0 * elevation_deg
        + 2.0 * rotator_angle_deg
        + 4.0 * fast_axis_offset_deg
    )
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    # Forward Mueller M maps sky → instrument; M is a CCW rotation by
    # ``angle_rad`` in the (Q, U) plane. Inverse M^T (= CW rotation)
    # is what corrects measured (Q_inst, U_inst) back to sky.
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

"""Stokes-cube assembly: per-HWP-cycle cubes and the median Stokes cube.

Per the new flow, ``Build Stokes Cubes`` consumes aligned data and the
Mueller-model parameters; it produces a Stokes cube per HWP cycle and a
median cube across cycles. Mueller demodulation and PA derotation are
folded into this step (not a separate Mueller-correction pass on raw data).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate as scipy_rotate, shift as scipy_shift

from nirc2pol.exceptions import IncompleteHwpSequenceError
from nirc2pol.instruments.base import PolarimetryData
from nirc2pol.polarimetry.double_diff import double_difference


def compute_stokes_cubes_per_hwp_cycle(dataset: PolarimetryData) -> None:
    """Build one ``(3, ny, nx)`` ``[I, Q, U]`` sky-aligned cube per HWP cycle.

    For each cycle from ``dataset.get_hwp_cycles()``:
    1. ``double_difference`` → instrument-frame ``(I, Q_inst, U_inst)``
    2. Mueller demodulation via the cycle-averaged Mueller matrix → sky-frame ``(Q, U)``
    3. Spatial derotation by the cycle-averaged ``sky_pa_deg`` → sky-aligned

    Writes ``dataset.output["stokes_cubes_per_cycle"]`` with shape
    ``(N_cycles, 3, ny, nx)``.
    """
    cycles = dataset.get_hwp_cycles()
    if not cycles:
        raise IncompleteHwpSequenceError("No complete HWP cycles found.")
    center = tuple(dataset.output["star_center"][0])  # (x, y)

    cubes = []
    for cycle in cycles:
        intensity, q_inst, u_inst = double_difference(dataset, cycle)

        # Cycle-averaged Mueller. Forward Mueller is sky → instrument; the
        # inverse (instrument → sky) is the transpose because it's a rotation.
        m = np.mean([dataset.get_mueller_matrix(i) for i in cycle], axis=0)
        m_inv = m.T
        q_sky = m_inv[1, 1] * q_inst + m_inv[1, 2] * u_inst
        u_sky = m_inv[2, 1] * q_inst + m_inv[2, 2] * u_inst

        pa = float(np.mean(dataset.sky_pa_deg[cycle]))
        i_aligned = _derotate_to_sky(intensity, pa, center)
        q_aligned = _derotate_to_sky(q_sky, pa, center)
        u_aligned = _derotate_to_sky(u_sky, pa, center)

        cubes.append(np.stack([i_aligned, q_aligned, u_aligned]))

    dataset.output["stokes_cubes_per_cycle"] = np.array(cubes)


def compute_median_stokes_cube(dataset: PolarimetryData) -> None:
    """Median of the per-cycle Stokes cubes along the cycle axis.

    Writes ``dataset.output["median_stokes_cube"]`` with shape ``(3, ny, nx)``.
    """
    cubes = dataset.output["stokes_cubes_per_cycle"]
    dataset.output["median_stokes_cube"] = np.nanmedian(cubes, axis=0)


def _derotate_to_sky(
    image: np.ndarray,
    sky_pa_deg: float,
    center_xy: tuple[float, float],
) -> np.ndarray:
    """Rotate ``image`` clockwise by ``sky_pa_deg`` around ``center_xy``.

    Matches pyklip's ``klip.rotate`` semantics (CW positive). Implementation
    is shift → scipy.ndimage.rotate (CCW positive, hence the sign flip) → shift back.
    """
    ny, nx = image.shape
    img_cy, img_cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    cx, cy = center_xy
    dy, dx = img_cy - cy, img_cx - cx

    shifted = scipy_shift(image, (dy, dx), order=3, mode="constant", cval=0.0)
    rotated = scipy_rotate(
        shifted, -sky_pa_deg, reshape=False, order=3, mode="constant", cval=0.0
    )
    return scipy_shift(rotated, (-dy, -dx), order=3, mode="constant", cval=0.0)

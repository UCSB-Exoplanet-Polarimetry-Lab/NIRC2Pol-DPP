"""Single and double differencing for HWP-modulated, dual-beam polarimetry.

A full HWP cycle uses four angles (0, 22.5, 45, 67.5 deg). For a dual-beam
(Wollaston) instrument:

- **Single difference** (within one HWP exposure): ``beam_diff[i] = data[i, 1] - data[i, 0]``
  cancels intensity but keeps Q-like or U-like signal depending on HWP angle.
- **Double difference** (between paired HWP exposures): ``Q_inst = beam_diff[0] - beam_diff[1]``,
  ``U_inst = beam_diff[2] - beam_diff[3]`` cancels residual additive instrument signatures.

The output is per-cycle ``(I, Q_inst, U_inst)`` in the instrument frame.
Mueller correction + sky derotation happen in :mod:`stokes`.
"""
from __future__ import annotations

import numpy as np

from nirc2pol.instruments.base import PolarimetryData


def double_difference(
    dataset: PolarimetryData,
    cycle_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply single + double differencing within one HWP cycle.

    Parameters
    ----------
    dataset
        Reduced, registered dataset (``data`` shape ``(N, 2, ny, nx)``).
    cycle_indices
        Frame indices for one complete HWP cycle. Length 4 for NIRC2.

    Returns
    -------
    intensity, q_inst, u_inst : np.ndarray, each shape (ny, nx)
        Cycle-averaged intensity, and double-differenced (Q, U) in the
        instrument frame.
    """
    frames = dataset.data[cycle_indices]              # (4, 2, ny, nx)
    beam_diffs = frames[:, 1] - frames[:, 0]          # (4, ny, nx)
    beam_sums = frames[:, 0] + frames[:, 1]           # (4, ny, nx)

    q_inst = beam_diffs[0] - beam_diffs[1]
    u_inst = beam_diffs[2] - beam_diffs[3]
    intensity = beam_sums.mean(axis=0)
    return intensity, q_inst, u_inst


def single_difference(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """``(frame_a - frame_b) / 2`` — paired-frame difference."""
    return 0.5 * (frame_a - frame_b)

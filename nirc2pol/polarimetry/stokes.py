"""Stokes-cube assembly: (I, Q, U) from frames grouped by HWP set."""
from __future__ import annotations

from nirc2pol.instruments.base import PolarimetryData


def compute_stokes_cube(dataset: PolarimetryData) -> None:
    """Build ``(N_sets, 3, ny, nx)`` Stokes cube ``[I, Q, U]``.

    Groups frames by HWP set using ``hwp_angles``; uses Stokes weights from
    :func:`nirc2pol.polarimetry.angles.hwp_to_stokes_weight`; propagates
    variance by quadrature; raises :class:`IncompleteHwpSequenceError` on
    partial sets. Writes the result to ``dataset.output["stokes_cube"]``.
    """
    raise NotImplementedError


def group_frames_by_hwp_set(dataset: PolarimetryData) -> list[list[int]]:
    """Return index groups, one per complete HWP set."""
    raise NotImplementedError

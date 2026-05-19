"""NIRC2PolarimetryData — concrete subclass wiring abstract methods to NIRC2 internals."""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from nirc2pol.instruments.base import PolarimetryData


class NIRC2PolarimetryData(PolarimetryData):
    """Keck/NIRC2 polarimetric dataset.

    Parameters
    ----------
    filelist
        FITS paths for a single observing sequence. If provided, :meth:`read_data`
        is invoked from ``__init__``.
    """

    def __init__(self, filelist: list[Path] | None = None) -> None:
        self.output = {}
        self._mueller_params: dict[str, float] = {}
        if filelist is not None:
            self.read_data(filelist)

    def read_data(self, filelist: list[Path]) -> None:
        """Read FITS files via ``io.read_nirc2_files`` and normalize headers."""
        raise NotImplementedError

    def get_mueller_matrix(self, frame_index: int) -> jnp.ndarray:
        """Dispatch to ``mueller/system.py::full_system_mueller`` for one frame."""
        raise NotImplementedError

    def get_mueller_matrix_sequence(self) -> jnp.ndarray:
        """(N, 4, 4) stack from ``mueller/system.py`` across all frames."""
        raise NotImplementedError

    def get_mueller_parameters(self) -> dict[str, float]:
        """Return current free Mueller parameters (HWP retardance, AO bench IP, ...)."""
        raise NotImplementedError

    def set_mueller_parameters(self, params: dict[str, float]) -> None:
        """Update free Mueller parameters in-place."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Write ``output`` cubes plus provenance headers to a NIRC2-style FITS file."""
        raise NotImplementedError

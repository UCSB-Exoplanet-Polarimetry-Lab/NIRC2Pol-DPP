"""Generic FITS read/write helpers shared across instruments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_fits_array(path: Path, ext: int | str = 0) -> tuple[np.ndarray, Any]:
    """Return ``(data, header)`` for one FITS extension."""
    raise NotImplementedError


def write_fits_array(
    path: Path,
    data: np.ndarray,
    header: Any | None = None,
    overwrite: bool = False,
) -> None:
    """Write a single-extension FITS file with optional header."""
    raise NotImplementedError


def write_fits_cube(
    path: Path,
    data: np.ndarray,
    variance: np.ndarray,
    header: Any | None = None,
    overwrite: bool = False,
) -> None:
    """Write a science cube + variance extension to a single FITS file."""
    raise NotImplementedError

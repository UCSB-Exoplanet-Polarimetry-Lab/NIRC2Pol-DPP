"""Abstract PolarimetryData contract — the only instrument symbol the science layer imports.

Subclasses populate the observational attributes in :meth:`read_data` and
implement the Mueller-matrix dispatch methods. Derived quantities (Stokes
cubes, star centers) are stored in :attr:`output`, not as attributes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


class PolarimetryData(ABC):
    """Instrument-agnostic polarimetric dataset.

    Attributes
    ----------
    data : np.ndarray, shape (N, ny, nx)
        Science frames in ADU or e-.
    variance : np.ndarray, shape (N, ny, nx)
        Per-pixel variance in (ADU)^2 or (e-)^2. Propagated through every step.
    wcs : list[astropy.wcs.WCS]
        One per frame.
    parangs : np.ndarray, shape (N,)
        Parallactic angle per frame, deg.
    hwp_angles : np.ndarray, shape (N,)
        Commanded HWP angle per frame, deg.
    elevations : np.ndarray, shape (N,)
        Telescope elevation per frame, deg.
    rotator_angles : np.ndarray, shape (N,)
        Instrument rotator angle per frame, deg.
    wavelengths : np.ndarray, shape (N,)
        Effective wavelength per frame, m.
    exposure_times : np.ndarray, shape (N,)
        Per-frame integration time, s.
    coadds : np.ndarray, shape (N,)
        Per-frame coadd count.
    filenames : list[pathlib.Path]
        Source FITS paths.
    prihdrs, exthdrs : list
        Original FITS headers, kept for provenance.
    pixel_scale : float
        Detector plate scale, arcsec/pixel.
    output : dict[str, np.ndarray]
        Derived products. Standard keys: ``"preprocessed"``, ``"star_center"``,
        ``"stokes_cube"``, ``"radial_stokes_cube"``.
    """

    data: np.ndarray
    variance: np.ndarray
    wcs: list[Any]
    parangs: np.ndarray
    hwp_angles: np.ndarray
    elevations: np.ndarray
    rotator_angles: np.ndarray
    wavelengths: np.ndarray
    exposure_times: np.ndarray
    coadds: np.ndarray
    filenames: list[Path]
    prihdrs: list[Any]
    exthdrs: list[Any]
    pixel_scale: float
    output: dict[str, np.ndarray]

    @abstractmethod
    def read_data(self, filelist: list[Path]) -> None:
        """Populate observational attributes from a list of FITS files."""

    @abstractmethod
    def get_mueller_matrix(self, frame_index: int) -> jnp.ndarray:
        """Return the (4, 4) Mueller matrix for a single frame."""

    @abstractmethod
    def get_mueller_matrix_sequence(self) -> jnp.ndarray:
        """Return the (N, 4, 4) stack of per-frame Mueller matrices."""

    @abstractmethod
    def get_mueller_parameters(self) -> dict[str, float]:
        """Return the current free Mueller-model parameters."""

    @abstractmethod
    def set_mueller_parameters(self, params: dict[str, float]) -> None:
        """Update free Mueller-model parameters (used by calibration/mueller_fit)."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write outputs in an instrument-appropriate FITS layout."""

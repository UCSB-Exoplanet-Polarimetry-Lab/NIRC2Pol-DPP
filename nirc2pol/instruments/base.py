"""Abstract PolarimetryData contract — the only instrument symbol the science layer imports.

Subclasses populate the observational attributes in :meth:`read_data` and
implement the Mueller-matrix dispatch and HWP-cycle-matching methods.
Derived quantities (Stokes cubes, star centers) live in :attr:`output`,
not as attributes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class PolarimetryData(ABC):
    """Instrument-agnostic polarimetric dataset.

    Attributes
    ----------
    data : np.ndarray, shape (N, n_beams, ny, nx)
        Science frames in ADU or e-. ``n_beams`` is 1 for single-beam
        instruments (no beam-splitter / Wollaston) and 2 for dual-beam
        instruments (e.g. NIRC2 with Wollaston prism).
    variance : np.ndarray, shape (N, n_beams, ny, nx)
        Per-pixel variance, same shape as ``data``.
    n_beams : int
        1 or 2. Set by the instrument subclass in ``read_data``.
    wcs : list[astropy.wcs.WCS]
        One per frame.
    parangs : np.ndarray, shape (N,)
        Raw parallactic angle (``PARANG`` header keyword) per frame, deg.
        Used by the Mueller model.
    sky_pa_deg : np.ndarray, shape (N,)
        Instrument-corrected on-sky PA per frame, deg (e.g. for NIRC2,
        ``PARANG + ROTPOSN - INSTANGL + zp_offset``). Used for spatial
        derotation to put sky North up. May equal ``parangs`` for
        instruments where no correction is needed.
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
        ``"stokes_cubes_per_cycle"``, ``"median_stokes_cube"``,
        ``"polarized_intensity"``, ``"dolp"``, ``"aolp"``,
        ``"radial_stokes_cube"``.
    """

    data: np.ndarray
    variance: np.ndarray
    n_beams: int
    wcs: list[Any]
    parangs: np.ndarray
    sky_pa_deg: np.ndarray
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
    def get_hwp_cycles(self) -> list[list[int]]:
        """Frame-index groups, one per complete HWP cycle.

        A "cycle" is the smallest set of HWP angles required for a full
        double-difference (typically 0, 22.5, 45, 67.5 deg). The instrument
        subclass decides what counts as complete and how to handle stragglers.
        """

    @abstractmethod
    def get_mueller_matrix(self, frame_index: int) -> np.ndarray:
        """Return the (4, 4) Mueller matrix for a single frame."""

    @abstractmethod
    def get_mueller_matrix_sequence(self) -> np.ndarray:
        """Return the (N, 4, 4) stack of per-frame Mueller matrices."""

    @abstractmethod
    def get_mueller_parameters(self) -> dict[str, float]:
        """Return the current free Mueller-model parameters."""

    @abstractmethod
    def set_mueller_parameters(self, params: dict[str, float]) -> None:
        """Update free Mueller-model parameters (used by mueller_fit)."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write outputs in an instrument-appropriate FITS layout."""

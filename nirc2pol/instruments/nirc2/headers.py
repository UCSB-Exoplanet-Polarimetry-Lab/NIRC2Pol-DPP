"""Keck/NIRC2 FITS keyword normalization.

Every NIRC2-specific FITS keyword (``ITIME``, ``COADDS``, ``CAMNAME``,
``FILNAME``, ``EL``, ``ROTPOSN``, ``PARANG``, ``WPLANGLE`` ...) is handled
here and nowhere else.
"""
from __future__ import annotations

from typing import Any


def normalize_header(prihdr: Any, exthdr: Any | None = None) -> dict[str, Any]:
    """Return a flat dict of pipeline-internal keys for one frame.

    Pulls NIRC2-specific keywords from the primary and extension headers and
    converts them to the internal schema (units in :doc:`/ARCHITECTURE`).
    """
    raise NotImplementedError


def get_itime_s(prihdr: Any) -> float:
    """Per-frame integration time in seconds."""
    raise NotImplementedError


def get_coadds(prihdr: Any) -> int:
    """Number of coadds for the frame."""
    raise NotImplementedError


def get_filter(prihdr: Any) -> str:
    """NIRC2 filter name from ``FILNAME``/``FWINAME``/``FWONAME``."""
    raise NotImplementedError


def get_camera(prihdr: Any) -> str:
    """NIRC2 camera (narrow/wide) from ``CAMNAME``."""
    raise NotImplementedError


def get_readout_mode(prihdr: Any) -> str:
    """Readout-mode identifier used for dark matching."""
    raise NotImplementedError


def get_parang_deg(prihdr: Any) -> float:
    """Parallactic angle, deg."""
    raise NotImplementedError


def get_elevation_deg(prihdr: Any) -> float:
    """Telescope elevation, deg — Mueller-model input."""
    raise NotImplementedError


def get_rotator_angle_deg(prihdr: Any) -> float:
    """Instrument rotator angle, deg — Mueller-model input."""
    raise NotImplementedError


def get_hwp_angle_deg(prihdr: Any) -> float:
    """Commanded HWP angle, deg."""
    raise NotImplementedError


def get_wavelength_m(prihdr: Any) -> float:
    """Effective wavelength for the current filter, m."""
    raise NotImplementedError


def get_frame_type(prihdr: Any) -> str:
    """One of ``"science"``, ``"dark"``, ``"flat"``, ``"sky"``."""
    raise NotImplementedError

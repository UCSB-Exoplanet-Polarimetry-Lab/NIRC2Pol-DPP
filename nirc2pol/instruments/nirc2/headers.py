"""Keck/NIRC2 FITS keyword normalization.

Every NIRC2-specific FITS keyword (``ITIME``, ``COADDS``, ``CAMNAME``,
``FILNAME``, ``EL``, ``ROTPOSN``, ``PARANG``, etc.) is handled here and
nowhere else.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

# Date of NIRC2 servicing that changed the narrow-cam PA offset.
_DATE_2015_04_13 = time.strptime("2015-4-13", "%Y-%m-%d")

# Filter → effective wavelength (m). Extend as needed; falls back to NaN.
_FILTER_WAVELENGTH_M: dict[str, float] = {
    "Kp": 2.124e-6,
    "Ks": 2.146e-6,
    "K": 2.196e-6,
    "H": 1.633e-6,
    "J": 1.248e-6,
    "Lp": 3.776e-6,
    "Ms": 4.670e-6,
}


def get_pa_deg(prihdr: Any) -> float:
    """Compute the NIRC2 PA (deg) for one frame from header keywords.

    Translation of the notebook's ``get_pa(..., mean_PA=False)`` path.
    Handles both vertical-angle and position-angle rotator modes; applies
    the narrow-cam zero-point offset that changed after the 2015-04-13
    servicing.
    """
    obsdate = time.strptime(prihdr["DATE-OBS"], "%Y-%m-%d")
    if prihdr["CAMNAME"].strip().lower() == "narrow":
        zp_offset = -0.252 if obsdate < _DATE_2015_04_13 else -0.262
    else:
        zp_offset = 0.0

    rotmode = prihdr.get("ROTMODE", "position angle").strip().lower()
    rotposn = prihdr["ROTPOSN"]
    instangl = prihdr["INSTANGL"]

    if rotmode == "vertical angle":
        return prihdr["PARANG"] + rotposn - instangl + zp_offset
    if rotmode == "position angle":
        return rotposn - instangl + zp_offset
    raise NotImplementedError(f"ROTMODE {rotmode!r} not supported")


def assign_hwp_angles_positionally(
    n_frames: int,
    cycle: tuple[float, ...] = (0.0, 22.5, 45.0, 67.5),
) -> np.ndarray:
    """Assign HWP angles by file order when the FITS header lacks an HWP key.

    NIRC2 polarimetry files frequently don't record the commanded HWP angle.
    The convention is that files are saved in cycle order, so the ``i``-th
    frame sits at ``cycle[i % len(cycle)]``.
    """
    return np.array([cycle[i % len(cycle)] for i in range(n_frames)])


def get_wavelength_m(prihdr: Any) -> float:
    """Effective wavelength for the current filter, m. NaN if unknown."""
    filter_name = (
        prihdr.get("FWINAME") or prihdr.get("FWONAME") or prihdr.get("FILNAME") or ""
    )
    return _FILTER_WAVELENGTH_M.get(filter_name.strip(), float("nan"))


def normalize_header(prihdr: Any, exthdr: Any | None = None) -> dict[str, Any]:
    """Flat dict of pipeline-internal keys for one frame (placeholder)."""
    raise NotImplementedError


def get_itime_s(prihdr: Any) -> float:
    """Per-frame integration time, s."""
    return float(prihdr["ITIME"])


def get_coadds(prihdr: Any) -> int:
    """Coadd count."""
    return int(prihdr["COADDS"])


def get_filter(prihdr: Any) -> str:
    """NIRC2 filter name."""
    return (
        prihdr.get("FWINAME") or prihdr.get("FWONAME") or prihdr.get("FILNAME") or ""
    ).strip()


def get_camera(prihdr: Any) -> str:
    """NIRC2 camera (``narrow``/``medium``/``wide``)."""
    return prihdr["CAMNAME"].strip().lower()


def get_readout_mode(prihdr: Any) -> str:
    """Readout-mode identifier derived from ``SAMPMODE`` + ``MULTISAM``."""
    return f"sampmode={prihdr.get('SAMPMODE', '?')}_multisam={prihdr.get('MULTISAM', '?')}"


def get_parang_deg(prihdr: Any) -> float:
    return float(prihdr.get("PARANG", 0.0))


def get_elevation_deg(prihdr: Any) -> float:
    return float(prihdr.get("EL", 0.0))


def get_rotator_angle_deg(prihdr: Any) -> float:
    return float(prihdr.get("ROTPPOSN", prihdr.get("ROTPOSN", 0.0)))


def get_hwp_angle_deg(prihdr: Any) -> float:
    """Per-frame HWP angle, deg.

    NIRC2 typically doesn't record this — see
    :func:`assign_hwp_angles_positionally` for the workaround.
    """
    return float(prihdr.get("WPLANGLE", float("nan")))


def get_frame_type(prihdr: Any) -> str:
    """One of ``"science"``, ``"dark"``, ``"flat"``, ``"sky"``."""
    raise NotImplementedError

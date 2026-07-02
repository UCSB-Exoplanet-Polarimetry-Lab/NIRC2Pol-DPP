"""NIRC2 FITS reading — raw files → data, variance, headers.

Handles the Wollaston dual-beam split at read time so downstream code sees
``data`` of shape ``(N, 2, ny, nx)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from nirc2pol.instruments.nirc2 import headers


def read_nirc2_files(
    filelist: list[Path],
    chop_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read NIRC2 FITS files, chop into dual beams, return a dict of arrays.

    The dict's keys match the attribute names on :class:`PolarimetryData`.
    """
    chop_kwargs = chop_kwargs or {}
    data, prihdrs = [], []
    for path in filelist:
        with fits.open(path) as hdul:
            chopped = chop_wollaston_frames(hdul[0].data, **chop_kwargs)
            data.append(chopped)
            prihdrs.append(hdul[0].header.copy())
    data = np.array(data, dtype=np.float64)

    n = len(prihdrs)
    parangs = np.array([h.get("PARANG", 0.0) for h in prihdrs])
    parangs[parangs < 0] += 360.0
    elevations = np.array([h.get("EL", 0.0) for h in prihdrs])
    # Note: notebook uses 'ROTPPOSN' (likely typo) in the demodulation; the
    # canonical NIRC2 keyword is 'ROTPOSN'. Reading both and preferring the
    # notebook's choice keeps the port numerically faithful.
    rotator_angles = np.array([h.get("ROTPPOSN", h.get("ROTPOSN", 0.0)) for h in prihdrs])
    calculated_pa = np.array([headers.get_pa_deg(h) for h in prihdrs])
    hwp_angles = headers.assign_hwp_angles_positionally(n)

    return {
        "data": data,
        "variance": np.clip(data, 1.0, None),  # Poisson floor; TODO: propagate
        "n_beams": data.shape[1],
        "parangs": parangs,
        "sky_pa_deg": calculated_pa,
        "elevations": elevations,
        "rotator_angles": rotator_angles,
        "hwp_angles": hwp_angles,
        "wavelengths": np.array([headers.get_wavelength_m(h) for h in prihdrs]),
        "exposure_times": np.array([h.get("ITIME", 0.0) for h in prihdrs]),
        "coadds": np.array([h.get("COADDS", 1) for h in prihdrs]),
        "filenames": [Path(p) for p in filelist],
        "prihdrs": prihdrs,
        "exthdrs": [None] * n,
        "wcs": [None] * n,
        "pixel_scale": 0.009942,  # narrow cam default; refine from header
    }


def chop_wollaston_frames(
    frame: np.ndarray,
    bottom_half_cutoff: int = 450,
    top_half_cuton: int = 508,
    x_offset: int = 13,
) -> np.ndarray:
    """Split one NIRC2 polarimetry frame into the two Wollaston beams.

    Returns
    -------
    np.ndarray, shape (2, ny_chop, nx_chop)
        ``out[0]`` = bottom (ordinary) beam, ``out[1]`` = top (extraordinary)
        beam, with the inter-beam x-shift removed.
    """
    top = frame[top_half_cuton:top_half_cuton + bottom_half_cutoff, :]
    bottom = frame[:bottom_half_cutoff, :]
    out = np.zeros((2, bottom.shape[0], bottom.shape[1] - x_offset), dtype=np.float64)
    out[0] = bottom[:, :-x_offset]
    out[1] = top[:, x_offset:]
    return out

"""Pixel-for-pixel reproduction of the AB Aur straight-PDI reduction.

Goal: run the framework code on the same 92 raw frames the notebook uses
and confirm the resulting median Qφ/Uφ match the notebook's reference cubes
``qphi_median.fits`` / ``uphi_median.fits``.

Skips the scipy.optimize minimization (we're verifying cells 8–19, not 21+).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from astropy.io import fits
from pyklip.klip import rotate as pyklip_rotate

from nirc2pol.instruments.nirc2 import NIRC2PolarimetryData
from nirc2pol.instruments.nirc2.mueller import full_system_mueller
from nirc2pol.polarimetry.double_diff import double_difference
from nirc2pol.polarimetry.radial_stokes import radial_stokes_single
from nirc2pol.reduction.sky import subtract_box_mean_background


RAW_DIR = Path("data/ab_aur/raw")
REF_QPHI = Path("/Users/brileylewis/Downloads/qphi_median.fits")
REF_UPHI = Path("/Users/brileylewis/Downloads/uphi_median.fits")


def file_num(path: str) -> int:
    return int(Path(path).stem.split("_")[1])


def collect_files() -> tuple[list[str], np.ndarray]:
    """Match the notebook's glob order (sort across both batches) and per-frame centers."""
    all_files = sorted(glob.glob(str(RAW_DIR / "reduced_*.fits")), key=file_num)
    selected, centers = [], []
    for f in all_files:
        n = file_num(f)
        if 857 <= n <= 900 or 915 <= n <= 930:
            selected.append(f)
            centers.append((786, 234))
        elif 932 <= n <= 963:
            selected.append(f)
            centers.append((600, 300))
    return selected, np.array(centers)


def register_per_frame(dataset: NIRC2PolarimetryData, centers: np.ndarray,
                       cutout_size: int = 400, padsize: int = 400) -> None:
    """Pad-and-crop registration with a different center per frame."""
    n, n_beams = dataset.data.shape[0], dataset.data.shape[1]
    half = cutout_size // 2
    new = np.empty((n, n_beams, cutout_size, cutout_size), dtype=dataset.data.dtype)
    for i in range(n):
        cx, cy = int(centers[i, 0]), int(centers[i, 1])
        for b in range(n_beams):
            padded = np.pad(dataset.data[i, b], ((padsize, padsize), (padsize, padsize)), constant_values=0)
            new[i, b] = padded[cy + padsize - half:cy + padsize + half,
                               cx + padsize - half:cx + padsize + half]
    dataset.data = new


def reduce_to_radial_stokes_median(
    dataset: NIRC2PolarimetryData,
    mask_center_xy: tuple[float, float] = (197.0, 203.0),
    mask_radius_px: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Notebook cells 8 → 19 reproduced via framework helpers + a hand-coded mask."""
    cycles = dataset.get_hwp_cycles()
    print(f"  {len(cycles)} HWP cycles")

    # NaN mask in instrument frame (notebook applies before sky derotation).
    ys, xs = np.indices(dataset.data.shape[-2:])
    cx, cy = mask_center_xy
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 < mask_radius_px ** 2

    qphis, uphis = [], []
    for cycle in cycles:
        _, q_inst, u_inst = double_difference(dataset, cycle)

        # Notebook averages the input angles, then evaluates Mueller once
        # (trig is nonlinear, so this differs from averaging the matrices).
        params = dataset.get_mueller_parameters()
        m = full_system_mueller(
            parang_deg=float(np.mean(dataset.parangs[cycle])),
            elevation_deg=float(np.mean(dataset.elevations[cycle])),
            rotator_angle_deg=float(np.mean(dataset.rotator_angles[cycle])),
            hwp_angle_deg=0.0,  # not used by this simplified Mueller
            wavelength_m=float(dataset.wavelengths[cycle[0]]),
            retardance_rad=params["retardance_rad"],
            fast_axis_offset_deg=params["fast_axis_offset_deg"],
        )
        m_inv = m.T
        q_sky = m_inv[1, 1] * q_inst + m_inv[1, 2] * u_inst
        u_sky = m_inv[2, 1] * q_inst + m_inv[2, 2] * u_inst

        # Central mask (NaN), then PA-derotate to sky orientation with pyklip's rotation.
        q_sky = np.where(mask, np.nan, q_sky)
        u_sky = np.where(mask, np.nan, u_sky)
        pa = float(np.mean(dataset.sky_pa_deg[cycle]))
        # pyklip.klip.rotate expects center as [x, y]
        q_rot = pyklip_rotate(q_sky, pa, list(mask_center_xy))
        u_rot = pyklip_rotate(u_sky, pa, list(mask_center_xy))

        # Per-cycle radial Stokes (notebook cell 17).
        qphi, uphi = radial_stokes_single(q_rot, u_rot, mask_center_xy)
        qphis.append(qphi)
        uphis.append(uphi)

    # Notebook cell 19 uses np.median, not nanmedian — NaN-propagating.
    qphi_med = np.median(np.stack(qphis), axis=0)
    uphi_med = np.median(np.stack(uphis), axis=0)
    return qphi_med, uphi_med


def compare(name: str, ours: np.ndarray, ref: np.ndarray) -> None:
    mask = np.isfinite(ours) & np.isfinite(ref)
    diff = ours[mask] - ref[mask]
    print(f"  {name}:")
    print(f"    finite-pixel overlap: {mask.sum()} / {ref.size}")
    print(f"    ref:  median={np.nanmedian(ref):.4g}, std={np.nanstd(ref):.4g}")
    print(f"    ours: median={np.nanmedian(ours):.4g}, std={np.nanstd(ours):.4g}")
    print(f"    diff: mean={diff.mean():.4g}, std={diff.std():.4g}, max|.|={np.abs(diff).max():.4g}")
    rel = np.linalg.norm(diff) / np.linalg.norm(ref[mask])
    print(f"    L2 relative residual: {rel:.4g}")


def main() -> None:
    files, centers = collect_files()
    print(f"Loading {len(files)} files. First/last: {Path(files[0]).name}, {Path(files[-1]).name}")
    dataset = NIRC2PolarimetryData(filelist=files)
    print(f"Raw data shape: {dataset.data.shape}")

    print("Subtracting box-mean background ...")
    subtract_box_mean_background(dataset, box=(25, 350, 50, 400))

    print("Per-frame registration ...")
    register_per_frame(dataset, centers)
    print(f"Registered data shape: {dataset.data.shape}")

    print("Building per-cycle radial Stokes + median ...")
    qphi_med, uphi_med = reduce_to_radial_stokes_median(dataset)

    print()
    print("=== Comparison vs reference ===")
    compare("Qφ", qphi_med, fits.getdata(str(REF_QPHI)))
    compare("Uφ", uphi_med, fits.getdata(str(REF_UPHI)))


if __name__ == "__main__":
    main()

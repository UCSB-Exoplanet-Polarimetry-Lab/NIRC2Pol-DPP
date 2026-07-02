"""Frame centering / alignment — populates ``dataset.output['star_center']``.

For NIRC2 polarimetry the dual beams come out of the Wollaston at known
detector locations, so registration is a pad-and-crop to a common centered
cutout (matches the notebook's ``rotate_cutouts(just_recenter=True)``).
"""
from __future__ import annotations

import numpy as np

from nirc2pol.instruments.base import PolarimetryData


def register_frames(
    dataset: PolarimetryData,
    center_xy: tuple[int, int] = (600, 300),
    cutout_size: int = 400,
    padsize: int = 400,
) -> None:
    """Pad and crop every beam to a ``cutout_size`` window centered on ``center_xy``.

    Stores the per-frame star center (the cutout center, in cropped-image
    pixel coordinates) in ``dataset.output["star_center"]``.
    """
    cx, cy = center_xy
    half = cutout_size // 2

    new_data = np.empty(
        (dataset.data.shape[0], dataset.data.shape[1], cutout_size, cutout_size),
        dtype=dataset.data.dtype,
    )
    for i in range(dataset.data.shape[0]):
        for b in range(dataset.data.shape[1]):
            beam = dataset.data[i, b]
            padded = np.pad(beam, ((padsize, padsize), (padsize, padsize)), constant_values=0)
            ys, ye = cy + padsize - half, cy + padsize + half
            xs, xe = cx + padsize - half, cx + padsize + half
            new_data[i, b] = padded[ys:ye, xs:xe]

    dataset.data = new_data
    dataset.variance = np.clip(new_data, 1.0, None)  # TODO: propagate properly
    dataset.output["star_center"] = np.tile([half, half], (new_data.shape[0], 1))


def centroid_single_frame(
    frame: np.ndarray, initial_guess: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Sub-pixel centroid of a single frame (not used by the AB Aur port)."""
    raise NotImplementedError

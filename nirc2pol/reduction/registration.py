"""Frame centering / alignment — populates ``dataset.output['star_center']``."""
from __future__ import annotations

import numpy as np

from nirc2pol.instruments.base import PolarimetryData


def register_frames(dataset: PolarimetryData) -> None:
    """Estimate the star centroid in every frame.

    Writes an ``(N, 2)`` array of ``(x, y)`` coordinates (zero-indexed,
    sub-pixel allowed) to ``dataset.output["star_center"]``. Does not modify
    ``data`` or ``variance``.
    """
    raise NotImplementedError


def centroid_single_frame(frame: np.ndarray, initial_guess: tuple[float, float] | None = None) -> tuple[float, float]:
    """Sub-pixel centroid of a single frame."""
    raise NotImplementedError

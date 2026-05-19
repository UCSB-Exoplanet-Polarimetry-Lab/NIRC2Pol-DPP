"""Optional count-rate normalization: data /= exptime*coadds; var /= (exptime*coadds)^2."""
from __future__ import annotations

from nirc2pol.instruments.base import PolarimetryData


def normalize_by_exposure(dataset: PolarimetryData) -> None:
    """Convert frames to counts/sec by dividing by ``exposure_times × coadds``.

    Marks the dataset as normalized (via a flag in ``dataset.output``) so a
    second invocation is a no-op rather than silently double-normalizing.
    Output is in counts/sec — no absolute flux calibration is performed.
    """
    raise NotImplementedError

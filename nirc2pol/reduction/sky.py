"""Sky / thermal background subtraction."""
from __future__ import annotations

from pathlib import Path

from nirc2pol.instruments.base import PolarimetryData


def subtract_sky(dataset: PolarimetryData, sky_library_path: Path) -> None:
    """Subtract the matching sky frame from each science frame.

    Updates ``dataset.data`` and ``dataset.variance`` in place.
    """
    raise NotImplementedError

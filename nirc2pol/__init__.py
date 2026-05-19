"""nirc2pol — polarimetric differential imaging pipeline for Keck/NIRC2."""
from __future__ import annotations

from nirc2pol import calibration, polarimetry, reduction
from nirc2pol.pipeline import Pipeline

__version__ = "0.0.0"

__all__ = ["Pipeline", "reduction", "polarimetry", "calibration"]

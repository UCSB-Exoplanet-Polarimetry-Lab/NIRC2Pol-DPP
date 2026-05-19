"""Domain exceptions used across the pipeline."""
from __future__ import annotations


class Nirc2PolError(Exception):
    """Base class for all nirc2pol errors."""


class NoMatchingCalibrationError(Nirc2PolError):
    """Raised when no calibration frame matches a science frame's settings."""


class IncompleteHwpSequenceError(Nirc2PolError):
    """Raised when a Stokes set is missing required HWP angles."""


class PipelineOrderingError(Nirc2PolError):
    """Raised when a pipeline step is registered before its dependencies."""

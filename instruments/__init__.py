"""Instrument-specific constants, header handling, and frame classification.
Currently: NIRC2 (``instruments.nirc2``)."""

from .base import PolarimetryData, config_csv, read_config

# Only the generic interface. The NIRC2 specifics stay at
# ``instruments.nirc2`` deliberately: flattening one instrument's constants
# into the package root would make ``from instruments import band_of``
# ambiguous the moment a second instrument exists.
__all__ = ["PolarimetryData", "config_csv", "read_config"]

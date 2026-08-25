"""NIRC2-Pol DPP: dual-beam polarimetry reduction for Keck/NIRC2.

Four layers, each talking only to the one below it::

    nirc2pol/polarimetry/   double differencing, Stokes cubes, radial Stokes, IP
    nirc2pol/reduction/     masters, calibration, sky subtraction, registration
    nirc2pol/instruments/   everything NIRC2-specific, behind PolarimetryData
    nirc2pol/utils/         Frame (image + header), FITS I/O, image operations

Every choice a reduction makes lives in a TOML config
(:class:`nirc2pol.reduction.config.ReductionConfig`); ``nirc2pol-reduce``
runs one end to end, and :func:`nirc2pol.recipe.run` is the same thing
callable from a notebook.
"""

# Kept as a plain literal: setuptools reads it out of this file without
# importing the package, so the version is available before the dependencies
# are.
__version__ = "0.1.0"

from . import instruments, polarimetry, reduction, utils

__all__ = ["instruments", "polarimetry", "reduction", "utils", "__version__"]

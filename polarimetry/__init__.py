"""Polarimetric science layer: double differencing, Stokes cube production,
radial Stokes, and the Mueller matrix model interface."""

from .stokes import (
    CRITICAL_ANGLES,
    azimuthal_angle,
    build_stokes_cube,
    build_stokes_cubes,
    double_difference,
    median_stokes_cube,
    normalized_single_difference,
    polarization_products,
    radial_stokes,
    rotate_qu,
    single_difference,
)
from .products import ProductWriter
from .mueller import (
    MuellerMatrixModel,
    build_corrected_stokes_cube,
    fit_empirical_cycle_correction,
)
from .instpol import (
    InstrumentalPolarization,
    fit_ip_uphi,
    mean_ip,
    measure_ip_annulus,
    measure_ip_cycle,
    subtract_ip,
)
from .fast_axis import (
    OFFSET_TO_FRAME,
    FastAxisResult,
    PreparedCycle,
    butterfly_phase,
    combine_at_offset,
    fit_fast_axis_on_sky,
    prepare_cycles,
    scan_fast_axis_offset,
    wrap_offset,
)

__all__ = [
    "CRITICAL_ANGLES",
    "FastAxisResult",
    "InstrumentalPolarization",
    "MuellerMatrixModel",
    "OFFSET_TO_FRAME",
    "PreparedCycle",
    "ProductWriter",
    "azimuthal_angle",
    "build_corrected_stokes_cube",
    "build_stokes_cube",
    "build_stokes_cubes",
    "butterfly_phase",
    "combine_at_offset",
    "double_difference",
    "fit_empirical_cycle_correction",
    "fit_fast_axis_on_sky",
    "fit_ip_uphi",
    "mean_ip",
    "measure_ip_annulus",
    "measure_ip_cycle",
    "median_stokes_cube",
    "normalized_single_difference",
    "polarization_products",
    "prepare_cycles",
    "radial_stokes",
    "rotate_qu",
    "scan_fast_axis_offset",
    "single_difference",
    "subtract_ip",
    "wrap_offset",
]

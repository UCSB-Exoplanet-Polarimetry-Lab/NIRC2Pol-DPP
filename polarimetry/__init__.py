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
    apply_mueller_model,
    build_corrected_stokes_cube,
    fit_empirical_cycle_correction,
)
from .instpol import (
    InstrumentalPolarization,
    fit_ip_uphi,
    fit_ip_uphi_all,
    mean_ip,
    subtract_ip,
)
from .fast_axis import (
    OFFSET_TO_FRAME,
    FastAxisResult,
    PreparedCycle,
    butterfly_phase,
    combine_at_offset,
    fit_fast_axis_butterfly,
    prepare_cycles,
    scan_fast_axis_offset_butterfly,
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
    "apply_mueller_model",
    "build_corrected_stokes_cube",
    "build_stokes_cube",
    "build_stokes_cubes",
    "butterfly_phase",
    "combine_at_offset",
    "double_difference",
    "fit_empirical_cycle_correction",
    "fit_fast_axis_butterfly",
    "fit_ip_uphi",
    "fit_ip_uphi_all",
    "mean_ip",
    "median_stokes_cube",
    "normalized_single_difference",
    "polarization_products",
    "prepare_cycles",
    "radial_stokes",
    "rotate_qu",
    "scan_fast_axis_offset_butterfly",
    "single_difference",
    "subtract_ip",
    "wrap_offset",
]

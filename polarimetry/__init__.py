"""Polarimetric science layer: double differencing, Stokes cube production,
radial Stokes, and the Mueller matrix model interface."""

from .stokes import (
    CRITICAL_ANGLES,
    build_stokes_cube,
    build_stokes_cubes,
    double_difference,
    median_stokes_cube,
    polarization_products,
    radial_stokes,
    rotate_qu,
    single_difference,
)
from .products import ProductWriter
from .mueller import (
    MuellerMatrixModel,
    RotationApproximationModel,
    build_corrected_stokes_cube,
    fit_empirical_cycle_correction,
)
from .instpol import (
    InstrumentalPolarization,
    fit_ip_uphi,
    mean_ip,
    measure_ip_annulus,
    measure_ip_edge,
    measure_ip_frame,
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
from .sequences import generate_sequence_epoch, sequence_dict, split_horizontal

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
from .sequences import generate_sequence_epoch, sequence_dict, split_horizontal

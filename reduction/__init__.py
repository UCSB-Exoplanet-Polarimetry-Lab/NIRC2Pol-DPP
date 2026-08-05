"""Science reduction layer, instrument-agnostic: master darks/flats/skies,
frame calibration, sky subtraction, and image registration. Instrument
specifics come in through ``instruments``; low-level helpers live in
``utils``."""

from .masters import (
    make_lamp_flats,
    make_master_darks,
    make_master_flats,
    make_master_masks,
    make_master_skies,
    make_masters,
    flat_sort_key,
    preferred_flat_type_for,
    split_polarimetric_flats,
)
from .calibrate import (
    find_closest_dark,
    find_closest_flat,
    find_closest_sky,
    find_matching_master,
    reduce_frame,
)
from .sky import (
    subtract_annulus_background,
    subtract_dither_pairs,
    subtract_mean_background,
    subtract_sky_frames,
)
from .registration import (
    center_frame,
    center_frames,
    derotate_frames,
    find_center,
    find_center_centroid,
    find_center_crosscorr,
    find_center_silhouette,
    find_center_symmetry,
    find_center_wings,
    find_center_smooth,
    median_combine,
    register_beam_stack,
    register_frames_to_template,
)

"""Low-level helpers: the Frame container, FITS I/O, generic image
operations, Gaussian fitting, and on-disk path conventions."""

from .frame import (
    Frame,
    all_header_keywords_match,
    frame_number,
    in_frame_range,
    framelist_to_cube,
    get_between,
    load_frames,
    select_frames,
    load_master,
    parse_date_obs,
    match_keys,
    save_frames,
    scrub_header,
)
from .imutils import (
    argquantile,
    crop,
    image_is_larger,
    make_annulus_mask,
    make_circle_mask,
    make_sigma_clip_mask,
    plus_mask,
    rotate_image_center,
    translate,
)
from .angles import (angles_match, is_critical_angle, mean_angle,
                     par_angle, sexagesimal_to_degrees,
                     small_angle_distance)
from .logs import ReductionLog, start_reduction_log
from .paths import ObslogPaths, link_frames, load_rejects, record_reject
from .provenance import (describe, drop_step, pipeline_version,
                         record_step, steps_of)

# The package's public API. Named explicitly rather than derived, because
# this is the list to consult before renaming something -- and because
# without it ``from utils import *`` also binds the submodule names
# (frame, paths, angles...), which would shadow a caller's own variables.
__all__ = [
    "Frame",
    "ObslogPaths",
    "ReductionLog",
    "all_header_keywords_match",
    "angles_match",
    "argquantile",
    "crop",
    "describe",
    "drop_step",
    "frame_number",
    "in_frame_range",
    "framelist_to_cube",
    "get_between",
    "image_is_larger",
    "is_critical_angle",
    "load_frames",
    "select_frames",
    "load_master",
    "link_frames",
    "load_rejects",
    "make_annulus_mask",
    "make_circle_mask",
    "make_sigma_clip_mask",
    "match_keys",
    "mean_angle",
    "par_angle",
    "parse_date_obs",
    "pipeline_version",
    "plus_mask",
    "record_reject",
    "record_step",
    "rotate_image_center",
    "save_frames",
    "scrub_header",
    "sexagesimal_to_degrees",
    "small_angle_distance",
    "start_reduction_log",
    "steps_of",
    "translate",
]

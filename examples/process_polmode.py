"""End-to-end NIRC2-Pol reduction: raw frames to Stokes cubes.

The script form of ``examples/tutorial.ipynb``: the same steps and the same
choices, run over a whole night laid out by :class:`utils.ObslogPaths`
instead of the small dataset bundled in ``examples/tutorial_data``. The
notebook explains each step and why it is done that way; this file is the
one you edit and run.

Follows the DPP block diagram / SPIE workflow as plain function calls, so
each step is easy to run and inspect on its own (e.g. in a notebook):

    1. sort raw frames by type (headers)
    2. build master darks / flats / skies
    3. pre-process science frames (dark, flat, bad pixels)
    4. measure the beam geometry, then HWP cycle matching
    5. fast axis offset (FAST_AXIS_METHOD) and instrumental polarization
       (IP_METHOD), chosen independently -- the butterfly fit settles the
       offset only
    6. Stokes cubes per cycle -- beam splitting, background subtraction,
       registration (REGISTER_METHOD) and double differencing all happen
       inside the builder
    7. median Stokes cube, PI / AoLP / DoLP and radial Stokes, written with
       full provenance by ``ProductWriter``

Everything the run reports is also written to one log file beside the
products, so the choices it made survive the terminal session.

There is deliberately no separate sky-subtraction step. The background is a
property of the instrument (``BACKGROUND_METHOD`` below) and is applied to
each Wollaston beam inside the Stokes builder, which calls
``instrument.subtract_background(instrument.split_beams(frame))``. Removing
a background from whole frames beforehand would measure it across both
beams at once, and would leave the instrument's own setting unused.

Edit the configuration block, then run from the repository root:

    python examples/process_polmode.py
"""

import glob
import logging
import os
import sys

# make the repo importable when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from instruments import nirc2
import numpy as np

from polarimetry import (ProductWriter, apply_mueller_model,
                         build_stokes_cubes, fit_ip_uphi,
                         subtract_residual_halo,
                         fit_ip_uphi_all, mean_ip,
                         fit_fast_axis_butterfly, median_stokes_cube)
from reduction import (fit_beam_geometry, make_master_darks,
                       make_master_flats,
                       make_master_masks, make_master_skies, reduce_frame)
from utils import (ObslogPaths, load_frames, load_rejects, save_frames,
                   select_frames, start_reduction_log)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("process_polmode")

# ---------------------------------------------------------------------------
# configuration
# Root folder holding ONE SUBFOLDER PER NIGHT -- not the folder your FITS
# files are in. For DATE below, frames are read from <ROOT>/<DATE>/raw/:
#
#     /data/nirc2pol/          <- OBSERVATIONS_ROOT
#       2025-12-08/            <- DATE
#         raw/  *.fits         <- your raw frames go here
#         reduced/ sequences/  <- written by this script
#
OBSERVATIONS_ROOT = "/path/to/data_polmode"
# UTC, as DATE-OBS records it. A Keck night runs 04:00-16:00 UTC, so one
# UTC date names a whole night -- one day after the HST evening.
DATE = "2025-12-08"

# TARGET only names the output files. What this reduction actually covers is
# set by SELECT_* below.
TARGET = "AB_Aur"

# Which frames become science products. Darks, flats and reduced frames are
# built for the whole night regardless; this narrows what goes on to HWP
# cycle matching and the Stokes products, which is where you want only the
# frames you trust. None means "no constraint".
#   SELECT_FRAME_RANGE = (932, 939)   inclusive, by the number in the filename
#   SELECT_FRAME_RANGE = [(857, 900), (915, 930)]   several runs at once
#   SELECT_TARGET      = "AB Aur"     matched ignoring case, spaces and _ / -
# For frames that are simply bad, prefer the reject file -- it keeps a reason
# and applies to every run of the night. See the note by load_rejects below.
SELECT_FRAME_RANGE = None
SELECT_TARGET = None

# Background, applied per Wollaston beam inside build_stokes_cubes. The band
# decides what is sensible and nirc2.check_background_choice warns if this is
# a poor fit: L'/M want "dither" or "mean_box", JHK want "annulus" or
# "mean_box". Box coordinates are in beam-cutout pixels, not full frames.
BACKGROUND_METHOD = "mean_box"        # "mean_box" | "annulus" | "dither" | None
BACKGROUND_BOX = (25, 350, 50, 400)   # (ylow, yhigh, xlow, xhigh)
BACKGROUND_ANNULUS = None             # (r_inner, r_outer) px, for "annulus"

# Beam geometry is measured from the data in step 4, not configured here.
# Set these only to override the measurement for a reduction.
BEAM_TOP_ROW = None
BEAM_X_OFFSET = None

# Centering algorithm used to register the two beams before differencing.
# The right choice depends on what the source looks like: "smooth_peak" for a
# point source, "min" for a saturated core that reads low, "wings" behind a
# coronagraph, "symmetry"/"centroid" for a resolved body, None to skip. A
# saturated L-prime core defeats "smooth_peak": its rim maxima are near-equal,
# so the finder hops between them from frame to frame and the combined PSF
# comes out doubled. Check by registering every frame and looking at the
# scatter of the centres it finds.
REGISTER_METHOD = "min"

# Subtract a master sky from every science frame during reduction. Off by
# default: combined with a mean-box or annulus background it removes the
# pedestal twice. Turn it on for dedicated sky frames with
# BACKGROUND_METHOD = None.
USE_MASTER_SKIES = False

# The fast axis offset and the instrumental polarization are two halves of
# one question, so they are chosen the same way: by naming the METHOD. Both
# default to "mm_model" -- the Mueller matrix model, which settles both at
# once and is the destination -- and that route is not built yet, so a
# reduction that has not thought about this stops with a clear error rather
# than quietly using an idealized rotation and no leakage correction.
#
# FAST_AXIS_METHOD
#   "mm_model"   take the offset from the Mueller matrix model (NOT YET
#                IMPLEMENTED -- raises). Settles the IP too, so IP_METHOD
#                must also be "mm_model".
#   "butterfly"  fit it from these data by the butterfly's orientation
#   "fixed"      use THETA_OFF below exactly as given, fitting nothing
#
# Fast axis offset [deg], used when FAST_AXIS_METHOD = "fixed". There is no
# trusted automatic source for it: an HWP ladder on an internal source
# returns theta_off + chi/2, where chi is the incident polarization angle in
# the instrument frame and is unknown for a dome or lamp source. -13 deg is
# the value measured on sky for this AB Aur night, consistent across the
# butterfly fit and a scan minimum; it is not a default for other data.
FAST_AXIS_METHOD = "mm_model"
THETA_OFF = -13.0
FIT_RADII = (25, 150)   # (r_inner, r_outer) px, the butterfly annulus

# Instrumental polarization: the I -> Q/U leakage that makes an unpolarized
# source come out with Q = ipq*I, U = ipu*I.
#
# Read the names as <method>_<scope>: how the leakage is measured, and what
# it is measured over -- which is also the level it is applied at, so a
# per_cycle measurement is removed from its own cycle.
#
#   "mm_model"                  from the Mueller matrix model (NOT YET
#                               IMPLEMENTED -- raises). Settles the fast axis
#                               too, so FAST_AXIS_METHOD must also be
#                               "mm_model"
#   "fit_uphi_per_cycle"        minimise the U_phi residual per cycle
#   "fit_uphi_all"              minimise it once across every cycle
#   None                        leave it uncorrected
#
# There is no "fit_uphi_per_frame": one exposure gives a single difference,
# +-Q or +-U, never both, so U_phi cannot be formed from it.
#
# Both fit_uphi_* routes ASSUME THE SOURCE IS AZIMUTHALLY POLARIZED and will
# return a confident, meaningless number where that is the hypothesis under
# test. They work on U_phi, where a tangentially polarized disk puts no
# signal by definition, so IP_MASK_RADIUS need only clear the core and the
# annulus may span the disk.
#
# The mask-edge ("edge_annulus") family is deliberately NOT offered here --
# see the note by measure_ip_coronagraph. It is untested on data we have.
#
# On AB Aur the two fit_uphi_* options agree to within 0.005% in ipq.
IP_METHOD = "mm_model"      # see the list above

# de Boer et al. (2020) final step: rotating the polarization directions
# mixes Q and U, so halo signal that cancelled in the instrument frame partly
# reappears in the sky frame. This removes what is left, measured over a
# disk-free annulus on the COMBINED planes. None skips it.
FINAL_HALO_ANNULUS = None   # (r_inner, r_outer)

IP_MASK_RADIUS = 22         # px, covering the saturated or occulted core
# ---------------------------------------------------------------------------


class NightPolData(nirc2.NIRC2PolarimetryData):
    """NIRC2 as configured for this night.

    Same pattern as the notebook's ``LpPolData``: the background and the beam
    geometry are the per-dataset choices, so they are all this subclass sets.
    Detector constants and the polarimetric rotation model live on the base
    class.
    """

    background_method = BACKGROUND_METHOD
    background_box = BACKGROUND_BOX
    background_annulus = BACKGROUND_ANNULUS
    top_row_start = BEAM_TOP_ROW
    beam_x_offset = BEAM_X_OFFSET


instrument = NightPolData()
paths = ObslogPaths(OBSERVATIONS_ROOT, DATE)
paths.make_folders()
# Frames excluded from every run of this night, each with a reason. Add one
# with:
#     from utils.paths import record_reject
#     record_reject(paths.rejects_file, "n0937.fits", "open AO loop")
rejects = load_rejects(paths.rejects_file)

# Everything the reduction reports -- which flat it matched, whether the band
# requirement was enforced, the beam geometry it measured, any centering
# fallback -- lands in one file beside the products. The console still shows
# it too; this is the copy that survives the terminal.
run_log = start_reduction_log(paths.log_file)
run_log.settings(date=DATE, target=TARGET, instrument=type(instrument).__name__,
                 background=instrument.describe_background(),
                 theta_off=THETA_OFF, fast_axis_method=FAST_AXIS_METHOD,
                 ip_method=IP_METHOD,
                 final_halo_annulus=FINAL_HALO_ANNULUS,
                 use_master_skies=USE_MASTER_SKIES,
                 register_method=REGISTER_METHOD,
                 beam_top_row=BEAM_TOP_ROW, beam_x_offset=BEAM_X_OFFSET)
log.info("background: %s", instrument.describe_background())
if instrument.top_row_start is not None:
    log.info("beam geometry overridden: top row %s, x offset %s",
             instrument.top_row_start, instrument.beam_x_offset)

# --- 1. sort raw frames by type -------------------------------------------
# *.fits* rather than *.fits so gzipped archive frames are picked up too
raw_files = sorted(glob.glob(os.path.join(paths.raw_folder, "*.fits*")))
sorted_files = instrument.sort_frames(raw_files)

# --- 2. master darks / flats / skies --------------------------------------
# instrument= supplies the detector mask, how to spot a polarimetric
# (critical-angle) flat set, and which flat type each band requires. Pass
# any of them explicitly to override. Flats come in two kinds -- dome and
# sky -- and each is split into polarimetric and not, so this can build up
# to four masters per filter.
darks = load_frames(sorted_files["darks"], rejects=rejects)
master_darks, dark_masks = make_master_darks(darks, instrument=instrument)
if master_darks:
    save_frames(paths.darks_file, master_darks)

master_flats, flat_masks = make_master_flats(
    load_frames(sorted_files["flats_dome"], rejects=rejects),
    load_frames(sorted_files["flats_sky"], rejects=rejects),
    master_darks,
    instrument=instrument,
)
if master_flats:
    save_frames(paths.flats_file, master_flats)

master_skies = None
if USE_MASTER_SKIES:
    master_skies, _ = make_master_skies(
        load_frames(sorted_files["flats_sky"], rejects=rejects),
        master_darks, instrument=instrument)
    if master_skies:
        save_frames(paths.skies_file, master_skies)

master_masks = make_master_masks(dark_masks, flat_masks)

# --- 3. pre-process science frames ----------------------------------------
# read once, not once per frame: this loads a FITS file
bad_pixel_mask = instrument.bad_pixel_mask()

sci_frames = load_frames(sorted_files["sci"], rejects=rejects)
# DATE above must be the UTC date the frames carry, since the masters and
# every product inherit it from the folder name
paths.check_frame_dates(sci_frames)
nirc2.make_frametable(sci_frames, paths.table_file)

reduced_frames = []
for frame in sci_frames:
    reduced = reduce_frame(
        frame, master_flats, master_darks, master_skies, master_masks,
        bad_pixel_mask=bad_pixel_mask,
        required_flat_types=instrument.required_flat_types,
        default_required_flat_type=instrument.default_required_flat_type,
        gain=instrument.gain(frame),
        saturation_limit=instrument.saturation_limit(frame),
    )
    reduced.save(os.path.join(paths.reduced_folder, reduced["RED-FN"]))
    reduced_frames.append(reduced)

# --- 3b. choose the frames this reduction covers --------------------------
# Everything above ran on the whole night. From here on it is just the frames
# selected, so the Stokes products are built from those alone. The choice is
# logged, so the reduction log records which frames the products came from.
reduced_frames = select_frames(reduced_frames, target=SELECT_TARGET,
                               frame_range=SELECT_FRAME_RANGE)

# --- 4. HWP cycle matching ------------------------------------------------
# Measure where the two beams sit, from these frames. The separation moves
# between epochs, so it is measured every time rather than looked up: a
# value written down once goes stale without saying so. Nothing downstream
# can undo a wrong one -- registration shifts both beams together -- so
# split_beams refuses until this has run.
if instrument.top_row_start is None or instrument.beam_x_offset is None:
    instrument.top_row_start, instrument.beam_x_offset = fit_beam_geometry(
        instrument, reduced_frames)
log.info("beam geometry: top row %d, x offset %d",
         instrument.top_row_start, instrument.beam_x_offset)

cycles = instrument.match_modulator_cycles(reduced_frames)

# --- 5. fast axis offset and instrumental polarization --------------------
# Both are chosen by naming a method; nothing here has a silent default. The
# Mueller matrix model settles the two together, so it is all or neither.
FAST_AXIS_METHODS = ("mm_model", "butterfly", "fixed")
IP_METHODS = ("mm_model", "fit_uphi_per_cycle", "fit_uphi_all", None)

if FAST_AXIS_METHOD not in FAST_AXIS_METHODS:
    raise ValueError(f"FAST_AXIS_METHOD must be one of {FAST_AXIS_METHODS}, "
                     f"not {FAST_AXIS_METHOD!r}")
if IP_METHOD == "fit_uphi_per_frame":
    raise ValueError(
        "There is no 'fit_uphi_per_frame'. One exposure yields a single "
        "difference -- +-Q or +-U, never both -- so U_phi cannot be formed "
        "from it and there is nothing to minimise. Use "
        "'fit_uphi_per_cycle' instead.")
if str(IP_METHOD).startswith("edge_annulus"):
    raise ValueError(
        "The edge_annulus routes are withdrawn for now. The estimator is "
        "sum(Q)/sum(I) over an annulus, which needs enough starlight in that "
        "annulus to be stable -- and on AB Aur there is no annulus that has "
        "it while also excluding the disk. Measured on that data, median I "
        "falls 63x between r = 22-40 px and r = 160-220 px, and the reported "
        "ipq grows with radius as the denominator shrinks: -0.9%, +0.4%, "
        "+0.5%, -1.9%, -3.4%, -8.6%. Use fit_uphi_all or "
        "fit_uphi_per_cycle.")
if IP_METHOD == "butterfly_joint":
    raise ValueError(
        "'butterfly_joint' is gone: the butterfly fit now determines the "
        "fast axis offset only, and the leakage is chosen separately with "
        "IP_METHOD. Note that every route currently offered needs the offset "
        "as an input, so the offset is fitted without a leakage removed and "
        "is biased; fit_fast_axis_butterfly takes ip= if you have one from "
        "elsewhere.")
if IP_METHOD not in IP_METHODS:
    raise ValueError(f"IP_METHOD must be one of {IP_METHODS}, not "
                     f"{IP_METHOD!r}. The name states the method and its "
                     f"scope, so a new route arrives as a new name rather "
                     f"than changing what an old one means.")

using_mm = (FAST_AXIS_METHOD == "mm_model", IP_METHOD == "mm_model")
if any(using_mm) and not all(using_mm):
    raise ValueError(
        "The Mueller matrix model gives the fast axis offset and the "
        "instrumental polarization together -- they are both terms of the "
        "same matrix -- so FAST_AXIS_METHOD and IP_METHOD must either both "
        f"be 'mm_model' or neither. Got FAST_AXIS_METHOD="
        f"{FAST_AXIS_METHOD!r}, IP_METHOD={IP_METHOD!r}.")

theta_off, ip = None, None
dd_kwargs = {"register_method": REGISTER_METHOD}

if all(using_mm):
    # Not fitted here: this applies a matrix determined elsewhere. Raises
    # NotImplementedError until that plumbing exists.
    theta_off, ip = apply_mueller_model(instrument, cycles)

# --- then the fast axis ---------------------------------------------------
if theta_off is None:
    if FAST_AXIS_METHOD == "fixed":
        if THETA_OFF is None:
            raise ValueError("FAST_AXIS_METHOD='fixed' uses THETA_OFF, which "
                             "is None. Set it, or pick another method.")
        theta_off = THETA_OFF
        log.info("fast axis: fixed at %g deg (nothing fitted)", theta_off)

    elif FAST_AXIS_METHOD == "butterfly":
        r_inner, r_outer = FIT_RADII
        result = fit_fast_axis_butterfly(instrument, cycles, ip=None,
                                         r_inner=r_inner, r_outer=r_outer,
                                         **dd_kwargs)
        theta_off = result.theta_off
        log.info("fast axis from the butterfly: %s", result.describe())
        if IP_METHOD is not None:
            log.warning(
                "Fast axis fitted with no leakage removed: every available "
                "IP route (%r here) needs the offset as an input, so the "
                "leakage cannot be measured first. The offset and the "
                "leakage are degenerate, so this offset is biased by however "
                "much IP there is. fit_fast_axis_butterfly takes ip= if you "
                "have a leakage from elsewhere.", IP_METHOD)

# --- and the leakage last, where it needed the offset ---------------------
if IP_METHOD == "fit_uphi_all":
    ip = fit_ip_uphi_all(instrument, cycles, theta_off,
                         mask_radius=IP_MASK_RADIUS, **dd_kwargs)
elif IP_METHOD == "fit_uphi_per_cycle":
    ip = [fit_ip_uphi(instrument, c, theta_off, mask_radius=IP_MASK_RADIUS,
                      **dd_kwargs)
          for c in cycles]
    log.info("per-cycle U_phi IP: ipq %s",
             ", ".join(f"{p.ipq:+.4f}" for p in ip))
    log.info("per-cycle IP standard error: ipq %.4f, ipu %.4f",
             mean_ip(ip).diagnostics["ipq_err"],
             mean_ip(ip).diagnostics["ipu_err"])

if isinstance(ip, list):
    log.info("instrumental polarization, per cycle (mean %s)",
             mean_ip(ip).describe())
elif ip is not None:
    log.info("instrumental polarization: %s", ip.describe())
elif IP_METHOD is None:
    log.warning("No instrumental polarization correction (IP_METHOD=None); "
                "the I -> Q/U leakage stays in the products.")

instrument.fast_axis_offset = theta_off

# --- 6. Stokes cubes ------------------------------------------------------
# The leakage is removed in the instrument frame, before Q/U are rotated to
# sky, so it is an argument here rather than subtracted from a finished cube.
# The FINAL_HALO_ANNULUS step below is a different quantity -- residual net
# polarization measured in the sky frame, after rotation -- not this one
# applied late.
stokes_cubes = build_stokes_cubes(instrument, cycles,
                                  fast_axis_offset=theta_off, ip=ip,
                                  **dd_kwargs)
median_cube = median_stokes_cube(stokes_cubes)

# de Boer's last step: the rotation into sky mixes Q and U, so halo signal
# that cancelled in the instrument frame partly reappears. Measured and
# removed in the sky frame, which is self-consistent -- see the note in
# polarimetry.instpol about why that is not the instrument-frame rule broken.
if FINAL_HALO_ANNULUS is not None:
    q_final, u_final, _ = subtract_residual_halo(
        median_cube[1], median_cube[2], median_cube[0],
        *FINAL_HALO_ANNULUS)
    median_cube = np.stack([median_cube[0], q_final, u_final])

# --- 7. products ----------------------------------------------------------
header = cycles[0][0].header.copy()
header["THETAOFF"] = (theta_off, "fast axis offset [deg]")

# ObslogPaths owns the night layout (raw/, reduced/, sequences/); the writer
# owns the product set and its provenance, so it is rooted at sequences/.
writer = ProductWriter(paths.sequences_folder, target=TARGET)
writer.save_stokes_cycles(stokes_cubes, cycles, header=header)
writer.save_median_stokes(median_cube, header=header)
writer.save_derived_products(median_cube, header=header)

run_log.finish()

print(f"Done: {len(stokes_cubes)} HWP cycles -> Stokes products in "
      f"{writer.output_dir}")
print(f"Log:  {run_log.path} ({run_log.warnings} warnings)")

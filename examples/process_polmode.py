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
    5. instrumental polarization and fast axis offset (both measured on sky)
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
from polarimetry import (ProductWriter, build_stokes_cubes,
                         fit_fast_axis_on_sky, median_stokes_cube)
from reduction import (fit_beam_geometry, make_master_darks,
                       make_master_flats,
                       make_master_masks, make_master_skies, reduce_frame)
from utils import (ObslogPaths, load_frames, load_rejects, save_frames,
                   start_reduction_log)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("process_polmode")

# ---------------------------------------------------------------------------
# configuration
OBSERVATIONS_FOLDER = "/path/to/data_polmode"  # contains <date>/raw/*.fits
# UTC, as DATE-OBS records it. A Keck night runs 04:00-16:00 UTC, so one
# UTC date names a whole night -- one day after the HST evening.
DATE = "2025-12-08"
TARGET = "AB_Aur"

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

# Fast axis offset [deg]. There is no trusted automatic source for it: an
# HWP ladder on an internal source returns theta_off + chi/2, where chi is
# the incident polarization angle in the instrument frame and is unknown for
# a dome or lamp source. Measure it on sky and set it here, or set
# FIT_ON_SKY to measure it from these data. -13 deg is the value measured on
# sky for this AB Aur night, consistent across three routes (offset alone,
# joint with IP, and a scan minimum); it is not a default for other data.
THETA_OFF = -13.0

# Fit theta_off and the instrumental polarization jointly from the data.
# Assumes an azimuthally polarized source (a scattered-light disk); on a
# point source, an AGN or a star field it returns a confident wrong answer.
FIT_ON_SKY = False
FIT_RADII = (25, 150)   # (r_inner, r_outer) px, the butterfly annulus
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
paths = ObslogPaths(OBSERVATIONS_FOLDER, DATE)
paths.make_folders()
rejects = load_rejects(paths.rejects_file)

# Everything the reduction reports -- which flat it matched, whether the band
# requirement was enforced, the beam geometry it measured, any centering
# fallback -- lands in one file beside the products. The console still shows
# it too; this is the copy that survives the terminal.
run_log = start_reduction_log(paths.log_file)
run_log.settings(date=DATE, target=TARGET, instrument=type(instrument).__name__,
                 background=instrument.describe_background(),
                 theta_off=THETA_OFF, fit_on_sky=FIT_ON_SKY,
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
# any of them explicitly to override.
darks = load_frames(sorted_files["darks"], rejects=rejects)
master_darks, dark_masks = make_master_darks(darks, instrument=instrument)
if master_darks:
    save_frames(paths.darks_file, master_darks)

master_flats, flat_masks = make_master_flats(
    load_frames(sorted_files["flats"], rejects=rejects),
    load_frames(sorted_files["flats_sky"], rejects=rejects),
    load_frames(sorted_files["flats_lampon"], rejects=rejects),
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
theta_off = THETA_OFF
ip = None
if FIT_ON_SKY:
    r_inner, r_outer = FIT_RADII
    result = fit_fast_axis_on_sky(instrument, cycles,
                                  r_inner=r_inner, r_outer=r_outer,
                                  register_method=REGISTER_METHOD)
    theta_off, ip = result.theta_off, result.ip
    log.info("fast axis on sky: %s", result.describe())
    if ip is not None:
        log.info("instrumental polarization: %s", ip.describe())
    # for an error bar on the IP, fit it per cycle instead and take the
    # scatter: mean_ip([fit_ip_uphi(instrument, c, theta_off) for c in cycles])

instrument.fast_axis_offset = theta_off

# --- 6. Stokes cubes ------------------------------------------------------
# IP is removed in the instrument frame, before Q/U are rotated to sky, so it
# is an argument here rather than something subtracted from a finished cube.
stokes_cubes = build_stokes_cubes(instrument, cycles,
                                  fast_axis_offset=theta_off, ip=ip,
                                  register_method=REGISTER_METHOD)
median_cube = median_stokes_cube(stokes_cubes)

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

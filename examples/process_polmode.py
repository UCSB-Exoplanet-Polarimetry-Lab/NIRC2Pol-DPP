"""End-to-end NIRC2-Pol reduction: raw frames to Stokes cubes.

Follows the DPP block diagram / SPIE workflow as plain function calls, so
each step is easy to run and inspect on its own (e.g. in a notebook):

    1. sort raw frames by type (headers)
    2. build master darks / flats / skies
    3. pre-process science frames (dark, flat, bad pixels)
    4. sky / dither subtraction
    5. HWP cycle matching
    6. beam splitting + image registration
    7. fast axis value from the calibration log
    8. Stokes cubes per cycle, median Stokes cube, PI/AoLP/DoLP,
       radial Stokes

Edit the configuration block, then run from the repository root:

    python examples/process_polmode.py
"""

import glob
import logging
import os
import sys

# make the repo importable when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from instruments import nirc2
from polarimetry import (build_stokes_cubes, median_stokes_cube,
                         polarization_products, radial_stokes)
from reduction import (make_master_darks, make_master_flats,
                       make_master_masks, make_master_skies, reduce_frame,
                       subtract_annulus_background)
from utils import Frame, ObslogPaths, load_frames, load_rejects, save_frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# configuration
OBSERVATIONS_FOLDER = "/path/to/data_polmode"  # contains <date>/raw/*.fits
DATE = "2025-12-04"
TARGET = "AB_Aur"
SKY_ANNULUS = (150, 200)     # (r_in, r_out) in px; None to skip
# ---------------------------------------------------------------------------

instrument = nirc2.NIRC2PolarimetryData()
paths = ObslogPaths(OBSERVATIONS_FOLDER, DATE)
paths.make_folders()
rejects = load_rejects(paths.rejects_file)

# --- 1. sort raw frames by type -------------------------------------------
raw_files = sorted(glob.glob(os.path.join(paths.raw_folder, "*.fits")))
sorted_files = instrument.sort_frames(raw_files)

# --- 2. master darks / flats / skies --------------------------------------
bad_pixel_mask = instrument.bad_pixel_mask()

darks = load_frames(sorted_files["darks"], rejects=rejects)
master_darks, dark_masks = make_master_darks(darks, bad_pixel_mask=bad_pixel_mask)
if master_darks:
    save_frames(paths.darks_file, master_darks)

master_flats, flat_masks = make_master_flats(
    load_frames(sorted_files["flats"], rejects=rejects),
    load_frames(sorted_files["flats_sky"], rejects=rejects),
    load_frames(sorted_files["flats_lampon"], rejects=rejects),
    load_frames(sorted_files["flats_lampoff"], rejects=rejects),
    master_darks,
    bad_pixel_mask=bad_pixel_mask,
)
if master_flats:
    save_frames(paths.flats_file, master_flats)

master_skies, _ = make_master_skies(
    load_frames(sorted_files["flats_sky"], rejects=rejects),
    master_darks, bad_pixel_mask=bad_pixel_mask)

master_masks = make_master_masks(dark_masks, flat_masks)

# --- 3. pre-process science frames ----------------------------------------
sci_frames = load_frames(sorted_files["sci"], rejects=rejects)
nirc2.make_frametable(sci_frames, paths.table_file)

reduced_frames = []
for frame in sci_frames:
    reduced = reduce_frame(
        frame, master_flats, master_darks, master_skies, master_masks,
        bad_pixel_mask=bad_pixel_mask,
        flat_exceptions=instrument.flat_exceptions,
        gain=instrument.gain(frame),
        saturation_limit=instrument.saturation_limit(frame),
    )
    reduced["RDNOISE"] = instrument.readnoise(reduced)
    reduced.save(os.path.join(paths.reduced_folder, reduced["RED-FN"]))
    reduced_frames.append(reduced)

# --- 4. sky subtraction (annulus; use subtract_dither_pairs for L') -------
if SKY_ANNULUS is not None:
    for frame in reduced_frames:
        frame.data = subtract_annulus_background(frame.data, *SKY_ANNULUS)

# --- 5. HWP cycle matching ------------------------------------------------
cycles = instrument.match_modulator_cycles(reduced_frames)

# --- 6. beam splitting + registration happen inside the Stokes builder via
#        instrument.split_beams; center the frames first so the beams land
#        on the star (see reduction.register_beam_stack for finer control)

# --- 7. fast axis value from the calibration log --------------------------
theta_off = nirc2.load_fast_axis_offset(DATE)
instrument.fast_axis_offset = theta_off

# --- 8. Stokes cubes and derived products ---------------------------------
stokes_cubes = build_stokes_cubes(instrument, cycles,
                                  fast_axis_offset=theta_off)
median_cube = median_stokes_cube(stokes_cubes)

header = cycles[0][0].header.copy()
header["THETAOFF"] = theta_off

Frame(stokes_cubes, header).save(
    os.path.join(paths.sequences_folder, f"{TARGET}_stokes_cubes.fits"))
Frame(median_cube, header).save(
    os.path.join(paths.sequences_folder, f"{TARGET}_median_stokes.fits"))

pi, aolp, dolp = polarization_products(median_cube)
q_phi, u_phi = radial_stokes(median_cube[1], median_cube[2])

for name, img in [("PI", pi), ("AoLP", aolp), ("DoLP", dolp),
                  ("Qphi", q_phi), ("Uphi", u_phi)]:
    Frame(img, header).save(
        os.path.join(paths.sequences_folder, f"{TARGET}_{name}.fits"))

print(f"Done: {len(stokes_cubes)} HWP cycles -> Stokes products in "
      f"{paths.sequences_folder}")

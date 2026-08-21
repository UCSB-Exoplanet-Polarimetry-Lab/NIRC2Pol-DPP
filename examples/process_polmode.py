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
    5. fast axis offset (cfg.fast_axis_method) and instrumental polarization
       (cfg.ip_method), chosen independently -- the butterfly fit settles the
       offset only
    6. Stokes cubes per cycle -- beam splitting, background subtraction,
       registration (cfg.register_method) and double differencing all happen
       inside the builder
    7. median Stokes cube, PI / AoLP / DoLP and radial Stokes, written with
       full provenance by ``ProductWriter``

Everything the run reports is also written to one log file beside the
products, so the choices it made survive the terminal session.

There is deliberately no separate sky-subtraction step. The background is a
property of the instrument (``cfg.background_method`` below) and is applied to
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
from reduction.config import ReductionConfig
from reduction import (fit_beam_geometry, make_master_darks,
                       make_master_flats,
                       make_master_masks, make_master_skies, reduce_frame)
from utils import (ObslogPaths, load_frames, load_rejects, save_frames,
                   select_frames, start_reduction_log)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("process_polmode")

# ---------------------------------------------------------------------------
# Every choice this reduction makes lives in a TOML config, not here. Generate
# a fresh one listing every option, its default and its allowed values with:
#
#     python -c "from reduction.config import ReductionConfig; \
#                print(ReductionConfig.template())" > my_night.toml
#
# then edit it and run:
#
#     python examples/process_polmode.py my_night.toml
#
# Instrument constants -- plate scale, detector epochs, the beam geometry
# search seed -- are a different thing and live in instruments/nirc2.toml.
CONFIG_PATH = (sys.argv[1] if len(sys.argv) > 1
               else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "reduction_config.toml"))
cfg = ReductionConfig.from_toml(CONFIG_PATH)
# ---------------------------------------------------------------------------


class NightPolData(nirc2.NIRC2PolarimetryData):
    """NIRC2 as configured for this night.

    Same pattern as the notebook's ``LpPolData``: the background and the beam
    geometry are the per-dataset choices, so they are all this subclass sets.
    Detector constants and the polarimetric rotation model live on the base
    class.
    """

    background_method = cfg.background_method
    background_box = cfg.background_box
    background_annulus = cfg.background_annulus
    top_row_start = cfg.beam_top_row
    beam_x_offset = cfg.beam_x_offset


instrument = NightPolData()
paths = ObslogPaths(cfg.observations_root, cfg.date)
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
run_log.settings(instrument=type(instrument).__name__,
                 background=instrument.describe_background(),
                 config=CONFIG_PATH, **cfg.describe())
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
if cfg.use_master_skies:
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
# cfg.date above must be the UTC date the frames carry, since the masters and
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
reduced_frames = select_frames(reduced_frames, target=cfg.select_target,
                               frame_range=cfg.select_frame_range)

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
# Both are chosen by naming a method. ReductionConfig has already checked
# that the pair is coherent -- the Mueller model settles the two together, so
# it is all or neither -- and that a fixed offset actually has a value.
theta_off, ip = None, None
dd_kwargs = {"register_method": cfg.register_method}

if cfg.fast_axis_method == "mm_model":
    # Not fitted here: this applies a matrix determined elsewhere. Raises
    # NotImplementedError until that plumbing exists.
    theta_off, ip = apply_mueller_model(instrument, cycles)

# --- then the fast axis ---------------------------------------------------
if theta_off is None:
    if cfg.fast_axis_method == "fixed":
        theta_off = cfg.theta_off
        log.info("fast axis: fixed at %g deg (nothing fitted)", theta_off)

    elif cfg.fast_axis_method == "butterfly":
        r_inner, r_outer = cfg.fit_radii
        result = fit_fast_axis_butterfly(instrument, cycles, ip=None,
                                         r_inner=r_inner, r_outer=r_outer,
                                         **dd_kwargs)
        theta_off = result.theta_off
        log.info("fast axis from the butterfly: %s", result.describe())
        if cfg.ip_method is not None:
            log.warning(
                "Fast axis fitted with no leakage removed: every available "
                "IP route (%r here) needs the offset as an input, so the "
                "leakage cannot be measured first. The offset and the "
                "leakage are degenerate, so this offset is biased by however "
                "much IP there is. fit_fast_axis_butterfly takes ip= if you "
                "have a leakage from elsewhere.", cfg.ip_method)

# --- and the leakage last, where it needed the offset ---------------------
if cfg.ip_method == "fit_uphi_all":
    ip = fit_ip_uphi_all(instrument, cycles, theta_off,
                         mask_radius=cfg.ip_mask_radius, **dd_kwargs)
elif cfg.ip_method == "fit_uphi_per_cycle":
    ip = [fit_ip_uphi(instrument, c, theta_off, mask_radius=cfg.ip_mask_radius,
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
elif cfg.ip_method is None:
    log.warning("No instrumental polarization correction (cfg.ip_method=None); "
                "the I -> Q/U leakage stays in the products.")

instrument.fast_axis_offset = theta_off

# --- 6. Stokes cubes ------------------------------------------------------
# The leakage is removed in the instrument frame, before Q/U are rotated to
# sky, so it is an argument here rather than subtracted from a finished cube.
# The cfg.final_halo_annulus step below is a different quantity -- residual net
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
if cfg.final_halo_annulus is not None:
    q_final, u_final, _ = subtract_residual_halo(
        median_cube[1], median_cube[2], median_cube[0],
        *cfg.final_halo_annulus)
    median_cube = np.stack([median_cube[0], q_final, u_final])

# --- 7. products ----------------------------------------------------------
header = cycles[0][0].header.copy()
header["THETAOFF"] = (theta_off, "fast axis offset [deg]")

# ObslogPaths owns the night layout (raw/, reduced/, sequences/); the writer
# owns the product set and its provenance, so it is rooted at sequences/.
writer = ProductWriter(paths.sequences_folder, target=cfg.target)
writer.save_stokes_cycles(stokes_cubes, cycles, header=header)
writer.save_median_stokes(median_cube, header=header)
writer.save_derived_products(median_cube, header=header)

run_log.finish()

print(f"Done: {len(stokes_cubes)} HWP cycles -> Stokes products in "
      f"{writer.output_dir}")
print(f"Log:  {run_log.path} ({run_log.warnings} warnings)")

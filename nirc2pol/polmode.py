"""The standard NIRC2-Pol reduction, raw frames to Stokes products.

One function, :func:`run`, doing the steps of the DPP block diagram / SPIE
workflow in order:

    1. sort raw frames by type (headers)
    2. build master darks / flats / skies
    3. choose the science frames this reduction covers
    4. pre-process them (dark, flat, bad pixels)
    5. measure the beam geometry -- before any dither subtraction, which
       would leave a negative image of the star for the centroid to find
    6. dither subtraction, when cfg.background_method asks for it
    7. HWP cycle matching
    8. fast axis offset (cfg.fast_axis_method) and instrumental polarization
       (cfg.ip_method), chosen independently -- the butterfly fit settles the
       offset only
    9. Stokes cubes per cycle -- beam splitting, background subtraction,
       registration (cfg.register_method) and double differencing all happen
       inside the builder
    10. median Stokes cube, PI / AoLP / DoLP and radial Stokes, written with
        full provenance by ``ProductWriter``

``nirc2pol-reduce night.toml`` is this function with a command line around
it. Calling it directly is the same reduction, and it hands back everything
it built, so a notebook can look at any of it::

    from nirc2pol.polmode import run
    from nirc2pol.reduction.config import ReductionConfig

    products = run(ReductionConfig.from_toml("night.toml"))
    products["cycles"]        # the HWP cycles it matched
    products["median_cube"]   # the combined Stokes cube

For a reduction that departs from this recipe, call the module functions
yourself rather than bending this one; ``examples/tutorial.ipynb`` walks
through them individually.

There is deliberately no separate sky-subtraction step. The background is a
property of the instrument (``cfg.background_method``) and is applied to each
Wollaston beam inside the Stokes builder, which calls
``instrument.subtract_background(instrument.split_beams(frame))``. Removing a
background from whole frames beforehand would measure it across both beams at
once, and would leave the instrument's own setting unused.
"""

import logging
import os

from nirc2pol.instruments import nirc2
from nirc2pol.polarimetry import (ProductWriter, apply_mueller_model,
                                  build_stokes_cubes, fit_ip_uphi,
                                  fit_ip_uphi_all, mean_ip,
                                  fit_fast_axis_butterfly, median_stokes_cube)
from nirc2pol.reduction import (fit_beam_geometry, make_master_darks,
                                make_master_flats, make_master_masks,
                                make_master_skies, reduce_frame,
                                subtract_dither_background)
from nirc2pol.utils import (ObslogPaths, load_frames, load_rejects,
                            read_headers, save_frames, select_frames,
                            start_reduction_log)

log = logging.getLogger(__name__)


def run(cfg, config_path=None):
    """Reduce one night, as ``cfg`` describes it.

    Parameters
    ----------
    cfg : ReductionConfig
        Every choice the reduction makes. Already validated by its own
        ``__post_init__``, so this function does not re-check the options.
    config_path : str, optional
        Where ``cfg`` was read from, recorded in the log as ``config_source``.
        The config itself is copied into the reduction folder as
        ``reduction_<date>.toml`` regardless, so the run can be repeated from
        its own folder; this only records where it originally came from.

    Returns
    -------
    dict
        Everything the run built, so a caller can inspect any of it:
        ``instrument``, ``paths``, ``cycles``, ``theta_off``, ``ip``,
        ``stokes_cubes``, ``median_cube``, ``writer`` and ``run_log``.

    Notes
    -----
    Writes as it goes -- masters, corrected frames, per-cycle and median
    Stokes products -- under ``cfg.reductions_root``. What is kept
    is itself a choice: see ``save_preproc`` and ``save_individual_cycles``.
    """
    # No subclass needed: the config sets the five per-dataset attributes on
    # the instance. Subclass NIRC2PolarimetryData only to change behaviour --
    # override a method, describe a night the base class cannot.
    instrument = cfg.configure(nirc2.NIRC2PolarimetryData())
    paths = ObslogPaths(cfg.reductions_root, cfg.date)

    # The frames stay where they are; raw/ gets links to the ones this run
    # reads, so the reduction folder records its own inputs and the archive
    # is never written to. Nothing is created until the frames are known to
    # exist, so a wrong raw_data_folder leaves no empty tree behind.
    raw_files = paths.link_raw_frames(cfg.raw_data_folder,
                                      frame_range=cfg.raw_range)

    paths.make_folders()
    # Frames excluded from every run of this night, each with a reason. Add one
    # with:
    #     from nirc2pol.utils.paths import record_reject
    #     record_reject(paths.rejects_file, "n0937.fits", "open AO loop")
    rejects = load_rejects(paths.rejects_file)

    # Everything the reduction reports -- which flat it matched, whether the
    # band requirement was enforced, the beam geometry it measured, any
    # centering fallback -- lands in one file beside the products. The console
    # still shows it too; this is the copy that survives the terminal.
    # Copy the config in beside the log. The log records the values, but a
    # file can be re-run, and the one that was passed on the command line
    # may be edited or gone by the time anybody comes back to this.
    saved_config = cfg.to_toml(paths.config_file)

    run_log = start_reduction_log(paths.log_file)
    run_log.settings(instrument=type(instrument).__name__,
                     background=instrument.describe_background(),
                     config=saved_config, config_source=config_path,
                     **cfg.describe())
    log.info("background: %s", instrument.describe_background())
    if instrument.top_row_start is not None:
        log.info("beam geometry overridden: top row %s, x offset %s",
                 instrument.top_row_start, instrument.beam_x_offset)

    # --- 1. sort raw frames by type ------------------------------------------
    # The frames were linked above, and cfg.raw_range decided which: that is
    # what is read off disk at all, and is distinct from select_frame_range,
    # which picks the science frames -- raw_range has to stay wide enough to
    # include the darks and flats, or the masters cannot be built.
    sorted_files = instrument.sort_frames(raw_files)

    # --- 2. master darks / flats / skies -------------------------------------
    # instrument= supplies the detector mask, how to spot a polarimetric
    # (critical-angle) flat set, and which flat type each band requires. Pass
    # any of them explicitly to override. Flats come in two kinds -- dome and
    # sky -- and each is split into polarimetric and not, so this can build up
    # to four masters per filter.
    # cfg.save_preproc decides whether the masters and the corrected science
    # frames are kept. They are built either way -- nothing downstream runs
    # without them -- so this is about disk, not about what the reduction does.
    darks = load_frames(sorted_files["darks"], rejects=rejects)
    master_darks, dark_masks = make_master_darks(
        darks, instrument=instrument, min_frames=cfg.master_min_frames)
    if master_darks and cfg.save_preproc:
        save_frames(paths.darks_file, master_darks)

    # Which bands the science frames are in, so the flats in those bands lead
    # the inventory. Headers only -- the frames themselves are not loaded
    # until step 3. Ordering only: find_closest_flat still requires the
    # filter to match, so this cannot change which flat a frame gets.
    science_bands = {nirc2.band_of(h)
                     for h in read_headers(sorted_files["sci"])}
    science_bands.discard(None)

    master_flats, flat_masks = make_master_flats(
        load_frames(sorted_files["flats_dome"], rejects=rejects),
        load_frames(sorted_files["flats_sky"], rejects=rejects),
        master_darks,
        instrument=instrument,
        required_flat_type=cfg.required_flat_type,
        allow_flat_without_dark=cfg.allow_flat_without_dark,
        min_frames=cfg.master_min_frames,
        science_bands=science_bands,
    )
    if master_flats and cfg.save_preproc:
        save_frames(paths.flats_file, master_flats)

    master_skies = None
    if cfg.use_master_skies:
        master_skies, _ = make_master_skies(
            load_frames(sorted_files["flats_sky"], rejects=rejects),
            master_darks, instrument=instrument,
            min_frames=cfg.master_min_frames,
            group_radius_arcsec=cfg.sky_group_radius,
            group_gap_minutes=cfg.sky_group_gap)
        if master_skies and cfg.save_preproc:
            save_frames(paths.skies_file, master_skies)

    master_masks = make_master_masks(dark_masks, flat_masks)

    # --- 3. choose the frames this reduction covers --------------------------
    # Before reducing, not after. sort_frames classifies by elimination --
    # anything that is not a dark or a flat is science -- so the science
    # bucket also holds acquisition and engineering frames, in whatever band
    # they were taken. Reducing the whole bucket first means spending the
    # time on frames that are about to be discarded, and failing outright on
    # one in a band this night has no flat for.
    #
    # The frame table is written for the WHOLE night, since it is the thing
    # you read while deciding what to select.
    sci_frames = load_frames(sorted_files["sci"], rejects=rejects)
    # cfg.date above must be the UTC date the frames carry, since the masters
    # and every product inherit it from the folder name
    paths.check_frame_dates(sci_frames)
    nirc2.make_frametable(sci_frames, paths.table_file)

    sci_frames = select_frames(sci_frames, target=cfg.select_target,
                               frame_range=cfg.select_frame_range)

    # --- 4. pre-process the frames selected ----------------------------------
    # read once, not once per frame: this loads a FITS file
    bad_pixel_mask = instrument.bad_pixel_mask()

    reduced_frames = []
    for frame in sci_frames:
        reduced = reduce_frame(
            frame, master_flats, master_darks, master_skies, master_masks,
            bad_pixel_mask=bad_pixel_mask,
            required_flat_types=instrument.required_flat_types,
            default_required_flat_type=instrument.default_required_flat_type,
            # The same override as the masters were built with: choosing the
            # kind for the build and then enforcing the band default when
            # matching would refuse the flats it had just made.
            required_flat_type=cfg.required_flat_type,
            allow_flat_type_mismatch=cfg.allow_flat_type_mismatch,
            allow_no_flat=cfg.allow_no_flat,
            skip_sky_sub=cfg.skip_sky_sub,
            sky_group_radius_arcsec=cfg.sky_group_radius,
            sky_max_radius_arcsec=cfg.sky_max_radius,
            replacement_method=cfg.replacement_method,
            gain=instrument.gain(frame),
            saturation_limit=instrument.saturation_limit(frame),
        )
        reduced_frames.append(reduced)

    # --- 5. beam geometry, measured before any dither subtraction ------------
    # Where the two beams sit. The separation moves between epochs, so it is
    # measured every time rather than looked up: a value written down once
    # goes stale without saying so. Nothing downstream can undo a wrong one --
    # registration shifts both beams together -- so split_beams refuses until
    # this has run.
    #
    # Before the dither, deliberately. A dither-subtracted frame carries a
    # NEGATIVE image of the star a few arcsec from the positive one, and the
    # centroid cannot tell them apart: on the 2025-12-06 standard it measured
    # beam_x_offset 29.64 with 102 px of scatter, where the same frames
    # before subtraction give 12.13 with 0.01 px. Geometry is a property of
    # the optics, so measure it on frames that still have one star in them.
    if instrument.top_row_start is None or instrument.beam_x_offset is None:
        instrument.top_row_start, instrument.beam_x_offset = fit_beam_geometry(
            instrument, reduced_frames)
    log.info("beam geometry: top row %d, x offset %d",
             instrument.top_row_start, instrument.beam_x_offset)

    # --- 6. dither subtraction -----------------------------------------------
    # At frame level, before the Wollaston beams are cut out, and pair-matched
    # within one HWP angle so it differences two skies rather than two
    # polarization states.
    if cfg.background_method == "dither":
        reduced_frames = subtract_dither_background(
            reduced_frames, instrument,
            tolerance_arcsec=cfg.dither_tolerance)

    # Saved after the dither, not before: these files are meant to be what
    # the Stokes cubes were built from, and DITHSUB records which frame was
    # subtracted from each.
    if cfg.save_preproc:
        for reduced in reduced_frames:
            reduced.save(os.path.join(paths.reduced_folder,
                                      reduced["RED-FN"]))

    # --- 7. HWP cycle matching -----------------------------------------------
    cycles = instrument.match_modulator_cycles(reduced_frames)

    # --- 8. fast axis offset and instrumental polarization -------------------
    # Both are chosen by naming a method. ReductionConfig has already checked
    # that the pair is coherent -- the Mueller model settles the two together,
    # so it is all or neither -- and that a fixed offset actually has a value.
    theta_off, ip = None, None
    dd_kwargs = {"register_method": cfg.register_method}

    if cfg.fast_axis_method == "mm_model":
        # Not fitted here: this applies a matrix determined elsewhere. Raises
        # NotImplementedError until that plumbing exists.
        theta_off, ip = apply_mueller_model(instrument, cycles)

    # --- then the fast axis --------------------------------------------------
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
                    "Fast axis fitted with no leakage removed: every "
                    "available IP route (%r here) needs the offset as an "
                    "input, so the leakage cannot be measured first. The "
                    "offset and the leakage are degenerate, so this offset "
                    "is biased by however much IP there is. "
                    "fit_fast_axis_butterfly takes ip= if you have a "
                    "leakage from elsewhere.", cfg.ip_method)

    # --- and the leakage last, where it needed the offset --------------------
    if cfg.ip_method == "fit_uphi_all":
        ip = fit_ip_uphi_all(instrument, cycles, theta_off,
                             mask_radius=cfg.ip_mask_radius, **dd_kwargs)
    elif cfg.ip_method == "fit_uphi_per_cycle":
        ip = [fit_ip_uphi(instrument, c, theta_off,
                          mask_radius=cfg.ip_mask_radius, **dd_kwargs)
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
        log.warning("No instrumental polarization correction "
                    "(cfg.ip_method=None); the I -> Q/U leakage stays in "
                    "the products.")

    instrument.fast_axis_offset = theta_off

    # --- 9. Stokes cubes -----------------------------------------------------
    # The leakage is removed in the instrument frame, before Q/U are rotated to
    # sky, so it is an argument here rather than subtracted from a finished
    # cube.

    stokes_cubes = build_stokes_cubes(instrument, cycles,
                                      fast_axis_offset=theta_off, ip=ip,
                                      **dd_kwargs)
    median_cube = median_stokes_cube(stokes_cubes)

    # --- 10. products ---------------------------------------------------------
    # THETAOFF is already on this header: build_stokes_cubes wrote it when
    # it resolved the offset it actually used.
    header = cycles[0][0].header.copy()

    # ObslogPaths owns the night layout (raw/, reduced/, sequences/); the
    # writer owns the product set and its provenance, so it is rooted at
    # sequences/.
    writer = ProductWriter(paths.sequences_folder, target=cfg.target,
                           overwrite=cfg.overwrite_products)

    # save_individual_cycles decides whether the per-cycle Stokes data is
    # kept: one FITS per cycle, each carrying its own cycle's header. Off,
    # the median cube and the derived products are all that is written, and
    # no cycle can be dropped or re-combined without reducing the night
    # again.
    if cfg.save_individual_cycles:
        writer.save_stokes_cycles(stokes_cubes, cycles, header=header)
    writer.save_median_stokes(median_cube, header=header)
    writer.save_derived_products(median_cube, header=header,
                                 derived=cfg.save_derived_quantities,
                                 radial=cfg.save_radial_stokes,
                                 dolp_min_intensity=cfg.dolp_min_intensity)

    run_log.finish()

    return {"instrument": instrument, "paths": paths, "cycles": cycles,
            "theta_off": theta_off, "ip": ip, "stokes_cubes": stokes_cubes,
            "median_cube": median_cube, "writer": writer, "run_log": run_log}

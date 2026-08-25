"""The standard NIRC2-Pol reduction, raw frames to Stokes products.

Ten stages, in the order of the DPP block diagram / SPIE workflow, plus
:func:`run` which composes them:

    1. :func:`sort_raw`         classify raw frames by type (headers)
    2. :func:`build_masters`    master darks / flats / skies
    3. :func:`select_science`   choose the frames this reduction covers
    4. :func:`reduce_science`   dark, flat, bad pixels
    5. :func:`set_beam_cutout`  the nominal cutout for the band
    6. :func:`subtract_dither`  frame-level background, when asked for
    7. :func:`match_cycles`     HWP cycle matching
    8. :func:`solve_fast_axis`  theta_off, then the leakage that needs it
    9. :func:`build_stokes`     per-cycle Stokes cubes, and the median
    10. :func:`write_products`  products, with full provenance

``nirc2pol-reduce night.toml`` is :func:`run` with a command line around it::

    from nirc2pol.recipe import run
    from nirc2pol.reduction.config import ReductionConfig

    products = run(ReductionConfig.from_toml("night.toml"))
    products["cycles"]        # the HWP cycles it matched
    products["median_cube"]   # the combined Stokes cube

**The stages are the point.** A reduction that departs from the recipe calls
them itself rather than bending :func:`run`, and calls only the ones it
wants::

    instrument, paths, rejects, run_log, raw_files = prepare_night(cfg)
    sorted_files = sort_raw(raw_files, instrument)
    masters = build_masters(sorted_files, instrument, paths, rejects, cfg)
    ...

Every stage reads its settings from ``cfg``, so none can be left out by
accident -- which is how a hand-written reduction comes to differ from the
pipeline without failing. To change one setting for one call, pass it: the
keyword wins over the config, and the rest still come from ``cfg``::

    reduce_science(sci, masters, instrument, cfg, skip_sky_sub=True)

``examples/run_step_by_step.ipynb`` runs the whole thing a stage at a time and
shows what each one is made of.

There is deliberately no separate sky-subtraction stage for ``mean_box`` or
``annulus``. Those are per-beam, applied inside the Stokes builder, which
calls ``instrument.subtract_background(instrument.split_beams(frame))``.
Removing a background from whole frames beforehand would measure it across
both beams at once, and would leave the instrument's own setting unused.
``dither`` is different -- it differences whole exposures, before the beams
are cut -- which is why it has a stage and they do not.
"""

import logging
import os
from typing import NamedTuple

from nirc2pol.instruments import nirc2
from nirc2pol.polarimetry import (ProductWriter, apply_mueller_model,
                                  build_stokes_cubes, fit_ip_uphi,
                                  fit_ip_uphi_all, mean_ip,
                                  fit_fast_axis_butterfly, median_stokes_cube)
from nirc2pol.reduction import (background_stages, make_master_darks,
                                make_master_flats, make_master_masks,
                                make_master_skies, reduce_frame,
                                subtract_dither_background)
from nirc2pol.utils import (Frame, ObslogPaths, frame_number, in_frame_range,
                            load_frames, load_master, load_rejects,
                            read_headers, save_frames, select_frames,
                            start_reduction_log)

log = logging.getLogger(__name__)


class _FromConfig:
    """Sentinel: this argument has not been given, so read it from ``cfg``."""

    __slots__ = ()

    def __repr__(self):
        return "FROM_CFG"


FROM_CFG = _FromConfig()
"""Default for every stage keyword: take the value from ``cfg``.

Not ``None``, because None is a legitimate *value* for several settings --
``required_flat_type=None`` means "use the band default", ``crop_size=None``
means "derive it from the dither throw", ``background_box=None`` means "no
box", ``ip_method=None`` means "leave the leakage in". With None as the
default, asking for any of those explicitly would silently fall back to the
config instead, which is the class of quiet substitution the stages exist to
remove.
"""


def _resolve(value, fallback):
    """``value`` unless it is :data:`FROM_CFG`, in which case ``fallback``."""
    return fallback if value is FROM_CFG else value


class Masters(NamedTuple):
    """The calibration frames a reduction is built on.

    A bundle, not an object with behaviour: it unpacks like a tuple and reads
    like a record, and :func:`reduce_science` takes it as one argument rather
    than four.
    """

    darks: list
    flats: list
    skies: object
    masks: object


def prepare_night(cfg, config_path=None):
    """Instrument, folders, rejects and the log — everything a stage needs.

    Parameters
    ----------
    cfg : ReductionConfig
        Every choice the reduction makes.
    config_path : str, optional
        Where ``cfg`` was read from, recorded in the log as ``config_source``.

    Returns
    -------
    tuple
        ``(instrument, paths, rejects, run_log, raw_files)``.

    Notes
    -----
    The frames stay where they are; ``raw/`` gets links to the ones this run
    reads, so the reduction folder records its own inputs and the archive is
    never written to. Nothing is created until the frames are known to exist,
    so a wrong ``raw_data_folder`` leaves no empty tree behind — which is why
    the linking happens before ``make_folders``.

    The config is copied in beside the log. The log records the *values*, but
    a file can be re-run, and the one passed on the command line may be edited
    or gone by the time anybody comes back to this.

    Reads ``reductions_root``, ``date``, ``raw_data_folder``, ``raw_range``,
    and everything ``cfg.describe()`` reports.
    """
    # No subclass needed: the config sets the five per-dataset attributes on
    # the instance. Subclass NIRC2PolarimetryData only to change behaviour --
    # override a method, describe a night the base class cannot.
    instrument = cfg.configure(nirc2.NIRC2PolarimetryData())
    paths = ObslogPaths(cfg.reductions_root, cfg.date)

    raw_files = paths.link_raw_frames(cfg.raw_data_folder,
                                      frame_range=cfg.raw_range)

    paths.make_folders()
    # Frames excluded from every run of this night, each with a reason. Add
    # one with:
    #     from nirc2pol.utils.paths import record_reject
    #     record_reject(paths.rejects_file, "n0937.fits", "open AO loop")
    rejects = load_rejects(paths.rejects_file)

    saved_config = cfg.to_toml(paths.config_file)

    # Everything the reduction reports -- which flat it matched, whether the
    # band requirement was enforced, any centering fallback -- lands in one
    # file beside the products. The console still shows it too; this is the
    # copy that survives the terminal.
    run_log = start_reduction_log(paths.log_file)
    run_log.settings(instrument=type(instrument).__name__,
                     background=instrument.describe_background(),
                     config=saved_config, config_source=config_path,
                     **cfg.describe())
    log.info("background: %s", instrument.describe_background())
    if instrument.top_row_start is not None:
        log.info("beam geometry overridden: top row %s, x offset %s",
                 instrument.top_row_start, instrument.beam_x_offset)

    return instrument, paths, rejects, run_log, raw_files


def sort_raw(raw_files, instrument):
    """Classify the linked frames into darks, flats and science, by header.

    Notes
    -----
    ``cfg.raw_range`` already decided which frames are read off disk at all,
    and that is distinct from ``select_frame_range``, which picks the science
    frames in :func:`select_science`. **raw_range has to stay wide enough to
    include the darks and flats**, or the masters cannot be built.
    """
    return instrument.sort_frames(raw_files)


def build_masters(sorted_files, instrument, paths, rejects, cfg, *,
                  required_flat_type=FROM_CFG,
                  allow_flat_without_dark=FROM_CFG,
                  master_min_frames=FROM_CFG, use_master_skies=FROM_CFG,
                  sky_group_radius=FROM_CFG, sky_group_gap=FROM_CFG,
                  save_preproc=FROM_CFG):
    """Master darks, flats and skies, plus the combined bad-pixel mask.

    Returns
    -------
    Masters

    Notes
    -----
    ``instrument=`` supplies the detector mask, how to spot a polarimetric
    (critical-angle) flat set, and which flat type each band requires. The
    keywords override those per run.

    Flats come in two kinds -- dome and sky -- and each is split into
    polarimetric and not, so this can build up to four masters per filter.

    ``science_bands`` is ordering only: the flats in the science bands lead
    the inventory, but ``find_closest_flat`` still requires the filter to
    match, so it cannot change which flat a frame gets. Headers only -- the
    science frames themselves are not loaded until :func:`select_science`.

    ``save_preproc`` decides whether the masters are kept on disk. They are
    built either way, since nothing downstream runs without them, so it is
    about disk rather than about what the reduction does.

    Reads ``required_flat_type``, ``allow_flat_without_dark``,
    ``master_min_frames``, ``use_master_skies``, ``sky_group_radius``,
    ``sky_group_gap``, ``save_preproc``.
    """
    required_flat_type = _resolve(required_flat_type, cfg.required_flat_type)
    allow_flat_without_dark = _resolve(allow_flat_without_dark,
                                       cfg.allow_flat_without_dark)
    master_min_frames = _resolve(master_min_frames, cfg.master_min_frames)
    use_master_skies = _resolve(use_master_skies, cfg.use_master_skies)
    sky_group_radius = _resolve(sky_group_radius, cfg.sky_group_radius)
    sky_group_gap = _resolve(sky_group_gap, cfg.sky_group_gap)
    save_preproc = _resolve(save_preproc, cfg.save_preproc)

    darks = load_frames(sorted_files["darks"], rejects=rejects)
    master_darks, dark_masks = make_master_darks(
        darks, instrument=instrument, min_frames=master_min_frames)
    if master_darks and save_preproc:
        save_frames(paths.darks_file, master_darks)

    science_bands = {nirc2.band_of(h)
                     for h in read_headers(sorted_files["sci"])}
    science_bands.discard(None)

    master_flats, flat_masks = make_master_flats(
        load_frames(sorted_files["flats_dome"], rejects=rejects),
        load_frames(sorted_files["flats_sky"], rejects=rejects),
        master_darks,
        instrument=instrument,
        required_flat_type=required_flat_type,
        allow_flat_without_dark=allow_flat_without_dark,
        min_frames=master_min_frames,
        science_bands=science_bands,
    )
    if master_flats and save_preproc:
        save_frames(paths.flats_file, master_flats)

    master_skies = None
    if use_master_skies:
        master_skies, _ = make_master_skies(
            load_frames(sorted_files["flats_sky"], rejects=rejects),
            master_darks, instrument=instrument,
            min_frames=master_min_frames,
            group_radius_arcsec=sky_group_radius,
            group_gap_minutes=sky_group_gap)
        if master_skies and save_preproc:
            save_frames(paths.skies_file, master_skies)

    master_masks = make_master_masks(dark_masks, flat_masks)
    if master_masks and save_preproc:
        save_master_masks(paths.masks_file, master_masks)

    return Masters(master_darks, master_flats, master_skies, master_masks)


def select_science(sorted_files, paths, rejects, cfg, *,
                   select_target=FROM_CFG, select_frame_range=FROM_CFG):
    """The science frames this reduction covers.

    Notes
    -----
    **Before reducing, not after.** ``sort_frames`` classifies by elimination
    -- anything that is not a dark or a flat is science -- so the science
    bucket also holds acquisition and engineering frames, in whatever band
    they were taken. Reducing the whole bucket first means spending the time
    on frames that are about to be discarded, and failing outright on one in a
    band this night has no flat for.

    The frame table is written for the WHOLE night, since it is the thing you
    read while deciding what to select.

    ``cfg.date`` must be the UTC date the frames carry, since the masters and
    every product inherit it from the folder name; ``check_frame_dates`` says
    so if it is not.

    Reads ``select_target``, ``select_frame_range``.
    """
    select_target = _resolve(select_target, cfg.select_target)
    select_frame_range = _resolve(select_frame_range, cfg.select_frame_range)

    sci_frames = load_frames(sorted_files["sci"], rejects=rejects)
    paths.check_frame_dates(sci_frames)
    nirc2.make_frametable(sci_frames, paths.table_file)

    return select_frames(sci_frames, target=select_target,
                         frame_range=select_frame_range)


def reduce_science(sci_frames, masters, instrument, cfg, *,
                   required_flat_type=FROM_CFG,
                   allow_flat_type_mismatch=FROM_CFG, allow_no_flat=FROM_CFG,
                   skip_sky_sub=FROM_CFG, sky_group_radius=FROM_CFG,
                   sky_max_radius=FROM_CFG, replacement_method=FROM_CFG):
    """Dark, flat and bad pixels, per frame. Nothing polarimetric yet.

    Notes
    -----
    ``required_flat_type`` is passed the same value the masters were built
    with: choosing the kind for the build and then enforcing the band default
    when matching would refuse the flats it had just made.

    The bad-pixel mask is read once rather than once per frame -- it loads a
    FITS file.

    Reads ``required_flat_type``, ``allow_flat_type_mismatch``,
    ``allow_no_flat``, ``skip_sky_sub``, ``sky_group_radius``,
    ``sky_max_radius``, ``replacement_method``.
    """
    required_flat_type = _resolve(required_flat_type, cfg.required_flat_type)
    allow_flat_type_mismatch = _resolve(allow_flat_type_mismatch,
                                        cfg.allow_flat_type_mismatch)
    allow_no_flat = _resolve(allow_no_flat, cfg.allow_no_flat)
    skip_sky_sub = _resolve(skip_sky_sub, cfg.skip_sky_sub)
    sky_group_radius = _resolve(sky_group_radius, cfg.sky_group_radius)
    sky_max_radius = _resolve(sky_max_radius, cfg.sky_max_radius)
    replacement_method = _resolve(replacement_method, cfg.replacement_method)

    bad_pixel_mask = instrument.bad_pixel_mask()

    reduced_frames = []
    for frame in sci_frames:
        reduced = reduce_frame(
            frame, masters.flats, masters.darks, masters.skies, masters.masks,
            bad_pixel_mask=bad_pixel_mask,
            required_flat_types=instrument.required_flat_types,
            default_required_flat_type=instrument.default_required_flat_type,
            required_flat_type=required_flat_type,
            allow_flat_type_mismatch=allow_flat_type_mismatch,
            allow_no_flat=allow_no_flat,
            skip_sky_sub=skip_sky_sub,
            sky_group_radius_arcsec=sky_group_radius,
            sky_max_radius_arcsec=sky_max_radius,
            replacement_method=replacement_method,
            gain=instrument.gain(frame),
            saturation_limit=instrument.saturation_limit(frame),
        )
        reduced_frames.append(reduced)

    return reduced_frames


def set_beam_cutout(reduced_frames, instrument, cfg, *,
                    beam_top_row=FROM_CFG, beam_x_offset=FROM_CFG):
    """Set where ``split_beams`` cuts the two beams out. **Mutates the
    instrument**, and returns the pair.

    Notes
    -----
    Nominal per-band values, not a measurement: they only have to contain each
    beam. Whatever offset they leave between the beams is removed per frame by
    ``align_beams`` during registration, which is both simpler and more
    accurate than choosing this pair well -- the beams are rotated ~0.37 deg
    relative to each other, so their separation depends on where the source
    sits in the field and no single pair is right at more than one position.

    Keyed on the config, not on the instrument: ``cfg.configure`` leaves the
    instrument's nominal values in place rather than nulling them, so testing
    the instrument would never fire and the band-specific entry would never be
    reached.

    Mutation rather than a return value alone, because ``split_beams`` reads
    these off the instrument, several calls down, with no argument to carry
    them.

    Reads ``beam_top_row``, ``beam_x_offset``.
    """
    beam_top_row = _resolve(beam_top_row, cfg.beam_top_row)
    beam_x_offset = _resolve(beam_x_offset, cfg.beam_x_offset)

    if beam_top_row is None or beam_x_offset is None:
        band = nirc2.band_of(reduced_frames[0]) if reduced_frames else None
        nominal_top, nominal_x = type(instrument).beam_geometry_for(band)
        if beam_top_row is None:
            instrument.top_row_start = nominal_top
        if beam_x_offset is None:
            instrument.beam_x_offset = nominal_x
    else:
        instrument.top_row_start = beam_top_row
        instrument.beam_x_offset = beam_x_offset

    log.info("beam cutout: top row %d, x offset %d (nominal; align_beams "
             "removes the residual per frame)",
             instrument.top_row_start, instrument.beam_x_offset)
    return instrument.top_row_start, instrument.beam_x_offset


def subtract_dither(reduced_frames, instrument, paths, cfg, *,
                    background_method=FROM_CFG, dither_tolerance=FROM_CFG,
                    save_preproc=FROM_CFG):
    """Frame-level dither subtraction, when the background chain asks for it.

    Notes
    -----
    At frame level, before the Wollaston beams are cut out, and pair-matched
    within one HWP angle so it differences two skies rather than two
    polarization states.

    Any other stage in the chain -- annulus, mean_box -- runs later and per
    beam, inside the Stokes builder, on what this leaves behind.

    The reduced frames are saved **after** the subtraction, not before: these
    files are meant to be what the Stokes cubes were built from, and
    ``DITHSUB`` records which frame was subtracted from each.

    Reads ``background_method``, ``dither_tolerance``, ``save_preproc``.
    """
    background_method = _resolve(background_method, cfg.background_method)
    dither_tolerance = _resolve(dither_tolerance, cfg.dither_tolerance)
    save_preproc = _resolve(save_preproc, cfg.save_preproc)

    if "dither" in background_stages(background_method):
        reduced_frames = subtract_dither_background(
            reduced_frames, instrument, tolerance_arcsec=dither_tolerance)

    if save_preproc:
        for reduced in reduced_frames:
            reduced.save(os.path.join(paths.reduced_folder,
                                      reduced["RED-FN"]))

    return reduced_frames


def match_cycles(reduced_frames, instrument):
    """Group the frames into complete HWP cycles.

    Notes
    -----
    Walks the frames in time order, grouping them by the four critical angles
    and recording the mapping in each frame's ``POLCYCLE``. Incomplete
    trailing groups are dropped with a warning.
    """
    return instrument.match_modulator_cycles(reduced_frames)


def solve_fast_axis(cycles, instrument, cfg, *, fast_axis_method=FROM_CFG,
                    theta_off=FROM_CFG, fit_radii=FROM_CFG,
                    ip_method=FROM_CFG, ip_mask_radius=FROM_CFG,
                    register_method=FROM_CFG):
    """The fast axis offset, then the leakage that needs it. **Sets
    ``instrument.fast_axis_offset``**, and returns ``(theta_off, ip)``.

    Notes
    -----
    Both are chosen by naming a method. ``ReductionConfig`` has already
    checked that the pair is coherent -- the Mueller model settles the two
    together, so it is all or neither -- and that a fixed offset has a value.

    **The order is forced.** Every available IP route needs the offset as an
    input, so the leakage cannot be measured first. A *fitted* offset is
    therefore measured with the leakage still in and is biased by however much
    of it there is, which is what the warning below says.

    Reads ``fast_axis_method``, ``theta_off``, ``fit_radii``, ``ip_method``,
    ``ip_mask_radius``, ``register_method``.
    """
    fast_axis_method = _resolve(fast_axis_method, cfg.fast_axis_method)
    theta_off_cfg = _resolve(theta_off, cfg.theta_off)
    fit_radii = _resolve(fit_radii, cfg.fit_radii)
    ip_method = _resolve(ip_method, cfg.ip_method)
    ip_mask_radius = _resolve(ip_mask_radius, cfg.ip_mask_radius)
    register_method = _resolve(register_method, cfg.register_method)

    theta_off, ip = None, None
    dd_kwargs = {"register_method": register_method}

    if fast_axis_method == "mm_model":
        # Not fitted here: this applies a matrix determined elsewhere. Raises
        # NotImplementedError until that plumbing exists.
        theta_off, ip = apply_mueller_model(instrument, cycles)

    if theta_off is None:
        if fast_axis_method == "fixed":
            theta_off = theta_off_cfg
            log.info("fast axis: fixed at %g deg (nothing fitted)", theta_off)

        elif fast_axis_method == "butterfly":
            r_inner, r_outer = fit_radii
            result = fit_fast_axis_butterfly(instrument, cycles, ip=None,
                                             r_inner=r_inner, r_outer=r_outer,
                                             **dd_kwargs)
            theta_off = result.theta_off
            log.info("fast axis from the butterfly: %s", result.describe())
            if ip_method is not None:
                log.warning(
                    "Fast axis fitted with no leakage removed: every "
                    "available IP route (%r here) needs the offset as an "
                    "input, so the leakage cannot be measured first. The "
                    "offset and the leakage are degenerate, so this offset "
                    "is biased by however much IP there is. "
                    "fit_fast_axis_butterfly takes ip= if you have a "
                    "leakage from elsewhere.", ip_method)

    if ip_method == "fit_uphi_all":
        ip = fit_ip_uphi_all(instrument, cycles, theta_off,
                             mask_radius=ip_mask_radius, **dd_kwargs)
    elif ip_method == "fit_uphi_per_cycle":
        ip = [fit_ip_uphi(instrument, c, theta_off,
                          mask_radius=ip_mask_radius, **dd_kwargs)
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
    elif ip_method is None:
        log.warning("No instrumental polarization correction "
                    "(cfg.ip_method=None); the I -> Q/U leakage stays in "
                    "the products.")

    instrument.fast_axis_offset = theta_off
    return theta_off, ip


def build_stokes(cycles, instrument, cfg, theta_off, ip, *,
                 register_method=FROM_CFG, crop_size=FROM_CFG):
    """Per-cycle Stokes cubes and their median.

    Notes
    -----
    Beam splitting, alignment, per-beam background, registration and the
    double difference all happen inside the builder, once per cycle.

    The leakage is removed in the instrument frame, **before** Q/U are rotated
    to sky, so it is an argument here rather than something subtracted from a
    finished cube.

    Reads ``register_method``, ``crop_size``.
    """
    register_method = _resolve(register_method, cfg.register_method)
    crop_size = _resolve(crop_size, cfg.crop_size)

    stokes_cubes = build_stokes_cubes(instrument, cycles,
                                      fast_axis_offset=theta_off, ip=ip,
                                      register_method=register_method,
                                      crop_size=crop_size)
    return stokes_cubes, median_stokes_cube(stokes_cubes)


def write_products(stokes_cubes, median_cube, cycles, paths, cfg, *,
                   target=FROM_CFG, save_individual_cycles=FROM_CFG,
                   save_derived_quantities=FROM_CFG,
                   save_radial_stokes=FROM_CFG, dolp_min_intensity=FROM_CFG,
                   overwrite_products=FROM_CFG):
    """Write the median cube and the derived products. Returns the writer.

    Notes
    -----
    ``THETAOFF`` is already on the header: ``build_stokes_cubes`` wrote it when
    it resolved the offset it actually used, so nothing stamps it on by hand.

    ``ObslogPaths`` owns the night layout (``raw/``, ``reduced/``,
    ``sequences/``); the writer owns the product set and its provenance, so it
    is rooted at ``sequences/``.

    ``save_individual_cycles`` decides whether the per-cycle Stokes data is
    kept: one FITS per cycle, each carrying its own cycle's header. Off, the
    median cube and the derived products are all that is written, and no cycle
    can be dropped or re-combined without reducing the night again.

    Reads ``target``, ``save_individual_cycles``, ``save_derived_quantities``,
    ``save_radial_stokes``, ``dolp_min_intensity``, ``overwrite_products``.
    """
    target = _resolve(target, cfg.target)
    save_individual_cycles = _resolve(save_individual_cycles,
                                      cfg.save_individual_cycles)
    save_derived_quantities = _resolve(save_derived_quantities,
                                       cfg.save_derived_quantities)
    save_radial_stokes = _resolve(save_radial_stokes, cfg.save_radial_stokes)
    dolp_min_intensity = _resolve(dolp_min_intensity, cfg.dolp_min_intensity)
    overwrite_products = _resolve(overwrite_products, cfg.overwrite_products)

    header = cycles[0][0].header.copy()

    writer = ProductWriter(paths.sequences_folder, target=target,
                           overwrite=overwrite_products)

    if save_individual_cycles:
        writer.save_stokes_cycles(stokes_cubes, cycles, header=header)
    writer.save_median_stokes(median_cube, header=header)
    writer.save_derived_products(median_cube, header=header,
                                 derived=save_derived_quantities,
                                 radial=save_radial_stokes,
                                 dolp_min_intensity=dolp_min_intensity)
    return writer


RESUME_LEVELS = (None, "masters", "reduced")


class ResumeError(ValueError):
    """A resume that cannot be honoured, and what to do instead.

    Its own type rather than a bare ValueError so a caller can tell "this
    folder does not match your config" from "something went wrong in the
    reduction". ``nirc2pol-reduce`` catches this one and prints the message;
    anything else it lets propagate, because a traceback says more about a
    bug than a summary would.

    Subclasses ValueError, so code that already catches that keeps working.
    """


def save_master_masks(path, masks):
    """Write the shape -> bad-pixel-mask dict from :func:`make_master_masks`.

    Stored as an ordinary multi-extension master file, one extension per
    shape, so it reads back with the same machinery as the darks and flats.
    The shape is the key and is recoverable from the array itself, so nothing
    else has to be recorded.

    Written only so a later run can resume: the mask is otherwise a byproduct
    of building the darks and flats, and vanishes with them.
    """
    save_frames(path, [Frame(mask.astype(float)) for mask in masks.values()])


def load_master_masks(path):
    """Read back :func:`save_master_masks`. ``{}`` when the file is absent."""
    return {tuple(f.data.shape): f.data > 0.5 for f in load_master(path)}


def load_masters(paths):
    """Masters a previous run wrote, or None if there are none to read.

    Returns
    -------
    Masters or None

    Raises
    ------
    ValueError
        When the darks and flats are there but the bad-pixel mask is not.
        The mask is a byproduct of building them, so a folder written before
        :func:`save_master_masks` existed has no copy, and resuming without
        it would reduce with no bad-pixel correction at all -- silently, and
        differently from the run that made the folder.
    """
    darks = load_master(paths.darks_file)
    flats = load_master(paths.flats_file)
    if not darks and not flats:
        return None

    masks = load_master_masks(paths.masks_file)
    if not masks:
        raise ResumeError(
            f"{paths.darks_file} and {paths.flats_file} are there but "
            f"{paths.masks_file} is not, so the bad-pixel mask cannot be "
            "recovered -- it is a byproduct of building the masters, not "
            "something rebuilt from them. Reduce from raw once to write it, "
            "then resume.")

    skies = load_master(paths.skies_file) or None
    log.info("resumed masters: %d dark(s), %d flat(s), %s sky(s), %d mask(s)",
             len(darks), len(flats),
             len(skies) if skies else "no", len(masks))
    return Masters(darks, flats, skies, masks)


def load_reduced(paths, cfg):
    """Corrected science frames a previous run wrote, or None if absent.

    Raises
    ------
    ValueError
        When the frames on disk do not match what ``cfg`` asks for. Two ways
        that happens, both of which would otherwise reduce quietly and wrong:

        A frame outside ``cfg.select_frame_range`` -- the folder was written
        for a different selection, so resuming would reduce the wrong frames.

        ``cfg.background_method`` asks for a dither and the frames carry no
        ``DITHSUB`` -- they were written before the dither stage ran, so
        resuming would go on with **no background subtracted at all**. That
        is exactly how an L' reduction came to carry its full thermal
        pedestal, so it is refused rather than warned about.
    """
    import glob as _glob

    files = sorted(_glob.glob(os.path.join(paths.reduced_folder, "*.fits")))
    if not files:
        return None

    frames = load_frames(files)

    if cfg.select_frame_range:
        stray = [os.path.basename(f) for f in files
                 if not in_frame_range(frame_number(f),
                                       cfg.select_frame_range)]
        if stray:
            raise ResumeError(
                f"{len(stray)} frame(s) in {paths.reduced_folder} fall "
                f"outside select_frame_range={cfg.select_frame_range}, "
                f"starting with {stray[0]}. That folder was written for a "
                "different selection; reduce from raw rather than resuming "
                "into it.")

    if "dither" in background_stages(cfg.background_method):
        undithered = [f for f in frames if not f.get("DITHSUB")]
        if undithered:
            raise ResumeError(
                f"background_method asks for a dither but {len(undithered)} "
                f"of {len(frames)} frames in {paths.reduced_folder} carry no "
                "DITHSUB, so they were written before the dither ran. "
                "Resuming would continue with no background subtracted at "
                "all. Reduce from raw.")

    paths.check_frame_dates(frames)
    log.info("resumed %d reduced frame(s) from %s",
             len(frames), paths.reduced_folder)
    return frames


def run(cfg, config_path=None, resume=None):
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
    resume : {None, "masters", "reduced"}, optional
        Pick up from what a previous run of the same folder left on disk.
        None reduces from raw.

        ``"masters"`` reloads the master darks, flats, skies and bad-pixel
        mask and re-runs the science reduction -- for iterating on the
        ``reduce_frame`` settings without rebuilding calibrations.

        ``"reduced"`` reloads the corrected frames and skips straight to
        cycle matching -- the bulk of the time, and what you want when
        iterating on ``theta_off``, the leakage, the crop or the products.

        Both need the previous run to have had ``save_preproc`` on, and both
        refuse rather than guess when what is on disk does not match ``cfg``:
        see :func:`load_masters` and :func:`load_reduced`.

    Returns
    -------
    dict
        Everything the run built, so a caller can inspect any of it:
        ``instrument``, ``paths``, ``cycles``, ``theta_off``, ``ip``,
        ``stokes_cubes``, ``median_cube``, ``writer`` and ``run_log``.

    Notes
    -----
    This is the composition and nothing else: the work is in the ten stages
    above, and they are the same functions a notebook calls. There is one
    implementation of each stage, so the two cannot drift.

    A resumed run takes the same stages from a later starting point, so a
    resume and a reduction from raw produce the same products bit for bit.

    Writes as it goes -- masters, corrected frames, per-cycle and median
    Stokes products -- under ``cfg.reductions_root``. What is kept is itself a
    choice: see ``save_preproc`` and ``save_individual_cycles``.
    """
    if resume not in RESUME_LEVELS:
        raise ResumeError(f"resume must be one of {RESUME_LEVELS}, "
                          f"not {resume!r}")

    instrument, paths, rejects, run_log, raw_files = prepare_night(
        cfg, config_path)

    reduced_frames = load_reduced(paths, cfg) if resume == "reduced" else None

    if reduced_frames is None:
        if resume == "reduced":
            raise ResumeError(
                f"resume='reduced' found no frames in {paths.reduced_folder}. "
                "A previous run has to have written them, which needs "
                "save_preproc on.")

        sorted_files = sort_raw(raw_files, instrument)

        masters = load_masters(paths) if resume == "masters" else None
        if masters is None:
            if resume == "masters":
                raise ResumeError(
                    f"resume='masters' found no masters for {cfg.date} in "
                    f"{cfg.reductions_root}. A previous run has to have "
                    "written them, which needs save_preproc on.")
            masters = build_masters(sorted_files, instrument, paths, rejects,
                                    cfg)

        sci_frames = select_science(sorted_files, paths, rejects, cfg)
        reduced_frames = reduce_science(sci_frames, masters, instrument, cfg)

        # The cutout is an instrument attribute, not something stored in the
        # frames, so it is set on every path including a resume -- below.
        set_beam_cutout(reduced_frames, instrument, cfg)
        reduced_frames = subtract_dither(reduced_frames, instrument, paths,
                                         cfg)
    else:
        set_beam_cutout(reduced_frames, instrument, cfg)

    cycles = match_cycles(reduced_frames, instrument)

    theta_off, ip = solve_fast_axis(cycles, instrument, cfg)
    stokes_cubes, median_cube = build_stokes(cycles, instrument, cfg,
                                             theta_off, ip)
    writer = write_products(stokes_cubes, median_cube, cycles, paths, cfg)

    run_log.finish()

    return {"instrument": instrument, "paths": paths, "cycles": cycles,
            "theta_off": theta_off, "ip": ip, "stokes_cubes": stokes_cubes,
            "median_cube": median_cube, "writer": writer, "run_log": run_log}

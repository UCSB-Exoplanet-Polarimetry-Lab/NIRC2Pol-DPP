"""Combine several reductions into one set of products.

Two nights of the same target are joined **after** each is reduced, not
before. That is not a stylistic preference. Pooling raw frames from two
nights is unsafe with this pipeline: ``find_closest_flat`` matches on filter
and detector size and never on date, so one night's frames can be
flat-fielded with the other night's flat without a word; ``make_master_darks``
groups on exposure settings alone, merging two nights' darks into one master;
and a pooled frame list hides which night each frame came from just when the
calibrations differ most.

Reduced separately, each night gets its own darks, flats and beam alignment,
and the per-cycle Stokes cubes -- one file per cycle, each carrying its own
cycle's header -- are the right thing to join::

    nirc2pol-reduce night1.toml       # -> .../AB_Aur_Dec07/sequences/
    nirc2pol-reduce night2.toml       # -> .../AB_Aur_Dec08/sequences/
    nirc2pol-combine combined.toml    # -> .../AB_Aur_combined/

The cycles are median-combined exactly as they are within one night, which is
sound because registration puts the star at ``((ny-1)/2, (nx-1)/2)`` in every
cube: equal shapes are already aligned.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, fields

import numpy as np

from nirc2pol.polarimetry import ProductWriter, median_stokes_cube
from nirc2pol.reduction.config import TomlConfig, config_field
from nirc2pol.utils import Frame, start_reduction_log

log = logging.getLogger(__name__)


@dataclass
class CombineConfig(TomlConfig):
    """Which reductions to combine, and where the result goes."""

    TEMPLATE_HEADER = (
        "# Combine config for NIRC2Pol-DPP.",
        "#",
        "# Joins reductions that have already been run, by median-combining",
        "# their per-cycle Stokes cubes. Each input must have been reduced",
        "# on its own, with its own darks, flats and beam geometry -- that",
        "# is what makes combining across nights safe.",
    )

    reductions: list = config_field(
        ["/path/to/first_reduction", "/path/to/second_reduction"],
        "The reduction folders to combine -- each one a reductions_root "
        "that has been run. Their per-cycle Stokes cubes are read from "
        "<reduction>/sequences/<target>_stokes_cycles/.",
        "paths")
    output_root: str = config_field(
        "/path/to/combined",
        "Where the combined products are written. A folder of its own, not "
        "one of the inputs.",
        "paths")
    save_derived_quantities: bool = config_field(
        True,
        "Write PI, AoLP and DoLP from the combined cube. They mean "
        "something for any source, so they are on by default.",
        "products")
    save_radial_stokes: bool = config_field(
        False,
        "Write Q_phi and U_phi from the combined cube. Off by default: they "
        "are defined about a centre, so they measure something only when "
        "the light is scattered from something at it. Match this to what "
        "the reductions being combined were run with.",
        "products")
    target: str = config_field(
        "AB_Aur",
        "Names the product files, and selects which cycle cubes to read from "
        "each input. Must match the target the reductions used.",
        "paths")

    def default_config_path(self):
        """``combine.toml`` inside ``output_root``, beside its log."""
        return os.path.join(self.output_root, "combine.toml")

    def __post_init__(self):
        """Accept a single folder as well as a list."""
        if isinstance(self.reductions, str):
            self.reductions = [self.reductions]
        if not self.reductions:
            raise ValueError("reductions is empty: nothing to combine.")


def cycle_files(reduction_root, target):
    """The per-cycle Stokes cubes one reduction wrote.

    Parameters
    ----------
    reduction_root : str
        A reduction folder -- what a reduction config calls
        ``reductions_root``.
    target : str
        The target name that reduction used, which prefixes its products.

    Returns
    -------
    list of str
        Paths, in cycle order.

    Raises
    ------
    FileNotFoundError
        When that reduction has no per-cycle cubes. The message names the
        folder and the pattern, since the usual causes are a target name that
        does not match and a reduction run with ``save_individual_cycles``
        off -- and the second one is unrecoverable without re-reducing.
    """
    folder = os.path.join(reduction_root, "sequences",
                          f"{target}_stokes_cycles")
    pattern = os.path.join(folder, f"{target}_stokes_cycle_*.fits")
    found = sorted(glob.glob(pattern))
    if not found:
        raise FileNotFoundError(
            f"No per-cycle Stokes cubes matching {pattern}. Check the target "
            f"name matches the one that reduction used, and that it ran with "
            f"save_individual_cycles on -- with it off the per-cycle data was "
            f"never written and cannot be recovered without reducing again.")
    return found


def run(cfg, config_path=None):
    """Median-combine the per-cycle Stokes cubes of several reductions.

    Parameters
    ----------
    cfg : CombineConfig
        Which reductions, and where the result goes.
    config_path : str, optional
        Path ``cfg`` was read from, recorded in the log so the run can be
        repeated from it.

    Returns
    -------
    dict
        ``cubes`` (every per-cycle cube that went in), ``median_cube``,
        ``writer``, ``run_log`` and ``sources`` (per input: folder, how many
        cycles, band and fast axis offset).

    Notes
    -----
    Every check that can refuse the combination runs before the output
    folder is created, so a wrong target name or a shape mismatch leaves
    nothing behind. The checks that only warn run after, so that they are
    recorded in the log rather than only shown on the console.
    """
    # Read and check first: nothing is created until the inputs are known
    # to be there and to go together.
    sources, frames = [], []
    for root in cfg.reductions:
        paths = cycle_files(root, cfg.target)
        loaded = [Frame.load(p) for p in paths]
        frames.extend(loaded)
        header = loaded[0].header
        sources.append({
            "reduction": os.path.abspath(os.path.expanduser(root)),
            "ncycles": len(loaded),
            "band": _band(header),
            "theta_off": header.get("THETAOFF"),
        })
        log.info("%s: %d cycle(s), band %s, theta_off %s",
                 root, len(loaded), sources[-1]["band"],
                 sources[-1]["theta_off"])

    shapes = {f.data.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(
            f"The cubes are not all the same shape: {sorted(shapes)}. "
            f"Different detector sizes cannot be combined -- the cubes are "
            f"medianed pixel by pixel, and only equal shapes are aligned.")

    # Everything that can refuse the combination has now run, and nothing
    # has been created. The checks that only *warn* come after the log
    # exists, so they land in it -- a warning about the inputs is exactly
    # what someone reads this log for later.
    out = os.path.abspath(os.path.expanduser(cfg.output_root))
    os.makedirs(out, exist_ok=True)
    saved_config = cfg.to_toml()
    run_log = start_reduction_log(os.path.join(out, "combine.log"))
    run_log.settings(config=saved_config, config_source=config_path,
                     ninputs=len(sources), ncycles=len(frames),
                     **cfg.describe())

    bands = {s["band"] for s in sources}
    if len(bands) > 1:
        # Not raised, because a genuinely deliberate cross-band combination
        # is the caller's business -- but it produces a picture of nothing
        # unless it was meant, and no other check would catch it.
        log.warning(
            "Combining %d different bands in one cube: %s. Q_phi in one band "
            "is not the same quantity as in another, so the median of the "
            "two is not a measurement of either. Combine per band unless you "
            "have a reason.", len(bands), ", ".join(sorted(map(str, bands))))

    offsets = {s["theta_off"] for s in sources}
    if len(offsets) > 1:
        # Deliberately not a warning. The fast axis offset is a per-epoch
        # calibration: if each night's is right, combining them is right,
        # and requiring them to agree would require the instrument not to
        # change between epochs. Recorded so the product says what went in.
        log.info("Per-epoch fast axis offsets: %s",
                 ", ".join(str(o) for o in sorted(map(str, offsets))))

    median_cube = median_stokes_cube([f.data for f in frames])
    log.info("%d cycle(s) from %d reduction(s) -> median %s",
             len(frames), len(sources), median_cube.shape)

    header = frames[0].header.copy()
    header["NCOMBINE"] = (len(frames), "HWP cycles median-combined")
    header["NREDUCT"] = (len(sources), "reductions combined")

    params = {
        "nreductions": len(sources),
        "ncycles": len(frames),
        "reductions": [s["reduction"] for s in sources],
        "cycles_each": [s["ncycles"] for s in sources],
        "bands": sorted(map(str, bands)),
        "theta_off_each": [s["theta_off"] for s in sources],
    }
    writer = ProductWriter(out, target=cfg.target)
    writer.save_median_stokes(median_cube, header=header, **params)
    writer.save_derived_products(median_cube, header=header,
                                 derived=cfg.save_derived_quantities,
                                 radial=cfg.save_radial_stokes, **params)

    run_log.finish()
    return {"cubes": frames, "median_cube": median_cube, "writer": writer,
            "run_log": run_log, "sources": sources}


def _band(header):
    """The observing band a product was taken in.

    Uses the instrument's own rule (``FWINAME``, falling back to the first
    token of ``FILTER``) so "same band" means the same thing here as it does
    when a flat is matched to a frame.
    """
    from nirc2pol.instruments.nirc2 import band_of
    return band_of(header)

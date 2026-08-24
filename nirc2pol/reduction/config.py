"""One place for every choice a reduction makes.

A reduction has a lot of knobs, and they used to live as loose constants at
the top of ``examples/process_polmode.py``, which made "where do the options
live?" an unanswerable question. They live here now: :class:`ReductionConfig`
holds every field, its default, what it means and what values it accepts, and
validates them on construction.

The per-night file you edit is TOML, and it is *generated* from this class by
:meth:`ReductionConfig.template`, so a new option appears in the template
automatically and the two cannot drift apart.

Instrument constants are a different thing and live in
``nirc2pol/instruments/nirc2.toml``: those are properties of the instrument,
the same for everyone reducing that night. This file is the choices *you*
make.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields

log = logging.getLogger(__name__)

#: Values in TOML that mean Python ``None``. TOML has no null, so a key that
#: is deliberately switched off is spelled ``"none"``; omitting the key
#: instead means "use the default", which is not always the same thing.
NONE_STRINGS = {"none", "null", ""}

#: Options that have moved, so a config written against an older version
#: gets told what to change rather than a list of every key that exists.
RENAMED_OPTIONS = {
    "observations_root": "raw_data_folder (where the frames are) plus "
                         "reductions_root (where this reduction is written) "
                         "-- they used to be the same folder, and are not "
                         "the same thing",
}

BACKGROUND_METHODS = ("mean_box", "annulus", "dither", None)
REGISTER_METHODS = ("smooth_peak", "quantile_peak", "max", "min", "gaussian",
                    "centroid", "silhouette", "symmetry", "wings", "crosscorr")
FLAT_TYPES = ("SKY", "DOME", None)
REPLACEMENT_METHODS = ("interpolation", "median")
FAST_AXIS_METHODS = ("mm_model", "butterfly", "fixed")
IP_METHODS = ("mm_model", "fit_uphi_per_cycle", "fit_uphi_all", None)

def config_field(default, doc, group, choices=None, unit=None):
    """A config field: its default, what it means, and what it accepts.

    The default is kept in the metadata as well, because a list default has
    to go through ``default_factory`` and is then not readable from
    ``field.default`` -- and :meth:`ReductionConfig.template` needs it.
    """
    meta = {"doc": doc, "group": group, "choices": choices, "unit": unit,
            "default": default}
    if isinstance(default, (list, dict, set)):
        return field(default_factory=lambda d=default: list(d), metadata=meta)
    return field(default=default, metadata=meta)


class TomlConfig:
    """TOML round-tripping for a config dataclass.

    The half of a config that is not about *what* the options are: reading
    one, writing one, and generating the annotated template that lists every
    option with its default. Shared, because there is more than one kind of
    config -- a reduction and a combine -- and only one of these should exist.

    A subclass is a ``@dataclass`` whose fields are built with
    :func:`config_field`, and which sets the two class attributes below.

    Attributes
    ----------
    TEMPLATE_HEADER : tuple of str
        Comment lines at the top of a generated template.
    RENAMED : dict
        Old option name -> what it became, so a config written against an
        older version is told what to change rather than handed a list of
        every key that exists.
    """

    TEMPLATE_HEADER = ("# Config for NIRC2Pol-DPP.",)
    RENAMED = {}

    @classmethod
    def from_toml(cls, path):
        """Read a reduction config. Keys not given keep their defaults.

        Sections are cosmetic -- they group the options for reading -- so a
        flat file works too. ``"none"`` means Python None; omitting a key
        means "use the default", which is not always the same thing.
        """
        import tomllib

        with open(path, "rb") as fh:
            raw = tomllib.load(fh)

        flat = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

        known = {f.name for f in fields(cls)}
        unknown = set(flat) - known
        if unknown:
            renamed = [f"{k} is now {cls.RENAMED[k]}"
                       for k in sorted(unknown) if k in cls.RENAMED]
            if renamed:
                # No path in the message: the caller that has one prefixes
                # it, and the other validation errors here do the same.
                raise ValueError(
                    "uses options that have been renamed: "
                    + "; ".join(renamed) + ".")
            raise ValueError(
                f"{path} sets options this version does not know: "
                f"{sorted(unknown)}. Known options: {sorted(known)}.")

        clean = {k: (None if isinstance(v, str) and v.lower() in NONE_STRINGS
                     else v)
                 for k, v in flat.items()}
        log.info("%s: %s (%d options set)", cls.__name__, path, len(clean))
        return cls(**clean)

    def describe(self):
        """Every option and its value, for the run log."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def default_config_path(self):
        """Where this config belongs when no path is given.

        Overridden by each config to name a file inside the folder it
        writes to, so ``to_toml()`` with no argument puts the config with
        the products it describes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no default location for its config, "
            f"so to_toml needs a path.")

    def to_toml(self, path=None):
        """Write this config to ``path`` as TOML, and return the path.

        With no path it goes to :meth:`default_config_path` -- inside the
        folder this config writes to, beside the products it describes.

        The counterpart to :meth:`from_toml`, and it round-trips: reading back
        what this writes gives an equal config.

        Its reason for existing is provenance. A config built inline -- in a
        notebook, say -- exists only in memory, so a reduction log can record
        the *values* but has nothing to point at, and nobody can re-run it
        from the log alone. Writing it out first gives the run a real file to
        name, which is what ``run_log.settings(config=...)`` records.
        """
        name = type(self).__name__
        path = self.default_config_path() if path is None else path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        text = self._render({f.name: getattr(self, f.name)
                             for f in fields(self)},
                            header=(f"# Written by {name}.to_toml.\n#\n"
                                    f"# Read it back with "
                                    f"{name}.from_toml(path)."))
        with open(path, "w") as fh:
            fh.write(text)
        log.info("%s written to %s", type(self).__name__, path)
        return path

    @classmethod
    def template(cls):
        """An annotated TOML file listing every option, default and choice.

        Generated from the dataclass, so it cannot fall behind it.
        """
        return cls._render({f.name: f.metadata["default"] for f in fields(cls)})

    @classmethod
    def _render(cls, values, header=None):
        """Render ``{field: value}`` as annotated TOML, grouped by section."""
        import textwrap

        out = list(cls.TEMPLATE_HEADER) + [""]
        if header is not None:
            out = header.splitlines() + ["#", ""]

        groups = {}
        for f in fields(cls):
            groups.setdefault(f.metadata["group"], []).append(f)

        for group, group_fields in groups.items():
            out.append(f"[{group}]")
            for f in group_fields:
                doc = f.metadata["doc"]
                if f.metadata.get("unit"):
                    doc += f" [{f.metadata['unit']}]"
                out += [("# " + line).rstrip() for line in _wrap(doc)]
                if f.metadata.get("choices"):
                    # Only worth listing when the prose above did not already
                    # walk through them one by one; otherwise it is a second
                    # copy of the same list.
                    named = sum(
                        1 for c in f.metadata["choices"]
                        if c is not None and f"  {c} " in doc)
                    if named < len(f.metadata["choices"]) - 1:
                        choices = ", ".join(
                            "none" if c is None else repr(c)
                            for c in f.metadata["choices"])
                        out += [("# " + line).rstrip()
                                for line in _wrap("choices: " + choices)]
                out.append(f"{f.name} = {_toml_value(values[f.name])}")
                out.append("")
        return "\n".join(out)


@dataclass
class ReductionConfig(TomlConfig):
    """Every choice one reduction makes, with defaults and allowed values."""

    TEMPLATE_HEADER = (
        "# Reduction config for NIRC2Pol-DPP.",
        "#",
        "# Every option this pipeline takes, with its default. Delete a",
        "# line to accept the default; set it to \"none\" to switch the",
        "# option off, which is not always the same thing.",
        "#",
        "# Instrument constants (plate scale, detector epochs, beam",
        "# geometry seed) are NOT here -- they live in",
        "# nirc2pol/instruments/nirc2.toml, because they are",
        "# properties of the instrument rather than choices about",
        "# this reduction.",
    )
    RENAMED = RENAMED_OPTIONS

    # ---- where the data is -------------------------------------------
    raw_data_folder: str = config_field(
        "/path/to/raw/frames",
        "Where the raw frames actually are. Read only -- the pipeline never "
        "writes here, so this can be an archive or shared space you do not "
        "own. The frames it reads are symlinked into "
        "reductions_root/raw/.",
        "paths")
    reductions_root: str = config_field(
        "/path/to/my_reduction",
        "Where THIS reduction is written: raw/ (links), reduced/, plots/, "
        "sequences/ and the masters. One folder per reduction, so two goes "
        "at the same night do not overwrite each other.",
        "paths")
    date: str = config_field(
        "2025-12-08",
        "The night, as DATE-OBS records it: UTC. A Keck night runs 04:00-16:00 "
        "UTC, so one UTC date names a whole night, one day after the HST "
        "evening. It locates nothing -- it names the masters, log and frame "
        "table, and is checked against the frames' own DATE-OBS.",
        "paths")
    target: str = config_field(
        "AB_Aur",
        "Names the output files only. What the reduction covers is set by "
        "select_target / select_frame_range.",
        "paths")

    # ---- which frames ------------------------------------------------
    select_target: str = config_field(
        None,
        "Keep only frames whose TARGNAME matches, ignoring case, spaces, "
        "underscores and hyphens. none keeps everything.",
        "selection")
    raw_range: list = config_field(
        None,
        "Which raw files to consider at all, as inclusive observation numbers "
        "read from the filenames -- one range [857, 993] or several. This is "
        "not the same question as select_frame_range: raw_range decides what "
        "is read off disk and sorted into darks, flats and science, so it has "
        "to be wide enough to include the calibrations. none reads every raw "
        "file in the night folder, which is what you want unless the folder "
        "holds more than one night's worth.",
        "selection")
    select_frame_range: list = config_field(
        None,
        "Keep only these observation numbers, read from the filenames. One "
        "range [932, 939], or several [[857, 900], [915, 930]]. Inclusive. "
        "none keeps everything.",
        "selection")

    # ---- background ---------------------------------------------------
    background_method: object = config_field(
        None,
        "How the sky/thermal background is removed, per Wollaston beam, "
        "inside the Stokes builder.\n"
        "\n"
        "  mean_box   the mean of a fixed box, set in background_box. Wants "
        "a corner with no source in it.\n"
        "  annulus    the median of an annulus around the star, set in "
        "background_annulus. Follows a background that varies across the "
        "frame better than a box in one corner.\n"
        "  dither     subtract the matching dithered exposure, so the sky is "
        "measured at the same time as the source. Needs a dither pattern in "
        "the data.\n"
        "  none       leave it in.\n"
        "\n"
        "Several may be chained, in the order applied, e.g.\n"
        "  background_method = [\"dither\", \"annulus\"]\n"
        "\n"
        "That combination is the one to reach for at L'. The dither removes "
        "the thermal pedestal AND its structure, which nothing else can, "
        "but it leaves a residual because the sky changed a little between "
        "the two exposures -- roughly a constant offset per beam. The "
        "annulus (or mean_box) then takes that out. On the 2025-12-06 "
        "standard the leftover was worth +22 ADU/px in U at the star, "
        "which is nothing beside a 1.2e5 core but sums over a 9000 px "
        "annulus into more signal than the star has out there.\n"
        "\n"
        "dither must be listed first if present: it runs on whole frames "
        "before the beams are cut out, so any other order is refused rather "
        "than quietly rearranged.\n"
        "\n"
        "L'/M want dither or mean_box, where the thermal background "
        "dominates; JHK want annulus or mean_box. Defaults to none because "
        "the right answer depends on the band and the data, and the region "
        "a box or annulus uses cannot be guessed -- a choice to make, not "
        "one to inherit. On-sky data almost always needs one.",
        "background")
    dither_tolerance: float = config_field(
        2.0,
        "How far apart two pointings may be and still count as the same "
        "dither position -- used ONLY when a frame has no RAOFF/DECOFF and "
        "the actual RA/DEC has to be used instead. Commanded offsets carry "
        "no jitter, so they are compared exactly.\n"
        "\n"
        "Keep this well BELOW the dither throw. A throw is only a few arcsec "
        "-- 3.0 on the 2025-12-06 L' data -- and a tolerance near it merges "
        "the two positions into one, leaving nothing to subtract. That is "
        "raised rather than passed over.",
        "background", unit="arcsec")
    background_box: list = config_field(
        None,
        "[ylow, yhigh, xlow, xhigh] for mean_box, in a source-free corner.",
        "background", unit="px")
    background_annulus: list = config_field(
        None, "[r_inner, r_outer] for annulus.", "background", unit="px")
    required_flat_type: str = config_field(
        None,
        "Override which KIND of flat this band requires -- SKY (twilight) or "
        "DOME (the dome screen). The instrument sets this per band already: "
        "sky at L'/M, where the dome lamp is swamped by the thermal "
        "background, and dome at JHK. none keeps that rule.\n"
        "\n"
        "Reducing with the wrong kind gives a wrong answer that still looks "
        "reasonable, so the rule is enforced rather than preferred -- a "
        "mismatch raises. Set this when you have a considered reason, such "
        "as sky flats being the only usable ones you took in JHK.",
        "calibration", choices=FLAT_TYPES)
    allow_flat_without_dark: bool = config_field(
        True,
        "Build a master flat even when no dark matches its exposure, tagging "
        "it +NODARK. On by default, because darks are routinely not taken "
        "for every flat and refusing meant a reduction stopped over a flat "
        "in a band it was never going to use.\n"
        "\n"
        "What keeps this safe is the ranking, not the flag: within a filter "
        "a +NODARK flat loses to EVERY dark-subtracted flat of the required "
        "type, even a non-polarimetric one built from far fewer frames. It "
        "is used only when nothing better exists in that filter -- and then "
        "it is the only alternative to no flat at all. It still warns when "
        "built, and the frame it calibrates records FLATTYPE ending +NODARK, "
        "because the dark current it carries survives normalization and "
        "divides into everything.",
        "calibration")
    allow_flat_type_mismatch: bool = config_field(
        False,
        "Downgrade a wrong-kind flat from an error to a warning, and reduce "
        "with it anyway. The frame records FLATMISM so the product says it "
        "happened. Where required_flat_type changes WHICH kind is demanded, "
        "this stops the demand being enforced at all -- so prefer setting "
        "required_flat_type when you know what you want.",
        "calibration")
    allow_no_flat: bool = config_field(
        False,
        "Reduce with no flat at all, dividing by ones. Off by default "
        "because it leaves the detector response in the data and is easy to "
        "miss afterwards.",
        "calibration")
    master_min_frames: int = config_field(
        3,
        "Fewest frames that can make a master dark, flat or sky. A group "
        "smaller than this is skipped with a warning -- and if that leaves "
        "no master at all, the reduction carries on without one, so a night "
        "short of calibration frames is quiet rather than fatal. Lower it "
        "deliberately, knowing a two-frame master barely rejects a cosmic "
        "ray.",
        "calibration")
    replacement_method: str = config_field(
        "interpolation",
        "How bad pixels are filled once identified.\n"
        "\n"
        "  interpolation  from the neighbours, which keeps a gradient "
        "across the pixel.\n"
        "  median         the median of the surrounding box, which is "
        "flatter but more robust where a whole region is bad.",
        "calibration", choices=REPLACEMENT_METHODS)
    skip_sky_sub: bool = config_field(
        True,
        "Do NOT subtract the master skies from each science frame. Reads as "
        "a double negative next to use_master_skies, and the two pair up: "
        "use_master_skies decides whether the masters are BUILT, this "
        "decides whether they are APPLIED. Both defaults off, so the sky is "
        "removed by background_method inside the Stokes builder instead -- "
        "per Wollaston beam, which is where it belongs. Set "
        "use_master_skies = true and skip_sky_sub = false to subtract a "
        "master sky frame the old way.",
        "background")
    use_master_skies: bool = config_field(
        False,
        "Subtract dedicated master sky frames. Off by default: combined with "
        "a mean-box or annulus background it removes the pedestal twice.",
        "background")

    # ---- beams and registration ---------------------------------------
    sky_group_radius: float = config_field(
        60.0,
        "How far the telescope may move before its sky frames count as a "
        "different set. Sky sets used to be merged whenever their exposure "
        "settings matched, whatever the telescope was pointing at, so three "
        "sets at three targets became one master and no later choice could "
        "recover them.",
        "skies", unit="arcsec")
    sky_group_gap: float = config_field(
        30.0,
        "How long a pause may be before the sky frames after it count as a "
        "different set. Position alone cannot separate two sets at the same "
        "place hours apart, and on L' the thermal background has moved on by "
        "then.",
        "skies", unit="min")
    sky_max_radius: float = config_field(
        600.0,
        "Warn when the nearest usable sky is further from the science frame "
        "than this. A sky that far off measures a different piece of "
        "atmosphere, and on L' the sky is most of what is being subtracted.",
        "skies", unit="arcsec")
    beam_top_row: int = config_field(
        None,
        "Where split_beams cuts the top beam out, if you are overriding it. "
        "none uses the nominal value for the band, which is the intended "
        "path.\n"
        "\n"
        "This is NOT a calibration and does not need to be exact. Whatever "
        "offset the cut leaves between the two beams is measured and "
        "removed per frame by align_beams during registration. Set it only "
        "if a beam is being cut off entirely -- which would mean the band "
        "is missing from the [beam_geometry] table in nirc2.toml.",
        "geometry", unit="px")
    beam_x_offset: int = config_field(None, "See beam_top_row.", "geometry",
                                      unit="px")
    register_method: str = config_field(
        "smooth_peak",
        "How the star is located, so both Wollaston beams can be shifted "
        "onto the image centre together. The right one depends on what the "
        "source looks like, and the wrong one fails quietly -- a centre off "
        "by a pixel leaks disk signal straight into U_phi.\n"
        "\n"
        "  smooth_peak    a point source. Peak of the Gaussian-smoothed "
        "image, refined by a Gaussian fit. No guess or tuning needed.\n"
        "  min            a SATURATED core. The centre reads dark, so "
        "smooth_peak hops between rim maxima of the donut and doubles the "
        "PSF; this takes the darkest pixel near the core instead.\n"
        "  wings          behind a CORONAGRAPH. Masks the core, which "
        "carries no position information, and aligns on the wings.\n"
        "  symmetry       a round RESOLVED body (planet, moon). "
        "Cross-correlates the image with its own 180-degree rotation; needs "
        "no template, threshold or guess.\n"
        "  silhouette     a resolved body with hotspots or albedo features. "
        "Geometric centre of the thresholded region, ignoring how bright "
        "each part of it is.\n"
        "  centroid       an extended source. Flux-weighted centroid above a "
        "threshold; still pulled about by bright surface features.\n"
        "  crosscorr      aligning a sequence to one reference. Needs a "
        "template= image, so it is not usable from this file alone.\n"
        "  gaussian       a 2D Gaussian fit, seeded from quantile_peak.\n"
        "  quantile_peak  the brightest pixel, ignoring outliers -- robust "
        "to isolated hot pixels where max is not.\n"
        "  max            the brightest pixel, full stop.\n"
        "  none           do not register at all. For a source too faint to "
        "find in a single frame.\n"
        "\n"
        "nirc2pol.reduction.registration documents the algorithms "
        "themselves, and takes any callable f(data) -> (cy, cx) too.",
        "geometry", choices=REGISTER_METHODS)

    # ---- fast axis -----------------------------------------------------
    fast_axis_method: str = config_field(
        "mm_model",
        "How theta_off is obtained. mm_model takes it from the Mueller matrix "
        "model, which settles the IP too (so ip_method must also be "
        "mm_model) and is NOT IMPLEMENTED yet -- it raises. butterfly fits it "
        "from the data. fixed uses theta_off below.",
        "fast_axis", choices=FAST_AXIS_METHODS)
    theta_off: float = config_field(
        None,
        "Fast axis offset, used when fast_axis_method = fixed. There is no "
        "trusted automatic source: an HWP ladder on an internal source "
        "returns theta_off + chi/2 with chi unknown.",
        "fast_axis", unit="deg")
    fit_radii: list = config_field(
        [25.0, 150.0],
        "[r_inner, r_outer] annulus for the butterfly fit. This works on "
        "U_phi, where a tangentially polarized disk contributes nothing by "
        "definition, so it should SPAN the disk and need only clear the core.",
        "fast_axis", unit="px")

    # ---- instrumental polarization -------------------------------------
    ip_method: str = config_field(
        "mm_model",
        "How the I -> Q/U leakage is measured, named <method>_<scope>: the "
        "scope is also the level it is applied at. mm_model is NOT "
        "IMPLEMENTED and raises. Both fit_uphi routes ASSUME AN AZIMUTHALLY "
        "POLARIZED SOURCE and return a confident, meaningless number where "
        "that is the hypothesis under test. none leaves it uncorrected.",
        "instrumental_polarization", choices=IP_METHODS)
    ip_mask_radius: float = config_field(
        22.0,
        "Radius excluded from the fit_uphi annulus, covering the saturated or "
        "occulted core.",
        "instrumental_polarization", unit="px")

    # ---- what is kept on disk ------------------------------------------
    save_preproc: bool = config_field(
        True,
        "Keep the master darks/flats/skies and every dark-, flat- and "
        "bad-pixel-corrected science frame. They are built either way, since "
        "nothing downstream can run without them; this only says whether "
        "they are written. Turn it off for a quick re-reduction whose "
        "calibration is already on disk.",
        "products")
    overwrite_products: bool = config_field(
        True,
        "Overwrite products already in the reduction folder. On by default, "
        "so re-running a reduction replaces its own output rather than "
        "failing halfway. Turn it off to make a folder write-once, and a "
        "second run will stop rather than quietly replace what is there.",
        "products")
    save_derived_quantities: bool = config_field(
        True,
        "Write PI, AoLP and DoLP -- polarized intensity, and the angle and "
        "degree of linear polarization. These mean something for any source, "
        "so they are on by default.",
        "products")
    save_radial_stokes: bool = config_field(
        False,
        "Write Q_phi and U_phi. Off by default because they are defined "
        "about a CENTRE: they mean something when the light is scattered "
        "from something at that centre, which is to say a disk. For a point "
        "source -- a polarization standard, say -- they are a rotation of Q "
        "and U about an arbitrary point, and Q_phi is not a measurement of "
        "anything. Turn it on for a disk, where U_phi is also the null "
        "channel worth judging the reduction by.",
        "products")
    dolp_min_intensity: float = config_field(
        0.001,
        "DoLP is written as NaN wherever |I| is below this fraction of the "
        "peak. DoLP is PI/I, so where I approaches zero -- most of a frame, "
        "once the background is gone -- it divides noise by noise and runs "
        "to enormous values that leave any display scaled by them showing "
        "nothing. Nothing is hidden: a ratio at I close to 0 carried no "
        "information. Raise it for a faint source, or set 0 to divide "
        "everywhere.",
        "products")
    save_individual_cycles: bool = config_field(
        True,
        "Keep the per-cycle Stokes cubes -- one FITS per HWP cycle in "
        "<target>_stokes_cycles/, each carrying its own cycle's header. Off, "
        "the median cube and the derived products are all that is written, "
        "and no cycle can be dropped or re-combined afterwards without "
        "reducing the night again.",
        "products")

    def __post_init__(self):
        """Check every enumerated field, and the rules that span fields."""
        for f in fields(self):
            choices = f.metadata.get("choices")
            value = getattr(self, f.name)
            if choices is None or value in choices:
                continue
            raise ValueError(
                f"{f.name} must be one of {choices}, not {value!r}")

        mm = (self.fast_axis_method == "mm_model", self.ip_method == "mm_model")
        if any(mm) and not all(mm):
            raise ValueError(
                "The Mueller matrix model gives the fast axis offset and the "
                "instrumental polarization together -- they are both terms of "
                "the same matrix -- so fast_axis_method and ip_method must "
                f"either both be 'mm_model' or neither. Got "
                f"fast_axis_method={self.fast_axis_method!r}, "
                f"ip_method={self.ip_method!r}.")

        if self.fast_axis_method == "fixed" and self.theta_off is None:
            raise ValueError(
                "fast_axis_method = 'fixed' uses theta_off, which is not set.")

        # background_method may name several stages, so it is validated by
        # background_stages rather than by the enumerated-choices loop above.
        from nirc2pol.reduction.sky import background_stages

        stages = background_stages(self.background_method)
        for stage in stages:
            if stage not in BACKGROUND_METHODS:
                raise ValueError(
                    f"background_method stage {stage!r} is not one of "
                    f"{[m for m in BACKGROUND_METHODS if m]}.")
        if "mean_box" in stages and not self.background_box:
            raise ValueError(
                "background_method includes 'mean_box', which needs "
                "background_box, [ylow, yhigh, xlow, xhigh] over a "
                "source-free region.")
        if "annulus" in stages and not self.background_annulus:
            raise ValueError(
                "background_method includes 'annulus', which needs "
                "background_annulus, [r_inner, r_outer].")

    # ------------------------------------------------------------------
    def default_config_path(self):
        """``reduction_<date>.toml`` inside ``reductions_root``.

        The same name :class:`nirc2pol.utils.ObslogPaths` derives, so a
        config written from a notebook lands exactly where a config written
        by a run does.
        """
        return os.path.join(self.reductions_root,
                            f"reduction_{self.date}.toml")

    def configure(self, instrument):
        """Apply the per-dataset settings to an instrument, and return it.

        These five attributes are the only ones a *reduction* chooses;
        everything else on the instrument -- plate scale, detector epochs,
        the rotation model -- is a property of the hardware and comes from
        ``nirc2pol/instruments/nirc2.toml``.

        You do not need a subclass to set them. They are read through
        ``self``, so assigning them on the instance works, which is what this
        does. Subclass only when you want to change *behaviour*: override a
        method, add an instrument, model a night the base class cannot
        describe. Setting five values is not a reason.

        The beam cutout is assigned even when None, which is the usual case:
        None means "use the nominal value for this band", which
        :func:`nirc2pol.polmode.run` fills in from the instrument's
        ``beam_geometry_for``. Setting it explicitly only moves where the
        beams are cut; it is not a calibration, because
        :func:`nirc2pol.reduction.align_beams` removes whatever offset the
        cut leaves.
        """
        instrument.background_method = self.background_method
        instrument.background_box = self.background_box
        instrument.background_annulus = self.background_annulus
        instrument.top_row_start = self.beam_top_row
        instrument.beam_x_offset = self.beam_x_offset
        return instrument


def _wrap(doc, width=72):
    """Wrap a field's doc for a comment block, keeping its own line breaks.

    A doc written as one paragraph wraps as one paragraph. A doc written
    with line breaks and indentation -- a list of what each choice means,
    say -- keeps them, with continuation lines indented to match, so a
    generated config can hold a small table and still read as one.
    """
    import textwrap

    lines = []
    for line in doc.splitlines():
        if not line.strip():
            lines.append("")
            continue
        indent = " " * (len(line) - len(line.lstrip()))
        # A list item hangs its continuations under the text; a plain
        # paragraph does not hang at all.
        hang = indent + "    " if indent else ""
        lines += textwrap.wrap(line.strip(), width,
                               initial_indent=indent,
                               subsequent_indent=hang)
    return lines


def _toml_value(value):
    """Render a default as TOML."""
    if value is None:
        return '"none"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return repr(value)

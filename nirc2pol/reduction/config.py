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

    def to_toml(self, path):
        """Write this config to ``path`` as TOML, and return the path.

        The counterpart to :meth:`from_toml`, and it round-trips: reading back
        what this writes gives an equal config.

        Its reason for existing is provenance. A config built inline -- in a
        notebook, say -- exists only in memory, so a reduction log can record
        the *values* but has nothing to point at, and nobody can re-run it
        from the log alone. Writing it out first gives the run a real file to
        name, which is what ``run_log.settings(config=...)`` records.
        """
        name = type(self).__name__
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
                out += ["# " + line
                        for line in textwrap.wrap(doc, 72)]
                if f.metadata.get("choices"):
                    choices = ", ".join(
                        "none" if c is None else repr(c)
                        for c in f.metadata["choices"])
                    out += ["# " + line for line in
                            textwrap.wrap("choices: " + choices, 72)]
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
    background_method: str = config_field(
        None,
        "How the sky/thermal background is removed, per Wollaston beam, "
        "inside the Stokes builder. L'/M want dither or mean_box; JHK want "
        "annulus or mean_box. Defaults to none because the right answer "
        "depends on the band and the data, and the region a mean_box or "
        "annulus uses cannot be guessed -- so this is a choice to make, not "
        "one to inherit. On-sky data almost always needs one.",
        "background", choices=BACKGROUND_METHODS)
    background_box: list = config_field(
        None,
        "[ylow, yhigh, xlow, xhigh] for mean_box, in a source-free corner.",
        "background", unit="px")
    background_annulus: list = config_field(
        None, "[r_inner, r_outer] for annulus.", "background", unit="px")
    use_master_skies: bool = config_field(
        False,
        "Subtract dedicated master sky frames. Off by default: combined with "
        "a mean-box or annulus background it removes the pedestal twice.",
        "background")

    # ---- beams and registration ---------------------------------------
    beam_top_row: int = config_field(
        None,
        "Beam geometry, if you are overriding it. none measures it from the "
        "data, which is the intended path -- the separation moves between "
        "epochs and a stale value fails silently.",
        "geometry", unit="px")
    beam_x_offset: int = config_field(None, "See beam_top_row.", "geometry", unit="px")
    register_method: str = config_field(
        "smooth_peak",
        "Star-centering algorithm. Use min for a saturated core: the default "
        "smooth_peak hops between rim maxima of the donut and doubles the PSF.",
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

        if self.background_method == "mean_box" and not self.background_box:
            raise ValueError(
                "background_method = 'mean_box' needs background_box, "
                "[ylow, yhigh, xlow, xhigh] over a source-free region.")
        if self.background_method == "annulus" and not self.background_annulus:
            raise ValueError(
                "background_method = 'annulus' needs background_annulus, "
                "[r_inner, r_outer].")

    # ------------------------------------------------------------------
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

        The beam geometry is assigned even when None, which is the usual
        case: None means "measure it from the data", and
        :func:`nirc2pol.reduction.fit_beam_geometry` fills it in
        afterwards. Call this before measuring, not after, or it will
        overwrite what was measured.
        """
        instrument.background_method = self.background_method
        instrument.background_box = self.background_box
        instrument.background_annulus = self.background_annulus
        instrument.top_row_start = self.beam_top_row
        instrument.beam_x_offset = self.beam_x_offset
        return instrument


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

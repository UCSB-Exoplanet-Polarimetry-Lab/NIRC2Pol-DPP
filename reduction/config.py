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
``instruments/nirc2.toml``: those are properties of the instrument, the same
for everyone reducing that night. This file is the choices *you* make.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields

log = logging.getLogger(__name__)

#: Values in TOML that mean Python ``None``. TOML has no null, so a key that
#: is deliberately switched off is spelled ``"none"``; omitting the key
#: instead means "use the default", which is not always the same thing.
NONE_STRINGS = {"none", "null", ""}

BACKGROUND_METHODS = ("mean_box", "annulus", "dither", None)
REGISTER_METHODS = ("smooth_peak", "quantile_peak", "max", "min", "gaussian",
                    "centroid", "silhouette", "symmetry", "wings", "crosscorr")
FAST_AXIS_METHODS = ("mm_model", "butterfly", "fixed")
IP_METHODS = ("mm_model", "fit_uphi_per_cycle", "fit_uphi_all", None)

#: Names that were once valid, and what to do instead. Kept so a config
#: written against an older version fails with directions rather than a bare
#: "not a valid choice".
RETIRED = {
    "butterfly_joint":
        "the butterfly fit now determines the fast axis offset only, and the "
        "leakage is chosen separately with ip_method",
    "edge_annulus_all":
        "the edge_annulus routes are withdrawn: sum(Q)/sum(I) needs an "
        "annulus that is both disk-free and bright, and on AB Aur no such "
        "annulus exists -- use fit_uphi_all",
    "edge_annulus_per_cycle": "withdrawn, see edge_annulus_all",
    "edge_annulus_per_frame": "withdrawn, see edge_annulus_all",
    "fit_uphi_per_frame":
        "one exposure gives a single difference, +-Q or +-U, never both, so "
        "U_phi cannot be formed from it -- use fit_uphi_per_cycle",
}


def _f(default, doc, group, choices=None, unit=None):
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


@dataclass
class ReductionConfig:
    """Every choice one reduction makes, with defaults and allowed values."""

    # ---- where the data is -------------------------------------------
    observations_root: str = _f(
        "/path/to/data_polmode",
        "Root folder holding ONE SUBFOLDER PER NIGHT -- not the folder your "
        "FITS files are in. Frames are read from <root>/<date>/raw/.",
        "paths")
    date: str = _f(
        "2025-12-08",
        "The night, as DATE-OBS records it: UTC. A Keck night runs 04:00-16:00 "
        "UTC, so one UTC date names a whole night, one day after the HST "
        "evening.",
        "paths")
    target: str = _f(
        "AB_Aur",
        "Names the output files only. What the reduction covers is set by "
        "select_target / select_frame_range.",
        "paths")

    # ---- which frames ------------------------------------------------
    select_target: str = _f(
        None,
        "Keep only frames whose TARGNAME matches, ignoring case, spaces, "
        "underscores and hyphens. none keeps everything.",
        "selection")
    select_frame_range: list = _f(
        None,
        "Keep only these observation numbers, read from the filenames. One "
        "range [932, 939], or several [[857, 900], [915, 930]]. Inclusive. "
        "none keeps everything.",
        "selection")

    # ---- background ---------------------------------------------------
    background_method: str = _f(
        None,
        "How the sky/thermal background is removed, per Wollaston beam, "
        "inside the Stokes builder. L'/M want dither or mean_box; JHK want "
        "annulus or mean_box. Defaults to none because the right answer "
        "depends on the band and the data, and the region a mean_box or "
        "annulus uses cannot be guessed -- so this is a choice to make, not "
        "one to inherit. On-sky data almost always needs one.",
        "background", choices=BACKGROUND_METHODS)
    background_box: list = _f(
        None,
        "[ylow, yhigh, xlow, xhigh] for mean_box, in a source-free corner.",
        "background", unit="px")
    background_annulus: list = _f(
        None, "[r_inner, r_outer] for annulus.", "background", unit="px")
    use_master_skies: bool = _f(
        False,
        "Subtract dedicated master sky frames. Off by default: combined with "
        "a mean-box or annulus background it removes the pedestal twice.",
        "background")

    # ---- beams and registration ---------------------------------------
    beam_top_row: int = _f(
        None,
        "Beam geometry, if you are overriding it. none measures it from the "
        "data, which is the intended path -- the separation moves between "
        "epochs and a stale value fails silently.",
        "geometry", unit="px")
    beam_x_offset: int = _f(None, "See beam_top_row.", "geometry", unit="px")
    register_method: str = _f(
        "smooth_peak",
        "Star-centering algorithm. Use min for a saturated core: the default "
        "smooth_peak hops between rim maxima of the donut and doubles the PSF.",
        "geometry", choices=REGISTER_METHODS)

    # ---- fast axis -----------------------------------------------------
    fast_axis_method: str = _f(
        "mm_model",
        "How theta_off is obtained. mm_model takes it from the Mueller matrix "
        "model, which settles the IP too (so ip_method must also be "
        "mm_model) and is NOT IMPLEMENTED yet -- it raises. butterfly fits it "
        "from the data. fixed uses theta_off below.",
        "fast_axis", choices=FAST_AXIS_METHODS)
    theta_off: float = _f(
        None,
        "Fast axis offset, used when fast_axis_method = fixed. There is no "
        "trusted automatic source: an HWP ladder on an internal source "
        "returns theta_off + chi/2 with chi unknown.",
        "fast_axis", unit="deg")
    fit_radii: list = _f(
        [25.0, 150.0],
        "[r_inner, r_outer] annulus for the butterfly fit. This works on "
        "U_phi, where a tangentially polarized disk contributes nothing by "
        "definition, so it should SPAN the disk and need only clear the core.",
        "fast_axis", unit="px")

    # ---- instrumental polarization -------------------------------------
    ip_method: str = _f(
        "mm_model",
        "How the I -> Q/U leakage is measured, named <method>_<scope>: the "
        "scope is also the level it is applied at. mm_model is NOT "
        "IMPLEMENTED and raises. Both fit_uphi routes ASSUME AN AZIMUTHALLY "
        "POLARIZED SOURCE and return a confident, meaningless number where "
        "that is the hypothesis under test. none leaves it uncorrected.",
        "instrumental_polarization", choices=IP_METHODS)
    ip_mask_radius: float = _f(
        22.0,
        "Radius excluded from the fit_uphi annulus, covering the saturated or "
        "occulted core.",
        "instrumental_polarization", unit="px")

    def __post_init__(self):
        """Check every enumerated field, and the rules that span fields."""
        for f in fields(self):
            choices = f.metadata.get("choices")
            value = getattr(self, f.name)
            if choices is None or value in choices:
                continue
            if value in RETIRED:
                raise ValueError(
                    f"{f.name} = {value!r} is retired: {RETIRED[value]}.")
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
            raise ValueError(
                f"{path} sets options this version does not know: "
                f"{sorted(unknown)}. Known options: {sorted(known)}.")

        clean = {k: (None if isinstance(v, str) and v.lower() in NONE_STRINGS
                     else v)
                 for k, v in flat.items()}
        log.info("Reduction config: %s (%d options set)", path, len(clean))
        return cls(**clean)

    def configure(self, instrument):
        """Apply the per-dataset settings to an instrument, and return it.

        These five attributes are the only ones a *reduction* chooses;
        everything else on the instrument -- plate scale, detector epochs,
        the rotation model -- is a property of the hardware and comes from
        ``instruments/nirc2.toml``.

        You do not need a subclass to set them. They are read through
        ``self``, so assigning them on the instance works, which is what this
        does. Subclass only when you want to change *behaviour*: override a
        method, add an instrument, model a night the base class cannot
        describe. Setting five values is not a reason.

        The beam geometry is assigned even when None, which is the usual
        case: None means "measure it from the data", and
        :func:`reduction.fit_beam_geometry` fills it in afterwards. Call this
        before measuring, not after, or it will overwrite what was measured.
        """
        instrument.background_method = self.background_method
        instrument.background_box = self.background_box
        instrument.background_annulus = self.background_annulus
        instrument.top_row_start = self.beam_top_row
        instrument.beam_x_offset = self.beam_x_offset
        return instrument

    def describe(self):
        """Every option and its value, for the reduction log."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def template(cls):
        """An annotated TOML file listing every option, default and choice.

        Generated from the dataclass, so it cannot fall behind it.
        """
        import textwrap

        out = [
            "# Reduction config for NIRC2Pol-DPP.",
            "#",
            "# Every option this pipeline takes, with its default. Delete a",
            "# line to accept the default; set it to \"none\" to switch the",
            "# option off, which is not always the same thing.",
            "#",
            "# Instrument constants (plate scale, detector epochs, beam",
            "# geometry seed) are NOT here -- they live in",
            "# instruments/nirc2.toml, because they are properties of the",
            "# instrument rather than choices about this reduction.",
            "",
        ]
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
                out.append(
                    f"{f.name} = {_toml_value(f.metadata['default'])}")
                out.append("")
        return "\n".join(out)


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

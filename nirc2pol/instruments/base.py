"""Abstract instrument interface for polarimetry data.

``PolarimetryData`` is the contract between the generic pipeline and one
instrument: everything the reduction and polarimetry layers need to know
about detector properties, header conventions, beam geometry, and the
polarization modulator lives behind this interface. Supporting a new
instrument means subclassing it (see ``nirc2pol/instruments/nirc2.py``).

The generic layers never read instrument headers directly — they call these
methods.
"""

from __future__ import annotations

import logging

from nirc2pol.reduction.sky import subtract_background
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

def read_config(path):
    """Read an instrument constants file.

    Parameters
    ----------
    path : str
        Path to the ``.toml``.

    Returns
    -------
    dict
        The parsed file: nested dicts of real types. TOML gives floats,
        integers, arrays and dates directly, so nothing needs casting at the
        call site, and keys keep their case -- which matters here, because
        band names are keys and ``Lp`` must not collide with ``L``.

    Raises
    ------
    FileNotFoundError
        If the file is missing. The constants are not optional, and silently
        falling back to hardcoded values would defeat the point of having
        them in one auditable place.
    """
    import tomllib

    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Instrument constants not found at {path}. This file holds the "
            "plate scale, detector epochs and beam geometry, so the module "
            "cannot be used without it.") from None


class PolarimetryData(ABC):
    """Everything instrument-specific the pipeline needs, in one object."""

    name = "abstract"
    plate_scale = None  # arcsec / pixel

    # Which flat type each band requires, e.g. {"Lp": "SKY", "H": "DOME"},
    # and the fallback for bands not listed. Which flat a band needs is a
    # property of the instrument and its bands, so it belongs here rather
    # than in the generic reduction code; pass it to reduce_frame and
    # make_master_flats. Empty means no requirement is enforced.
    required_flat_types = {}
    default_required_flat_type = None

    # header keyword holding the polarization modulator position (HWP angle
    # or, for NIRC2, the image rotator position)
    modulator_keyword = None

    # number of modulator positions in one complete cycle
    modulator_cycle_length = 4

    # header keyword naming the target. Frames of different targets are never
    # one modulator cycle, however well their angles happen to line up across
    # the boundary, so match_modulator_cycles breaks on a change here.
    target_keyword = "TARGNAME"

    # header keyword naming where the modulator is parked, and the values
    # meaning it sits in the beam or out of it. Flats need this: one taken
    # with the modulator removed is an ordinary flat, while one taken with it
    # in the beam carries its transmission and is only usable as part of a
    # complete modulation cycle.
    modulator_name_keyword = None
    modulator_in_names = ()
    modulator_out_names = ()

    # the modulator's critical angles [deg], ordered (Q+, Q-, U+, U-)
    critical_angles = (0.0, 45.0, 22.5, 67.5)

    #
    # background subtraction
    #
    # On-sky data must configure this
    # Recommended by band: L uses dither pairs or a mean box; JHK use an annulus around the source or a mean box.
    background_method = "mean_box"   # "mean_box" | "annulus" | "dither" | None
    background_box = None                 # (ylow, yhigh, xlow, xhigh)
    background_annulus = None             # (r_inner, r_outer) in pixels

    _warned_background_choice = False

    # One-shot warning flags. They are class attributes, so a message is
    # emitted once per class for the life of the *process* -- in a long
    # session (a notebook kernel run twice, a batch over several nights) it
    # will not repeat. Call reset_warnings() between reductions to re-arm
    # them.
    _warned_no_background = False
    _warned_uncalibrated_offset = False

    @classmethod
    def reset_warnings(cls):
        """Re-arm every one-shot warning on this class.

        The warn-once flags persist for the life of the process, which is
        right within one reduction and wrong across several: the second
        night reduced in the same session would stay silent about a missing
        background or an uncalibrated fast axis offset.
        """
        for klass in cls.__mro__:
            for name in list(vars(klass)):
                if name.startswith(("_warned_", "_announced_")):
                    setattr(klass, name, False)

    def subtract_background(self, stack):
        """Remove the sky/thermal background from each beam of a stack.

        Parameters
        ----------
        stack : ndarray
            Beam stack to correct.

        Returns
        -------
        ndarray
            The corrected stack.

        Notes
        -----
        The subtractions, and the dispatch between them, are in
        :func:`nirc2pol.reduction.sky.subtract_background`; this supplies the three
        per-dataset settings the instrument carries and nothing else.

        The one piece of logic kept here is the warning for a ``mean_box``
        with no box, because saying it once per instrument needs the
        ``_warned_*`` flag that :meth:`reset_warnings` clears.
        """
        if self.background_method == "mean_box" and self.background_box is None:
            if not type(self)._warned_no_background:
                log.warning(
                    "%s has background_method='mean_box' but no "
                    "background_box set, so no background is being "
                    "removed. On-sky data should configure one (or set "
                    "background_method=None to say the omission is "
                    "deliberate).", type(self).__name__)
                type(self)._warned_no_background = True
            return stack

        return subtract_background(stack, self.background_method,
                                   box=self.background_box,
                                   annulus=self.background_annulus)

    def check_background_choice(self, header):
        """Warn once if the background method suits the band badly.

        Parameters
        ----------
        header : Header
            A frame from the dataset, for reading its band.

        Notes
        -----
        Does nothing by default. An instrument that knows which methods suit
        which of its bands overrides this

        Warns once per instrument class, via a ``_warned_*`` flag so that
        :meth:`reset_warnings` clears it along with the others.
        """

    def describe_beam_geometry(self):
        """One-line description of how the beams were cut out, for provenance.

        Returns
        -------
        str
            Empty by default. An instrument whose beam extraction is
            configurable should say what it used, since a product cannot be
            reproduced without it.

        Notes
        -----
        A method rather than the caller reading attributes: how the two
        beams are separated is the instrument's business
        """
        return ""

    def describe_background(self):
        """One-line description of the background setting, for provenance."""
        if self.background_method is None:
            return "none"
        if self.background_method == "mean_box":
            return f"mean_box{self.background_box}"
        if self.background_method == "dither":
            return "dither pairs (frame level)"
        return f"annulus{self.background_annulus}"

    #
    # detector properties
    #

    @abstractmethod
    def gain(self, header):
        """Detector gain [e-/ADU] for a frame."""

    @abstractmethod
    def saturation_limit(self, header):
        """Saturation / linearity limit [ADU] for a frame."""

    def readnoise(self, header):
        """Read noise [e-]; default 0 if the instrument doesn't model it."""
        return 0.0

    @abstractmethod
    def bad_pixel_mask(self):
        """Static detector bad pixel mask as a boolean array."""

    #
    # header handling
    #

    @abstractmethod
    def sort_frames(self, filenames, **kwargs):
        """Classify raw files into a dict of filename lists with keys
        "sci", "flats", "flats_sky", "flats_lampon", "darks"."""

    @abstractmethod
    def north_angle(self, header):
        """Mean angle to north [deg] for a frame; derotate by minus this."""

    def modulator_angle(self, header):
        """Modulator (HWP / rotator) position [deg] for a frame."""
        if self.modulator_keyword is None:
            raise NotImplementedError(
                f"{self.name} does not define modulator_keyword")
        return header[self.modulator_keyword]

    #
    # polarimetry geometry
    #

    @abstractmethod
    def split_beams(self, frame):
        """Extract the two orthogonally-polarized beams from a frame.

        Returns an array of shape ``(2, ny, nx)`` with the two beam images
        registered to each other (beam 0 and beam 1 cover the same sky).
        """

    def occulting_radius(self, header):
        """Radius [px] of the focal-plane occulting mask, or None.

        Used by the mask-edge instrumental-polarization method, which needs
        an annulus just outside the mask where the light is the star's own
        (assumed unpolarized) PSF. Returning None means "not coronagraphic",
        and the caller must supply radii itself — for saturated but
        unocculted data the equivalent boundary is the edge of the saturated
        core, which the instrument cannot know.
        """
        return None

    @abstractmethod
    def qu_rotation_angle(self, header, fast_axis_offset=None):
        """Angle [deg] by which the measured polarization frame is rotated
        relative to sky for a frame — the instrument's Mueller-matrix (or
        rotation-approximation) model. Used by
        ``polarimetry.stokes.build_stokes_cube`` to rotate the double
        differences into sky Q/U."""

    #
    # modulator cycle matching (concrete — instruments rarely need to
    # override)
    #

    def modulator_in_beam(self, frame):
        """Was the modulator in the beam when this frame was taken?

        Parameters
        ----------
        frame : Frame
            Frame to test.

        Returns
        -------
        bool or None
            True in the beam, False parked out of it, and None when the header
            does not say -- data predating the keyword, or a position named by
            neither list. None means *unknown*, not "out": callers must not
            treat it as a licence to use the frame as an ordinary flat.
        """
        if not self.modulator_name_keyword:
            return None
        name = frame.get(self.modulator_name_keyword)
        if name is None:
            return None
        name = str(name).strip().lower()
        if name in {str(n).strip().lower() for n in self.modulator_in_names}:
            return True
        if name in {str(n).strip().lower() for n in self.modulator_out_names}:
            return False
        return None

    def match_modulator_cycles(self, frames, atol=0.1):
        """Group a time-ordered list of frames into complete modulator
        cycles of ``modulator_cycle_length`` distinct positions.

        Walks the frame list in order, accumulating frames until
        ``modulator_cycle_length`` distinct modulator angles have been seen;
        that group becomes one cycle. Incomplete trailing groups are dropped
        with a warning. Repeats at the same position within a cycle are kept
        (a cycle is a list of frames, one entry per exposure).

        A change of target also ends the group in progress. Without that, a
        pointing ending mid-cycle at, say, 0 and 45 followed by a slew to
        another object starting at 22.5 and 67.5 presents four distinct angles
        in a row and would be accepted as one cycle -- a double difference
        between two objects. Selecting the frames you mean to reduce
        (:func:`nirc2pol.utils.frame.select_frames`) is the real answer; this is the
        guard for when that has not been done.

        Returns a list of cycles; each cycle is a list of frames ordered as
        observed. The frame -> cycle mapping is recorded in each frame's
        POLCYCLE header keyword.
        """
        from nirc2pol.utils.angles import angles_match

        cycles = []
        current = []
        current_angles = []
        current_target = None
        discarded = 0

        for frame in frames:
            angle = self.modulator_angle(frame)
            seen = any(angles_match(angle, a, atol) for a in current_angles)

            target = frame.get(self.target_keyword) if self.target_keyword \
                else None
            changed_target = bool(current) and target != current_target

            # A repeat before the cycle is complete means the sequence was
            # interrupted (an aborted cycle, or extra frames at the end of a
            # pointing). A change of target means the telescope moved. Either
            # way those frames belong to no complete cycle: drop the partial
            # group rather than letting it absorb frames from the next
            # pointing.
            if changed_target:
                log.warning(
                    "Target changed from %r to %r partway through a modulator "
                    "cycle, so the %d frame(s) already accumulated are "
                    "dropped. They cannot be combined with frames of a "
                    "different object.",
                    current_target, target, len(current))
            if seen or changed_target:
                discarded += len(current)
                current = []
                current_angles = []

            if not current:
                current_target = target

            current.append(frame)
            current_angles.append(angle)

            if len(current_angles) == self.modulator_cycle_length:
                for f in current:
                    f["POLCYCLE"] = len(cycles)
                cycles.append(current)
                current = []
                current_angles = []

        discarded += len(current)
        if discarded:
            log.warning(
                "Dropped %d frames that do not form a complete %d-position "
                "modulator cycle (interrupted or trailing sequences)",
                discarded, self.modulator_cycle_length)

        log.info("Matched %d complete modulator cycles", len(cycles))
        return cycles

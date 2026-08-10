"""Abstract instrument interface for polarimetry data.

``PolarimetryData`` is the contract between the generic pipeline and one
instrument: everything the reduction and polarimetry layers need to know
about detector properties, header conventions, beam geometry, and the
polarization modulator lives behind this interface. Supporting a new
instrument means subclassing it (see ``instruments/nirc2.py``).

The generic layers never read instrument headers directly — they call these
methods.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)


class PolarimetryData(ABC):
    """Everything instrument-specific the pipeline needs, in one object."""

    name = "abstract"
    plate_scale = None  # arcsec / pixel

    # filter substitution rules for flat matching, passed to
    # reduction.calibrate.find_closest_flat (empty by default)
    flat_exceptions = {}

    # header keyword holding the polarization modulator position (HWP angle
    # or, for NIRC2, the image rotator position)
    modulator_keyword = None

    # number of modulator positions in one complete cycle
    modulator_cycle_length = 4

    # the modulator's critical angles [deg], ordered (Q+, Q-, U+, U-)
    critical_angles = (0.0, 45.0, 22.5, 67.5)

    #
    # background subtraction
    #
    # Removed from every extracted beam before differencing. On-sky data
    # must configure this: at L' the thermal pedestal reaches tens of
    # thousands of counts and swamps registration, the I-proportional
    # instrumental-polarization term, and any PI/I ratio. Calibration
    # sequences (dome flats, fast axis ladders) set
    # ``background_method = None``, because there the illumination *is*
    # the signal.
    # Recommended by band: L'/M use dither pairs or a mean box (the thermal
    # pedestal is large and structured); JHK use an annulus around the
    # source or a mean box.
    background_method = "mean_box"   # "mean_box" | "annulus" | "dither" | None
    background_box = None                 # (ylow, yhigh, xlow, xhigh)
    background_annulus = None             # (r_inner, r_outer) in pixels

    _warned_no_background = False
    _warned_uncalibrated_offset = False

    def subtract_background(self, stack):
        """Remove the sky/thermal pedestal from each beam of a stack."""
        from reduction.sky import (subtract_annulus_background,
                                   subtract_mean_background)

        if self.background_method is None:
            return stack

        if self.background_method == "mean_box":
            if self.background_box is None:
                if not type(self)._warned_no_background:
                    log.warning(
                        "%s has background_method='mean_box' but no "
                        "background_box set, so no background is being "
                        "removed. On-sky data should configure one (or set "
                        "background_method=None to say the omission is "
                        "deliberate).", type(self).__name__)
                    type(self)._warned_no_background = True
                return stack
            return subtract_mean_background(stack, box=self.background_box)

        if self.background_method == "annulus":
            if self.background_annulus is None:
                raise ValueError("background_method='annulus' requires "
                                 "background_annulus=(r_inner, r_outer)")
            return subtract_annulus_background(stack, *self.background_annulus)

        if self.background_method == "dither":
            # Handled at frame level, before beam extraction, by
            # reduction.sky.subtract_dither_pairs; nothing to do per beam.
            return stack

        raise ValueError(f"Unknown background_method "
                         f"{self.background_method!r}")

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
        "sci", "flats", "flats_sky", "flats_lampon", "flats_lampoff",
        "darks"."""

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

    @abstractmethod
    def qu_rotation_angle(self, header, fast_axis_offset=0.0):
        """Angle [deg] by which the measured polarization frame is rotated
        relative to sky for a frame — the instrument's Mueller-matrix (or
        rotation-approximation) model. Used by
        ``polarimetry.stokes.build_stokes_cube`` to rotate the double
        differences into sky Q/U."""

    #
    # modulator cycle matching (concrete — instruments rarely need to
    # override)
    #

    def match_modulator_cycles(self, frames, atol=0.1):
        """Group a time-ordered list of frames into complete modulator
        cycles of ``modulator_cycle_length`` distinct positions.

        Walks the frame list in order, accumulating frames until
        ``modulator_cycle_length`` distinct modulator angles have been seen;
        that group becomes one cycle. Incomplete trailing groups are dropped
        with a warning. Repeats at the same position within a cycle are kept
        (a cycle is a list of frames, one entry per exposure).

        Returns a list of cycles; each cycle is a list of frames ordered as
        observed. The frame -> cycle mapping is recorded in each frame's
        POLCYCLE header keyword.
        """
        from utils.angles import angles_match

        cycles = []
        current = []
        current_angles = []
        discarded = 0

        for frame in frames:
            angle = self.modulator_angle(frame)
            seen = any(angles_match(angle, a, atol) for a in current_angles)

            # A repeat before the cycle is complete means the sequence was
            # interrupted (an aborted cycle, or extra frames at the end of a
            # pointing). Those frames belong to no complete cycle: drop the
            # partial group rather than letting it absorb frames from the
            # next pointing.
            if seen:
                discarded += len(current)
                current = []
                current_angles = []

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

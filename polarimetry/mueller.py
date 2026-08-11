"""Mueller matrix model interface.

The full NIRC2-Pol Mueller matrix model is still under development; until
it is available, the pipeline uses the idealized rotation approximation
(SPIE Eq. 3), which lives in each instrument's ``qu_rotation_angle``.

This module defines the interface that the full model will implement, so
Stokes cube generation can swap it in without restructuring. Packages like
``pyMuellerMat`` / ``pyPolCal`` are candidates for the implementation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from utils.angles import mean_angle

log = logging.getLogger(__name__)


class MuellerMatrixModel(ABC):
    """Instrumental polarization model: converts measured Stokes vectors to
    sky Stokes vectors for a given instrument state (header)."""

    @abstractmethod
    def correct(self, stokes_cube, header):
        """Apply the instrumental polarization correction to a
                ``(3, ny, nx)`` measured Stokes cube [I, Q, U], returning the
                corrected cube in the sky frame.

        Rotate instrument-frame Q/U into the sky frame.

        Parameters
        ----------
        Q, U : ndarray
            Instrument-frame Stokes planes.
        header : Frame or Header
            Frame supplying the rotation angle.

        Returns
        -------
        tuple of ndarray
            ``(Q_sky, U_sky)``.
        """


class RotationApproximationModel(MuellerMatrixModel):
    """Idealized model: pure rotation of Q/U by the instrument's
    polarimetric rotation angle, with no cross-talk or instrumental
    polarization terms. Wraps ``instrument.qu_rotation_angle``."""

    def __init__(self, instrument, fast_axis_offset=0.0):
        """Bind the model to an instrument and a fast axis offset.

        Parameters
        ----------
        instrument : PolarimetryData
            Instrument whose rotation model is used.
        fast_axis_offset : float, optional
            theta_off in degrees.
        """
        self.instrument = instrument
        self.fast_axis_offset = fast_axis_offset

    def correct(self, stokes_cube, header):
        """Rotate instrument-frame Q/U into the sky frame.

        Parameters
        ----------
        Q, U : ndarray
            Instrument-frame Stokes planes.
        header : Frame or Header
            Frame supplying the rotation angle.

        Returns
        -------
        tuple of ndarray
            ``(Q_sky, U_sky)``.
        """
        from .stokes import rotate_qu

        I, Q, U = stokes_cube
        theta = self.instrument.qu_rotation_angle(header,
                                                  self.fast_axis_offset)
        q_sky, u_sky = rotate_qu(Q, U, theta)
        return np.stack([I, q_sky, u_sky], axis=0)


#
# TEMPORARY empirical correction — remove when the real Mueller matrix
# model (ref. 31 of the SPIE paper) is available.
#
# Ports the commissioning notebook's calc_quphi minimization: for each HWP
# cycle, fit small residual beam shifts plus instrumental-polarization
# leakage terms (ipq, ipu) by minimizing the U_phi signal. The IP terms
# absorb I -> Q/U crosstalk that the ideal rotation model cannot represent;
# the full Mueller model will supersede all of this.
#


def _registered_stacks(instrument, cycle, critical_angles, atol,
                       register_method, crop_size):
    """One mean registered beam stack per critical angle, optionally cropped
    around the star (which registration puts at the stack center)."""
    from reduction.registration import register_beam_stack
    from utils.imutils import crop

    from .stokes import _angles_match

    stacks = []
    for angle in critical_angles:
        matches = []
        for frame in cycle:
            if not _angles_match(instrument.modulator_angle(frame), angle,
                                 atol):
                continue
            stack = instrument.subtract_background(instrument.split_beams(frame))
            if register_method is not None:
                stack, _ = register_beam_stack(stack, method=register_method)
            if crop_size is not None:
                stack = np.stack([crop(b, (crop_size, crop_size))[0]
                                  for b in stack])
            matches.append(stack)
        if not matches:
            raise ValueError(f"No frames at modulator angle {angle} in cycle")
        stacks.append(np.nanmean(matches, axis=0))
    return stacks


def _corrected_qu(stacks, oe_shifts, frame_shifts, ipq, ipu):
    """Assemble Q, U, I from the four per-angle beam stacks with residual
    beam shifts, frame-to-frame shifts, and IP leakage subtraction."""
    from utils.imutils import translate

    diffs, sums = [], []
    for i, stack in enumerate(stacks):
        bot = translate(stack[0], *oe_shifts[i])
        d, s = stack[1] - bot, stack[1] + bot
        if i > 0:
            d = translate(d, *frame_shifts[i - 1])
            s = translate(s, *frame_shifts[i - 1])
        diffs.append(d)
        sums.append(s)

    from .instpol import InstrumentalPolarization, subtract_ip

    Q = 0.5 * (diffs[0] - diffs[1])
    U = 0.5 * (diffs[2] - diffs[3])
    I = 0.25 * (sums[0] + sums[1] + sums[2] + sums[3])
    Q, U = subtract_ip(Q, U, 0.5 * (sums[0] + sums[1]),
                       InstrumentalPolarization(ipq, ipu, method="uphi_min"),
                       I_u=0.5 * (sums[2] + sums[3]))
    return Q, U, I


def _unpack(x):
    """Split the flat parameter vector into its parts.

    Parameters
    ----------
    x : ndarray
        The 16 fitted parameters.

    Returns
    -------
    oe_shifts : ndarray
        ``(4, 2)`` bottom-beam shifts, one per critical angle.
    frame_shifts : ndarray
        ``(3, 2)`` shifts of frames 2-4 relative to frame 1.
    ipq, ipu : float
        Instrumental polarization terms.
    """
    oe_shifts = x[0:8].reshape(4, 2)
    frame_shifts = x[8:14].reshape(3, 2)
    return oe_shifts, frame_shifts, x[14], x[15]


def fit_empirical_cycle_correction(instrument, cycle, fast_axis_offset,
                                   mask_radius=20, crop_size=400,
                                   critical_angles=None, atol=1.0,
                                   register_method="smooth_peak",
                                   derotate=True, maxiter=2000,
                                   maxfev=None):
    """TEMPORARY: fit per-cycle beam shifts and IP leakage by minimizing
    U_phi (the commissioning notebook's calc_quphi minimization).

    Fits 16 parameters per HWP cycle: a residual (dy, dx) shift of the
    bottom beam at each of the 4 critical angles, a (dy, dx) shift of
    frames 2-4 relative to frame 1, and the instrumental-polarization
    leakage fractions ipq / ipu (I -> Q/U crosstalk, subtracted as
    ``ip * single_sum``). The fast axis offset is held fixed (it comes from
    the calibration log, not this fit).

    Assumes an azimuthally-polarized source so U_phi is noise-only; pixels
    within ``mask_radius`` of the star are excluded. Slow (a Powell search
    over 16 parameters with image shifts per step) — crop tightly.

    Returns a dict with ``oe_shifts``, ``frame_shifts``, ``ipq``, ``ipu``,
    and the initial/final U_phi std. Pass it to
    :func:`build_corrected_stokes_cube`. Will be superseded by the full
    Mueller matrix model.
    """
    from scipy.optimize import minimize

    from utils.imutils import make_circle_mask

    from .stokes import CRITICAL_ANGLES, radial_stokes, rotate_qu

    log.warning("Using the TEMPORARY empirical U_phi-minimization "
                "correction; to be replaced by the Mueller matrix model")

    critical_angles = critical_angles or CRITICAL_ANGLES
    stacks = _registered_stacks(instrument, cycle, critical_angles, atol,
                                register_method, crop_size)
    base_rot = float(mean_angle(
        [instrument.qu_rotation_angle(f, fast_axis_offset) for f in cycle]))
    north = (float(mean_angle([instrument.north_angle(f) for f in cycle]))
             if derotate else 0.0)

    # Spatially derotating the Q/U maps and measuring the U_phi std over a
    # circular annulus is exactly equivalent to adding 2*north to the Q/U
    # rotation angle (rotating pixels by alpha shifts every azimuthal angle
    # phi by alpha, and only 2*phi + theta enters U_phi) — so the objective
    # needs no image rotation per optimizer step.
    eff_rot = base_rot + 2.0 * north
    shape = stacks[0][0].shape
    annulus = (make_circle_mask(shape, min(shape) // 2 - 1)
               & ~make_circle_mask(shape, mask_radius))

    def objective(x):
        """U_phi scatter for one trial parameter vector; lower is better."""
        Q, U, _ = _corrected_qu(stacks, *_unpack(x))
        _, u_phi = radial_stokes(*rotate_qu(Q, U, eff_rot))
        return float(np.nanstd(u_phi[annulus]))

    x0 = np.zeros(16)
    initial_std = objective(x0)
    options = {"maxiter": maxiter}
    if maxfev is not None:
        options["maxfev"] = maxfev
    result = minimize(objective, x0, method="Powell", options=options)
    oe_shifts, frame_shifts, ipq, ipu = _unpack(result.x)

    log.info("Empirical fit: U_phi std %.4g -> %.4g | ipq = %.4f, "
             "ipu = %.4f", initial_std, result.fun, ipq, ipu)
    return {"oe_shifts": oe_shifts, "frame_shifts": frame_shifts,
            "ipq": float(ipq), "ipu": float(ipu),
            "uphi_std_initial": initial_std,
            "uphi_std_final": float(result.fun)}


def build_corrected_stokes_cube(instrument, cycle, correction,
                                fast_axis_offset, crop_size=400,
                                critical_angles=None, atol=1.0,
                                register_method="smooth_peak",
                                derotate=True):
    """TEMPORARY: build a ``(3, ny, nx)`` Stokes cube [I, Q', U'] applying
    a fitted empirical correction (see
    :func:`fit_empirical_cycle_correction`). Use the same ``crop_size`` /
    ``register_method`` as the fit."""
    from utils.imutils import rotate_image_center

    from .stokes import CRITICAL_ANGLES, rotate_qu

    critical_angles = critical_angles or CRITICAL_ANGLES
    stacks = _registered_stacks(instrument, cycle, critical_angles, atol,
                                register_method, crop_size)
    Q, U, I = _corrected_qu(stacks, correction["oe_shifts"],
                            correction["frame_shifts"], correction["ipq"],
                            correction["ipu"])

    base_rot = float(mean_angle(
        [instrument.qu_rotation_angle(f, fast_axis_offset) for f in cycle]))
    q_sky, u_sky = rotate_qu(Q, U, base_rot)

    if derotate:
        north = float(mean_angle([instrument.north_angle(f) for f in cycle]))
        I = rotate_image_center(I, -north)
        q_sky = rotate_image_center(q_sky, -north)
        u_sky = rotate_image_center(u_sky, -north)

    return np.stack([I, q_sky, u_sky], axis=0)

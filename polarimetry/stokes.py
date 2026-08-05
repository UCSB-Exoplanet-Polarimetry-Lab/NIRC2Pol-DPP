"""Double differencing and Stokes cube production.

Implements Section 3.5 of Lewis et al. (SPIE), assuming an idealized system
(the rotation-approximation instrument model); the full Mueller matrix model
will slot in via ``polarimetry/mueller.py`` once available.

For each HWP cycle of critical angles (0, 45, 22.5, 67.5 deg), with
``I_top(theta)`` and ``I_bottom(theta)`` the two orthogonal polarization
states (ordinary/extraordinary beams)::

    Q = 1/2 [ (I_top(0)    - I_bottom(0))    - (I_top(45)   - I_bottom(45))   ]
    U = 1/2 [ (I_top(22.5) - I_bottom(22.5)) - (I_top(67.5) - I_bottom(67.5)) ]

The measured Q/U are rotated into the sky frame by the instrument's
polarimetric rotation angle theta_rot (for NIRC2:
``-2*PARANG + 2*EL + 2*ROTPDEST + 4*theta_off``)::

    Q' =  Q cos(theta_rot) + U sin(theta_rot)
    U' = -Q sin(theta_rot) + U cos(theta_rot)

Outputs: Stokes cubes per HWP cycle, a median Stokes cube, derived products
(PI / AoLP / DoLP), and optionally radial Stokes (Q_phi / U_phi).
"""

from __future__ import annotations

import logging

import numpy as np

from utils.angles import mean_angle

log = logging.getLogger(__name__)

# canonical HWP critical angles [deg]: (Q+, Q-, U+, U-)
CRITICAL_ANGLES = (0.0, 45.0, 22.5, 67.5)


def single_difference(beam_stack):
    """Single difference and sum from a registered ``(2, ny, nx)`` beam
    stack (beam 0 = bottom, beam 1 = top): returns
    ``(I_top - I_bottom, I_top + I_bottom)``."""
    beam_stack = np.asarray(beam_stack)
    return beam_stack[1] - beam_stack[0], beam_stack[1] + beam_stack[0]


def _angles_match(a, b, atol):
    """Circular comparison of modulator angles modulo 180 deg (so -0.002
    matches 0, and 179.9 matches 0)."""
    from utils.angles import angles_match

    return angles_match(a, b, atol)


def _mean_frame_at_angle(instrument, cycle, angle, atol, register_method,
                         register_kwargs=None):
    """Mean single difference and sum of all frames in the cycle whose
    modulator angle matches ``angle``. Each frame's beam stack is centered
    on the star first (unless ``register_method`` is None), so frames can
    be combined and differenced across the cycle."""
    from reduction.registration import register_beam_stack

    diffs, sums = [], []
    for frame in cycle:
        if _angles_match(instrument.modulator_angle(frame), angle, atol):
            stack = instrument.subtract_background(instrument.split_beams(frame))
            if register_method is not None:
                stack, _ = register_beam_stack(stack, method=register_method,
                                               **(register_kwargs or {}))
            d, s = single_difference(stack)
            diffs.append(d)
            sums.append(s)
    if not diffs:
        raise ValueError(f"No frames at modulator angle {angle} in cycle")
    return np.nanmean(diffs, axis=0), np.nanmean(sums, axis=0)


def _check_background(instrument, cycle):
    """Warn once if the background setting looks wrong for the band."""
    if getattr(instrument, "_bkg_checked", False):
        return
    instrument._bkg_checked = True
    try:
        from instruments.nirc2 import band_of, check_background_choice
        check_background_choice(band_of(cycle[0].header),
                                instrument.background_method)
    except Exception:
        pass


def double_difference(instrument, cycle, critical_angles=CRITICAL_ANGLES,
                      atol=1.0, register_method="smooth_peak",
                      register_kwargs=None):
    """Double differences for one HWP cycle.

    ``cycle`` is a list of frames covering all four critical angles (from
    ``instrument.match_modulator_cycles``). Returns ``(Q, U, I)`` in the
    instrument frame, where I averages the single sums over all four angles.

    ``register_method`` selects the star-centering algorithm applied to each
    beam stack (see ``reduction.registration.find_center``); None to skip.
    ``register_kwargs`` are passed through to that algorithm - needed by
    ``crosscorr``, which requires a ``template=`` reference image.
    """
    _check_background(instrument, cycle)

    a_qp, a_qm, a_up, a_um = critical_angles

    diff_qp, sum_qp = _mean_frame_at_angle(instrument, cycle, a_qp, atol,
                                           register_method,
                                           register_kwargs)
    diff_qm, sum_qm = _mean_frame_at_angle(instrument, cycle, a_qm, atol,
                                           register_method,
                                           register_kwargs)
    diff_up, sum_up = _mean_frame_at_angle(instrument, cycle, a_up, atol,
                                           register_method,
                                           register_kwargs)
    diff_um, sum_um = _mean_frame_at_angle(instrument, cycle, a_um, atol,
                                           register_method,
                                           register_kwargs)

    Q = 0.5 * (diff_qp - diff_qm)
    U = 0.5 * (diff_up - diff_um)
    I = 0.25 * (sum_qp + sum_qm + sum_up + sum_um)

    return Q, U, I


def rotate_qu(Q, U, theta_rot_deg):
    """Rotate measured Q/U into the sky frame by ``theta_rot`` [deg]::

        Q' =  Q cos(theta_rot) + U sin(theta_rot)
        U' = -Q sin(theta_rot) + U cos(theta_rot)
    """
    theta = np.radians(theta_rot_deg)
    q_sky = Q * np.cos(theta) + U * np.sin(theta)
    u_sky = -Q * np.sin(theta) + U * np.cos(theta)
    return q_sky, u_sky


def build_stokes_cube(instrument, cycle, fast_axis_offset=0.0,
                      critical_angles=CRITICAL_ANGLES, atol=1.0,
                      register_method="smooth_peak", derotate=True,
                      register_kwargs=None):
    """Build one ``(3, ny, nx)`` Stokes cube [I, Q', U'] from one HWP cycle.

    Splits and registers the beams, double-differences the cycle, rotates
    Q/U to the sky frame using the cycle-averaged instrument rotation angle,
    and (optionally) spatially derotates all three planes to north-up
    east-left using the cycle-averaged north angle.
    """
    Q, U, I = double_difference(instrument, cycle,
                                critical_angles=critical_angles, atol=atol,
                                register_method=register_method,
                                register_kwargs=register_kwargs)

    theta_rot = float(mean_angle(
        [instrument.qu_rotation_angle(f, fast_axis_offset) for f in cycle]))
    q_sky, u_sky = rotate_qu(Q, U, theta_rot)

    north = None
    if derotate:
        from utils.imutils import rotate_image_center

        north = float(mean_angle([instrument.north_angle(f) for f in cycle]))
        I = rotate_image_center(I, -north)
        q_sky = rotate_image_center(q_sky, -north)
        u_sky = rotate_image_center(u_sky, -north)

    from utils.provenance import record_step

    record_step(cycle[0], "stokes cube",
                instrument=instrument.name, nframes=len(cycle),
                background=instrument.describe_background(),
                critical_angles=list(critical_angles),
                registration=register_method,
                fast_axis_offset=fast_axis_offset,
                qu_rotation=theta_rot,
                north_angle=(north if north is not None else "not derotated"))

    return np.stack([I, q_sky, u_sky], axis=0)


def build_stokes_cubes(instrument, cycles, fast_axis_offset=0.0, **kwargs):
    """Stokes cubes for every HWP cycle: returns a ``(ncycles, 3, ny, nx)``
    array."""
    cubes = [build_stokes_cube(instrument, cycle,
                               fast_axis_offset=fast_axis_offset, **kwargs)
             for cycle in cycles]
    return np.stack(cubes, axis=0)


def median_stokes_cube(stokes_cubes):
    """Median-combine per-cycle Stokes cubes into one ``(3, ny, nx)`` cube."""
    return np.nanmedian(np.asarray(stokes_cubes), axis=0)


def polarization_products(stokes_cube):
    """Derived quantities from a ``(3, ny, nx)`` Stokes cube [I, Q, U]:

    Returns ``(PI, AoLP, DoLP)`` — polarized intensity ``sqrt(Q^2 + U^2)``,
    angle of linear polarization ``0.5 * arctan2(U, Q)`` [deg], and degree
    of linear polarization ``PI / I``.
    """
    I, Q, U = stokes_cube
    pi = np.sqrt(Q**2 + U**2)
    aolp = 0.5 * np.degrees(np.arctan2(U, Q))
    with np.errstate(divide="ignore", invalid="ignore"):
        dolp = pi / I
    return pi, aolp, dolp


def azimuthal_angle(shape, center=None):
    """Azimuthal angle phi [rad] around ``center = (cy, cx)`` for each
    pixel. Since only 2*phi enters the radial Stokes formulas, the choice of
    reference axis (+x vs -x) does not matter."""
    ny, nx = shape
    if center is None:
        center = ((ny - 1) / 2, (nx - 1) / 2)
    cy, cx = center
    yy, xx = np.mgrid[:ny, :nx]
    return np.arctan2(yy - cy, xx - cx)


def radial_stokes(Q, U, center=None):
    """Radial Stokes parameters (SPIE Eqs. 6-7)::

        Q_phi =  Q cos(2 phi) + U sin(2 phi)
        U_phi = -Q sin(2 phi) + U cos(2 phi)

    Disk signal is positive in Q_phi while U_phi contains noise. This sign
    convention was verified empirically on the AB Aur commissioning data
    (2025-12-07 L'): with these signs the tangentially-polarized disk comes
    out positive, matching the notebook cell that produced the reference
    qphi_median. (The IRDAP-style ``-Q cos - U sin`` form gives *negative*
    disk signal for this instrument's image parity — don't "fix" the sign
    without rechecking on sky.) ``center = (cy, cx)`` is the star position
    (default: image center).
    """
    phi = azimuthal_angle(np.shape(Q), center=center)
    q_phi = Q * np.cos(2 * phi) + U * np.sin(2 * phi)
    u_phi = -Q * np.sin(2 * phi) + U * np.cos(2 * phi)
    return q_phi, u_phi

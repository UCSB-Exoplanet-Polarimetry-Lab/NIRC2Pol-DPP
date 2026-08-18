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


def normalized_single_difference(beam_stack, mask=None):
    """Flux-weighted normalized single difference, ``sum(d) / sum(s)``.

    For a registered ``(2, ny, nx)`` beam stack this is the fractional
    imbalance between the two Wollaston beams over ``mask`` (default: every
    finite pixel) — the single-exposure analogue of ``q = Q/I``.

    What it *means* depends on where you measure it and is the caller's
    business. Over a region of intrinsically unpolarized starlight, such as
    just outside an occulting mask or a saturated core, it is an estimate of
    the instrumental I -> Q/U leakage for that exposure, which is how
    ``double_difference(ip_frame_annulus=...)`` uses it. Over a polarized
    source it is simply the modulated signal.

    Note this is *not* an ipq/ipu pair: one exposure sits at a single HWP
    angle, so it carries one modulated combination of Q and U (at HWP 0 it
    is +Q, at 45 deg -Q, at 22.5 deg +U), not both.

    Returns a plain float. Raises ``ValueError`` if the mask selects nothing
    or the summed intensity is zero, since the ratio is then meaningless
    rather than merely noisy.
    """
    d, s = single_difference(beam_stack)
    m = np.isfinite(s) if mask is None else (mask & np.isfinite(s))
    if not m.any():
        raise ValueError("normalized_single_difference: mask selects no "
                         "finite pixels")
    total = float(np.nansum(s[m]))
    if not np.isfinite(total) or total == 0.0:
        raise ValueError("normalized_single_difference: summed intensity is "
                         "zero over the mask")
    return float(np.nansum(d[m])) / total


def _angles_match(a, b, atol):
    """Circular comparison of modulator angles modulo 180 deg (so -0.002
        matches 0, and 179.9 matches 0).

    Parameters
    ----------
    a, b : float
        Modulator angles in degrees.
    atol : float
        Tolerance in degrees.

    Returns
    -------
    bool
        True if they match modulo 180 deg.
    """
    from utils.angles import angles_match

    return angles_match(a, b, atol)


def _mean_frame_at_angle(instrument, cycle, angle, atol, register_method,
                         register_kwargs=None, ip_frame_annulus=None):
    """Mean single difference and sum of all frames in the cycle whose
        modulator angle matches ``angle``. Each frame's beam stack is centered
        on the star first (unless ``register_method`` is None), so frames can
        be combined and differenced across the cycle.

        With ``ip_frame_annulus = (r_inner, r_outer)`` each frame's own
        instrumental leakage is measured in that annulus and removed from its
        single difference before averaging, catching leakage that varies within
        a cycle. See :func:`normalized_single_difference`.

    Parameters
    ----------
    instrument : PolarimetryData
        Instrument supplying beam splitting and the modulator angle.
    cycle : list of Frame
        Frames of one HWP cycle.
    angle : float
        Critical angle to select.
    atol : float
        Tolerance when matching that angle.
    register_method : str or None
        Centering algorithm; None to skip registration.
    register_kwargs : dict, optional
        Extra arguments for the centering algorithm.
    ip_frame_annulus : tuple of float, optional
        ``(r_inner, r_outer)`` for the per-exposure leakage removal.

    Returns
    -------
    diff, sums : ndarray
        Mean single difference and mean single sum over the matching frames.

    Raises
    ------
    ValueError
        If no frame in the cycle sits at this angle.
    """
    from reduction.registration import register_beam_stack

    from .instpol import _annulus

    diffs, sums = [], []
    for frame in cycle:
        if _angles_match(instrument.modulator_angle(frame), angle, atol):
            stack = instrument.subtract_background(instrument.split_beams(frame))
            if register_method is not None:
                stack, _ = register_beam_stack(stack, method=register_method,
                                               **(register_kwargs or {}))
            d, s = single_difference(stack)
            if ip_frame_annulus is not None:
                ratio = normalized_single_difference(
                    stack, _annulus(s.shape, *ip_frame_annulus))
                d = d - ratio * s
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
                      register_kwargs=None, ip=None,
                      ip_frame_annulus=None):
    """Double differences for one HWP cycle.

    ``cycle`` is a list of frames covering all four critical angles (from
    ``instrument.match_modulator_cycles``). Returns ``(Q, U, I)`` in the
    instrument frame, where I averages the single sums over all four angles.

    ``register_method`` selects the star-centering algorithm applied to each
    beam stack (see ``reduction.registration.find_center``); None to skip.
    ``register_kwargs`` are passed through to that algorithm - needed by
    ``crosscorr``, which requires a ``template=`` reference image.

    ``ip`` is an optional ``instpol.InstrumentalPolarization`` removed from
    Q/U here, in the instrument frame, which is the only place it is correct
    to do so: once Q/U have been rotated by ``theta_rot`` the leakage vector
    would have to be rotated with them. ``ip_frame_annulus`` instead removes
    a leakage measured per exposure; the two are independent and can be
    combined.
    """
    _check_background(instrument, cycle)

    a_qp, a_qm, a_up, a_um = critical_angles

    diff_qp, sum_qp = _mean_frame_at_angle(instrument, cycle, a_qp, atol,
                                           register_method,
                                           register_kwargs, ip_frame_annulus)
    diff_qm, sum_qm = _mean_frame_at_angle(instrument, cycle, a_qm, atol,
                                           register_method,
                                           register_kwargs, ip_frame_annulus)
    diff_up, sum_up = _mean_frame_at_angle(instrument, cycle, a_up, atol,
                                           register_method,
                                           register_kwargs, ip_frame_annulus)
    diff_um, sum_um = _mean_frame_at_angle(instrument, cycle, a_um, atol,
                                           register_method,
                                           register_kwargs, ip_frame_annulus)

    Q = 0.5 * (diff_qp - diff_qm)
    U = 0.5 * (diff_up - diff_um)
    I = 0.25 * (sum_qp + sum_qm + sum_up + sum_um)

    if ip is not None:
        from .instpol import subtract_ip

        # the matched intensity of each HWP pair, not the four-angle mean:
        # Q and U are formed from different exposures, so each is scaled by
        # the intensity its own pair actually carried
        Q, U = subtract_ip(Q, U, 0.5 * (sum_qp + sum_qm), ip,
                           I_u=0.5 * (sum_up + sum_um))

    return Q, U, I


def rotate_qu(Q, U, theta_rot_deg):
    """Rotate measured Q/U into the sky frame by ``theta_rot`` [deg]::

            Q' =  Q cos(theta_rot) + U sin(theta_rot)
            U' = -Q sin(theta_rot) + U cos(theta_rot)

    Parameters
    ----------
    Q, U : ndarray
        Instrument-frame Stokes planes.
    theta_rot_deg : float
        Rotation angle in degrees, from the instrument's rotation model.

    Returns
    -------
    tuple of ndarray
        ``(Q_sky, U_sky)``.
    """
    theta = np.radians(theta_rot_deg)
    q_sky = Q * np.cos(theta) + U * np.sin(theta)
    u_sky = -Q * np.sin(theta) + U * np.cos(theta)
    return q_sky, u_sky


def build_stokes_cube(instrument, cycle, fast_axis_offset=0.0,
                      critical_angles=CRITICAL_ANGLES, atol=1.0,
                      register_method="smooth_peak", derotate=True,
                      register_kwargs=None, ip=None,
                      ip_frame_annulus=None):
    """Build one ``(3, ny, nx)`` Stokes cube [I, Q', U'] from one HWP cycle.

        Splits and registers the beams, double-differences the cycle, rotates
        Q/U to the sky frame using the cycle-averaged instrument rotation angle,
        and (optionally) spatially derotates all three planes to north-up
        east-left using the cycle-averaged north angle.

    Parameters
    ----------
    instrument : PolarimetryData
        Instrument supplying beam geometry and the rotation model.
    cycle : list of Frame
        One complete HWP cycle.
    fast_axis_offset : float, optional
        theta_off in degrees. Measure it on sky; the 0 deg default is not a
        calibration and warns once.
    critical_angles : tuple of float, optional
        The four modulation angles.
    atol : float, optional
        Tolerance when matching them.
    register_method : str, optional
        Centering algorithm; None to skip.
    derotate : bool, optional
        Rotate the planes to north-up east-left.
    register_kwargs : dict, optional
        Extra arguments for the centering algorithm.
    ip : InstrumentalPolarization, optional
        Leakage removed in the instrument frame.
    ip_frame_annulus : tuple of float, optional
        ``(r_inner, r_outer)`` for per-exposure leakage removal.

    Returns
    -------
    ndarray
        ``(3, ny, nx)`` cube of ``[I, Q, U]``.
    """
    Q, U, I = double_difference(instrument, cycle,
                                critical_angles=critical_angles, atol=atol,
                                register_method=register_method,
                                register_kwargs=register_kwargs, ip=ip,
                                ip_frame_annulus=ip_frame_annulus)

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

    # replace=True because this frame is an *input*: the record is written
    # here so the writers, which build product headers from cycle[0], can
    # find it. Rebuilding the same cycle -- scanning fast axis offsets, say
    # -- would otherwise leave a stack of records disagreeing about how the
    # cube in hand was made.
    record_step(cycle[0], "stokes cube", replace=True,
                instrument=instrument.name, nframes=len(cycle),
                background=instrument.describe_background(),
                critical_angles=list(critical_angles),
                registration=register_method,
                fast_axis_offset=fast_axis_offset,
                instrumental_polarization=(ip.describe() if ip is not None
                                           else "none"),
                ip_frame_annulus=(str(ip_frame_annulus)
                                  if ip_frame_annulus else "none"),
                qu_rotation=theta_rot,
                north_angle=(north if north is not None else "not derotated"))

    return np.stack([I, q_sky, u_sky], axis=0)


def build_stokes_cubes(instrument, cycles, fast_axis_offset=0.0, **kwargs):
    """Stokes cubes for every HWP cycle: returns a ``(ncycles, 3, ny, nx)``
        array.

    Parameters
    ----------
    instrument : PolarimetryData
        Instrument to reduce with.
    cycles : list of list of Frame
        The cycles to reduce.
    fast_axis_offset : float, optional
        theta_off in degrees.
    **kwargs
        Passed to :func:`build_stokes_cube`.

    Returns
    -------
    ndarray
        ``(ncycles, 3, ny, nx)``.
    """
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
        reference axis (+x vs -x) does not matter.

    Parameters
    ----------
    shape : tuple of int
        ``(ny, nx)`` of the grid.
    center : tuple of float, optional
        ``(cy, cx)``; defaults to the image centre.

    Returns
    -------
    ndarray
        Azimuth in radians at each pixel. Only ``2*phi`` enters the radial
        Stokes definitions, so the choice of reference axis does not matter.
    """
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
        (2025-12-08 UT L'): with these signs the tangentially-polarized disk comes
        out positive, matching the notebook cell that produced the reference
        qphi_median. (The IRDAP-style ``-Q cos - U sin`` form gives *negative*
        disk signal for this instrument's image parity — don't "fix" the sign
        without rechecking on sky.) ``center = (cy, cx)`` is the star position
        (default: image center).

    Parameters
    ----------
    Q, U : ndarray
        Sky-frame Stokes planes.
    center : tuple of float, optional
        Centre of the azimuthal pattern -- the star. Defaults to the image
        centre, which is right only if registration put the star there.

    Returns
    -------
    tuple of ndarray
        ``(Q_phi, U_phi)``. Tangential polarization gives positive Q_phi, and
        U_phi is the null channel.
    """
    phi = azimuthal_angle(np.shape(Q), center=center)
    q_phi = Q * np.cos(2 * phi) + U * np.sin(2 * phi)
    u_phi = -Q * np.sin(2 * phi) + U * np.cos(2 * phi)
    return q_phi, u_phi

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
from reduction.registration import register_beam_stack

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
                         register_kwargs=None):
    """Mean single difference and sum of all frames in the cycle whose
        modulator angle matches ``angle``. Each frame's beam stack is centered
        on the star first (unless ``register_method`` is None), so frames can
        be combined and differenced across the cycle.

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

    Returns
    -------
    diff, sums : ndarray
        Mean single difference and mean single sum over the matching frames.

    Raises
    ------
    ValueError
        If no frame in the cycle sits at this angle.
    """

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


def _check_cycle_exposure(cycle):
    """Warn when one cycle's frames were not all taken the same way.

    Parameters
    ----------
    cycle : list of Frame
        The frames of one modulator cycle.

    Notes
    -----
    The double difference subtracts frames from each other directly, so
    frames of unequal depth would contribute unequally. Dividing by ITIME
    during reduction normalises that away, which is why this warns rather
    than refuses -- the arithmetic is sound either way. It is still worth
    saying, because an exposure change inside a single HWP cycle usually
    means something happened during the observation rather than something
    intended.

    Warns per cycle, not once per run: each mixed cycle is a separate fact
    about the data, and reporting only the first would hide the rest.
    """
    for keyword in ("ITIME", "COADDS"):
        values = {f.get(keyword) for f in cycle if f.get(keyword) is not None}
        if len(values) > 1:
            log.warning(
                "Frames in this cycle differ in %s: %s. They are normalised "
                "per frame during reduction so the double difference is "
                "still valid, but an exposure change within one HWP cycle "
                "is worth checking.", keyword,
                ", ".join(str(v) for v in sorted(values, key=str)))


def double_difference(instrument, cycle, critical_angles=CRITICAL_ANGLES,
                      atol=1.0, register_method="smooth_peak",
                      register_kwargs=None, ip=None):
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
    would have to be rotated with them.
    """
    instrument.check_background_choice(cycle[0].header)
    _check_cycle_exposure(cycle)

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


def build_stokes_cube(instrument, cycle, fast_axis_offset=None,
                      critical_angles=CRITICAL_ANGLES, atol=1.0,
                      register_method="smooth_peak", derotate=True,
                      register_kwargs=None, ip=None):
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

    Returns
    -------
    ndarray
        ``(3, ny, nx)`` cube of ``[I, Q, U]``.
    """
    Q, U, I = double_difference(instrument, cycle,
                                critical_angles=critical_angles, atol=atol,
                                register_method=register_method,
                                register_kwargs=register_kwargs, ip=ip)

    # None rather than 0.0 as the default, so that "not specified" stays
    # distinguishable all the way down to qu_rotation_angle. A literal 0.0
    # is a deliberate request -- the fast axis solver evaluates the rotation
    # at zero offset before scanning -- whereas None means nobody chose, and
    # only the second case should warn. With 0.0 as the default here the
    # warning could never fire on the ordinary path.
    theta_rot = float(mean_angle(
        [instrument.qu_rotation_angle(f, fast_axis_offset) for f in cycle]))
    effective_offset = (instrument.fast_axis_offset if fast_axis_offset is None
                        else fast_axis_offset)
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
                beam_geometry=instrument.describe_beam_geometry(),
                background=instrument.describe_background(),
                critical_angles=list(critical_angles),
                registration=register_method,
                fast_axis_offset=effective_offset,
                instrumental_polarization=(ip.describe() if ip is not None
                                           else "none"),
                qu_rotation=theta_rot,
                north_angle=(north if north is not None else "not derotated"))

    return np.stack([I, q_sky, u_sky], axis=0)


def _ip_per_cycle(ip, ncycles):
    """One leakage per cycle, from either a single value or a sequence.

    ``ip`` may be a single :class:`InstrumentalPolarization` applied to every
    cycle, a sequence of one per cycle, or None. The per-cycle form is what
    lets a leakage measured on a cycle be removed from *that* cycle, which is
    the point of the ``per_cycle`` scopes -- otherwise a per-cycle
    measurement can only ever be averaged and applied uniformly.
    """
    if ip is None or hasattr(ip, "ipq"):
        return [ip] * ncycles

    ips = list(ip)
    if len(ips) != ncycles:
        raise ValueError(
            f"ip has {len(ips)} entries but there are {ncycles} cycles. Pass "
            f"one InstrumentalPolarization to apply the same leakage to "
            f"every cycle, or exactly one per cycle.")
    return ips


def build_stokes_cubes(instrument, cycles, fast_axis_offset=None,
                       **kwargs):
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
        Passed to :func:`build_stokes_cube`. ``ip`` is special: it may be a
        single :class:`InstrumentalPolarization` applied to every cycle, or a
        sequence of one per cycle, so a leakage measured on a cycle can be
        removed from that same cycle.

    Returns
    -------
    ndarray
        ``(ncycles, 3, ny, nx)``.
    """
    cycles = list(cycles)
    ips = _ip_per_cycle(kwargs.pop("ip", None), len(cycles))
    cubes = [build_stokes_cube(instrument, cycle,
                               fast_axis_offset=fast_axis_offset, ip=ip,
                               **kwargs)
             for cycle, ip in zip(cycles, ips)]
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

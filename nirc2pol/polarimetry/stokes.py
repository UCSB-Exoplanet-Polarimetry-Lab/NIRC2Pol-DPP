"""Double differencing and Stokes cube production.

Implements Section 3.5 of Lewis et al. (SPIE), assuming an idealized system
(the rotation-approximation instrument model); the full Mueller matrix model
will slot in via ``nirc2pol/polarimetry/mueller.py`` once available.

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

from nirc2pol.utils.angles import mean_angle
from nirc2pol.reduction.registration import register_beam_stack

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
    finite pixel) -- the single-exposure analogue of ``q = Q/I``.

    What it *means* depends entirely on where you measure it, and that is the
    caller's business. Over a region of intrinsically unpolarized starlight it
    estimates the instrumental I -> Q/U leakage for that exposure. Over a
    polarized source it is simply the modulated signal. Over a flat it tells
    you how the two beams' throughput compares, which is how you would check
    whether a flat set is imprinting its source's polarization.

    Note this is *not* an ipq/ipu pair: one exposure sits at a single HWP
    angle, so it carries one modulated combination of Q and U (at HWP 0 it
    is +Q, at 45 deg -Q, at 22.5 deg +U), not both.

    .. warning::

       As a leakage estimate this inherits the weakness of every annulus
       ratio: ``d`` contains any real polarized signal in the mask, so the
       mask has to exclude the source, and the ratio destabilises wherever
       the summed intensity is small. Measured on AB Aur the equivalent
       quantity drifted from -0.9% to -8.6% between r = 22-40 px and
       r = 200-224 px purely as the denominator shrank. Nothing in the
       pipeline uses it for that purpose any more; it is kept as a
       measurement primitive, not as a correction.

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
    from nirc2pol.utils.angles import angles_match

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


def _dither_is_pending(instrument, frames):
    """Does this instrument declare a dither the frames have not had?

    Returns True only when ``"dither"`` is in the background chain AND at
    least one frame lacks ``DITHSUB``. The header is the record of what was
    actually done, so it is what decides -- which makes applying the dither
    idempotent, and lets a caller who already ran it pass through untouched.
    """
    from nirc2pol.reduction.sky import background_stages

    if "dither" not in background_stages(
            getattr(instrument, "background_method", None)):
        return False
    return any(f.get("DITHSUB") in (None, "") for f in frames)


def _crop_planes(planes, size):
    """Crop planes to ``size`` about the registered centre.

    Registration puts the source at ``((ny-1)/2, (nx-1)/2)`` and
    :func:`azimuthal_angle` takes its origin from the same expression, so the
    crop is taken about that exact point and sized odd -- which lands the
    source on ``((size-1)/2)`` in the result. Half a pixel of disagreement
    here leaks about 2% of a tangential disk into U_phi, which is what
    ``validate_synthetic`` check 6 exists to catch.
    """
    from nirc2pol.utils.imutils import crop

    ny, nx = np.asarray(planes[0]).shape
    centre = ((ny - 1) / 2.0, (nx - 1) / 2.0)
    try:
        return tuple(crop(plane, (size, size), center=centre)[0]
                     for plane in planes)
    except ValueError:
        log.warning("A %d px crop does not fit a %dx%d beam, so the planes "
                    "are left uncropped and any dither ghost stays in them.",
                    size, ny, nx)
        return tuple(planes)


def _crop_size_for(planes, cycle, requested=None, margin=8.0):
    """How large a crop keeps the source and drops its dither ghost.

    Parameters
    ----------
    planes : sequence of ndarray
        The registered planes, source already at the array centre.
    cycle : list of Frame
        Supplies ``DITHSEP``, the throw in pixels recorded at dither time.
    requested : int or None, optional
        An explicit ``crop_size``. 0 or negative disables cropping. None
        derives one.
    margin : float, optional
        Pixels of clearance between the crop edge and the ghost.

    Returns
    -------
    int or None
        Side length of the crop, or None for no crop.

    Raises
    ------
    ValueError
        When the throw is no larger than the source, so no crop can separate
        them.

    Notes
    -----
    Derived rather than configured. A pixel count asked of a user means
    converting a throw from arcsec by hand, and getting it wrong fails
    quietly one way -- ghost still in the image -- and destructively the
    other -- source clipped. The throw comes from the commanded offsets and
    the plate scale; the source radius from its own curve of growth.
    """
    from nirc2pol.utils.imutils import curve_of_growth, growth_radius

    if requested is not None:
        return int(requested) if int(requested) > 0 else None

    throw = None
    for frame in cycle:
        value = frame.get("DITHSEP")
        if value:
            throw = float(value)
            break
    if not throw:
        return None                    # not dithered: no ghost to remove

    I = np.asarray(planes[0], dtype=float)
    radii, enclosed = curve_of_growth(I)
    r_src = growth_radius(radii, enclosed, 0.9)
    if not np.isfinite(r_src) or r_src <= 0:
        r_src = 0.0

    if throw <= 2.0 * r_src:
        raise ValueError(
            f"The dither throw is {throw:.0f} px but the source reaches "
            f"{r_src:.0f} px, so it overlaps its own negative ghost and no "
            "crop can separate them. That also means the dither cannot "
            "clean this target: either dither further, or use a different "
            "background_method.")

    half = throw - r_src - margin
    half = min(half, (I.shape[0] - 1) / 2.0, (I.shape[1] - 1) / 2.0)
    if half <= r_src:
        return None                    # array is the binding constraint
    size = int(2 * int(half) + 1)      # odd, so the centre stays a pixel
    log.info("Cropping to %d px: dither throw %.0f px, source radius %.0f px "
             "at 90%% of the flux, %.0f px margin.", size, throw, r_src, margin)
    return size


def build_stokes_cube(instrument, cycle, fast_axis_offset=None,
                      critical_angles=CRITICAL_ANGLES, atol=1.0,
                      register_method="smooth_peak", derotate=True,
                      register_kwargs=None, ip=None, crop_size=None):
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
    if _dither_is_pending(instrument, cycle):
        raise ValueError(
            "background_method declares 'dither' but these frames have no "
            "DITHSUB, and one cycle is not enough to apply it: a dither "
            "partner is usually in another cycle. Call "
            "reduction.subtract_dither_background over all the frames "
            "first, or use build_stokes_cubes, which does it for you.")

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
        from nirc2pol.utils.imutils import rotate_image_center

        north = float(mean_angle([instrument.north_angle(f) for f in cycle]))
        I = rotate_image_center(I, -north)
        q_sky = rotate_image_center(q_sky, -north)
        u_sky = rotate_image_center(u_sky, -north)

    from nirc2pol.utils.provenance import record_step

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

    # A readable keyword as well as the provenance line. The offset was only
    # ever in the header because a caller happened to stamp it there, and
    # anything reading it back -- nirc2pol.combine, or a person -- got None
    # from products built any other way. It is known here, so it is written
    # here, and every product built from this cycle's header inherits it.
    cycle[0]["THETAOFF"] = (effective_offset, "fast axis offset [deg]")

    # Crop out the dither ghost, last, so everything above sees the full
    # field and only the product is trimmed.
    #
    # An explicit size only. Deriving one HERE would measure the source
    # separately in every cycle and get a slightly different answer each
    # time -- 88, 89, 90, 91 px on the 2025-12-06 Io data -- so the cubes
    # would not stack. build_stokes_cubes derives it once and passes it down.
    if crop_size and int(crop_size) > 0:
        I, q_sky, u_sky = _crop_planes((I, q_sky, u_sky), int(crop_size))

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
    cycles = [list(cycle) for cycle in cycles]

    # A config that declares a dither should get one. The stage runs on whole
    # frames before the beams are cut, so recipe.run used to be the only
    # caller and a notebook that set background_method=["dither"] quietly got
    # no background subtraction at all -- the L' pedestal left in, and a
    # flux-weighted centre pulled off the target by it.
    #
    # Applied over the FLATTENED frame list, because a dither partner is
    # usually in a different cycle and a per-cycle pass would find none.
    # subtract_dither_background returns frames in input order, so the cycles
    # reassemble by position.
    flat = [f for cycle in cycles for f in cycle]
    if flat and _dither_is_pending(instrument, flat):
        from nirc2pol.reduction.sky import subtract_dither_background

        log.info("background_method declares 'dither' and these frames have "
                 "not had it, so it is being applied now over all %d frames "
                 "of %d cycles.", len(flat), len(cycles))
        done = subtract_dither_background(flat, instrument)
        at = 0
        for cycle in cycles:
            cycle[:] = done[at:at + len(cycle)]
            at += len(cycle)

    ips = _ip_per_cycle(kwargs.pop("ip", None), len(cycles))

    # Build uncropped, then decide the crop once from the combined result and
    # apply the same size to every cycle. The source radius has to be
    # measured from a built plane, and measuring it per cycle gives a
    # slightly different answer each time -- so cubes cropped independently
    # would not stack.
    requested = kwargs.pop("crop_size", None)
    cubes = [build_stokes_cube(instrument, cycle,
                               fast_axis_offset=fast_axis_offset, ip=ip,
                               **kwargs)
             for cycle, ip in zip(cycles, ips)]
    stacked = np.stack(cubes, axis=0)

    if cycles:
        size = _crop_size_for(np.nanmedian(stacked, axis=0), cycles[0],
                              requested=requested)
        if size:
            stacked = np.stack([np.stack(_crop_planes(cube, size), axis=0)
                                for cube in stacked], axis=0)
    return stacked


def median_stokes_cube(stokes_cubes):
    """Median-combine per-cycle Stokes cubes into one ``(3, ny, nx)`` cube."""
    return np.nanmedian(np.asarray(stokes_cubes), axis=0)


def polarization_products(stokes_cube, min_intensity_frac=0.001):
    """Derived quantities from a ``(3, ny, nx)`` Stokes cube [I, Q, U].

    Parameters
    ----------
    stokes_cube : ndarray
        ``(3, ny, nx)`` as ``[I, Q, U]``.
    min_intensity_frac : float, optional
        DoLP is NaN wherever ``|I|`` is below this fraction of ``max(I)``.
        Set to 0 to divide everywhere.

    Returns
    -------
    tuple of ndarray
        ``(PI, AoLP, DoLP)`` -- polarized intensity ``sqrt(Q^2 + U^2)``,
        angle of linear polarization ``0.5 * arctan2(U, Q)`` in degrees, and
        degree of linear polarization ``PI / I``.

    Notes
    -----
    Only DoLP is masked, because only DoLP is a ratio. Where ``I`` approaches
    zero -- which is most of a frame once the background is subtracted -- it
    divides noise by noise and runs away: on a real L' standard-star frame it
    spanned -2.0e6 to +9.6e5, with 3.7% of pixels exceeding 100%
    polarization, which leaves any display scaled by those extremes showing
    nothing at all.

    Masking hides no information, since a ratio at ``I ~ 0`` had none. The
    absolute value is used deliberately: a pixel where ``I`` came out
    negative from noise is exactly as unusable a denominator as one near
    zero.
    """
    I, Q, U = stokes_cube
    pi = np.sqrt(Q**2 + U**2)
    aolp = 0.5 * np.degrees(np.arctan2(U, Q))
    with np.errstate(divide="ignore", invalid="ignore"):
        dolp = pi / I

    if min_intensity_frac:
        peak = np.nanmax(I)
        if np.isfinite(peak) and peak > 0:
            dolp = np.where(np.abs(I) < min_intensity_frac * peak,
                            np.nan, dolp)
    return pi, aolp, dolp


def aperture_polarization(stokes_cube, center=None, radius=None,
                          background=None, background_method="plane",
                          mask=None):
    """Integrated q, u, p and position angle in a circular aperture.

    Parameters
    ----------
    stokes_cube : ndarray
        ``(3, ny, nx)`` as ``[I, Q, U]``.
    center : tuple of float, optional
        ``(cy, cx)``. Defaults to the centroid found by
        :func:`nirc2pol.utils.imutils.curve_of_growth`.
    radius : float, optional
        Aperture radius in pixels. Defaults to the radius enclosing 90% of
        the flux, from the curve of growth.
    background : tuple of float, optional
        ``(r_inner, r_outer)`` annulus to measure the background in. None
        skips the subtraction, which is almost never right -- see the notes.
    background_method : {"plane", "median", None}, optional
        How to model the background in that annulus. ``"plane"`` fits a
        tilted plane, ``"median"`` a constant.
    mask : ndarray of bool, optional
        True where a pixel may be used, for excluding a companion or the
        zero-filled wedge left by registration.

    Returns
    -------
    dict
        ``radius``, ``center``, ``I``, ``Q``, ``U`` (summed, background
        removed), ``q``, ``u``, ``p`` and ``theta`` in degrees [0, 180), plus
        ``background`` as the per-pixel ``(I, Q, U)`` removed at the centre.

    Notes
    -----
    **The background has to come off Q and U, not only I.** They are
    differences, so the sky is expected to cancel -- and it does not, quite:
    at L-prime the thermal background moves between HWP positions and leaves
    a residual. On the 2025-12-06 standard, U carried a detector-scale
    gradient worth +22 ADU/px at the star. That is nothing next to a
    1.2e5 ADU/px core, but an annulus at r=60-80 has ~9000 pixels and sums it
    into more signal than the star has out there. Left in, p ran from 0.9% at
    r=8 to 4.1% at r=100 and the angle swung 50 degrees; removed, p is flat
    at 0.88-0.95% over that whole range.

    A plane rather than a constant because that gradient was not centred on
    the star, so its level under the aperture differs from its level in the
    annulus.

    **A drifting p is a diagnostic, not a fact about the source.** Aperture
    losses cannot bias it: q, u and I are integrated over the same pixels, so
    clipping the PSF divides out. If p changes with radius, something else is
    -- background, a neighbour, or beam misalignment -- and the fix is to
    find it rather than to pick an aperture.

    This does not correct instrumental polarization. On NIRC2 the I -> Q/U
    leakage is of order 1-2%, which is larger than a typical standard star's
    signal, so a value from here is not comparable with a catalogue one
    unless ``cfg.ip_method`` removed it first.

    To make it comparable -- and to get the fast axis offset out of the same
    measurement -- see
    :func:`nirc2pol.polarimetry.fast_axis.fit_theta_off_polstd`, which runs
    this function once per HWP cycle and solves for theta_off, the leakage
    and the polarimetric efficiency together against a catalogue angle.
    """
    from nirc2pol.utils.imutils import curve_of_growth, growth_radius

    cube = np.asarray(stokes_cube, dtype=float)
    if cube.ndim != 3 or cube.shape[0] < 3:
        raise ValueError("expected a (3, ny, nx) Stokes cube, got shape "
                         f"{cube.shape}")
    I, Q, U = cube[0], cube[1], cube[2]
    ny, nx = I.shape

    radii, enclosed = curve_of_growth(I, center=center, mask=mask)
    if center is None:
        finite = np.where(np.isfinite(I), I, 0.0)
        py, px = np.unravel_index(np.argmax(finite), finite.shape)
        yy, xx = np.mgrid[:ny, :nx]
        box = (np.abs(yy - py) <= 15) & (np.abs(xx - px) <= 15)
        w = np.clip(finite, 0, None) * box
        center = ((yy * w).sum() / w.sum(), (xx * w).sum() / w.sum())
    if radius is None:
        radius = growth_radius(radii, enclosed, 0.9)

    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - center[0], xx - center[1])
    usable = np.isfinite(I) & np.isfinite(Q) & np.isfinite(U)
    if mask is not None:
        usable &= mask

    removed = {}
    if background is not None and background_method:
        ann = usable & (r >= background[0]) & (r < background[1])
        if ann.sum() < 10:
            raise ValueError(
                f"background annulus {background} holds {int(ann.sum())} "
                "usable pixels, too few to fit. Widen it, or check that the "
                "mask is not excluding it.")
        planes = {}
        if background_method == "plane":
            A = np.column_stack([xx[ann], yy[ann], np.ones(int(ann.sum()))])
            for name, arr in (("I", I), ("Q", Q), ("U", U)):
                coef, *_ = np.linalg.lstsq(A, arr[ann], rcond=None)
                planes[name] = coef[0] * xx + coef[1] * yy + coef[2]
        elif background_method == "median":
            for name, arr in (("I", I), ("Q", Q), ("U", U)):
                planes[name] = np.full_like(I, np.nanmedian(arr[ann]))
        else:
            raise ValueError(
                f"unknown background_method {background_method!r}; "
                "use 'plane', 'median' or None")
        I, Q, U = I - planes["I"], Q - planes["Q"], U - planes["U"]
        cy_i, cx_i = int(round(center[0])), int(round(center[1]))
        removed = {name: float(pl[cy_i, cx_i]) for name, pl in planes.items()}

    ap = usable & (r <= radius)
    it, qt, ut = np.sum(I[ap]), np.sum(Q[ap]), np.sum(U[ap])
    q, u = (qt / it, ut / it) if it else (np.nan, np.nan)
    return {
        "radius": float(radius),
        "center": (float(center[0]), float(center[1])),
        "npix": int(ap.sum()),
        "I": float(it), "Q": float(qt), "U": float(ut),
        "q": float(q), "u": float(u),
        "p": float(np.hypot(q, u)),
        "theta": float(np.degrees(0.5 * np.arctan2(u, q)) % 180.0),
        "background": removed,
    }


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

"""Determining the HWP fast axis offset, by two routes with different costs.

theta_off cannot be had from an internal source. Fitting
``A*cos(4*(theta - theta_fit))`` to an HWP ladder returns

    theta_fit = theta_off + chi/2

where ``chi`` is the incident polarization angle in the instrument frame:
the phase is degenerate between the offset and the source's own angle, and
the mod-45 deg wrap on theta_off does not rescue it because ``chi/2`` is mod
90 deg. For a lamp or a dome screen ``chi`` is whatever the optics impose and
is unknown, so the answer is unknowably wrong, and the dated calibration log
that recorded such answers was deleted rather than repaired.

Everything here is a way of supplying ``chi`` from outside the instrument.
There are two, and they fail in different directions.

**The butterfly** (:func:`fit_fast_axis_butterfly`, and the routines carrying
``butterfly`` in their names). Light singly scattered by a circumstellar disk
is polarized tangentially to the scattering plane, so its angle at every
point is fixed by where that point sits relative to the star -- geometry
supplies ``chi``, with no catalogue involved. In this pipeline's convention
(SPIE Eq. 6) all of the signal then belongs in ``Q_phi``::

    Q_phi = +Q cos2phi + U sin2phi
    U_phi = -Q sin2phi + U cos2phi

A wrong theta_off rotates the Q/U frame, turning the familiar four-lobe
pattern rigidly and spilling signal into U_phi; measuring how far it has
turned recovers the offset. The cost is that **it assumes the source is
azimuthally polarized**. On a target where that is the hypothesis under test
it will happily rotate a genuine U_phi signal into Q_phi and report a
confident number, and on an AGN, a merger or a star field the pattern it is
measuring is not there at all.

**A polarized standard** (:func:`fit_theta_off_polstd`). A star whose
polarization angle on sky someone else has measured supplies ``chi`` from a
catalogue instead of from geometry, so it assumes nothing whatever about the
source's morphology -- it is a point source and that is the whole model. It
also does something the butterfly cannot: instrumental polarization is fixed
in the instrument frame while the star's polarization is fixed in the sky
frame, so a standard observed across a spread of field rotation separates
theta_off from the leakage, instead of absorbing one into the other. The
costs are that it needs an external number, that it inherits that number's
calibration and any wavelength extrapolation behind it, and that it needs
real field rotation before the leakage actually separates.

Both routes absorb a constant north-angle or image-parity error into
theta_off, and neither can detect one on its own.

The two share their machinery: :class:`PreparedCycle` and
:func:`prepare_cycles` reduce a sequence to the point where theta_off is
still free, which is what makes trying an offset one arithmetic step rather
than a re-reduction, and both solvers work from there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# theta_rot carries 4*theta_off (SPIE Eq. 3), and Q/U rotate by theta_rot,
# so one degree of offset turns the polarization frame by four.
OFFSET_TO_FRAME = 4.0


def wrap_offset(theta_off):
    """Wrap a fast axis offset into (-22.5, 22.5] deg.

    ``theta_off`` enters the rotation as ``4*theta_off`` and Q/U are
    invariant under 180 deg, so offsets differing by 45 deg are physically
    indistinguishable. Anything outside this range is the same solution
    written differently.
    """
    return float((np.asarray(theta_off) + 22.5) % 45.0 - 22.5)


def butterfly_phase(Q, U, center=None, r_inner=0.0, r_outer=None):
    """How far the butterfly pattern is turned, in degrees.

    The primitive. Give it any Q/U pair — a median cube, one cycle, a
    synthetic test — and it answers a single question: by what extra
    rotation ``delta`` of the Q/U frame would all the azimuthal signal land
    in ``Q_phi``?

    Because only ``2*phi + delta`` enters the radial Stokes definitions, an
    extra frame rotation acts on the integrated radial Stokes as a plain
    2-vector rotation, so the solution is closed-form:

        delta = atan2(sum(U_phi), sum(Q_phi))

    with the branch that makes ``sum(Q_phi)`` positive, i.e. tangential
    rather than radial polarization.

    Parameters
    ----------
    Q, U : ndarray
        Stokes planes in whatever frame you want to test.
    center : (cy, cx), optional
        Centre of the azimuthal pattern — the star. Defaults to the image
        centre, which is correct only if registration put the star there.
    r_inner, r_outer : float
        Annulus [px] over which to integrate. Exclude the occulted or
        saturated core with ``r_inner``; ``r_outer`` defaults to the largest
        circle fitting in the frame.

    Returns
    -------
    float
        ``delta`` in degrees, in (-180, 180]. Divide by
        :data:`OFFSET_TO_FRAME` to convert to a fast axis offset.

    Notes
    -----
    Returns a *frame* rotation, not an offset — the factor of 4 between them
    is the caller's business, and :func:`fit_fast_axis_butterfly` applies it.

    The estimate is flux-weighted through the sums, so it is dominated by
    the brightest part of the disk. It says nothing about whether an
    azimuthal pattern is present at all: on noise it returns a number with
    no meaning. Check ``sum(Q_phi)`` against the noise before believing it.
    """
    from .stokes import radial_stokes

    Q = np.asarray(Q, dtype=float)
    U = np.asarray(U, dtype=float)
    ny, nx = Q.shape
    if center is None:
        center = ((ny - 1) / 2.0, (nx - 1) / 2.0)
    if r_outer is None:
        r_outer = min(ny, nx) / 2.0 - 1.0

    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - center[0], xx - center[1])
    mask = (r >= r_inner) & (r <= r_outer) & np.isfinite(Q) & np.isfinite(U)
    if not mask.any():
        raise ValueError(f"annulus r={r_inner}-{r_outer} px is empty on a "
                         f"{ny}x{nx} frame")

    q_phi, u_phi = radial_stokes(Q, U, center=center)
    return float(np.degrees(np.arctan2(np.nansum(u_phi[mask]),
                                       np.nansum(q_phi[mask]))))


@dataclass
class PreparedCycle:
    """One HWP cycle reduced to the point where theta_off is still free.

    Holds the instrument-frame Stokes planes plus the rotation angle that
    would apply at ``theta_off = 0``. Trying a different offset is then one
    ``rotate_qu`` call rather than a re-reduction, which is what makes
    scanning cheap.

    ``derotated`` records whether ``prepare_cycles`` folded the north angle
    into ``rot_at_zero``. **That fold is exact for radial Stokes and wrong
    for a position angle**, so it is not a detail to be looked up later: see
    :func:`prepare_cycles`, and the guard in :func:`measure_cycles`.
    """

    Q: np.ndarray
    U: np.ndarray
    I: np.ndarray
    rot_at_zero: float      # eff. Q/U rotation [deg] at theta_off = 0
    derotated: bool = False  # is 2*north folded into rot_at_zero?

    def sky_qu(self, theta_off, ip=None):
        """Rotate into the sky frame for a trial offset, removing ``ip``
                first (in the instrument frame, where it belongs).

        Parameters
        ----------
        theta_off : float
            Trial fast axis offset in degrees.
        ip : InstrumentalPolarization, optional
            Leakage removed before rotating.

        Returns
        -------
        tuple of ndarray
            ``(Q_sky, U_sky)`` for this cycle at that offset.
        """
        from .instpol import subtract_ip
        from .stokes import rotate_qu

        Q, U = subtract_ip(self.Q, self.U, self.I, ip)
        return rotate_qu(Q, U, self.rot_at_zero
                         + OFFSET_TO_FRAME * float(theta_off))


def prepare_cycles(instrument, cycles, derotate=True, **dd_kwargs):
    """Reduce cycles to :class:`PreparedCycle` objects, theta_off still free.

        Runs ``double_difference`` once per cycle and records the Q/U rotation
        angle at ``theta_off = 0``. Spatial derotation is folded into that angle
        rather than applied to pixels: rotating an image by an angle shifts every
        azimuth by the same amount, and only ``2*phi + theta`` enters the radial
        Stokes, so ``eff_rot = base_rot + 2*north`` is exactly equivalent and
        costs no interpolation.

        ``**dd_kwargs`` are forwarded to ``double_difference``
        (``register_method``, ``register_kwargs``, ``critical_angles``, ...).

    Parameters
    ----------
    instrument : PolarimetryData
        Instrument to reduce with.
    cycles : list of list of Frame
        Cycles to prepare.
    derotate : bool, optional
        Fold the north angle into the rotation, as the science reduction does.

        **Only correct for radial Stokes.** The equivalence above holds
        because only ``2*phi + theta`` enters Q_phi and U_phi, so turning the
        image and turning the frame are the same thing. For the *absolute*
        position angle of a point source they are not: ``build_stokes_cube``
        rotates Q/U by ``theta_rot`` and then rotates the image by ``-north``
        without touching Q/U again, so folding ``2*north`` in here adds a
        rotation the science products do not have. Measured on the
        2025-12-06 standard it moves the star's position angle from 100.8 deg
        to 179.9. Pass ``derotate=False`` for anything reading an angle off a
        point source -- :func:`measure_cycles` refuses the folded form rather
        than let it pass silently.
    **dd_kwargs
        Passed to ``double_difference``.

    Returns
    -------
    list of PreparedCycle
        One per cycle, with theta_off still free.
    """
    from nirc2pol.utils.angles import mean_angle

    from .stokes import double_difference

    prepared = []
    for cycle in cycles:
        Q, U, I = double_difference(instrument, cycle, **dd_kwargs)
        base = float(mean_angle(
            [instrument.qu_rotation_angle(f, 0.0) for f in cycle]))
        north = (float(mean_angle([instrument.north_angle(f) for f in cycle]))
                 if derotate else 0.0)
        prepared.append(PreparedCycle(Q, U, I, base + 2.0 * north,
                                      derotated=bool(derotate)))
    return prepared


def combine_at_offset(prepared, theta_off, ip=None):
    """Median-combine prepared cycles at a trial offset.

    Each cycle is rotated into the sky frame with its *own* rotation angle
    before combining — they differ, since parallactic angle and elevation
    change through a sequence — and only then median-combined.

    Returns ``(Q, U, I)`` in the sky frame.
    """
    qs, us = zip(*(p.sky_qu(theta_off, ip) for p in prepared))
    return (np.nanmedian(qs, axis=0), np.nanmedian(us, axis=0),
            np.nanmedian([p.I for p in prepared], axis=0))


def _uphi_score(prepared, theta_off, ip, center, r_inner, r_outer,
                score="uphi_sum"):
    """Score one trial offset; lower is better.

    Parameters
    ----------
    prepared : list of PreparedCycle
        Cycles to combine.
    theta_off : float
        Trial offset in degrees.
    ip : InstrumentalPolarization or None
        Leakage removed before rotating.
    center : tuple of float or None
        Centre of the azimuthal pattern.
    r_inner, r_outer : float
        Annulus over which to score.
    score : {"uphi_sum", "uphi_std"}
        Which statistic to use; see :func:`scan_fast_axis_offset_butterfly`.

    Returns
    -------
    float
        The score.

    Raises
    ------
    ValueError
        If ``score`` is not one of the two known statistics.
    """
    from .stokes import radial_stokes

    Q, U, _ = combine_at_offset(prepared, theta_off, ip)
    ny, nx = Q.shape
    c = center or ((ny - 1) / 2.0, (nx - 1) / 2.0)
    ro = r_outer if r_outer is not None else min(ny, nx) / 2.0 - 1.0
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - c[0], xx - c[1])
    mask = (r >= r_inner) & (r <= ro)
    q_phi, u_phi = radial_stokes(Q, U, center=c)

    if score == "uphi_std":
        return float(np.nanstd(u_phi[mask]))
    if score != "uphi_sum":
        raise ValueError(f"unknown score {score!r}; expected 'uphi_sum' or "
                         f"'uphi_std'")
    denom = float(np.nansum(q_phi[mask]))
    return float(abs(np.nansum(u_phi[mask])) / max(abs(denom), 1e-12))


def scan_fast_axis_offset_butterfly(prepared, offsets=None, ip=None, center=None,
                          r_inner=0.0, r_outer=None, score="uphi_sum"):
    """Score a grid of trial offsets. The diagnostic, not the solver.

    **Assumes the source is azimuthally polarized**, exactly as
    :func:`fit_fast_axis_butterfly` does: every available score is a U_phi
    statistic, and driving U_phi to zero is only meaningful when all the
    signal belongs in Q_phi.

    :func:`fit_fast_axis_butterfly` returns one number; this returns the whole
    curve behind it, which is the only way to see whether that number sits
    at a real, isolated minimum. Worth doing at least once per dataset: on
    AB Aur the joint offset/IP minimum turned out to sit ~6 deg away from
    the value the lamp ladder claimed, and only the curve made that obvious.

    Parameters
    ----------
    prepared : list of PreparedCycle
        From :func:`prepare_cycles`.
    offsets : array-like, optional
        Trial offsets [deg]. Defaults to the full non-degenerate range,
        (-22.5, 22.5] in 0.25 deg steps.
    ip : InstrumentalPolarization, optional
        Held fixed while scanning. Note the offset and the leakage are
        partly degenerate, so a scan at fixed ``ip`` is a slice through a
        2-D surface, not the profile of a 1-D one.
    score : {"uphi_sum", "uphi_std"}
        What "bad" means. ``"uphi_sum"`` is ``|sum(U_phi)| / |sum(Q_phi)|``
        over the annulus, which goes to zero at the solution and is the
        quantity :func:`butterfly_phase` solves for directly.

        ``"uphi_std"`` is the spatial scatter of U_phi, which is what you
        want if the concern is structured residuals rather than a frame
        rotation — but it is **blind to the rotation itself**: a rotation
        error puts ``U_phi = -sin(delta) * Q_phi``, so on a smooth disk the
        leaked signal is near-uniform and contributes almost nothing to the
        scatter. Use it as a second opinion, not as the primary metric.

    Returns
    -------
    (offsets, scores) : tuple of ndarray
        Lower is better.
    """
    if offsets is None:
        offsets = np.arange(-22.5, 22.5, 0.25)
    offsets = np.asarray(offsets, dtype=float)
    scores = np.array([_uphi_score(prepared, t, ip, center, r_inner, r_outer,
                                   score)
                       for t in offsets])
    return offsets, scores


@dataclass
class FastAxisResult:
    """Outcome of :func:`fit_fast_axis_butterfly`."""

    theta_off: float
    ip: object = None                 # the IP removed before fitting, if any
    n_iter: int = 0
    converged: bool = False
    delta_history: tuple = ()
    scan: tuple = ()                  # (offsets, scores) if requested

    def describe(self):
        """One-line summary of the fit, for logs.

        Returns
        -------
        str
            The offset, the IP that was removed before fitting if any, and
            whether it converged.
        """
        s = f"theta_off={self.theta_off:+.4f} deg (on-sky butterfly"
        if self.ip is not None:
            s += f", IP removed first: {self.ip.describe()}"
        return s + f", {self.n_iter} iter, converged={self.converged})"


def fit_fast_axis_butterfly(instrument, cycles, ip=None, center=None,
                         r_inner=20.0, r_outer=None, max_iter=10, tol=1e-3,
                         derotate=True, scan=False, prepared=None,
                         **dd_kwargs):
    """Measure theta_off from the butterfly's orientation. The solver.

    **Assumes the source is azimuthally polarized** -- a disk in scattered
    light, whose signal all belongs in Q_phi. The offset is read off how far
    the butterfly has turned, so on a target where azimuthal polarization is
    the hypothesis under test this returns a confident, meaningless number.

    Iterates one closed-form step until it stops moving: rotate to sky at the
    current offset, measure how far the butterfly is still turned
    (:func:`butterfly_phase`), and correct the offset by ``delta / 4``.

    **This fits the offset only.** The leakage is a separate choice, made
    through the IP routines in :mod:`nirc2pol.polarimetry.instpol`, and is supplied
    here through ``ip`` rather than fitted alongside. That matters because
    the two are **degenerate**: a constant leakage tilts the integrated
    radial Stokes just as a frame rotation does, so an offset fitted with the
    leakage still in it is biased by however much IP there is.

    Which means the order the two are done in is not free. Every IP route
    currently offered as an ``ip_method`` -- :func:`nirc2pol.polarimetry.fit_ip_uphi`
    and :func:`nirc2pol.polarimetry.fit_ip_uphi_all` -- takes the offset as an *input*,
    so the offset has to be fitted first with ``ip=None``, and is biased.

    A leakage established some other way -- one that did not itself need an
    offset -- can be passed in here instead, which removes the bias. Nothing
    in the package currently produces one, so this path is for a value you
    bring from outside.

    Parameters
    ----------
    ip : InstrumentalPolarization, optional
        A leakage to remove before fitting, not one to fit. Any measurement
        that did not itself need an offset can go here, and the offset comes
        back unbiased. Leaving this None fits the offset with the leakage
        still present, which biases it.
    r_inner, r_outer : float
        Annulus [px] holding the disk -- this fit works on U_phi, where a
        tangentially polarized disk contributes nothing by definition, so the
        annulus should *span* the disk rather than avoid it -- the opposite
        of what an estimator working on Q/U would want, where the disk does
        contribute. ``r_inner`` must clear the
        occulted or saturated core; the default 20 px suits NIRC2
        coronagraphic data and should be checked against the actual mask.
    prepared : list of PreparedCycle, optional
        Reuse an existing reduction instead of redoing it — handy when
        fitting and scanning the same data.
    scan : bool
        Also compute the score curve and return it on the result. Cheap
        relative to the reduction, and worth having.

    Returns
    -------
    FastAxisResult
        ``theta_off`` is wrapped into (-22.5, 22.5]; offsets 45 deg apart
        are the same solution.

    Warnings
    --------
    Assumes azimuthal polarization. On an AGN, a merger or a star field this
    returns a number that means nothing — the pattern it is measuring is not
    there. Check that the integrated Q_phi is significant first.
    """
    if prepared is None:
        prepared = prepare_cycles(instrument, cycles, derotate=derotate,
                                  **dd_kwargs)

    theta_off = 0.0
    history, converged = [], False
    for it in range(1, max_iter + 1):
        Q, U, _ = combine_at_offset(prepared, theta_off, ip)
        delta = butterfly_phase(Q, U, center=center, r_inner=r_inner,
                                r_outer=r_outer)
        theta_off = wrap_offset(theta_off + delta / OFFSET_TO_FRAME)
        history.append(delta)

        if abs(delta / OFFSET_TO_FRAME) < tol:
            converged = True
            break

    result = FastAxisResult(theta_off=theta_off, ip=ip, n_iter=it,
                            converged=converged,
                            delta_history=tuple(history))
    if scan:
        result.scan = scan_fast_axis_offset_butterfly(prepared, ip=ip, center=center,
                                            r_inner=r_inner, r_outer=r_outer)
    if not converged:
        log.warning("On-sky fast axis fit did not converge in %d iterations "
                    "(last correction %.4f deg); the source may not be "
                    "azimuthally polarized", max_iter,
                    history[-1] / OFFSET_TO_FRAME if history else float("nan"))
    log.info("On-sky fast axis: %s", result.describe())
    return result


# ---------------------------------------------------------------------------
# The polarized-standard route. Same PreparedCycle machinery, different
# source of chi: a catalogue angle instead of the geometry of a disk.
# ---------------------------------------------------------------------------


@dataclass
class ApertureCycles:
    """Per-cycle aperture polarimetry, with theta_off still free.

    What :func:`measure_cycles` hands to the solver. One complex number per
    HWP cycle plus the rotation that would apply at ``theta_off = 0``, which
    is all the fit needs -- the images are not carried any further.

    Attributes
    ----------
    z : ndarray of complex
        ``q + i*u`` in the **instrument** frame, one per cycle, normalized
        by the intensity in the same aperture.
    base : ndarray
        Each cycle's ``rot_at_zero`` [deg]: the Q/U rotation at zero offset,
        with the north angle already folded in.
    flux : ndarray
        Summed intensity in the aperture, background removed. Not used by
        the fit; kept because a cycle whose flux is wildly out of line with
        the rest is usually a cycle worth dropping.
    center : tuple of float
        ``(cy, cx)`` the aperture was placed at -- resolved once and shared
        by every cycle.
    radius : float
        Aperture radius [px], likewise shared.
    """

    z: np.ndarray
    base: np.ndarray
    flux: np.ndarray
    center: tuple
    radius: float


def _angular_span(angles_deg):
    """Circular range of a set of angles [deg]: the smallest arc holding all.

    ``max - min`` is wrong for angles -- a set straddling 0 would report
    nearly 360 deg of spread when it has almost none. This finds the largest
    empty gap and returns what is left.
    """
    a = np.sort(np.mod(np.asarray(angles_deg, dtype=float), 360.0))
    if a.size < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, a[:1] + 360.0]))
    return float(360.0 - gaps.max())


def measure_cycles(prepared, center=None, radius=None, background=None,
                   background_method="plane", mask=None, growth_frac=0.9):
    """Aperture polarimetry on every prepared cycle, in the instrument frame.

    The measurement half of the polarized-standard route. Each cycle's Q, U
    and I are summed over one aperture on the star and written as a single
    complex ``z = q + i*u``, which is the form that rotates cleanly:
    :func:`nirc2pol.polarimetry.rotate_qu` is exactly multiplication by
    ``exp(-i*theta_rot)``, so a whole sequence's worth of geometry reduces to
    one number per cycle and the offset never has to touch a pixel again.

    Parameters
    ----------
    prepared : list of PreparedCycle
        From :func:`prepare_cycles`, **with ``derotate=False``**. These hold
        instrument-frame Q/U/I and ``rot_at_zero``, which is precisely the
        per-cycle rotation the fit needs, so no re-reduction happens here.
        Cycles prepared with ``derotate=True`` are refused: that fold is
        right for radial Stokes and wrong for a position angle.
    center : tuple of float, optional
        ``(cy, cx)`` of the star. Resolved from the median intensity if not
        given.
    radius : float, optional
        Aperture radius [px]. Defaults to the radius enclosing
        ``growth_frac`` of the median intensity.
    background : tuple of float, optional
        ``(r_inner, r_outer)`` annulus, passed to
        :func:`nirc2pol.polarimetry.aperture_polarization`. **Use it.** The
        background does not cancel out of Q and U as cleanly as differencing
        suggests, and left in it has been measured moving p by a factor of
        four and the angle by tens of degrees.
    background_method : {"plane", "median", None}, optional
        How that annulus is modelled. A plane by default, because the
        residual gradient is not centred on the star.
    mask : ndarray of bool, optional
        True where a pixel may be used. Needed in practice, not decoration:
        a companion or the zero-filled wedge registration leaves behind will
        otherwise sit inside the background annulus and be fitted as sky.
    growth_frac : float, optional
        Enclosed-flux fraction defining the default radius.

    Returns
    -------
    ApertureCycles

    Notes
    -----
    **The aperture is resolved once and reused for every cycle.**
    ``aperture_polarization`` will happily pick its own centre and radius per
    call, from its own curve of growth; letting it do that here would give
    each cycle a subtly different aperture and put the difference straight
    into the fit as scatter. So the defaults are settled first, on the
    median intensity, and then passed explicitly.

    p is a ratio over a shared aperture, so clipping the PSF cannot bias it
    -- which is why a p that moves with radius is a statement about the
    background rather than about the star. :func:`curve_of_growth_polarization`
    is the standing check on that.
    """
    from .stokes import aperture_polarization

    prepared = list(prepared)
    if not prepared:
        raise ValueError("measure_cycles needs at least one prepared cycle")
    if any(getattr(c, "derotated", False) for c in prepared):
        raise ValueError(
            "these cycles were prepared with derotate=True, which folds "
            "2*north into rot_at_zero. That fold is exact for radial Stokes, "
            "where only 2*phi + theta enters, but it is wrong for the "
            "position angle of a point source: build_stokes_cube rotates Q/U "
            "by theta_rot and derotates the image without touching Q/U "
            "again, so the fold adds a rotation the science products do not "
            "have. On the 2025-12-06 standard it moves the measured angle by "
            "79 deg. Re-run prepare_cycles with derotate=False.")

    if center is None or radius is None:
        # Resolve the defaults through aperture_polarization itself rather
        # than repeating its centroid and curve-of-growth logic here, so the
        # two cannot drift apart. Q and U are zeroed because only I decides
        # where the star is and how big the aperture should be.
        reference = np.nanmedian([p.I for p in prepared], axis=0)
        blank = np.zeros_like(reference)
        probe = aperture_polarization(np.stack([reference, blank, blank]),
                                      center=center, radius=radius,
                                      background=None, mask=mask)
        center = probe["center"] if center is None else center
        radius = probe["radius"] if radius is None else radius
        log.info("aperture resolved once for all cycles: centre (%.2f, %.2f), "
                 "r = %.1f px (%.0f%% enclosed)", center[0], center[1], radius,
                 100 * growth_frac)

    z, base, flux = [], [], []
    for cycle in prepared:
        result = aperture_polarization(
            np.stack([cycle.I, cycle.Q, cycle.U]), center=center,
            radius=radius, background=background,
            background_method=background_method, mask=mask)
        z.append(result["q"] + 1j * result["u"])
        base.append(cycle.rot_at_zero)
        flux.append(result["I"])

    return ApertureCycles(np.array(z), np.array(base, dtype=float),
                          np.array(flux, dtype=float),
                          (float(center[0]), float(center[1])), float(radius))


@dataclass
class PolStdResult:
    """Outcome of :func:`fit_theta_off_polstd`.

    Attributes
    ----------
    theta_off, theta_off_err : float
        The offset [deg], wrapped into (-22.5, 22.5], and a jackknife error
        over cycles. The error is statistical only: it says nothing about
        whether the catalogue angle was right, and that is the dominant term
        in practice -- a catalogue good to 2 deg puts 1 deg here on its own.
    ip : InstrumentalPolarization or None
        The leakage, when ``fit_ip=True``. None otherwise, which means the
        leakage is still in ``theta_off``, not that there is none.
    efficiency, efficiency_err : float
        ``|c|``: how much of the standard's catalogue polarization came
        through. Not purely an instrument property -- it absorbs the
        catalogue p and any wavelength extrapolation behind it, so read a
        value far from 1 as a doubtful p before reading it as a doubtful
        instrument.
    n_cycles : int
        Cycles pooled, over every dataset.
    rotation_span : float
        Circular range [deg] of the source term's phase. This is what makes
        the leakage separable; see ``condition_number``.
    condition_number : float
        Of the weighted design matrix. Large means the fit is amplifying
        noise to tell ip and the source apart -- the number to look at
        before believing a jointly fitted leakage.
    residuals : ndarray
        Weighted complex residuals, one per cycle.
    per_cycle_theta_off : tuple of float
        Each cycle's own closed-form offset, with any fitted leakage removed
        first. Their spread is the honest scatter, and a single cycle far
        from the rest shows up here and nowhere else.
    aperture : tuple
        One ``((cy, cx), radius)`` per dataset, in the order given.
    standards : tuple of str
        ``describe()`` of each standard used, for provenance.
    diagnostics : dict
        The covariance-based error for comparison with the jackknife,
        whether the fit was weighted, and the per-dataset cycle counts.
    """

    theta_off: float
    theta_off_err: float = float("nan")
    ip: object = None                 # InstrumentalPolarization, if fitted
    efficiency: float = float("nan")
    efficiency_err: float = float("nan")
    n_cycles: int = 0
    rotation_span: float = float("nan")
    condition_number: float = float("nan")
    residuals: np.ndarray = None
    per_cycle_theta_off: tuple = ()
    aperture: tuple = ()              # (center, radius)
    standards: tuple = ()
    diagnostics: dict = None

    def describe(self):
        """One-line summary of the fit, for logs."""
        s = (f"theta_off={self.theta_off:+.4f}+/-{self.theta_off_err:.4f} deg "
             f"(polarized standard, {self.n_cycles} cycles, "
             f"{self.rotation_span:.1f} deg of rotation")
        if self.ip is not None:
            s += f", IP fitted jointly: {self.ip.describe()}"
        return s + f", efficiency {self.efficiency:.3f}, cond {self.condition_number:.1f})"


def _polstd_datasets(prepared, standard):
    """Normalise the two call shapes into a list of (prepared, standard).

    ``fit_theta_off_polstd(cycles, standard)`` for one sequence, or
    ``fit_theta_off_polstd([(cycles_a, std_a), (cycles_b, std_b)])`` for
    several pooled together.
    """
    if standard is not None:
        return [(list(prepared), standard)]

    datasets = []
    for entry in prepared:
        # Deliberately not a bare two-way unpack: a sequence of exactly two
        # PreparedCycle objects passed without a standard would unpack
        # without complaint and be read as one cycle plus one standard.
        pair = tuple(entry) if isinstance(entry, (tuple, list)) else ()
        if len(pair) != 2 or not (hasattr(pair[1], "p")
                                  and hasattr(pair[1], "theta")):
            raise ValueError(
                "fit_theta_off_polstd takes either (prepared, standard) or a "
                "list of (prepared, standard) pairs. Got an entry that is not "
                f"a (cycles, PolarizedStandard) pair: {entry!r}. If you meant "
                "one sequence, pass the standard as the second argument.")
        datasets.append((list(pair[0]), pair[1]))
    if not datasets:
        raise ValueError("fit_theta_off_polstd needs at least one dataset")
    return datasets


def _solve(design, observed, weights, fit_ip):
    """Weighted complex least squares; returns (x, cond, residuals, cov)."""
    A = design * weights[:, None]
    b = observed * weights
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    residuals = b - A @ x
    cond = float(np.linalg.cond(A))

    # Complex least squares on n complex observations is 2n real ones, and
    # each complex parameter is two real ones.
    dof = 2 * len(b) - 2 * (2 if fit_ip else 1)
    cov = None
    if dof > 0:
        sigma2 = float(np.sum(np.abs(residuals) ** 2)) / dof
        try:
            cov = sigma2 * np.linalg.inv(A.conj().T @ A)
        except np.linalg.LinAlgError:
            cov = None
    return x, cond, residuals, cov


def fit_theta_off_polstd(prepared=None, standard=None, fit_ip=False,
                         center=None, radius=None, background=None,
                         background_method="plane", mask=None,
                         measurements=None, low_span_deg=30.0):
    """Measure theta_off against a standard's known polarization. The solver.

    The alternative to :func:`fit_fast_axis_butterfly`, and the one that
    assumes nothing about the source's morphology. Where the butterfly gets
    the incident polarization angle from the geometry of a disk, this gets it
    from a catalogue, so it works on a point source and says nothing about
    whether anything is azimuthally polarized.

    The model is one line. A source of sky-frame polarization
    ``p*exp(2i*theta_known)``, seen through a rotation ``theta_rot =
    base_k + 4*theta_off`` and a leakage fixed in the instrument frame, gives
    each cycle's aperture measurement as::

        z_k = ip + c * s_k       s_k = p * exp(i(2*theta_known + base_k))
                                 c   = efficiency * exp(i * 4*theta_off)

    which is **linear in ip and c**. Two complex unknowns, four real ones --
    ipq, ipu, the polarimetric efficiency ``|c|`` and
    ``theta_off = arg(c)/4`` -- and no optimizer: it is a least squares with
    a closed form, the same shape as :func:`butterfly_phase`.

    Parameters
    ----------
    prepared : list of PreparedCycle, or list of (prepared, standard) pairs
        One sequence with ``standard`` given, or several pooled by passing
        pairs and leaving ``standard`` as None. Pooling is not a convenience:
        two standards at *different* sky angles separate the leakage even
        when neither sequence has much field rotation, which is the realistic
        way to get an IP measurement out of short calibration sequences.
    standard : PolarizedStandard, optional
        The known polarization. Omit when passing pairs.
    fit_ip : bool, optional
        Fit the I -> Q/U leakage jointly. Off by default because that is the
        well-posed estimator. Which one is *better* is a real trade and
        depends on the data -- see the note below; do not reach for
        ``fit_ip=True`` reflexively.
    center, radius, background, background_method, mask
        Passed to :func:`measure_cycles`. Ignored if ``measurements`` is
        given.
    measurements : ApertureCycles, optional
        Reuse an existing measurement instead of redoing the photometry --
        useful when fitting the same data several ways. Only valid for a
        single dataset.
    low_span_deg : float, optional
        Below this much field rotation, ``fit_ip=True`` warns. Not a refusal:
        pooled standards at different angles can be well conditioned with
        very little rotation.

    Returns
    -------
    PolStdResult
        ``theta_off`` wrapped into (-22.5, 22.5]; offsets 45 deg apart are
        the same solution. ``ip`` is an
        :class:`~nirc2pol.polarimetry.InstrumentalPolarization` when fitted
        and None otherwise.

    Warnings
    --------
    **The two known inputs are not equally trustworthy, and they fail
    independently.** ``standard.theta`` enters the *phase* of ``s_k`` and
    ``standard.p`` enters its *modulus*, so a p that is wrong -- and at L' it
    usually is, being a long Serkowski extrapolation -- moves the fitted
    efficiency and leaves theta_off essentially where it was. Read a bad
    efficiency as a bad p before reading it as a bad offset.

    The angle is the demanding input, and by a factor of two: theta_known
    enters as ``2*theta_known`` while theta_off enters as ``4*theta_off``, so
    an error ``d`` in the catalogue angle moves theta_off by ``d/2``. The
    catalogue angle has to be good to about 2 deg for theta_off to be good
    to 1.

    **A constant north-angle or image-parity error is absorbed into
    theta_off** and cannot be detected here. Two standards at different sky
    angles can detect it; one cannot, however many cycles it has.

    **``fit_ip=True`` needs field rotation.** With ``s_k`` nearly constant
    the leakage and the source term are the same column and the fit cannot
    tell them apart. The result carries ``rotation_span`` and
    ``condition_number`` so this is visible rather than implied.

    Notes
    -----
    **Choosing fit_ip is a bias-against-variance trade, and at a short
    rotation span the biased answer can be the better one.** Leaving the
    leakage in does not scatter the offset, it displaces it; fitting it
    removes the displacement but amplifies the noise by the condition
    number, which is what a short span makes large. Measured on synthetics
    -- 27 cycles, a 0.85% leakage against a 0.93% star, ``sigma`` the
    per-cycle scatter on q and u::

        span    sigma      fit_ip=False          fit_ip=True
                          bias   scatter       bias   scatter
        16 deg  0.05%     0.70     0.08        0.03     1.78
        16 deg  0.20%     0.71     0.29        2.48    11.05
        60 deg  0.20%     0.67     0.30        0.01     1.95
        60 deg  0.50%     0.75     0.82        0.20     7.16

    So at 16 deg -- which is what the Schulte 19 sequence actually has --
    fitting the leakage wins only while the per-cycle scatter stays below
    roughly a twentieth of the star's own polarization, and loses badly
    above it. At 60 deg it wins comfortably.

    The practical test needs no theory: **fit it both ways.** If the two
    offsets differ by less than the ``theta_off_err`` that ``fit_ip=True``
    reports, the fit is not measuring the leakage, and the tighter biased
    answer is the more useful one -- with the bias stated. If they differ by
    much more, the leakage is real and being removed.

    The bias itself is not enormous and does not grow without limit: for a
    leakage comparable to the star's own polarization it is of order 0.7 deg
    in theta_off, roughly independent of span until the rotation coverage
    becomes very wide, at which point it averages down.
    """
    from .instpol import InstrumentalPolarization
    from nirc2pol.utils.angles import mean_angle

    datasets = _polstd_datasets(prepared, standard)
    if measurements is not None and len(datasets) > 1:
        raise ValueError("measurements= reuses a single dataset's photometry; "
                         "it cannot stand in for several pooled datasets")

    # theta_err is what the weighting uses, since the angle is what
    # constrains the offset. Mixing weighted and unweighted rows would let a
    # standard that simply forgot to state an error dominate the fit, so it
    # is all or nothing.
    have_errors = all(d[1].theta_err for d in datasets)
    if not have_errors and any(d[1].theta_err for d in datasets):
        log.warning("Some standards give theta_err and some do not; weighting "
                    "every cycle equally rather than letting the ones without "
                    "a stated error dominate.")

    z, s, weights, spans, cycles_per, apertures = [], [], [], [], [], []
    for cycles, std in datasets:
        measured = (measurements if measurements is not None
                    else measure_cycles(cycles, center=center, radius=radius,
                                        background=background,
                                        background_method=background_method,
                                        mask=mask))
        apertures.append((measured.center, measured.radius))
        phase = np.radians(2.0 * std.theta + measured.base)
        s_k = std.p * np.exp(1j * phase)

        if have_errors:
            # An angle error d on the standard displaces the source term by
            # |s|*2d, so that -- not the angle itself -- is the sigma.
            sigma = np.abs(s_k) * 2.0 * np.radians(std.theta_err)
            w = 1.0 / np.where(sigma > 0, sigma, np.inf)
        else:
            w = np.ones_like(measured.base)

        z.append(measured.z)
        s.append(s_k)
        weights.append(w)
        spans.append(np.degrees(phase))
        cycles_per.append(len(measured.z))

    z = np.concatenate(z)
    s = np.concatenate(s)
    weights = np.concatenate(weights)
    rotation_span = _angular_span(np.concatenate(spans))
    n = len(z)

    if n < (2 if fit_ip else 1):
        raise ValueError(f"{n} cycle(s) cannot constrain "
                         f"{'ip and the offset' if fit_ip else 'the offset'}")

    design = (np.column_stack([np.ones(n, dtype=complex), s]) if fit_ip
              else s[:, None])
    x, cond, residuals, cov = _solve(design, z, weights, fit_ip)

    ip_complex = complex(x[0]) if fit_ip else 0j
    c = complex(x[-1])
    theta_off = wrap_offset(np.degrees(np.angle(c)) / OFFSET_TO_FRAME)
    efficiency = float(np.abs(c))

    # Per-parameter errors from the covariance. A complex parameter's error
    # is spread over its two real components, hence the halving.
    theta_off_err = efficiency_err = float("nan")
    ip_err = None
    if cov is not None:
        var_c = float(np.real(cov[-1, -1])) / 2.0
        if var_c >= 0:
            efficiency_err = float(np.sqrt(var_c))
            if efficiency > 0:
                theta_off_err = float(np.degrees(np.sqrt(var_c) / efficiency)
                                      / OFFSET_TO_FRAME)
        if fit_ip:
            var_ip = float(np.real(cov[0, 0])) / 2.0
            ip_err = float(np.sqrt(var_ip)) if var_ip >= 0 else None

    # Each cycle's own closed-form answer, with any fitted leakage removed
    # first. The spread across these is the honest scatter, and it is also
    # the only place a single bad cycle shows itself.
    per_cycle = np.degrees(np.angle((z - ip_complex) * np.conj(s))) / OFFSET_TO_FRAME
    per_cycle = np.array([wrap_offset(t) for t in per_cycle])
    if len(per_cycle):
        # Circular, in the 4*theta_off domain where the wrap actually lives:
        # a plain mean is wrong across it, the same bug already fixed once in
        # stokes.py.
        per_cycle_mean = wrap_offset(
            mean_angle(OFFSET_TO_FRAME * per_cycle, period=360.0)
            / OFFSET_TO_FRAME)
    else:
        per_cycle_mean = float("nan")

    jackknife = _jackknife_theta_off(design, z, weights, fit_ip)
    if np.isfinite(jackknife):
        theta_off_err = jackknife

    ip = None
    if fit_ip:
        ip = InstrumentalPolarization(
            float(ip_complex.real), float(ip_complex.imag),
            method="polstd", scope="sequence_joint",
            diagnostics={"ip_err": ip_err,
                         "condition_number": cond,
                         "rotation_span": rotation_span,
                         "n_cycles": n,
                         "standards": [d[1].name for d in datasets]})

    result = PolStdResult(
        theta_off=theta_off, theta_off_err=theta_off_err, ip=ip,
        efficiency=efficiency, efficiency_err=efficiency_err, n_cycles=n,
        rotation_span=rotation_span, condition_number=cond,
        residuals=residuals, per_cycle_theta_off=tuple(per_cycle),
        aperture=tuple(apertures),
        standards=tuple(d[1].describe() for d in datasets),
        diagnostics={"per_cycle_mean": per_cycle_mean,
                     "theta_off_err_covariance": (
                         float(np.degrees(np.sqrt(np.real(cov[-1, -1]) / 2.0)
                                          / efficiency) / OFFSET_TO_FRAME)
                         if cov is not None and efficiency > 0 else None),
                     "cycles_per_dataset": cycles_per,
                     "weighted": bool(have_errors),
                     "fit_ip": bool(fit_ip)})

    if fit_ip and rotation_span < low_span_deg:
        log.warning(
            "Fitting IP jointly across only %.1f deg of field rotation "
            "(condition number %.1f). The leakage and the source term are "
            "nearly the same column at this span, so it is ip and the "
            "efficiency that are poorly determined -- theta_off is the "
            "parameter the angle constrains and it survives this far better. "
            "Compare this fit against fit_ip=False: if they differ by less "
            "than theta_off_err here, the leakage is not being measured and "
            "the biased fit is tighter. Pooling a second standard at a "
            "different sky angle separates them properly.",
            rotation_span, cond)
    if not fit_ip:
        log.warning(
            "theta_off fitted with no leakage removed. On NIRC2 the I -> Q/U "
            "leakage is of order 1-2%%, comparable to or larger than a "
            "standard's own polarization in the infrared, and it displaces "
            "this offset by of order 0.7 deg on synthetics. Whether "
            "fit_ip=True is an improvement depends on the rotation span -- "
            "fit it both ways and compare the difference against the error "
            "bar the joint fit reports.")
    log.info("Polarized standard fast axis: %s", result.describe())
    return result


def _jackknife_theta_off(design, observed, weights, fit_ip):
    """Leave-one-out standard error on theta_off [deg].

    Quoted alongside the covariance error because the two fail differently:
    the covariance assumes the residuals are what the model says they are,
    while this only assumes the cycles are exchangeable. Where they disagree,
    something is wrong with one cycle rather than with the noise.
    """
    n = len(observed)
    npar = 2 if fit_ip else 1
    if n < npar + 2:
        return float("nan")

    keep = np.ones(n, dtype=bool)
    offsets = []
    for i in range(n):
        keep[i] = False
        x, *_ = _solve(design[keep], observed[keep], weights[keep], fit_ip)
        offsets.append(np.degrees(np.angle(complex(x[-1]))) / OFFSET_TO_FRAME)
        keep[i] = True

    # In the 4*theta_off domain, where the 45 deg wrap becomes a 180 deg one
    # and deviations can be taken modulo it without special cases.
    scaled = OFFSET_TO_FRAME * np.array(offsets)
    mean = np.angle(np.mean(np.exp(1j * np.radians(scaled))), deg=True)
    deviations = (scaled - mean + 180.0) % 360.0 - 180.0
    variance = (n - 1) / n * float(np.sum(deviations ** 2))
    return float(np.sqrt(variance) / OFFSET_TO_FRAME)


def curve_of_growth_polarization(prepared, theta_off, radii=None, ip=None,
                                 center=None, background=None,
                                 background_method="plane", mask=None):
    """Sky-frame p and position angle against aperture radius. The diagnostic.

    Not a fit and not circular: ``theta_off`` is held **fixed** at whatever
    you already believe, and the question asked is whether the star's measured
    polarization depends on how much of it you integrate. It should not. p is
    a ratio over a shared aperture, so clipping the PSF divides out; if p or
    the angle move with radius, something that is not the star is in the
    aperture -- a residual background, a neighbour, a beam misalignment -- and
    the answer is to find it, not to choose a radius.

    This is the standing check on the background treatment. Before the
    background was taken off Q and U as well as I, this curve ran p from 0.9%
    to 4.1% and swung the angle by tens of degrees; after, it is flat.

    Parameters
    ----------
    prepared : list of PreparedCycle
        From :func:`prepare_cycles`.
    theta_off : float
        Held fixed [deg]. Use the value you are testing -- from
        :func:`fit_theta_off_polstd`, from the butterfly, or a trial.
    radii : array-like, optional
        Aperture radii [px]. Defaults to 10..100 in steps of 10.
    ip : InstrumentalPolarization, optional
        Removed in the instrument frame before rotating, where it belongs.
    center, background, background_method, mask
        As for :func:`measure_cycles`.

    Returns
    -------
    dict of ndarray
        ``radius``, ``p``, ``theta`` [deg, 0-180), ``q``, ``u`` -- the
        cycle-combined sky-frame values at each radius.

    Notes
    -----
    Costs one aperture measurement per radius per cycle, and each of those
    builds its own curve of growth internally, so a long radius list over
    many cycles is not free. Start coarse.

    Cycles are combined by averaging the complex sky-frame values, which is
    the vector mean of q and u -- the right thing for Stokes parameters,
    where averaging p and theta separately would not be.
    """
    if radii is None:
        radii = np.arange(10.0, 101.0, 10.0)
    radii = np.asarray(radii, dtype=float)
    ip_complex = 0j if ip is None else complex(ip.ipq, ip.ipu)

    q, u, p, theta = [], [], [], []
    for r in radii:
        measured = measure_cycles(prepared, center=center, radius=float(r),
                                  background=background,
                                  background_method=background_method,
                                  mask=mask)
        rotation = np.radians(measured.base
                              + OFFSET_TO_FRAME * float(theta_off))
        sky = np.mean((measured.z - ip_complex) * np.exp(-1j * rotation))
        q.append(sky.real)
        u.append(sky.imag)
        p.append(abs(sky))
        theta.append(np.degrees(0.5 * np.angle(sky)) % 180.0)

    return {"radius": radii, "q": np.array(q), "u": np.array(u),
            "p": np.array(p), "theta": np.array(theta)}

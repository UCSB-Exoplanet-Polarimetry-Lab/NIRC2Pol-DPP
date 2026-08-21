"""Determining the HWP fast axis offset from an azimuthally polarized source.

Every routine here measures the offset from the orientation of the
**butterfly** -- the four-lobe pattern a tangentially polarized disk makes in
Q/U -- which is why they carry ``butterfly`` in their names. That is a real
restriction, not a detail of the implementation: see the warning at the end.

There is no trusted lamp-ladder route to theta_off. Fitting
``A*cos(4*(theta - theta_fit))`` to an HWP ladder returns

    theta_fit = theta_off + chi/2

where ``chi`` is the incident polarization angle in the instrument frame:
the phase is degenerate between the offset and the source's own angle, and
the mod-45 deg wrap on theta_off does not rescue it because ``chi/2`` is mod
90 deg. For an internal source ``chi`` is whatever the optics impose and is
unknown, so the answer is unknowably wrong.

On sky the missing ingredient comes from geometry instead of a catalogue.
Light singly scattered by a circumstellar disk is polarized **tangentially**
to the scattering plane, so its angle at every point is fixed by where that
point sits relative to the star. In this pipeline's convention (SPIE Eq. 6)
that means all of the signal belongs in ``Q_phi`` and ``U_phi`` is noise:

    Q_phi = +Q cos2phi + U sin2phi
    U_phi = -Q sin2phi + U cos2phi

A wrong theta_off rotates the Q/U frame, which turns the familiar four-lobe
"butterfly" pattern rigidly and spills signal into U_phi. Measuring how far
it has turned recovers the offset.

**This assumes the source is azimuthally polarized.** On a target where that
is the hypothesis under test, these routines will happily rotate a genuine
U_phi signal into Q_phi and report a confident number. There is currently no
route to theta_off that does not make this assumption -- the lamp-ladder
route is degenerate, as above, and its log was deleted deliberately.
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
    """

    Q: np.ndarray
    U: np.ndarray
    I: np.ndarray
    rot_at_zero: float      # eff. Q/U rotation [deg] at theta_off = 0

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
    **dd_kwargs
        Passed to ``double_difference``.

    Returns
    -------
    list of PreparedCycle
        One per cycle, with theta_off still free.
    """
    from utils.angles import mean_angle

    from .stokes import double_difference

    prepared = []
    for cycle in cycles:
        Q, U, I = double_difference(instrument, cycle, **dd_kwargs)
        base = float(mean_angle(
            [instrument.qu_rotation_angle(f, 0.0) for f in cycle]))
        north = (float(mean_angle([instrument.north_angle(f) for f in cycle]))
                 if derotate else 0.0)
        prepared.append(PreparedCycle(Q, U, I, base + 2.0 * north))
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
    through the IP routines in :mod:`polarimetry.instpol`, and is supplied
    here through ``ip`` rather than fitted alongside. That matters because
    the two are **degenerate**: a constant leakage tilts the integrated
    radial Stokes just as a frame rotation does, so an offset fitted with the
    leakage still in it is biased by however much IP there is.

    Which means the order the two are done in is not free. Every IP route
    currently offered as an ``ip_method`` -- :func:`polarimetry.fit_ip_uphi`
    and :func:`polarimetry.fit_ip_uphi_all` -- takes the offset as an *input*,
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

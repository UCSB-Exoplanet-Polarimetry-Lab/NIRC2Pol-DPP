"""Empirical instrumental polarization (IP) measurement and removal.

IP here means the I -> Q/U crosstalk that leaks a fraction of the total
intensity into the polarized channels: an unpolarized source comes out with
``Q = ipq * I`` and ``U = ipu * I``. It is a property of the optical train,
not of the sky.

One way to measure it is provided: :func:`fit_ip_uphi`, and its all-cycle
counterpart :func:`fit_ip_uphi_all`, which minimize the U_phi residual.

**That route assumes the source is azimuthally polarized, and there is
currently no alternative that does not.** A mask-edge estimator used to live
here -- normalized Stokes just outside the occulting mask or saturated core,
where the light is the star's own and taken to be unpolarized, which assumes
nothing about the target. It was withdrawn because it needs an annulus that
is both disk-free and bright, and on the data available no such annulus
exists: measured on AB Aur the median intensity fell 63x between r = 22-40 px
and r = 160-220 px, and the reported ipq tracked the shrinking denominator
rather than the instrument, reaching -8.6% at the largest radii. Every choice
made U_phi worse than no correction at all. See the git history if it is
wanted back.

So on a target where azimuthal polarization is the hypothesis under test,
this module has nothing to offer, and that is the honest position until the
Mueller matrix model lands.

These are a stopgap until that model, and are applied at the same point in
the chain it will occupy.

**IP must be removed in the instrument frame, before the rotation into
sky.** ``Q_sky = Q cos(theta_rot) + U sin(theta_rot)``, so subtracting
``ipq * I`` from a sky-frame Q is only correct if the IP vector is rotated
along with it. Everything here therefore operates on the output of
``double_difference``, never on a finished sky-frame cube.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class InstrumentalPolarization:
    """A measured I -> Q/U leakage, and where it came from.

    Attributes
    ----------
    ipq, ipu : float
        Leakage fractions. An unpolarized source of total intensity ``I``
        appears as ``Q = ipq * I``, ``U = ipu * I``, in the **instrument**
        frame.
    method : str
        How it was measured. Produced by this package:

        ``"uphi_min"``
            Nelder-Mead minimization of the U_phi scatter
            (:func:`fit_ip_uphi`, :func:`fit_ip_uphi_all`, and
            ``mueller.fit_empirical_cycle_correction``).
        The field is free-form, so an experimental route may use another
        label; nothing validates it.
    scope : str
        What it was measured over, and therefore how it is applied:
        ``"cycle"`` (one per HWP cycle), ``"sequence_joint"`` (one value
        fitted or measured over the whole sequence at once) or
        ``"sequence_mean"`` (per-cycle values averaged, by
        :func:`mean_ip`).
    diagnostics : dict
        Whatever the measuring routine wants to record — pixel counts,
        before/after residuals, radii used. Carried into the FITS
        provenance so a number can be traced back to how it was obtained.
    """

    ipq: float
    ipu: float
    method: str
    scope: str = "cycle"
    diagnostics: dict = field(default_factory=dict)

    @property
    def magnitude(self):
        """Total leakage fraction, ``hypot(ipq, ipu)``."""
        return float(np.hypot(self.ipq, self.ipu))

    @property
    def angle(self):
        """Position angle of the leakage [deg, 0-180), ``0.5*atan2(u, q)``.

        Useful as a sanity check: a stable instrument should give a
        repeatable angle across epochs even when the magnitude drifts.
        """
        return float(0.5 * np.degrees(np.arctan2(self.ipu, self.ipq)) % 180.0)

    def describe(self):
        """One-line summary for logs and FITS provenance."""
        return (f"ipq={self.ipq:+.5f} ipu={self.ipu:+.5f} "
                f"({100 * self.magnitude:.3f}% at {self.angle:.1f}deg) "
                f"method={self.method} scope={self.scope}")


def subtract_ip(Q, U, I, ip, I_u=None):
    """Remove an I -> Q/U leakage from instrument-frame Stokes planes.

    This is the only place the correction itself is written down; every
    method in this module funnels through it.

    Parameters
    ----------
    Q, U, I : ndarray
        Instrument-frame Stokes planes, **before** ``rotate_qu``.
    ip : InstrumentalPolarization or None
        The leakage to remove. ``None`` returns the inputs unchanged, so
        callers can pass an optional correction straight through.
    I_u : ndarray, optional
        Intensity to scale the U correction by, when it differs from ``I``.
        Q and U are built from different exposures, so each HWP pair has its
        own total intensity; using the matched one is slightly more faithful
        than the four-angle mean. Defaults to ``I``.

    Returns
    -------
    (Q, U) : tuple of ndarray
        Corrected planes. ``I`` is unchanged and not returned.

    Notes
    -----
    Applying this to a sky-frame cube is a bug, not a shortcut: the leakage
    is fixed in the instrument, so once Q/U have been rotated by
    ``theta_rot`` the correction would have to be rotated too.
    """
    if ip is None:
        return Q, U
    return Q - ip.ipq * I, U - ip.ipu * (I if I_u is None else I_u)


def _uphi_fit_terms(instrument, cycle, fast_axis_offset, mask_radius,
                    crop_size, critical_angles, atol, register_method,
                    derotate):
    """Everything one cycle contributes to a U_phi-minimizing IP fit.

    Returns ``(Q0, U0, Iq, Iu, eff_rot, annulus)``. Shared by the per-cycle
    and all-cycle fits so the two cannot drift apart in what they optimize.
    """
    from utils.angles import mean_angle
    from utils.imutils import make_circle_mask

    from .mueller import _registered_stacks
    from .stokes import CRITICAL_ANGLES, single_difference

    critical_angles = critical_angles or CRITICAL_ANGLES
    stacks = _registered_stacks(instrument, cycle, critical_angles, atol,
                                register_method, crop_size)
    base_rot = float(mean_angle(
        [instrument.qu_rotation_angle(f, fast_axis_offset) for f in cycle]))
    north = (float(mean_angle([instrument.north_angle(f) for f in cycle]))
             if derotate else 0.0)
    eff_rot = base_rot + 2.0 * north

    shape = stacks[0][0].shape
    annulus = (make_circle_mask(shape, min(shape) // 2 - 1)
               & ~make_circle_mask(shape, mask_radius))

    diffs, sums = zip(*(single_difference(s) for s in stacks))
    Q0 = 0.5 * (diffs[0] - diffs[1])
    U0 = 0.5 * (diffs[2] - diffs[3])
    Iq = 0.5 * (sums[0] + sums[1])
    Iu = 0.5 * (sums[2] + sums[3])
    return Q0, U0, Iq, Iu, eff_rot, annulus


def _uphi_residual(terms, ipq, ipu):
    """U_phi in the annulus for one cycle's terms at a trial (ipq, ipu)."""
    from .stokes import radial_stokes, rotate_qu

    Q0, U0, Iq, Iu, eff_rot, annulus = terms
    q, u = rotate_qu(Q0 - ipq * Iq, U0 - ipu * Iu, eff_rot)
    _, u_phi = radial_stokes(q, u)
    return u_phi[annulus]


def fit_ip_uphi_all(instrument, cycles, fast_axis_offset, mask_radius=20,
                    crop_size=400, critical_angles=None, atol=1.0,
                    register_method="smooth_peak", derotate=True):
    """Fit a single ipq/ipu across *every* cycle at once.

    The all-cycle counterpart to :func:`fit_ip_uphi`, which fits each cycle
    separately. Both minimize the same U_phi scatter; this one asks for the
    one leakage pair that best explains all the cycles together, pooling
    their annulus pixels into a single objective.

    Prefer this when the leakage is a property of the optics rather than of
    the night: two free parameters against every cycle's data, instead of two
    per cycle. That economy is the argument for it, not a large gain in the
    result -- measured on AB Aur (23 cycles, theta_off = -13.1) the two routes
    agree closely::

        no IP correction                    U_phi std  5.699
        one ipq/ipu for all cycles          U_phi std  5.386   ipq -0.761%
        per-cycle fits, then mean_ip        U_phi std  5.365   ipq -0.765%

    Both remove real leakage and land within 0.005% of each other in ipq, and
    averaging per-cycle fits was, if anything, a shade better. So choose on
    what you believe about the instrument, not on these numbers.

    Prefer :func:`fit_ip_uphi` per cycle when the leakage really does vary
    within the sequence -- a rotator that moved, or a configuration change
    partway through -- and :func:`mean_ip` to summarise those fits, which also
    gives an error bar from the scatter.

    Note that per-cycle fitting is only harmless because this fit has two
    parameters. ``mueller.fit_empirical_cycle_correction`` fits sixteen per
    cycle, and there the same per-cycle freedom does real damage: on the same
    data it lowers each cycle's own U_phi but raises the median's from 5.70 to
    9.07 and drops the Q_phi correlation from 0.973 to 0.926, because the
    cycles' corrections disagree and that disagreement does not cancel.

    Parameters
    ----------
    instrument : PolarimetryData
        Supplies the rotation model and beam geometry.
    cycles : list of list of Frame
        Every complete HWP cycle, as from ``match_modulator_cycles``.
    fast_axis_offset : float
        Held fixed, as for :func:`fit_ip_uphi`.
    mask_radius, crop_size, critical_angles, atol, register_method, derotate
        As for :func:`fit_ip_uphi`.

    Returns
    -------
    InstrumentalPolarization
        ``scope="sequence_joint"`` -- one value fitted jointly over the whole
        sequence, as against :func:`mean_ip`'s ``"sequence_mean"``, which
        averages separate per-cycle fits. Same extent, different provenance,
        so the names say which. The pooled U_phi scatter before and after is
        in ``diagnostics``, along with the number of cycles used.

    Warnings
    --------
    **Assumes the source is azimuthally polarized**, exactly as
    :func:`fit_ip_uphi` does. Do not use it where that is the hypothesis
    under test.
    """
    import numpy as np
    from scipy.optimize import minimize

    cycles = list(cycles)
    if not cycles:
        raise ValueError("fit_ip_uphi_all needs at least one cycle")

    terms = [_uphi_fit_terms(instrument, c, fast_axis_offset, mask_radius,
                             crop_size, critical_angles, atol,
                             register_method, derotate)
             for c in cycles]

    def objective(x):
        """Pooled U_phi scatter over every cycle; lower is better."""
        pooled = np.concatenate([_uphi_residual(t, x[0], x[1]) for t in terms])
        return float(np.nanstd(pooled))

    initial = objective(np.zeros(2))
    result = minimize(objective, np.zeros(2), method="Nelder-Mead")
    ipq, ipu = float(result.x[0]), float(result.x[1])

    ip = InstrumentalPolarization(
        ipq, ipu, method="uphi_min", scope="sequence_joint",
        diagnostics={"uphi_std_initial": initial,
                     "uphi_std_final": float(result.fun),
                     "n_cycles": len(cycles),
                     "mask_radius": float(mask_radius),
                     "crop_size": int(crop_size) if crop_size else None})
    log.info("All-cycle U_phi-minimization IP over %d cycles: %s | "
             "pooled U_phi std %.4g -> %.4g",
             len(cycles), ip.describe(), initial, result.fun)
    return ip


def fit_ip_uphi(instrument, cycle, fast_axis_offset, mask_radius=20,
                crop_size=400, critical_angles=None, atol=1.0,
                register_method="smooth_peak", derotate=True):
    """Fit ipq/ipu by minimizing the U_phi residual over one HWP cycle.

    For a source polarized azimuthally about its centre — a protoplanetary
    disk in scattered light — all of the signal belongs in Q_phi and U_phi
    is noise. Any leakage tilts flux into U_phi, so the (ipq, ipu) that
    minimize ``nanstd(U_phi)`` are an estimate of the leakage.

    Two parameters, not the sixteen of
    ``mueller.fit_empirical_cycle_correction``: that routine also fits
    residual beam and frame shifts, and is the right tool when those are
    the problem. This one is much faster and does not risk absorbing real
    structure into shift parameters.

    This fits *one cycle*. To fit a single leakage pair across every cycle at
    once, use :func:`fit_ip_uphi_all` -- usually the better choice, since the
    leakage is normally a property of the optics rather than of the cycle.
    Fitting per cycle and averaging with :func:`mean_ip` is a third thing
    again, and not equivalent: see the note in :func:`fit_ip_uphi_all`.

    Parameters
    ----------
    fast_axis_offset : float
        Held fixed. Fitting it here as well is possible but degenerate with
        the IP terms — see :func:`polarimetry.fast_axis.fit_fast_axis_butterfly`,
        which does the joint fit deliberately.
    mask_radius : float
        Pixels within this radius of the centre are excluded, covering the
        saturated or occulted core.
    crop_size : int
        Work on a central cutout this size. The optimizer evaluates the
        objective repeatedly, so a tight crop matters.
    derotate : bool
        Whether the science reduction derotates to north-up. The objective
        needs no image rotation either way: rotating pixels by an angle
        shifts every azimuth by that angle, and only ``2*phi + theta``
        enters U_phi, so derotation folds into the Q/U rotation angle as
        ``eff_rot = base_rot + 2*north``.

    Returns
    -------
    InstrumentalPolarization
        With ``uphi_std_initial`` / ``uphi_std_final`` in ``diagnostics``.
        A final value close to the initial one means the fit found nothing
        to remove, which is information, not success.

    Warnings
    --------
    **Assumes the source is azimuthally polarized.** Do not use it on a
    target where that is the hypothesis under test — it will happily rotate
    a genuine U_phi signal into Q_phi and report a confident answer. There is
    currently no IP route here that avoids the assumption, so on an AGN, a
    merger or a star field the answer is to measure the leakage some other
    way and pass it in, not to reach for another function in this module.
    """
    from scipy.optimize import minimize

    terms = _uphi_fit_terms(instrument, cycle, fast_axis_offset, mask_radius,
                            crop_size, critical_angles, atol, register_method,
                            derotate)

    def objective(x):
        """U_phi scatter for a trial ``(ipq, ipu)``; lower is better."""
        return float(np.nanstd(_uphi_residual(terms, x[0], x[1])))

    initial = objective(np.zeros(2))
    result = minimize(objective, np.zeros(2), method="Nelder-Mead")
    ipq, ipu = float(result.x[0]), float(result.x[1])

    ip = InstrumentalPolarization(
        ipq, ipu, method="uphi_min", scope="cycle",
        diagnostics={"uphi_std_initial": initial,
                     "uphi_std_final": float(result.fun),
                     "mask_radius": float(mask_radius),
                     "crop_size": (int(crop_size) if crop_size
                                   else None)})
    log.info("U_phi-minimization IP: %s | U_phi std %.4g -> %.4g",
             ip.describe(), initial, result.fun)
    return ip


def mean_ip(ips, method=None):
    """Average a list of per-cycle measurements into one sequence value.

        The scatter across cycles is the honest error bar on the mean, so it is
        recorded in ``diagnostics`` as ``ipq_err`` / ``ipu_err`` (standard error
        of the mean). If that scatter is comparable to the values themselves,
        the leakage is not well measured and the correction should be treated
        with suspicion rather than applied silently.

    Parameters
    ----------
    ips : list of InstrumentalPolarization
        Per-cycle measurements.
    method : str, optional
        Method recorded on the result; defaults to that of the first input.

    Returns
    -------
    InstrumentalPolarization
        The mean, scope ``"sequence"``, with ``ipq_err`` and ``ipu_err`` in
        ``diagnostics``.

    Raises
    ------
    ValueError
        If ``ips`` is empty.
    """
    if not ips:
        raise ValueError("no measurements to average")
    q = np.array([ip.ipq for ip in ips], dtype=float)
    u = np.array([ip.ipu for ip in ips], dtype=float)
    n = max(len(q), 1)
    return InstrumentalPolarization(
        float(q.mean()), float(u.mean()),
        method=method or ips[0].method, scope="sequence_mean",
        diagnostics={"n": len(ips),
                     "ipq_err": float(q.std() / np.sqrt(n)),
                     "ipu_err": float(u.std() / np.sqrt(n))})

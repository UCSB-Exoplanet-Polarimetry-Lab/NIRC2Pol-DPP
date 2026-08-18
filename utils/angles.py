"""Circular angle arithmetic.

Modulator and rotator angles wrap, and treating them as plain numbers has
broken this pipeline twice: a PCUPR reading of ``-0.002`` for the "0 deg"
HWP position matched no critical angle when compared naively, and the
arithmetic mean of a cycle straddling a wrap landed 180 deg from the truth.
Everything here compares and averages angles circularly.
"""

def angles_match(a, b, atol=1.0, period=180.0):
    """Compare two angles circularly.

    Parameters
    ----------
    a, b : float
        Angles in degrees.
    atol : float, optional
        Tolerance in degrees.
    period : float, optional
        Wrap period. The default 180 suits polarization angles and HWP
        positions, where a half turn is the same state; pass 360 for
        pointing angles.

    Returns
    -------
    bool
        True if the angles agree within ``atol`` modulo ``period``.

    Notes
    -----
    The circular comparison is not decoration. PCUPR reads ``-0.002`` for
    the nominal 0 deg HWP position; taken modulo 180 that becomes 179.998,
    and a direct comparison matched none of the critical angles, producing
    zero usable cycles.
    """
    half = period / 2.0
    return abs((a - b + half) % period - half) <= atol


def is_critical_angle(angle, critical_angles, atol=1.0, period=180.0):
    """Is an angle one of the modulator's critical angles?

    Parameters
    ----------
    angle : float
        Angle in degrees, as read from the header.
    critical_angles : iterable of float
        The instrument's critical angles, e.g. ``(0, 45, 22.5, 67.5)``.
    atol : float, optional
        Tolerance in degrees.
    period : float, optional
        Wrap period, as for :func:`angles_match`.

    Returns
    -------
    bool
        True if ``angle`` matches any of them circularly.
    """
    return any(angles_match(angle, c, atol, period) for c in critical_angles)


def mean_angle(angles, period=360.0):
    """Circular mean of a set of angles.

    Parameters
    ----------
    angles : array_like
        Angles in degrees.
    period : float, optional
        Wrap period; 360 for pointing angles, 180 for polarization angles.

    Returns
    -------
    float
        The circular mean, returned in the branch nearest ``angles[0]`` so
        it stays continuous with the input rather than being forced into
        ``[0, period)``. NaN for an empty input.

    Notes
    -----
    A plain ``np.mean`` is wrong whenever the set straddles a wrap. The
    AB Aur 2025-12-08 UT data has a cycle whose PARANG runs ``-112.75`` and
    ``246.94`` — the same direction, 360 deg apart. Unwrapped the second is
    ``-113.06``, so the true mean is ``-112.905``, while the arithmetic mean
    of the raw values is ``+67.095``.
    """
    import numpy as np

    angles = np.asarray(angles, dtype=float)
    if angles.size == 0:
        return float("nan")
    scale = 2.0 * np.pi / period
    mean = np.angle(np.mean(np.exp(1j * angles * scale))) / scale
    ref = float(angles.flat[0])
    return float(ref + (mean - ref + period / 2.0) % period - period / 2.0)

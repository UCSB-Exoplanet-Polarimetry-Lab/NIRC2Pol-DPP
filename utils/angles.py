"""Circular angle arithmetic.

Modulator and rotator angles wrap, and treating them as plain numbers has
broken this pipeline twice: a PCUPR reading of ``-0.002`` for the "0 deg"
HWP position matched no critical angle when compared naively, and the
arithmetic mean of a cycle straddling a wrap landed 180 deg from the truth.
Everything here compares and averages angles circularly. It also holds the
plain spherical-geometry helpers -- sexagesimal parsing, small-angle
separation, parallactic angle
"""

import numpy as np


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
    """
    import numpy as np

    angles = np.asarray(angles, dtype=float)
    if angles.size == 0:
        return float("nan")
    scale = 2.0 * np.pi / period
    mean = np.angle(np.mean(np.exp(1j * angles * scale))) / scale
    ref = float(angles.flat[0])
    return float(ref + (mean - ref + period / 2.0) % period - period / 2.0)


def sexagesimal_to_degrees(value):
    """Sexagesimal string to a decimal number.

    Parameters
    ----------
    value : str or float
        ``'HH:MM:SS.S'`` or ``'+DD:MM:SS'``. A number passes through, so a
        header that already holds decimals needs no special case.

    Returns
    -------
    float
        The value in whatever unit the leading field was -- hours in, hours
        out; degrees in, degrees out.

    Notes
    -----
    The sign is taken from the leading field, so ``'-00:30:00'`` is -0.5 and
    not +0.5: negative declinations between 0 and -1 would otherwise come out
    with the wrong sign, since ``float('-00') == 0``.
    """
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    sign = -1.0 if parts[0].strip().startswith("-") else 1.0
    numbers = [abs(float(p)) for p in parts]
    return sign * sum(n / 60.0**i for i, n in enumerate(numbers))


def small_angle_distance(a, b):
    """Angular distance between two ``(ra, dec)`` pairs in degrees.

    Parameters
    ----------
    a, b : tuple of float
        ``(ra, dec)`` pairs in degrees.

    Returns
    -------
    float
        Separation in degrees, under the small-angle approximation: the RA
        difference is scaled by ``cos(dec)`` but no spherical law of cosines
        is used, so it degrades for widely separated points and near the
        poles. Fine for the arcsecond-scale comparisons it is used for.
    """
    (ra_a, dec_a), (ra_b, dec_b) = a, b
    return np.sqrt(((ra_a - ra_b) * np.cos(np.deg2rad(dec_a))) ** 2
                   + (dec_a - dec_b) ** 2)


def par_angle(hour_angle, dec, lat):
    """Parallactic angle in degrees. Source: pyKLIP.

    Parameters
    ----------
    hour_angle : float
        Hour angle in *hours*, not degrees -- it is multiplied by 15 here.
    dec : float
        Declination in degrees.
    lat : float
        Observatory latitude in degrees. Passed in rather than read from a
        constant, which is what keeps this function site-agnostic.

    Returns
    -------
    float
        Parallactic angle in degrees.
    """
    ha_rad = np.deg2rad(hour_angle * 15.0)
    dec_rad = np.deg2rad(dec)
    lat_rad = np.deg2rad(lat)

    parallang = -np.arctan2(
        -np.sin(ha_rad),
        np.cos(dec_rad) * np.tan(lat_rad) - np.sin(dec_rad) * np.cos(ha_rad),
    )
    return np.rad2deg(parallang)

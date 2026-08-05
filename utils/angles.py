"""Angle comparison helpers.

Modulator (HWP) angles need circular comparison: a nominal 0 deg position
often reads back as -0.002, and angles are equivalent modulo 180 deg for a
half-wave plate.
"""

from __future__ import annotations


def angles_match(a, b, atol=1.0, period=180.0):
    """True if angles ``a`` and ``b`` agree within ``atol``, compared
    circularly modulo ``period`` (so -0.002 matches 0, and 179.9 matches 0).
    """
    half = period / 2.0
    return abs((a - b + half) % period - half) <= atol


def is_critical_angle(angle, critical_angles, atol=1.0, period=180.0):
    """True if ``angle`` matches any of the modulator's critical angles."""
    return any(angles_match(angle, c, atol, period) for c in critical_angles)


def mean_angle(angles, period=360.0):
    """Circular mean of ``angles`` [deg], modulo ``period``.

    A plain ``np.mean`` is wrong whenever a set of angles straddles a wrap:
    the AB Aur 2025-12-07 sequence has a cycle whose PARANG runs -112.75,
    246.94 (the same direction, 360 deg apart), and the arithmetic mean of
    that cycle lands ~90 deg from the truth. The result is returned in the
    branch nearest the first angle, so it stays continuous with the input
    rather than being forced into [0, period).
    """
    import numpy as np

    angles = np.asarray(angles, dtype=float)
    if angles.size == 0:
        return float("nan")
    scale = 2.0 * np.pi / period
    mean = np.angle(np.mean(np.exp(1j * angles * scale))) / scale
    ref = float(angles.flat[0])
    return float(ref + (mean - ref + period / 2.0) % period - period / 2.0)

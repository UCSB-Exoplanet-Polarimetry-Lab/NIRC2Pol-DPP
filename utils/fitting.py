"""2D Gaussian fitting for PSF centering. Translated from AIR.jl fitting.jl."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .imutils import translate


def gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y, offset):
    return amplitude * np.exp(
        -((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2))
    ) + offset


def fit_2d_gaussian(data, initial_guess, fixed_sigma=None):
    """Least-squares fit of a 2D Gaussian to an image.

    ``initial_guess`` is ``[amplitude, x0, y0, sigma_x, sigma_y, offset]``,
    or ``[amplitude, x0, y0, offset]`` when ``fixed_sigma`` is given.
    Coordinates are 0-based pixel indices (x = column, y = row).

    Returns the fitted parameter array in the same order as the guess.
    """
    data = np.asarray(data, dtype=float)
    rows, cols = data.shape
    yy, xx = np.mgrid[:rows, :cols]

    if fixed_sigma is not None:
        def model(p):
            amp, x0, y0, offset = p
            return gaussian_2d(xx, yy, amp, x0, y0, fixed_sigma, fixed_sigma, offset)
    else:
        def model(p):
            amp, x0, y0, sx, sy, offset = p
            return gaussian_2d(xx, yy, amp, x0, y0, sx, sy, offset)

    result = least_squares(lambda p: (data - model(p)).ravel(),
                           np.asarray(initial_guess, dtype=float))
    return result.x


def fit_and_translate(data, initial_guess, fixed_sigma=None, fill=0.0):
    """Fit a 2D Gaussian and translate the image so the fitted peak lands on
    the image center. Returns ``(translated, cx, cy)`` with the fitted peak
    position in the original image."""
    params = fit_2d_gaussian(data, initial_guess, fixed_sigma=fixed_sigma)
    cx, cy = params[1], params[2]

    h, w = data.shape
    translated = translate(data, h / 2 - cy, w / 2 - cx, fill=fill)
    return translated, cx, cy

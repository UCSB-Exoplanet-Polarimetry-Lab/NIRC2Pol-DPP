"""Generic image utilities: cropping, shifting, rotating, masks.

Translated from AIR.jl's utils.jl and angles.jl (the rotation helper only —
the NIRC2/Keck north-angle calculation lives in instruments/nirc2.py).

Conventions: arrays are indexed ``data[y, x]`` and coordinates are given as
``(cy, cx)`` pairs, 0-based.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def image_is_larger(a, b):
    """True if array ``a`` is strictly larger than ``b`` in every dimension."""
    return all(sa > sb for sa, sb in zip(np.shape(a), np.shape(b)))


def crop(img, crop_size, center=None):
    """Crop an image to ``crop_size = (height, width)`` around ``center =
    (cy, cx)`` (defaults to the image center).

    Returns ``(cropped, offset_y, offset_x)``. Add the offsets to go from
    cropped to original coordinates; subtract to go the other way.
    """
    img = np.asarray(img)
    h, w = img.shape
    crop_h, crop_w = crop_size

    if (h, w) == (crop_h, crop_w):
        return img, 0, 0

    if center is None:
        cy, cx = h / 2, w / 2
    else:
        cy, cx = center

    start_row = int(round(cy - crop_h / 2))
    start_col = int(round(cx - crop_w / 2))
    end_row = start_row + crop_h
    end_col = start_col + crop_w

    if start_row < 0 or start_col < 0 or end_row > h or end_col > w:
        raise ValueError(
            f"Crop out of bounds: img {img.shape}, center ({cy}, {cx}), "
            f"crop_size {crop_size}"
        )

    return img[start_row:end_row, start_col:end_col], start_row, start_col


def translate(img, dy, dx, fill=0.0, order=1):
    """Shift image content by ``(dy, dx)``: a feature at (y, x) moves to
    (y + dy, x + dx). Bilinear interpolation by default."""
    return ndimage.shift(
        np.asarray(img, dtype=float), (dy, dx), order=order, cval=fill,
        prefilter=(order > 1),
    )


def rotate_image_center(img, angle_degrees, fill=np.nan, center=None,
                        flipx=False):
    """Rotate an image about its center using ``pyklip.klip.rotate``.

    Sign convention (kept from this module's original scipy backend):
    positive angles rotate image features clockwise when displayed with
    ``origin='lower'`` — i.e. this calls ``pyklip.klip.rotate(img,
    -angle_degrees, center)``, since pyklip's positive angle is
    counterclockwise.

    Pixels rotated in from outside the image become NaN; pass ``fill`` to
    replace them (default keeps NaN, which downstream nanmedian collapses
    handle). ``center = (cy, cx)`` defaults to the image center; ``flipx``
    is passed through to pyklip for left-handed coordinate systems.
    """
    from .frame import Frame

    if isinstance(img, Frame):
        rotated = rotate_image_center(img.data, angle_degrees, fill=fill,
                                      center=center, flipx=flipx)
        return Frame(rotated, img.header.copy())

    from pyklip.klip import rotate as pyklip_rotate

    img = np.asarray(img, dtype=float)
    ny, nx = img.shape
    if center is None:
        pyklip_center = [(nx - 1) / 2, (ny - 1) / 2]  # pyklip wants [x, y]
    else:
        cy, cx = center
        pyklip_center = [cx, cy]

    rotated = pyklip_rotate(img, -angle_degrees, pyklip_center, flipx=flipx)

    if fill is not None and not np.isnan(fill):
        rotated = np.where(np.isnan(rotated), fill, rotated)
    return rotated


def make_circle_mask(shape, radius, center=None):
    """Boolean mask, True inside a circle of ``radius`` pixels around
    ``center = (cy, cx)`` (defaults to image center)."""
    sy, sx = shape
    if center is None:
        cy, cx = (sy - 1) / 2, (sx - 1) / 2
    else:
        cy, cx = center
    yy, xx = np.ogrid[:sy, :sx]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def make_annulus_mask(shape, inner_radius, outer_radius, center=None):
    """Boolean mask, True between ``inner_radius`` and ``outer_radius``."""
    sy, sx = shape
    if center is None:
        cy, cx = (sy - 1) / 2, (sx - 1) / 2
    else:
        cy, cx = center
    yy, xx = np.ogrid[:sy, :sx]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (r2 >= inner_radius**2) & (r2 <= outer_radius**2)


def make_sigma_clip_mask(data, n_sigma=9.0):
    """Mask pixels brighter than ``median + n_sigma * std`` (hot pixels /
    cosmic rays). Only the upper tail is clipped, matching AIR.jl."""
    data = np.asarray(data)
    if data.size == 0:
        return np.zeros_like(data, dtype=bool)
    threshold = np.median(data) + n_sigma * np.std(data)
    return data > threshold


def plus_mask(mask, radius=1):
    """Grow a boolean mask into a "+" shape: each True pixel spreads
    ``radius`` steps up/down/left/right. Used for saturated pixels, which
    bleed along detector rows/columns."""
    cross = ndimage.generate_binary_structure(2, 1)  # 4-connected "+" kernel
    return ndimage.binary_dilation(mask, structure=cross, iterations=radius)


def argquantile(img, quantile):
    """Index ``(iy, ix)`` of the pixel whose value is closest to the given
    quantile of the image. A robust way to locate the peak while ignoring
    isolated hot pixels."""
    img = np.asarray(img)
    val = np.nanquantile(img, quantile)
    flat_idx = np.nanargmin(np.abs(img - val))
    return np.unravel_index(flat_idx, img.shape)


def measure_background(data, mask_radius=200, peak_quantile=0.9999):
    """Median background level, measured outside a circle of ``mask_radius``
    pixels around the (quantile-located) peak."""
    data = np.asarray(data)
    cy, cx = argquantile(data, peak_quantile)
    outside = ~make_circle_mask(data.shape, mask_radius, center=(cy, cx))
    background_pixels = data[outside]
    if background_pixels.size == 0:
        return 0.0
    return float(np.median(background_pixels))


def key_with_most_items(d):
    """Key of the dict whose value has the largest ``len``."""
    if not d:
        raise ValueError("dictionary is empty")
    return max(d, key=lambda k: len(d[k]))

"""Generic image utilities: cropping, shifting, rotating, masks.

Translated from AIR.jl's utils.jl and angles.jl (the rotation helper only —
the NIRC2/Keck north-angle calculation lives in nirc2pol/instruments/nirc2.py).

Conventions: arrays are indexed ``data[y, x]`` and coordinates are given as
``(cy, cx)`` pairs, 0-based.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def image_is_larger(a, b):
    """True if array ``a`` is strictly larger than ``b`` in every dimension.

    Parameters
    ----------
    a, b : array_like
        Arrays, or anything ``np.shape`` accepts. Only shapes are compared.

    Returns
    -------
    bool
        True when every axis of ``a`` exceeds the matching axis of ``b``.
        Equal sizes count as *not* larger, which is what callers want when
        deciding whether a calibration frame can be trimmed to fit.
    """
    return all(sa > sb for sa, sb in zip(np.shape(a), np.shape(b)))


def crop(img, crop_size, center=None):
    """Crop an image to a given size about a given centre.

    Parameters
    ----------
    img : array_like
        2D image, indexed ``[y, x]``.
    crop_size : tuple of int
        ``(height, width)`` of the output.
    center : tuple of float, optional
        ``(cy, cx)`` about which to crop, 0-based. Defaults to the image
        centre.

    Returns
    -------
    cropped : ndarray
        View of ``img`` with shape ``crop_size``.
    offset_y, offset_x : int
        Position of the crop's origin in the original array. Add them to go
        from cropped to original coordinates, subtract to go the other way.

    Raises
    ------
    ValueError
        If the requested region falls outside the image
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
    """Shift image content by a (possibly fractional) offset.

    Parameters
    ----------
    img : array_like
        2D image.
    dy, dx : float
        Shift in pixels. A feature at ``(y, x)`` moves to ``(y + dy, x + dx)``.
    fill : float, optional
        Value for pixels shifted in from outside the frame.
    order : int, optional
        Spline order for the interpolation; 1 (bilinear) by default.
        Higher orders ring around sharp edges, which matters on a saturated
        or occulted core.

    Returns
    -------
    ndarray
        Shifted image, always float.
    """
    return ndimage.shift(
        np.asarray(img, dtype=float), (dy, dx), order=order, cval=fill,
        prefilter=(order > 1),
    )


def rotate_image_center(img, angle_degrees, fill=np.nan, center=None,
                        flipx=False):
    """Rotate an image about a centre, using pyklip's implementation.

    Parameters
    ----------
    img : array_like or Frame
        2D image. A :class:`nirc2pol.utils.frame.Frame` is rotated and returned as a
        Frame with its header carried across.
    angle_degrees : float
        Rotation angle. **Positive rotates features clockwise** when
        displayed with ``origin='lower'``.
    fill : float, optional
        Replacement for pixels rotated in from outside. The default NaN is
        deliberate: downstream ``nanmedian`` combines then ignore them,
        whereas zeros would drag a median toward zero at the edges.
    center : tuple of float, optional
        ``(cy, cx)`` to rotate about; defaults to the image centre.
    flipx : bool, optional
        Passed through to pyklip for left-handed coordinate systems.

    Returns
    -------
    ndarray or Frame
        Rotated image, matching the input type.
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
    """Boolean mask of a filled circle.

    Parameters
    ----------
    shape : tuple of int
        ``(ny, nx)`` of the mask.
    radius : float
        Circle radius in pixels. The boundary is inclusive.
    center : tuple of float, optional
        ``(cy, cx)``; defaults to the image centre, ``((ny-1)/2, (nx-1)/2)``.

    Returns
    -------
    ndarray of bool
        True inside the circle.
    """
    sy, sx = shape
    if center is None:
        cy, cx = (sy - 1) / 2, (sx - 1) / 2
    else:
        cy, cx = center
    yy, xx = np.ogrid[:sy, :sx]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def make_annulus_mask(shape, inner_radius, outer_radius, center=None):
    """Boolean mask of an annulus.

    Parameters
    ----------
    shape : tuple of int
        ``(ny, nx)`` of the mask.
    inner_radius, outer_radius : float
        Radii in pixels. Both boundaries are inclusive.
    center : tuple of float, optional
        ``(cy, cx)``; defaults to the image centre.

    Returns
    -------
    ndarray of bool
        True between the two radii.

    Notes
    -----
    Used for background estimation and for the mask-edge instrumental
    polarization measurement, where the annulus must hold starlight and no
    disk signal.
    """
    sy, sx = shape
    if center is None:
        cy, cx = (sy - 1) / 2, (sx - 1) / 2
    else:
        cy, cx = center
    yy, xx = np.ogrid[:sy, :sx]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (r2 >= inner_radius**2) & (r2 <= outer_radius**2)


def make_sigma_clip_mask(data, n_sigma=9.0):
    """Mask the bright tail of an image: hot pixels and cosmic rays.

    Parameters
    ----------
    data : array_like
        Image to clip.
    n_sigma : float, optional
        Threshold in standard deviations above the median.

    Returns
    -------
    ndarray of bool
        True where ``data > median + n_sigma * std``. An empty input gives
        an empty mask rather than raising.

    Notes
    -----
    Only the *upper* tail is clipped, matching AIR.jl. Dead and cold pixels
    are the static bad-pixel mask's job, not this one.
    """
    data = np.asarray(data)
    if data.size == 0:
        return np.zeros_like(data, dtype=bool)
    threshold = np.median(data) + n_sigma * np.std(data)
    return data > threshold


def plus_mask(mask, radius=1):
    """Grow a boolean mask into a "+" shape.

    Parameters
    ----------
    mask : ndarray of bool
        Mask to grow.
    radius : int, optional
        Steps to spread up, down, left and right. **Must be >= 1.**

    Returns
    -------
    ndarray of bool
        The dilated mask.

    Warnings
    --------
    ``radius`` is passed straight to ``ndimage.binary_dilation`` as
    ``iterations``, where ``0`` does not mean "no dilation" but "repeat
    until nothing changes" — which fills the entire frame. For a
    saturated-pixel mask that would blank the whole detector.
    """
    cross = ndimage.generate_binary_structure(2, 1)  # 4-connected "+" kernel
    return ndimage.binary_dilation(mask, structure=cross, iterations=radius)


def argquantile(img, quantile):
    """Locate the pixel closest to a given quantile of the image.

    Parameters
    ----------
    img : array_like
        Image to search. NaNs are ignored.
    quantile : float
        Quantile in [0, 1]. Values just below 1 find the peak while
        stepping over isolated hot pixels, which is the usual reason to
        prefer this over ``argmax``.

    Returns
    -------
    tuple of int
        ``(iy, ix)`` of the matching pixel.

    Raises
    ------
    ValueError
        If every pixel is NaN, propagated from ``np.nanargmin``.
    """
    img = np.asarray(img)
    val = np.nanquantile(img, quantile)
    flat_idx = np.nanargmin(np.abs(img - val))
    return np.unravel_index(flat_idx, img.shape)




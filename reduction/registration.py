"""Image registration: locate the star and center/derotate frames.

There are several reasonable ways to find the star, so the method is
swappable (SPIE Sec. 3.3): pass ``method=`` one of the built-in names below
or any callable ``f(data) -> (cy, cx)`` to drop in your own algorithm.

Built-in methods:

- ``"smooth_peak"`` (default) — peak of the Gaussian-smoothed image,
  refined to subpixel with a Gaussian fit in a small window around it.
  Fully automatic: no starting guess, crop region, or tuning required.
- ``"quantile_peak"`` — pixel at a high quantile of the flux
  distribution; robust to isolated hot pixels.
- ``"max"`` — brightest pixel.
- ``"min"`` — darkest pixel near the core, for saturated PSFs (searched
  within ``search_radius`` of the quantile peak).
- ``"gaussian"`` — 2D Gaussian fit seeded from the quantile peak.
- ``"centroid"`` — flux-weighted centroid of everything above a threshold.
  For extended sources; still pulled by bright surface features.
- ``"silhouette"`` — geometric center of the thresholded source region,
  ignoring brightness variations within it. Suits a resolved body (planet,
  moon) whose disk carries hotspots or albedo features.
- ``"symmetry"`` — center of symmetry, found by cross-correlating the image
  with its own 180-degree rotation. Uses the whole source, needs no
  template, threshold, or starting guess; the natural choice for a round
  resolved body.
- ``"wings"`` — align on the PSF wings, masking out the core. The method
  for coronagraphic data: the occulted core carries no reliable position
  information, but the extended wings stay symmetric about the true star.
- ``"crosscorr"`` — cross-correlate against a ``template=`` image and
  return the position matching the template's center. Use when you want a
  whole sequence aligned to one reference (see
  :func:`register_frames_to_template`).

Works on 2D frames and on ``(2, ny, nx)`` beam stacks (both beams shifted
by the offset measured on their mean, so the beams stay registered to each
other).
"""

from __future__ import annotations

import logging

import numpy as np

from utils.frame import Frame, framelist_to_cube
from utils.imutils import (argquantile, crop, make_circle_mask,
                           rotate_image_center, translate)

log = logging.getLogger(__name__)


def find_center_smooth(data, smooth_sigma=3.0, fit_window=15):
    """Fully automatic star finder: peak of the Gaussian-smoothed image
        (immune to hot pixels and cosmic rays), refined to subpixel with a 2D
        Gaussian fit in a ``fit_window`` box around the smoothed peak. Needs no
        starting guess or crop region.

    Parameters
    ----------
    data : array_like
        2D image to search.
    smooth_sigma : float, optional
        Gaussian smoothing width in pixels, also used as the fixed sigma of
        the refinement fit.
    fit_window : int, optional
        Side of the box, in pixels, used for the subpixel fit.

    Returns
    -------
    tuple of float
        ``(cy, cx)``, the source position in 0-based pixel coordinates.

    Notes
    -----
    Falls back to the integer smoothed peak, with a warning, if the fit fails
    or wanders outside its window. That fallback once hid a missing import for
    an entire session: every reduction silently lost subpixel precision while
    looking fine. If this warns often, investigate rather than ignore it.
    """
    from scipy.ndimage import gaussian_filter, median_filter

    # median filter first: annihilates isolated hot pixels / cosmic rays
    # that could survive Gaussian smoothing alone
    smoothed = gaussian_filter(median_filter(np.nan_to_num(data), size=3),
                               smooth_sigma)
    cy, cx = np.unravel_index(np.argmax(smoothed), smoothed.shape)

    half = fit_window // 2
    y0 = min(max(0, cy - half), data.shape[0] - fit_window)
    x0 = min(max(0, cx - half), data.shape[1] - fit_window)
    window = np.nan_to_num(data[y0:y0 + fit_window, x0:x0 + fit_window])

    background = float(np.median(window))
    try:
        params = fit_2d_gaussian(
            window,
            [window.max() - background, cx - x0, cy - y0, background],
            fixed_sigma=smooth_sigma)
        fit_cy, fit_cx = y0 + params[2], x0 + params[1]
    except Exception:
        log.warning("Subpixel Gaussian refinement failed, using smoothed "
                    "peak pixel")
        return float(cy), float(cx)

    # reject a fit that wandered out of its window; the peak is still good
    if not (y0 <= fit_cy <= y0 + fit_window and x0 <= fit_cx <= x0 + fit_window):
        log.warning("Subpixel fit left the window, using smoothed peak pixel")
        return float(cy), float(cx)
    return float(fit_cy), float(fit_cx)


def _prepare_source_image(data, smooth_sigma=2.0, clip_percentile=90.0):
    """Smooth an image and clip its bright tail, returning
        ``(prepared, source_level)``.

        Clipping matters for resolved bodies: a volcano or storm can outshine
        the disk several times over, and any measure referenced to the peak
        then describes the feature instead of the body. The clip level is a
        percentile *of the source pixels only*, so a small bright feature
        cannot set the scale. ``source_level`` is the clipped maximum, i.e. a
        robust estimate of the body's own surface brightness.

    Parameters
    ----------
    data : array_like
        2D image to search.
    smooth_sigma : float, optional
        Gaussian smoothing width in pixels.
    clip_percentile : float, optional
        Percentile *of the source pixels alone* at which to clip the bright
        tail.

    Returns
    -------
    prepared : ndarray
        Smoothed and clipped image.
    source_level : float
        The clipped maximum, i.e. a robust brightness scale.
    """
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(np.nan_to_num(data), smooth_sigma)

    # rough source region, then a robust brightness level within it
    rough = smoothed > 0.2 * np.nanmax(smoothed)
    if not rough.any():
        return smoothed, float(np.nanmax(smoothed))

    level = float(np.nanpercentile(smoothed[rough], clip_percentile))
    if level <= 0:
        level = float(np.nanmax(smoothed))
    return np.minimum(smoothed, level), level


def _threshold_mask(data, threshold_frac, smooth_sigma, clip_percentile=90.0):
    """Boolean mask of the source: pixels above ``threshold_frac`` of the
        body's own (clip-robust) brightness, largest connected region only.

    Parameters
    ----------
    data : array_like
        2D image to search.
    threshold_frac : float
        Fraction of the source level above which a pixel counts as source.
    smooth_sigma : float, optional
        Gaussian smoothing width applied before thresholding.
    clip_percentile : float, optional
        Passed to :func:`_prepare_source_image`.

    Returns
    -------
    ndarray of bool
        True on source pixels.
    """
    from scipy.ndimage import label

    prepared, level = _prepare_source_image(data, smooth_sigma,
                                            clip_percentile)
    mask = prepared > threshold_frac * level

    labels, n = label(mask)
    if n > 1:  # keep the largest blob, drop noise specks
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == sizes.argmax()
    return mask


def find_center_centroid(data, threshold_frac=0.2, smooth_sigma=2.0):
    """Flux-weighted centroid of the source region. Suits extended objects;
        bright surface features still pull the centroid toward them, so prefer
        ``symmetry`` or ``crosscorr`` for a body with strong features.

    Parameters
    ----------
    data : array_like
        2D image to search.
    threshold_frac : float, optional
        Fraction of the source level defining the source region.
    smooth_sigma : float, optional
        Gaussian smoothing width in pixels.

    Returns
    -------
    tuple of float
        ``(cy, cx)``, the source position in 0-based pixel coordinates.
    """
    mask = _threshold_mask(data, threshold_frac, smooth_sigma)
    prepared, _ = _prepare_source_image(data, smooth_sigma)
    weights = np.where(mask, prepared, 0.0)
    total = weights.sum()
    if total <= 0:
        log.warning("Centroid found no flux, falling back to smoothed peak")
        return find_center_smooth(data)
    yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]
    return float((yy * weights).sum() / total), float((xx * weights).sum() / total)


def find_center_silhouette(data, threshold_frac=0.5, smooth_sigma=2.0):
    """Geometric center of the source silhouette: the unweighted mean
        position of all pixels above the threshold.

        Because it ignores how bright each pixel is, this locates the center of
        a resolved body's disk even when volcanoes, storms, or albedo features
        dominate the flux — unlike peak or centroid methods. Choose
        ``threshold_frac`` near half the typical disk brightness so the mask
        traces the limb.

    Parameters
    ----------
    data : array_like
        2D image to search.
    threshold_frac : float, optional
        Fraction of the source level defining the silhouette.
    smooth_sigma : float, optional
        Gaussian smoothing width in pixels.

    Returns
    -------
    tuple of float
        ``(cy, cx)``, the source position in 0-based pixel coordinates.
    """
    mask = _threshold_mask(data, threshold_frac, smooth_sigma)
    if not mask.any():
        log.warning("Silhouette mask empty, falling back to smoothed peak")
        return find_center_smooth(data)
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def _phase_shift(reference, moving, upsample=20):
    """Sub-pixel shift (dy, dx) that best aligns ``moving`` onto
        ``reference``, via upsampled phase cross-correlation.

    Parameters
    ----------
    moving, reference : ndarray
        Images to align; ``moving`` is shifted onto ``reference``.
    upsample : int, optional
        Sub-pixel precision factor for the cross-correlation.

    Returns
    -------
    tuple of float
        ``(dy, dx)`` shift in pixels.
    """
    from skimage.registration import phase_cross_correlation

    shift, _, _ = phase_cross_correlation(
        np.nan_to_num(reference), np.nan_to_num(moving),
        upsample_factor=upsample, normalization=None)
    return float(shift[0]), float(shift[1])


def find_center_symmetry(data, upsample=20, smooth_sigma=2.0,
                         clip_percentile=90.0):
    """Center of symmetry of a source, from cross-correlation of the image
        with its own 180-degree rotation.

        For a source symmetric about ``c``, the rotated copy is displaced by
        ``2c`` relative to the original, so the correlation peak gives the
        center directly. This uses all the source's flux (not just a peak or a
        threshold) and needs no template or starting guess, which makes it a
        good default for round resolved bodies. Bright surface features are
        clipped first (see :func:`_prepare_source_image`), so a volcano or
        storm does not drag the correlation peak toward itself.

    Parameters
    ----------
    data : array_like
        2D image to search.
    upsample : int, optional
        Sub-pixel precision of the cross-correlation.
    smooth_sigma : float, optional
        Gaussian smoothing width in pixels.
    clip_percentile : float, optional
        Percentile at which the bright tail is clipped, so a single hot spot
        cannot drag the symmetry centre.

    Returns
    -------
    tuple of float
        ``(cy, cx)``, the source position in 0-based pixel coordinates.

    Notes
    -----
    The centre follows from ``c = (N - 1 + s) / 2`` for a symmetry shift ``s``.
    The sign of that expression was wrong once and had to be settled against
    synthetic disks; do not adjust it without doing the same.
    """
    img, _ = _prepare_source_image(data, smooth_sigma, clip_percentile)
    flipped = img[::-1, ::-1]
    dy, dx = _phase_shift(img, flipped, upsample)

    # flipping maps index i to (N-1-i); a shift s between image and flip
    # puts the symmetry center at c = (N - 1 + s) / 2 along each axis
    # (verified against synthetic disks at known positions)
    ny, nx = img.shape
    return (ny - 1 + dy) / 2.0, (nx - 1 + dx) / 2.0


def find_center_crosscorr(data, template, template_center=None, upsample=20):
    """Locate the source by cross-correlating ``data`` against ``template``.

        Returns the position in ``data`` corresponding to ``template_center``
        (default: the template's geometric center), so a whole sequence can be
        aligned to one reference image regardless of source morphology.

    Parameters
    ----------
    data : array_like
        Image to locate within.
    template : ndarray
        Reference image the sequence is being aligned to.
    template_center : tuple of float, optional
        Point in the template whose match is sought; defaults to the
        template's geometric centre.
    upsample : int, optional
        Sub-pixel precision of the cross-correlation.

    Returns
    -------
    tuple of float
        ``(cy, cx)``, the source position in 0-based pixel coordinates.

    Notes
    -----
    Unlike the peak finders, this returns the point matching the *template
    centre*, not the brightest source. In a crowded or sourceless field that
    is the only sensible reference, and the "centre far from the brightest
    peak" sanity warning in :func:`register_beam_stack` is skipped for it.
    """
    if template_center is None:
        template_center = ((template.shape[0] - 1) / 2,
                           (template.shape[1] - 1) / 2)

    dy, dx = _phase_shift(np.nan_to_num(data), np.nan_to_num(template),
                          upsample)
    return template_center[0] + dy, template_center[1] + dx


def find_center_wings(data, r_inner=25.0, r_outer=120.0, upsample=20,
                      smooth_sigma=1.0, iterations=3, center=None):
    """Locate a star by aligning its PSF wings, ignoring the core.

        Intended for coronagraphic data: behind an occulting mask the core is
        suppressed and distorted, so peak- and centroid-based methods latch
        onto the mask edge or a diffraction spot. The extended wings, however,
        remain symmetric about the true star position, so this restricts the
        image to an annulus (``r_inner`` to ``r_outer``, in pixels) and finds
        the symmetry center of what is left.

        The annulus must be concentric with the star for an unbiased answer,
        so the center is refined iteratively from a provisional estimate
        (``center``, else the whole-image symmetry center). Choose ``r_inner``
        just outside the occulting spot and ``r_outer`` where the wings fade
        into background.

    Parameters
    ----------
    data : array_like
        2D image to search.
    r_inner, r_outer : float, optional
        Annulus of PSF wings used, in pixels. ``r_inner`` should clear the
        occulting spot and ``r_outer`` reach where the wings fade.
    upsample : int, optional
        Sub-pixel precision of the cross-correlation.
    smooth_sigma : float, optional
        Gaussian smoothing width in pixels.
    iterations : int, optional
        Refinement passes; the annulus must be concentric with the star for
        an unbiased answer, so the centre is re-derived each pass.
    center : tuple of float, optional
        Starting estimate; defaults to the smoothed peak.

    Returns
    -------
    tuple of float
        ``(cy, cx)``, the source position in 0-based pixel coordinates.

    Notes
    -----
    Seeded from the smoothed peak rather than whole-image symmetry. Seeding
    from symmetry returns the cutout centre when the star sits well off-centre,
    and the wing annulus then never contains the star at all -- which is how a
    DoAr 44 reduction ended up with 60 px of registration scatter.
    """
    from scipy.ndimage import gaussian_filter

    img = np.nan_to_num(np.asarray(data, dtype=float))
    if smooth_sigma:
        img = gaussian_filter(img, smooth_sigma)

    if center is None:
        # Provisional center from the smoothed peak, not whole-image
        # symmetry: when the source sits well away from the middle of a
        # large cutout, the symmetry centre of the whole frame is the
        # frame centre, and the wing annulus then never contains the star.
        center = find_center_smooth(data)

    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    for _ in range(iterations):
        rr = np.hypot(yy - center[0], xx - center[1])
        wings = np.where((rr >= r_inner) & (rr <= r_outer), img, 0.0)

        # symmetry center of the wings alone
        dy, dx = _phase_shift(wings, wings[::-1, ::-1], upsample)
        new_center = ((ny - 1 + dy) / 2.0, (nx - 1 + dx) / 2.0)

        if np.hypot(new_center[0] - center[0],
                    new_center[1] - center[1]) < 0.01:
            center = new_center
            break
        center = new_center

    return center


def find_center_quantile_peak(data, quantile=0.9999999):
    """Locate the source as the pixel at a high quantile of the image.

    Cheap and robust to isolated hot pixels, but integer-precision only: it
    returns a pixel index, not a subpixel position.

    Parameters
    ----------
    data : array_like
        2D image to search.
    quantile : float, optional
        Quantile whose pixel is taken as the source.

    Returns
    -------
    tuple of float
        ``(cy, cx)``, the source position in 0-based pixel coordinates.
    """
    return argquantile(data, quantile)


def find_center_max(data):
    """Locate the source as the single brightest pixel.

    The least robust option: one hot pixel or cosmic ray wins outright. Useful
    only on clean synthetic data or as a deliberate comparison against the
    smoothed finders.

    Parameters
    ----------
    data : array_like
        2D image to search.

    Returns
    -------
    tuple of float
        ``(cy, cx)``, the source position in 0-based pixel coordinates.
    """
    return np.unravel_index(np.nanargmax(data), data.shape)


def find_center_min(data, search_radius=30):
    """Darkest pixel near the core — for saturated PSFs, where the core
        reads low. Searches within ``search_radius`` of the quantile peak.

    Parameters
    ----------
    data : array_like
        2D image to search.
    search_radius : float, optional
        Radius in pixels around the frame centre to search.

    Returns
    -------
    tuple of float
        ``(cy, cx)``, the source position in 0-based pixel coordinates.
    """
    cy, cx = argquantile(data, 0.9999)
    region = make_circle_mask(data.shape, search_radius, center=(cy, cx))
    masked = np.where(region, data, np.nan)
    return np.unravel_index(np.nanargmin(masked), data.shape)


def find_center_gaussian(data, fixed_sigma=5.0, quantile=0.9999999,
                         background_radius=50.0):
    """2D Gaussian fit seeded from the quantile peak.

        The background is the median outside ``background_radius`` px of that
        seed. On a cutout not much bigger than the radius that annulus can be
        empty or tiny, so it falls back to the median of the whole frame rather
        than letting a NaN background propagate into the fit.

    Parameters
    ----------
    data : array_like
        2D image to search.
    fixed_sigma : float, optional
        PSF width held fixed during the fit.
    quantile : float, optional
        Quantile used to seed the fit, via :func:`find_center_quantile_peak`.
    background_radius : float, optional
        Radius outside which the background median is measured.

    Returns
    -------
    tuple of float
        ``(cy, cx)``, the source position in 0-based pixel coordinates.
    """
    try:
        cy, cx = argquantile(data, quantile)
    except ValueError as exc:
        # argquantile cannot seed off an all-NaN frame. There is no centre to
        # return here, so fail with a message that names the cause rather than
        # letting "All-NaN slice encountered" surface from three frames down.
        raise ValueError(
            "find_center_gaussian: cannot locate a source, the frame has no "
            f"finite pixels ({exc})") from exc

    outside = ~make_circle_mask(data.shape, background_radius, center=(cy, cx))
    if outside.sum() < 16:
        log.warning("Background annulus outside %.0f px of (%d, %d) holds %d "
                    "pixels on a %dx%d frame; using the whole-frame median "
                    "instead", background_radius, cy, cx, int(outside.sum()),
                    *data.shape)
        outside = np.ones(data.shape, dtype=bool)

    background = float(np.nanmedian(data[outside]))
    amplitude = float(data[cy, cx]) - background
    if not (np.isfinite(background) and np.isfinite(amplitude)):
        log.warning("Non-finite background (%r) or amplitude (%r) for the "
                    "Gaussian fit; using the quantile peak", background,
                    amplitude)
        return float(cy), float(cx)

    try:
        params = fit_2d_gaussian(
            data, [amplitude, cx, cy, background], fixed_sigma=fixed_sigma)
    except Exception as exc:
        log.warning("Gaussian centering fit failed (%s); using the quantile "
                    "peak", exc)
        return float(cy), float(cx)

    fit_cy, fit_cx = float(params[2]), float(params[1])
    if not (0 <= fit_cy < data.shape[0] and 0 <= fit_cx < data.shape[1]):
        log.warning("Gaussian fit left the frame at (%.1f, %.1f); using the "
                    "quantile peak", fit_cy, fit_cx)
        return float(cy), float(cx)
    return fit_cy, fit_cx  # (cy, cx)


_CENTER_METHODS = {
    "smooth_peak": find_center_smooth,
    "quantile_peak": find_center_quantile_peak,
    "max": find_center_max,
    "min": find_center_min,
    "gaussian": find_center_gaussian,
    "centroid": find_center_centroid,
    "silhouette": find_center_silhouette,
    "symmetry": find_center_symmetry,
    "wings": find_center_wings,
    "crosscorr": find_center_crosscorr,
}


def find_center(data, method="smooth_peak", search_center=None,
                search_radius=None, **kwargs):
    """Locate the star: returns ``(cy, cx)``.

        ``method`` is a built-in name or any callable
        ``f(data, **kwargs) -> (cy, cx)``.

        ``search_center`` / ``search_radius`` restrict the search to a circular
        region, which is what you need when something else in the field can
        outshine the target: a nearby star in a crowded field, a detector
        artefact, or the second Wollaston beam. Everything outside the circle
        is zeroed before the finder runs.

    Parameters
    ----------
    data : array_like
        2D image to search.
    method : str, optional
        Which algorithm to use; see the table above.
    search_center : tuple of float, optional
        Restrict the search to a region around this point.
    search_radius : float, optional
        Half-size of that region, in pixels.
    **kwargs
        Passed to the chosen algorithm -- ``crosscorr`` in particular needs
        ``template=``.

    Returns
    -------
    tuple of float
        ``(cy, cx)``, the source position in 0-based pixel coordinates.

    Raises
    ------
    ValueError
        If ``method`` is not one of the known algorithms.
    """
    finder = _CENTER_METHODS.get(method, method)
    if not callable(finder):
        raise ValueError(f"Unknown centering method {method!r}; options: "
                         f"{sorted(_CENTER_METHODS)} or a callable")

    if search_center is not None and search_radius is not None:
        data = np.where(make_circle_mask(np.shape(data), search_radius,
                                         center=search_center),
                        np.nan_to_num(data), 0.0)
    return finder(data, **kwargs)


def center_frame(frame, method="smooth_peak", background_radius=50,
                 fill=None, **kwargs):
    """Shift a frame so the star lands on the image center.

        The background level (used to fill edges revealed by the shift) is the
        median outside a ``background_radius`` circle around the star. The
        original star position is stored in the CX / CY header keywords
        (0-based pixels).

    Parameters
    ----------
    frame : Frame
        Frame to shift.
    method : str, optional
        Centering algorithm, as for :func:`find_center`.
    fill : float, optional
        Value for pixels shifted in from outside.
    **kwargs
        Passed to the centering algorithm.

    Returns
    -------
    Frame
        A new frame with the star at the image centre; the input is unchanged.
    """
    cy, cx = find_center(frame.data, method=method, **kwargs)

    if fill is None:
        outside = ~make_circle_mask(frame.shape, background_radius,
                                    center=(int(cy), int(cx)))
        fill = float(np.median(frame.data[outside]))

    h, w = frame.shape
    shifted = translate(frame.data, h / 2 - cy, w / 2 - cx, fill=fill)

    centered = Frame(shifted, frame.header.copy())
    centered["CX"] = float(cx)
    centered["CY"] = float(cy)

    log.info("Frame %s: star at (%.1f, %.1f)",
             frame.get("RED-FN", frame.get("FILENAME")), cy, cx)
    return centered


def center_frames(frames, **kwargs):
    """Center every frame in a list (see :func:`center_frame`).

    Parameters
    ----------
    frames : iterable of Frame
        Frames to centre.
    **kwargs
        Passed to :func:`center_frame`.

    Returns
    -------
    list of Frame
        The centred frames.
    """
    return [center_frame(f, **kwargs) for f in frames]


def measure_beam_offset(stack, method="centroid", **kwargs):
    """How far the star in beam 1 sits from the star in beam 0.

    Locates the source independently in each beam of a ``(2, ny, nx)`` stack
    and returns the difference. A correctly split stack measures ~0; anything
    larger is an error in the beam extraction geometry, since the two beams
    are two images of the same sky taken through one optic.

    Parameters
    ----------
    stack : ndarray
        ``(2, ny, nx)`` beam stack, beam 0 bottom and beam 1 top, as returned
        by an instrument's ``split_beams``. Subtract the background first:
        a threshold-based centroid on a raw thermal-background beam measures
        the pedestal, which is common to both beams, and so reports ~0
        however wrong the geometry is.
    method : str, optional
        Centering algorithm, as for :func:`find_center`. The default
        ``"centroid"`` is chosen for this job rather than to match whatever
        the reduction registers with: it is sub-pixel, cheap, and because it
        weights the whole source its errors largely cancel in a difference
        between two images of the same thing. Measured against a hand
        centroid on real data it recovers the offset to 0.05 px with 0.04 px
        scatter frame to frame.

        Do not pass ``"smooth_peak"`` here. It reports whichever local
        maximum is brightest, which on a saturated donut is a different rim
        peak in each beam -- on real L-prime data that yields a spurious
        17 px offset. ``"min"`` is no good either: it returns whole-pixel
        positions, so it cannot see the sub-pixel disagreement this is for.
    **kwargs
        Passed to the centering algorithm.

    Returns
    -------
    tuple of float
        ``(dy, dx)`` in pixels, beam 1 minus beam 0. Add these to the
        instrument's ``top_row_start`` and ``beam_x_offset`` to correct the
        geometry -- both are plain additions, because beam 1 stack row *j*
        is detector row ``j + top_row_start`` and stack column *j* is
        detector column ``j + beam_x_offset``.

    Notes
    -----
    This measures a *relative* offset, which is exactly what registration
    cannot fix: :func:`register_beam_stack` finds one centre on the mean of
    the two beams and shifts both by it, deliberately preserving whatever
    offset lies between them. On a bright star the measurement repeats to
    ~0.05 px between frames, so a residual above ~1 px is real.

    The measurement is only as good as the centering method. On a source the
    finder cannot lock onto -- a faint or extended target, or a saturated
    core under the default ``"smooth_peak"`` -- the two beams can disagree
    for reasons that have nothing to do with the geometry.
    """
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3 or stack.shape[0] != 2:
        raise ValueError("expected a (2, ny, nx) beam stack, got shape "
                         f"{stack.shape}")
    cy0, cx0 = find_center(stack[0], method=method, **kwargs)
    cy1, cx1 = find_center(stack[1], method=method, **kwargs)
    return (cy1 - cy0, cx1 - cx0)


def register_beam_stack(stack, method="smooth_peak", fill=0.0,
                        check_beam_alignment=True, beam_alignment_tol=1.5,
                        beam_alignment_method="centroid", **kwargs):
    """Center a ``(2, ny, nx)`` beam stack on the star.

        The star is located on the *mean* of the two beams and both beams are
        shifted by that same offset, preserving their relative registration.
        Returns ``(centered_stack, (cy, cx))``.

    Parameters
    ----------
    stack : ndarray
        ``(2, ny, nx)`` beam stack, beam 0 bottom and beam 1 top.
    method : str, optional
        Centering algorithm, as for :func:`find_center`.
    fill : float, optional
        Value for pixels shifted in from outside.
    check_beam_alignment : bool, optional
        Warn when the two beams are not aligned with each other, which means
        the instrument's beam extraction geometry is wrong. On by default:
        this function cannot fix such an offset and nothing downstream can
        either, so silence would be the only other outcome.
    beam_alignment_tol : float, optional
        Offset in pixels above which to warn. The default of 1.5 clears the
        ~0.7 px that integer row/column slicing can leave behind even when
        the geometry is the best available, plus the measurement error, while
        still catching the multi-pixel errors that matter.
    beam_alignment_method : str, optional
        Centering algorithm for the check, as for :func:`measure_beam_offset`.
        Deliberately independent of ``method``: the algorithm that registers
        best is not always one that can measure a sub-pixel offset, and
        ``"min"`` and ``"smooth_peak"`` in particular cannot.
    **kwargs
        Passed to the centering algorithm.

    Returns
    -------
    centered : ndarray
        The shifted ``(2, ny, nx)`` stack.
    center : tuple of float
        ``(cy, cx)`` that was found.

    Notes
    -----
    Both beams are shifted by the *same* offset, found on their mean, so their
    relative registration is preserved -- that alignment is what the double
    difference depends on. Warns when the chosen centre is far from the
    smoothed peak, since that usually means the finder locked onto the frame
    edge rather than the star.
    """
    stack = np.asarray(stack, dtype=float)
    mean_beam = np.nanmean(stack, axis=0)
    cy, cx = find_center(mean_beam, method=method, **kwargs)

    # The two beams are one sky through one optic, so any offset between
    # them is a beam-extraction error. Shifting both by a single offset
    # preserves it by design, and the double difference then turns it into a
    # dipole, so warn here -- this is the last point where the two beams are
    # still separable. A finder that fails on one beam is not evidence of
    # anything, so a failure to measure is not reported.
    if check_beam_alignment:
        try:
            dy, dx = measure_beam_offset(stack, method=beam_alignment_method)
        except Exception:
            pass
        else:
            if np.hypot(dy, dx) > beam_alignment_tol:
                log.warning(
                    "Beams are misaligned by (dy=%+.2f, dx=%+.2f) px, above "
                    "the %.2f px tolerance: the beam extraction geometry is "
                    "probably wrong. Registration shifts both beams "
                    "together, so this offset will survive into the double "
                    "difference as a dipole. Add these to the instrument's "
                    "top_row_start and beam_x_offset (see "
                    "instrument.fit_beam_geometry), or pass "
                    "check_beam_alignment=False if the source is one that "
                    "%r centering cannot locate per beam.",
                    dy, dx, beam_alignment_tol, beam_alignment_method)

    # Sanity check: a centre far from the flux-weighted position of the
    # source usually means the finder locked onto the frame rather than the
    # star. Warn rather than silently registering to nonsense.
    # ``crosscorr`` deliberately returns the point matching the template
    # centre, which in a crowded field is nowhere near the brightest
    # source, so the check does not apply to it.
    ref_cy, ref_cx = find_center_smooth(mean_beam)
    if method != "crosscorr" and np.hypot(cy - ref_cy, cx - ref_cx) > 30:
        log.warning("Centering method %r returned (%.1f, %.1f) but the "
                    "smoothed peak is at (%.1f, %.1f) - %.0f px away; the "
                    "registration is probably wrong", method, cy, cx,
                    ref_cy, ref_cx, np.hypot(cy - ref_cy, cx - ref_cx))

    h, w = mean_beam.shape
    out = np.stack([translate(beam, h / 2 - cy, w / 2 - cx, fill=fill)
                    for beam in stack], axis=0)
    return out, (cy, cx)


def derotate_frames(frames, north_angle_func):
    """Derotate frames to put north up.

        ``north_angle_func`` maps a frame's header to its north angle — e.g.
        ``instruments.nirc2.calculate_north_angle``, whose first return value is
        the mean angle. Each frame is rotated by minus that angle, matching the
        AIR.jl call. NOTE: verify the rotation sign on real data (a field with a
        known companion PA) — Julia and scipy image conventions differ, so the
        derotation direction should be sanity-checked once against sky data.

    Parameters
    ----------
    frames : iterable of Frame
        Frames to derotate.
    north_angle_func : callable
        Maps a frame to its north angle in degrees, e.g.
        ``instruments.nirc2.calculate_north_angle``. A tuple return is
        accepted and its first element used.

    Returns
    -------
    list of Frame
        Frames rotated by minus their north angle.
    """
    derotated = []
    for frame in frames:
        angle = north_angle_func(frame)
        if isinstance(angle, tuple):
            angle = angle[0]
        derotated.append(rotate_image_center(frame, -angle))
    return derotated


def median_combine(frames, crop_size=None, crop_center=None):
    """Median-combine a list of frames into one image, optionally cropping
        the result. Returns a Frame carrying the first frame's header.

    Parameters
    ----------
    frames : iterable of Frame
        Frames to combine.
    crop_size : tuple of int, optional
        Crop applied to the result.
    crop_center : tuple of float, optional
        Centre of that crop.

    Returns
    -------
    Frame
        The combined image, carrying the first frame's header.
    """
    cube = framelist_to_cube(frames)
    combined = np.nanmedian(cube, axis=0)
    if crop_size is not None:
        combined, _, _ = crop(combined, crop_size, center=crop_center)
    return Frame(combined, frames[0].header.copy())


def register_frames_to_template(images, template=None, upsample=20):
    """Align a sequence of images to a common reference by cross-correlation.

        ``template`` defaults to the median of the input images, which is a
        robust reference for a repeated observation. Returns
        ``(aligned, shifts)`` where ``shifts`` lists the applied ``(dy, dx)``.

        Alignment is relative: the sequence ends up co-registered on the
        template's frame, not on any absolute center.

    Parameters
    ----------
    images : iterable of array_like
        Images to align.
    template : ndarray, optional
        Reference; defaults to the median of ``images``, which is a robust
        reference for a repeated observation.
    upsample : int, optional
        Sub-pixel precision of the cross-correlation.

    Returns
    -------
    aligned : list of ndarray
        The aligned images.
    shifts : list of tuple
        The ``(dy, dx)`` applied to each.
    """
    images = [np.asarray(im, dtype=float) for im in images]
    if template is None:
        template = np.nanmedian(images, axis=0)

    aligned, shifts = [], []
    for im in images:
        dy, dx = _phase_shift(template, im, upsample)
        aligned.append(translate(im, dy, dx))
        shifts.append((dy, dx))
    return aligned, shifts

def _gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """2D Gaussian model used by :func:`fit_2d_gaussian`.

    Parameters
    ----------
    x, y : ndarray
        Coordinate grids.
    amplitude : float
        Peak height above ``offset``.
    x0, y0 : float
        Centre position.
    sigma_x, sigma_y : float
        Widths along each axis.
    offset : float
        Constant background.

    Returns
    -------
    ndarray
        The model evaluated on the grid.
    """
    return amplitude * np.exp(
        -((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2))
    ) + offset


def fit_2d_gaussian(data, initial_guess, fixed_sigma=None):
    """Least-squares fit of a 2D Gaussian to an image.

        ``initial_guess`` is ``[amplitude, x0, y0, sigma_x, sigma_y, offset]``,
        or ``[amplitude, x0, y0, offset]`` when ``fixed_sigma`` is given.
        Coordinates are 0-based pixel indices (x = column, y = row).

        Returns the fitted parameter array in the same order as the guess.

    Parameters
    ----------
    data : array_like
        Image to fit.
    initial_guess : sequence of float
        Starting parameters, in the same order as the return value.
    fixed_sigma : float, optional
        Hold the width fixed at this value, dropping the two sigma
        parameters from the fit.

    Returns
    -------
    ndarray
        Fitted parameters, ordered as ``initial_guess`` was.
    """
    from scipy.optimize import least_squares

    data = np.asarray(data, dtype=float)
    rows, cols = data.shape
    yy, xx = np.mgrid[:rows, :cols]

    if fixed_sigma is not None:
        def model(p):
            """Gaussian model evaluated on the pixel grid for parameters p."""
            amp, x0, y0, offset = p
            return _gaussian_2d(xx, yy, amp, x0, y0, fixed_sigma, fixed_sigma,
                                offset)
    else:
        def model(p):
            """Gaussian model evaluated on the pixel grid for parameters p."""
            amp, x0, y0, sx, sy, offset = p
            return _gaussian_2d(xx, yy, amp, x0, y0, sx, sy, offset)

    result = least_squares(lambda p: (data - model(p)).ravel(),
                           np.asarray(initial_guess, dtype=float))
    return result.x
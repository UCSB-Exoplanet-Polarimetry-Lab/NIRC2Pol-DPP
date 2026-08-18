"""Match calibration frames to science frames and apply the calibration.

Translated from AIR.jl's reduction.jl. Instrument-agnostic: the bad pixel
mask, gain, saturation limit, and any filter-substitution exceptions are all
parameters. NIRC2-specific values come from instruments/nirc2.py.

The main entry point is :func:`reduce_frame`, which applies::

    reduced = (frame - dark) / flat        (optionally sky-subtracted first)

followed by coadd normalization, bad-pixel replacement, and gain scaling.
"""

from __future__ import annotations

import logging

import numpy as np

from utils.frame import Frame, all_header_keywords_match
from utils.imutils import crop, image_is_larger, plus_mask
from . import defaults

log = logging.getLogger(__name__)

# set once when a flat-type check has to be skipped for want of a table
_WARNED_NO_FLAT_TYPE_TABLE = False


def find_matching_master(frame, masters, keylist):
    """First master whose header matches ``frame`` on every keyword in
        ``keylist``. Returns ``(index, master)`` or ``(None, None)``.

    Parameters
    ----------
    frame : Frame
        Science frame to match.
    masters : list of Frame
        Candidate calibration masters.
    keylist : list of str
        Header keywords that must all agree.

    Returns
    -------
    index : int or None
        Position of the match in the list, or None.
    master : Frame or None
        The matching master, or None when nothing matched.
    """
    if not masters:
        return None, None
    for i, m in enumerate(masters):
        if all_header_keywords_match(frame, m, keylist):
            return i, m
    return None, None


def find_closest_flat(frame, master_flats, ranked_keylists=None,
                      exceptions=None, required_flat_type=None,
                      allow_flat_type_mismatch=False,
                      required_flat_types=None,
                      default_required_flat_type=None):
    """Find the best-matching flat.

        Two things are mandatory. The filter must always match — a flat in another filter describes the
        wrong throughput pattern — but exposure settings are irrelevant, since
        the flat is normalized. Detector size is only a preference: a flat
        covering a *larger* region is trimmed to the frame, so a full-frame
        flat can calibrate a subarray exposure. Matching therefore tries
        "same filter and same size" first, then "same filter, any size"
        (``reduction.defaults.RANKED_FLATS_KEYLISTS``). Within each level the
        flats keep the preference order set by ``make_master_flats``
        (polarimetric first, then the band-appropriate lamp/sky type).

        The flat *type* must also match what the band requires: sky flats at
        L'/M, where the dome lamp is swamped by thermal background, and lamp
        flats in JHK. Reducing with the wrong kind produces a wrong answer that
        still looks reasonable, so a mismatch raises ``ValueError`` rather than
        substituting. ``required_flat_type`` overrides the band default and
        ``allow_flat_type_mismatch=True`` downgrades the error to a warning for
        anyone who genuinely wants it.

        ``exceptions`` maps a filter substring to a tuple of acceptable
        substitute filter names, for filters that have no flats of their own
        (e.g. NIRC2 narrowband — see ``instruments.nirc2.FLAT_EXCEPTIONS``).

    Parameters
    ----------
    frame : Frame
        Science frame needing a flat.
    master_flats : list of Frame
        Candidates, already in preference order from
        :func:`reduction.masters.make_master_flats`.
    ranked_keylists : list of list of str, optional
        Matching criteria from strictest to loosest; defaults to
        ``reduction.defaults.RANKED_FLATS_KEYLISTS``.
    exceptions : dict, optional
        Maps a filter substring to acceptable substitutes.
    required_flat_type : str, optional
        Override the band's required type, ``"SKY"`` or ``"LAMP"``.
    allow_flat_type_mismatch : bool, optional
        Downgrade a type mismatch from an error to a warning, recording
        ``FLATMISM`` in the returned flat's header.

    Returns
    -------
    index : int or None
        Position of the match in the list, or None.
    master : Frame or None
        The matching master, or None when nothing matched.

    Raises
    ------
    ValueError
        If the best match is not the type the band requires and
        ``allow_flat_type_mismatch`` is False.

    Notes
    -----
    The band requirement can only be checked when the instrument's table is
    supplied via ``required_flat_types`` (from
    ``instrument.required_flat_types``), since which flat a band needs
    differs per instrument. Without it the check is skipped and said so
    once, rather than being quietly dropped.
    """
    from .masters import required_flat_type_for
    ranked_keylists = (ranked_keylists if ranked_keylists is not None
                       else defaults.RANKED_FLATS_KEYLISTS)
    exceptions = exceptions or {}

    ind, matched_flat = None, None

    for key, alternates in exceptions.items():
        if key in frame["FILTER"]:
            for alt in alternates:
                for i, m in enumerate(master_flats):
                    if alt in m["FILTER"]:
                        ind, matched_flat = i, m
                        log.warning("Using %s flat for %s frames!",
                                    matched_flat["FILTER"], key)

    if matched_flat is None:
        for keylist in ranked_keylists:
            ind, matched_flat = find_matching_master(frame, master_flats,
                                                     keylist)
            if matched_flat is not None:
                break

    if matched_flat is None:
        log.warning("No matching flat found for %s, %s",
                    frame.get("FILENAME"), frame.get("FILTER"))
        return None, None

    band = str(frame.get("FWINAME")
               or str(frame.get("FILTER", "")).split("+")[0].strip())
    wanted = required_flat_type_for(band, required_flat_type,
                                    required_flat_types,
                                    default_required_flat_type)
    got = str(matched_flat.get("FLATTYPE", "")).split("+")[0] or "UNKNOWN"
    if wanted is None:
        # No instrument table, so there is nothing to check against. Say so
        # once: silently skipping a requirement is how a wrong flat gets
        # through looking plausible, which is the failure this check exists
        # to prevent.
        global _WARNED_NO_FLAT_TYPE_TABLE
        if not _WARNED_NO_FLAT_TYPE_TABLE:
            _WARNED_NO_FLAT_TYPE_TABLE = True
            log.warning(
                "No flat-type table supplied, so the band requirement is "
                "not being enforced (this frame is %s-band with a %s flat). "
                "Pass required_flat_types=instrument.required_flat_types to "
                "check it.", band, got)
    elif got != wanted:
        available = sorted({str(f.get("FLATTYPE", "UNKNOWN")).split("+")[0]
                            for f in master_flats})
        message = (f"{band}-band data requires a {wanted} flat but the best "
                   f"match is {got}; available types: {available}. Reducing "
                   f"with the wrong kind of flat gives a wrong answer that "
                   f"still looks plausible.")
        if not allow_flat_type_mismatch:
            raise ValueError(message)
        log.warning("%s Proceeding because allow_flat_type_mismatch=True.",
                    message)
        matched_flat = Frame(matched_flat.data, matched_flat.header.copy())
        matched_flat["FLATMISM"] = (True, "flat type does not match the band")

    if matched_flat.shape != frame.shape:
        if image_is_larger(matched_flat.data, frame.data):
            log.info("Trimming %s flat from %s to %s for %s",
                     matched_flat.get("FILTER"), matched_flat.shape,
                     frame.shape, frame.get("FILENAME"))
            cropped, _, _ = crop(matched_flat.data, frame.shape)
            matched_flat = Frame(cropped, matched_flat.header.copy())
            matched_flat["FLATTRIM"] = (True, "flat trimmed to the frame size")
        else:
            log.warning("Flat %s is smaller than the frame %s, cannot use "
                        "it for %s", matched_flat.shape, frame.shape,
                        frame.get("FILENAME"))
            return None, None

    return ind, matched_flat


def find_closest_dark(frame, master_darks, ranked_keylists=None):
    """Find the best-matching dark, relaxing the match criteria step by step
        (see ``reduction.defaults.RANKED_DARKS_KEYLISTS``). When only ITIME
        matches, the dark is rescaled by the frame's COADDS and cropped to size.

    Parameters
    ----------
    frame : Frame
        Science frame needing a dark.
    master_darks : list of Frame
        Candidates.
    ranked_keylists : list of list of str, optional
        Matching criteria from strictest to loosest; defaults to
        ``reduction.defaults.RANKED_DARKS_KEYLISTS``.

    Returns
    -------
    index : int or None
        Position of the match in the list, or None.
    master : Frame or None
        The matching master, or None when nothing matched.

    Notes
    -----
    The loosest level matches on ITIME alone, at which point the dark is
    rescaled by the frame's COADDS and cropped to size -- so one set of
    full-frame darks can serve every subarray of a night.
    """
    ranked_keylists = (ranked_keylists if ranked_keylists is not None
                       else defaults.RANKED_DARKS_KEYLISTS)

    if not master_darks:
        return None, None

    for rank, keylist in enumerate(ranked_keylists):
        ind, matched_dark = find_matching_master(frame, master_darks, keylist)
        if matched_dark is None:
            continue

        # for looser matches, rescale by coadds
        if "COADDS" not in keylist:
            log.warning("Rescaling dark frame by COADDS %s -> %s",
                        matched_dark["COADDS"], frame["COADDS"])
            data = (matched_dark.data / matched_dark["COADDS"]
                    * frame["COADDS"])
            matched_dark = Frame(data, matched_dark.header.copy())

        # loosest match may not even be the same size
        if matched_dark.shape != frame.shape:
            if image_is_larger(matched_dark.data, frame.data):
                cropped, _, _ = crop(matched_dark.data, frame.shape)
                matched_dark = Frame(cropped, matched_dark.header.copy())
            else:
                log.warning("Dark frame is smaller than the target frame, "
                            "not cropping: %s", frame.get("FILENAME"))
                return None, None

        return ind, matched_dark

    log.warning("No matching dark found for %s, %s, %s, %s",
                frame.get("FILENAME"), frame.get("FILTER"),
                frame.get("ITIME"), frame.get("COADDS"))
    return None, None


def find_closest_sky(frame, master_skies, ranked_keylists=None):
    """Find the best-matching sky, relaxing FILTER/ITIME/COADDS step by step
        and rescaling by exposure time and coadds for the looser matches.

    Parameters
    ----------
    frame : Frame
        Science frame needing a sky.
    master_skies : list of Frame
        Candidates.
    ranked_keylists : list of list of str, optional
        Matching criteria from strictest to loosest.

    Returns
    -------
    index : int or None
        Position of the match in the list, or None.
    master : Frame or None
        The matching master, or None when nothing matched.
    """
    ranked_keylists = (ranked_keylists if ranked_keylists is not None
                       else defaults.RANKED_SKIES_KEYLISTS)

    if not master_skies:
        return None, None

    for rank, keylist in enumerate(ranked_keylists):
        ind, matched_sky = find_matching_master(frame, master_skies, keylist)
        if matched_sky is None:
            continue

        if "COADDS" not in keylist and "ITIME" in keylist:
            log.warning("Rescaling sky frame by COADDS %s -> %s",
                        matched_sky["COADDS"], frame["COADDS"])
            data = matched_sky.data / matched_sky["COADDS"] * frame["COADDS"]
            matched_sky = Frame(data, matched_sky.header.copy())
        elif "ITIME" not in keylist:
            log.warning("Rescaling sky frame by ITIME %s -> %s and "
                        "COADDS %s -> %s",
                        matched_sky["ITIME"], frame["ITIME"],
                        matched_sky["COADDS"], frame["COADDS"])
            data = (matched_sky.data
                    / (matched_sky["ITIME"] * matched_sky["COADDS"])
                    * (frame["ITIME"] * frame["COADDS"]))
            matched_sky = Frame(data, matched_sky.header.copy())

        if matched_sky.shape != frame.shape:
            if image_is_larger(matched_sky.data, frame.data):
                cropped, _, _ = crop(matched_sky.data, frame.shape)
                matched_sky = Frame(cropped, matched_sky.header.copy())
            else:
                log.warning("Sky frame is smaller than the target frame, "
                            "not cropping: %s", frame.get("FILENAME"))
                return None, None

        return ind, matched_sky

    log.warning("No matching sky found")
    return None, None


def reduce_frame(frame, master_flats, master_darks, master_skies=None,
                 masks=None, bad_pixel_mask=None, flat_exceptions=None,
                 required_flat_type=None, allow_flat_type_mismatch=False,
                 required_flat_types=None, default_required_flat_type=None,
                 bad_pixel_mask_size=9, bad_pixel_plus_mask_size=11,
                 gain=1.0, saturation_limit=1e12, skip_sky_sub=True,
                 div_coadds=True, div_itime=False,
                 replacement_method="interpolation"):
    """Dark-subtract, flat-divide, and clean up a single science frame.

    Parameters
    ----------
    frame : Frame
        Raw science frame.
    master_flats, master_darks, master_skies : list of Frame
        Master calibration frames; the best match is chosen per frame.
    masks : dict, optional
        Extra bad-pixel masks keyed by array shape, as produced by
        :func:`reduction.masters.make_master_masks`.
    bad_pixel_mask : ndarray of bool, optional
        Static detector bad-pixel mask (e.g. from
        ``instruments.nirc2.load_bad_pixel_mask()``).
    flat_exceptions : dict, optional
        Filter substitution rules passed to :func:`find_closest_flat`.
    required_flat_types : mapping, optional
        Band to required flat type, from ``instrument.required_flat_types``.
        Without it the band requirement cannot be enforced.
    default_required_flat_type : str, optional
        Fallback for bands that mapping does not list, from
        ``instrument.default_required_flat_type``.
    gain : float
        Multiplicative gain (e-/ADU) applied at the end.
    saturation_limit : float
        Pixels above this (within the bad-pixel mask) get "+"-shaped
        replacement, since saturation bleeds along detector columns.
    replacement_method : {"interpolation", "median"}
        How to fill bad pixels.

    Returns the reduced Frame with bookkeeping header keywords (DARKSUB,
    FLATDIV, SKYSUB, DIVCOADD, DIVITIME, GAIN, RED-FN).
    """
    reduced = frame.copy()
    masks = masks or {}

    _, matched_flat = find_closest_flat(
        reduced, master_flats, exceptions=flat_exceptions,
        required_flat_type=required_flat_type,
        allow_flat_type_mismatch=allow_flat_type_mismatch,
        required_flat_types=required_flat_types,
        default_required_flat_type=default_required_flat_type)
    _, matched_dark = find_closest_dark(reduced, master_darks)

    if matched_dark is None:
        reduced["DARKSUB"] = False
        dark_data = np.zeros(reduced.shape)
    else:
        reduced["DARKSUB"] = True
        dark_data = matched_dark.data

    if matched_flat is None:
        reduced["FLATDIV"] = False
        flat_data = np.ones(reduced.shape)
    else:
        reduced["FLATDIV"] = True
        flat_data = matched_flat.data

    reduced["SKYSUB"] = False
    if master_skies and not skip_sky_sub:
        _, matched_sky = find_closest_sky(frame, master_skies)
        if matched_sky is not None:
            log.info("Subtracting sky frame from %s", reduced.get("FILENAME"))
            reduced.data -= matched_sky.data
            reduced["SKYSUB"] = True

    reduced.data = (reduced.data - dark_data) / flat_data

    # assemble the full bad-pixel mask: detector mask + shape-matched extra
    # masks + non-finite pixels
    if bad_pixel_mask is not None:
        mask = np.asarray(bad_pixel_mask, dtype=bool)
        if mask.shape != reduced.shape:
            mask, _, _ = crop(mask, reduced.shape)
        mask = mask.copy()
    else:
        mask = np.zeros(reduced.shape, dtype=bool)

    if reduced.shape in masks:
        mask |= masks[reduced.shape]

    mask |= ~np.isfinite(reduced.data)

    reduced["DIVCOADD"] = False
    if div_coadds:
        reduced.data /= reduced["COADDS"]
        reduced["DIVCOADD"] = True

    # saturated pixels bleed into a "+" shape, so they get a wider mask
    sat_mask = mask & (reduced.data > saturation_limit)
    mask_without_sat = mask & ~sat_mask
    sat_plus_mask = plus_mask(sat_mask)

    if replacement_method == "interpolation":
        local_interpolate_bad_pixels(reduced.data, mask_without_sat,
                                     bad_pixel_mask_size)
        local_interpolate_bad_pixels(reduced.data, sat_plus_mask,
                                     bad_pixel_plus_mask_size)
    elif replacement_method == "median":
        local_median_replace_bad_pixels(reduced.data, mask_without_sat,
                                        bad_pixel_mask_size)
        local_median_replace_bad_pixels(reduced.data, sat_plus_mask,
                                        bad_pixel_plus_mask_size)
    else:
        raise ValueError(f"Invalid replacement method: {replacement_method}")

    reduced["DIVITIME"] = False
    if div_itime:
        reduced.data /= reduced["ITIME"]
        reduced["DIVITIME"] = True

    reduced.data *= gain
    reduced["GAIN"] = gain

    reduced["RED-FN"] = f"reduced_{int(reduced['FRAMENO']):04d}.fits"

    from utils.provenance import record_step

    record_step(reduced, "dark/flat reduction",
                dark=(matched_dark.get("FILENAME", "?")
                      if matched_dark is not None else "none"),
                flat=(matched_flat.get("FILENAME", "?")
                      if matched_flat is not None else "none"),
                polflat=(matched_flat.get("POLFLAT")
                         if matched_flat is not None else None),
                gain=gain, saturation=saturation_limit,
                div_coadds=div_coadds, div_itime=div_itime,
                badpix=replacement_method)

    return reduced


def local_median_replace_bad_pixels(data, mask, median_size, fail_val=0.0):
    """Replace masked pixels in-place with the median of the good pixels in
        a ``median_size`` x ``median_size`` window around each.

    Parameters
    ----------
    data : ndarray
        Image, modified in place.
    mask : ndarray of bool
        True where pixels must be replaced.
    median_size : int
        Side of the neighbourhood used for each replacement.
    fail_val : float, optional
        Value used where the neighbourhood holds no good pixels.

    Returns
    -------
    None
        ``data`` is modified in place.
    """
    half = median_size // 2
    height, width = data.shape

    for i, j in zip(*np.nonzero(mask)):
        i0, i1 = max(0, i - half), min(height, i + half + 1)
        j0, j1 = max(0, j - half), min(width, j + half + 1)

        window = data[i0:i1, j0:j1]
        window_mask = mask[i0:i1, j0:j1]

        good = window[~window_mask]
        data[i, j] = np.median(good) if good.size else fail_val


def local_interpolate_bad_pixels(data, mask, kernel_size):
    """Replace masked pixels in-place by thin-plate-spline interpolation of
        the good pixels in a local window around each bad pixel.

    Parameters
    ----------
    data : ndarray
        Image, modified in place.
    mask : ndarray of bool
        True where pixels must be replaced.
    kernel_size : int
        Side of the neighbourhood interpolated over.

    Returns
    -------
    None
        ``data`` is modified in place.
    """
    half = kernel_size // 2
    height, width = data.shape

    for i, j in zip(*np.nonzero(mask)):
        i0, i1 = max(0, i - half), min(height, i + half + 1)
        j0, j1 = max(0, j - half), min(width, j + half + 1)

        interpolate_bad_pixels(data[i0:i1, j0:j1], mask[i0:i1, j0:j1])


def interpolate_bad_pixels(data, mask, fail_val=0.0):
    """Thin-plate-spline interpolate all masked pixels of ``data`` in-place
        from the unmasked pixels.

    Parameters
    ----------
    data : ndarray
        Image, modified in place.
    mask : ndarray of bool
        True where pixels must be replaced.
    fail_val : float, optional
        Value used when the interpolation cannot be solved.

    Returns
    -------
    None
        ``data`` is modified in place.

    Notes
    -----
    Clustered bad pixels can leave the radial-basis-function solve singular,
    because the surviving good pixels are collinear. That is why the Cygnus A
    detector defect is blanked to NaN rather than having its mask dilated --
    growing the mask made the interpolator fail outright.
    """
    from scipy.interpolate import RBFInterpolator

    good = np.argwhere(~mask)
    bad = np.argwhere(mask)

    if bad.size == 0:
        return
    if good.shape[0] < 3:  # not enough points to interpolate
        data[mask] = fail_val
        return

    itp = RBFInterpolator(good.astype(float), data[~mask],
                          kernel="thin_plate_spline")
    data[mask] = itp(bad.astype(float))

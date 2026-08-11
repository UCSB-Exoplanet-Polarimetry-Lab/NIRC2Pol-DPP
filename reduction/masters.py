"""Build master calibration frames (darks, flats, skies) from raw frames.

Translated from AIR.jl's reduction.jl (make_masters) and
generic_reduce/02_make_masters.jl, but instrument-agnostic: the detector
bad-pixel mask and the header keywords used to group frames are all
parameters. Anything NIRC2-specific (which keywords to use, where the bad
pixel mask lives) is supplied by the caller — see instruments/nirc2.py.

Typical use::

    from instruments import nirc2
    bpm = nirc2.load_bad_pixel_mask()

    master_darks, dark_masks = make_master_darks(dark_frames, bad_pixel_mask=bpm)
    master_flats, flat_masks = make_master_flats(
        flat_frames, sky_frames, lampon_frames,
        master_darks, bad_pixel_mask=bpm)
    master_masks = make_master_masks(dark_masks, flat_masks)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

from utils.frame import Frame, framelist_to_cube, match_keys
from utils.imutils import crop
from .calibrate import find_closest_dark
from . import defaults

log = logging.getLogger(__name__)


def make_masters(frames, keylist, bad_pixel_mask=None, n_sigma=6.0,
                 median_size=7, method=np.nanmedian, min_frames=3):
    """Group frames by the header keywords in ``keylist`` and combine each
        group into a master frame.

        For each group: frames are stacked and combined with ``method`` (median
        by default), a hot-pixel mask is built by OR-ing per-frame sigma-clip
        masks with the detector ``bad_pixel_mask``, and masked pixels are
        replaced by a local median.

        Returns ``(master_frames, master_masks)``, both dicts keyed by the tuple
        of header values.

    Parameters
    ----------
    frames : list of Frame
        Frames to group and combine.
    keylist : list of str
        Header keywords defining a group.
    bad_pixel_mask : ndarray of bool, optional
        Static detector mask, OR-ed into each group's mask.
    n_sigma : float, optional
        Sigma-clip threshold for hot pixels.
    median_size : int, optional
        Neighbourhood used to replace masked pixels.
    method : callable, optional
        Combiner, ``np.nanmedian`` by default.
    min_frames : int, optional
        Groups with fewer frames are skipped with a warning.

    Returns
    -------
    masters : dict
        Maps the group key to the combined Frame.
    masks : dict
        Maps the group key to that group's bad-pixel mask.
    """
    frame_dict = match_keys(frames, keylist)

    for key in list(frame_dict):
        if len(frame_dict[key]) < min_frames:
            log.warning("Not enough frames for key %s, skipping...", key)
            del frame_dict[key]
        else:
            log.info("Making master for key -> %s, count -> %d", key,
                     len(frame_dict[key]))

    master_frames = {}
    master_masks = {}
    for key, group in frame_dict.items():
        stack = framelist_to_cube(group)
        master = Frame(method(stack, axis=0))

        # per-frame sigma clip masks, OR-ed together
        from utils.imutils import make_sigma_clip_mask

        mask = np.zeros(master.shape, dtype=bool)
        for f in group:
            mask |= make_sigma_clip_mask(f.data, n_sigma)

        # fold in the detector bad pixel mask, cropped to size if needed
        if bad_pixel_mask is not None:
            bpm = np.asarray(bad_pixel_mask, dtype=bool)
            if bpm.shape != master.shape:
                bpm, _, _ = crop(bpm, master.shape)
            mask |= bpm

        # replace masked pixels with the local median
        median_master = ndimage.median_filter(master.data, size=median_size)
        master.data[mask] = median_master[mask]

        # repopulate the header with the grouping keys so masters can be
        # matched to science frames later
        for k, v in zip(keylist, key):
            master[k] = v

        master["FILENAME"] = group[0].get("FILENAME", "")
        master["NFRAMES"] = len(group)
        master["NSIGMASK"] = n_sigma
        master["MEDSIZE"] = median_size
        master["NPIXMASK"] = int(mask.sum())
        master["MAMEDIAN"] = float(np.median(master.data[~mask]))
        master["MAMEAN"] = float(np.mean(master.data[~mask]))
        master["MASTD"] = float(np.std(master.data[~mask]))

        master_frames[key] = master
        master_masks[key] = mask

    return master_frames, master_masks


def make_master_darks(dark_frames, keylist=None, bad_pixel_mask=None,
                      min_frames=3, **kwargs):
    """Build master darks, one per unique combination of ``keylist`` header
        values. Returns ``(master_darks, masks)`` as flat lists.

    Parameters
    ----------
    dark_frames : list of Frame
        Dark frames.
    keylist : list of str, optional
        Grouping keywords; defaults to ``defaults.DARKS_KEYLIST``.
    bad_pixel_mask : ndarray of bool, optional
        Static detector mask.
    min_frames : int, optional
        Minimum frames per master.
    **kwargs
        Passed to :func:`make_masters`.

    Returns
    -------
    masters : list of Frame
        One master dark per exposure setting.
    masks : list of ndarray
        Matching bad-pixel masks.
    """
    keylist = keylist if keylist is not None else defaults.DARKS_KEYLIST

    if len(dark_frames) < min_frames:
        log.warning("Not enough dark frames found, skipping...")
        return [], []

    masters, masks = make_masters(dark_frames, keylist,
                                  bad_pixel_mask=bad_pixel_mask,
                                  min_frames=min_frames, **kwargs)
    return list(masters.values()), list(masks.values())


def split_polarimetric_flats(flat_frames, modulator_keyword, critical_angles,
                             atol=1.0):
    """Split flats into (polarimetric, regular).

        *Polarimetric* flats are taken as a discrete sequence at the modulator's
        critical angles; combining a full set averages the polarized response
        of the flat source over the modulation cycle. Flats taken while the
        modulator rotates continuously (calibration sweeps) do not form such a
        set and are treated as regular flats.

    Parameters
    ----------
    flat_frames : list of Frame
        Flats to split.
    modulator_keyword : str
        Header keyword holding the modulator angle, e.g. ``"PCUPR"``.
    critical_angles : iterable of float
        The instrument's critical angles.
    atol : float, optional
        Tolerance in degrees, compared circularly.

    Returns
    -------
    polarimetric : list of Frame
        Flats taken at a critical angle.
    regular : list of Frame
        Everything else.
    """
    from utils.angles import is_critical_angle

    pol, regular = [], []
    for f in flat_frames:
        angle = f.get(modulator_keyword)
        if angle is not None and is_critical_angle(float(angle),
                                                   critical_angles, atol):
            pol.append(f)
        else:
            regular.append(f)
    return pol, regular


def make_flats(flat_frames, master_darks, keylist=None, bad_pixel_mask=None,
               flattype="REGULAR", min_frames=3, polarimetric=False, **kwargs):
    """Build master flats: median-combine, subtract the best-matching master
        dark, then normalize by the median. ``FLATTYPE`` records how each flat
        was made (and whether a dark was available); ``POLFLAT`` records whether
        the flats were a critical-angle polarimetric set.

    Parameters
    ----------
    flat_frames : list of Frame
        Flats to combine.
    master_darks : list of Frame
        Darks to subtract from them.
    keylist : list of str, optional
        Grouping keywords.
    bad_pixel_mask : ndarray of bool, optional
        Static detector mask.
    flattype : str, optional
        Tag written to ``FLATTYPE``; suffixed ``+NODARK`` when no dark matched.
    min_frames : int, optional
        Minimum frames per master.
    polarimetric : bool, optional
        Mark the results ``POLFLAT``, which ranks them first.
    **kwargs
        Passed to :func:`make_masters`.

    Returns
    -------
    masters : dict
        Maps the group key to the normalized master flat.
    masks : dict
        Matching bad-pixel masks.
    """
    keylist = keylist if keylist is not None else defaults.FLATS_KEYLIST

    if len(flat_frames) < min_frames:
        log.warning("Not enough flat frames found, skipping...")
        return {}, {}

    master_flats, masks = make_masters(flat_frames, keylist,
                                       bad_pixel_mask=bad_pixel_mask,
                                       min_frames=min_frames, **kwargs)

    for key, flat in master_flats.items():
        flat["POLFLAT"] = bool(polarimetric)
        _, matched_dark = find_closest_dark(flat, master_darks)
        if matched_dark is not None:
            log.info("Subtracting dark from master flat %s", key)
            flat.data -= matched_dark.data
            flat["FLATTYPE"] = flattype
        else:
            log.warning("No matching dark found for master flat %s", key)
            flat["FLATTYPE"] = f"{flattype}+NODARK"

        flat.data /= flat["MAMEDIAN"]

    return master_flats, masks


def make_master_skies(sky_frames, master_darks, keylist=None,
                      bad_pixel_mask=None, min_frames=3, **kwargs):
    """Build master skies: like flats (dark-subtracted) but *not* normalized,
        since skies are subtracted rather than divided. Returns flat lists.

    Parameters
    ----------
    sky_frames : list of Frame
        Sky frames.
    master_darks : list of Frame
        Darks to subtract.
    keylist : list of str, optional
        Grouping keywords.
    bad_pixel_mask : ndarray of bool, optional
        Static detector mask.
    min_frames : int, optional
        Minimum frames per master.
    **kwargs
        Passed to :func:`make_masters`.

    Returns
    -------
    masters : list of Frame
        Master skies, **not** normalized -- they are subtracted, not divided.
    masks : list of ndarray
        Matching bad-pixel masks.
    """
    keylist = keylist if keylist is not None else defaults.FLATS_KEYLIST

    if len(sky_frames) < min_frames:
        log.warning("Not enough sky frames found, skipping...")
        return [], []

    master_skies, masks = make_masters(sky_frames, keylist,
                                       bad_pixel_mask=bad_pixel_mask,
                                       min_frames=min_frames, **kwargs)

    for key, sky in master_skies.items():
        _, matched_dark = find_closest_dark(sky, master_darks)
        if matched_dark is not None:
            log.info("Subtracting dark from master sky %s", key)
            sky.data -= matched_dark.data
            sky["FLATTYPE"] = "SKY"
        else:
            log.warning("No matching dark found for master sky %s", key)
            sky["FLATTYPE"] = "SKY+NODARK"

    return list(master_skies.values()), list(masks.values())


def make_lamp_flats(lampon_frames, master_darks, keylist=None,
                    bad_pixel_mask=None, min_frames=3, **kwargs):
    """Build lamp flats: lamp-on minus the matched master dark, normalized
        by the median.

        Lamp-off frames are not used. They are meaningless in JHK, and at L' the
        dome lamp is swamped by thermal background so sky flats are used
        instead, which leaves the dark as the thing to subtract.

    Parameters
    ----------
    lampon_frames : list of Frame
        Lamp-on dome flats.
    master_darks : list of Frame
        Darks to subtract.
    keylist : list of str, optional
        Grouping keywords.
    bad_pixel_mask : ndarray of bool, optional
        Static detector mask.
    min_frames : int, optional
        Minimum frames per master.
    **kwargs
        Passed to :func:`make_masters`.

    Returns
    -------
    masters : dict
        Maps the group key to the master flat, tagged ``LAMP`` when a dark was
        subtracted and ``LAMP+NODARK`` when none matched.
    masks : dict
        Matching bad-pixel masks.

    Notes
    -----
    The tag is load-bearing, not bookkeeping:
    :func:`reduction.calibrate.find_closest_flat` refuses a flat whose type is
    not the one the band requires, and :func:`flat_sort_key` ranks ``+NODARK``
    variants below dark-subtracted ones.
    """
    keylist = keylist if keylist is not None else defaults.FLATS_KEYLIST

    def _make(frames, label):
        """Combine one labelled set of frames, or skip it if too few.

        Parameters
        ----------
        frames : list of Frame
            Frames to combine.
        label : str
            Name used in the "not enough frames" warning.

        Returns
        -------
        masters, masks : dict
            As :func:`make_masters`, or empty dicts when there were too few.
        """
        if len(frames) < min_frames:
            log.warning("Not enough %s frames found, skipping...", label)
            return {}, {}
        return make_masters(frames, keylist, bad_pixel_mask=bad_pixel_mask,
                            min_frames=min_frames, **kwargs)

    master_lampon, lampon_masks = _make(lampon_frames, "lamp-on")

    master_flat_lamp = {}
    master_flat_lamp_masks = {}
    for key, flat in master_lampon.items():
        _, matched_dark = find_closest_dark(flat, master_darks)
        if matched_dark is None:
            log.warning("No matching dark for lamp flat %s; it will rank "
                        "below dark-subtracted flats", key)
            sub_frame = np.zeros(flat.shape)
            flat["FLATTYPE"] = "LAMP+NODARK"
        else:
            sub_frame = matched_dark.data
            flat["FLATTYPE"] = "LAMP"
        master_flat_lamp_masks[key] = lampon_masks[key]

        flat.data -= sub_frame
        flat.data /= np.median(flat.data)
        master_flat_lamp[key] = flat

    return master_flat_lamp, master_flat_lamp_masks


def required_flat_type_for(band, override=None):
    """Which flat type a band requires: sky flats in the thermal infrared,
        lamp flats in the near infrared. ``override`` (e.g. "SKY") wins, letting
        a user ask for sky flats in JHK.

        This is a requirement rather than a preference -- reducing L' data with
        a lamp flat gives a wrong answer that still looks reasonable -- and is
        enforced by :func:`reduction.calibrate.find_closest_flat`.

    Parameters
    ----------
    band : str
        Observing band, e.g. from ``instruments.nirc2.band_of``.
    override : str, optional
        Force a type, ``"SKY"`` or ``"LAMP"``; case-insensitive.

    Returns
    -------
    str
        The required flat type.
    """
    if override:
        return str(override).upper()
    key = str(band or "").strip()
    return defaults.REQUIRED_FLAT_TYPE_BY_BAND.get(
        key, defaults.DEFAULT_REQUIRED_FLAT_TYPE)


def flat_sort_key(flat, required_type=None):
    """Sort key implementing the flat preference order.

        1. polarimetric flats (critical-angle sets) before all others
        2. flats with a dark subtracted before "+NODARK" variants
        3. the band-required type (sky for L'/M, lamp for JHK) before others
        4. more frames first

    Parameters
    ----------
    flat : Frame
        Master flat to rank.
    required_type : str, optional
        Override the band default, as for :func:`required_flat_type_for`.

    Returns
    -------
    tuple
        Sort key; lower sorts first.
    """
    flattype = str(flat.get("FLATTYPE", ""))
    base = flattype.split("+")[0]
    nodark = "NODARK" in flattype

    band = str(flat.get("FWINAME")
               or str(flat.get("FILTER", "")).split("+")[0].strip())
    wanted = required_flat_type_for(band, required_type)

    if base == wanted:
        type_rank = 0
    elif base in ("SKY", "LAMP"):
        type_rank = 1          # the other real flat-field kind
    else:
        type_rank = 2          # generic "REGULAR"

    return (not flat.get("POLFLAT", False), nodark, type_rank,
            -flat.get("NFRAMES", 0))


def make_master_flats(flat_frames, sky_frames, lampon_frames,
                      master_darks, keylist=None, bad_pixel_mask=None,
                      modulator_keyword=None, critical_angles=None,
                      required_flat_type=None, **kwargs):
    """Build every available kind of flat and return a single ranked list:
        for any science frame, the first matching flat in the list is the best
        available one.

        When ``modulator_keyword`` and ``critical_angles`` are given (from the
        instrument), dome/regular flats are split into a *polarimetric* set
        (taken at the critical angles) and everything else. Polarimetric flats
        are ranked ahead of all other flats, so they are used whenever they
        exist and older data without them fall back to regular flats
        automatically.

        Ordering (see :func:`flat_sort_key`): polarimetric flats first, then
        dark-subtracted before darkless, then the band-required type — sky
        flats for L'/M where the dome lamp is swamped by thermal background,
        lamp flats for JHK — then the set built from the most frames.
        ``required_flat_type`` ("SKY" or "LAMP") overrides the band default,
        e.g. to use sky flats in JHK.

        Ordering is only a preference among *valid* flats; the type requirement
        itself is enforced later, per science frame, by
        :func:`reduction.calibrate.find_closest_flat`.

        Returns ``(master_flats, masks)`` as flat lists.

    Parameters
    ----------
    flat_frames : list of Frame
        Generic flats.
    sky_frames : list of Frame
        Sky flats.
    lampon_frames : list of Frame
        Lamp-on dome flats.
    master_darks : list of Frame
        Darks to subtract.
    keylist : list of str, optional
        Grouping keywords.
    bad_pixel_mask : ndarray of bool, optional
        Static detector mask.
    modulator_keyword : str, optional
        Enables the polarimetric split when given with ``critical_angles``.
    critical_angles : iterable of float, optional
        The instrument's critical angles.
    required_flat_type : str, optional
        Override the band's required type.
    **kwargs
        Passed through to the individual flat builders.

    Returns
    -------
    flats : list of Frame
        All masters, in preference order.
    masks : list of ndarray
        Matching bad-pixel masks.
    """
    pol_flats, pol_masks = {}, {}
    if modulator_keyword is not None and critical_angles is not None:
        pol_frames, flat_frames = split_polarimetric_flats(
            flat_frames, modulator_keyword, critical_angles)
        if pol_frames:
            log.info("Found %d polarimetric flats at critical angles "
                     "(%d other flats)", len(pol_frames), len(flat_frames))
            pol_flats, pol_masks = make_flats(
                pol_frames, master_darks, keylist=keylist,
                bad_pixel_mask=bad_pixel_mask, polarimetric=True, **kwargs)
        else:
            log.info("No critical-angle polarimetric flats found; "
                     "falling back to regular flats")

    master_flats, flats_masks = make_flats(
        flat_frames, master_darks, keylist=keylist,
        bad_pixel_mask=bad_pixel_mask, **kwargs)

    master_flats_sky, sky_masks = make_flats(
        sky_frames, master_darks, keylist=keylist,
        bad_pixel_mask=bad_pixel_mask, flattype="SKY", **kwargs)

    master_flats_lamp, lamp_masks = make_lamp_flats(
        lampon_frames, master_darks, keylist=keylist,
        bad_pixel_mask=bad_pixel_mask, **kwargs)

    # polarimetric flats keep separate keys so they are never merged away
    combined = {**master_flats, **master_flats_sky, **master_flats_lamp}
    flats = list(combined.values()) + list(pol_flats.values())
    flats.sort(key=lambda f: flat_sort_key(f, required_flat_type))

    if flats:
        log.info("Flat preference order: %s",
                 ", ".join(f"{f.get('FILTER')}/{f.get('FLATTYPE')}"
                           f"{'/POL' if f.get('POLFLAT') else ''}"
                           f"(n={f['NFRAMES']})" for f in flats[:6]))

    combined_masks = {**flats_masks, **sky_masks, **lamp_masks, **pol_masks}
    return flats, list(combined_masks.values())


def make_master_masks(*mask_lists):
    """Combine all master masks, OR-ing together those with the same shape.

        Returns a dict mapping shape -> combined boolean mask, which is what
        :func:`reduction.calibrate.reduce_frame` expects for its ``masks``
        argument.

    Parameters
    ----------
    *mask_lists
        Any number of lists of boolean masks.

    Returns
    -------
    dict
        Maps array shape to the OR of every mask of that shape, which is what
        :func:`reduction.calibrate.reduce_frame` expects for ``masks``.
    """
    by_shape = {}
    for masks in mask_lists:
        for mask in masks:
            mask = np.asarray(mask, dtype=bool)
            key = mask.shape
            if key in by_shape:
                by_shape[key] = by_shape[key] | mask
            else:
                by_shape[key] = mask
    return by_shape

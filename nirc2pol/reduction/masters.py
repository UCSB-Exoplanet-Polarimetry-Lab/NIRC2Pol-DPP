"""Build master calibration frames (darks, flats, skies) from raw frames.

Translated from AIR.jl's reduction.jl (make_masters) and
generic_reduce/02_make_masters.jl, but instrument-agnostic: the detector
bad-pixel mask and the header keywords used to group frames are all
parameters. Anything NIRC2-specific (which keywords to use, where the bad
pixel mask lives) is supplied by the caller — see nirc2pol/instruments/nirc2.py.

Typical use::

    from nirc2pol.instruments import nirc2
    bpm = nirc2.load_bad_pixel_mask()

    master_darks, dark_masks = make_master_darks(dark_frames, bad_pixel_mask=bpm)
    master_flats, flat_masks = make_master_flats(
        dome_frames, sky_frames,
        master_darks, bad_pixel_mask=bpm)
    master_masks = make_master_masks(dark_masks, flat_masks)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

from nirc2pol.utils.frame import (Frame, framelist_to_cube,
                                  group_by_pointing, match_keys)
from nirc2pol.utils.imutils import crop
from .calibrate import find_closest_dark
from nirc2pol.utils.provenance import record_step

from . import defaults

log = logging.getLogger(__name__)


def _instrument_default(instrument, value, attr, call=False):
    """Fall back to an instrument attribute, leaving explicit values alone.

    Parameters
    ----------
    instrument : PolarimetryData or None
        Source of the default. None means there is nothing to fall back to.
    value : object
        What the caller passed. Anything but None is returned untouched, so
        an explicit argument always beats the instrument.
    attr : str
        Attribute to read.
    call : bool, optional
        Call the attribute rather than reading it, for the ones that are
        methods. Only reached when the value is actually needed, which
        matters for ``bad_pixel_mask`` -- it reads a FITS file.

    Returns
    -------
    object
        The caller's value, the instrument's, or None.

    Notes
    -----
    This duck-types: it reads attributes off whatever it is handed and
    imports nothing from ``instruments``. The reduction layer stays
    instrument-agnostic; an instrument here is just an object carrying the
    right attribute names.
    """
    if value is not None or instrument is None:
        return value
    got = getattr(instrument, attr, None)
    return got() if (call and callable(got)) else got


def make_masters(frames, keylist, bad_pixel_mask=None, n_sigma=6.0,
                 median_size=7, method=np.nanmedian, min_frames=3,
                 kind="master"):
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
            # Say where and when, not only which key. Splitting skies by
            # pointing makes small groups where a merged master used to hide
            # them, and a dropped sky set has to be visible as a sky set.
            dropped = frame_dict[key]
            log.warning(
                "Only %d frame(s) for key %s, fewer than min_frames=%d, so "
                "no master is built from them: %s%s", len(dropped), key,
                min_frames,
                ", ".join(str(f.get("FILENAME")) for f in dropped[:4]),
                _where_and_when(dropped[0]))
            del frame_dict[key]
        else:
            log.info("Making master for key -> %s, count -> %d", key,
                     len(frame_dict[key]))

    master_frames = {}
    master_masks = {}
    for key, group in frame_dict.items():
        stack = framelist_to_cube(group)
        # Keep the group's header rather than starting empty and adding
        # back only the grouping keys. A master rebuilt from its keylist
        # knows nothing else about how it was taken -- which is why a master
        # flat, grouped without the sampling keywords, could not be matched
        # to a master dark that had them.
        master = Frame(method(stack, axis=0), group[0].header.copy())

        # per-frame sigma clip masks, OR-ed together
        from nirc2pol.utils.imutils import make_sigma_clip_mask

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
            # None means the group's frames did not carry this keyword, so
            # the master must not either: writing it would give the master a
            # keyword its own inputs lacked, and a science frame without it
            # would then fail to match.
            if v is not None:
                master[k] = v

        master["FILENAME"] = group[0].get("FILENAME", "")
        master["NFRAMES"] = len(group)
        master["NSIGMASK"] = n_sigma
        master["MEDSIZE"] = median_size
        master["NPIXMASK"] = int(mask.sum())
        master["MAMEDIAN"] = float(np.median(master.data[~mask]))
        master["MAMEAN"] = float(np.mean(master.data[~mask]))
        master["MASTD"] = float(np.std(master.data[~mask]))

        # A master outlives the reduction that made it -- it is written to
        # disk and matched to science frames later -- so it has to be able
        # to say what built it, and with which pipeline version.
        record_step(master, f"{kind} combination",
                    nframes=len(group), combine=getattr(method, "__name__",
                                                        str(method)),
                    n_sigma=n_sigma, median_size=median_size,
                    masked_pixels=int(mask.sum()),
                    detector_mask=bad_pixel_mask is not None)

        master_frames[key] = master
        master_masks[key] = mask

    return master_frames, master_masks


def make_master_darks(dark_frames, keylist=None, bad_pixel_mask=None,
                      min_frames=3, instrument=None, **kwargs):
    """Build master darks, one per unique combination of ``keylist`` header
        values. Returns ``(master_darks, masks)`` as flat lists.

    Parameters
    ----------
    dark_frames : list of Frame
        Dark frames.
    keylist : list of str, optional
        Grouping keywords; defaults to ``defaults.DARKS_KEYLIST``.
    bad_pixel_mask : ndarray of bool, optional
        Static detector mask. Defaults to ``instrument.bad_pixel_mask()``.
    min_frames : int, optional
        Minimum frames per master.
    instrument : PolarimetryData, optional
        Supplies ``bad_pixel_mask`` when it is not given explicitly.
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
    bad_pixel_mask = _instrument_default(instrument, bad_pixel_mask,
                                         "bad_pixel_mask", call=True)

    # How to tell whether the modulator was in the beam. Bound method rather
    # than a value, so it is taken from the instrument directly.
    in_beam = getattr(instrument, "modulator_in_beam", None)

    if len(dark_frames) < min_frames:
        log.warning("Not enough dark frames found, skipping...")
        return [], []

    masters, masks = make_masters(dark_frames, keylist, kind="dark",
                                  bad_pixel_mask=bad_pixel_mask,
                                  min_frames=min_frames, **kwargs)
    return list(masters.values()), list(masks.values())


def split_polarimetric_flats(flat_frames, modulator_keyword, critical_angles,
                             atol=1.0, in_beam=None):
    """Split flats into (polarimetric, regular), discarding what is neither.

        A flat falls into one of three states, and only two of them are usable.

        **Regular** flats are taken with the modulator *out of the beam*. They
        see the bare optics, so they are ordinary flat fields.

        **Polarimetric** flats are taken with the modulator in the beam,
        stepping through a *complete* set of its critical angles. Combining the
        full set averages the polarized response of the flat source over the
        modulation cycle.

        **Neither** is a flat taken with the modulator in the beam that is not
        part of a complete cycle -- a partial set, or a continuous calibration
        sweep. It cannot serve as a polarimetric flat, because the cycle does
        not close and the source's own polarization stays in it; twilight sky
        is strongly polarized, so a set of 0, 45 and 67.5 has its Q pair cancel
        while the lone U angle does not. It cannot serve as a regular flat
        either, because the modulator was in the beam and its transmission is
        baked in. Such flats are dropped with a warning rather than quietly
        demoted.

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
    in_beam : callable, optional
        ``in_beam(frame)`` returning True when the modulator was in the beam,
        False when parked out of it and None when unknown -- normally
        ``instrument.modulator_in_beam``. Without it every frame is unknown
        and the split falls back to judging on angle alone, which is what this
        did before the modulator position was recorded.

    Returns
    -------
    polarimetric : list of Frame
        Flats forming a complete cycle at the critical angles.
    regular : list of Frame
        Flats taken with the modulator out of the beam, plus frames whose
        modulator position is unknown and which do not look like a cycle.

    Notes
    -----
    Completeness is judged per filter, since flats of different bands are never
    combined into one master; a complete Lp cycle therefore cannot vouch for a
    partial H one.

    "Complete" means every critical angle appears the *same number of times*,
    not merely that each appears once. The cycle average only cancels the flat
    source's polarization when the angles carry equal weight, so a set holding
    two frames at one angle and one at each of the others is lopsided in the
    same way a missing angle is. Surplus frames at over-represented angles are
    set aside -- the earliest are kept, so whole cycles survive in the order
    observed -- leaving the largest balanced set.
    """
    from nirc2pol.utils.angles import angles_match, is_critical_angle

    def band_of(flat):
        """The flat's band, as flat_sort_key reads it."""
        return str(flat.get("FWINAME")
                   or str(flat.get("FILTER", "")).split("+")[0].strip())

    candidates, regular, unusable = [], [], []

    for f in flat_frames:
        state = in_beam(f) if in_beam is not None else None
        angle = f.get(modulator_keyword)
        at_critical = (angle is not None
                       and is_critical_angle(float(angle), critical_angles,
                                             atol))
        if state is False:
            regular.append(f)                  # modulator out: a plain flat
        elif state is True:
            # in the beam: only a complete cycle is usable, and an angle that
            # is not critical at all cannot be part of one
            (candidates if at_critical else unusable).append((f, state))
        else:
            # unknown: judge on angle alone, as before PCUNAME was recorded
            (candidates.append((f, state)) if at_critical
             else regular.append(f))

    by_band = {}
    for f, state in candidates:
        by_band.setdefault(band_of(f), []).append((f, state))

    pol, trimmed = [], []
    for band, group in by_band.items():
        seen = [float(f.get(modulator_keyword)) for f, _ in group]
        missing = [a for a in critical_angles
                   if not any(angles_match(a, s, atol) for s in seen)]

        if missing:
            log.warning("%d %s flat(s) sit at critical angles but the set is "
                        "missing %s, so it is not a complete modulation cycle.",
                        len(group), band or "unfiltered",
                        ", ".join(f"{a:g}" for a in missing))
            for f, state in group:
                # only frames known to have had the modulator in the beam are
                # unusable; an unknown one cannot be shown to carry its
                # transmission, so it stays an ordinary flat as it always did
                (unusable if state is True else regular).append(
                    (f, state) if state is True else f)
            continue

        # Every angle is present. It also has to be present the *same number
        # of times*: the cycle average only cancels the source's polarization
        # if each angle carries equal weight, so 2 x 67.5 against one each of
        # the others is the same fault as a missing angle, just smaller.
        buckets = {}
        for (f, state), angle in zip(group, seen):
            for a in critical_angles:
                if angles_match(a, angle, atol):
                    buckets.setdefault(a, []).append((f, state))
                    break

        counts = {a: len(items) for a, items in buckets.items()}
        keep_per_angle = min(counts.values())

        surplus = []
        for a in sorted(buckets):
            items = buckets[a]
            pol += [f for f, _ in items[:keep_per_angle]]
            surplus += items[keep_per_angle:]

        if surplus:
            log.warning(
                "%s flats cover the cycle unevenly (%s), so %d balanced "
                "cycle(s) are kept and %d surplus frame(s) set aside: %s. "
                "These are sound frames -- they are dropped only because an "
                "angle observed more often than the others weights the cycle "
                "average toward it, and the flat source's own polarization "
                "stops cancelling.",
                band or "Unfiltered",
                ", ".join(f"{a:g}deg x{counts[a]}" for a in sorted(counts)),
                keep_per_angle, len(surplus),
                ", ".join(str(f.get("FILENAME") or "?") for f, _ in surplus))
        trimmed += surplus

    # surplus frames whose modulator position is unknown cannot be shown to
    # carry its transmission, so they stay ordinary flats; the rest are dropped
    regular += [f for f, state in trimmed if state is not True]

    if unusable:
        names = ", ".join(str(f.get("FILENAME") or "?") for f, _ in unusable)
        log.warning(
            "DISCARDING %d flat(s) taken with the modulator IN the beam but "
            "not forming a complete cycle: %s. They are not polarimetric flats "
            "-- the cycle does not close, so the flat source's own "
            "polarization stays in them -- and they are not ordinary flats "
            "either, because the modulator's transmission is baked in. Take "
            "the missing angles, or take flats with the modulator parked out "
            "of the beam.", len(unusable), names)

    order = {id(f): i for i, f in enumerate(flat_frames)}
    pol.sort(key=lambda f: order[id(f)])
    regular.sort(key=lambda f: order[id(f)])
    return pol, regular


def make_flats(flat_frames, master_darks, keylist=None, bad_pixel_mask=None,
               flattype="DOME", min_frames=3, polarimetric=False,
               allow_flat_without_dark=False, **kwargs):
    """Build master flats: median-combine, subtract the best-matching master
        dark, then normalize by the median. ``FLATTYPE`` records which kind of
        flat this is -- ``"DOME"`` or ``"SKY"`` -- and ``POLFLAT`` records
        whether it was a critical-angle polarimetric set. The two are
        independent: a polarimetric flat can be either kind.

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
        Tag written to ``FLATTYPE``: ``"DOME"`` or ``"SKY"``.
    min_frames : int, optional
        Minimum frames per master.
    polarimetric : bool, optional
        Mark the results ``POLFLAT``.
    allow_flat_without_dark : bool, optional
        Build a flat even when no dark matched it. Off by default, because a
        flat that keeps its own dark current is not a flat field: the pedestal
        survives normalization as a multiplicative error that divides straight
        into every science frame. Turning it on tags the result ``+NODARK`` so
        the choice is recorded in the product.
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
        log.warning("Only %d %s%s flat frames, fewer than the %d required, so "
                    "no master is built from them. Note that a mixed set split "
                    "by HWP angle can leave both halves below this threshold.",
                    len(flat_frames), "polarimetric " if polarimetric else "",
                    flattype, min_frames)
        return {}, {}

    master_flats, masks = make_masters(flat_frames, keylist,
                                       kind=f"{flattype.lower()} flat",
                                       bad_pixel_mask=bad_pixel_mask,
                                       min_frames=min_frames, **kwargs)

    for key, flat in master_flats.items():
        flat["POLFLAT"] = bool(polarimetric)
        _, matched_dark = find_closest_dark(flat, master_darks)
        if matched_dark is not None:
            log.info("Subtracting dark from master flat %s", key)
            flat.data -= matched_dark.data
            flat["FLATTYPE"] = flattype
        elif allow_flat_without_dark:
            log.warning("Building master flat %s with NO DARK SUBTRACTED "
                        "because allow_flat_without_dark=True. Its dark "
                        "current survives normalization and will divide into "
                        "every frame this flat calibrates.", key)
            flat["FLATTYPE"] = f"{flattype}+NODARK"
        else:
            raise ValueError(
                f"No dark matches master flat {key}, so it cannot be dark "
                f"subtracted. A flat still holding its dark current is not a "
                f"flat field -- the pedestal survives normalization and "
                f"divides into every science frame. Take darks matching the "
                f"flats' exposure, or pass allow_flat_without_dark=True to "
                f"build it anyway on purpose.")

        flat.data /= flat["MAMEDIAN"]

    return master_flats, masks


def make_master_skies(sky_frames, master_darks, keylist=None,
                      bad_pixel_mask=None, min_frames=3,
                      group_radius_arcsec=60.0, group_gap_minutes=30.0,
                      instrument=None, **kwargs):
    """Build master skies: like flats (dark-subtracted) but *not* normalized,
        since skies are subtracted rather than divided. Returns flat lists.

        Sky frames are split into observing groups by pointing and time
        first, and the group joins the keylist, so sets taken at different
        places stay different masters. Without that, a night with three sky
        sets at three targets median-combines them into one -- the exposure
        settings are identical, which is all the keylist used to look at --
        and no later choice can recover what was averaged away.

    Parameters
    ----------
    sky_frames : list of Frame
        Sky frames.
    master_darks : list of Frame
        Darks to subtract.
    keylist : list of str, optional
        Grouping keywords.
    bad_pixel_mask : ndarray of bool, optional
        Static detector mask. Defaults to ``instrument.bad_pixel_mask()``.
    min_frames : int, optional
        Minimum frames per master. Note that splitting by group makes groups
        smaller than the merged whole, so this can drop a set that used to
        be absorbed into a larger master; it is warned about by pointing.
    group_radius_arcsec : float, optional
        How far the telescope may move within one sky set.
    group_gap_minutes : float, optional
        How long a pause may be within one sky set.
    instrument : PolarimetryData, optional
        Supplies ``bad_pixel_mask`` when it is not given explicitly.
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
    bad_pixel_mask = _instrument_default(instrument, bad_pixel_mask,
                                         "bad_pixel_mask", call=True)

    # Where and when, added to the grouping key. The masters keep OBSGRP
    # afterwards because make_masters repopulates the grouping keywords, and
    # each master inherits its group's first frame, so its RA/DEC/UTC are
    # the group's -- which is what find_closest_sky matches on.
    if sky_frames:
        group_by_pointing(sky_frames, radius_arcsec=group_radius_arcsec,
                          gap_minutes=group_gap_minutes, keyword="OBSGRP")
        keylist = list(keylist) + ["OBSGRP"]

    # How to tell whether the modulator was in the beam. Bound method rather
    # than a value, so it is taken from the instrument directly.
    in_beam = getattr(instrument, "modulator_in_beam", None)

    if len(sky_frames) < min_frames:
        log.warning("Not enough sky frames found, skipping...")
        return [], []

    master_skies, masks = make_masters(sky_frames, keylist, kind="sky",
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


def required_flat_type_for(band, override=None, flat_types=None,
                           default=None):
    """Which flat type a band requires: sky flats in the thermal infrared,
        dome flats in the near infrared. ``override`` (e.g. "SKY") wins, letting
        a user ask for sky flats in JHK.

        This is a requirement rather than a preference -- reducing L' data with
        a dome flat gives a wrong answer that still looks reasonable -- and is
        enforced by :func:`nirc2pol.reduction.calibrate.find_closest_flat`.

    Parameters
    ----------
    band : str
        Observing band, e.g. from ``instruments.nirc2.band_of``.
    override : str, optional
        Force a type, ``"SKY"`` or ``"DOME"``; case-insensitive.
    flat_types : mapping, optional
        Band to required type, from the instrument
        (``instrument.required_flat_types``). Which flat a band needs is a
        property of the instrument, so this code does not carry a table of
        its own.
    default : str, optional
        Type for bands the mapping does not list, from
        ``instrument.default_required_flat_type``.

    Returns
    -------
    str or None
        The required flat type, or None when no mapping was supplied and no
        default given -- meaning the requirement cannot be evaluated and the
        caller should not enforce one.
    """
    if override:
        return str(override).upper()
    key = str(band or "").strip()
    wanted = (flat_types or {}).get(key, default)
    return str(wanted).upper() if wanted else None


def _where_and_when(frame):
    """`` (at RA/Dec ..., 09:03 UTC)`` for a log line, or "" if unknown."""
    from nirc2pol.utils.frame import observed_at, pointing_of

    point, when = pointing_of(frame), observed_at(frame)
    if point is None and when is None:
        return ""
    where = f"RA/Dec {point[0]:.3f} {point[1]:+.3f}" if point else "unknown"
    clock = f", {when:%H:%M} UTC" if when else ""
    return f" (at {where}{clock})"


def describe_flat(flat):
    """One-line description of a master flat: filter, kind, pol, frames.

    Shared by the log lines and the header bookkeeping so a flat is named
    the same way wherever it is mentioned.

    Parameters
    ----------
    flat : Frame
        A master flat.

    Returns
    -------
    str
        e.g. ``"Lp + Wollaston SKY POL (n=4)"``.
    """
    pol = " POL" if flat.get("POLFLAT") else ""
    return (f"{flat.get('FILTER')} {flat.get('FLATTYPE')}{pol} "
            f"(n={flat.get('NFRAMES', 0)})")


def flat_sort_key(flat, required_type=None, flat_types=None,
                  default_flat_type=None, science_bands=None):
    """Sort key implementing the flat preference order.

        0. flats in a band the science data actually uses, when
           ``science_bands`` says which those are. Without it the leading key
           is the flat's own band as a string, so which band heads the list is
           decided by the alphabet -- H before Lp -- which reads like a
           judgement and is not one.
        1. filter, which groups the list; a flat in the wrong filter is never
           eligible in the first place, so this only makes the order readable
        2. the band-required type (sky for L'/M, dome for JHK) before the other
           kind -- the wrong illumination is a worse error than losing the
           critical-angle property
        3. dark-subtracted before ``+NODARK``, so ``SKY`` outranks
           ``SKY+NODARK``
        4. polarimetric (critical-angle) sets before the rest
        5. more frames first

        A ``+NODARK`` flat only exists when someone passed
        ``allow_flat_without_dark=True`` to :func:`make_flats`, which is rare.
        It ranks above the wrong *type* but below anything of its own type
        that has a dark, and above polarimetric: an unsubtracted pedestal
        survives normalization and divides a real photometric error into every
        frame, which costs more than losing the critical-angle property.

    Parameters
    ----------
    flat : Frame
        Master flat to rank.
    required_type : str, optional
        Override the band default, as for :func:`required_flat_type_for`.
    flat_types : mapping, optional
        Band to required type, from ``instrument.required_flat_types``.
    default_flat_type : str, optional
        Fallback for bands the mapping does not list.
    science_bands : iterable of str, optional
        Bands the science frames are in. Flats in one of them sort first.
        Ordering only: which flat a frame actually gets is decided per frame
        by :func:`nirc2pol.reduction.calibrate.find_closest_flat`, where the
        filter must match, so this cannot change what is used -- only what
        the list looks like when read.

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
    wanted = required_flat_type_for(band, required_type, flat_types,
                                   default_flat_type)

    type_rank = 0 if base == wanted else 1
    off_band = bool(science_bands) and band not in set(science_bands)

    return (off_band, band, type_rank, nodark,
            not flat.get("POLFLAT", False), -flat.get("NFRAMES", 0))


def make_master_flats(dome_frames, sky_frames,
                      master_darks, keylist=None, bad_pixel_mask=None,
                      modulator_keyword=None, critical_angles=None,
                      required_flat_type=None, required_flat_types=None,
                      default_required_flat_type=None,
                      allow_flat_without_dark=False, science_bands=None,
                      instrument=None, **kwargs):
    """Build every available kind of flat and return a single ranked list:
        for any science frame, the first matching flat in the list is the best
        available one.

        Pass ``instrument`` and the instrument-derived arguments below are
        filled in from it; anything given explicitly still wins.

        A flat is described by two independent properties. Its *kind* is where
        the light came from -- ``"DOME"`` (the dome screen, lamp on or off) or
        ``"SKY"`` (twilight) -- and that is what the band requirement is about.
        Whether it is *polarimetric* is a separate question, decided by the HWP
        angle, and either kind can be: a polarimetric dome flat and a
        polarimetric sky flat are both ordinary things to have. So when
        ``modulator_keyword`` and ``critical_angles`` are known, both incoming
        lists are split at the critical angles, giving up to four masters from
        two kinds.

        Ordering (see :func:`flat_sort_key`): by filter, then the band-required
        kind — sky flats for L'/M where the dome lamp is swamped by thermal
        background, dome flats for JHK — then polarimetric before not, then the
        set built from the most frames. ``required_flat_type`` ("SKY" or
        "DOME") overrides the band default, e.g. to use sky flats in JHK.

        Ordering is only a preference among *valid* flats; the type requirement
        itself is enforced later, per science frame, by
        :func:`nirc2pol.reduction.calibrate.find_closest_flat`.

        Returns ``(master_flats, masks)`` as flat lists.

    Parameters
    ----------
    dome_frames : list of Frame
        Dome flats, lamp on or off.
    sky_frames : list of Frame
        Twilight sky flats.
    master_darks : list of Frame
        Darks to subtract. Every flat needs one; see
        ``allow_flat_without_dark``.
    keylist : list of str, optional
        Grouping keywords.
    bad_pixel_mask : ndarray of bool, optional
        Static detector mask.
    modulator_keyword : str, optional
        Enables the polarimetric split when given with ``critical_angles``.
    critical_angles : iterable of float, optional
        The instrument's critical angles.
    required_flat_types : mapping, optional
        Band to required flat type, from ``instrument.required_flat_types``.
    default_required_flat_type : str, optional
        Fallback for bands that mapping does not list.
    required_flat_type : str, optional
        Override the band's required type.
    allow_flat_without_dark : bool, optional
        Build flats that no dark matched, tagging them ``+NODARK``. Off by
        default; see :func:`make_flats`.
    science_bands : iterable of str, optional
        Bands the science frames are in, so flats in those bands lead the
        list. Ordering only -- see :func:`flat_sort_key`.
    **kwargs
        Passed through to the individual flat builders.
    instrument : PolarimetryData, optional
        Supplies ``modulator_keyword``, ``critical_angles``,
        ``required_flat_types``, ``default_required_flat_type`` and
        ``bad_pixel_mask`` when they are not given explicitly.
        ``required_flat_type`` is deliberately not among them: that one is a
        per-reduction override, not a property of the instrument.

    Returns
    -------
    flats : list of Frame
        All masters, in preference order.
    masks : list of ndarray
        Matching bad-pixel masks.
    """
    # Everything below is a property of the instrument, so take it from the
    # instrument unless the caller overrode it. Restating these at each call
    # site is how one goes missing, and the one that matters is
    # required_flat_types: without it the band flat-type rule is not
    # enforced at all, which is precisely the silent failure FLATCHK exists
    # to surface.
    modulator_keyword = _instrument_default(instrument, modulator_keyword,
                                            "modulator_keyword")
    critical_angles = _instrument_default(instrument, critical_angles,
                                          "critical_angles")
    required_flat_types = _instrument_default(instrument, required_flat_types,
                                              "required_flat_types")
    default_required_flat_type = _instrument_default(
        instrument, default_required_flat_type, "default_required_flat_type")
    bad_pixel_mask = _instrument_default(instrument, bad_pixel_mask,
                                         "bad_pixel_mask", call=True)

    # How to tell whether the modulator was in the beam. Bound method rather
    # than a value, so it is taken from the instrument directly.
    in_beam = getattr(instrument, "modulator_in_beam", None)

    def build_kind(frames, flattype):
        """Build one kind of flat, split into polarimetric and not."""
        if not frames:
            return [], []

        if modulator_keyword is not None and critical_angles is not None:
            pol_frames, plain_frames = split_polarimetric_flats(
                frames, modulator_keyword, critical_angles, in_beam=in_beam)
            if pol_frames:
                log.info("Found %d polarimetric %s flats at critical angles "
                         "(%d other %s flats)", len(pol_frames),
                         flattype.lower(), len(plain_frames),
                         flattype.lower())
            elif plain_frames:
                log.info("No complete polarimetric %s flat cycle, so ordinary "
                         "%s flats are used", flattype.lower(),
                         flattype.lower())
            else:
                log.warning("No usable %s flats: none form a complete "
                            "polarimetric cycle, and none were taken with the "
                            "modulator out of the beam.", flattype.lower())
            groups = [(pol_frames, True), (plain_frames, False)]
        else:
            groups = [(frames, False)]

        built, built_masks = [], []
        for group_frames, polarimetric in groups:
            if not group_frames:
                continue
            masters, masks = make_flats(
                group_frames, master_darks, keylist=keylist,
                bad_pixel_mask=bad_pixel_mask, flattype=flattype,
                polarimetric=polarimetric,
                allow_flat_without_dark=allow_flat_without_dark, **kwargs)
            built += list(masters.values())
            built_masks += list(masks.values())
        return built, built_masks

    # Concatenated, not merged. make_flats returns dicts keyed by the grouping
    # key, which does not include FLATTYPE or POLFLAT, so merging them dropped
    # a sky flat whenever a dome flat shared its filter and exposure settings
    # -- and at L' the sky flat is the one the band requires. Nothing here is a
    # duplicate: sort_frames puts each raw frame in exactly one bucket, and the
    # polarimetric split partitions each bucket.
    dome_flats, dome_masks = build_kind(dome_frames, "DOME")
    sky_flats, sky_masks = build_kind(sky_frames, "SKY")

    flats = dome_flats + sky_flats
    flats.sort(key=lambda f: flat_sort_key(
        f, required_flat_type, required_flat_types,
        default_required_flat_type, science_bands))

    if flats:
        # An inventory, not a decision. Which flat a frame gets is settled
        # per frame in find_closest_flat, on a filter that must match; this
        # order only decides which of several *equally matching* flats is
        # reached first, and reads as a ranking of the night if it is not
        # said plainly.
        log.info("%d master flat(s) available: %s", len(flats),
                 ", ".join(describe_flat(f) for f in flats[:6]))

    masks = dome_masks + sky_masks
    return flats, masks


def make_master_masks(*mask_lists):
    """Combine all master masks, OR-ing together those with the same shape.

        Returns a dict mapping shape -> combined boolean mask, which is what
        :func:`nirc2pol.reduction.calibrate.reduce_frame` expects for its ``masks``
        argument.

    Parameters
    ----------
    *mask_lists
        Any number of lists of boolean masks.

    Returns
    -------
    dict
        Maps array shape to the OR of every mask of that shape, which is what
        :func:`nirc2pol.reduction.calibrate.reduce_frame` expects for ``masks``.
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

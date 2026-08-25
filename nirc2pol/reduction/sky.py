"""Sky / dither / background subtraction, as its own pipeline stage.

Runs on pre-processed (dark-subtracted, flat-divided) frames. Two options:

Options (SPIE Sec. 3.2):

- :func:`subtract_annulus_background` — annulus median around the star,
  usually sufficient for J / H / Kp without dedicated sky frames.
- :func:`subtract_dither_background` — the pipeline's dither stage:
  subtracts, from each frame, the nearest in time at a different dither
  position and the same HWP angle. Needed in L' where the thermal
  background is rapidly varying and spatially structured.
- :func:`subtract_dither_pairs` — the simpler primitive underneath the idea:
  two lists of frames, paired by order.
- :func:`subtract_sky_frames` — subtract a matched master sky frame
  (built from sky flats).
- :func:`subtract_mean_background` — subtract the mean level of an empty
  box region, per beam.
"""

from __future__ import annotations

import logging

import numpy as np

from nirc2pol.utils.imutils import argquantile, make_annulus_mask
from .calibrate import find_closest_sky

log = logging.getLogger(__name__)


def subtract_annulus_background(data, r_in, r_out, center=None):
    """Subtract the median level of an annulus around the star (located by
        quantile peak if ``center = (cy, cx)`` is not given). Works on a 2D
        image or each plane of a stack. Returns a new array.

    Parameters
    ----------
    data : array_like
        2D image or a stack; each plane is treated separately.
    r_in, r_out : float
        Annulus radii in pixels.
    center : tuple of float, optional
        ``(cy, cx)``; located by quantile peak when not given.

    Returns
    -------
    ndarray
        A new array with the annulus median removed. The input is unchanged.

    Notes
    -----
    Subtracts one scalar, so it assumes the background is flat across the
    frame. That holds in JHK but not at L' or M, where the thermal pedestal is
    large and structured -- see
    ``instruments.nirc2.check_background_choice``.
    """
    data = np.asarray(data, dtype=float).copy()

    def _sub(img):
        """Subtract this plane's background in place.

        Subtract this plane background in place.
        """
        c = center if center is not None else argquantile(img, 0.9999)
        annulus = make_annulus_mask(img.shape, r_in, r_out, center=c)
        img -= np.nanmedian(img[annulus])

    if data.ndim == 2:
        _sub(data)
    else:
        for plane in data.reshape(-1, *data.shape[-2:]):
            _sub(plane)
    return data


def subtract_dither_pairs(frames_a, frames_b):
    """Pairwise dither subtraction: subtract each frame at dither position B
        from its counterpart at position A (matched by order). Returns new
        frames from position A with the B background removed.

    Parameters
    ----------
    frames_a, frames_b : list of Frame
        Frames at the two dither positions, paired by order.

    Returns
    -------
    list of Frame
        New frames from position A with B subtracted, each recording the
        subtracted file in ``DITHSUB``. Warns and pairs up to the shorter list
        if the lengths differ.

    Notes
    -----
    The cleanest background removal at L' and M: the pedestal is measured
    through the same optics moments apart, so its structure cancels rather
    than being approximated by a single number.
    """
    if len(frames_a) != len(frames_b):
        log.warning("Dither position lists differ in length (%d vs %d); "
                    "pairing up to the shorter one",
                    len(frames_a), len(frames_b))

    out = []
    for fa, fb in zip(frames_a, frames_b):
        result = fa.copy()
        result.data = fa.data - fb.data
        result["DITHSUB"] = fb.get("FILENAME", "")
        out.append(result)
    return out


def dither_positions(frames, tolerance_arcsec=2.0):
    """Group frames by where the telescope was nodded to.

    Parameters
    ----------
    frames : list of Frame
        Frames to group.
    tolerance_arcsec : float, optional
        Used **only** on the fallback path. See the notes.

    Returns
    -------
    dict
        Position label -> the frames taken there.

    Notes
    -----
    Positions come from the distinct ``RAOFF`` / ``DECOFF`` pairs, compared
    exactly, because those are commanded offsets and carry no jitter: on the
    2025-12-06 L' data the actual pointing within one commanded offset varies
    by 0.00 arcsec. A tolerance there would only risk merging two positions
    that are genuinely different, and a dither throw is small -- 3 arcsec on
    that data -- so there is very little room before it does.

    ``tolerance_arcsec`` therefore applies only when those keywords are
    missing and the actual ``RA`` / ``DEC`` has to be used instead, where
    guiding really does move between frames.

    This is deliberately not :func:`nirc2pol.utils.frame.group_by_pointing`,
    which walks frames in time order and starts a new group whenever the
    telescope moves: an ABAB dither would come out as four groups rather than
    two positions.
    """
    from nirc2pol.utils.frame import pointing_of

    groups = {}
    fallback = []
    for frame in frames:
        ra_off, dec_off = frame.get("RAOFF"), frame.get("DECOFF")
        if ra_off is None or dec_off is None:
            fallback.append(frame)
            continue
        groups.setdefault((float(ra_off), float(dec_off)), []).append(frame)

    if fallback:
        from nirc2pol.utils.angles import small_angle_distance

        log.warning(
            "%d frame(s) have no RAOFF/DECOFF, so their dither position is "
            "taken from RA/DEC within %.1f arcsec instead.",
            len(fallback), tolerance_arcsec)
        anchors = []
        for frame in fallback:
            point = pointing_of(frame)
            if point is None:
                groups.setdefault("unknown", []).append(frame)
                continue
            for label, anchor_point in anchors:
                if (small_angle_distance(anchor_point, point) * 3600.0
                        <= tolerance_arcsec):
                    groups.setdefault(label, []).append(frame)
                    break
            else:
                label = f"radec{len(anchors)}"
                anchors.append((label, point))
                groups.setdefault(label, []).append(frame)

    return groups


def _dither_throw_px(positions, instrument):
    """Smallest separation between dither positions, in pixels.

    Parameters
    ----------
    positions : dict
        Position label -> frames, as :func:`dither_positions` returns. Labels
        are ``(ra_off, dec_off)`` tuples in arcsec on the commanded path.
    instrument : PolarimetryData
        Supplies ``plate_scale`` in arcsec per pixel.

    Returns
    -------
    float or None
        The throw, or None when it cannot be derived -- fallback labels carry
        no offsets, and an instrument may not declare a plate scale.

    Notes
    -----
    Only the *magnitude* is wanted, so the detector rotation does not enter:
    a separation in arcsec divided by the plate scale is a separation in
    pixels whatever angle the field sits at. Checked against 2025-12-06 Io,
    where a measured 290-316 px ghost separation is 2.9-3.1 arcsec, the
    commanded throw.

    The *smallest* separation, because the crop has to clear the nearest
    ghost, not the average one.
    """
    scale = getattr(instrument, "plate_scale", None)
    if not scale:
        return None
    offsets = [label for label in positions if isinstance(label, tuple)]
    if len(offsets) < 2:
        return None

    seps = [np.hypot(a[0] - b[0], a[1] - b[1])
            for i, a in enumerate(offsets) for b in offsets[i + 1:]]
    seps = [s for s in seps if s > 0]
    return float(min(seps) / scale) if seps else None


def subtract_dither_background(frames, instrument, tolerance_arcsec=2.0,
                               critical_angles=None, atol=1.0):
    """Subtract the sky measured at another dither position, per frame.

    The cleanest background removal at L' and M: the pedestal is measured
    through the same optics moments apart, so its structure cancels rather
    than being approximated by a single number.

    Parameters
    ----------
    frames : list of Frame
        Pre-processed science frames -- dark-subtracted and flat-divided,
        before the Wollaston beams are cut out.
    instrument : PolarimetryData
        Supplies the modulator angle of each frame and the critical angles.
    tolerance_arcsec : float, optional
        Fallback position tolerance; see :func:`dither_positions`.
    critical_angles : iterable of float, optional
        Defaults to the instrument's.
    atol : float, optional
        How close a modulator angle must be to a critical angle to count as
        it, in degrees.

    Returns
    -------
    list of Frame
        New frames with the background removed, each recording the file
        subtracted in ``DITHSUB``. Frames that could not be paired are
        returned unchanged, without that keyword.

    Raises
    ------
    ValueError
        When every frame is at the same dither position, so there is no sky
        to subtract. Raised rather than returned untouched: choosing
        ``dither`` and quietly getting no background subtraction at all is
        the failure this exists to prevent.

    Notes
    -----
    **Pairs are matched within one HWP angle.** Subtracting a 45 degree frame
    from a 0 degree frame differences two polarization states, not two skies,
    and destroys the signal it is supposed to clean. The angle is matched to
    the nearest *critical* angle rather than compared raw, because the same
    nominal position reads 45.002 on one frame and 45.0025 on the next.

    **Each frame is paired with the single nearest in time** at a different
    position, so an uneven split keeps everything. With 6 frames at one
    position and 3 at the other -- which is what the 2025-12-06 standard-star
    sequence has, at every angle -- all 9 come out subtracted, where pairing
    the two lists by order would keep 3. The price is that a sky frame used
    by two science frames correlates their noise. It buys a 28-60 second
    match against a 20 minute sequence, which is the whole point of dithering
    at L' in the first place.
    """
    from nirc2pol.utils.angles import angles_match
    from nirc2pol.utils.frame import observed_at

    frames = list(frames)
    if not frames:
        return frames

    positions = dither_positions(frames, tolerance_arcsec)
    if len(positions) < 2:
        only = next(iter(positions), "unknown")
        raise ValueError(
            f"background_method='dither' needs frames at more than one "
            f"position, but all {len(frames)} are at {only}. Either these "
            f"frames were not dithered -- in which case use mean_box or "
            f"annulus -- or the positions were merged: a dither throw is "
            f"only a few arcsec, so a tolerance near it collapses them.")

    where = {id(f): label for label, group in positions.items() for f in group}
    throw_px = _dither_throw_px(positions, instrument)
    when = {id(f): observed_at(f) for f in frames}
    critical_angles = (critical_angles if critical_angles is not None
                       else instrument.critical_angles)

    # Bucket by critical angle, so a pair is always two skies and never two
    # polarization states.
    buckets = {}
    for frame in frames:
        angle = instrument.modulator_angle(frame)
        for critical in critical_angles:
            if angles_match(angle, critical, atol):
                buckets.setdefault(critical, []).append(frame)
                break
        else:
            buckets.setdefault(None, []).append(frame)

    # Keyed by input frame, not appended: the buckets below walk angle by
    # angle, and match_modulator_cycles walks frames in TIME order. Returning
    # them grouped by angle presents four frames at the same angle in a row,
    # which reads as a repeat before the cycle is complete and drops every
    # partial group -- no cycles at all, from a subtraction that worked.
    done, subtracted, unpaired = {}, 0, 0
    for critical, group in sorted(buckets.items(),
                                  key=lambda kv: (kv[0] is None, kv[0])):
        labels = {where[id(f)] for f in group}
        if len(labels) < 2:
            log.warning(
                "Modulator angle %s has %d frame(s), all at dither position "
                "%s, so none of them has a sky to subtract and they are left "
                "with their background in.", critical, len(group),
                next(iter(labels)))
            for frame in group:
                done[id(frame)] = frame
            unpaired += len(group)
            continue

        for frame in group:
            others = [o for o in group if where[id(o)] != where[id(frame)]]
            here = when[id(frame)]
            if here is not None and all(when[id(o)] is not None
                                        for o in others):
                partner = min(others,
                              key=lambda o: abs((when[id(o)] - here)
                                                .total_seconds()))
            else:
                partner = others[0]

            result = frame.copy()
            result.data = frame.data - partner.data
            result["DITHSUB"] = (str(partner.get("FILENAME", "")),
                                 "dither frame subtracted as sky")
            # How far the ghost sits from the source, in pixels. Recorded
            # rather than recomputed because it is what sizes the crop that
            # removes the ghost, and a number that travels with the data can
            # be checked against the image it describes.
            if throw_px is not None:
                result["DITHSEP"] = (float(throw_px),
                                     "dither throw [px]: ghost offset")
            done[id(frame)] = result
            subtracted += 1

    log.info("Dither subtraction: %d frame(s) from %d position(s) across %d "
             "angle(s); %d subtracted, %d left unpaired",
             len(frames), len(positions), len(buckets), subtracted, unpaired)
    return [done[id(f)] for f in frames]


def subtract_sky_frames(frames, master_skies):
    """Subtract the best-matching master sky from each frame (in place on
        copies). Frames with no matching sky are passed through unchanged, with
        the SKYSUB header keyword recording what happened.

    Parameters
    ----------
    frames : iterable of Frame
        Science frames.
    master_skies : list of Frame
        Candidate master skies.

    Returns
    -------
    list of Frame
        Copies with the best-matching sky subtracted where one was found.
        ``SKYSUB`` records what happened for each frame, so a frame that was
        passed through unchanged is identifiable afterwards.
    """
    out = []
    for frame in frames:
        result = frame.copy()
        result["SKYSUB"] = False
        _, matched_sky = find_closest_sky(frame, master_skies)
        if matched_sky is not None:
            result.data -= matched_sky.data
            result["SKYSUB"] = True
        out.append(result)
    return out


def subtract_mean_background(data, box=None):
    """Subtract the mean of a background box from an image (or from each
        plane of a stack of images).

        ``box`` is ``(ylow, yhigh, xlow, xhigh)``; defaults to the whole image.
        Returns a new array.

    Parameters
    ----------
    data : array_like
        2D image or a stack; each plane is treated separately.
    box : tuple of int, optional
        ``(ylow, yhigh, xlow, xhigh)`` region to average; the whole image by
        default.

    Returns
    -------
    ndarray
        A new array with the mean removed. The input is unchanged.
    """
    data = np.asarray(data, dtype=float).copy()

    def _sub(img):
        """Subtract this plane background in place."""
        if box is None:
            img -= np.nanmean(img)
        else:
            ylow, yhigh, xlow, xhigh = box
            img -= np.nanmean(img[ylow:yhigh, xlow:xhigh])

    if data.ndim == 2:
        _sub(data)
    else:
        for plane in data.reshape(-1, *data.shape[-2:]):
            _sub(plane)
    return data


def background_stages(method):
    """Normalise a background setting into an ordered tuple of stages.

    Parameters
    ----------
    method : str or None or sequence
        One stage (``"dither"``), several in the order they are applied
        (``["dither", "annulus"]``), or None for no subtraction.

    Returns
    -------
    tuple
        The stages. ``()`` when nothing is to be done.

    Raises
    ------
    ValueError
        If a stage is repeated, if None is mixed with real stages, or if
        ``"dither"`` is present but not first.

    Notes
    -----
    Chaining exists because the two kinds of subtraction do different jobs.
    ``dither`` differences two exposures taken seconds apart, which removes
    the thermal pedestal *and its structure* -- the only thing that can, at
    L-prime. What it leaves is a residual, because the sky did change a
    little between the two, and that residual is a roughly constant offset
    per beam. ``annulus`` or ``mean_box`` after it removes that.

    The residual is small and matters anyway. On the 2025-12-06 standard,
    after a clean dither subtraction, U still carried a detector-scale
    gradient worth +22 ADU/px at the star -- nothing beside a 1.2e5 core,
    but an annulus at r=60-80 holds ~9000 pixels and sums it into more
    signal than the star has out there, which ran the measured polarization
    from 0.9% to 4.1%.

    ``"dither"`` must come first because it is not a beam-level operation at
    all: it happens in :func:`nirc2pol.recipe.run`, on whole frames, before
    the beams are cut out. Writing it anywhere else would describe an order
    that cannot happen, so that is an error rather than a silent
    reordering.
    """
    if method is None:
        return ()
    if isinstance(method, str):
        stages = (method,)
    else:
        stages = tuple(method)

    if not stages:
        return ()
    if any(s is None for s in stages):
        if len(stages) > 1:
            raise ValueError(
                f"background_method {list(stages)!r} mixes none with real "
                "stages. Use none on its own to mean no subtraction, or list "
                "only the stages you want.")
        return ()
    if len(set(stages)) != len(stages):
        raise ValueError(
            f"background_method {list(stages)!r} repeats a stage. Running "
            "one twice removes nothing the first pass left.")
    if "dither" in stages and stages[0] != "dither":
        raise ValueError(
            f"background_method {list(stages)!r} puts 'dither' after another "
            "stage, but dither is applied to whole frames before the beams "
            "are cut out, so it always runs first. Write it first, or drop "
            "it.")
    return stages


def subtract_background(stack, method, box=None, annulus=None):
    """Apply the background subtraction named by ``method``.

    Parameters
    ----------
    stack : ndarray
        Image or beam stack to correct.
    method : str or None or sequence
        Which subtraction to apply, or several in order -- see
        :func:`background_stages`. None returns the input untouched, which
        is how a caller says the omission is deliberate.
    box : tuple of int, optional
        ``(ylow, yhigh, xlow, xhigh)`` for ``"mean_box"``.
    annulus : tuple of float, optional
        ``(r_inner, r_outer)`` in pixels for ``"annulus"``.

    Returns
    -------
    ndarray
        The corrected data, or the input unchanged for ``None`` and
        ``"dither"``.

    Raises
    ------
    ValueError
        If a method needs parameters it was not given, or is not recognised.

    Notes
    -----
    ``"dither"`` is a no-op here by design: dither pairs are differenced at
    frame level by :func:`subtract_dither_background`, before the beams are
    cut out, so there is nothing left to do per beam. It is still accepted
    here, and still worth naming in a chain, so that one setting describes
    the whole background treatment rather than half of it.

    Every sky subtraction the pipeline knows about is in this module,
    including the choice between them. Instruments carry which method to use
    and its parameters, since those are per-dataset settings, but none of
    them implements a subtraction.
    """
    for stage in background_stages(method):
        if stage == "mean_box":
            if box is None:
                raise ValueError("background method 'mean_box' requires "
                                 "box=(ylow, yhigh, xlow, xhigh)")
            stack = subtract_mean_background(stack, box=box)
        elif stage == "annulus":
            if annulus is None:
                raise ValueError("background method 'annulus' requires "
                                 "annulus=(r_inner, r_outer)")
            stack = subtract_annulus_background(stack, *annulus)
        elif stage == "dither":
            continue          # done at frame level, before the beams are cut
        else:
            raise ValueError(f"Unknown background method {stage!r}")
    return stack

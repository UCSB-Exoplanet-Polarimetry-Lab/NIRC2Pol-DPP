"""Lightweight FITS frame container and I/O.

A ``Frame`` is just image data plus its FITS header, mirroring AIR.jl's use of
``AstroImage``. Header keywords are accessed dict-style::

    frame = Frame.load("n0001.fits")
    itime = frame["ITIME"]
    frame["RED-FN"] = "reduced_0001.fits"

Data is always stored as float64 numpy arrays in the standard numpy/astropy
convention: ``data[y, x]``, i.e. axis 0 is NAXIS2 (rows) and axis 1 is NAXIS1
(columns).
"""

from __future__ import annotations

import os
import re

import logging
from datetime import date as _date

import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)


class Frame:
    """A 2D image and its FITS header, kept together.

    The unit every pipeline stage passes around. Header access is
    dict-style, mirroring AIR.jl's ``AstroImage``, so ``frame["ITIME"]``
    reads a keyword and ``frame["DARKSUB"] = True`` writes one.

    Attributes
    ----------
    data : ndarray
        Image data, always float. Integer FITS data is converted on load so
        that arithmetic never silently truncates.
    header : astropy.io.fits.Header
        The FITS header. Empty rather than None when not supplied.
    """

    def __init__(self, data, header=None):
        """Wrap an array and a header.

        Parameters
        ----------
        data : array_like
            Image data; converted to float.
        header : astropy.io.fits.Header or dict, optional
            Header to attach. A plain dict is accepted and converted.
        """
        self.data = np.asarray(data, dtype=float)
        self.header = fits.Header(header) if header is not None else fits.Header()

    # dict-style header access, like AstroImage in AIR.jl
    def __getitem__(self, key):
        """Read a header keyword. Raises ``KeyError`` if absent."""
        return self.header[key]

    def __setitem__(self, key, value):
        """Write a header keyword; ``value`` may be a ``(value, comment)``
        tuple, as astropy allows."""
        self.header[key] = value

    def __contains__(self, key):
        """True if the header carries this keyword."""
        return key in self.header

    def get(self, key, default=None):
        """Read a header keyword, returning ``default`` when it is absent.

        Parameters
        ----------
        key : str
            Keyword to read.
        default : object, optional
            Value returned when the keyword is missing.

        Returns
        -------
        object
            The keyword value, or ``default``.
        """
        return self.header.get(key, default)

    @property
    def shape(self):
        """Shape of the image data, ``(ny, nx)``."""
        return self.data.shape

    def copy(self):
        """Deep copy: both the data and the header are copied, so edits to
        the result cannot reach back into the original."""
        return Frame(self.data.copy(), self.header.copy())

    @classmethod
    def load(cls, filename, hdu=0):
        """Read a Frame from a FITS file.

        Parameters
        ----------
        filename : str
            Path to read. Gzipped files are handled by astropy.
        hdu : int, optional
            Which HDU to take data and header from.

        Returns
        -------
        Frame
            The loaded frame. ``FILENAME`` is stamped into the header from
            the path when the file does not already carry one, so a frame
            can be traced back to disk after being passed around.
        """
        with fits.open(filename) as hdul:
            frame = cls(np.asarray(hdul[hdu].data, dtype=float), hdul[hdu].header)
        if "FILENAME" not in frame.header:
            frame["FILENAME"] = os.path.basename(str(filename))
        return frame

    def save(self, filename, overwrite=True):
        """Write the Frame to a FITS file.

        Parameters
        ----------
        filename : str
            Path to write.
        overwrite : bool, optional
            Replace an existing file.

        Notes
        -----
        Falls back to :func:`scrub_header` when astropy refuses the header.
        Some NIRC2 headers carry malformed CONTINUE cards that cannot be
        written back out; without the fallback no product from those nights
        could be saved at all. The fallback warns, since it drops the
        offending cards.
        """
        try:
            fits.PrimaryHDU(
                data=self.data,
                header=without_structural_cards(self.header)).writeto(
                filename, overwrite=overwrite)
        except fits.verify.VerifyError:
            # some NIRC2 headers carry malformed CONTINUE cards
            log.warning("Malformed header cards in %s; writing a scrubbed "
                        "header", filename)
            fits.PrimaryHDU(
                data=self.data,
                header=scrub_header(
                    without_structural_cards(self.header))).writeto(
                filename, overwrite=overwrite)

    def __repr__(self):
        """Short representation: filename and data shape."""
        name = self.get("FILENAME", "<no filename>")
        return f"Frame({name}, shape={self.data.shape})"


# Cards that describe the HDU's structure rather than the observation.
_STRUCTURAL_CARDS = ("SIMPLE", "XTENSION", "BITPIX", "NAXIS", "EXTEND",
                     "PCOUNT", "GCOUNT", "BSCALE", "BZERO", "BLANK",
                     "CHECKSUM", "DATASUM")


def without_structural_cards(header):
    """Header copy with the cards astropy manages itself removed.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header to clean.

    Returns
    -------
    astropy.io.fits.Header
        A copy without SIMPLE, BITPIX, NAXIS/NAXISn, EXTEND, PCOUNT, GCOUNT
        and the checksums.

    Notes
    -----
    Only used when writing. The keywords stay in the in-memory header
    because NAXIS1 and NAXIS2 are matching keywords -- a dark has to agree
    with its frame on readout size -- and astropy restores them on load.
    """
    clean = header.copy()
    for key in list(clean):
        if key in _STRUCTURAL_CARDS or (key.startswith("NAXIS")
                                        and key[5:].isdigit()):
            del clean[key]
    return clean


def scrub_header(header):
    """Drop malformed CONTINUE chunks from a FITS header.

    Astropy stores a card plus its CONTINUEs as one consolidated card whose
    ``_image`` is the concatenation of 80-byte chunks. Some NIRC2 headers
    (e.g. the 2026-05-26 reduction) contain a CONTINUE whose value field is
    not a quoted string, which astropy refuses to write back out. This walks
    the chunks of each card and keeps everything up to the first bad
    CONTINUE, so a product can still be saved.
    """
    clean = fits.Header()
    for card in header._cards:
        image = card._image
        if image is None:            # card built in memory, not from a file
            try:
                clean.append(card, end=True)
            except Exception:
                pass
            continue
        if len(image) > 80:
            chunks = [image[i:i + 80] for i in range(0, len(image), 80)]
            kept = [chunks[0]]
            for chunk in chunks[1:]:
                if chunk.startswith("CONTINUE"):
                    value = chunk[8:].lstrip()
                    if not value.startswith("'"):
                        break
                kept.append(chunk)
            image = "".join(kept)
        try:
            clean.append(fits.Card.fromstring(image), end=True)
        except Exception:
            pass
    return clean


def pointing_of(frame):
    """Where the telescope was pointing, in degrees.

    Parameters
    ----------
    frame : Frame or Header
        Anything with ``RA`` and ``DEC``.

    Returns
    -------
    tuple of float, or None
        ``(ra_deg, dec_deg)``, or None when either is missing or unreadable.

    Notes
    -----
    NIRC2 stores ``RA`` sexagesimally **in hours** (``'04:55:45.53'``) and
    ``DEC`` in degrees. :func:`nirc2pol.utils.angles.sexagesimal_to_degrees`
    returns whatever unit went in, so the RA is multiplied by 15 here -- once,
    in one place, rather than at each call site where forgetting it would
    understate every separation by that factor.
    """
    from nirc2pol.utils.angles import sexagesimal_to_degrees

    ra, dec = frame.get("RA"), frame.get("DEC")
    if ra is None or dec is None:
        return None
    try:
        return (sexagesimal_to_degrees(ra) * 15.0,
                sexagesimal_to_degrees(dec))
    except (TypeError, ValueError):
        return None


def observed_at(frame):
    """When a frame was taken, as a datetime.

    Parameters
    ----------
    frame : Frame or Header
        Anything with ``DATE-OBS`` and ``UTC``.

    Returns
    -------
    datetime.datetime, or None
        None when either keyword is missing or unreadable.

    Notes
    -----
    NIRC2 has no ``MJD-OBS``: the date and the time of day are separate
    keywords, and ``UTC`` is sexagesimal hours.
    """
    import datetime

    from nirc2pol.utils.angles import sexagesimal_to_degrees

    date, utc = frame.get("DATE-OBS"), frame.get("UTC")
    if not date or utc is None:
        return None
    try:
        day = parse_date_obs(str(date))
        hours = sexagesimal_to_degrees(utc)
    except (TypeError, ValueError):
        return None
    return (datetime.datetime.combine(day, datetime.time())
            + datetime.timedelta(hours=float(hours)))


def group_by_pointing(frames, radius_arcsec=60.0, gap_minutes=30.0,
                      keyword="OBSGRP"):
    """Number frames into observing groups, and stamp the number on each.

    A new group starts when the telescope has moved further than
    ``radius_arcsec`` from where the current group started, or when more than
    ``gap_minutes`` has passed since the previous frame. Both matter: two sky
    sets visited back to back are only told apart by position, and one
    pointing revisited hours later is only told apart by time.

    Parameters
    ----------
    frames : list of Frame
        Frames in the order observed. Sorted by time internally when every
        frame carries one, so a caller need not have sorted them.
    radius_arcsec : float, optional
        How far the telescope may move within one group.
    gap_minutes : float, optional
        How long a pause may be within one group.
    keyword : str, optional
        Header keyword the group index is written to.

    Returns
    -------
    dict
        Group index -> the frames in it, in order.

    Notes
    -----
    Frames with no readable pointing all land in one group and are warned
    about: they cannot be told apart, and silently giving each its own group
    would fragment a night into single-frame sets that then fall below any
    minimum-frames rule.
    """
    from nirc2pol.utils.angles import small_angle_distance

    if not frames:
        return {}

    times = {id(f): observed_at(f) for f in frames}
    if all(times[id(f)] is not None for f in frames):
        frames = sorted(frames, key=lambda f: times[id(f)])

    unplaced = [f for f in frames if pointing_of(f) is None]
    if unplaced:
        log.warning(
            "%d of %d frame(s) have no readable RA/DEC, so they cannot be "
            "grouped by where they were taken and are kept together: %s",
            len(unplaced), len(frames),
            ", ".join(str(f.get("FILENAME")) for f in unplaced[:4]))

    groups = {}
    index = -1
    anchor_point = None
    previous_time = None

    for frame in frames:
        point = pointing_of(frame)
        when = times[id(frame)]

        moved = (anchor_point is not None and point is not None
                 and small_angle_distance(anchor_point, point) * 3600.0
                 > radius_arcsec)
        paused = (previous_time is not None and when is not None
                  and (when - previous_time).total_seconds() / 60.0
                  > gap_minutes)

        if index < 0 or moved or paused:
            index += 1
            anchor_point = point
        elif anchor_point is None:
            anchor_point = point

        frame[keyword] = (index, "observing group: pointing and time")
        groups.setdefault(index, []).append(frame)
        if when is not None:
            previous_time = when

    if len(groups) > 1:
        log.info("%d frame(s) fall into %d observing group(s) by pointing "
                 "and time", len(frames), len(groups))
    return groups


def read_headers(paths):
    """Headers only, without reading a single pixel.

    For the questions that are answered by the header -- what band is this
    night, what was the telescope pointing at -- where loading the data would
    cost hundreds of megabytes to look at a few keywords.

    Parameters
    ----------
    paths : iterable of str
        FITS files.

    Returns
    -------
    list of Header
        In the order given. Files that cannot be opened are skipped with a
        warning rather than stopping a reduction over one bad file.
    """
    from astropy.io import fits

    headers = []
    for path in paths:
        try:
            headers.append(fits.getheader(path))
        except Exception as exc:
            log.warning("Could not read the header of %s: %s", path, exc)
    return headers


def load_frames(frame_paths, rejects=()):
    """Load several FITS files, skipping rejected ones.

    Parameters
    ----------
    frame_paths : iterable of str
        Paths to load.
    rejects : container, optional
        Anything supporting ``in``: a list of filenames, or the
        ``{filename: reason}`` mapping from
        :func:`nirc2pol.utils.paths.load_rejects`. Matching is on the *basename*, so
        a reject list is portable between machines.

    Returns
    -------
    list of Frame
        The frames that were not rejected, in the order given.
    """
    frames = []
    for fn in frame_paths:
        if os.path.basename(str(fn)) in rejects:
            continue
        frames.append(Frame.load(fn))
    return frames


def frame_number(path_or_frame):
    """Observation number from a frame's filename, e.g. ``n0932.fits.gz`` -> 932.

    Parameters
    ----------
    path_or_frame : str or Frame
        A path, or a Frame carrying ``FILENAME``.

    Returns
    -------
    int or None
        The first run of digits in the basename, or None if there is none.
    """
    if hasattr(path_or_frame, "header"):
        name = str(path_or_frame.get("FILENAME") or "")
    else:
        name = str(path_or_frame)
    match = re.search(r"(\d+)", os.path.basename(name))
    return int(match.group(1)) if match else None


def _normalize_target(name):
    """Fold a target name for comparison: case, spaces, underscores, hyphens.

    ``"AB_Aur"``, ``"AB Aur"`` and ``"ab-aur"`` all become ``"abaur"``, so a
    label written one way in a script finds a header written another way.
    """
    return re.sub(r"[\s_\-]+", "", str(name)).lower()


def _as_ranges(frame_range):
    """Normalise a frame_range argument to a list of inclusive pairs.

    Accepts a single ``(first, last)`` or several, ``[(857, 900), (915, 930)]``.
    A night is often observed in runs broken by a slew or a filter change, so
    the plural form is the common case; the singular is kept because most
    selections are one run.
    """
    if frame_range is None:
        return None

    pairs = list(frame_range)
    if pairs and isinstance(pairs[0], (int, float)):
        pairs = [pairs]                      # a bare (first, last)

    ranges = []
    for pair in pairs:
        first, last = pair
        if first > last:
            raise ValueError(
                f"frame_range {(first, last)} runs backwards: the first frame "
                f"number must not exceed the last. Pass several ranges as a "
                f"list of pairs, e.g. [(857, 900), (915, 930)].")
        ranges.append((int(first), int(last)))
    return ranges


def in_frame_range(path_or_frame, frame_range):
    """Is this frame's observation number inside any of ``frame_range``?

    The path-level counterpart to :func:`select_frames`. That one reads
    headers, so it needs frames already loaded; this reads only the filename,
    so it can narrow a glob *before* anything is opened -- which is what a
    ``raw_range`` is for, when the folder holds more than you want to read.

    Accepts the same shapes as ``select_frames(frame_range=...)``: one
    ``(first, last)`` or several. A file with no number in its name is not in
    any range.
    """
    n = frame_number(path_or_frame)
    if n is None:
        return False
    return any(lo <= n <= hi for lo, hi in _as_ranges(frame_range))


def select_frames(frames, target=None, frame_range=None,
                  target_keyword="TARGNAME"):
    """Narrow a list of frames to the ones a reduction should cover.

    This is about *scope*, not quality: masters and reduced frames are built
    for a whole night, and this picks the subset that goes on to become
    science products. Frames that are simply *bad* belong in the reject file
    instead, where they carry a reason and persist across runs -- see
    :func:`nirc2pol.utils.paths.record_reject`.

    Parameters
    ----------
    frames : list of Frame
        Frames to choose from.
    target : str, optional
        Keep frames whose target matches, compared with separators and case
        folded away, so ``"AB_Aur"`` matches a header reading ``"AB Aur"``.
        Matched as a substring, so it also works against ``target_keyword=
        "OBJECT"`` when OBJECT carries more than the name.
    frame_range : tuple of int or list of tuple, optional
        Inclusive observation numbers, read from the filenames. Either one
        range, ``(932, 939)``, which keeps n0932 through n0939, or several,
        ``[(857, 900), (915, 930), (932, 963)]`` -- a night broken by a slew
        or a filter change is several runs, and this is how an obslog refers
        to them. A frame is kept if it falls in any of the ranges.
    target_keyword : str, optional
        Header keyword holding the target name.

    Returns
    -------
    list of Frame
        The frames that matched every criterion given, in the order supplied.
        Criteria combine with AND, and ``None`` means no constraint, so
        calling with nothing returns the list unchanged.
    """
    criteria = []
    kept = list(frames)

    if target is not None:
        wanted = _normalize_target(target)
        kept = [f for f in kept
                if wanted in _normalize_target(f.get(target_keyword) or "")]
        criteria.append(f"{target_keyword} matching {target!r}")

    ranges = _as_ranges(frame_range)
    if ranges is not None:
        numbered = [(f, frame_number(f)) for f in kept]
        unnumbered = [f for f, n in numbered if n is None]
        if unnumbered:
            log.warning("%d frames have no number in their filename and "
                        "cannot be selected by frame_range; they are dropped.",
                        len(unnumbered))
        kept = [f for f, n in numbered
                if n is not None and any(lo <= n <= hi for lo, hi in ranges)]
        criteria.append("frames "
                        + ", ".join(f"{lo}-{hi}" for lo, hi in ranges))

    if criteria:
        log.info("Selected %d of %d frames by %s", len(kept), len(frames),
                 " and ".join(criteria))
        if not kept:
            log.warning("Selection left NO frames. Check the criteria "
                        "against the night's frame table -- nothing "
                        "downstream will have anything to work on.")

    return kept


def save_frames(filename, frames, overwrite=True):
    """Save several Frames into one multi-extension FITS file.

    Parameters
    ----------
    filename : str
        Path to write.
    frames : iterable of Frame
        Frames to store. The first becomes the primary HDU and the rest
        image extensions.
    overwrite : bool, optional
        Replace an existing file.

    Raises
    ------
    ValueError
        If ``frames`` is empty.

    Notes
    -----
    Used for master dark, flat and sky files, where one file per exposure
    setting is more convenient than one per master. Read back with
    :func:`load_master`. Falls back to scrubbed headers on a
    ``VerifyError``, as :meth:`Frame.save` does.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("no frames to save")
    hdus = [fits.PrimaryHDU(data=frames[0].data,
                            header=without_structural_cards(frames[0].header))]
    hdus += [fits.ImageHDU(data=f.data,
                           header=without_structural_cards(f.header))
             for f in frames[1:]]
    try:
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
    except fits.verify.VerifyError:
        log.warning("Malformed header cards; writing scrubbed headers to %s",
                    filename)
        for h in hdus:
            h.header = scrub_header(h.header)
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)


def load_master(filename):
    """Read a multi-extension master file back into Frames.

    Parameters
    ----------
    filename : str
        Path written by :func:`save_frames`.

    Returns
    -------
    list of Frame
        One per HDU carrying data; empty extensions are skipped, and a
        missing file gives ``[]`` rather than raising, so a reduction can
        ask for masters that were never built.
    """
    if not os.path.isfile(filename):
        return []
    frames = []
    with fits.open(filename) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            frames.append(Frame(np.asarray(hdu.data, dtype=float), hdu.header))
    return frames


def framelist_to_cube(frames):
    """Stack Frames or arrays into a cube.

    Parameters
    ----------
    frames : iterable of Frame or array_like
        Images to stack; all must share a shape.

    Returns
    -------
    ndarray
        Cube of shape ``(nframes, ny, nx)``.
    """
    arrs = [f.data if isinstance(f, Frame) else np.asarray(f) for f in frames]
    return np.stack(arrs, axis=0)


def match_keys(frames, keylist):
    """Group frames by the values of some header keywords.

    Parameters
    ----------
    frames : iterable of Frame
        Frames to group.
    keylist : list of str
        Keywords whose values form the grouping key.

    Returns
    -------
    dict
        Maps ``tuple(frame[k] for k in keylist)`` to the list of frames
        with those values. This is how master darks and flats are split
        into one master per exposure setting.
    """
    matched = {}
    for f in frames:
        key = tuple(f[k] if k in f.header else None for k in keylist)
        matched.setdefault(key, []).append(f)
    return matched


def all_header_keywords_match(frame_a, frame_b, keywords):
    """Do two frames agree on every one of these keywords?

    Parameters
    ----------
    frame_a, frame_b : Frame
        Frames to compare.
    keywords : iterable of str
        Keywords that must match.

    Returns
    -------
    bool
        True if every keyword has the same value in both.

    """
    for keyword in keywords:
        in_a = keyword in frame_a.header
        in_b = keyword in frame_b.header
        if in_a != in_b:
            return False          # one knows it, the other does not
        if in_a and frame_a[keyword] != frame_b[keyword]:
            return False
    return True


def get_between(frames, frameno_range):
    """Select frames by frame number.

    Parameters
    ----------
    frames : iterable of Frame
        Frames to filter.
    frameno_range : tuple of int
        ``(lo, hi)``, both inclusive.

    Returns
    -------
    list of Frame
        Those whose ``FRAMENO`` falls in the range, order preserved.
    """
    lo, hi = frameno_range
    return [f for f in frames if lo <= f["FRAMENO"] <= hi]


def parse_date_obs(date_obs):
    """Parse a FITS ``DATE-OBS`` value into a date.

    Parameters
    ----------
    date_obs : str or datetime.date
        Either a bare date, ``'2025-12-08'``, or a full timestamp,
        ``'2025-12-08T09:38:16.614'``. Only the first ten characters are
        read, so both work.

    Returns
    -------
    datetime.date
        The observing date. NIRC2 records DATE-OBS in UTC, and a Keck night
        runs 04:00-16:00 UTC, so one UTC date names a whole night.

    Raises
    ------
    ValueError
        If the value is not a parseable date. Empty or missing values are
        the caller's to handle -- guessing a date would be worse than
        stopping.
    """
    if isinstance(date_obs, _date):
        return date_obs
    return _date.fromisoformat(str(date_obs)[:10])

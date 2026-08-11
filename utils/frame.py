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

import logging

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
            fits.PrimaryHDU(data=self.data, header=self.header).writeto(
                filename, overwrite=overwrite)
        except fits.verify.VerifyError:
            # some NIRC2 headers carry malformed CONTINUE cards
            log.warning("Malformed header cards in %s; writing a scrubbed "
                        "header", filename)
            fits.PrimaryHDU(data=self.data,
                            header=scrub_header(self.header)).writeto(
                filename, overwrite=overwrite)

    def __repr__(self):
        """Short representation: filename and data shape."""
        name = self.get("FILENAME", "<no filename>")
        return f"Frame({name}, shape={self.data.shape})"


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


def load_frames(frame_paths, rejects=()):
    """Load several FITS files, skipping rejected ones.

    Parameters
    ----------
    frame_paths : iterable of str
        Paths to load.
    rejects : container, optional
        Anything supporting ``in``: a list of filenames, or the
        ``{filename: reason}`` mapping from
        :func:`utils.paths.load_rejects`. Matching is on the *basename*, so
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
    hdus = [fits.PrimaryHDU(data=frames[0].data, header=frames[0].header)]
    hdus += [fits.ImageHDU(data=f.data, header=f.header) for f in frames[1:]]
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

    Raises
    ------
    KeyError
        If a frame is missing one of the keywords.
    """
    matched = {}
    for f in frames:
        key = tuple(f[k] for k in keylist)
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
    return all(frame_a[k] == frame_b[k] for k in keywords)


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

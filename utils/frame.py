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
    """Image data + FITS header."""

    def __init__(self, data, header=None):
        self.data = np.asarray(data, dtype=float)
        self.header = fits.Header(header) if header is not None else fits.Header()

    # dict-style header access, like AstroImage in AIR.jl
    def __getitem__(self, key):
        return self.header[key]

    def __setitem__(self, key, value):
        self.header[key] = value

    def __contains__(self, key):
        return key in self.header

    def get(self, key, default=None):
        return self.header.get(key, default)

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return Frame(self.data.copy(), self.header.copy())

    @classmethod
    def load(cls, filename, hdu=0):
        with fits.open(filename) as hdul:
            frame = cls(np.asarray(hdul[hdu].data, dtype=float), hdul[hdu].header)
        if "FILENAME" not in frame.header:
            frame["FILENAME"] = os.path.basename(str(filename))
        return frame

    def save(self, filename, overwrite=True):
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
    """Load a list of FITS files into Frames, skipping any whose basename is
    in ``rejects``."""
    frames = []
    for fn in frame_paths:
        if os.path.basename(str(fn)) in rejects:
            continue
        frames.append(Frame.load(fn))
    return frames


def save_frames(filename, frames, overwrite=True):
    """Save a list of Frames into a single multi-extension FITS file
    (primary HDU is the first frame). Used for master dark/flat/sky files."""
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
    """Load a multi-extension FITS master file (as written by
    :func:`save_frames`) into a list of Frames. Returns [] if the file does
    not exist."""
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
    """Stack a list of Frames (or 2D arrays) into a cube with shape
    ``(nframes, ny, nx)``."""
    arrs = [f.data if isinstance(f, Frame) else np.asarray(f) for f in frames]
    return np.stack(arrs, axis=0)


def match_keys(frames, keylist):
    """Group frames by the values of the given header keywords.

    Returns a dict mapping ``tuple(frame[k] for k in keylist)`` -> list of
    frames with those header values.
    """
    matched = {}
    for f in frames:
        key = tuple(f[k] for k in keylist)
        matched.setdefault(key, []).append(f)
    return matched


def all_header_keywords_match(frame_a, frame_b, keywords):
    """True if the two frames have identical values for every keyword."""
    return all(frame_a[k] == frame_b[k] for k in keywords)


def get_between(frames, frameno_range):
    """Select frames whose FRAMENO lies in [lo, hi] (inclusive)."""
    lo, hi = frameno_range
    return [f for f in frames if lo <= f["FRAMENO"] <= hi]

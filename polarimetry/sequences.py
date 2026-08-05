"""Build polarimetry sequences from reduced frames.

Translated from AIR.jl's process_polmode/01_generate_sequence.jl. In pol
mode the Wollaston prism splits each frame into two orthogonally-polarized
beams (top and bottom half of the detector); each reduced frame becomes an
"up" and a "dn" frame.
"""

from __future__ import annotations

import logging

from utils.frame import Frame, get_between, match_keys

log = logging.getLogger(__name__)


def split_horizontal(frame):
    """Split a pol-mode frame into its two orthogonally-polarized beams.

    Returns ``(up_frame, dn_frame)``, each the same size as the input with
    the other beam's half zeroed out. The ``SPLIT`` header keyword records
    which beam each frame holds ("up" = top half, rows ``ny//2:``).
    """
    data = frame.data
    nrows = data.shape[0]
    mid = nrows // 2

    up_data = frame.data * 0.0
    dn_data = frame.data * 0.0

    up_data[mid:, :] = data[mid:, :]
    dn_data[:mid, :] = data[:mid, :]

    up_frame = Frame(up_data, frame.header.copy())
    dn_frame = Frame(dn_data, frame.header.copy())
    up_frame["SPLIT"] = "up"
    dn_frame["SPLIT"] = "dn"

    return up_frame, dn_frame


def generate_sequence_epoch(reduced_frames, frameno_range, rescale_frames=False):
    """Select the object frames for one epoch, split each into its two
    polarized beams, and group into sequences.

    Parameters
    ----------
    reduced_frames : list of Frame
        Output of the generic reduction step.
    frameno_range : (int, int)
        Inclusive FRAMENO range of the object frames for this epoch
        (found by inspecting the frame table).
    rescale_frames : bool
        If True, divide every frame by its ITIME and return a single
        sequence. If False (default), group frames by (ITIME, FILTER) so
        each sequence has consistent exposure settings.

    Returns a list of sequences, where each sequence is a list of Frames.
    """
    obj_frames = get_between(reduced_frames, frameno_range)
    log.info("Found %d object frames between %s...", len(obj_frames),
             frameno_range)

    sequence_frames = []
    for frame in obj_frames:
        up_frame, dn_frame = split_horizontal(frame)
        sequence_frames.append(up_frame)
        sequence_frames.append(dn_frame)

    if rescale_frames:
        log.info("Rescaling frames by exposure time...")
        rescaled = []
        for frame in sequence_frames:
            new_frame = frame.copy()
            new_frame.data /= frame["ITIME"]
            rescaled.append(new_frame)
        return [rescaled]

    grouped = match_keys(sequence_frames, ["ITIME", "FILTER"])
    for i, (key, seq) in enumerate(grouped.items()):
        log.info("Sequence %d: %d frames for key %s...", i, len(seq), key)
    return list(grouped.values())


def sequence_dict(frames, keylist, ignore=None):
    """Group frames by header keys, dropping frames whose header matches any
    key/value pair in ``ignore``. Returns a dict key-tuple -> list of frames
    (empty groups removed)."""
    ignore = ignore or {}
    matched = match_keys(frames, keylist)

    out = {}
    for key, group in matched.items():
        kept = []
        for f in group:
            ignored = False
            for k, v in ignore.items():
                if f[k] == v:
                    log.info("Ignoring frame %s due to ignore condition: "
                             "%s = %s", f.get("RED-FN"), k, v)
                    ignored = True
                    break
            if not ignored:
                kept.append(f)
        if kept:
            out[key] = kept
    return out

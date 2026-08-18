"""Record what the pipeline did, in the FITS headers of its own products.

Every processing step appends a HISTORY line of the form::

    HISTORY DPP <step>: key=value, key=value   [2026-07-29T04:12:33]

so a finished product carries an ordered, human-readable account of how it
was made, and stamps ``DPPVER`` (pipeline version / git commit) and
``DPPDATE`` (when it was processed) once.

Use :func:`record_step` from any pipeline stage; use :func:`steps_of` to
read the record back out of a product.
"""

from __future__ import annotations

import datetime
import os
import subprocess

_VERSION_CACHE = None
_STEP_PREFIX = "DPP "


def pipeline_version():
    """Version string for the running pipeline.

    Returns
    -------
    str
        The short git commit of this repository, suffixed ``+local`` when
        the working tree is dirty, or ``"unknown"`` when git is unavailable
        (an installed copy, or no repository). Cached after the first call.

    Notes
    -----
    Stamped into every product as ``DPPVER``, so a reduction can be traced
    to the code that made it. ``+local`` is a warning that the commit alone
    does not identify what ran.
    """
    global _VERSION_CACHE
    if _VERSION_CACHE is not None:
        return _VERSION_CACHE

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        _VERSION_CACHE = (commit + ("+local" if dirty else "")) if commit \
            else "unknown"
    except Exception:
        _VERSION_CACHE = "unknown"
    return _VERSION_CACHE


def _format_value(value):
    """Render one provenance parameter compactly for a HISTORY card.

    Parameters
    ----------
    value : object
        Value to render. Floats use ``%.6g`` (so ``8.0`` becomes ``8``),
        lists and tuples are bracketed and space-separated, everything else
        falls back to ``str``.

    Returns
    -------
    str
        The rendered value.
    """
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + " ".join(_format_value(v) for v in value) + "]"
    return str(value)


# marks a HISTORY card that continues the step above
_CONTINUATION = "    "


def drop_step(target, step):
    """Remove any previously recorded instance of one step.

    Parameters
    ----------
    target : Frame or astropy.io.fits.Header
        Product to edit.
    step : str
        Step name, as passed to :func:`record_step`.

    Returns
    -------
    int
        How many records were removed.
    """
    header = getattr(target, "header", target)
    cards = [str(h) for h in header.get("HISTORY", [])]
    if not cards:
        return 0

    prefix = f"{_STEP_PREFIX}{step}"
    kept, removed, dropping = [], 0, False
    for text in cards:
        if text.startswith(_STEP_PREFIX):
            dropping = text.startswith(prefix)
            removed += dropping
            if not dropping:
                kept.append(text)
        elif dropping and text.startswith(_CONTINUATION):
            continue                      # a continuation of the dropped step
        else:
            dropping = False
            kept.append(text)

    if removed:
        del header["HISTORY"]
        for text in kept:
            header.add_history(text)
    return removed


def record_step(target, step, replace=False, **params):
    """Record one processing step in a header.

    Parameters
    ----------
    target : Frame or astropy.io.fits.Header
        Where to write. A ``Frame``'s header is used.
    step : str
        Name of the stage, e.g. ``"dark/flat reduction"`` or
        ``"stokes cube"``.
    **params
        Settings worth reproducing, rendered by :func:`_format_value`.
    replace : bool, optional
        Drop any earlier record of this same step first. Use it for a step
        that can be re-run over the same frames -- rebuilding a Stokes cube
        at several fast axis offsets would otherwise leave several records
        that contradict each other, with nothing to say which one produced
        the data in hand.

    Returns
    -------
    astropy.io.fits.Header
        The header that was written to.

    Notes
    -----
    Appends a HISTORY card of the form::

        DPP <step>: key=value, key=value   [2026-07-29T04:12:33Z]

    The timestamp is UTC, with the trailing ``Z`` saying so, to match the
    UTC in every NIRC2 header keyword.

    and stamps ``DPPVER`` and ``DPPDATE`` once, on first use. Lines longer
    than a HISTORY card's payload are wrapped across several cards rather
    than silently truncated, so a long parameter list survives the round
    trip to disk.
    """
    header = getattr(target, "header", target)

    if replace:
        drop_step(header, step)

    # UTC, and marked as such. Everything else in a NIRC2 product is UTC --
    # DATE-OBS, UT, MJD -- and a naive local timestamp cannot be compared
    # with any of them, or with the same product reduced on another machine.
    now = datetime.datetime.now(datetime.timezone.utc)

    if "DPPVER" not in header:
        header["DPPVER"] = (pipeline_version(), "NIRC2Pol-DPP version")
        header["DPPDATE"] = (now.isoformat(timespec="seconds"),
                             "pipeline processing date (UTC)")

    detail = ", ".join(f"{k}={_format_value(v)}" for k, v in params.items())
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    text = f"{_STEP_PREFIX}{step}: {detail} [{stamp}]" if detail \
        else f"{_STEP_PREFIX}{step} [{stamp}]"

    limit = 68  # HISTORY card payload
    while text:
        header.add_history(text[:limit])
        text = ((_CONTINUATION + text[limit:])
                if len(text) > limit else "")
    return header


def steps_of(target):
    """Read the recorded pipeline steps back out of a product.

    Parameters
    ----------
    target : Frame or astropy.io.fits.Header
        Product to inspect.

    Returns
    -------
    list of str
        The ``DPP``-prefixed HISTORY lines, in the order they were written,
        with any continuation cards rejoined so a long parameter list reads
        back whole. Other HISTORY cards, such as those astropy adds itself,
        are ignored.
    """
    header = getattr(target, "header", target)
    steps, in_step = [], False
    for card in header.get("HISTORY", []):
        text = str(card)
        if text.startswith(_STEP_PREFIX):
            steps.append(text)
            in_step = True
        elif in_step and text.startswith(_CONTINUATION):
            # A wrapped continuation of the step above. Rejoining is not
            # cosmetic: record_step splits long parameter lists across
            # cards, so without this everything past the first card is
            # invisible to every reader even though it is on disk.
            steps[-1] += text[len(_CONTINUATION):]
        else:
            in_step = False
    return steps


def describe(target):
    """Format a product's full provenance for reading.

    Parameters
    ----------
    target : Frame or astropy.io.fits.Header
        Product to describe.

    Returns
    -------
    str
        A multi-line summary: the pipeline version and processing date,
        then one indented line per recorded step.
    """
    header = getattr(target, "header", target)
    lines = [f"NIRC2Pol-DPP {header.get('DPPVER', 'unknown')} "
             f"processed {header.get('DPPDATE', 'unknown')}"]
    lines += ["  " + s for s in steps_of(header)]
    return "\n".join(lines)

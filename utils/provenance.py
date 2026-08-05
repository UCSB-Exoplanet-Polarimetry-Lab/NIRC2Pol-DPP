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
    """Version string for the running pipeline: the git commit of this
    repository if available, else "unknown"."""
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
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + " ".join(_format_value(v) for v in value) + "]"
    return str(value)


def record_step(target, step, **params):
    """Record one processing step in a header.

    ``target`` may be a ``Frame`` or an ``astropy.io.fits.Header``. ``step``
    names the stage ("dark/flat", "double-difference", ...) and ``params``
    are the settings worth reproducing. Long lines are wrapped across
    several HISTORY cards so nothing is silently truncated.
    """
    header = getattr(target, "header", target)

    if "DPPVER" not in header:
        header["DPPVER"] = (pipeline_version(), "NIRC2Pol-DPP version")
        header["DPPDATE"] = (datetime.datetime.now().isoformat(timespec="seconds"),
                             "pipeline processing date")

    detail = ", ".join(f"{k}={_format_value(v)}" for k, v in params.items())
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    text = f"{_STEP_PREFIX}{step}: {detail} [{stamp}]" if detail \
        else f"{_STEP_PREFIX}{step} [{stamp}]"

    limit = 68  # HISTORY card payload
    while text:
        header.add_history(text[:limit])
        text = ("    " + text[limit:]) if len(text) > limit else ""
    return header


def steps_of(target):
    """List the pipeline steps recorded in a header, in order."""
    header = getattr(target, "header", target)
    return [str(h) for h in header.get("HISTORY", [])
            if str(h).startswith(_STEP_PREFIX)]


def describe(target):
    """Human-readable provenance summary of a product."""
    header = getattr(target, "header", target)
    lines = [f"NIRC2Pol-DPP {header.get('DPPVER', 'unknown')} "
             f"processed {header.get('DPPDATE', 'unknown')}"]
    lines += ["  " + s for s in steps_of(header)]
    return "\n".join(lines)

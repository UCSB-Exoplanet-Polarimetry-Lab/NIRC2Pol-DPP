"""Capture a whole reduction into one log file.

Every choice the pipeline makes -- which flat it matched, whether the band
requirement was enforced, the beam geometry it measured, a centering method
falling back, a cycle with mixed exposure -- is already reported through the
standard :mod:`logging` module. Attaching a file handler therefore captures
the lot without any of those call sites knowing about it.

What this adds beyond a plain handler is the two things a log needs to be
worth keeping: a header saying which pipeline version wrote it and when, and
a footer saying how long it took and how many warnings went by. 
"""

from __future__ import annotations

import datetime
import logging
import os
import time

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"


class _WarningCounter(logging.Filter):
    """Counts records at WARNING or above, passing everything through."""

    def __init__(self):
        super().__init__()
        self.warnings = 0
        self.errors = 0

    def filter(self, record):
        if record.levelno >= logging.ERROR:
            self.errors += 1
        elif record.levelno >= logging.WARNING:
            self.warnings += 1
        return True


class ReductionLog:
    """A file handler on the root logger, with a header and a footer.

    Use it as a context manager, or call :meth:`finish` yourself

    Parameters
    ----------
    path : str
        File to write. Its directory is created if missing.
    level : int, optional
        Lowest level to record. INFO by default, which is where the
        pipeline reports what it chose; DEBUG adds the per-frame detail.
    mode : str, optional
        ``"w"`` to start fresh, ``"a"`` to append to an earlier run.

    Attributes
    ----------
    path : str
        The file being written.
    warnings, errors : int
        How many have been seen so far.
    """

    def __init__(self, path, level=logging.INFO, mode="w"):
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        self._counter = _WarningCounter()
        self._handler = logging.FileHandler(self.path, mode=mode)
        self._handler.setLevel(level)
        # logging.Formatter renders %(asctime)s with time.localtime unless
        # told otherwise, so the trailing Z in _DATEFMT would be a lie on any
        # machine not set to UTC -- and every date in a NIRC2 header is UTC.
        formatter = logging.Formatter(_FORMAT, _DATEFMT)
        formatter.converter = time.gmtime
        self._handler.setFormatter(formatter)
        self._handler.addFilter(self._counter)

        root = logging.getLogger()
        self._restore_level = root.level
        if root.level > level or root.level == logging.NOTSET:
            root.setLevel(level)
        root.addHandler(self._handler)

        self._started = datetime.datetime.now(datetime.timezone.utc)
        self._finished = False
        self._write_header()

    @property
    def warnings(self):
        """Warnings recorded so far."""
        return self._counter.warnings

    @property
    def errors(self):
        """Errors recorded so far."""
        return self._counter.errors

    def _write_header(self):
        from utils.provenance import pipeline_version

        self._handler.stream.write(
            f"# NIRC2Pol-DPP {pipeline_version()}\n"
            f"# reduction started {self._started.isoformat(timespec='seconds')}\n"
            f"# times below are UTC, as every date in a NIRC2 header is\n\n")
        self._handler.stream.flush()

    def settings(self, **values):
        """Record the configuration this reduction ran with.

        Parameters
        ----------
        **values
            Any settings worth reproducing. Logged at INFO, so they land in
            the file and on the console together.

        Notes
        -----
        The log otherwise records what *happened*; this records what was
        *asked for*, which is the other half of reproducing a result.
        """
        log = logging.getLogger("reduction")
        for key in sorted(values):
            log.info("setting %s = %s", key, values[key])

    def finish(self):
        """Write the footer and detach. Safe to call more than once."""
        if self._finished:
            return
        self._finished = True

        elapsed = (datetime.datetime.now(datetime.timezone.utc)
                   - self._started).total_seconds()
        verdict = "no warnings"
        if self._counter.errors:
            verdict = (f"{self._counter.errors} error(s), "
                       f"{self._counter.warnings} warning(s)")
        elif self._counter.warnings:
            verdict = f"{self._counter.warnings} warning(s) -- worth a read"

        self._handler.stream.write(
            f"\n# finished in {elapsed:.1f} s with {verdict}\n")
        self._handler.stream.flush()

        root = logging.getLogger()
        root.removeHandler(self._handler)
        root.setLevel(self._restore_level)
        self._handler.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.finish()
        return False


def start_reduction_log(path, level=logging.INFO, mode="w"):
    """Begin capturing the reduction into ``path``.

    Parameters
    ----------
    path : str
        File to write.
    level : int, optional
        Lowest level to record.
    mode : str, optional
        ``"w"`` to overwrite, ``"a"`` to append.

    Returns
    -------
    ReductionLog
        Call :meth:`~ReductionLog.finish` when the reduction ends, or use it
        as a context manager.
    """
    return ReductionLog(path, level=level, mode=mode)

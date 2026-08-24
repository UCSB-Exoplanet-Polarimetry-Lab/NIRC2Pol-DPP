"""Standard on-disk layout for one reduction.

Descended from AIR.jl's ObslogPaths. Everything here is *output*: the folder
a reduction writes, which is deliberately not the folder the raw frames came
from. Those are two different things -- an archive of frames you may not own
and did not create, and the result of one reduction of them -- and conflating
them means the pipeline cannot be pointed at an archive at all, and writes
into it when it can::

    reductions_root/
        raw/          symlinks to the frames this reduction reads
        reduced/      dark-subtracted, flat-divided frames
        sequences/    centered / derotated / combined products
        plots/
        reduction_<date>.toml   the config this run used
        master_darks_<date>.fits
        master_flats_<date>.fits
        master_skies_<date>.fits
        master_mask_<date>.fits
        reduction_<date>.log
        <date>_rejects.toml
        <date>_reduced_frames_table.txt

``date`` locates nothing. It names the masters and the log, and is checked
against the frames' own DATE-OBS -- which is all it was ever really doing.
:func:`link_frames` populates ``raw/`` from wherever the frames actually live.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field

from nirc2pol.utils.frame import in_frame_range, parse_date_obs


log = logging.getLogger(__name__)


@dataclass
class ObslogPaths:
    """Folder layout for one reduction, rooted where it writes.

    Descended from AIR.jl's ``ObslogPaths``. Constructed from the folder this
    reduction writes to and the date of the night it covers; every other path
    is derived, so a reduction script names them once::

        /home/you/reductions/AB_Aur_Lp/   <- reductions_root
            raw/                          <- symlinks, see link_frames
            reduced/  sequences/  plots/
            master_darks_2025-12-08.fits
            reduction_2025-12-08.log

    The raw frames are somewhere else entirely, and stay there: pass their
    folder to :meth:`link_raw_frames`.

    Parameters
    ----------
    reductions_root : str
        Folder this reduction writes to. Not where the raw frames live.
    date : str
        The night, UTC (e.g. ``"2026-06-05"``). Names the masters, log and
        frame table, and is checked against the frames' DATE-OBS by
        :meth:`check_frame_dates`. It locates nothing.

    Attributes
    ----------
    raw_folder, reduced_folder, sequences_folder, plots_folder : str
        Subfolders of ``reductions_root``.
    darks_file, flats_file, skies_file, master_mask_file : str
        Multi-extension master files, as written by
        :func:`nirc2pol.utils.frame.save_frames`.
    rejects_file : str
        TOML list of frames to exclude; see :func:`load_rejects`.
    config_file : str
        Where the config a run used is copied, beside the log.
    """

    reductions_root: str
    date: str

    raw_folder: str = field(init=False)
    reduced_folder: str = field(init=False)
    plots_folder: str = field(init=False)
    sequences_folder: str = field(init=False)

    rejects_file: str = field(init=False)
    table_file: str = field(init=False)
    config_file: str = field(init=False)

    darks_file: str = field(init=False)
    flats_file: str = field(init=False)
    skies_file: str = field(init=False)
    masks_file: str = field(init=False)

    def __post_init__(self):
        """Derive every path from ``reductions_root`` and ``date``."""
        self.raw_folder = os.path.join(self.reductions_root, "raw")
        self.reduced_folder = os.path.join(self.reductions_root, "reduced")
        self.plots_folder = os.path.join(self.reductions_root, "plots")
        self.sequences_folder = os.path.join(self.reductions_root, "sequences")

        # everything the reduction logged, in one file beside the products
        self.log_file = os.path.join(
            self.reductions_root, f"reduction_{self.date}.log")

        # ...and the config it ran with, beside the log. A run records the
        # config's *values* in the log either way; this is the file, so the
        # reduction can be repeated from its own folder without hunting for
        # whichever copy was passed on the command line and hoping it has
        # not been edited since.
        self.config_file = os.path.join(
            self.reductions_root, f"reduction_{self.date}.toml")

        self.rejects_file = os.path.join(
            self.reductions_root, f"{self.date}_rejects.toml")
        self.table_file = os.path.join(
            self.reductions_root, f"{self.date}_reduced_frames_table.txt")

        # Masters carry the date because they belong to the dataset they
        # were taken with. Darks and flats are taken with every dataset and
        # are not interchangeable between them; a bare "darks.fits" invites
        # reuse and leaves no trace when it happens. The date matches the
        # rejects and frame-table files above.
        self.darks_file = os.path.join(
            self.reductions_root, f"master_darks_{self.date}.fits")
        self.flats_file = os.path.join(
            self.reductions_root, f"master_flats_{self.date}.fits")
        self.skies_file = os.path.join(
            self.reductions_root, f"master_skies_{self.date}.fits")
        self.masks_file = os.path.join(
            self.reductions_root, f"master_mask_{self.date}.fits")

    def raw_files(self, frame_range=None, pattern="*.fits*"):
        """The night's raw frames, in name order.

        Parameters
        ----------
        frame_range : tuple or list of tuple, optional
            Keep only frames whose observation number falls inside one of
            these inclusive ranges; see
            :func:`nirc2pol.utils.frame.in_frame_range`. This is read from
            the filename, so nothing is opened to apply it.
        pattern : str, optional
            Glob for the raw files. The default ends in ``*`` so gzipped
            archive frames (``n0902.fits.gz``) are picked up too.

        Returns
        -------
        list of str
            Absolute paths, sorted.

        Raises
        ------
        FileNotFoundError
            When there is nothing to reduce, either way of getting there:
            ``raw_folder`` empty, or files there but ``frame_range``
            excluding every one. One exception type because from the
            caller's side the two are the same condition, and the command
            line can then report both as the user errors they are rather
            than as tracebacks.

        Notes
        -----
        Raising here is the point. Without it an empty night reads as zero
        darks, zero flats and zero science frames, each merely logged, and
        the reduction runs on for several steps before failing somewhere
        that cannot say what was actually wrong.
        """
        found = sorted(glob.glob(os.path.join(self.raw_folder, pattern)))

        if not found:
            raise FileNotFoundError(
                f"No raw frames matching {pattern!r} in {self.raw_folder}. "
                f"That folder is filled by link_raw_frames from wherever the "
                f"frames actually live -- cfg.raw_data_folder for a "
                f"reduction.")

        if frame_range is None:
            return found

        kept = [f for f in found if in_frame_range(f, frame_range)]
        if not kept:
            raise FileNotFoundError(
                f"raw_range {frame_range} excluded all {len(found)} frame(s) "
                f"in {self.raw_folder}. It is read from the filename, so "
                f"check the numbers against what is there.")
        return kept

    def link_raw_frames(self, source_folder, frame_range=None,
                        pattern="*.fits*"):
        """Fill ``raw_folder`` with links to the frames this run will read.

        Thin wrapper on :func:`link_frames`, which carries the detail.

        Parameters
        ----------
        source_folder : str
            Where the raw frames actually are. Never written to.
        frame_range : tuple or list of tuple, optional
            Which frames to link; see :func:`link_frames`.
        pattern : str, optional
            Glob for the raw files.

        Returns
        -------
        list of str
            The links, sorted -- the frames this reduction reads.
        """
        return link_frames(source_folder, self.raw_folder,
                           frame_range=frame_range, pattern=pattern)

    def check_frame_dates(self, frames, keyword="DATE-OBS"):
        """Warn when the dataset folder's date disagrees with its frames.

        Parameters
        ----------
        frames : iterable of Frame
            Frames belonging to this dataset.
        keyword : str, optional
            Header keyword holding the observing date. NIRC2 records
            ``DATE-OBS`` in UTC.

        Returns
        -------
        set of str
            The distinct dates found in the frames.
        """
        found = set()
        for frame in frames:
            raw = str(frame.get(keyword, "")).strip()
            if raw:
                found.add(parse_date_obs(raw).isoformat())
        if found and self.date not in found:
            log.warning(
                "Dataset folder is dated %s but its frames say %s=%s. That "
                "keyword is UTC, and a Hawai'i night is a single UTC date one "
                "day after the HST evening -- check which the folder is "
                "named for, because masters and products inherit this date.",
                self.date, keyword, ", ".join(sorted(found)))
        return found

    def make_folders(self):
        """Create the folders this reduction *writes* to.

        ``raw_folder`` is deliberately not among them. It is an input: the
        frames are either in it already or they are not, so creating it can
        only ever produce an empty directory -- and that manufactures the
        very folder :meth:`raw_files` is about to report as empty, hiding a
        mistyped date or a wrong root behind a directory this call made
        itself.
        """
        for folder in (self.reduced_folder, self.plots_folder,
                       self.sequences_folder):
            os.makedirs(folder, exist_ok=True)


def link_frames(source_folder, dest_folder, frame_range=None,
                pattern="*.fits*"):
    """Symlink raw frames into a reduction's own folder.

    The frames stay where they are -- an archive, shared space, a mounted
    volume -- and the reduction folder gets links to exactly the ones it
    reads. That makes ``dest_folder`` a record of this run's inputs without
    copying a byte, and it is why a reduction never has to be run inside the
    data it reduces.

    Parameters
    ----------
    source_folder : str
        Where the frames actually are. Only read from.
    dest_folder : str
        Where the links go, created if absent -- but only once there is
        something to put in it.
    frame_range : tuple or list of tuple, optional
        Link only frames whose observation number falls inside one of these
        inclusive ranges; see :func:`nirc2pol.utils.frame.in_frame_range`.
        Read from the filename, so nothing is opened. None links everything
        matching ``pattern``.
    pattern : str, optional
        Glob for the frames. The default ends in ``*`` so gzipped archive
        frames (``n0902.fits.gz``) are picked up too.

    Returns
    -------
    list of str
        The paths in ``dest_folder``, sorted. These are what to reduce.

    Raises
    ------
    FileNotFoundError
        When ``source_folder`` holds nothing matching, or ``frame_range``
        excluded every file. Checked *before* anything is created, so a
        mistyped source leaves no empty folders behind.

    Notes
    -----
    Idempotent, and careful about what it replaces. A link already pointing
    at the same file is left alone. A **stale** link -- one pointing
    somewhere else, which is what a re-pointed source leaves behind -- is
    replaced. A **real file** of that name is never touched, only warned
    about: that is somebody's actual data, and this function has no business
    deleting it.

    Link targets are absolute, so the reduction folder can be moved without
    breaking them. Source and destination being the same folder is a no-op:
    the frames are already where they need to be.
    """
    source_folder = os.path.abspath(os.path.expanduser(source_folder))
    dest_folder = os.path.abspath(os.path.expanduser(dest_folder))

    found = sorted(glob.glob(os.path.join(source_folder, pattern)))
    if not found:
        raise FileNotFoundError(
            f"No frames matching {pattern!r} in {source_folder}.")

    if frame_range is not None:
        kept = [f for f in found if in_frame_range(f, frame_range)]
        if not kept:
            raise FileNotFoundError(
                f"Frame range {frame_range} excluded all {len(found)} "
                f"frame(s) in {source_folder}. It is read from the filename, "
                f"so check the numbers against what is there.")
        found = kept

    if source_folder == dest_folder:
        log.info("%d frame(s) already in %s; nothing to link",
                 len(found), dest_folder)
        return found

    os.makedirs(dest_folder, exist_ok=True)

    links, created, reused, skipped = [], 0, 0, 0
    for src in found:
        dest = os.path.join(dest_folder, os.path.basename(src))
        if os.path.islink(dest):
            if os.path.realpath(dest) == os.path.realpath(src):
                reused += 1
                links.append(dest)
                continue
            # Points somewhere else: a source that has moved, or a config
            # re-pointed at different data. Replacing a link destroys
            # nothing.
            os.unlink(dest)
        elif os.path.exists(dest):
            log.warning(
                "%s is a real file, not a link, so it is left as it is and "
                "will be read instead of %s. Remove it if that is not what "
                "you want -- this will not delete data it did not create.",
                dest, src)
            skipped += 1
            links.append(dest)
            continue

        os.symlink(src, dest)
        created += 1
        links.append(dest)

    log.info("%d frame(s) in %s: %d linked, %d already there, %d left alone",
             len(links), dest_folder, created, reused, skipped)
    return sorted(links)


def load_rejects(rejects_file):
    """Load rejected frames from a TOML file, as ``{filename: reason}``.

    Two forms are accepted. A plain list, which is what earlier reductions
    used and which loses the reason::

        rejects = ["n0123.fits", "n0456.fits"]

    or a table, which keeps it::

        [rejects]
        "n0123.fits" = "open AO loop"
        "n0456.fits" = "satellite trail"

    Returns a dict either way (empty reasons for the list form), and ``{}``
    when the file does not exist. Callers testing membership are unaffected,
    since ``in`` on a dict checks its keys -- which is how
    :func:`nirc2pol.utils.frame.load_frames` uses it.
    """
    import tomllib

    if not os.path.isfile(rejects_file):
        return {}
    with open(rejects_file, "rb") as f:
        entry = tomllib.load(f).get("rejects", [])

    if isinstance(entry, dict):
        return {str(k): str(v) for k, v in entry.items()}
    return {str(name): "" for name in entry}


def record_reject(rejects_file, filename, reason=""):
    """Add or update a rejected frame, preserving the existing entries.

    Rewrites the file in the table form, so a file that started as a plain
    list is upgraded in place and keeps the reasons added from then on.
    Returns the full mapping as written.

    ``tomllib`` is read-only, so the table is written directly rather than
    taking a dependency on a TOML writer; the structure is two levels deep
    and the only escaping needed is for quotes and backslashes.
    """
    rejects = load_rejects(rejects_file)
    rejects[str(os.path.basename(str(filename)))] = str(reason)

    def _quote(text):
        """TOML-quote a string, escaping backslashes and double quotes."""
        return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = ["# Frames excluded from reductions, with why.",
             "# Read by utils.paths.load_rejects; written by record_reject.",
             "", "[rejects]"]
    lines += [f"{_quote(name)} = {_quote(rejects[name])}"
              for name in sorted(rejects)]
    with open(rejects_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    return rejects

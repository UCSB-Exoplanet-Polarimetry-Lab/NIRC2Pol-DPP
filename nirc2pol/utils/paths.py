"""Standard on-disk layout for one night of observations.

Mirrors AIR.jl's ObslogPaths::

    observations_folder/
        <date>/
            raw/          raw FITS frames
            reduced/      dark-subtracted, flat-divided frames
            sequences/    centered / derotated / combined products
            plots/
            darks.fits    master darks   (multi-extension)
            flats.fits    master flats
            skies.fits    master skies
            master_mask.fits
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
    """Standard folder layout for one night of observations.

    Mirrors AIR.jl's ``ObslogPaths``. Constructed from the root
    observations folder and a date; every other path is derived, so a
    reduction script names the night once.

    The layout is two levels deep -- the root holds one folder per night, and
    the frames sit inside that, not in the root itself::

        /data/nirc2pol/          <- observations_folder
          2025-12-08/            <- date
            raw/  *.fits         <- raw frames
            reduced/ sequences/  <- written by the reduction

    Parameters
    ----------
    observations_folder : str
        Root folder holding one subfolder per night. Not the folder the FITS
        files themselves are in; see the tree above.
    date : str
        Night identifier, used as the subfolder name (e.g. ``"2026-06-05"``).

    Attributes
    ----------
    data_folder : str
        ``observations_folder/date``.
    raw_folder, reduced_folder, sequences_folder, plots_folder : str
        The per-night subfolders.
    darks_file, flats_file, skies_file, master_mask_file : str
        Multi-extension master files, as written by
        :func:`nirc2pol.utils.frame.save_frames`.
    rejects_file : str
        TOML list of frames to exclude; see :func:`load_rejects`.
    """

    observations_folder: str
    date: str

    data_folder: str = field(init=False)
    raw_folder: str = field(init=False)
    reduced_folder: str = field(init=False)
    plots_folder: str = field(init=False)
    sequences_folder: str = field(init=False)

    rejects_file: str = field(init=False)
    table_file: str = field(init=False)

    darks_file: str = field(init=False)
    flats_file: str = field(init=False)
    skies_file: str = field(init=False)
    masks_file: str = field(init=False)

    def __post_init__(self):
        """Derive every path from ``observations_folder`` and ``date``."""
        self.data_folder = os.path.join(self.observations_folder, self.date)

        self.raw_folder = os.path.join(self.data_folder, "raw")
        self.reduced_folder = os.path.join(self.data_folder, "reduced")
        self.plots_folder = os.path.join(self.data_folder, "plots")
        self.sequences_folder = os.path.join(self.data_folder, "sequences")

        # everything the reduction logged, in one file beside the products
        self.log_file = os.path.join(
            self.data_folder, f"reduction_{self.date}.log")

        self.rejects_file = os.path.join(self.data_folder, f"{self.date}_rejects.toml")
        self.table_file = os.path.join(
            self.data_folder, f"{self.date}_reduced_frames_table.txt")

        # Masters carry the date because they belong to the dataset they
        # were taken with. Darks and flats are taken with every dataset and
        # are not interchangeable between them; a bare "darks.fits" invites
        # reuse and leaves no trace when it happens. The date matches the
        # rejects and frame-table files above.
        self.darks_file = os.path.join(
            self.data_folder, f"master_darks_{self.date}.fits")
        self.flats_file = os.path.join(
            self.data_folder, f"master_flats_{self.date}.fits")
        self.skies_file = os.path.join(
            self.data_folder, f"master_skies_{self.date}.fits")
        self.masks_file = os.path.join(
            self.data_folder, f"master_mask_{self.date}.fits")

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
            When the night has no frames to reduce, either way of getting
            there: nothing matching in ``raw_folder``, or files there but
            ``frame_range`` excluding every one. The message says which, and
            says so explicitly when the frames turn out to be sitting in
            ``observations_folder`` instead -- the layout mistake this
            exists to catch. One exception type because from the caller's
            side the two are the same condition, and the command line can
            then report both as the user errors they are rather than as
            tracebacks.

        Notes
        -----
        Raising here is the point. Without it an empty night reads as zero
        darks, zero flats and zero science frames, each merely logged, and
        the reduction runs on for several steps before failing somewhere
        that cannot say what was actually wrong.
        """
        found = sorted(glob.glob(os.path.join(self.raw_folder, pattern)))

        if not found:
            # The usual cause: an archive folder holding the frames directly,
            # with no <date>/raw/ beneath it. Say so rather than making the
            # user work it out from an empty result.
            loose = sorted(glob.glob(os.path.join(self.observations_folder,
                                                  pattern)))
            hint = ""
            if loose:
                hint = (f" {len(loose)} frame(s) do sit directly in "
                        f"{self.observations_folder}; this layout is two "
                        f"levels deep, so they belong in {self.raw_folder} "
                        f"(a symlink is enough).")
            raise FileNotFoundError(
                f"No raw frames matching {pattern!r} in {self.raw_folder}."
                + hint)

        if frame_range is None:
            return found

        kept = [f for f in found if in_frame_range(f, frame_range)]
        if not kept:
            raise FileNotFoundError(
                f"raw_range {frame_range} excluded all {len(found)} frame(s) "
                f"in {self.raw_folder}. It is read from the filename, so "
                f"check the numbers against what is there.")
        return kept

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

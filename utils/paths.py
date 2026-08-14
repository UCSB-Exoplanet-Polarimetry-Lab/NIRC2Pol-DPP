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
import os
from dataclasses import dataclass, field


@dataclass
class ObslogPaths:
    """Standard folder layout for one night of observations.

    Mirrors AIR.jl's ``ObslogPaths``. Constructed from the root
    observations folder and a date; every other path is derived, so a
    reduction script names the night once.

    Parameters
    ----------
    observations_folder : str
        Root folder holding one subfolder per night.
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
        :func:`utils.frame.save_frames`.
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

        self.rejects_file = os.path.join(self.data_folder, f"{self.date}_rejects.toml")
        self.table_file = os.path.join(
            self.data_folder, f"{self.date}_reduced_frames_table.txt")

        self.darks_file = os.path.join(self.data_folder, "darks.fits")
        self.flats_file = os.path.join(self.data_folder, "flats.fits")
        self.skies_file = os.path.join(self.data_folder, "skies.fits")
        self.masks_file = os.path.join(self.data_folder, "master_mask.fits")

    def make_folders(self):
        """Create the night's subfolders, leaving any that already exist."""
        for folder in (self.raw_folder, self.reduced_folder,
                       self.plots_folder, self.sequences_folder):
            os.makedirs(folder, exist_ok=True)


def make_and_clear(folder_path, glob_pattern):
    """Create a folder if needed; if it already exists, delete files matching
    ``glob_pattern`` inside it."""
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    else:
        for fn in glob.glob(os.path.join(folder_path, glob_pattern)):
            os.remove(fn)


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
    :func:`utils.frame.load_frames` uses it.
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

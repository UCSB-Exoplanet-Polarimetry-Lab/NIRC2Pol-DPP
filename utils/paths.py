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
    """Load the list of rejected frame filenames from a TOML file with a
    top-level ``rejects`` list. Returns [] if the file doesn't exist."""
    import tomllib

    if not os.path.isfile(rejects_file):
        return []
    with open(rejects_file, "rb") as f:
        return tomllib.load(f)["rejects"]

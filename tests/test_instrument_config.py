"""The instrument constants file: parsing, the per-epoch beam geometry table,
and the lookup that replaces hardcoding geometry in every reduction script."""

from configparser import ConfigParser
from datetime import date

import numpy as np
import pytest

from instruments import nirc2
from instruments.nirc2 import NIRC2PolarimetryData
from utils import Frame


def _config(text):
    parser = ConfigParser(inline_comment_prefixes=(";",))
    parser.optionxform = str
    parser.read_string(text)
    return parser


TWO_EPOCHS = """
[beam_geometry.early]
from_date = 2025-12-01
to_date = 2025-12-31
bands = Lp, L
top_row_start = 504
beam_x_offset = 12

[beam_geometry.later]
from_date = 2026-06-01
to_date = 2026-06-30
bands = Kp
top_row_start = 529
beam_x_offset = 14
"""


def _header(date_obs, band="Lp"):
    return Frame(np.zeros((4, 4)),
                 {"DATE-OBS": date_obs, "FWINAME": band}).header


# -- the shipped file --------------------------------------------------------

def test_shipped_config_parses_and_holds_the_measured_epoch():
    """A typo here would break every reduction, so it is worth a test."""
    assert nirc2.PLATE_SCALE == pytest.approx(0.009942)
    entries = {g.label: g for g in nirc2.BEAM_GEOMETRIES}
    assert "2025-12_lprime" in entries
    measured = entries["2025-12_lprime"]
    assert (measured.top_row_start, measured.beam_x_offset) == (504, 12)


def test_band_names_keep_their_case():
    """ConfigParser lowercases option names by default, which would collide
    Lp with L and stop either matching what band_of returns."""
    assert "Lp" in nirc2.RECOMMENDED_BACKGROUND
    assert "Ks" in nirc2.RECOMMENDED_BACKGROUND
    assert nirc2.RECOMMENDED_BACKGROUND["Lp"] == ("dither", "mean_box")


def test_missing_config_is_an_error_not_a_silent_fallback():
    with pytest.raises(FileNotFoundError):
        nirc2.read_config("/nonexistent/nirc2.ini")


# -- the lookup --------------------------------------------------------------

def test_lookup_matches_on_both_date_and_band():
    geoms = nirc2.load_beam_geometries(_config(TWO_EPOCHS))
    assert nirc2.beam_geometry_for(_header("2025-12-07"),
                                   geoms).top_row_start == 504
    assert nirc2.beam_geometry_for(_header("2026-06-15", "Kp"),
                                   geoms).top_row_start == 529


@pytest.mark.parametrize("date_obs, band", [
    ("2026-01-15", "Lp"),      # right band, gap between epochs
    ("2025-12-07", "Kp"),      # right date, band never measured then
])
def test_unmeasured_epoch_refuses_rather_than_borrowing(date_obs, band):
    """Borrowing a neighbouring value is exactly how the beams ended up
    3.8 px apart, so an unmatched frame has to stop the reduction."""
    geoms = nirc2.load_beam_geometries(_config(TWO_EPOCHS))
    with pytest.raises(ValueError, match="no beam geometry recorded"):
        nirc2.beam_geometry_for(_header(date_obs, band), geoms)


def test_overlapping_entries_are_refused_not_silently_picked():
    overlapping = TWO_EPOCHS + """
[beam_geometry.overlap]
from_date = 2025-12-05
to_date = 2025-12-09
bands = Lp
top_row_start = 999
beam_x_offset = 99
"""
    geoms = nirc2.load_beam_geometries(_config(overlapping))
    with pytest.raises(ValueError, match="entries match"):
        nirc2.beam_geometry_for(_header("2025-12-07"), geoms)


def test_frame_without_a_date_cannot_be_looked_up():
    with pytest.raises(ValueError, match="no DATE-OBS"):
        nirc2.beam_geometry_for(Frame(np.zeros((4, 4)), {"FWINAME": "Lp"}).header)


# -- how split_beams uses it -------------------------------------------------

class LookedUp(NIRC2PolarimetryData):
    """Geometry left unset, so it has to come from the per-epoch table."""

    beam_height = 60
    top_row_start = None
    beam_x_offset = None
    background_method = None


def _frame(date_obs="2025-12-07"):
    data = np.zeros((600, 120))
    data[504 + 20, 40] = 100.0          # something inside the top beam
    return Frame(data, {"DATE-OBS": date_obs, "FWINAME": "Lp"})


def test_split_beams_configures_itself_from_the_table():
    """The point of the file: scripts stop copying the numbers around."""
    stack = LookedUp().split_beams(_frame())
    assert stack.shape == (2, 60, 120 - 12)
    # the injected pixel landed where top_row_start=504 says it should
    assert np.unravel_index(np.argmax(stack[1]), stack[1].shape)[0] == 20


def test_explicit_attributes_beat_the_table():
    """A new epoch must be reducible before anyone edits the config."""
    inst = LookedUp()
    inst.top_row_start = 500
    stack = inst.split_beams(_frame())
    assert np.unravel_index(np.argmax(stack[1]), stack[1].shape)[0] == 24


def test_unknown_epoch_stops_the_reduction():
    with pytest.raises(ValueError, match="no beam geometry recorded"):
        LookedUp().split_beams(_frame(date_obs="2024-03-03"))

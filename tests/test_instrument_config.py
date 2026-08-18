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


# -- split_beams will not guess ---------------------------------------------

class Unconfigured(NIRC2PolarimetryData):
    """Geometry unset, as an instrument arrives before it is measured."""

    beam_height = 60
    top_row_start = None
    beam_x_offset = None
    background_method = None


def _frame():
    data = np.zeros((600, 120))
    data[504 + 20, 40] = 100.0
    return Frame(data, {"DATE-OBS": "2025-12-08", "FWINAME": "Lp"})


def test_split_beams_refuses_until_the_geometry_is_measured():
    """It used to consult a per-epoch table. Measuring is now a step of the
    reduction, so arriving here unset means the step was skipped."""
    with pytest.raises(ValueError, match="fit_beam_geometry"):
        Unconfigured().split_beams(_frame())


def test_split_beams_uses_the_geometry_once_it_is_set():
    inst = Unconfigured()
    inst.top_row_start, inst.beam_x_offset = 504, 12
    stack = inst.split_beams(_frame())
    assert stack.shape == (2, 60, 120 - 12)
    assert np.unravel_index(np.argmax(stack[1]), stack[1].shape)[0] == 20

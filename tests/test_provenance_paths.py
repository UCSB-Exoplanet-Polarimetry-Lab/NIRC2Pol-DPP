"""The audit trail: provenance headers, and the reject list."""

import numpy as np
import pytest

from utils.frame import Frame
from utils.paths import load_rejects, record_reject
from utils.provenance import describe, record_step, steps_of


def test_record_step_round_trips():
    frame = Frame(np.zeros((4, 4)), {})
    record_step(frame, "dark/flat reduction", darksub=True, gain=8.0)
    steps = steps_of(frame)
    assert any("dark/flat reduction" in s for s in steps)
    # record_step formats floats with %.6g, so 8.0 renders as "8"
    assert any("gain=8" in s for s in steps)


def test_record_step_accumulates_in_order():
    frame = Frame(np.zeros((4, 4)), {})
    record_step(frame, "first")
    record_step(frame, "second")
    steps = steps_of(frame)
    assert len(steps) == 2
    assert "first" in steps[0] and "second" in steps[1]


def test_provenance_survives_a_fits_round_trip(tmp_path):
    """The real risk: HISTORY cards silently not reaching disk.

    Nothing would raise — the products would simply lose their record of how
    they were made, which is only noticed much later when someone asks.
    """
    out = tmp_path / "p.fits"
    frame = Frame(np.zeros((4, 4)), {})
    record_step(frame, "stokes cube", fast_axis_offset=11.36,
                registration="smooth_peak")
    frame.save(str(out))

    back = Frame.load(str(out))
    trail = describe(back)
    assert "stokes cube" in trail
    assert "11.36" in trail
    assert "smooth_peak" in trail


def test_describe_is_readable_when_there_is_nothing():
    assert isinstance(describe(Frame(np.zeros((2, 2)), {})), str)


# --- rejects -----------------------------------------------------------

def test_load_rejects_missing_file(tmp_path):
    assert load_rejects(str(tmp_path / "nope.toml")) == {}


def test_load_rejects_list_form(tmp_path):
    """The original format, which carries no reasons."""
    f = tmp_path / "r.toml"
    f.write_text('rejects = ["n0001.fits", "n0002.fits"]\n')
    assert load_rejects(str(f)) == {"n0001.fits": "", "n0002.fits": ""}


def test_load_rejects_table_form_keeps_reasons(tmp_path):
    f = tmp_path / "r.toml"
    f.write_text('[rejects]\n"n0003.fits" = "open AO loop"\n')
    assert load_rejects(str(f)) == {"n0003.fits": "open AO loop"}


def test_record_reject_upgrades_a_list_file_in_place(tmp_path):
    f = tmp_path / "r.toml"
    f.write_text('rejects = ["n0001.fits"]\n')
    record_reject(str(f), "n0009.fits", "satellite trail")

    got = load_rejects(str(f))
    assert got["n0001.fits"] == "", "the existing entry must survive"
    assert got["n0009.fits"] == "satellite trail"


def test_record_reject_takes_a_basename_from_a_path(tmp_path):
    f = tmp_path / "r.toml"
    record_reject(str(f), "/data/2026-06-05/raw/n0042.fits", "cloud")
    assert "n0042.fits" in load_rejects(str(f))


def test_record_reject_updates_an_existing_reason(tmp_path):
    f = tmp_path / "r.toml"
    record_reject(str(f), "n0001.fits", "first guess")
    record_reject(str(f), "n0001.fits", "actually the AO loop opened")
    got = load_rejects(str(f))
    assert len(got) == 1
    assert got["n0001.fits"] == "actually the AO loop opened"


def test_record_reject_escapes_quotes_and_backslashes(tmp_path):
    f = tmp_path / "r.toml"
    messy = 'seeing was "bad", see C:\\logs'
    record_reject(str(f), "n0001.fits", messy)
    assert load_rejects(str(f))["n0001.fits"] == messy


def test_rejects_membership_works_for_load_frames(tmp_path):
    """load_frames tests `basename in rejects`, so a dict must behave."""
    f = tmp_path / "r.toml"
    record_reject(str(f), "n0001.fits", "cloud")
    assert "n0001.fits" in load_rejects(str(f))

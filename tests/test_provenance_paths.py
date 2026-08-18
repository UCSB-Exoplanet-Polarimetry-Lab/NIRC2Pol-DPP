"""The audit trail: provenance headers, and the reject list."""

import datetime
import logging
import os

import numpy as np
from astropy.io import fits
import pytest

from utils.frame import Frame
from utils.paths import ObslogPaths, load_rejects, record_reject
from utils.provenance import (describe, drop_step, record_step,
                              steps_of)


def test_record_step_round_trips():
    """A recorded step, with its parameters, reads back out of the header."""
    frame = Frame(np.zeros((4, 4)), {})
    record_step(frame, "dark/flat reduction", darksub=True, gain=8.0)
    steps = steps_of(frame)
    assert any("dark/flat reduction" in s for s in steps)
    # record_step formats floats with %.6g, so 8.0 renders as "8"
    assert any("gain=8" in s for s in steps)


def test_record_step_accumulates_in_order():
    """Several steps accumulate in the order they were recorded."""
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
    """An unprocessed product still describes itself without raising."""
    assert isinstance(describe(Frame(np.zeros((2, 2)), {})), str)


# --- rejects -----------------------------------------------------------

def test_load_rejects_missing_file(tmp_path):
    """A missing reject file gives an empty mapping, not an error."""
    assert load_rejects(str(tmp_path / "nope.toml")) == {}


def test_load_rejects_list_form(tmp_path):
    """The original format, which carries no reasons."""
    f = tmp_path / "r.toml"
    f.write_text('rejects = ["n0001.fits", "n0002.fits"]\n')
    assert load_rejects(str(f)) == {"n0001.fits": "", "n0002.fits": ""}


def test_load_rejects_table_form_keeps_reasons(tmp_path):
    """The table form preserves why each frame was dropped."""
    f = tmp_path / "r.toml"
    f.write_text('[rejects]\n"n0003.fits" = "open AO loop"\n')
    assert load_rejects(str(f)) == {"n0003.fits": "open AO loop"}


def test_record_reject_upgrades_a_list_file_in_place(tmp_path):
    """Writing to a legacy list file upgrades it and keeps its entries."""
    f = tmp_path / "r.toml"
    f.write_text('rejects = ["n0001.fits"]\n')
    record_reject(str(f), "n0009.fits", "satellite trail")

    got = load_rejects(str(f))
    assert got["n0001.fits"] == "", "the existing entry must survive"
    assert got["n0009.fits"] == "satellite trail"


def test_record_reject_takes_a_basename_from_a_path(tmp_path):
    """A full path is reduced to its basename, matching how rejects are used."""
    f = tmp_path / "r.toml"
    record_reject(str(f), "/data/2026-06-05/raw/n0042.fits", "cloud")
    assert "n0042.fits" in load_rejects(str(f))


def test_record_reject_updates_an_existing_reason(tmp_path):
    """Re-recording a frame replaces its reason rather than duplicating it."""
    f = tmp_path / "r.toml"
    record_reject(str(f), "n0001.fits", "first guess")
    record_reject(str(f), "n0001.fits", "actually the AO loop opened")
    got = load_rejects(str(f))
    assert len(got) == 1
    assert got["n0001.fits"] == "actually the AO loop opened"


def test_record_reject_escapes_quotes_and_backslashes(tmp_path):
    """Quotes and backslashes in a reason survive the TOML round trip."""
    f = tmp_path / "r.toml"
    messy = 'seeing was "bad", see C:\\logs'
    record_reject(str(f), "n0001.fits", messy)
    assert load_rejects(str(f))["n0001.fits"] == messy


def test_rejects_membership_works_for_load_frames(tmp_path):
    """load_frames tests `basename in rejects`, so a dict must behave."""
    f = tmp_path / "r.toml"
    record_reject(str(f), "n0001.fits", "cloud")
    assert "n0001.fits" in load_rejects(str(f))


def test_a_long_parameter_list_survives_the_round_trip():
    """record_step wraps long lines across HISTORY cards. The reader has to
    rejoin them: writing the data to disk is no use if every reader shows
    only the first 68 characters, which is what happened before -- steps_of
    filtered on the DPP prefix and the continuation cards do not carry it."""
    header = fits.Header()
    record_step(header, "dark/flat reduction",
                dark="n1013.fits", flat="n0024.fits", polflat=True,
                flat_checked=True, flat_mismatch=False, gain=8.0,
                saturation=4500.0, div_coadds=True, div_itime=False,
                badpix="interpolation")

    assert len(header["HISTORY"]) > 1, "this list must be long enough to wrap"

    step = steps_of(header)[0]
    for expected in ("dark=n1013.fits", "flat_checked=T", "flat_mismatch=F",
                     "badpix=interpolation"):
        assert expected in step, f"{expected} lost in the wrap"
    assert "    " not in step, "continuation padding leaked into the text"


def test_drop_step_removes_a_wrapped_step_whole():
    """A wrapped step must go with its continuations, or the leftovers get
    reattached to whatever step precedes them."""
    header = fits.Header()
    header.add_history("not ours, leave alone")
    record_step(header, "keep me", a=1)
    record_step(header, "dark/flat reduction", dark="n1013.fits",
                flat="n0024.fits", gain=8.0, saturation=4500.0,
                div_coadds=True, badpix="interpolation", polflat=True)

    assert drop_step(header, "dark/flat reduction") == 1
    remaining = steps_of(header)
    assert len(remaining) == 1 and "keep me" in remaining[0]
    assert not any("n1013" in str(h) for h in header["HISTORY"]), \
        "a continuation card survived its step"
    assert any("not ours" in str(h) for h in header["HISTORY"]), \
        "someone else's HISTORY was collateral damage"


def test_record_step_replace_keeps_only_the_latest():
    header = fits.Header()
    record_step(header, "stokes cube", replace=True, fast_axis_offset=-13)
    record_step(header, "stokes cube", replace=True, fast_axis_offset=-11)
    steps = steps_of(header)
    assert len(steps) == 1
    assert "fast_axis_offset=-11" in steps[0]


def test_master_filenames_carry_the_date():
    """Masters belong to the dataset they were taken with; darks and flats
    are taken with every dataset and are not interchangeable."""
    paths = ObslogPaths("/data", "2025-12-07")
    for attr in ("darks_file", "flats_file", "skies_file", "masks_file"):
        name = os.path.basename(getattr(paths, attr))
        assert "2025-12-07" in name, f"{attr} is not dated: {name}"
        assert name.startswith("master_")


def test_masters_record_how_they_were_built():
    """A master outlives its reduction, so it has to say what made it."""
    from reduction.masters import make_master_darks
    darks = [Frame(np.ones((8, 8)) * i,
                   {"NAXIS1": 8, "NAXIS2": 8, "ITIME": 1.0, "COADDS": 1,
                    "SAMPMODE": 3, "READS": 1, "FILENAME": f"d{i}.fits"})
             for i in range(3)]
    masters, _ = make_master_darks(darks)

    step = steps_of(masters[0].header)
    assert step, "a master with no provenance cannot be audited later"
    assert "dark combination" in step[0]
    assert "nframes=3" in step[0]
    assert "DPPVER" in masters[0].header, "no pipeline version on the master"


def test_provenance_timestamps_are_utc():
    """Every other date in a NIRC2 product is UTC. A naive local stamp
    cannot be compared with any of them, nor with the same product reduced
    on a machine in another timezone."""
    header = fits.Header()
    record_step(header, "test step", a=1)

    stamped = datetime.datetime.fromisoformat(header["DPPDATE"])
    assert stamped.tzinfo is not None, "the stamp does not say which zone"
    assert stamped.utcoffset() == datetime.timedelta(0)

    now = datetime.datetime.now(datetime.timezone.utc)
    assert abs((now - stamped).total_seconds()) < 300, \
        "stamp is not the current UTC time (local time would show as an offset)"
    assert steps_of(header)[0].rstrip("]").endswith("Z"), \
        "the step timestamp should be marked UTC"


def test_a_folder_dated_for_the_hst_evening_is_flagged(caplog):
    """A Keck night is one UTC date, one day after the local evening. Naming
    the folder for the evening is how that drifts, and masters and products
    inherit the folder's date."""
    frames = [Frame(np.zeros((2, 2)), {"DATE-OBS": "2025-12-08"})]

    with caplog.at_level(logging.WARNING):
        found = ObslogPaths("/data", "2025-12-07").check_frame_dates(frames)
    assert found == {"2025-12-08"}
    assert "2025-12-08" in " ".join(r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ObslogPaths("/data", "2025-12-08").check_frame_dates(frames)
    assert not caplog.records, "the matching date must not warn"

"""Frame I/O, and the header scrubbing that keeps some nights saveable."""

import numpy as np
import pytest
from astropy.io import fits

from utils.frame import (Frame, get_between, load_master, match_keys,
                         save_frames, scrub_header)


def _header_with_bad_continue():
    """A header carrying the malformed CONTINUE that astropy refuses to write.

    Astropy stores a long card plus its CONTINUEs as one consolidated card
    whose ``_image`` is a concatenation of 80-byte chunks. The 2026-05-26
    NIRC2 reduction contains a CONTINUE whose value field is not a quoted
    string; without scrubbing, every product from that night fails to save.
    """
    good = fits.Card.fromstring("OBJECT  = 'AB Aur'".ljust(80))
    keep = fits.Card.fromstring("ITIME   = 0.45".ljust(80))
    broken = fits.Card.fromstring(
        "LONGKEY = 'start of a long value&'".ljust(80)
        + "CONTINUE  not_a_quoted_string".ljust(80))
    header = fits.Header()
    for card in (good, keep, broken):
        header.append(card, end=True)
    return header


def test_scrub_header_keeps_the_good_keywords():
    scrubbed = scrub_header(_header_with_bad_continue())
    assert scrubbed["OBJECT"] == "AB Aur"
    assert scrubbed["ITIME"] == pytest.approx(0.45)


def test_scrub_header_lets_the_frame_save(tmp_path):
    """The point of the exercise: a product from that night can be written."""
    out = tmp_path / "scrubbed.fits"
    frame = Frame(np.ones((4, 4)), _header_with_bad_continue())
    frame.save(str(out))
    assert out.exists()
    assert Frame.load(str(out))["OBJECT"] == "AB Aur"


def test_scrub_header_survives_in_memory_cards():
    """Cards built in memory have ``_image = None``.

    Not handling that was a second crash on top of the first, so it gets its
    own case: a header mixing file-parsed and in-memory cards must scrub.
    """
    header = _header_with_bad_continue()
    header["INMEM"] = (7, "built in memory, no _image")
    scrubbed = scrub_header(header)
    assert scrubbed["INMEM"] == 7
    assert scrubbed["OBJECT"] == "AB Aur"


def test_frame_save_load_round_trip(tmp_path):
    out = tmp_path / "f.fits"
    data = np.arange(24, dtype=float).reshape(4, 6)
    Frame(data, {"OBJECT": "test", "ITIME": 2.0}).save(str(out))
    back = Frame.load(str(out))
    assert np.allclose(back.data, data)
    assert back["OBJECT"] == "test"
    assert back["FILENAME"] == "f.fits"


def test_save_frames_and_load_master_round_trip(tmp_path):
    out = tmp_path / "masters.fits"
    frames = [Frame(np.full((3, 3), float(i)), {"NFRAMES": i})
              for i in range(1, 4)]
    save_frames(str(out), frames)
    back = load_master(str(out))
    assert len(back) == 3
    assert [np.mean(f.data) for f in back] == pytest.approx([1.0, 2.0, 3.0])


def test_get_between_selects_a_frame_number_range():
    frames = [Frame(np.zeros((2, 2)), {"FRAMENO": n}) for n in (5, 6, 7, 8, 9)]
    assert [f["FRAMENO"] for f in get_between(frames, (6, 8))] == [6, 7, 8]


def test_match_keys_groups_by_header_values():
    frames = [Frame(np.zeros((2, 2)), {"FILTER": f, "ITIME": t})
              for f, t in (("H", 1.0), ("H", 1.0), ("Kp", 1.0))]
    groups = match_keys(frames, ["FILTER", "ITIME"])
    assert len(groups) == 2
    assert sorted(len(v) for v in groups.values()) == [1, 2]

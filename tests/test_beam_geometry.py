"""Beam extraction geometry: the unsafe default is gone, and a wrong one is
caught rather than silently folded into the double difference."""

import logging

import numpy as np
import pytest

from instruments.nirc2 import NIRC2PolarimetryData
from reduction import measure_beam_offset, register_beam_stack

BEAM_H, NY, NX = 60, 140, 80
TOP, XOFF = 70, 5
STAR = (28.0, 30.0)          # (y, x) of the star inside the bottom beam


class TinyNIRC2(NIRC2PolarimetryData):
    """NIRC2 on a small synthetic detector, with the geometry set."""

    beam_height = BEAM_H
    bottom_row_start = 0
    top_row_start = TOP
    beam_x_offset = XOFF
    background_method = None


def _gaussian(shape, cy, cx, sigma=2.5, amp=1000.0):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def detector(dy=0.0, dx=0.0):
    """A frame whose two beams are offset from each other by (dy, dx) once
    split with the nominal geometry."""
    y, x = STAR
    return (_gaussian((NY, NX), y, x)
            + _gaussian((NY, NX), TOP + y + dy, XOFF + x + dx))


def test_split_beams_refuses_to_guess_the_geometry():
    """No default: a wrong one is silent and unrecoverable, so it must be set."""
    with pytest.raises(ValueError) as excinfo:
        NIRC2PolarimetryData().split_beams(detector())
    message = str(excinfo.value)
    assert "top_row_start" in message and "beam_x_offset" in message
    # the error has to say how to fix it, not just that it is broken
    assert "fit_beam_geometry" in message


def test_split_beams_accepts_a_trial_geometry():
    """Overrides are what make the geometry measurable in the first place."""
    inst = TinyNIRC2()
    star_row = int(STAR[0])
    shifted = inst.split_beams(detector(), top_row_start=TOP + 4)
    # the top beam now starts 4 rows late, so its star sits 4 rows earlier
    assert np.argmax(shifted[1][:, int(STAR[1])]) == star_row - 4
    assert np.argmax(shifted[0][:, int(STAR[1])]) == star_row


@pytest.mark.parametrize("dy, dx", [(0.0, 0.0), (3.0, -2.0), (-4.0, 1.5)])
def test_measure_beam_offset_recovers_an_injected_offset(dy, dx):
    stack = TinyNIRC2().split_beams(detector(dy, dx))
    got_dy, got_dx = measure_beam_offset(stack)
    assert got_dy == pytest.approx(dy, abs=0.15)
    assert got_dx == pytest.approx(dx, abs=0.15)


def test_measure_beam_offset_rejects_a_non_beam_stack():
    with pytest.raises(ValueError, match=r"\(2, ny, nx\)"):
        measure_beam_offset(np.zeros((3, 20, 20)))


def test_register_beam_stack_warns_when_the_beams_disagree(caplog):
    """The offset survives registration by design, so this is the last place
    it can be reported."""
    stack = TinyNIRC2().split_beams(detector(dy=4.0, dx=0.0))
    with caplog.at_level(logging.WARNING):
        register_beam_stack(stack, method="centroid")
    assert any("misaligned" in r.getMessage() for r in caplog.records)


def test_register_beam_stack_is_quiet_when_the_beams_agree(caplog):
    stack = TinyNIRC2().split_beams(detector())
    with caplog.at_level(logging.WARNING):
        register_beam_stack(stack, method="centroid")
    assert not any("misaligned" in r.getMessage() for r in caplog.records)


def test_alignment_check_can_be_switched_off(caplog):
    """A source the centroid cannot locate per beam would warn forever."""
    stack = TinyNIRC2().split_beams(detector(dy=4.0, dx=0.0))
    with caplog.at_level(logging.WARNING):
        register_beam_stack(stack, method="centroid",
                            check_beam_alignment=False)
    assert not any("misaligned" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("trial", [(TOP + 3, XOFF - 2), (TOP - 2, XOFF + 1),
                                   (TOP, XOFF)])
def test_fit_beam_geometry_converges_from_a_wrong_trial(trial):
    """One pass is exact: the residual is a pure translation."""
    assert TinyNIRC2().fit_beam_geometry(detector(), *trial) == (TOP, XOFF)


def test_fit_beam_geometry_flags_a_half_pixel_tie(caplog):
    """Rounding decides these arbitrarily, so the caller must be told."""
    inst = TinyNIRC2()
    with caplog.at_level(logging.WARNING):
        inst.fit_beam_geometry(detector(dx=0.5), TOP, XOFF)
    assert any("between pixels" in r.getMessage() for r in caplog.records)

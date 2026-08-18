"""Beam extraction geometry: the unsafe default is gone, and a wrong one is
caught rather than silently folded into the double difference."""

import logging

import numpy as np
import pytest

from instruments.nirc2 import NIRC2PolarimetryData
from reduction import (fit_beam_geometry, measure_beam_offset,
                       register_beam_stack)

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
    assert fit_beam_geometry(TinyNIRC2(), detector(), *trial) == (TOP, XOFF)


def test_fit_beam_geometry_flags_a_half_pixel_tie(caplog):
    """Rounding decides these arbitrarily, so the caller must be told."""
    inst = TinyNIRC2()
    with caplog.at_level(logging.WARNING):
        fit_beam_geometry(inst, detector(dx=0.5), TOP, XOFF)
    assert any("between pixels" in r.getMessage() for r in caplog.records)


def test_crop_does_not_change_the_measured_offset():
    """The same box is cut from both beams, so the difference must survive."""
    stack = TinyNIRC2().split_beams(detector(dy=3.0, dx=-1.5))
    full = measure_beam_offset(stack)
    cropped = measure_beam_offset(stack, crop_size=40)
    assert cropped[0] == pytest.approx(full[0], abs=0.05)
    assert cropped[1] == pytest.approx(full[1], abs=0.05)


def test_crop_falls_back_when_the_box_does_not_fit():
    """A box larger than the beam must not cost us the check entirely."""
    stack = TinyNIRC2().split_beams(detector(dy=3.0, dx=0.0))
    assert measure_beam_offset(stack, crop_size=10_000)[0] == pytest.approx(
        3.0, abs=0.15)


def test_implausible_offset_is_a_failed_measurement_not_a_misalignment(caplog):
    """Two beams of one field cannot really be 120 px apart; saying
    "misaligned" there would send someone hunting a geometry bug that is
    actually a centering failure."""
    scene = np.zeros((300, 300))
    scene[145:155, 145:155] = 1000.0
    stack = np.stack([scene, np.roll(scene, 120, axis=0)])
    with caplog.at_level(logging.WARNING):
        register_beam_stack(stack, method="centroid")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "failed measurement" in text
    assert "Beams are misaligned by" not in text


def test_a_credible_offset_is_still_called_a_misalignment(caplog):
    """The cutoff must not swallow the errors the guard exists to catch."""
    stack = TinyNIRC2().split_beams(detector(dy=4.0, dx=0.0))
    with caplog.at_level(logging.WARNING):
        register_beam_stack(stack, method="centroid")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "Beams are misaligned by" in text
    assert "failed measurement" not in text


# --- measured, not looked up ------------------------------------------------

def test_the_seed_is_only_a_starting_point():
    """A seed several pixels off must reach the same answer, or it is a
    default in disguise."""
    answers = {fit_beam_geometry(TinyNIRC2(), detector(), TOP + d, XOFF)
               for d in (-4, -2, 0, 2, 4)}
    assert answers == {(TOP, XOFF)}, f"seed-dependent: {answers}"


def test_several_frames_are_averaged_before_rounding():
    """Rounding each frame separately lets a geometry near a half pixel come
    out differently frame to frame; averaging the sub-pixel offsets first
    settles it once."""
    inst = TinyNIRC2()
    # rounded separately these disagree -- 0.2 gives +0, 1.2 gives +1
    assert fit_beam_geometry(inst, detector(dx=0.2), TOP, XOFF) == (TOP, XOFF)
    assert fit_beam_geometry(inst, detector(dx=1.2), TOP, XOFF) == (TOP, XOFF + 1)

    # averaged first, 0.7 rounds once, to +1
    combined = fit_beam_geometry(inst, [detector(dx=0.2), detector(dx=1.2)],
                                 TOP, XOFF)
    assert combined == (TOP, XOFF + 1)


def test_a_single_frame_is_accepted_as_well_as_a_list():
    """A bare ndarray has no .header, and list() on a 2D array iterates its
    rows -- so a naive check turns one image into a list of 1D slices."""
    one = fit_beam_geometry(TinyNIRC2(), detector(), TOP, XOFF)
    listed = fit_beam_geometry(TinyNIRC2(), [detector()], TOP, XOFF)
    assert one == listed


def test_the_instrument_describes_the_geometry_it_used():
    """Measured per reduction, so a product cannot be reproduced unless it
    records which values produced it."""
    inst = TinyNIRC2()
    assert str(TOP) in inst.describe_beam_geometry()
    assert str(XOFF) in inst.describe_beam_geometry()

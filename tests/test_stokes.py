"""Double differencing: the identities the whole reduction rests on."""

import logging

import numpy as np

from utils import Frame
import pytest

from conftest import NX, NY, synth_cycle
from polarimetry.stokes import (_check_cycle_exposure, build_stokes_cube,
                                double_difference,
                                radial_stokes, rotate_qu, single_difference)


def test_single_difference_convention():
    """beam 0 is the bottom, beam 1 the top; the difference is top - bottom."""
    stack = np.stack([np.full((4, 4), 2.0), np.full((4, 4), 5.0)])
    d, s = single_difference(stack)
    assert np.allclose(d, 3.0)
    assert np.allclose(s, 7.0)


def test_static_additive_pattern_cancels(instrument):
    """A detector pattern present in every exposure double-differences away.

    This is why the chevron pattern in the Cygnus A Kp data survives into
    Stokes I but is suppressed roughly fourteen-fold in Q and U.
    """
    cycle = synth_cycle(theta_off=0.0, ipq=0.0, ipu=0.0, amplitude=1.0)
    rng = np.random.default_rng(0)
    pattern = rng.normal(0, 50.0, size=(2 * NY, NX))
    dirty = []
    for f in cycle:
        g = f.copy()
        g.data = g.data + pattern
        dirty.append(g)

    Q0, U0, _ = double_difference(instrument, cycle, register_method=None)
    Q1, U1, _ = double_difference(instrument, dirty, register_method=None)
    assert np.allclose(Q0, Q1, atol=1e-9)
    assert np.allclose(U0, U1, atol=1e-9)


def test_stokes_cube_recovers_injected_disk(instrument, truth):
    """The full chain returns the tangential disk it was given.

    Only true when handed the injected theta_off: this is the test that
    would catch a sign or factor error in the rotation model.
    """
    cycle = synth_cycle(truth["theta_off"], parang=15.0, el=50.0, rot=30.0)
    cube = build_stokes_cube(instrument, cycle,
                             fast_axis_offset=truth["theta_off"],
                             register_method=None, derotate=False)
    q_phi, u_phi = radial_stokes(cube[1], cube[2])
    yy, xx = np.mgrid[:NY, :NX]
    r = np.hypot(yy - (NY - 1) / 2, xx - (NX - 1) / 2)
    ring = (r >= 14) & (r <= 32)
    assert np.nansum(q_phi[ring]) > 0
    assert abs(np.nansum(u_phi[ring])) < 0.01 * np.nansum(q_phi[ring])


def test_wrong_offset_leaks_into_uphi(instrument, truth):
    """The signal the on-sky fit keys on: a bad offset spills into U_phi."""
    cycle = synth_cycle(truth["theta_off"], parang=15.0)
    cube = build_stokes_cube(instrument, cycle,
                             fast_axis_offset=truth["theta_off"] + 5.0,
                             register_method=None, derotate=False)
    q_phi, u_phi = radial_stokes(cube[1], cube[2])
    yy, xx = np.mgrid[:NY, :NX]
    r = np.hypot(yy - (NY - 1) / 2, xx - (NX - 1) / 2)
    ring = (r >= 14) & (r <= 32)
    assert abs(np.nansum(u_phi[ring])) > 0.05 * abs(np.nansum(q_phi[ring]))


def test_rotate_qu_round_trip():
    """Rotating Q/U and back returns the original, so the sign is consistent."""
    Q, U = np.array([[1.0]]), np.array([[0.5]])
    q, u = rotate_qu(*rotate_qu(Q, U, 37.0), -37.0)
    assert q == pytest.approx(Q)
    assert u == pytest.approx(U)


def test_radial_stokes_sign_convention():
    """SPIE Eq. 6: tangential polarization gives positive Q_phi.

    Verified on sky against AB Aur; do not 'fix' this without rechecking
    against real data.
    """
    ny = nx = 64
    yy, xx = np.mgrid[:ny, :nx]
    phi = np.arctan2(yy - (ny - 1) / 2, xx - (nx - 1) / 2)
    Q, U = np.cos(2 * phi), np.sin(2 * phi)      # purely tangential
    q_phi, u_phi = radial_stokes(Q, U)
    r = np.hypot(yy - (ny - 1) / 2, xx - (nx - 1) / 2)
    ring = (r > 8) & (r < 28)
    assert np.all(q_phi[ring] > 0.99)
    assert np.allclose(u_phi[ring], 0.0, atol=1e-9)


def test_incomplete_cycle_raises(instrument):
    """A cycle missing a critical angle raises rather than silently combining."""
    cycle = synth_cycle(0.0)[:2]
    with pytest.raises(ValueError, match="No frames at modulator angle"):
        double_difference(instrument, cycle, register_method=None)


def test_rebuilding_a_cycle_replaces_its_record_rather_than_stacking(instrument,
                                                                     truth):
    """build_stokes_cube writes its provenance to cycle[0], an *input*, so the
    writers can find it. Scanning fast axis offsets would otherwise leave a
    pile of records disagreeing about how the cube in hand was made."""
    from utils.provenance import steps_of

    cycle = synth_cycle(truth["theta_off"], parang=15.0, el=50.0, rot=30.0)
    for offset in (5.0, 6.0, 7.0):
        build_stokes_cube(instrument, cycle, fast_axis_offset=offset,
                          register_method=None, derotate=False)

    records = [s for s in steps_of(cycle[0].header) if "stokes cube" in s]
    assert len(records) == 1, "one build, one record"
    assert "fast_axis_offset=7" in records[0], "and it is the build that ran last"


# --- one exposure per cycle ------------------------------------------------

def _exposure_cycle(itimes, coadds=None):
    """A four-frame cycle with the given per-frame exposure settings."""
    coadds = coadds or [1] * len(itimes)
    return [Frame(np.ones((4, 4)),
                  {"ITIME": t, "COADDS": c, "PCUPR": angle})
            for t, c, angle in zip(itimes, coadds, (0.0, 45.0, 22.5, 67.5))]


def test_a_uniform_cycle_says_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        _check_cycle_exposure(_exposure_cycle([0.45] * 4))
    assert not caplog.records


def test_a_mixed_exposure_cycle_is_flagged(caplog):
    """The double difference subtracts frames from each other, so unequal
    depth would contribute unequally. ITIME division normalises that, which
    is why this warns rather than refuses -- but an exposure change inside
    one HWP cycle is worth knowing about."""
    with caplog.at_level(logging.WARNING):
        _check_cycle_exposure(_exposure_cycle([0.45, 0.45, 0.9, 0.45]))
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "ITIME" in text
    assert "0.45" in text and "0.9" in text, "the values found must be named"


def test_mixed_coadds_are_flagged_too(caplog):
    with caplog.at_level(logging.WARNING):
        _check_cycle_exposure(_exposure_cycle([0.45] * 4, coadds=[45, 45, 1, 45]))
    assert any("COADDS" in r.getMessage() for r in caplog.records)


def test_every_mixed_cycle_is_reported_not_just_the_first(caplog):
    """Warning once would hide every case after the first, and each mixed
    cycle is a separate fact about the data."""
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            _check_cycle_exposure(_exposure_cycle([0.45, 0.9, 0.45, 0.45]))
    assert len([r for r in caplog.records if "ITIME" in r.getMessage()]) == 3

"""Double differencing: the identities the whole reduction rests on."""

import numpy as np
import pytest

from conftest import NX, NY, synth_cycle
from polarimetry.stokes import (build_stokes_cube, double_difference,
                                radial_stokes, rotate_qu, single_difference)


def test_single_difference_convention():
    """beam 0 is the bottom, beam 1 the top; the difference is top - bottom."""
    stack = np.stack([np.full((4, 4), 2.0), np.full((4, 4), 5.0)])
    d, s = single_difference(stack)
    assert np.allclose(d, 3.0)
    assert np.allclose(s, 7.0)


def test_unpolarized_source_gives_zero_qu(instrument):
    """No injected polarization anywhere, so Q and U must vanish."""
    cycle = synth_cycle(theta_off=0.0, ipq=0.0, ipu=0.0, amplitude=0.0)
    Q, U, I = double_difference(instrument, cycle, register_method=None)
    assert np.allclose(Q, 0.0, atol=1e-9)
    assert np.allclose(U, 0.0, atol=1e-9)
    assert np.all(I > 0)


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
    cycle = synth_cycle(0.0)[:2]
    with pytest.raises(ValueError, match="No frames at modulator angle"):
        double_difference(instrument, cycle, register_method=None)

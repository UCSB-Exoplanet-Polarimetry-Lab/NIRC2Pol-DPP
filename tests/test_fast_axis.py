"""On-sky fast axis determination."""

import numpy as np
import pytest

from conftest import NX, NY, disk_radial_stokes, synth_cycle
from polarimetry.fast_axis import (OFFSET_TO_FRAME, butterfly_phase,
                                   combine_at_offset, fit_fast_axis_on_sky,
                                   prepare_cycles, scan_fast_axis_offset,
                                   wrap_offset)


def _rotated_butterfly(delta_deg, amplitude=1.0):
    """Q/U for a tangential disk whose frame is turned by ``delta_deg``."""
    q_phi, u_phi, _, phi = disk_radial_stokes(amplitude=amplitude)
    Q = q_phi * np.cos(2 * phi) - u_phi * np.sin(2 * phi)
    U = q_phi * np.sin(2 * phi) + u_phi * np.cos(2 * phi)
    c, s = np.cos(np.radians(delta_deg)), np.sin(np.radians(delta_deg))
    return Q * c - U * s, Q * s + U * c


def test_butterfly_phase_zero_on_tangential_disk():
    Q, U = _rotated_butterfly(0.0)
    assert butterfly_phase(Q, U, r_inner=10, r_outer=36) == pytest.approx(
        0.0, abs=1e-6)


@pytest.mark.parametrize("delta", [-90.0, -25.0, -3.0, 5.0, 40.0, 120.0])
def test_butterfly_phase_recovers_injected_rotation(delta):
    Q, U = _rotated_butterfly(delta)
    got = butterfly_phase(Q, U, r_inner=10, r_outer=36)
    assert got == pytest.approx(delta, abs=1e-6)


def test_butterfly_phase_branch_keeps_qphi_positive():
    """A radially polarized source is 180 deg away, not 0."""
    Q, U = _rotated_butterfly(180.0)
    assert abs(butterfly_phase(Q, U, r_inner=10, r_outer=36)) == pytest.approx(
        180.0, abs=1e-6)


def test_butterfly_phase_rejects_empty_annulus():
    Q, U = _rotated_butterfly(0.0)
    with pytest.raises(ValueError, match="empty"):
        butterfly_phase(Q, U, r_inner=500, r_outer=600)


def test_wrap_offset_folds_45_degree_degeneracy():
    for base in (-3.25, 0.0, 11.0, 22.4):
        for k in (-2, -1, 0, 1, 2):
            assert wrap_offset(base + 45.0 * k) == pytest.approx(base)


def test_offset_to_frame_is_four():
    """theta_rot carries 4*theta_off, so one degree of offset is four of frame."""
    assert OFFSET_TO_FRAME == 4.0


def test_prepare_cycles_leaves_offset_free(instrument, clean_cycles, truth):
    """A prepared cycle rotated at the injected offset gives back Q_phi."""
    from polarimetry.stokes import radial_stokes

    prepared = prepare_cycles(instrument, clean_cycles, derotate=False,
                              register_method=None)
    Q, U, _ = combine_at_offset(prepared, truth["theta_off"])
    q_phi, u_phi = radial_stokes(Q, U)
    yy, xx = np.mgrid[:NY, :NX]
    r = np.hypot(yy - (NY - 1) / 2, xx - (NX - 1) / 2)
    ring = (r >= 14) & (r <= 32)
    assert np.nansum(q_phi[ring]) > 0
    assert abs(np.nansum(u_phi[ring])) < 0.01 * abs(np.nansum(q_phi[ring]))


def test_fit_fast_axis_recovers_offset_without_ip(instrument, clean_cycles,
                                                  truth):
    res = fit_fast_axis_on_sky(instrument, clean_cycles, fit_ip=False,
                               r_inner=13.0, r_outer=33.0, derotate=False,
                               register_method=None)
    assert res.theta_off == pytest.approx(truth["theta_off"], abs=0.05)
    assert res.converged


def test_fit_fast_axis_joint_with_ip(instrument, cycles, truth):
    """With leakage present, the joint fit still finds the offset."""
    res = fit_fast_axis_on_sky(instrument, cycles, fit_ip=True,
                               r_inner=13.0, r_outer=33.0, derotate=False,
                               register_method=None)
    assert res.theta_off == pytest.approx(truth["theta_off"], abs=0.5)
    assert res.ip is not None


def test_ip_biases_the_offset_when_ignored(instrument, cycles, truth):
    """The degeneracy this module warns about, made explicit.

    Fitting the offset with the leakage left in gives a different answer
    from fitting them together. If this ever stops being true the joint fit
    has stopped doing anything.
    """
    ignored = fit_fast_axis_on_sky(instrument, cycles, fit_ip=False,
                                   r_inner=13.0, r_outer=33.0, derotate=False,
                                   register_method=None)
    joint = fit_fast_axis_on_sky(instrument, cycles, fit_ip=True,
                                 r_inner=13.0, r_outer=33.0, derotate=False,
                                 register_method=None)
    assert abs(joint.theta_off - truth["theta_off"]) < \
        abs(ignored.theta_off - truth["theta_off"])


def test_scan_has_minimum_at_the_fitted_offset(instrument, clean_cycles,
                                               truth):
    prepared = prepare_cycles(instrument, clean_cycles, derotate=False,
                              register_method=None)
    offsets, scores = scan_fast_axis_offset(prepared, r_inner=13.0,
                                            r_outer=33.0)
    best = offsets[np.argmin(scores)]
    assert best == pytest.approx(truth["theta_off"], abs=0.5)


def test_scan_returns_matching_lengths(instrument, clean_cycles):
    prepared = prepare_cycles(instrument, clean_cycles, derotate=False,
                              register_method=None)
    grid = np.arange(-10.0, 10.0, 1.0)
    offsets, scores = scan_fast_axis_offset(prepared, offsets=grid,
                                            r_inner=13.0, r_outer=33.0)
    assert len(offsets) == len(scores) == len(grid)
    assert np.all(np.isfinite(scores))


def test_offset_is_degenerate_modulo_45(instrument, truth):
    """Injecting theta_off + 45 deg is the same physical configuration."""
    shifted = [synth_cycle(truth["theta_off"] + 45.0, 0.0, 0.0, parang=pa)
               for pa in (-20.0, 0.0, 20.0)]
    res = fit_fast_axis_on_sky(instrument, shifted, fit_ip=False,
                               r_inner=13.0, r_outer=33.0, derotate=False,
                               register_method=None)
    assert res.theta_off == pytest.approx(wrap_offset(truth["theta_off"]),
                                          abs=0.05)

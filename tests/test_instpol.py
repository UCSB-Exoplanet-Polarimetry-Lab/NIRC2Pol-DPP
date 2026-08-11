"""Instrumental polarization: measurement and removal."""

import numpy as np
import pytest

from conftest import NX, NY, synth_cycle
from polarimetry.instpol import (InstrumentalPolarization, fit_ip_uphi,
                                 mean_ip, measure_ip_annulus,
                                 measure_ip_cycle, subtract_ip)
from polarimetry.stokes import (double_difference,
                                normalized_single_difference, rotate_qu)


def test_subtract_ip_is_exact():
    """Removing a leakage equal to the signal leaves exactly zero."""
    I = np.full((16, 16), 100.0)
    Q = np.full((16, 16), 3.0)
    U = np.full((16, 16), -2.0)
    ip = InstrumentalPolarization(0.03, -0.02, method="manual")
    q, u = subtract_ip(Q, U, I, ip)
    assert np.allclose(q, 0.0)
    assert np.allclose(u, 0.0)


def test_subtract_ip_separate_u_intensity():
    """U can be scaled by its own HWP pair's intensity."""
    I_q, I_u = np.full((8, 8), 100.0), np.full((8, 8), 200.0)
    Q, U = np.full((8, 8), 1.0), np.full((8, 8), 2.0)
    ip = InstrumentalPolarization(0.01, 0.01, method="manual")
    q, u = subtract_ip(Q, U, I_q, ip, I_u=I_u)
    assert np.allclose(q, 0.0)
    assert np.allclose(u, 0.0)


def test_measure_ip_annulus_recovers_injection():
    """The annulus estimator returns the injected leakage exactly."""
    yy, xx = np.mgrid[:NY, :NX]
    r = np.hypot(yy - (NY - 1) / 2, xx - (NX - 1) / 2)
    I = np.exp(-r ** 2 / (2 * 20.0 ** 2)) * 1000 + 1.0
    ip_true = (0.017, -0.009)
    got = measure_ip_annulus(I * ip_true[0], I * ip_true[1], I, 14, 40)
    assert got.ipq == pytest.approx(ip_true[0], abs=1e-9)
    assert got.ipu == pytest.approx(ip_true[1], abs=1e-9)
    assert got.diagnostics["npix"] > 0


def test_measure_ip_annulus_rejects_empty_annulus():
    """An annulus containing no finite pixels raises."""
    a = np.ones((16, 16))
    with pytest.raises(ValueError, match="no finite pixels"):
        measure_ip_annulus(a, a, a, 100, 200)


def test_measure_ip_cycle_recovers_injection(instrument, truth):
    """The mask-edge method, per cycle, on data with a known leakage.

    The annulus sits just outside the disk's inner edge where the halo
    dominates, so the disk's own polarization contributes little.
    """
    cycle = synth_cycle(truth["theta_off"], truth["ipq"], truth["ipu"])
    ip = measure_ip_cycle(instrument, cycle, r_inner=4.0, r_outer=11.0,
                         register_method=None)
    assert ip.ipq == pytest.approx(truth["ipq"], abs=2e-3)
    assert ip.ipu == pytest.approx(truth["ipu"], abs=2e-3)
    assert ip.method == "edge_annulus"


def test_measure_ip_cycle_uses_instrument_occulting_radius(instrument, truth):
    """With no radii given it falls back to the instrument's mask size."""
    cycle = synth_cycle(truth["theta_off"], truth["ipq"], truth["ipu"])
    ip = measure_ip_cycle(instrument, cycle, register_method=None)
    assert ip.diagnostics["r_inner"] == pytest.approx(12.0)


def test_measure_ip_cycle_without_radius_raises():
    """Non-coronagraphic data must be given an explicit annulus."""
    from conftest import SyntheticPolarimetryData

    class NoMask(SyntheticPolarimetryData):
        """An instrument with no coronagraph, to exercise the missing-radius path."""
        def occulting_radius(self, header):
            """No occulting mask on this instrument."""
            return None

    cycle = synth_cycle(0.0)
    with pytest.raises(ValueError, match="No occulting radius"):
        measure_ip_cycle(NoMask(), cycle, register_method=None)


def test_normalized_single_difference_sign_convention(instrument, truth):
    """Pins the beam-order and sign conventions.

    Fixes the chain ``split_beams`` returns ``[bottom, top]`` ->
    ``single_difference`` computes ``top - bottom`` -> an ipq leakage reads
    positive. Mutation testing says only two tests in the suite notice if
    that inverts, and a silent inversion would flip every polarization angle
    downstream.

    The relative flip between HWP 0 and 45 used to be asserted here too, but
    the fixture builds the frames as ``(q_i, -q_i, u_i, -u_i)``, so it was
    injected by construction and could not fail.
    """
    from utils.imutils import make_annulus_mask

    cycle = synth_cycle(theta_off=0.0, ipq=truth["ipq"], ipu=0.0,
                        parang=0.0, el=0.0, rot=0.0)
    mask = make_annulus_mask((NY, NX), 4.0, 11.0)
    r0 = normalized_single_difference(
        instrument.split_beams(cycle[0]), mask)
    assert r0 == pytest.approx(truth["ipq"], abs=2e-3)


def test_normalized_single_difference_rejects_empty_mask(instrument):
    """A mask selecting nothing raises rather than dividing by zero."""
    cycle = synth_cycle(0.0)
    empty = np.zeros((NY, NX), dtype=bool)
    with pytest.raises(ValueError, match="no finite pixels"):
        normalized_single_difference(instrument.split_beams(cycle[0]), empty)


def test_ip_frame_annulus_removes_leakage(instrument, truth):
    """The per-frame hook in double_difference cancels a constant leakage."""
    cycle = synth_cycle(truth["theta_off"], truth["ipq"], truth["ipu"])
    Q_raw, U_raw, I = double_difference(instrument, cycle,
                                        register_method=None)
    Q_fix, U_fix, _ = double_difference(instrument, cycle,
                                        register_method=None,
                                        ip_frame_annulus=(4.0, 11.0))
    yy, xx = np.mgrid[:NY, :NX]
    r = np.hypot(yy - (NY - 1) / 2, xx - (NX - 1) / 2)
    halo = (r >= 4) & (r <= 11)
    # leakage is proportional to I, so it shows up as a net offset there
    assert abs(np.nansum(Q_fix[halo])) < abs(np.nansum(Q_raw[halo])) / 10
    assert abs(np.nansum(U_fix[halo])) < abs(np.nansum(U_raw[halo])) / 10


def test_fit_ip_uphi_recovers_injection(instrument, truth):
    """U_phi minimization finds the leakage on an azimuthally polarized disk."""
    cycle = synth_cycle(truth["theta_off"], truth["ipq"], truth["ipu"])
    ip = fit_ip_uphi(instrument, cycle, truth["theta_off"], mask_radius=6,
                     crop_size=None, register_method=None, derotate=False)
    assert ip.ipq == pytest.approx(truth["ipq"], abs=3e-3)
    assert ip.ipu == pytest.approx(truth["ipu"], abs=3e-3)
    assert ip.diagnostics["uphi_std_final"] < ip.diagnostics["uphi_std_initial"]


def test_fit_ip_uphi_finds_nothing_on_clean_data(instrument, truth):
    """With no leakage injected, the fit returns nothing to remove."""
    cycle = synth_cycle(truth["theta_off"], 0.0, 0.0)
    ip = fit_ip_uphi(instrument, cycle, truth["theta_off"], mask_radius=6,
                     crop_size=None, register_method=None, derotate=False)
    assert abs(ip.ipq) < 3e-3
    assert abs(ip.ipu) < 3e-3


def test_mean_ip_reports_scatter():
    """Averaging keeps the cycle-to-cycle scatter as the error bar."""
    ips = [InstrumentalPolarization(q, u, method="edge_annulus")
           for q, u in ((0.010, 0.004), (0.012, 0.006), (0.008, 0.002))]
    avg = mean_ip(ips)
    assert avg.ipq == pytest.approx(0.010)
    assert avg.ipu == pytest.approx(0.004)
    assert avg.scope == "sequence"
    assert avg.diagnostics["ipq_err"] > 0


def test_ip_magnitude_and_angle():
    """Magnitude and position angle follow from ipq/ipu, wrapped to [0, 180)."""
    ip = InstrumentalPolarization(0.01, 0.0, method="manual")
    assert ip.magnitude == pytest.approx(0.01)
    assert ip.angle == pytest.approx(0.0)
    ip = InstrumentalPolarization(0.0, 0.01, method="manual")
    assert ip.angle == pytest.approx(45.0)
    # a negative u must wrap into [0, 180), not come back negative -- the
    # only case where the modulo in .angle does any work
    ip = InstrumentalPolarization(0.01, -0.01, method="manual")
    assert ip.angle == pytest.approx(157.5)


def test_flux_weighting_beats_per_pixel_ratio(instrument, truth):
    """The estimator is sum(d)/sum(s), not the mean of d/s per pixel.

    Both are unbiased on noiseless data, so the choice only shows up once
    there is noise: a per-pixel ratio weights every pixel equally, including
    the faint ones where the denominator is small and the ratio is wild,
    while the ratio of sums weights by flux. Over an annulus spanning a
    factor of ~5 in intensity the difference is large.

    Without this test the estimator can be swapped for the one the docstring
    warns against and the rest of the suite still passes.
    """
    from utils.imutils import make_annulus_mask

    mask = make_annulus_mask((NY, NX), 4.0, 40.0)
    weighted_err, per_pixel_err = [], []
    for seed in range(12):
        cycle = synth_cycle(theta_off=0.0, ipq=truth["ipq"], ipu=0.0,
                            parang=0.0, el=0.0, rot=0.0,
                            seed=seed, noise=3.0)
        stack = instrument.split_beams(cycle[0])
        d, s = stack[1] - stack[0], stack[1] + stack[0]
        weighted = np.nansum(d[mask]) / np.nansum(s[mask])
        per_pixel = np.nanmean(d[mask] / s[mask])
        weighted_err.append(abs(weighted - truth["ipq"]))
        per_pixel_err.append(abs(per_pixel - truth["ipq"]))

    assert normalized_single_difference(
        instrument.split_beams(cycle[0]), mask) == pytest.approx(weighted)
    assert np.mean(weighted_err) < np.mean(per_pixel_err)

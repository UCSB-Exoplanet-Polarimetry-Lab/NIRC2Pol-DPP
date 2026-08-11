"""End to end: raw FITS on disk through to written products.

Every other test in this suite starts from a hand-built beam stack, which
skips the whole front of the pipeline. This one writes a synthetic night of
raw frames with a known detector model (``raw = signal * flat + dark``) and
a known polarimetric truth, then runs the real code over it: header
classification, master darks and flats, per-frame reduction, cycle matching,
the on-sky fast axis fit, the IP measurement, Stokes cubes, product writing
and the provenance trail.

It is the only test that touches reduction/masters.py, reduction/calibrate.py,
polarimetry/products.py, utils/provenance.py or pipeline.py.
"""

import glob
import os

import numpy as np
import pytest

from conftest import (CRITICAL, E2E_BEAM_HEIGHT, E2E_NX, E2E_XOFF,
                      _e2e_beam_signal)
from pipeline import Pipeline
from polarimetry import (ProductWriter, build_stokes_cubes, fit_fast_axis_on_sky,
                         measure_ip_cycle, median_stokes_cube, radial_stokes,
                         subtract_ip)
from polarimetry.instpol import mean_ip
from reduction import (make_master_darks, make_master_flats, make_master_masks,
                       reduce_frame)
from utils import Frame, load_frames
from utils.provenance import describe


def _disk_annulus(shape):
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - (ny - 1) / 2, xx - (nx - 1) / 2)
    return (r >= 7.0) & (r <= 17.0)


def _reduce_night(night):
    """Sort, build masters, reduce. Returns (sorted_files, reduced, inst)."""
    inst = night["instrument"]
    files = sorted(glob.glob(os.path.join(str(night["dir"]), "*.fits")))
    sorted_files = inst.sort_frames(files)

    master_darks, dark_masks = make_master_darks(
        load_frames(sorted_files["darks"]), bad_pixel_mask=inst.bad_pixel_mask())
    master_flats, flat_masks = make_master_flats(
        load_frames(sorted_files["flats"]), [], [], [], master_darks,
        bad_pixel_mask=inst.bad_pixel_mask(),
        modulator_keyword=inst.modulator_keyword,
        critical_angles=inst.critical_angles)
    masks = make_master_masks(dark_masks, flat_masks)

    reduced = [reduce_frame(f, master_flats, master_darks, None, masks,
                            bad_pixel_mask=inst.bad_pixel_mask(),
                            gain=1.0, saturation_limit=1e12)
               for f in load_frames(sorted_files["sci"])]
    return sorted_files, master_darks, master_flats, reduced, inst


def test_sort_frames_classifies_the_night(synthetic_night):
    """The real header classifier, on headers it has never seen before."""
    inst = synthetic_night["instrument"]
    files = sorted(glob.glob(os.path.join(str(synthetic_night["dir"]), "*.fits")))
    sorted_files = inst.sort_frames(files)

    assert len(sorted_files["darks"]) == 5
    assert len(sorted_files["flats"]) == 12          # lamp-on, no lamp-off pairs
    assert len(sorted_files["sci"]) == 4 * synthetic_night["truth"]["n_cycles"]


def test_masters_and_reduction_invert_the_detector(synthetic_night):
    """reduce_frame must undo raw = signal * flat + dark."""
    _, master_darks, master_flats, reduced, inst = _reduce_night(synthetic_night)

    assert master_darks, "no master dark was built"
    assert master_flats, "no master flat was built"
    assert np.allclose(master_darks[0].data,
                       synthetic_night["dark"], atol=1e-6)

    # the master flat is normalized, so compare its shape not its scale
    mf = master_flats[0].data
    truth = synthetic_night["flat"]
    good = np.isfinite(mf) & (truth > 0.5)
    ratio = (mf / truth)[good]
    assert np.std(ratio) / np.mean(ratio) < 1e-6, "flat shape not recovered"

    # A reduced science frame must be the injected signal back, up to the
    # single global factor the flat normalization introduces. Checking the
    # ratio is *constant* is what makes this bite: skipping the dark leaves
    # a dark/flat term that varies across the detector, and skipping the
    # flat division leaves the flat's own gradient behind. Either one makes
    # the ratio non-constant while leaving the polarimetry almost intact,
    # because a term common to all four HWP angles cancels in the double
    # difference.
    beams = inst.split_beams(reduced[0])
    assert np.all(np.isfinite(beams))
    assert beams.shape == (2, E2E_BEAM_HEIGHT, E2E_NX - E2E_XOFF)

    truth_frame = _e2e_beam_signal(
        synthetic_night["truth"]["theta_off"],
        synthetic_night["truth"]["ipq"], synthetic_night["truth"]["ipu"],
        parang=-20.0, el=50.0, rot=0.0, hwp_index=0)
    truth_beams = inst.split_beams(Frame(truth_frame, {}))
    bright = truth_beams > 0.05 * np.nanmax(truth_beams)
    ratio = beams[bright] / truth_beams[bright]
    assert np.std(ratio) / abs(np.mean(ratio)) < 1e-6, (
        "reduced frame is not a constant multiple of the injected signal: "
        "dark subtraction or flat division is wrong")


def test_end_to_end_recovers_the_injected_polarimetry(synthetic_night,
                                                      tmp_path):
    """The whole chain, then compare against what was injected."""
    truth = synthetic_night["truth"]
    _, _, _, reduced, inst = _reduce_night(synthetic_night)

    cycles = inst.match_modulator_cycles(reduced)
    assert len(cycles) == truth["n_cycles"]

    # fast axis, on sky, from the butterfly
    res = fit_fast_axis_on_sky(inst, cycles, fit_ip=True, r_inner=7.0,
                               r_outer=17.0, derotate=False,
                               register_method=None)
    assert res.theta_off == pytest.approx(truth["theta_off"], abs=0.5)

    # instrumental polarization, from an annulus inside the disk
    ips = [measure_ip_cycle(inst, c, r_inner=2.0, r_outer=5.0,
                            register_method=None) for c in cycles]
    ip = mean_ip(ips)
    assert ip.ipq == pytest.approx(truth["ipq"], abs=4e-3)
    assert ip.ipu == pytest.approx(truth["ipu"], abs=4e-3)

    # Stokes cubes with the recovered calibration
    cubes = build_stokes_cubes(inst, cycles,
                               fast_axis_offset=truth["theta_off"],
                               register_method=None, derotate=False, ip=ip)
    med = median_stokes_cube(cubes)
    q_phi, u_phi = radial_stokes(med[1], med[2])
    ring = _disk_annulus(q_phi.shape)

    assert np.nansum(q_phi[ring]) > 0, "tangential disk did not come back"
    assert abs(np.nansum(u_phi[ring])) < 0.05 * np.nansum(q_phi[ring])

    # products round-trip through disk
    out = tmp_path / "products"
    writer = ProductWriter(str(out), target="Synthetic")
    header = cycles[0][0].header.copy()
    header["THETAOFF"] = res.theta_off
    writer.save_median_stokes(med, header=header, ncycles=len(cycles))
    writer.save_stokes_cycles(cubes, cycles, header=header)

    written = {os.path.basename(f)
               for f in glob.glob(os.path.join(str(out), "*.fits"))}
    assert "Synthetic_median_stokes.fits" in written, (
        f"expected Synthetic_median_stokes.fits, got {sorted(written)}")
    reloaded = Frame.load(os.path.join(str(out), "Synthetic_median_stokes.fits"))
    assert reloaded.data.shape == med.shape
    assert np.allclose(np.nan_to_num(reloaded.data), np.nan_to_num(med))

    # the provenance trail records how it was made
    trail = describe(reloaded)
    assert "median" in trail.lower() or "stokes" in trail.lower()


def test_ip_must_be_removed_before_the_sky_rotation(synthetic_night):
    """Correcting in the wrong frame demonstrably degrades the answer.

    The stronger form of an invariant that was previously asserted only as
    "the two orders differ" -- true of almost any implementation. Here the
    right order recovers the injected disk and the wrong order does not, so
    the test can actually fail.
    """
    truth = synthetic_night["truth"]
    _, _, _, reduced, inst = _reduce_night(synthetic_night)
    cycles = inst.match_modulator_cycles(reduced)
    ip = mean_ip([measure_ip_cycle(inst, c, r_inner=2.0, r_outer=5.0,
                                   register_method=None) for c in cycles])

    # right: ip removed inside double_difference, in the instrument frame
    right = median_stokes_cube(build_stokes_cubes(
        inst, cycles, fast_axis_offset=truth["theta_off"],
        register_method=None, derotate=False, ip=ip))

    # wrong: the same numbers subtracted after rotation to sky
    sky = median_stokes_cube(build_stokes_cubes(
        inst, cycles, fast_axis_offset=truth["theta_off"],
        register_method=None, derotate=False))
    wrong_q, wrong_u = subtract_ip(sky[1], sky[2], sky[0], ip)

    ring = _disk_annulus(right[1].shape)
    _, u_right = radial_stokes(right[1], right[2])
    _, u_wrong = radial_stokes(wrong_q, wrong_u)

    def rms(a):
        return float(np.sqrt(np.nanmean(a[ring] ** 2)))

    # U_phi is the null channel: doing it right leaves less in it.
    # RMS, not the integral -- a leakage subtracted in the sky frame rotates
    # through every azimuth, so it sums to nearly zero while leaving plenty
    # of scatter behind. The integral would call this a pass.
    assert rms(u_right) < 0.1 * rms(u_wrong)


def test_orchestrator_matches_the_function_chain(synthetic_night):
    """pipeline.Pipeline produces the same result as calling the steps.

    The only coverage pipeline.py has, and it also exercises
    run(from_step=...) resuming partway through.
    """
    truth = synthetic_night["truth"]
    inst = synthetic_night["instrument"]
    files = sorted(glob.glob(os.path.join(str(synthetic_night["dir"]), "*.fits")))

    _, _, _, reduced_direct, _ = _reduce_night(synthetic_night)
    direct = median_stokes_cube(build_stokes_cubes(
        inst, inst.match_modulator_cycles(reduced_direct),
        fast_axis_offset=truth["theta_off"], register_method=None,
        derotate=False))

    def step_sort(ctx):
        return ctx["instrument"].sort_frames(files)

    def step_masters(ctx):
        bpm = ctx["instrument"].bad_pixel_mask()
        darks, dmask = make_master_darks(
            load_frames(ctx["sort"]["darks"]), bad_pixel_mask=bpm)
        flats, fmask = make_master_flats(
            load_frames(ctx["sort"]["flats"]), [], [], [], darks,
            bad_pixel_mask=bpm,
            modulator_keyword=ctx["instrument"].modulator_keyword,
            critical_angles=ctx["instrument"].critical_angles)
        return {"darks": darks, "flats": flats,
                "masks": make_master_masks(dmask, fmask)}

    def step_reduce(ctx):
        m = ctx["masters"]
        return [reduce_frame(f, m["flats"], m["darks"], None, m["masks"],
                             bad_pixel_mask=ctx["instrument"].bad_pixel_mask(),
                             gain=1.0, saturation_limit=1e12)
                for f in load_frames(ctx["sort"]["sci"])]

    def step_cycles(ctx):
        return ctx["instrument"].match_modulator_cycles(ctx["reduce"])

    def step_stokes(ctx):
        return median_stokes_cube(build_stokes_cubes(
            ctx["instrument"], ctx["cycles"],
            fast_axis_offset=truth["theta_off"], register_method=None,
            derotate=False))

    pipe = (Pipeline({"instrument": inst})
            .add_step("sort", step_sort)
            .add_step("masters", step_masters)
            .add_step("reduce", step_reduce)
            .add_step("cycles", step_cycles)
            .add_step("stokes", step_stokes))
    ctx = pipe.run()

    assert len(ctx["cycles"]) == truth["n_cycles"]
    assert np.allclose(np.nan_to_num(ctx["stokes"]), np.nan_to_num(direct))

    # resuming partway through must reuse the earlier context, not rerun it
    ctx["cycles"] = ctx["cycles"][:2]
    pipe.run(from_step="stokes")
    assert pipe.context["stokes"].shape == direct.shape
    assert not np.allclose(np.nan_to_num(pipe.context["stokes"]),
                           np.nan_to_num(direct)), \
        "run(from_step=...) appears to have rerun the earlier steps"

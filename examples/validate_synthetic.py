"""End-to-end synthetic validation: inject known polarization, recover it.

Builds fake NIRC2-Pol frames from a known truth -- a star plus a
tangentially-polarized disk, at a known fast-axis offset, known beam
geometry and known telescope angles -- pushes them through the real
pipeline, and checks what comes back is what went in.

The forward model is the exact inverse of the pipeline's own chain, which is
the only way this test is worth anything. Taken from the code, not assumed:

    split_beams      stack[0] = data[0:450, :-dx],  stack[1] = data[top:top+450, dx:]
    single_difference    d = top - bottom,  s = top + bottom
    double_difference    Q = (d_0 - d_45)/2,  U = (d_22.5 - d_67.5)/2
    qu_rotation_angle    theta_rot = -2*PARANG + 2*EL + 2*ROTPDEST + 4*theta_off
    rotate_qu            Q' =  Q cos(t) + U sin(t),  U' = -Q sin(t) + U cos(t)
    radial_stokes        Q_phi =  Q cos(2phi) + U sin(2phi)
                         U_phi = -Q sin(2phi) + U cos(2phi)

So to inject a sky-frame (Q_sky, U_sky) we rotate *backwards* into the
instrument frame and modulate the two beams:

    Q_inst = Q_sky cos(t) - U_sky sin(t)
    U_inst = Q_sky sin(t) + U_sky cos(t)
    top = (I + d)/2,  bottom = (I - d)/2,  d = +-Q_inst or +-U_inst by angle

PARANG/EL/ROTPDEST are held constant within a cycle and varied between
cycles: the pipeline rotates by the cycle *average*, so a within-cycle
gradient would make the truth ambiguous rather than the test meaningful.

Run:
    python examples/validate_synthetic.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from instruments.nirc2 import NIRC2PolarimetryData
from polarimetry import build_stokes_cubes, median_stokes_cube, radial_stokes
from polarimetry.stokes import azimuthal_angle
from reduction import fit_beam_geometry
from utils import Frame

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("validate_synthetic")

# --- the truth we inject ---------------------------------------------------
TOP_ROW_START = 505
BEAM_X_OFFSET = 13
THETA_OFF = -13.1        # deg
P_DISK = 0.08            # tangential polarization fraction in the disk
CRITICAL = (0.0, 45.0, 22.5, 67.5)
NY, NX = 1024, 1024
BEAM_H = 450

# one (PARANG, EL, ROTPDEST) per cycle, constant within it
CYCLE_ANGLES = [(0.0, 0.0, 0.0), (37.0, 51.0, 12.0), (-104.0, 63.0, -25.0),
                (152.0, 44.0, 7.0)]


class SynthPol(NIRC2PolarimetryData):
    """The synthetic instrument: known geometry, no background to remove."""

    top_row_start = TOP_ROW_START
    beam_x_offset = BEAM_X_OFFSET
    background_method = None


def truth_scene(shape):
    """A star plus a tangentially-polarized ring, in the sky frame.

    Returns ``(I, Q_sky, U_sky, disk_mask, q_phi)``. U_phi is identically
    zero by construction, so any U_phi the pipeline reports is error.
    """
    ny, nx = shape
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - cy, xx - cx)

    star = 5000.0 * np.exp(-0.5 * (r / 6.0) ** 2)
    disk = 300.0 * np.exp(-0.5 * ((r - 70.0) / 18.0) ** 2)
    I = star + disk + 20.0

    # tangential: Q_phi = p * I_disk, U_phi = 0. Invert radial_stokes.
    phi = azimuthal_angle(shape, center=(cy, cx))
    q_phi = P_DISK * disk
    Q = q_phi * np.cos(2 * phi)
    U = q_phi * np.sin(2 * phi)
    return I, Q, U, disk > 0.05 * disk.max(), q_phi


def synth_frame(I, Q_inst, U_inst, angle, parang, el, rotpdest):
    """One fake raw frame at one HWP angle, with both beams laid in."""
    a_qp, a_qm, a_up, a_um = CRITICAL
    if np.isclose(angle, a_qp):
        d = Q_inst
    elif np.isclose(angle, a_qm):
        d = -Q_inst
    elif np.isclose(angle, a_up):
        d = U_inst
    else:
        d = -U_inst

    top, bottom = (I + d) / 2.0, (I - d) / 2.0

    data = np.zeros((NY, NX))
    data[0:BEAM_H, 0:NX - BEAM_X_OFFSET] = bottom
    data[TOP_ROW_START:TOP_ROW_START + BEAM_H, BEAM_X_OFFSET:NX] = top

    f = Frame(data)
    f["PCUPR"] = float(angle)
    f["PCUNAME"] = "hwp_center"
    f["PARANG"] = float(parang)
    f["EL"] = float(el)
    f["ROTPDEST"] = float(rotpdest)
    f["TARGNAME"] = "SYNTH"
    f["OBJECT"] = "synthetic_hwp_%.1f" % angle
    f["DATE-OBS"] = "2025-12-08"
    f["ITIME"] = 0.45
    f["COADDS"] = 45
    f["FILTER"] = "Lp + Wollaston"
    f["FWINAME"] = "Lp"
    f["FILENAME"] = "synth_%05.1f.fits" % angle
    return f


def make_cycles(theta_off):
    """Synthetic frames for every cycle, plus the injected sky-frame truth."""
    shape = (BEAM_H, NX - BEAM_X_OFFSET)
    I, Q_sky, U_sky, disk_mask, q_phi = truth_scene(shape)

    cycles = []
    for parang, el, rotpdest in CYCLE_ANGLES:
        p = parang + 360.0 if parang < 0 else parang     # as the pipeline does
        t = np.radians(-2.0 * p + 2.0 * el + 2.0 * rotpdest + 4.0 * theta_off)
        # rotate backwards into the instrument frame
        Q_inst = Q_sky * np.cos(t) - U_sky * np.sin(t)
        U_inst = Q_sky * np.sin(t) + U_sky * np.cos(t)
        cycles.append([synth_frame(I, Q_inst, U_inst, a, parang, el, rotpdest)
                       for a in CRITICAL])
    return cycles, I, Q_sky, U_sky, disk_mask, q_phi


def report(name, got, want, mask, tol):
    """Compare two planes over a mask and say whether it passed."""
    err = np.abs(got - want)[mask]
    scale = np.abs(want)[mask].max() or 1.0
    rel = err.max() / scale
    ok = rel < tol
    print("  %-46s max err %.3e (%.2e of peak)  %s"
          % (name, err.max(), rel, "OK" if ok else "** FAIL **"))
    return ok


def main():
    inst = SynthPol()
    inst.fast_axis_offset = THETA_OFF
    shape = (BEAM_H, NX - BEAM_X_OFFSET)
    cycles, I, Q_sky, U_sky, disk_mask, q_phi_true = make_cycles(THETA_OFF)
    frames = [f for c in cycles for f in c]
    passed = []

    print("Injected: theta_off = %g deg, p_disk = %.3f, geometry = (%d, %d)"
          % (THETA_OFF, P_DISK, TOP_ROW_START, BEAM_X_OFFSET))
    print("          %d cycles x 4 angles = %d frames of %dx%d"
          % (len(cycles), len(frames), NY, NX))

    # --- 1. beam geometry is recovered from the data ----------------------
    print("\n1. Beam geometry recovered by fit_beam_geometry")
    top_fit, dx_fit = fit_beam_geometry(inst, frames)
    print("  measured (%d, %d), injected (%d, %d)  %s"
          % (top_fit, dx_fit, TOP_ROW_START, BEAM_X_OFFSET,
             "OK" if (top_fit, dx_fit) == (TOP_ROW_START, BEAM_X_OFFSET)
             else "** FAIL **"))
    passed.append((top_fit, dx_fit) == (TOP_ROW_START, BEAM_X_OFFSET))

    # --- 2. Stokes recovery, no registration ------------------------------
    # registration off so this isolates the polarimetric chain; the star is
    # already centred, so shifting it can only add interpolation error.
    print("\n2. Stokes planes, registration off, derotation off")
    cubes = build_stokes_cubes(inst, cycles, fast_axis_offset=THETA_OFF,
                               register_method=None, derotate=False)
    med = median_stokes_cube(cubes)
    passed.append(report("I", med[0], I, np.ones(shape, bool), 1e-10))
    passed.append(report("Q_sky", med[1], Q_sky, disk_mask, 1e-10))
    passed.append(report("U_sky", med[2], U_sky, disk_mask, 1e-10))

    # --- 3. radial Stokes: all signal in Q_phi, none in U_phi -------------
    print("\n3. Radial Stokes (tangential disk: U_phi must be zero)")
    qphi, uphi = radial_stokes(med[1], med[2])
    passed.append(report("Q_phi vs injected p*I_disk", qphi, q_phi_true,
                         disk_mask, 1e-10))
    passed.append(report("U_phi (should be 0)", uphi,
                         np.zeros(shape), disk_mask, 1e-10))
    print("  U_phi/Q_phi peak ratio in the disk: %.2e"
          % (np.abs(uphi[disk_mask]).max() / np.abs(qphi[disk_mask]).max()))

    # --- 4. the offset scan finds the injected theta_off ------------------
    print("\n4. theta_off scan: U_phi residual should minimise at the truth")
    best, curve = None, []
    for trial in np.arange(THETA_OFF - 6, THETA_OFF + 6.01, 1.0):
        c = build_stokes_cubes(inst, cycles, fast_axis_offset=float(trial),
                               register_method=None, derotate=False)
        m = median_stokes_cube(c)
        _, u = radial_stokes(m[1], m[2])
        s = float(np.nanstd(u[disk_mask]))
        curve.append((float(trial), s))
        if best is None or s < best[1]:
            best = (float(trial), s)
    for t, s in curve:
        mark = "  <-- minimum" if (t, s) == best else ""
        print("    theta_off %+6.1f   U_phi std %10.4f%s" % (t, s, mark))
    ok = abs(best[0] - THETA_OFF) <= 1.0
    print("  minimum at %+.1f, injected %+.1f  %s"
          % (best[0], THETA_OFF, "OK" if ok else "** FAIL **"))
    passed.append(ok)

    # --- 5. registration accuracy, and what it costs ----------------------
    # The star is at the exact beam-stack centre, so a perfect centering
    # algorithm shifts by nothing and the result stays exact. What this
    # measures is how far each algorithm is from that, and what the resulting
    # sub-pixel resample does to a sharp feature. "min" is not included: it
    # exists for a saturated core, and this synthetic star has none, so it
    # locks onto unrelated structure 23 px away -- a property of the scene,
    # not a fault in the algorithm.
    print("\n5. Registration: centering accuracy and its cost")
    from reduction.registration import find_center
    stack = inst.split_beams(cycles[0][0])
    ny, nx = stack.shape[1:]
    truth_c = ((ny - 1) / 2, (nx - 1) / 2)
    mean_beam = np.mean(stack, axis=0)

    for method in ("centroid", "smooth_peak"):
        c = find_center(mean_beam, method=method)
        derr = float(np.hypot(c[0] - truth_c[0], c[1] - truth_c[1]))
        cubes_r = build_stokes_cubes(inst, cycles, fast_axis_offset=THETA_OFF,
                                     register_method=method, derotate=False)
        med_r = median_stokes_cube(cubes_r)
        qphi_r, uphi_r = radial_stokes(med_r[1], med_r[2])
        qerr = (np.abs(qphi_r - q_phi_true)[disk_mask].max()
                / np.abs(q_phi_true)[disk_mask].max())
        ratio = (np.abs(uphi_r[disk_mask]).max()
                 / np.abs(qphi_r[disk_mask]).max())
        ok = derr < 0.05
        print("  %-12s centre off by %.3f px -> Q_phi err %.2e, "
              "U_phi/Q_phi %.2e  %s"
              % (method, derr, qerr, ratio, "OK" if ok else "** FAIL **"))
        passed.append(ok)
    print("  (both algorithms find the star; the residual below is not theirs)")

    # --- 6. the centre convention, now a regression guard -----------------
    # This test found a real bug. register_beam_stack used to end with
    #     translate(beam, h / 2 - cy, w / 2 - cx)
    # putting the star at (h/2, w/2) -- (225.0, 505.5) here -- while
    # radial_stokes takes its azimuth origin from azimuthal_angle, which
    # defaults to ((ny-1)/2, (nx-1)/2) = (224.5, 505.0). Half a pixel apart in
    # each axis, on every reduction. Both centring sites now use (n-1)/2, the
    # convention azimuthal_angle and the fast-axis fits already used. The old
    # value is kept below as the contrast: it is what this guards against.
    print("\n6. Centre convention: registration vs radial_stokes")
    cubes_c = build_stokes_cubes(inst, cycles, fast_axis_offset=THETA_OFF,
                                 register_method="centroid", derotate=False)
    med_c = median_stokes_cube(cubes_c)
    ny_, nx_ = med_c[1].shape
    peak = np.abs(q_phi_true)[disk_mask].max()
    leaks = {}
    for label, ctr in [("agreeing   ((ny-1)/2, (nx-1)/2)", None),
                       ("old target (ny/2, nx/2)", (ny_ / 2, nx_ / 2))]:
        qp, up = radial_stokes(med_c[1], med_c[2], center=ctr)
        qerr = np.abs(qp - q_phi_true)[disk_mask].max() / peak
        leak = np.abs(up[disk_mask]).max() / np.abs(qp[disk_mask]).max()
        leaks[label.split()[0]] = leak
        print("    %-34s Q_phi err %.3e  U_phi/Q_phi %.3e"
              % (label, qerr, leak))
    ok = leaks["agreeing"] < 1e-10
    print("  centres agree, so a tangential disk leaks nothing into U_phi  %s"
          % ("OK" if ok else "** FAIL **"))
    print("  (half a pixel of disagreement costs ~2%, into the very channel")
    print("   theta_off and the IP fits are judged by -- hence the guard)")
    passed.append(ok)

    print("\n%s  (%d/%d checks passed)"
          % ("ALL CHECKS PASSED" if all(passed) else "** SOME CHECKS FAILED **",
             sum(passed), len(passed)))
    return 0 if all(passed) else 1


if __name__ == "__main__":
    sys.exit(main())

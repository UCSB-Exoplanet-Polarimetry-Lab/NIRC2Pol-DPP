"""Validate the pipeline against the AB Aur commissioning notebook.

Processes the same 92 frames (2025-12-07, FRAMENO 857-900, 915-930,
932-963) as working-ab-aur_07022026.ipynb, but through the pipeline: PCUPR
based HWP cycle matching and automatic centering instead of order-based
grouping and hand-tuned cutout centers.

Compares the resulting median Q_phi / U_phi against the notebook's saved
qphi_median.fits / uphi_median.fits (in ~/AB-Aur). Known bookkeeping
differences: the notebook omits the 1/2 factors in the double differences
(so its Q/U/I are 2x ours) and its Q_phi uses the opposite sign convention
in the cell that produced the saved medians; the comparison accounts for
both via a fitted scale factor.

Two things differ from the version this replaces, both deliberate:

* **Beam geometry is measured, not inherited.** ``top_row_start`` and
  ``beam_x_offset`` are no longer class defaults -- they are None until
  measured, because a stale value fails silently: registration shifts both
  beams together, so a wrong separation is never corrected and lands in the
  double difference as a dipole. This script now measures them from these
  frames, as process_polmode.py and the tutorial do.
* **register_method is "min".** AB Aur's core is saturated, so the default
  peak-based centering finds the wrong thing; the tutorial overrides it the
  same way. The old script took whatever the default was, so a comparison
  against its numbers is not quite like for like.

Run from anywhere:
    python examples/compare_ab_aur.py
    python examples/compare_ab_aur.py --theta-off -13.1
    python examples/compare_ab_aur.py --ip fit_uphi_all
    python examples/compare_ab_aur.py --empirical
"""

import glob
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from instruments.nirc2 import NIRC2PolarimetryData
from polarimetry import (build_stokes_cubes, fit_ip_uphi, fit_ip_uphi_all,
                         mean_ip, median_stokes_cube, radial_stokes)
from reduction import fit_beam_geometry
from utils import crop, frame_number, load_frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("compare_ab_aur")

DATA_DIR = "/home/shared/exoserver/NIRC2_Pol/jaykes_reduction/2025-12-07/reduced"
NOTEBOOK_DIR = os.path.expanduser("~/AB-Aur")

# Outputs stay outside the repo: this script lives in examples/, and its FITS
# and PNG products are not something to drop into a checkout.
OUT_DIR = os.path.expanduser("~/validation")

# This script deliberately pins both choices that process_polmode.py exposes
# as FAST_AXIS_METHOD and IP_METHOD, because it is a comparison against a
# fixed reference rather than a reduction:
#
#   fast axis - "fixed", at the notebook's own offset (below). Fitting it
#               would compare the pipeline against a moving target.
#   IP        - None by default, matching the notebook, which applied no
#               leakage correction. --ip switches it on to see what one does.
#
# Both the butterfly fit and the U_phi scan prefer about -13 rather than the
# notebook's value; --theta-off exists to try that.
NOTEBOOK_OFFSET = -8.0278

# AB Aur's core is saturated, so centre on the minimum rather than the peak.
REGISTER_METHOD = "min"

BKG_BOX = (25, 350, 50, 400)  # notebook's sub_mean_bkg box, per beam
CROP = 400                    # notebook's final cutout size

# The notebook's frame selection, as three disjoint runs. select_frames takes
# one inclusive range, so this stays an explicit predicate.
FRAME_RANGES = ((857, 900), (915, 930), (932, 963))


class ABAurData(NIRC2PolarimetryData):
    """NIRC2 pol data with the notebook's per-beam mean background
    subtraction folded into beam extraction (L' thermal background)."""

    background_method = "mean_box"
    background_box = BKG_BOX


def wanted(path):
    """Is this one of the notebook's frames?"""
    n = frame_number(path)
    return n is not None and any(lo <= n <= hi for lo, hi in FRAME_RANGES)


def resolve_theta_off(argv):
    """Read --theta-off from the command line, defaulting to the notebook's."""
    if "--theta-off" not in argv:
        return NOTEBOOK_OFFSET
    return float(argv[argv.index("--theta-off") + 1])


def resolve_ip_method(argv):
    """Read --ip from the command line. None means no leakage correction."""
    if "--ip" not in argv:
        return None
    method = argv[argv.index("--ip") + 1]
    if method not in ("fit_uphi_all", "fit_uphi_per_cycle", "none"):
        raise SystemExit(
            "--ip must be fit_uphi_all, fit_uphi_per_cycle or none, not %r. "
            "These are the same names process_polmode.py uses for IP_METHOD. "
            "The mm_model route is not implemented yet, and the "
            "edge_annulus_* routes are not offered here because this "
            "comparison pins the offset rather than fitting it, which is "
            "what those are useful for." % method)
    return None if method == "none" else method


def fit_ip(inst, cycles, theta_off, method):
    """The IP for a named method, or None. Same vocabulary as IP_METHOD."""
    if method is None:
        return None
    if method == "fit_uphi_all":
        return fit_ip_uphi_all(inst, cycles, theta_off, mask_radius=22,
                               crop_size=CROP,
                               register_method=REGISTER_METHOD)
    per_cycle = [fit_ip_uphi(inst, c, theta_off, mask_radius=22,
                             crop_size=CROP, register_method=REGISTER_METHOD)
                 for c in cycles]
    return mean_ip(per_cycle)


def main():
    theta_off = resolve_theta_off(sys.argv)
    ip_method = resolve_ip_method(sys.argv)
    inst = ABAurData()

    files = sorted(f for f in glob.glob(os.path.join(DATA_DIR, "reduced*.fits"))
                   if wanted(f))
    log.info("Found %d files", len(files))
    frames = load_frames(files)

    # Measured from these frames rather than looked up: the separation is a
    # property of the epoch, and nothing downstream can undo a wrong one.
    inst.top_row_start, inst.beam_x_offset = fit_beam_geometry(inst, frames)
    log.info("beam geometry: top row %d, x offset %d",
             inst.top_row_start, inst.beam_x_offset)

    cycles = inst.match_modulator_cycles(frames)

    # star lands at the beam-stack center after registration
    ny, nx = inst.beam_height, 1024 - inst.beam_x_offset
    star_center = ((ny - 1) / 2, (nx - 1) / 2)

    # --- Stokes cubes ---
    ip = fit_ip(inst, cycles, theta_off, ip_method)
    if ip is not None:
        log.info("instrumental polarization (%s): %s", ip_method,
                 ip.describe())
    cubes = build_stokes_cubes(inst, cycles, fast_axis_offset=theta_off,
                               register_method=REGISTER_METHOD, ip=ip)
    med = median_stokes_cube(cubes)

    # Crop to the notebook's 400x400 field around the star. crop quantises its
    # origin (int(round(cy - crop_h/2))), so the star does not land exactly on
    # the crop's centre -- up to a pixel out. Carry the offset through and tell
    # radial_stokes where the star actually is, rather than letting it assume
    # the centre: a pixel of error there leaks disk signal into U_phi.
    I, sr, sc = crop(med[0], (CROP, CROP), center=star_center)
    Q, _, _ = crop(med[1], (CROP, CROP), center=star_center)
    U, _, _ = crop(med[2], (CROP, CROP), center=star_center)
    star_in_crop = (star_center[0] - sr, star_center[1] - sc)
    log.info("star sits at (%.1f, %.1f) in the %dx%d crop",
             star_in_crop[0], star_in_crop[1], CROP, CROP)
    qphi, uphi = radial_stokes(Q, U, center=star_in_crop)

    os.makedirs(OUT_DIR, exist_ok=True)

    def out(name, ext):
        """Output path tagged with the offset, so runs do not clobber."""
        return os.path.join(OUT_DIR, f"{name}_theta{theta_off:g}.{ext}")

    fits.PrimaryHDU(np.stack([I, Q, U])).writeto(
        out("abaur_median_stokes_pipeline", "fits"), overwrite=True)
    fits.PrimaryHDU(np.stack([qphi, uphi])).writeto(
        out("abaur_qphi_uphi_pipeline", "fits"), overwrite=True)

    # --- notebook reference products ---
    nb_qphi = fits.getdata(os.path.join(NOTEBOOK_DIR, "qphi_median.fits"))
    nb_uphi = fits.getdata(os.path.join(NOTEBOOK_DIR, "uphi_median.fits"))

    # the notebook star sits at (y=203, x=197) in its cutouts; ours at the
    # crop center. Align by shifting the comparison annulus, not the images:
    # compare radial profiles + pixel correlation on overlap.
    ours_c = star_in_crop
    nb_c = (203.0, 197.0)

    yy, xx = np.mgrid[:CROP, :CROP]
    r_ours = np.hypot(yy - ours_c[0], xx - ours_c[1])
    r_nb = np.hypot(yy - nb_c[0], xx - nb_c[1])
    ann_ours = (r_ours > 22) & (r_ours < 150)
    ann_nb = (r_nb > 22) & (r_nb < 150)

    # shift notebook image so its star matches our center, for pixelwise stats
    from utils.imutils import translate
    nb_qphi_al = translate(np.nan_to_num(nb_qphi), ours_c[0] - nb_c[0],
                           ours_c[1] - nb_c[1])
    nb_uphi_al = translate(np.nan_to_num(nb_uphi), ours_c[0] - nb_c[0],
                           ours_c[1] - nb_c[1])

    # empirical disk-sign check: is the disk positive in each product?
    log.info("Disk sign check, annulus mean Q_phi: pipeline %.2f | notebook %.2f",
             float(np.nanmean(qphi[ann_ours])),
             float(np.nanmean(nb_qphi[ann_nb])))

    sel = ann_ours & np.isfinite(qphi) & np.isfinite(nb_qphi_al) \
        & (nb_qphi_al != 0)
    a, b = qphi[sel], nb_qphi_al[sel]
    scale = float(np.nansum(a * b) / np.nansum(a * a))
    corr = float(np.corrcoef(a, b)[0, 1])
    resid = b - scale * a
    log.info("Q_phi comparison in annulus 22 < r < 150 "
             "(theta_off = %g, register_method = %s, IP = %s):",
             theta_off, REGISTER_METHOD, ip_method or "none")
    log.info("  correlation (pipeline vs notebook): %.4f", corr)
    log.info("  fitted scale notebook/pipeline: %.3f (expect ~2 from the "
             "notebook's missing 1/2 factors, sign per convention)", scale)
    log.info("  residual std / notebook Q_phi std: %.3f",
             float(np.nanstd(resid) / np.nanstd(b)))
    log.info("  U_phi std ratio ours/notebook: %.3f",
             float(np.nanstd(uphi[sel]) / np.nanstd(nb_uphi_al[sel])))

    # --- comparison figure ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    nb_v = np.nanstd(b)
    our_v = nb_v / abs(scale)

    panels = [
        (axes[0, 0], qphi, our_v, "Pipeline Q$_\\phi$ (median, auto-centered)"),
        (axes[0, 1], np.sign(scale) * nb_qphi / abs(scale), our_v,
         "Notebook Q$_\\phi$ (rescaled to pipeline units)"),
        (axes[0, 2], np.sign(scale) * nb_qphi_al / abs(scale) - qphi, our_v,
         "Difference (notebook $-$ pipeline)"),
        (axes[1, 0], uphi, our_v, "Pipeline U$_\\phi$"),
        (axes[1, 1], np.sign(scale) * nb_uphi / abs(scale), our_v,
         "Notebook U$_\\phi$ (rescaled)"),
    ]
    for ax, img, v, title in panels:
        im = ax.imshow(img, origin="lower", vmin=-v, vmax=2.5 * v,
                       cmap="inferno")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)

    # radial profiles
    ax = axes[1, 2]
    bins = np.arange(22, 150, 4)

    def prof(img, r):
        return [np.nanmedian(img[(r >= lo) & (r < lo + 4)]) for lo in bins]

    ax.plot(bins, prof(qphi, r_ours), label="pipeline Q$_\\phi$")
    ax.plot(bins, prof(np.sign(scale) * nb_qphi / abs(scale), r_nb), "--",
            label="notebook Q$_\\phi$ (rescaled)")
    ax.plot(bins, prof(uphi, r_ours), label="pipeline U$_\\phi$", alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("radius [px]")
    ax.set_ylabel("median in annulus")
    ax.set_title(f"Radial profiles | corr = {corr:.3f}, "
                 f"$\\theta_{{off}}$ = {theta_off:g}$^\\circ$")
    ax.legend(fontsize=9)

    fig.suptitle("AB Aur 2025-12-07 L': NIRC2Pol-DPP vs commissioning "
                 "notebook", fontsize=14)
    fig.tight_layout()
    out_png = out("abaur_comparison", "png")
    fig.savefig(out_png, dpi=110)
    log.info("Wrote %s", out_png)

    if "--empirical" not in sys.argv:
        return

    # --- TEMPORARY empirical correction: per-cycle beam shifts + ipq/ipu --
    from polarimetry import (build_corrected_stokes_cube,
                             fit_empirical_cycle_correction)

    corrections, corr_cubes = [], []
    for i, cycle in enumerate(cycles):
        c = fit_empirical_cycle_correction(inst, cycle, theta_off,
                                           mask_radius=22, crop_size=CROP,
                                           maxfev=2500)
        log.info("cycle %02d: U_phi std %.3f -> %.3f | ipq %+.4f ipu %+.4f",
                 i, c["uphi_std_initial"], c["uphi_std_final"],
                 c["ipq"], c["ipu"])
        corrections.append(c)
        corr_cubes.append(build_corrected_stokes_cube(
            inst, cycle, c, theta_off, crop_size=CROP))

    med_c = median_stokes_cube(corr_cubes)
    qphi_c, uphi_c = radial_stokes(med_c[1], med_c[2])

    fits.PrimaryHDU(np.stack([qphi_c, uphi_c])).writeto(
        out("abaur_qphi_uphi_empirical", "fits"), overwrite=True)

    ipqs = [c["ipq"] for c in corrections]
    ipus = [c["ipu"] for c in corrections]
    sel_c = sel & np.isfinite(qphi_c)
    corr_c = float(np.corrcoef(qphi_c[sel_c], nb_qphi_al[sel_c])[0, 1])
    log.info("Empirical correction summary:")
    log.info("  ipq = %.4f +/- %.4f, ipu = %.4f +/- %.4f",
             np.mean(ipqs), np.std(ipqs), np.mean(ipus), np.std(ipus))
    log.info("  Q_phi corr vs notebook: plain %.4f -> corrected %.4f",
             corr, corr_c)
    log.info("  U_phi std in annulus: plain %.3f -> corrected %.3f "
             "(notebook, rescaled: %.3f)",
             float(np.nanstd(uphi[sel])), float(np.nanstd(uphi_c[sel_c])),
             float(np.nanstd(nb_uphi_al[sel] / abs(scale))))

    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    for ax, img, title in [
            (axes2[0, 0], qphi, "Plain Q$_\\phi$ (rotation model only)"),
            (axes2[0, 1], qphi_c, "Corrected Q$_\\phi$ (+shifts, ipq/ipu)"),
            (axes2[0, 2], np.sign(scale) * nb_qphi / abs(scale),
             "Notebook Q$_\\phi$ (rescaled)"),
            (axes2[1, 0], uphi, "Plain U$_\\phi$"),
            (axes2[1, 1], uphi_c, "Corrected U$_\\phi$"),
            (axes2[1, 2], np.sign(scale) * nb_uphi / abs(scale),
             "Notebook U$_\\phi$ (rescaled)")]:
        im = ax.imshow(img, origin="lower", vmin=-our_v, vmax=2.5 * our_v,
                       cmap="inferno")
        ax.set_title(title, fontsize=11)
        fig2.colorbar(im, ax=ax, fraction=0.046)

    ax = axes2[0, 3]
    ax.plot(bins, prof(qphi, r_ours), label="plain")
    ax.plot(bins, prof(qphi_c, r_ours), label="corrected")
    ax.plot(bins, prof(np.sign(scale) * nb_qphi / abs(scale), r_nb), "--",
            label="notebook (rescaled)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("radius [px]"); ax.set_ylabel("median Q$_\\phi$")
    ax.set_title(f"Q$_\\phi$ profiles | corr {corr:.3f} -> {corr_c:.3f}")
    ax.legend(fontsize=9)

    ax = axes2[1, 3]
    ax.plot(ipqs, "o-", label=f"ipq ({np.mean(ipqs):+.4f})")
    ax.plot(ipus, "s-", label=f"ipu ({np.mean(ipus):+.4f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("HWP cycle"); ax.set_ylabel("fitted IP fraction")
    ax.set_title("Per-cycle IP leakage terms")
    ax.legend(fontsize=9)

    fig2.suptitle("AB Aur: TEMPORARY empirical correction (beam shifts + "
                  "ipq/ipu, $\\theta_{off}$ fixed at "
                  f"{theta_off:g}$^\\circ$)", fontsize=14)
    fig2.tight_layout()
    out2 = out("abaur_comparison_empirical", "png")
    fig2.savefig(out2, dpi=110)
    log.info("Wrote %s", out2)


if __name__ == "__main__":
    main()

"""Measure the HWP fast axis offset against a polarized standard star.

A calibration run, not a science one. It reduces a sequence on a star whose
polarization angle on sky is already known, asks what ``theta_off`` makes the
pipeline reproduce that angle, and prints the answer. What you do with it is
hand it to the science reduction::

    fast_axis_method = "fixed"
    theta_off = <the number this printed>

which is why nothing here is wired into ``polmode.run``: the standard is a
different target from the science target, usually a different night, and the
offset is the only thing that travels between them.

    python polarized_standard.py my_standard.toml

The other route to the same number is
:func:`nirc2pol.polarimetry.fit_fast_axis_butterfly`, which reads it off the
orientation of a tangentially polarized disk. Neither is a check on the other
in the sense of sharing assumptions -- the butterfly needs a disk and assumes
it is azimuthally polarized, this needs a catalogue and assumes that
catalogue -- which is exactly what makes agreement between them worth
something.

Two things this script exists to make hard to get wrong:

**derotate=False.** ``prepare_cycles`` can fold the north angle into the Q/U
rotation, and for radial Stokes that is exact. For the position angle of a
point source it is not, and on the 2025-12-06 standard the difference is
79 degrees. ``measure_cycles`` refuses folded cycles rather than let that
pass, but the right thing is to not fold them.

**Fit it both ways.** Leaving the leakage in displaces the offset; fitting it
removes the displacement and amplifies the noise. Which wins depends on the
rotation span and the per-cycle scatter, and the only reliable way to find
out is to look at both, which is what this prints.
"""

import logging
import os
import sys

import numpy as np

from nirc2pol.polarimetry import (PolarizedStandard, curve_of_growth_polarization,
                                  fit_theta_off_polstd, measure_cycles,
                                  prepare_cycles)
from nirc2pol.polmode import run
from nirc2pol.reduction.config import ReductionConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CONFIG_PATH = (sys.argv[1] if len(sys.argv) > 1
               else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "reduction_config.toml"))

# ---------------------------------------------------------------------------
# What is known about the star, from outside this pipeline. There is no
# bundled catalogue on purpose -- a value written here can be traced to
# whoever chose it, and one shipped in the package cannot. If the catalogue
# value is optical, carry it to the observing band with
# ``nirc2pol.polarimetry.serkowski_p`` first, and put an honest p_err on the
# result: at L' that extrapolation is worth a factor rather than a percent.
#
# theta_err is the one that matters. It is what the fit weights by, and the
# catalogue angle propagates into theta_off at HALF its own error -- 2 deg
# there is 1 deg here.
# ---------------------------------------------------------------------------
STANDARD = PolarizedStandard(
    name="REPLACE ME",
    p=0.0093,               # fraction, not percent, in the observed band
    theta=103.0,            # sky position angle [deg]
    p_err=0.004,
    theta_err=2.0,
    band="Lp",
    reference="fill this in -- it is the one input that cannot be recomputed",
)

# The aperture. Read the radius off the curve of growth below rather than
# choosing it here: p is a ratio over a shared aperture, so PSF clipping
# cannot bias it, and a p that moves with radius is telling you about the
# background instead.
RADIUS = 30.0
BACKGROUND = (110.0, 165.0)     # annulus for the residual background plane
MASK = None                     # exclude a companion or a registration wedge

cfg = ReductionConfig.from_toml(CONFIG_PATH)
products = run(cfg, config_path=CONFIG_PATH)
instrument, cycles = products["instrument"], products["cycles"]

# derotate=False: see the module docstring. register_method comes from the
# config so the photometry sees the same registration the products did.
prepared = prepare_cycles(instrument, cycles, derotate=False,
                          register_method=cfg.register_method)
print(f"\n{len(prepared)} complete HWP cycles")

# --- is the measurement stable? -------------------------------------------
# The standing check on the background treatment. Both columns should be
# flat. If p climbs with radius, or the angle walks, the background is not
# being removed properly and no fit below is worth reading.
print("\ncurve of growth in polarization (theta_off held at the config value):")
cog = curve_of_growth_polarization(prepared, cfg.theta_off,
                                   radii=[10, 20, 30, 40, 60, 80],
                                   background=BACKGROUND, mask=MASK)
print(f"  {'r [px]':>7} {'p [%]':>9} {'theta [deg]':>12}")
for r, p, t in zip(cog["radius"], cog["p"], cog["theta"]):
    print(f"  {r:7.0f} {100 * p:9.3f} {t:12.2f}")

# --- how noisy is one cycle? ----------------------------------------------
# This decides whether the joint IP fit can work at all. The offset and the
# leakage separate through field rotation, and the separation is amplified by
# the condition number, so a large per-cycle scatter at a short span means
# the joint fit returns noise however well posed it looks.
measured = measure_cycles(prepared, radius=RADIUS, background=BACKGROUND,
                          mask=MASK)
sky = measured.z * np.exp(-1j * np.radians(measured.base
                                           + 4.0 * cfg.theta_off))
sigma = float(np.hypot(sky.real.std(), sky.imag.std()) / np.sqrt(2))
print(f"\naperture r = {RADIUS:.0f} px at ({measured.center[0]:.2f}, "
      f"{measured.center[1]:.2f})")
print(f"  measured p = {100 * abs(sky.mean()):.3f}%, "
      f"sky angle = {np.degrees(0.5 * np.angle(sky.mean())) % 180:.2f} deg "
      f"at theta_off = {cfg.theta_off}")
print(f"  per-cycle scatter on q, u = {100 * sigma:.3f}%")

# --- the fit, both ways ---------------------------------------------------
plain = fit_theta_off_polstd(prepared, STANDARD, radius=RADIUS,
                             background=BACKGROUND, mask=MASK)
joint = fit_theta_off_polstd(prepared, STANDARD, radius=RADIUS,
                             background=BACKGROUND, mask=MASK, fit_ip=True)

print(f"\n{'':14} {'theta_off':>18} {'efficiency':>12}")
print(f"  {'leakage left in':12} {plain.theta_off:+10.3f} "
      f"+/-{plain.theta_off_err:5.3f} {plain.efficiency:12.3f}")
print(f"  {'leakage fitted':12} {joint.theta_off:+10.3f} "
      f"+/-{joint.theta_off_err:5.3f} {joint.efficiency:12.3f}"
      f"   ipq {joint.ip.ipq:+.4f}  ipu {joint.ip.ipu:+.4f}")
print(f"  {joint.rotation_span:.1f} deg of field rotation, "
      f"condition number {joint.condition_number:.0f}")

difference = abs(plain.theta_off - joint.theta_off)
if difference < joint.theta_off_err:
    print(f"\n  The two differ by {difference:.2f} deg against a joint-fit "
          f"error of {joint.theta_off_err:.2f} deg, so the leakage is NOT "
          f"being measured here.\n  Use the first number and state that it "
          f"carries the leakage -- on synthetics that displaces theta_off by "
          f"of order 0.7 deg.\n  To do better, observe a second standard at a "
          f"different sky angle and pool them:\n"
          f"      fit_theta_off_polstd([(cycles_a, std_a), (cycles_b, std_b)], "
          f"fit_ip=True)")
else:
    print(f"\n  The two differ by {difference:.2f} deg, more than the joint "
          f"fit's {joint.theta_off_err:.2f} deg error, so the leakage is "
          f"real and being removed.\n  Use the second number.")

# The catalogue angle is the input least likely to be right, and the whole
# dependence on it is one line -- so print the line rather than only the
# point estimate, and the reader can re-evaluate it when a better value
# turns up without reducing the night again.
reference_angle = float(np.degrees(0.5 * np.angle(sky.mean())) % 180.0)
print(f"\n  For any other catalogue angle, without re-reducing:")
print(f"      theta_off = {cfg.theta_off} - (theta_known - "
      f"{reference_angle:.2f}) / 2")

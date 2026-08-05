# Fast axis offset measurements

Every HWP fast-axis calibration sequence found under
`/home/shared/exoserver/NIRC2_Pol`, fitted with the expression from the
SPIE paper:

    normalized beam difference = A * cos(4 * (theta_HWP - theta_off)) + C

A *fast axis calibration* is a single HWP ladder stepping 0 -> 180 deg at a
**fixed** image rotator position. Sequences that step the image rotator
(`pol_cal_imr_*`) are a different calibration and are excluded.

Sequences are dark-subtracted (and flat-fielded where available) before
fitting; flux is summed in the two beam apertures (top y 575-985, bottom
y 40-450, x 250-950). Values are 3-sigma clipped where one bad point
dominated the fit.

## Good calibrations

| Band | UT date | theta_off [deg] | +/- | Amplitude | R^2 | Sequence |
|---|---|---|---|---|---|---|
| **H** | 2026-06-05 | **+10.653** | 0.325 | 1.25 % | 0.992 | `h_fast_axis` frames 3-21 |
| **Kp** | 2026-06-05 | **+11.356** | 0.215 | 1.20 % | 0.996 | `Kp_fast_axis` frames 24-42 |
| **J** | 2026-06-07 | **+10.507** | 0.170 | 1.03 % | 0.998 | `j_fast_axis_cal` (1 clipped) |
| **J** | 2026-06-29 | **+10.356** | 0.236 | 1.22 % | 0.996 | `JPolFlat` |
| **Lp** | 2025-12-03 | **-8.180** | 0.144 | 0.93 % | 0.998 | Dome_Flats L-Band frames 340-358 |
| H | 2025-10-02 | +19.810 | 0.198 | 1.49 % | 0.994 | `h_hwp_modulation` |
| Lp | 2026-05-27 | +19.627 | 0.656 | 0.045 % | 0.970 | `FastAxisCal` (weak signal) |

## Band or epoch dependence: what the data can and cannot show

The four June 2026 calibrations — J, H and Kp, taken within four weeks —
agree at **+10.4 to +11.4 deg**, a spread of about 1 deg across three
bands. Within that epoch, the offset is essentially band-independent
across JHK.

The 2025-12-03 Lp calibration gives **-8.18**, which does not match the
June 2026 cluster. That difference is **degenerate between band and
epoch**: the only high-quality Lp measurement is from December 2025, and
all the high-quality JHK measurements are from June 2026, so nothing here
separates "L' differs from JHK" from "the zero point changed between
epochs". The one near-contemporaneous Lp point (2026-05-27, +19.63) has
20x lower modulation amplitude and is too weak to settle it.

Fall 2025 calibrations are excluded entirely — the calibration procedure
differed enough then that those values are not trusted, so they are not in
the log. (For reference only: a 2025-10-02 H ladder fits to +19.81.)

**Practical guidance**: use the calibration closest in time to the science
data, in the same band. Resolving band vs epoch properly needs one epoch
with good ladders in several bands *including* L'.

## Weak or unusable sequences

These fitted poorly and should not be used: Lp 2025-12-07
`hwp_modulation_test` (amp 0.018 %), Lp 2025-12-08 `HWP_modulation`
(monotonic drift, not a modulation), Lp 2026-06-01 `pol_cal` (R^2 0.005),
H2O_ice 2026-06-01 / 2026-06-29 (amp ~0.01 %, R^2 < 0.2). The 2026-07-06
H2O_ice sequence does fit (-21.41 +/- 1.11, R^2 0.908) but H2O_ice is not
a science band here.

**Always look at the plot before trusting a fit.** The 2026-06-07 J
sequence fitted to +6.9 +/- 3.8 (R^2 0.46) purely because of one bad point
at HWP = 0; clipped it is +10.51 +/- 0.17 (R^2 0.998), consistent with the
other J sequence.

## Notes

- Modulation amplitude drives the uncertainty. Sequences with ~1 % show
  clean two-cycle sinusoids and pin theta_off to ~0.2 deg; those with
  ~0.01-0.05 % are dominated by noise or background drift. At L' the large
  unpolarized thermal background dilutes the signal, so L' sequences need
  care (the good 2025-12-03 one reaches 0.93 %).
- The model is degenerate modulo 45 deg; values are wrapped into
  (-22.5, +22.5].
- Header keywords: sequences before ~2025-12-01 carry no `PCUPR`. The HWP
  angle is then taken from the `OBJECT` string, or from file order across
  the ladder (0 -> 180) when OBJECT does not encode it, as for the
  2025-12-03 L-band data.

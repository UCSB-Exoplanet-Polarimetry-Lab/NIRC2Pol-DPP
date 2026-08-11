# NIRC2Pol-DPP Tutorial

This tutorial walks through a complete reduction of Keck/NIRC2 polarimetry
(NIRC2-Pol) data, from raw FITS frames to Stokes cubes and radial Stokes
images, using real commissioning data (AB Aurigae, 2025-12-07, L' +
Wollaston).

**What the pipeline does**, in order:

```
raw frames ──► sort by type ──► master darks/flats ──► reduce science frames
   (dark subtraction, flat fielding, bad-pixel cleanup)
      ──► sky subtraction ──► HWP cycle matching ──► beam splitting
      ──► image registration ──► double differencing ──► Stokes cubes
      ──► median Stokes cube ──► PI / AoLP / DoLP / radial Stokes
```

**How the code is organized** (each layer only talks to the one below it):

| layer | package | contents |
|---|---|---|
| orchestration | `pipeline.py` | chain steps into a "recipe" (optional) |
| polarimetry | `polarimetry/` | double differencing, Stokes cubes, radial Stokes, Mueller interface |
| reduction | `reduction/` | masters, calibration, sky subtraction, registration |
| instrument | `instruments/` | everything NIRC2-specific, behind the `PolarimetryData` interface |
| helpers | `utils/` | `Frame` (image + header), FITS I/O, image operations |

Every step is a plain function: you can run the whole chain, or any single
step, from a script or notebook.

## 0. Setup

Requirements: Python ≥ 3.11 with `numpy`, `scipy`, `astropy`, and `pyklip`
(used for image rotation). On mueller these live in the `nirc2p` conda
environment.

Run everything from the repository root (or add it to `sys.path`):

```python
import sys, os, glob
sys.path.insert(0, os.path.abspath(".."))  # repo root, if running from examples/

import numpy as np
import matplotlib.pyplot as plt

# the four layers
from utils import Frame, ObslogPaths, load_frames, save_frames, crop
from reduction import (make_master_darks, make_master_flats, make_master_masks,
                       reduce_frame, subtract_mean_background)
from instruments import nirc2
from instruments.nirc2 import NIRC2PolarimetryData
from polarimetry import (build_stokes_cubes, median_stokes_cube,
                         polarization_products, radial_stokes)
```

## 1. The data model: `Frame` and the instrument object

The pipeline passes around `Frame` objects — just image data plus the FITS
header, with dict-style keyword access. Data follow the numpy convention
`data[y, x]`.

```python
DATA = "/home/shared/exoserver/NIRC2_Pol/jaykes_reduction/2025-12-07"

frame = Frame.load(f"{DATA}/raw/n0932.fits")   # an AB Aur science frame
print(frame)
print("shape:", frame.shape)
print("OBJECT:", frame["OBJECT"], "| FILTER:", frame["FILTER"],
      "| ITIME:", frame["ITIME"], "| HWP angle (PCUPR):", frame.get("PCUPR"))
```

Everything NIRC2-specific — header keywords, detector constants, beam
geometry, the polarimetric rotation model — lives in one object,
`NIRC2PolarimetryData`. The rest of the pipeline only talks to this
interface, so supporting another instrument means subclassing
`instruments.base.PolarimetryData`.

On-sky data must also say how the sky/thermal background is removed. This
is a pipeline setting, not something each script does itself, so a
reduction cannot silently skip it. Which method suits depends on the band:
L' and M sit on a huge thermal pedestal and want dither pairs or a mean
box; JHK usually want an annulus around the source. Calibration sequences
(dome flats, fast axis ladders) set ``background_method = None``, because
there the illumination *is* the signal.

```python
class LpPolData(NIRC2PolarimetryData):
    """NIRC2 pol data for L': mean-box background (this run was not dithered)."""

    background_method = "mean_box"        # "mean_box" | "annulus" | "dither" | None
    background_box = (25, 350, 50, 400)   # (ylow, yhigh, xlow, xhigh)

instrument = LpPolData()
print("background:", instrument.describe_background())
print("modulator keyword:", instrument.modulator_keyword)
print("HWP critical angles:", instrument.critical_angles)
print("gain [e-/ADU]:", instrument.gain(frame))
print("saturation [ADU]:", instrument.saturation_limit(frame))
```

## 2. Sorting raw frames

`instrument.sort_frames` classifies raw files from their headers alone:
shutter closed → dark; telescope at the dome-flat position with AO loops
open → lamp-on/off flat (split by count level); OBJECT containing
"sky"/"twi" → sky flat; everything else → science.

*The full night here has 1265 frames — sorting loads each one, so this takes
a few minutes. For the tutorial we sort a subset covering darks, flats, and
the AB Aur science sequence.*

```python
def frameno(path):
    return int(os.path.basename(path)[1:5])   # raw files are n####.fits

# For a real reduction just glob the whole raw folder; for the tutorial we
# take a subset: sky flats + the AB Aur sequence + its matching darks.
raw_files = [f for f in sorted(glob.glob(f"{DATA}/raw/*.fits"))
             if 901 <= frameno(f) <= 913     # sky flats (Lp + Wollaston)
             or 932 <= frameno(f) <= 963     # AB Aur science frames
             or 984 <= frameno(f) <= 993]    # darks matching the science settings

sorted_files = instrument.sort_frames(raw_files)
for kind, files in sorted_files.items():
    print(f"{kind:14s} {len(files)} frames")
```

## 3. Master darks and flats

Frames are grouped by exposure settings (`NAXIS1/2, ITIME, COADDS,
SAMPMODE, READS`, plus `FILTER` for flats); each group is median-combined,
sigma-clipped, cleaned with the detector bad-pixel mask, and stamped with
bookkeeping keywords (`NFRAMES`, `MAMEDIAN`, `FLATTYPE`, ...).

Flats are dark-subtracted and normalized to a median of 1.

**The kind of flat is a requirement, not a preference.** L' and M must use
sky flats — the dome lamp is swamped by thermal background there — and JHK
must use lamp flats. Reducing with the wrong kind gives a wrong answer that
still looks plausible, so `find_closest_flat` raises rather than
substituting. Pass `required_flat_type="SKY"` to override the band default
(some observers want skies in JHK too), or
`allow_flat_type_mismatch=True` to proceed anyway, which records `FLATMISM`
in the header.

Lamp-*off* flats are not used at all: they are meaningless in JHK, and at L'
sky flats are used regardless, so a lamp flat is simply the lamp-on frames
minus the matched dark.

Among *valid* flats there is still an ordering — polarimetric (critical-angle)
sets first, then dark-subtracted before darkless, then the most frames.

```python
bad_pixel_mask = instrument.bad_pixel_mask()   # static NIRC2 mask

darks = load_frames(sorted_files["darks"])
master_darks, dark_masks = make_master_darks(darks, bad_pixel_mask=bad_pixel_mask)

master_flats, flat_masks = make_master_flats(
    load_frames(sorted_files["flats"]),
    load_frames(sorted_files["flats_sky"]),
    load_frames(sorted_files["flats_lampon"]),
    master_darks,
    bad_pixel_mask=bad_pixel_mask,
    # tell it how to spot a polarimetric (critical-angle) flat set so those
    # are preferred; data without them falls back to regular flats
    modulator_keyword=instrument.modulator_keyword,
    critical_angles=instrument.critical_angles,
)
master_masks = make_master_masks(dark_masks, flat_masks)

print(f"{len(master_darks)} master darks, {len(master_flats)} master flats")
for f in master_flats[:3]:
    print("  flat:", f["FILTER"], "| FLATTYPE", f["FLATTYPE"],
          "| polarimetric:", f.get("POLFLAT"), "| from", f["NFRAMES"], "frames")
```

## 4. Reducing the science frames

`reduce_frame` applies, per frame:

1. find the best-matching dark (relaxing the match criteria step by step)
   and flat — the **filter must always match**, size is only preferred (a
   larger flat is trimmed to the frame), and exposure settings are ignored
   because the flat is normalized
2. `reduced = (raw − dark) / flat`
3. divide by COADDS
4. replace bad pixels (detector mask + sigma-clip masks + NaNs) by local
   thin-plate-spline interpolation; saturated pixels get a wider "+"-shaped
   replacement
5. multiply by the gain; record everything in header keywords
   (`DARKSUB`, `FLATDIV`, `DIVCOADD`, `GAIN`, `RED-FN`)

This chain is validated bit-for-bit against AIR.jl's reduction of this same
night.

```python
sci_frames = load_frames(sorted_files["sci"])
print(f"{len(sci_frames)} science frames")

reduced_frames = []
for f in sci_frames:
    r = reduce_frame(f, master_flats, master_darks, None, master_masks,
                     bad_pixel_mask=bad_pixel_mask,
                     flat_exceptions=instrument.flat_exceptions,
                     gain=instrument.gain(f),
                     saturation_limit=instrument.saturation_limit(f))
    reduced_frames.append(r)

r = reduced_frames[0]
print(r["RED-FN"], "| dark subtracted:", r["DARKSUB"],
      "| flat divided:", r["FLATDIV"])
```

In production you would save each frame
(`r.save(os.path.join(paths.reduced_folder, r["RED-FN"]))`) using the
standard folder layout from `ObslogPaths` — see `examples/process_polmode.py`
for the full script version.

**Sky subtraction** comes next when needed. Options in `reduction.sky`:
`subtract_annulus_background` (JHK, usually sufficient),
`subtract_dither_pairs` (L', rapidly varying thermal background),
`subtract_sky_frames` (dedicated master skies). In this tutorial the L'
background is handled per-beam inside `LpPolData.split_beams`, so we skip
this step.

## 5. HWP cycle matching

NIRC2-Pol modulates with the half-wave plate stepping through the four
critical angles **0°, 45°, 22.5°, 67.5°** (header keyword `PCUPR`).
`match_modulator_cycles` walks the frames in time order and groups them into
complete cycles, recording the mapping in each frame's `POLCYCLE` keyword.
Incomplete trailing groups are dropped with a warning.

```python
cycles = instrument.match_modulator_cycles(reduced_frames)
print(f"{len(cycles)} complete HWP cycles")
print("cycle 0 HWP angles:",
      [round(instrument.modulator_angle(f), 1) for f in cycles[0]])
```

## 6. Stokes cubes: split, register, double-difference, rotate

`build_stokes_cubes` does the polarimetric core, per cycle:

1. **beam splitting** — `instrument.split_beams` cuts the two orthogonally
   polarized Wollaston beams (bottom = beam 0, top = beam 1, x-offset
   removed) → a registered `(2, ny, nx)` stack
2. **registration** — the star is located on the mean of the two beams and
   both are shifted together so it lands at the image center.
   `register_method` picks the algorithm, and the right choice depends on
   what the source looks like:

   | source | method |
   |---|---|
   | point source | `"smooth_peak"` (default) |
   | saturated core | `"min"` |
   | resolved body (planet, moon) | `"symmetry"`, `"centroid"`, `"silhouette"` |
   | behind a coronagraph | `"wings"` — masks the core, uses the PSF wings |
   | aligning to a reference | `"crosscorr"` with `template=` |
   | not detectable in one frame | `None` — do not register at all |

   Peak finding fails badly on a resolved body (it locks onto a volcanic
   hotspot ~50 px from Io's disk centre) and on an occulted star. Pass
   `search_center` / `search_radius` to confine the search when a brighter
   neighbour or a detector artefact sits in the field. Any callable
   returning `(cy, cx)` also works.
3. **double differencing** (SPIE Eqs. 1–2, note the ½ factors):
   `Q = ½[(top−bot)(0°) − (top−bot)(45°)]`,
   `U = ½[(top−bot)(22.5°) − (top−bot)(67.5°)]`,
   `I = ¼ Σ (top+bot)`
4. **rotation to sky** (Eqs. 3–5): `θ_rot = −2·PARANG + 2·EL + 2·ROTPDEST
   + 4·θ_off`, then `Q′ = Q cosθ + U sinθ`, `U′ = −Q sinθ + U cosθ`
5. **derotation** to north-up east-left (pyklip rotation)

The **fast axis offset θ_off** has to be determined **on sky** and passed
in explicitly. There is no calibration log to read it from: fitting an HWP
ladder on an internal source returns θ_off + χ/2, where χ is the incident
polarization angle in the instrument frame, and χ is not known for a dome or
lamp source. Leaving θ_off at its 0 deg default does not rotate Q/U into the
sky frame correctly, and the pipeline warns once if you do.

```python
THETA_OFF = 0.0   # [deg] -- REPLACE with a value measured on sky
theta_off = THETA_OFF

stokes_cubes = build_stokes_cubes(instrument, cycles,
                                  fast_axis_offset=theta_off)
print("per-cycle Stokes cubes:", stokes_cubes.shape,
      "  # (ncycles, [I,Q,U], ny, nx)")
```

## 7. Final products

Median-combine the per-cycle cubes, then derive polarized intensity (PI),
angle and degree of linear polarization (AoLP, DoLP), and the radial Stokes
images. With the pipeline's sign convention (SPIE Eqs. 6–7) a
tangentially-polarized disk is **positive in Q_phi** while U_phi contains
noise — verified on this AB Aur dataset.

```python
med = median_stokes_cube(stokes_cubes)          # (3, ny, nx): I, Q, U

# crop to the star (registration put it at the beam-stack center)
ny, nx = instrument.beam_height, 1024 - instrument.beam_x_offset
star = ((ny - 1) / 2, (nx - 1) / 2)
I, Q, U = (crop(p, (400, 400), center=star)[0] for p in med)

pi, aolp, dolp = polarization_products(np.stack([I, Q, U]))
q_phi, u_phi = radial_stokes(Q, U)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
scale = float(np.nanstd(q_phi))          # common stretch for Q_phi and U_phi

im = axes[0].imshow(np.arcsinh(I / np.nanstd(I)), origin="lower", cmap="inferno")
axes[0].set_title("Stokes I (arcsinh stretch)")
fig.colorbar(im, ax=axes[0], fraction=0.046)

for ax, img, title in [(axes[1], q_phi, "Q$_\\phi$ (disk signal)"),
                       (axes[2], u_phi, "U$_\\phi$ (should be noise)")]:
    im = ax.imshow(img, origin="lower", cmap="inferno",
                   vmin=-scale, vmax=2.5 * scale)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig("tutorial_products.png", dpi=100)

# average disk signal in an annulus around the star (star is at the center)
yy, xx = np.mgrid[:400, :400]
radius = np.hypot(yy - 199.5, xx - 199.5)
annulus = (radius > 25) & (radius < 150)
print("Q_phi mean in annulus:", round(float(np.nanmean(q_phi[annulus])), 2))
print("U_phi std  in annulus:", round(float(np.nanstd(u_phi[annulus])), 2))
```

Save products as FITS with full provenance in the header:

```python
header = cycles[0][0].header.copy()
header["THETAOFF"] = theta_off

Frame(np.stack([I, Q, U]), header).save("tutorial_median_stokes.fits")
Frame(np.stack([q_phi, u_phi]), header).save("tutorial_qphi_uphi.fits")
print("saved tutorial_median_stokes.fits, tutorial_qphi_uphi.fits")
```

## 8. Instrumental polarization

Instrumental polarization (IP) is the I → Q/U crosstalk that leaks a
fraction of the total intensity into the polarized channels: an unpolarized
source comes out with `Q = ipq * I`, `U = ipu * I`. It is a property of the
optical train, not the sky, and on NIRC2 it runs of order 1–2 %.

Two ways to measure it, because neither works everywhere.

**Minimizing U_phi** (`fit_ip_uphi`) needs a bright, azimuthally polarized
source — a disk. All the real signal belongs in Q_phi, so the (ipq, ipu)
that minimize the U_phi residual estimate the leakage.

```python
from polarimetry import fit_ip_uphi, mean_ip

ips = [fit_ip_uphi(instrument, c, theta_off, mask_radius=22, crop_size=400)
       for c in cycles]
ip = mean_ip(ips)          # the scatter across cycles is the error bar
print(ip.describe())
```

On the AB Aur commissioning data this gives **ipq = −1.11 ± 0.44 %**,
reproducing the −1.2 % measured previously by a different route. Note the
error bar: three of its eight cycles have U_phi residuals five times the
rest and return nonsense (one gives ipq = −4.1 %). The per-cycle
`diagnostics["uphi_std_final"]` is a usable quality flag — cycles where it
barely improves are worth dropping.

**The mask edge** (`measure_ip_cycle`) needs high-contrast data instead. Just
outside an occulting mask the light is the star's own PSF, assumed
intrinsically unpolarized, so the normalized Stokes there measure the
leakage directly. The annulus defaults to the coronagraph size, which the
instrument reads from `SLITNAME`:

```python
from polarimetry import measure_ip_cycle

ips = [measure_ip_cycle(instrument, c) for c in cycles]   # radii from the mask
```

DoAr 44 behind corona150 gives a 7.5–15 px annulus and **1.85 % at 127°**;
HD 377 behind corona400 gives 20–40 px and **0.92 % at 11°**. Both were
taken in H a week apart, so treat that difference as a caution rather than a
measurement: the annulus sits close in, where a real disk or coronagraph
residuals can violate the unpolarized assumption.

Either way, apply it through `build_stokes_cubes`:

```python
cubes = build_stokes_cubes(instrument, cycles,
                           fast_axis_offset=theta_off, ip=ip)
```

**The correction has to happen in the instrument frame**, before Q/U are
rotated to sky — which is why it is an argument to the cube builder rather
than something you subtract from a finished product. Subtracting `ipq * I`
from a sky-frame Q is only correct if the leakage vector is rotated too.

`ip_frame_annulus=(r_in, r_out)` instead removes a leakage measured
per *exposure* rather than per cycle, which catches variation within a
cycle. It is noisier; check it is needed before switching it on.

`polarimetry/mueller.py` still holds `fit_empirical_cycle_correction`, the
older 16-parameter fit of beam shifts *and* IP together. Prefer `fit_ip_uphi`
unless residual beam misalignment is the actual problem; both are stopgaps
until the Mueller matrix model lands.

## 8a. Fast axis offset, on sky

There is no calibration log for θ_off, and there is no lamp-ladder route to
one. Fitting `A cos(4(θ − θ_fit))` to an HWP ladder returns

    θ_fit = θ_off + χ/2

where χ is the incident polarization angle in the instrument frame. The
phase is degenerate between the offset and the source's own angle, and for
an internal source χ is unknown — so every ladder-derived value carries an
unknown error.

On sky the missing ingredient comes from geometry. A disk's scattered light
is polarized tangentially, so its angle is fixed by position, and a wrong
θ_off shows up as the four-lobe butterfly being rotated:

```python
from polarimetry import butterfly_phase, fit_fast_axis_on_sky, scan_fast_axis_offset
from polarimetry.fast_axis import prepare_cycles

res = fit_fast_axis_on_sky(instrument, cycles, r_inner=25, r_outer=150)
print(res.describe())

# and check it is a real minimum rather than trusting one number
prepared = prepare_cycles(instrument, cycles)
offsets, scores = scan_fast_axis_offset(prepared, r_inner=25, r_outer=150)
```

`butterfly_phase(Q, U)` is the primitive underneath: give it any Q/U pair and
it returns how far the pattern is turned, in degrees. Divide by 4 to get an
offset — `theta_rot` carries `4 * theta_off`, so one degree of offset turns
the polarization frame by four.

On AB Aur all three routes agree: **−12.97°** fitting the offset alone,
**−12.85°** fitting it jointly with the IP, and a scan minimum at
**−13.00°**. The retired lamp-ladder value for that night was −8.18°. The
disagreement is the point rather than a problem — it is roughly the size of
the unmodelled rotation seen on this data before, and χ/2 is exactly the
kind of constant offset that would produce it.

Fit the IP jointly (the default): the two are degenerate, since a constant
leakage tilts the integrated radial Stokes just as a frame rotation does.

**This assumes the source is azimuthally polarized.** On an AGN, a merger or
a star field these routines will happily rotate a genuine U_phi signal into
Q_phi and report a confident number.

## 8b. Writing products, and reading their provenance

Rather than saving files by hand, `ProductWriter` writes the whole set —
reduced frames, one Stokes cube per HWP cycle *and* a stacked cube, the
median Stokes cube, and PI / AoLP / DoLP / Q_phi / U_phi — under a single
output directory you set in one place:

```python
from polarimetry import ProductWriter

writer = ProductWriter("/path/to/output", target="AB_Aur")
writer.save_reduced(reduced_frames)
writer.save_stokes_cycles(stokes_cubes, cycles, header=header)
writer.save_median_stokes(median_cube, header=header)
writer.save_derived_products(median_cube, header=header)
```

Every product records how it was made, so a file can be audited months
later without the script that produced it:

```python
from utils.provenance import describe
print(describe(Frame.load("output/AB_Aur_median_stokes.fits")))
```

```
NIRC2Pol-DPP c71c0e9 processed 2026-07-29T07:15:56
  DPP dark/flat reduction: dark=n1013.fits, flat=n0024.fits, polflat=T
  DPP stokes cube: instrument=NIRC2, background=annulus(150, 220), ...
  DPP median-combined Stokes cube: ncycles=16
```

## 9. The pipeline orchestrator (optional)

For push-button reductions, chain the steps into a `Pipeline`: each step is
a function of the shared context dict, its result is stored under the step's
name, and every intermediate product stays inspectable afterwards.

```python
from pipeline import Pipeline

pipe = Pipeline({"instrument": instrument})
pipe.add_step("cycles", lambda ctx: ctx["instrument"].match_modulator_cycles(reduced_frames))
pipe.add_step("stokes", lambda ctx: build_stokes_cubes(
    ctx["instrument"], ctx["cycles"], fast_axis_offset=theta_off))
ctx = pipe.run()
ctx["stokes"].shape        # every product remains accessible
```

`pipe.run(from_step="stokes")` re-runs from a given step after you tweak a
parameter.

## 10. Conventions, gotchas, and troubleshooting

**Conventions**
- Data arrays are `data[y, x]`; coordinates are `(cy, cx)`, 0-based.
- Double differences carry the ½ factors (SPIE Eqs. 1–2). Reductions that
  omit them (e.g. the commissioning notebooks) are exactly **2×** these
  units.
- Q_phi = +Q cos2φ + U sin2φ (Eq. 6): tangential disk signal is positive.
- `rotate_image_center(img, a)` rotates clockwise in `origin="lower"`
  display (pyklip backend, sign-matched); derotation uses `−north_angle`.
- θ_off must be measured on sky and passed explicitly; there is no trusted
  automatic source, and the 0 deg default is not a calibration.

**Gotchas**
- *Saturated PSF cores*: at L' the core often reads low (a "donut") —
  the default `smooth_peak` centering handles this; `method="min"` is an
  alternative.
- *Field-mask edges*: the Wollaston field mask produces strongly
  "polarized-looking" stripes at the beam edges — mask them in analysis.
- *Inner ~15 px after the empirical correction*: sub-pixel shifts of the
  bright core cause interpolation ringing there.
- *Known limitation*: at the calibrated θ_off, on-sky data show a residual
  ≈25° of unmodeled polarimetric rotation (elevated disk-region U_phi, seen
  on both AB Aur and HL Tau) — this awaits the Mueller matrix model.
- *Beam geometry is per-epoch*: the beams sit (508, 13) px apart in the
  December 2025 L' data but (536, 14) in the 2026 H-band data. Measure it
  for a new epoch rather than inheriting it.
- *Instrumental polarization* runs 1–2 % of Stokes I and makes a four-lobe
  quadrupole in Q_phi and U_phi. Its orientation tracks the polarization
  angle, so Q_phi and U_phi can look "swapped" — that is the IP angle, not
  a bug.
- *Pre-December-2025 data* carries no `PCUPR`; the HWP angle is recovered
  from `OBJECT` with a warning (`check_pol_headers`). Some 2026 frames also
  have malformed CONTINUE cards, which `Frame.save` scrubs automatically.

**Judging a detection**
- **U_phi is the null channel.** If U_phi is comparable to Q_phi there is
  no detection, however suggestive the image looks.
- **PI is positively biased** (it is √(Q²+U²)), so noise alone produces
  signal. Use the signed, unbiased Q_phi for a faint search, and debias PI
  before quoting it.
- **Binning is a test, not just cosmetics.** A real extended source gains
  roughly √N in significance when binned; if the significance stays flat or
  falls, the "source" is noise.

**Troubleshooting
- `No frames at modulator angle X in cycle` — the HWP sequence was
  interrupted; check `POLCYCLE` assignments and drop stragglers (frames
  are matched to the critical angles circularly, so PCUPR = −0.002 ≡ 0).
- Dropped trailing frames warning from cycle matching — normal if the
  sequence didn't end on a complete cycle.
- `No matching dark/flat` warnings — the frame is reduced anyway with
  `DARKSUB`/`FLATDIV` = False; check your calibration coverage.
- Frames missing required keywords are skipped by `sort_frames` with a
  warning listing the missing keys.

## API quick reference

| task | function |
|---|---|
| load/save FITS | `utils.Frame.load / .save`, `load_frames`, `save_frames` |
| night folder layout | `utils.ObslogPaths` |
| sort raw frames | `instrument.sort_frames(files)` |
| master darks / flats | `reduction.make_master_darks / make_master_flats` |
| reduce a frame | `reduction.reduce_frame` |
| sky subtraction | `reduction.subtract_annulus_background / subtract_dither_pairs / subtract_sky_frames` |
| star finding / centering | `reduction.find_center / center_frames / register_beam_stack` |
| HWP cycles | `instrument.match_modulator_cycles(frames)` |
| Stokes cubes | `polarimetry.build_stokes_cubes` |
| combine + products | `polarimetry.median_stokes_cube / polarization_products / radial_stokes` |
| instrumental polarization | `polarimetry.fit_ip_uphi / measure_ip_cycle / measure_ip_annulus / subtract_ip` |
| fast axis, on sky | `polarimetry.fit_fast_axis_on_sky / butterfly_phase / scan_fast_axis_offset` |
| older joint beam-shift + IP fit | `polarimetry.fit_empirical_cycle_correction` (TEMPORARY) |
| exclude bad frames | `utils.load_rejects / record_reject` |
| write products | `polarimetry.ProductWriter(output_dir, target=...)` |
| read provenance | `utils.provenance.describe(frame)` |


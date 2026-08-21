# Tutorial dataset

The smallest subset of a real night that runs `examples/tutorial.ipynb` end to end,
bundled so the tutorial works without access to the observatory archive.

**Source:** AB Aurigae, 2025-12-08 UT, Keck/NIRC2 in L' + Wollaston (narrow
camera, no coronagraph). The full night is 1265 frames; these are 15 of them,
gzipped. They are otherwise untouched — real frames with real headers, which
is half the point of the exercise.

| frames | role | why these |
|---|---|---|
| `n0902`–`n0905` | sky flats | one **complete** HWP cycle at 0, 45, 22.5, 67.5 deg. L' **requires** sky flats, and these classify as such from their `OBJECT` names |
| `n0984`–`n0986` | darks | same 3-frame floor; ITIME 0.45 s × 45 coadds, matching the science |
| `n0932`–`n0939` | AB Aur science | two complete HWP cycles at 0, 45, 22.5, 67.5 deg |

Two cycles rather than one: `median_stokes_cube` over a single cycle is a
no-op, and the tutorial's U_phi and radial-profile discussion would have
nothing to show.

The sky flats **must** cover all four critical angles. These were taken with
the HWP in the beam (`PCUNAME = hwp_center`), and such a flat is only usable as
part of a complete cycle: combining the full set averages the flat source's own
polarization away, and twilight sky is strongly polarized. A partial set is
neither a polarimetric flat (the cycle does not close) nor an ordinary one (the
HWP's transmission is baked in), so `split_polarimetric_flats` discards it with
a warning rather than quietly demoting it.

The night holds three full cycles, `n0902`–`n0913`. Only the first is bundled,
to keep the repo small; add the other two from the archive path below for a
higher signal-to-noise flat. `n0901` is a stray 67.5 deg frame before the
sequence proper and is deliberately **not** included — a second frame at one
angle would weight the cycle average unevenly. `split_polarimetric_flats` now
catches that case itself and sets the surplus frame aside with a warning, so
including it would be harmless but noisy.

Frames are gzipped, ~2.3 MB each rather than 4.06 MB. `astropy` reads
`.fits.gz` transparently, so nothing in the pipeline needs to know — only the
glob pattern does, which is why the tutorial matches `*.fits*`.

To reduce the whole night instead, point `DATA` at
`/home/shared/exoserver/NIRC2_Pol/20251207` on the lab machine and narrow it
with `select_frames(frames, frame_range=(932, 939))` or
`select_frames(frames, target="AB Aur")`. That folder is named for the HST
evening; the frames inside it are all `DATE-OBS = 2025-12-08`, since a Keck
night runs 04:00-16:00 UTC and so falls entirely in the following UTC date.
The pipeline works in UTC throughout, and `ObslogPaths.check_frame_dates`
warns when a dataset folder is named for the local evening instead.

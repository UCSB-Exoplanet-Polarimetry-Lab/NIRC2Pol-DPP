# Tutorial dataset

The smallest subset of a real night that runs `docs/tutorial.md` end to end,
bundled so the tutorial works without access to the observatory archive.

**Source:** AB Aurigae, 2025-12-07 UT, Keck/NIRC2 in L' + Wollaston (narrow
camera, no coronagraph). The full night is 1265 frames; these are 14 of them,
gzipped. They are otherwise untouched — real frames with real headers, which
is half the point of the exercise.

| frames | role | why these |
|---|---|---|
| `n0901`–`n0903` | sky flats | `make_masters` needs 3 frames for a master. L' **requires** sky flats, and these classify as such from their `OBJECT` names |
| `n0984`–`n0986` | darks | same 3-frame floor; ITIME 0.45 s × 45 coadds, matching the science |
| `n0932`–`n0939` | AB Aur science | two complete HWP cycles at 0, 45, 22.5, 67.5 deg |

Two cycles rather than one: `median_stokes_cube` over a single cycle is a
no-op, and the tutorial's U_phi and radial-profile discussion would have
nothing to show.

The sky flats do not need to cover all four critical angles.
`make_master_flats` splits only the generic flats bucket into polarimetric
sets; sky flats are built separately and tagged `SKY`.

Frames are gzipped, ~2.3 MB each rather than 4.06 MB. `astropy` reads
`.fits.gz` transparently, so nothing in the pipeline needs to know — only the
glob pattern does, which is why the tutorial matches `*.fits*`.

To reduce the whole night instead, point `DATA` at
`/home/shared/exoserver/NIRC2_Pol/jaykes_reduction/2025-12-07` on the lab
machine and restore a frame-range filter.

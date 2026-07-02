# nirc2pol — Architectural Conventions

Authoritative reference for units, frames, sign conventions, and the layered
dependency invariant.

## Layered dependency invariant

```
High-level pipeline interface
└── pipeline.py                          orchestrator that pulls multiple
                                         steps together into a "recipe"
        │
        ▼
Science reduction layer
├── reduction/                           dark sub, flat fielding, BP masking,
│                                       sky / dither subtraction, registration
└── polarimetry/                         double differencing, Stokes cubes,
                                         Mueller matrix model interface,
                                         derived products, radial Stokes
        │
        ▼
Instrument-specific layer
├── instruments/base.py                  abstract PolarimetryData
│                                       (including HWP cycle matching)
└── instruments/nirc2/                   NIRC2 data handling +
                                         fast-axis calibration +
                                         Mueller-model math
        │
        ▼
Misc / helper layer
├── utils/                               io and FITS helpers
└── parallel.py                          mp.Pool wrapper
```

`reduction/`, `polarimetry/`, and `pipeline.py` import **only** from
`instruments/base.py` — never from `instruments/nirc2/` directly.
Enforced by `tests/test_invariants.py`.

## Data flow (PDI reduction)

```
Science Data ─┐
              │
              ▼
        Pre-processing  ◄── Darks, Flats
       (dark sub, flat field, BP mask)
              │
              ▼
       Pre-processed Data
              │
              ▼
    Sky / Dither Subtraction  ◄── Sky Flats (optional, JHK)
              │
              ▼
        Sky-subtracted Data
              │
              ▼
        Image Registration
              │
              ▼
          Aligned Data
              │
              ▼
        Build Stokes Cubes  ◄── Fast Axis Value, Mueller Matrix Model Params
              │
              ├──► Stokes Cubes per HWP Cycle
              ├──► Median Stokes Cube
              ├──► PI / DoLP / AoLP
              └──► Radial Stokes Cubes (optional)

Fast Axis Calibration  ◄── Darks, Flats, Fast Axis Calibration Flats
       → Fast Axis Value
```

## Physical conventions

- **Angles:** degrees at external interfaces. Radians only internally,
  with a `_rad` suffix.
- **Position angle:** degrees east of north.
- **Array indexing:** `data[frame, y, x]` — `y` is the first spatial axis.
- **HWP angle:** commanded in degrees; positive = counterclockwise looking
  into the beam.
- **Stokes convention:** `(I, Q, U, V)` IAU. `V` is not computed (linear only).
- **Radial Stokes sign:** `Qφ > 0` = azimuthal polarization (disk-like);
  `Qφ < 0` = radial.
- **AoLP:** `0.5 * atan2(U, Q)`, degrees east of north.
- **Centering:** star coordinates in `(x, y)` pixel order, zero-indexed;
  sub-pixel fractions permitted.

## Array library boundary

| Layer | Library | Reason |
|---|---|---|
| `utils/`, `instruments/nirc2/{io,headers,fast_axis}.py`, `reduction/calibration_files.py` | NumPy | I/O and bookkeeping |
| Remainder of `reduction/` | NumPy | No gradient flow needed |
| `polarimetry/`, `instruments/nirc2/mueller.py` | JAX | Differentiable Mueller path |

Use `jnp` for JAX, `np` for NumPy. Convert explicitly at boundaries.

## Output storage

Derived quantities live in `dataset.output: dict[str, np.ndarray]`, not on
the class.

| Key | Shape | Populated by |
|---|---|---|
| `preprocessed` | `(N, ny, nx)` | reduction steps |
| `star_center` | `(N, 2)` | `reduction/registration.py` |
| `stokes_cubes_per_cycle` | `(N_cycles, 3, ny, nx)` | `polarimetry/stokes.py` |
| `median_stokes_cube` | `(3, ny, nx)` | `polarimetry/stokes.py` |
| `polarized_intensity` | `(ny, nx)` | `polarimetry/derived_products.py` |
| `dolp` | `(ny, nx)` | `polarimetry/derived_products.py` |
| `aolp` | `(ny, nx)` | `polarimetry/derived_products.py` |
| `radial_stokes_cube` | `(2, ny, nx)` | `polarimetry/radial_stokes.py` |

## Variance propagation

Every step that touches `dataset.data` must also touch `dataset.variance`.

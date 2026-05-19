# nirc2pol — Architectural Conventions

Authoritative reference for units, frames, sign conventions, and the layered
dependency invariant. Read alongside `CLAUDE.md`.

## Layered dependency invariant

```
pipeline.py
    │
    ▼
reduction/ · polarimetry/ · calibration/      (science layer)
    │
    ▼
instruments/base.py                            (abstract contract)
    │
    ▼
instruments/nirc2/                             (concrete instrument)
    │
    ▼
io/ · parallel.py · numpy · jax · astropy      (infrastructure)
```

`reduction/`, `polarimetry/`, `calibration/`, and `pipeline.py` import **only**
from `instruments/base.py` — never from `instruments/nirc2/` directly. Enforced
by `tests/test_invariants.py`.

## Physical conventions

- **Angles:** degrees at all external interfaces. Radians only internally, with
  the `_rad` suffix on variable names.
- **Position angle:** degrees east of north.
- **Array indexing:** `data[frame, y, x]` — `y` is the first spatial axis.
- **HWP angle:** commanded in degrees; positive = counterclockwise looking into
  the beam.
- **Stokes convention:** `(I, Q, U, V)` in IAU standard. `V` is not computed
  (linear polarimetry only).
- **Radial Stokes sign:** `Qφ > 0` = azimuthal polarization (disk-like);
  `Qφ < 0` = radial polarization.
- **Centering:** star coordinates in `(x, y)` pixel order, zero-indexed.
  Sub-pixel fractions are permitted.

## Array library boundary

| Layer | Library | Reason |
|---|---|---|
| `io/`, `instruments/nirc2/{io,headers}.py`, `reduction/calibration_files.py` | NumPy | I/O and bookkeeping |
| `reduction/` (excluding calibration_files) | NumPy | No gradient flow |
| `polarimetry/`, `calibration/`, `instruments/nirc2/mueller/` | JAX | Differentiable Mueller path |

Use `jnp` for JAX, `np` for NumPy. Convert explicitly at boundaries.

## Output storage

Derived quantities live in `dataset.output: dict[str, np.ndarray]`, not on the
dataset class. Standard keys:

| Key | Shape | Populated by |
|---|---|---|
| `preprocessed` | `(N, ny, nx)` | reduction steps |
| `star_center` | `(N, 2)` | `reduction/registration.py` |
| `stokes_cube` | `(N_sets, 3, ny, nx)` | `polarimetry/stokes.py` |
| `radial_stokes_cube` | `(N_sets, 2, ny, nx)` | `polarimetry/radial_stokes.py` |

## Variance propagation

Every step that touches `dataset.data` must also touch `dataset.variance`.
See the matching section in `CLAUDE.md` for the per-step rules.

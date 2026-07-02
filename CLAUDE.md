# CLAUDE.md — nirc2pol Pipeline

Authoritative reference for Claude Code working on the `nirc2pol` data
processing pipeline. Read fully before starting any session. Pair with
[`ARCHITECTURE.md`](ARCHITECTURE.md) for physical conventions and the data-flow
diagram.

---

## Project vision

`nirc2pol` is a modular, extensible Python pipeline for polarimetric imaging.
The immediate target is Keck/NIRC2 in polarimetry mode, but the architecture
is designed so adding a future instrument (SPHERE, GPI polarimetry, etc.)
requires only writing a new instrument subpackage — all science and reduction
code is generic.

**Scope:** The pipeline's job is to turn raw polarimetric frames into
calibrated Stokes cubes per HWP cycle, a median Stokes cube, derived
products (PI, DoLP, AoLP), and optionally radial Stokes cubes (Qφ, Uφ).
**No PSF subtraction (no KLIP, no ADI). No photometry (no flux calibration,
no contrast curves).** The stellar PSF is unpolarized, so it cancels
naturally in Qφ; that *is* the PSF-subtraction equivalent for polarized
signal.

Design inspirations:
- **pyKLIP** (`https://pyklip.readthedocs.io`): abstract instrument base
  class pattern, clean separation of instrument-specific vs generic code.
  Architectural pattern only — no KLIP algorithm.
- **CHARIS-DPP** (`https://github.com/thaynecurrie/charis-dpp`): sequential
  pipeline workflow, parameter-file approach, explicit reduction steps.
- **AIR.jl** (`https://github.com/jsnguyen/AIR.jl`): NIRC2-specific dark
  subtraction, flat fielding, header-based frame matching, master
  calibration construction. Reference for `reduction/calibration_files.py`
  and `reduction/dark_flat.py`. Translate Julia → Python.

---

## Repository layout

```
NIRC2Pol-DPP/
├── pyproject.toml
├── ARCHITECTURE.md                # layered diagram, data flow, conventions
├── README.md
├── docs/
│
├── nirc2pol/
│   ├── __init__.py
│   ├── exceptions.py                       # domain exceptions
│   ├── parallel.py                         # multiprocessing helpers
│   ├── pipeline.py                         # orchestrator
│   │
│   ├── instruments/
│   │   ├── __init__.py
│   │   ├── base.py                         # Abstract PolarimetryData
│   │   │                                   # (incl. HWP cycle matching)
│   │   └── nirc2/
│   │       ├── __init__.py                 # exports NIRC2PolarimetryData
│   │       ├── data.py                     # subclass shell + dispatch
│   │       ├── io.py                       # FITS reading
│   │       ├── headers.py                  # Keck FITS keyword normalization
│   │       ├── mueller.py                  # Mueller-model math (flat file)
│   │       └── fast_axis.py                # HWP fast-axis calibration
│   │
│   ├── reduction/
│   │   ├── __init__.py
│   │   ├── calibration_files.py            # build masters (AIR.jl make_masters)
│   │   ├── dark_flat.py                    # apply masters (AIR.jl mass_reduce)
│   │   ├── sky.py                          # subtract_sky_flat / subtract_dither_sky
│   │   └── registration.py                 # populates star_center
│   │
│   ├── polarimetry/
│   │   ├── __init__.py
│   │   ├── angles.py                       # HWP→Stokes weights, PA utilities
│   │   ├── double_diff.py                  # single + double differencing
│   │   ├── mueller.py                      # Mueller-model interface
│   │   ├── stokes.py                       # per-cycle + median Stokes cubes
│   │   ├── derived_products.py             # PI, DoLP, AoLP
│   │   ├── radial_stokes.py                # Qφ, Uφ (optional)
│   │   ├── mueller_fit.py                  # fit Mueller free params
│   │   ├── efficiency.py                   # polarimetric efficiency vs λ
│   │   └── instrumental_pol.py             # IP from unpolarized standards
│   │
│   └── utils/
│       ├── __init__.py
│       ├── fits_utils.py                   # generic FITS read/write helpers
│       └── param_file.py                   # YAML/TOML config parsing
│
└── tests/
    ├── instruments/
    ├── reduction/
    ├── polarimetry/
    ├── test_invariants.py                  # architectural test
    └── test_pipeline.py
```

---

## Abstract base class contract (`instruments/base.py`)

**Raw/observational attributes** every subclass must populate in `read_data()`:

| Attribute | Type | Units | Description |
|---|---|---|---|
| `data` | `np.ndarray (N, ny, nx)` | ADU or e⁻ | Science frames |
| `variance` | `np.ndarray (N, ny, nx)` | (ADU)² or (e⁻)² | Per-pixel variance |
| `wcs` | `list[astropy.wcs.WCS]` | — | One per frame |
| `parangs` | `np.ndarray (N,)` | deg | Parallactic angle per frame |
| `hwp_angles` | `np.ndarray (N,)` | deg | Commanded HWP angle per frame |
| `elevations` | `np.ndarray (N,)` | deg | Telescope elevation |
| `rotator_angles` | `np.ndarray (N,)` | deg | Instrument rotator angle |
| `wavelengths` | `np.ndarray (N,)` | m | Effective wavelength per frame |
| `exposure_times` | `np.ndarray (N,)` | s | Per-frame ITIME |
| `coadds` | `np.ndarray (N,)` | — | Per-frame COADDS |
| `filenames` | `list[Path]` | — | Source file paths |
| `prihdrs`, `exthdrs` | `list` | — | Original FITS headers (provenance) |
| `pixel_scale` | `float` | arcsec/pixel | Detector plate scale |

**Derived quantities do NOT live on the class.** All Stokes cubes, star
centers, PI/DoLP/AoLP maps, and radial Stokes cubes live in
`dataset.output`.

**Required abstract methods:**

```python
class PolarimetryData(ABC):
    @abstractmethod
    def read_data(self, filelist: list[Path]) -> None: ...

    @abstractmethod
    def get_hwp_cycles(self) -> list[list[int]]:
        """Frame-index groups, one per complete HWP cycle."""

    @abstractmethod
    def get_mueller_matrix(self, frame_index: int) -> jnp.ndarray:
        """(4, 4) Mueller matrix for a single frame."""

    @abstractmethod
    def get_mueller_matrix_sequence(self) -> jnp.ndarray:
        """(N, 4, 4) stack of per-frame Mueller matrices."""

    @abstractmethod
    def get_mueller_parameters(self) -> dict[str, float]: ...

    @abstractmethod
    def set_mueller_parameters(self, params: dict[str, float]) -> None: ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write standard output in instrument-appropriate FITS format."""
```

### Output storage

```python
dataset.output: dict[str, np.ndarray] = {}
# Standard keys populated by pipeline steps:
# "preprocessed"               – after dark/flat/sky/register
# "star_center"                – (N, 2) from registration
# "stokes_cubes_per_cycle"     – (N_cycles, 3, ny, nx) for I/Q/U per cycle
# "median_stokes_cube"         – (3, ny, nx) median across cycles
# "polarized_intensity"        – (ny, nx) PI = sqrt(Q² + U²)
# "dolp"                       – (ny, nx) sqrt(Q² + U²) / I
# "aolp"                       – (ny, nx) 0.5 * atan2(U, Q), deg E of N
# "radial_stokes_cube"         – (2, ny, nx) Qφ, Uφ (optional)
```

### Invariant

**`reduction/`, `polarimetry/`, and `pipeline.py` import only from
`instruments/base.py`.** They must never import from `instruments/nirc2/`
directly. Enforced via `tests/test_invariants.py`.

---

## Mueller-matrix model architecture

The Mueller-model math lives inside the instrument subpackage as a single
file (`instruments/nirc2/mueller.py`). The science-layer **interface** —
applying `M⁻¹` and propagating variance — lives in
`polarimetry/mueller.py` and only touches the abstract dataset.

```
polarimetry/mueller.py::correct_instrumental_polarization(dataset)
        │
        │  ONLY calls the abstract interface:
        │
        ▼
PolarimetryData.get_mueller_matrix_sequence()   [abstract]
        │
        │  subclass dispatch
        │
        ▼
NIRC2PolarimetryData.get_mueller_matrix_sequence()
        │
        ▼
instruments/nirc2/mueller.py::full_system_mueller(
    elevation, rotator_angle, hwp_angle, wavelength,
    retardance, fast_axis_offset
)
        │
        └── composes telescope (M1+M2+M3) · AO bench · HWP
```

Physical notes:
- Keck M3 sits at 45° and dominates instrumental polarization.
- Mueller matrix is a function of elevation, rotator angle, HWP angle, and
  wavelength. **Do not hardcode as a constant.**
- NIRC2 HWP retardance is measured (~0.97π) and the fast-axis offset is
  measured per night by `instruments/nirc2/fast_axis.py`.
- Refs: Wiktorowicz & Matthews (2008); van Holstein et al. (2020).

### Calibration hook pattern

`polarimetry/mueller_fit.py` fits free parameters (retardance, AO bench IP
fraction, fast-axis offset, coating constants) without importing any
concrete instrument. It calls `dataset.get_mueller_parameters()` /
`dataset.set_mueller_parameters()` and re-evaluates through
`dataset.get_mueller_matrix_sequence()`.

---

## Fast-axis calibration (`instruments/nirc2/fast_axis.py`)

NIRC2-specific. Uses dedicated **Fast Axis Calibration Flats** plus the
matching darks/flats to measure the HWP fast-axis offset (deg). The
resulting scalar is stored on the dataset via
`NIRC2PolarimetryData.set_fast_axis()` and consumed by
`full_system_mueller()` downstream.

```python
fast_axis_deg, sigma = calibrate_fast_axis(
    fac_flat_paths, master_dark, master_flat
)
dataset.set_fast_axis(fast_axis_deg)
```

Runs once per observing night, not per cycle.

---

## Reduction (`reduction/`)

Four modules. `calibration_files.py` and `dark_flat.py` follow AIR.jl's
`make_masters.jl` → `mass_reduce.jl` two-phase design.

### `reduction/calibration_files.py` — master construction

- Sort raw frames by type using normalized headers from
  `instruments/nirc2/headers.py`.
- **Master darks:** group by (integration time, coadds, readout mode);
  sigma-clip stack; write provenance to header.
- **Master flats:** group by (filter, camera); dark-subtract each raw flat;
  normalize; sigma-clip stack.
- **Bad pixel masks:** hot pixels from dark variance, dead pixels from flat
  response.

### `reduction/dark_flat.py` — application

- Match each science frame to its master dark by (ITIME, COADDS, readout
  mode).
- **If no matching dark exists, raise `NoMatchingCalibrationError`.**
  Never silently substitute.
- Subtract dark, divide by flat, propagate variance.
- Bad-pixel handling as two separate functions:
  `interpolate_bad_pixels()` and `mask_bad_pixels()`.

All NIRC2-specific header keywords (`ITIME`, `COADDS`, `CAMNAME`,
`FILNAME`, etc.) are handled exclusively in
`instruments/nirc2/headers.py`.

### `reduction/sky.py` — sky / dither subtraction

Two functions matching the new data flow:

- `subtract_sky_flat(dataset, sky_flat_path)` — use pre-built Sky Flats
  (optional for JHK).
- `subtract_dither_sky(dataset)` — median-combine the dithered science
  frames (with the source masked) and subtract that median.

The pipeline selects one of the two via the config file.

### `reduction/registration.py`

Frame centering and alignment. Populates `dataset.output["star_center"]`
as `(N, 2)` array of `(x, y)` coordinates.

---

## Polarimetry (`polarimetry/`)

The scientific core. Eight modules.

### `polarimetry/angles.py`

`hwp_to_stokes_weight(hwp_angle_deg, parang_deg, rotator_angle_deg) -> dict`
is the **single source of truth** for mapping HWP commanded angle → Stokes
weights. Docstring must cite NIRC2 documentation and the derivation must be
validated against observations of a polarized standard with known PA before
any science run. Also houses sky↔detector PA utilities and sign conventions.

### `polarimetry/double_diff.py` — single + double differencing

- `double_difference(dataset, cycle_indices) -> (Q, U, Q_var, U_var)` —
  one complete HWP cycle in, ``(Q, U)`` images (+ variances) out.
- `single_difference(frame_a, frame_b)` — half-cycle paired difference.

This is the dominant noise-canceling step for HWP-modulated PDI on a single
detector: instrumental additive offsets cancel pairwise.

### `polarimetry/mueller.py` — Mueller-model interface

- `correct_instrumental_polarization(dataset)` — applies per-frame Mueller
  inversion via `dataset.get_mueller_matrix_sequence()`.
- `propagate_variance(variance, mueller_inverse)` — propagates per-pixel
  variance through `M⁻¹`.

Never imports a concrete instrument.

### `polarimetry/stokes.py` — Build Stokes Cubes

- `compute_stokes_cubes_per_hwp_cycle(dataset)` — calls `double_difference`
  per cycle from `dataset.get_hwp_cycles()`, then Mueller correction. Writes
  `dataset.output["stokes_cubes_per_cycle"]` of shape `(N_cycles, 3, ny, nx)`.
- `compute_median_stokes_cube(dataset)` — median across cycles. Writes
  `dataset.output["median_stokes_cube"]` of shape `(3, ny, nx)`.
- Raises `IncompleteHwpSequenceError` on partial cycles — don't silently
  drop frames.

### `polarimetry/derived_products.py` — PI / DoLP / AoLP

Each function reads `median_stokes_cube` and writes one key under
`dataset.output`. Sign conventions live in `ARCHITECTURE.md`.

### `polarimetry/radial_stokes.py` — optional PDI product

`compute_radial_stokes(dataset)` computes `(Qφ, Uφ)` from `(Q, U)` and
`dataset.output["star_center"]`. Reference: Schmid et al. (2006).
Sign convention: `Qφ > 0` for azimuthal polarization (disk),
`Qφ < 0` for radial. Writes `dataset.output["radial_stokes_cube"]`.

### `polarimetry/{mueller_fit,efficiency,instrumental_pol}.py`

All three operate on the abstract `PolarimetryData` interface only:

- `mueller_fit.fit_mueller_parameters(dataset, expected, free_names, ...)` —
  JAX-autodiff fit of named free Mueller parameters against a standard.
- `efficiency.polarimetric_efficiency(dataset) -> array` — fractional
  polarization recovery vs wavelength.
- `instrumental_pol.measure_instrumental_pol(dataset) -> dict` — residual
  (Q/I, U/I) of an unpolarized standard; diagnostic only.

---

## Variance propagation

Every reduction and polarimetry step that modifies `data` must also update
`variance`. Non-negotiable.

- Dark subtraction: `var += dark_var`
- Flat fielding: standard error propagation for division
- Sky subtraction: `var += sky_var`
- Single/double differencing: linear combination → quadrature sum
- Mueller inversion: propagate through `M⁻¹` (test carefully)
- Median Stokes cube: appropriate median-statistics variance
- PI: `var(PI) = (Q² var(Q) + U² var(U)) / (Q² + U²)`
- Radial Stokes: linear rotation of `(Q, U)` → straightforward

Utility lives in `polarimetry/mueller.py::propagate_variance` and
`utils/fits_utils.py`.

---

## JAX vs NumPy boundary

| Module | Array library | Reason |
|---|---|---|
| `utils/`, `reduction/calibration_files.py`, `instruments/nirc2/{headers,io,fast_axis}.py` | NumPy | I/O and bookkeeping |
| `reduction/{dark_flat,sky,registration}.py` | NumPy | No gradients needed |
| `polarimetry/*` | JAX | Mueller inversion may be differentiated |
| `instruments/nirc2/mueller.py` | JAX | Calibration fits flow through this |

Use `jnp` for JAX, `np` for NumPy. Convert explicitly at boundaries; never
mix silently.

---

## Parallelism

`parallel.py` wraps `multiprocessing.Pool` with a consistent interface.
Apply to:
- Per-frame operations in `reduction/` (dark subtraction, registration)
- Bootstrap iterations in `polarimetry/mueller_fit.py`

Pattern from pyKLIP: `numthreads` parameter, default `os.cpu_count() - 1`.
Sequential fallback (`numthreads=1`) for debugging.

---

## Pipeline orchestrator (`pipeline.py`)

Step-registration pattern rather than hard-coded methods:

```python
class Pipeline:
    def __init__(self, dataset: PolarimetryData, config: dict):
        self.dataset = dataset
        self.config = config
        self.steps: list[tuple[str, Callable]] = []

    def add_step(self, step: Callable, name: str | None = None) -> None: ...
    def run(self) -> None: ...
    def run_until(self, step_name: str) -> None: ...
    def sanity_check(self) -> list[str]:
        """Return list of warnings: missing HWP angles, incomplete cycles, etc."""
```

Convenience constructor: `Pipeline.default_pdi(dataset, config)` assembles
the standard sequence:

```
subtract_dark → divide_flat → (subtract_sky_flat OR subtract_dither_sky)
   → register_frames → correct_instrumental_polarization
   → compute_stokes_cubes_per_hwp_cycle → compute_median_stokes_cube
   → compute_polarized_intensity → compute_degree_of_linear_polarization
   → compute_angle_of_linear_polarization
   → [compute_radial_stokes if config says so]
```

### Ordering constraints

- `compute_radial_stokes` requires `dataset.output["star_center"]` →
  must follow `register_frames`.
- `compute_stokes_cubes_per_hwp_cycle` requires Mueller correction →
  must follow `correct_instrumental_polarization`.
- `compute_median_stokes_cube` requires
  `compute_stokes_cubes_per_hwp_cycle`.
- All derived products require `compute_median_stokes_cube`.
- Mueller calls require `set_fast_axis` to have been invoked (typically by
  a separate calibration script before pipeline run).

The `Pipeline.sanity_check()` method validates these dependencies before
`run()`.

---

## No flag parameters

Follow the style rule strictly:
- NOT `apply_bad_pixel_mask(method="interp" | "nan")` →
  `interpolate_bad_pixels()` and `mask_bad_pixels()`.
- NOT `build_master(kind="dark" | "flat")` → `build_master_dark()` and
  `build_master_flat()`.
- NOT `subtract_sky(dataset, mode="flat" | "dither")` →
  `subtract_sky_flat()` and `subtract_dither_sky()`.

---

## Physical conventions

Documented in [`ARCHITECTURE.md`](ARCHITECTURE.md). Brief recap:

- **Angles:** degrees externally; radians internally with `_rad` suffix.
- **Position angle:** degrees east of north.
- **Array indexing:** `data[frame, y, x]`.
- **HWP angle:** commanded, deg; positive = CCW into the beam.
- **Stokes:** `(I, Q, U, V)` IAU; `V` not computed.
- **Radial Stokes sign:** `Qφ > 0` = azimuthal; `Qφ < 0` = radial.
- **AoLP:** `0.5 * atan2(U, Q)`, deg E of N.
- **Centering:** `(x, y)` pixel order, zero-indexed; sub-pixel OK.

---

## Code style

- Type hints on ALL signatures and class attributes.
- NumPy-style docstrings on all public functions: Summary, Parameters,
  Returns, Raises.
- `pathlib.Path` for paths (not strings).
- `logging` module (not `print`).
- f-strings for formatting.
- Functions ~25 lines — extract private helpers (`_name`) to stay short.
- Files split at ~300 lines.
- **No flag parameters** — see above.
- Comments explain WHY, not WHAT. No commented-out code.
- Units in variable names: `wavelength_m`, `separation_arcsec`, `angle_deg`.
- Guard against division by zero and NaN propagation.
- Document physical assumptions; cite equations and papers in docstrings.

---

## Engineering priorities (when values conflict)

1. Correctness
2. Explicit over clever
3. Edge cases matter
4. DRY — repeated literals → constants; repeated logic → functions
5. Well-tested — every public function, every error path, edge cases
6. "Engineered enough" — simplest correct solution, not fragile,
   not over-abstracted

---

## Testing

Every public function has a test. Every error path has a test. Edge cases:
- Missing darks
- Incomplete HWP cycle (e.g. only 0° and 45°, no 22.5°/67.5°)
- NaN frames
- Mismatched header keywords across frames
- Zero and near-zero variance pixels
- Zero `I` pixels (DoLP divide-by-zero guard)
- Star center at exact pixel vs sub-pixel position (radial Stokes regridding)

Polarimetry validation tests:
- Synthetic data with known polarization fraction and angle → verify
  `(I, Q, U)` recovery within uncertainty.
- Unpolarized standard → residual IP < 0.1%.
- Polarized standard with known PA → verify angle convention.
- Synthetic azimuthally polarized disk → verify `Qφ > 0` everywhere,
  `Uφ ≈ 0`.

Architectural test (`tests/test_invariants.py`):
- Import every module in `reduction/`, `polarimetry/`, and `pipeline.py`
  and assert none transitively import from `nirc2pol.instruments.nirc2`.

### After every implementation task
1. `pytest tests/<relevant_file> -v`
2. Fix implementation (not tests) on failure.
3. Report "N passed, 0 failed".
4. Commit passing work before moving to the next task.

---

## Git workflow

### Session start
```bash
git status
git log --oneline -10
```
Report output before doing any work.

### Commit discipline
- Commit after every completed task and every passing test run.
- Never bundle unrelated changes.
- Never commit broken code.
- WIP: commit with `wip:` prefix or stash.

### Commit message format
```
<type>(<scope>): <short summary>

<optional body: what changed and why>

Tests: N passed, 0 failed
```
Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`. Always include
test results when code was changed.

### Branch strategy
- Solo development on a working branch (currently `development`), not `main`.
- Exploratory: `experiment/<short-description>`.
- Never force-push without explicit instruction.

### Session end
- All completed work committed (no unstaged changes).
- Run `git log --oneline -5` and report session commits as summary.
- Save session state to memory so the next session can continue.

---

## Build order

Each piece independently testable before the next depends on it:

1. `instruments/base.py` — abstract class skeleton + structural tests
2. `instruments/nirc2/headers.py` — pure functions on synthetic headers
3. `instruments/nirc2/io.py` + `data.py` — basic read (Mueller still stubbed)
4. `reduction/calibration_files.py`
5. `reduction/dark_flat.py`, `reduction/sky.py`, `reduction/registration.py`
6. `instruments/nirc2/fast_axis.py` — needs master darks/flats from step 4
7. `instruments/nirc2/mueller.py` — pure math, compare to known matrices
8. Wire `fast_axis` + `mueller` hooks into `instruments/nirc2/data.py`
9. `polarimetry/angles.py`, `polarimetry/double_diff.py`
10. `polarimetry/mueller.py` (interface)
11. `polarimetry/stokes.py` — per-cycle then median
12. `polarimetry/derived_products.py` — PI, DoLP, AoLP
13. `polarimetry/radial_stokes.py` (optional product)
14. `polarimetry/{mueller_fit,efficiency,instrumental_pol}.py`
15. `pipeline.py` — integration tests last

---

## Target end-user API

```python
import nirc2pol
from nirc2pol.instruments.nirc2 import NIRC2PolarimetryData
from nirc2pol.instruments.nirc2.fast_axis import calibrate_fast_axis
from pathlib import Path

dataset = NIRC2PolarimetryData(filelist=list(Path("data/").glob("*.fits")))

# One-off: fast-axis calibration (per night)
fa_deg, _ = calibrate_fast_axis(fac_flat_paths, master_dark, master_flat)
dataset.set_fast_axis(fa_deg)

# High-level: one call
pipe = nirc2pol.Pipeline.default_pdi(dataset, config=Path("reduction.yaml"))
warnings = pipe.sanity_check()
pipe.run()

# Or compose your own pipeline
pipe = nirc2pol.Pipeline(dataset, config)
pipe.add_step(nirc2pol.reduction.subtract_dark)
pipe.add_step(nirc2pol.reduction.divide_flat)
pipe.add_step(nirc2pol.reduction.subtract_dither_sky)
pipe.add_step(nirc2pol.reduction.register_frames)
pipe.add_step(nirc2pol.polarimetry.correct_instrumental_polarization)
pipe.add_step(nirc2pol.polarimetry.compute_stokes_cubes_per_hwp_cycle)
pipe.add_step(nirc2pol.polarimetry.compute_median_stokes_cube)
pipe.add_step(nirc2pol.polarimetry.compute_polarized_intensity)
pipe.add_step(nirc2pol.polarimetry.compute_degree_of_linear_polarization)
pipe.add_step(nirc2pol.polarimetry.compute_angle_of_linear_polarization)
pipe.add_step(nirc2pol.polarimetry.compute_radial_stokes)  # optional
pipe.run()

# Inspect results
cubes  = dataset.output["stokes_cubes_per_cycle"]  # (N_cycles, 3, ny, nx)
median = dataset.output["median_stokes_cube"]      # (3, ny, nx): I, Q, U
pi     = dataset.output["polarized_intensity"]     # (ny, nx)
dolp   = dataset.output["dolp"]                    # (ny, nx)
aolp   = dataset.output["aolp"]                    # (ny, nx)
radial = dataset.output["radial_stokes_cube"]      # (2, ny, nx): Qφ, Uφ
dataset.save(Path("reduced.fits"))
```

---

## Key references

- Wiktorowicz & Matthews (2008) — Keck telescope Mueller matrix
- van Holstein et al. (2020) — instrumental-polarization correction methodology
- Schmid et al. (2006) — radial Stokes (Qφ, Uφ) formalism
- AIR.jl: `https://github.com/jsnguyen/AIR.jl` — NIRC2 dark/flat reference (Julia)
- CHARIS-DPP: `https://github.com/thaynecurrie/charis-dpp` — pipeline workflow reference
- pyKLIP: `https://pyklip.readthedocs.io` — abstract instrument class pattern

# NIRC2-Pol-DPP
NIRC2 Polarimetry Data Processing Pipeline

!! heavily under construction !!

inspired by:

jayke's julia reduction: https://github.com/jsnguyen/AIR.jl

keck nirc2 "calibrated data" page: https://koa.ipac.caltech.edu/UserGuide/NIRC2/calibrated_data.html

page that shows existing pipelines for keck instruments: https://www2.keck.hawaii.edu/inst/drp.html

reduction software from twilight zone programs: https://nirc2-reduce.readthedocs.io/en/latest/description.html

Jessica Lu's IRAF/pyRAF pipeline + her one for distortion: https://keck-datareductionpipelines.github.io/KAI/ and https://github.com/jluastro/nirc2_distortion
--> Natasha Abrams (Jessica's grad student) also recently upgraded this to move away from IRAF https://astro.berkeley.edu/people/natasha-abrams


## Getting started

- **[Tutorial](docs/tutorial.md)** — full walkthrough from raw frames to
  Stokes cubes and radial Stokes images, with conventions, gotchas, and an
  API quick reference. Also available as a runnable notebook:
  [examples/tutorial.ipynb](examples/tutorial.ipynb).
- [examples/process_polmode.py](examples/process_polmode.py) — the same
  reduction as a plain script.

## Architecture

Layered design (see the SPIE proceedings for details), separating
instrument-specific I/O from the science reduction:

- `pipeline.py` — high-level orchestrator that pulls steps into a "recipe"
- `reduction/` — science reduction layer: master darks/flats/skies,
  frame calibration, sky/dither subtraction, image registration
- `polarimetry/` — polarimetric science layer: HWP double differencing,
  Stokes cube production (per-cycle + median), PI/AoLP/DoLP, radial Stokes,
  Mueller matrix model interface
- `instruments/` — instrument-specific layer: abstract `PolarimetryData`
  class (`base.py`, incl. HWP cycle matching) and the NIRC2 implementation
  (`nirc2.py`, incl. fast axis calibration and the polarimetric rotation
  model)
- `utils/` — low-level helpers: FITS I/O (`Frame`), image operations,
  Gaussian fitting, path conventions
- `examples/process_polmode.py` — end-to-end reduction from raw frames to
  Stokes products, as plain function calls

Run with the `nirc2p` conda environment (numpy / scipy / astropy).

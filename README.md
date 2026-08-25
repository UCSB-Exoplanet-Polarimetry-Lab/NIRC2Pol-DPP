# NIRC2-Pol DPP
Last Updated: 14 Aug 2026

**THE PIPELINE IS STILL UNDER DEVELOPMENT AND HAS NOT BEEN FULLY TESTED**

The NIRC2 Polarimetry Data Processing Pipeline is an open-source, Python-based, user-friendly polarimetric data reduction for Keck/NIRC2-Pol. NIRC2 Polarimetry (NIRC2-Pol, or nirc2p) is a dual-channel polarimetry mode on the Keck II NIRC2 infrared imager. Dual-channel polarimetry uses a polarizing beamsplitter to split the incoming light into two orthogonal polarization states, and a half-wave plate (HWP) to modulate the angle of polarization. Through cycles of four critical HWP angles (0°, 45°, 22.5°, 67.5°), it is possible to recover the linear Stokes vector components Q and U (see de Boer+ 2020 for a description of dual-channel polarimetry and double differencing). NIRC2-Pol enables polarimetric observations in JHKL’ bands in combination with multiple existing NIRC2 modes, such as grism spectroscopy and high-contrast coronagraphic imaging, and both NGS and LGS AO. This is useful for many science cases, from solar system objects to circumstellar disks and active galactic nuclei. NIRC2-Pol was developed as part of the Precision Calibration Unit (PCU2) project on Keck II. Operational scripts for the mode can be found at [NIRC2Pol-Ops](https://github.com/UCSB-Exoplanet-Polarimetry-Lab/NIRC2Pol-Ops). A [draft version of the NIRC2-Pol operations/observer's guide](https://docs.google.com/document/d/1xZ5t1CYUM9_GUHD_lKeaxhGwf5xAPUc301j2dv2oiKI/edit?tab=t.v9hqfo1pspp7#heading=h.ej8cynj3sfoq) is also available.

## Documentation

Coming soon!

## Features
- Dark subtraction
- Both polarimetric and standard flat fielding
- Multiple options for image registration
- Stokes cube generation with options for combining HWP cycles, converting to radial Stokes
- Multiple methods for empirical instrumental polarization correction and HWP fast axis determination
- Mueller matrix model correction from Zhang et al. 2026 [coming soon]

## Installation

Python ≥ 3.11 (the pipeline reads its configuration with `tomllib`), plus
`numpy`, `scipy`, `astropy`, `scikit-image` and `pyklip`.

The conda route builds the environment and installs the pipeline into it:

```
conda env create -f environment.yml
conda activate nirc2p
```

Or install into an environment you already have, from a checkout:

```
pip install -e .
```

...or straight from GitHub, without one:

```
pip install git+https://github.com/UCSB-Exoplanet-Polarimetry-Lab/NIRC2Pol-DPP.git
```

Either way `import nirc2pol` then works from any directory, so a notebook does
not have to live in the repository or edit `sys.path` to find it:

```python
from nirc2pol.reduction.config import ReductionConfig
from nirc2pol.polarimetry import build_stokes_cubes, radial_stokes
from nirc2pol.instruments.nirc2 import NIRC2PolarimetryData
```

## Running a reduction

Every choice a reduction makes lives in one TOML file. Write a fresh one,
listing every option with its default and its allowed values:

```
nirc2pol-reduce --template > my_night.toml
```

Edit it -- at minimum these four -- then run it:

```toml
[paths]
raw_data_folder = "/data/NIRC2_Pol/20251207"   # where the frames are
reductions_root = "/home/you/reductions/AB_Aur_Lp"  # where this run goes
date            = "2025-12-08"                 # UTC, as DATE-OBS records it
target          = "AB_Aur"
```

```
nirc2pol-reduce my_night.toml
```

Re-running after changing one setting does not have to redo the whole night:

```
nirc2pol-reduce my_night.toml --resume reduced   # straight to cycle matching
nirc2pol-reduce my_night.toml --resume masters   # reuse the calibrations
```

`--resume reduced` reloads the corrected frames and skips to cycle matching,
which is most of the time -- use it when iterating on the fast axis offset,
the leakage, the crop or the products. `--resume masters` reloads the
calibrations and re-runs the science reduction, for iterating on how frames
are flat-fielded and sky-subtracted. Either way the products come out the same
as a reduction from raw. Both need the earlier run to have had `save_preproc`
on, and both refuse rather than guess when the folder does not match the
config -- frames from a different selection, or frames written before the
dither stage ran, which would otherwise carry on with no background subtracted
at all.

Those first two are deliberately separate. The frames stay wherever they are
-- an archive, shared space, a mounted volume -- and are only ever read; the
run symlinks the ones it needs into `reductions_root/raw/` and writes
everything else there too, so a reduction folder is a self-contained record of
its own inputs and nothing is written back into the data:

```
reductions_root/
    raw/          symlinks to the frames this run read
    reduced/      dark-subtracted, flat-divided frames
    sequences/    per-cycle Stokes cubes, median cube, PI / AoLP / DoLP,
                  Q_phi / U_phi
    plots/
    master_darks_<date>.fits, master_flats_<date>.fits
    reduction_<date>.log
```

`date` locates nothing: it names the masters and the log, and is checked
against the frames' own `DATE-OBS`.

## Combining several nights

Reduce each night on its own, then join the results:

```
nirc2pol-combine --template > combined.toml
nirc2pol-combine combined.toml
```

It median-combines the per-cycle Stokes cubes of the reductions you list. The
join happens *after* reduction rather than before, so that each night keeps
its own darks, flats and beam geometry -- pooling raw frames from two nights
would let one night's flat calibrate the other night's data, since flats are
matched on filter and detector size and not on date.

The same reduction is one call from a notebook, which returns everything it
built so you can look at any of it:

```python
from nirc2pol.reduction.config import ReductionConfig
from nirc2pol.recipe import run

products = run(ReductionConfig.from_toml("my_night.toml"))
products["median_cube"].shape
```

`examples/tutorial.ipynb` walks through the same steps individually, on a
small dataset bundled with the repository, and explains what each one is for.

## Contributions / Questions

If you wish to contribute to the NIRC2-Pol DPP, or have any questions about its use, please open an issue on the GitHub to start a discussion.

## Citing the NIRC2-Pol DPP

If you use the NIRC2-Pol DPP in your work, please cite the following works (updated bibTeX citations coming soon):
(1) NIRC2-Pol's first light paper (Lewis et al. 2026b, in prep)
```
@article{lewis2026nirc,
  title={NIRC2-Pol: First Light of Near-Infrared Polarimetry on Keck II},
  author={Lewis, Briley L. and Zhang, Rebecca and Millar-Blanchaer, Maxwell and Marin, Eduardo and Nguyen, Jayke and Melby, William and others},
  year={In Prep.}
}
```
(2) NIRC2-Pol DPP design SPIE proceeding (Lewis et al. 2026a)
```
@misc{lewis2026opensourcedataprocessingpipeline,
      title={An open-source data processing pipeline for Keck / NIRC2-Polarimetry}, 
      author={Briley L. Lewis and Maxwell A. Millar-Blanchaer and Rebecca Zhang and Jayke Nguyen and Max Brodheim and Ashish Uhlmann},
      year={2026},
      eprint={2608.14864},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2608.14864}, 
}
```
(3) NIRC2-Pol preliminary Mueller matrix model calibration SPIE proceeding (Zhang et al. 2026)
```
@misc{zhang2026enablingquantitativepolarimetrykecknirc2,
      title={Enabling Quantitative Polarimetry for Keck/NIRC2: Preliminary Mueller Matrix Model Calibration}, 
      author={Manxuan Zhang and Briley L. Lewis and Maxwell A. Millar-Blanchaer and Eduardo Marin and Jayke S. Nguyen and William Melby and Carlos Alvarez and Jaren N. Ashcraft and Mahawa Cisse and Charles-Antoine Claveau and Jacques-Robert Delorme and Greg Doppmann and Michael P. Fitzgerald and Matthew Freeman and Percy Gomez and Trisha Hammen and Ryan Hersey and Nemanja Jovanovic and Marc Kassis and Scott Lilley and Jessica Lu and James E. Lyke and Keith Matthews and Dimitri Mawet and Thomas McIntosh and Max Service and Lauren Simmons and Jacob Taylor and Rob G. van Holstein and Ed Wetherell},
      year={2026},
      eprint={2608.14873},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2608.14873}, 
}
```

## Acknowledgements 

NIRC2-Pol PI: Max Millar-Blanchaer (UCSB)
NIRC2-Pol Team: Briley Lewis, Rebecca Zhang (UCSB); Jayke Nguyen (UCSD); Ryan Hersey, Thomas McIntosh, Jaren Ashcraft (UCSB); Will Melby (U of A); Mike Fitzgerald (UCLA); Dimitri Mawet, Nem Jovanovic, Keith Matthews (Caltech)
PCU2 Team: Jessica Lu, Charles-Antoine Claveau, Matthew Freeman (Berkeley); Eduardo Marin, Scott Lilley, Ed Wetherell, Jacob Taylor, Mahawa Cisse, Lauren Simmons, Carlos Alvarez, Percy Gomez, Max Service, Trisha Hammen, Jim Lyke, Greg Doppmann (Keck)

Some of the data presented herein were obtained at Keck Observatory, which is a private 501(c)3 non-profit organization operated as a scientific partnership among the California Institute of Technology, the University of California, and the National Aeronautics and Space Administration. The Observatory was made possible by the generous financial support of the W. M. Keck Foundation. The authors wish to recognize and acknowledge the very significant cultural role and reverence that the summit of Maunakea has always had within the Native Hawaiian community. We are most fortunate to have the opportunity to conduct observations from this mountain.

This material is based upon work supported by the National Science Foundation Astronomy \& Astrophysics Postdoctoral Fellowship Award No. 2401654. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation. This work was also supported by the Mt. Cuba Astronomical Foundation and the University of California Observatories Mini-Grant Program.

This repository was written with the help of Claude Version 1.24012.9 (o3c61d) using Fable 5 and Opus 5. Thank you to Jayke Nguyen for writing [AIR.jl](https://github.com/jsnguyen/AIR.jl) which was adapted into Python for the basic reduction steps in the DPP (e.g. dark subtraction, flat fielding).

-------------------------

Contact: Briley Lewis, brileylewis@ucsb.edu

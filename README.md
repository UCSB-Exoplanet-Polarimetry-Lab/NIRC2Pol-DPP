# NIRC2-Pol DPP
Last Updated: 5 Aug 2026

The NIRC2 Polarimetry Data Processing Pipeline is an open-source, Python-based, user-friendly polarimetric data reduction for Keck/NIRC2-Pol. NIRC2 Polarimetry (NIRC2-Pol, or nirc2p) is a dual-channel polarimetry mode on the Keck II NIRC2 infrared imager. Dual-channel polarimetry uses a polarizing beamsplitter to split the incoming light into two orthogonal polarization states, and a half-wave plate (HWP) to modulate the angle of polarization. Through cycles of four critical HWP angles (0°, 45°, 22.5°, 67.5°), it is possible to recover the linear Stokes vector components Q and U (see de Boer+ 2020 for a description of dual-channel polarimetry and double differencing). NIRC2-Pol enables polarimetric observations in JHKL’ bands in combination with multiple existing NIRC2 modes, such as grism spectroscopy and high-contrast coronagraphic imaging, and both NGS and LGS AO. This is useful for many science cases, from solar system objects to circumstellar disks and active galactic nuclei. NIRC2-Pol was developed as part of the Precision Calibration Unit (PCU2) project on Keck II. Operational scripts for the mode can be found at [NIRC2Pol-Ops](https://github.com/UCSB-Exoplanet-Polarimetry-Lab/NIRC2Pol-Ops). A [draft version of the NIRC2-Pol operations/observer's guide](https://docs.google.com/document/d/1xZ5t1CYUM9_GUHD_lKeaxhGwf5xAPUc301j2dv2oiKI/edit?tab=t.v9hqfo1pspp7#heading=h.ej8cynj3sfoq) is also available.

## Documentation

Coming soon!

## Features
- Dark subtraction
- Both polarimetric and standard flat fielding
- Multiple options for image registration
- Stokes cube generation with options for combining HWP cycles, converting to radial Stokes
- Mueller matrix model correction from Zhang et al. 2026 [coming soon]

## Installation

The NIRC2-Pol DPP requires Python ≥ 3.11 with `numpy`, `scipy`, `astropy`, and `pyklip`.

Further instructions coming soon!

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
@article{lewis2026dpp,
  title={An open-source data processing pipeline for Keck /
NIRC2-Polarimetry},
  author={Lewis, Briley L. and Zhang, Rebecca and Millar-Blanchaer, Maxwell and Nguyen, Jayke and Brodheim, Max and Uhlmann, Ashish},
  year={Submitted}
}
```
(3) NIRC2-Pol preliminary Mueller matrix model calibration SPIE proceeding (Zhang et al. 2026)
```
@article{zhang2026mueller,
  title={Enabling Quantitative Polarimetry for Keck/NIRC2:
Preliminary Mueller Matrix Model Calibration},
  author={Zhang, Rebecca and Lewis, Briley L. and Millar-Blanchaer, Maxwell and Marin, Eduardo and Nguyen, Jayke and Melby, William and others},
  year={In Prep.}
}
```

## Acknowledgements 

NIRC2-Pol PI: Max Millar-Blanchaer (UCSB)
NIRC2-Pol Team: Briley Lewis, Rebecca Zhang (UCSB); Jayke Nguyen (UCSD); Ryan Hersey, Thomas McIntosh, Jaren Ashcraft (UCSB); Will Melby (U of A); Mike Fitzgerald (UCLA); Dimitri Mawet, Nem Jovanovic, Keith Matthews (Caltech)
PCU2 Team: Jessica Lu, Charles-Antoine Claveau, Matthew Freeman (Berkeley); Eduardo Marin, Scott Lilley, Ed Wetherell, Jacob Taylor, Mahawa Cisse, Lauren Simmons, Carlos Alvarez, Paul Richards, Percy Gomez, Max Service, Trisha Hammen, Jim Lyke, Greg Doppmann (Keck)

Some of the data presented herein were obtained at Keck Observatory, which is a private 501(c)3 non-profit organization operated as a scientific partnership among the California Institute of Technology, the University of California, and the National Aeronautics and Space Administration. The Observatory was made possible by the generous financial support of the W. M. Keck Foundation. The authors wish to recognize and acknowledge the very significant cultural role and reverence that the summit of Maunakea has always had within the Native Hawaiian community. We are most fortunate to have the opportunity to conduct observations from this mountain.

This material is based upon work supported by the National Science Foundation Astronomy \& Astrophysics Postdoctoral Fellowship Award No. 2401654. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation. This work was also supported by the Mt. Cuba Astronomical Foundation and the University of California Observatories Mini-Grant Program.

This repository was written with the help of Claude Version 1.24012.9 (o3c61d) using Fable 5 and Opus 5. Thank you to Jayke Nguyen for writing [AIR.jl](https://github.com/jsnguyen/AIR.jl) which was adapted into Python for the basic reduction steps in the DPP (e.g. dark subtraction, flat fielding).

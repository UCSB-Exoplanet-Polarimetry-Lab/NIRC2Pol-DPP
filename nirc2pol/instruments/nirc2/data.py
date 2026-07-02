"""NIRC2PolarimetryData — concrete subclass wiring abstract methods to NIRC2 internals."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from nirc2pol.instruments.base import PolarimetryData
from nirc2pol.instruments.nirc2 import io as nirc2_io
from nirc2pol.instruments.nirc2.mueller import full_system_mueller

# Default Mueller-model free parameters (from working-ab-aur2 notebook).
_DEFAULT_MUELLER_PARAMS: dict[str, float] = {
    "fast_axis_offset_deg": -8.0278,
    "retardance_rad": float(np.pi),  # ideal half-wave; refine via mueller_fit
}


class NIRC2PolarimetryData(PolarimetryData):
    """Keck/NIRC2 polarimetric dataset.

    Parameters
    ----------
    filelist
        FITS paths for a single observing sequence. If provided,
        :meth:`read_data` is invoked from ``__init__``.
    chop_kwargs
        Optional kwargs forwarded to
        :func:`nirc2pol.instruments.nirc2.io.chop_wollaston_frames`.
    """

    cycle_size: int = 4  # frames per HWP cycle (0, 22.5, 45, 67.5)

    def __init__(
        self,
        filelist: list[Path] | None = None,
        chop_kwargs: dict | None = None,
    ) -> None:
        self.output: dict[str, np.ndarray] = {}
        self._mueller_params: dict[str, float] = dict(_DEFAULT_MUELLER_PARAMS)
        self._chop_kwargs = chop_kwargs or {}
        if filelist is not None:
            self.read_data(filelist)

    def read_data(self, filelist: list[Path]) -> None:
        d = nirc2_io.read_nirc2_files(filelist, chop_kwargs=self._chop_kwargs)
        for key, value in d.items():
            setattr(self, key, value)

    def get_hwp_cycles(self) -> list[list[int]]:
        """Group frame indices into complete HWP cycles of ``cycle_size`` frames."""
        n = self.data.shape[0]
        n_complete = (n // self.cycle_size) * self.cycle_size
        return [
            list(range(i, i + self.cycle_size))
            for i in range(0, n_complete, self.cycle_size)
        ]

    def get_mueller_matrix(self, frame_index: int) -> np.ndarray:
        return full_system_mueller(
            parang_deg=float(self.parangs[frame_index]),
            elevation_deg=float(self.elevations[frame_index]),
            rotator_angle_deg=float(self.rotator_angles[frame_index]),
            hwp_angle_deg=float(self.hwp_angles[frame_index]),
            wavelength_m=float(self.wavelengths[frame_index]),
            retardance_rad=self._mueller_params["retardance_rad"],
            fast_axis_offset_deg=self._mueller_params["fast_axis_offset_deg"],
        )

    def get_mueller_matrix_sequence(self) -> np.ndarray:
        return np.array(
            [self.get_mueller_matrix(i) for i in range(self.data.shape[0])]
        )

    def get_mueller_parameters(self) -> dict[str, float]:
        return dict(self._mueller_params)

    def set_mueller_parameters(self, params: dict[str, float]) -> None:
        self._mueller_params.update(params)

    def save(self, path: Path) -> None:
        """Write the median Stokes cube + derived products to a FITS file.

        Layout: PRIMARY = median Stokes ``(3, ny, nx)``; extensions for
        ``polarized_intensity``, ``dolp``, ``aolp``, ``radial_stokes_cube``
        if present in ``output``.
        """
        median = self.output["median_stokes_cube"]
        hdul = fits.HDUList([fits.PrimaryHDU(median.astype(np.float32))])
        for key in ("polarized_intensity", "dolp", "aolp", "radial_stokes_cube"):
            if key in self.output:
                hdul.append(fits.ImageHDU(self.output[key].astype(np.float32), name=key.upper()))
        hdul.writeto(path, overwrite=True)

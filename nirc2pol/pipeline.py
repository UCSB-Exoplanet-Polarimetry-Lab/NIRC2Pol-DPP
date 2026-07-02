"""Pipeline orchestrator — step registration, ordering checks, run/run_until."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from nirc2pol.exceptions import PipelineOrderingError
from nirc2pol.instruments.base import PolarimetryData

Step = Callable[[PolarimetryData, dict[str, Any]], None]

logger = logging.getLogger(__name__)


class Pipeline:
    """Compose and run an ordered sequence of reduction + polarimetry steps."""

    def __init__(self, dataset: PolarimetryData, config: dict[str, Any]) -> None:
        self.dataset = dataset
        self.config = config
        self.steps: list[tuple[str, Step]] = []

    def add_step(self, step: Step, name: str | None = None) -> None:
        """Append a step. ``name`` defaults to ``step.__name__``."""
        self.steps.append((name or step.__name__, step))

    def run(self) -> None:
        """Run every registered step in order. Calls :meth:`sanity_check` first."""
        for warning in self.sanity_check():
            logger.warning(warning)
        for name, step in self.steps:
            logger.info("Running step: %s", name)
            step(self.dataset, self.config)

    def run_until(self, step_name: str) -> None:
        """Run registered steps up to and including ``step_name``."""
        for name, step in self.steps:
            step(self.dataset, self.config)
            if name == step_name:
                return
        raise PipelineOrderingError(f"step {step_name!r} not registered")

    def sanity_check(self) -> list[str]:
        """Return a list of warning strings; empty list means OK."""
        warnings: list[str] = []
        n = getattr(self.dataset, "data", None)
        if n is None:
            warnings.append("dataset.data is not populated — did read_data run?")
        if not self.steps:
            warnings.append("no steps registered")
        return warnings

    @classmethod
    def default_pdi(
        cls, dataset: PolarimetryData, config: dict[str, Any]
    ) -> "Pipeline":
        """Standard NIRC2 PDI sequence matching the working AB Aur notebook.

        Skips dark/flat (assumed already applied upstream) and skips the
        separate Mueller-correction step (folded into Stokes-cube assembly).
        """
        from nirc2pol import polarimetry, reduction

        pipe = cls(dataset, config)
        pipe.add_step(
            lambda d, c: reduction.subtract_box_mean_background(
                d, c.get("box", (25, 350, 50, 400))
            ),
            name="subtract_box_mean_background",
        )
        pipe.add_step(
            lambda d, c: reduction.register_frames(
                d,
                center_xy=c.get("center_xy", (600, 300)),
                cutout_size=c.get("cutout_size", 400),
                padsize=c.get("padsize", 400),
            ),
            name="register_frames",
        )
        pipe.add_step(polarimetry.compute_stokes_cubes_per_hwp_cycle, name="compute_stokes_cubes_per_hwp_cycle")
        pipe.add_step(polarimetry.compute_median_stokes_cube, name="compute_median_stokes_cube")
        pipe.add_step(polarimetry.compute_polarized_intensity, name="compute_polarized_intensity")
        pipe.add_step(polarimetry.compute_degree_of_linear_polarization, name="compute_degree_of_linear_polarization")
        pipe.add_step(polarimetry.compute_angle_of_linear_polarization, name="compute_angle_of_linear_polarization")
        if config.get("compute_radial_stokes", True):
            pipe.add_step(polarimetry.compute_radial_stokes, name="compute_radial_stokes")
        return pipe

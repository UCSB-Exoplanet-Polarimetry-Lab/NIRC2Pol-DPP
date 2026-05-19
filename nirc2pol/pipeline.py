"""Pipeline orchestrator — step registration, ordering checks, run/run_until."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nirc2pol.instruments.base import PolarimetryData

Step = Callable[[PolarimetryData, dict[str, Any]], None]


class Pipeline:
    """Compose and run an ordered sequence of reduction + polarimetry steps.

    Parameters
    ----------
    dataset
        Instrument-specific :class:`PolarimetryData` subclass instance.
    config
        Parameter dict (typically loaded from YAML/TOML via
        :func:`nirc2pol.io.param_file.load_config`).
    """

    def __init__(self, dataset: PolarimetryData, config: dict[str, Any]) -> None:
        self.dataset = dataset
        self.config = config
        self.steps: list[tuple[str, Step]] = []

    def add_step(self, step: Step, name: str | None = None) -> None:
        """Append a step. ``name`` defaults to ``step.__name__``."""
        raise NotImplementedError

    def run(self) -> None:
        """Run every registered step in order. Calls :meth:`sanity_check` first."""
        raise NotImplementedError

    def run_until(self, step_name: str) -> None:
        """Run registered steps up to and including the one named ``step_name``."""
        raise NotImplementedError

    def sanity_check(self) -> list[str]:
        """Return a list of warning strings (empty = OK).

        Catches: missing HWP angles, incomplete Stokes sets, steps registered
        out of order (e.g. ``compute_radial_stokes`` before ``register_frames``).
        """
        raise NotImplementedError

    @classmethod
    def default_pdi(cls, dataset: PolarimetryData, config: dict[str, Any]) -> "Pipeline":
        """Standard PDI sequence: dark → flat → sky → register → Mueller → Stokes → radial Stokes."""
        raise NotImplementedError

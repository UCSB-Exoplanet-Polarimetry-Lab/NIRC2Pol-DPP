"""YAML / TOML config-file parsing for pipeline parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """Load a pipeline config from YAML or TOML, dispatched by file suffix."""
    raise NotImplementedError


def validate_config(config: dict[str, Any]) -> list[str]:
    """Return a list of human-readable warnings; empty list means OK."""
    raise NotImplementedError

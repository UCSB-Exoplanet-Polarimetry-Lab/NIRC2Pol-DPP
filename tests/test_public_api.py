"""What each package offers, and what a wildcard import gets you.

``__init__.py`` re-exports are what let code move between modules without
breaking callers -- which happened repeatedly while ``fit_beam_geometry``,
``parse_date_obs`` and the sky-subtraction dispatch were relocated. The cost
is a hand-maintained list, and a hand-maintained list drifts: the dispatch
went unexported for several commits, and ``__all__`` was first written here
naming a class that had already been deleted.
"""

import importlib

import pytest

PACKAGES = ["utils", "reduction", "polarimetry", "instruments"]


@pytest.mark.parametrize("name", PACKAGES)
def test_every_exported_name_resolves(name):
    """A name in __all__ that does not exist makes ``import *`` raise, which
    is how the deleted RotationApproximationModel was caught."""
    module = importlib.import_module(name)
    missing = [n for n in module.__all__ if not hasattr(module, n)]
    assert not missing, f"{name}.__all__ names nothing: {missing}"


@pytest.mark.parametrize("name", PACKAGES)
def test_a_wildcard_import_gets_the_api_and_nothing_else(name):
    """Without __all__, ``from utils import *`` also binds frame, paths,
    angles and the rest, which would shadow a caller's own variables."""
    module = importlib.import_module(name)
    namespace = {}
    exec(f"from {name} import *", namespace)  # noqa: S102
    got = {n for n in namespace if not n.startswith("__")}

    assert got == set(module.__all__)
    leaked = [n for n in got
              if getattr(getattr(module, n, None), "__package__", None) == name]
    assert not leaked, f"submodules leaked into the wildcard: {leaked}"


@pytest.mark.parametrize("name", PACKAGES)
def test_the_export_list_is_sorted_and_unique(name):
    """Sorted so additions land in an obvious place rather than at the end,
    where two people appending at once conflict."""
    exported = importlib.import_module(name).__all__
    assert exported == sorted(exported), f"{name}.__all__ is out of order"
    assert len(exported) == len(set(exported)), f"{name}.__all__ has repeats"


def test_the_instrument_package_stays_generic():
    """NIRC2 specifics live at instruments.nirc2. Flattening them into the
    package root would make ``from instruments import band_of`` ambiguous
    the moment a second instrument exists."""
    import instruments

    assert set(instruments.__all__) == {"PolarimetryData", "config_csv",
                                        "read_config"}
    for nirc2_only in ("band_of", "NIRC2PolarimetryData", "PLATE_SCALE"):
        assert nirc2_only not in instruments.__all__

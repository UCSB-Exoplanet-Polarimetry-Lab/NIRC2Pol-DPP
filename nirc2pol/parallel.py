"""Multiprocessing helper wrapping ``multiprocessing.Pool`` with a uniform API.

Pattern borrowed from pyKLIP: ``numthreads`` parameter, default
``os.cpu_count() - 1``. ``numthreads=1`` runs sequentially for debugging.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")


def parallel_map(
    fn: Callable[[T], U],
    items: Iterable[T],
    numthreads: int | None = None,
) -> list[U]:
    """Apply ``fn`` to every item, optionally in parallel.

    Parameters
    ----------
    fn
        Pure function to apply per item.
    items
        Iterable of inputs.
    numthreads
        ``None`` → ``os.cpu_count() - 1``. ``1`` → sequential (for debugging).
    """
    raise NotImplementedError

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

"""Observability helpers and context managers."""

@contextmanager
def span(name: str) -> Iterator[None]:
    yield

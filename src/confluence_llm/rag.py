from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

from store import ScoredDoc, get_store


@runtime_checkable
class SupportsSearch(Protocol):
    def search(
        self,
        query: str,
        k: int = 5,
        user_profile: dict[str, object] | None = None,
    ) -> list[ScoredDoc]:
        ...


class RAG:
    def __init__(self, store: object | None = None) -> None:
        # Accept a broad type at the boundary and cast to our protocol internally.
        backing = store if store is not None else get_store()
        self.store: SupportsSearch = cast(SupportsSearch, backing)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        user_profile: dict[str, object] | None = None,
    ) -> list[ScoredDoc]:
        return self.store.search(query, k=k, user_profile=user_profile or {})

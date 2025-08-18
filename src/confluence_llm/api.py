from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Query
from permissions import UserProfile
from rag import RAG
from store import get_store

app = FastAPI(title="DCPS Confluence RAG")

# Concrete instances (avoid using functions as types/classes)
STORE = get_store()
RAG_PIPE = RAG(STORE)

def _resolve_ingest() -> Callable[[Any, Sequence[str] | None], Any]:
    """Load the ingest function at request time to avoid static symbol issues."""
    try:
        ingest_mod = importlib.import_module("ingest")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Unable to import 'ingest' module.",
        ) from e

    func = getattr(ingest_mod, "ingest_all", None) or getattr(ingest_mod, "ingest", None)
    if not callable(func):
        raise HTTPException(
            status_code=500,
            detail="No callable 'ingest_all' or 'ingest' found in ingest.py.",
        )

    return cast(Callable[[Any, Sequence[str] | None], Any], func)


def _to_jsonable_list(items: Any) -> list[dict[str, Any]]:
    """Normalize result(s) to JSON-serializable list of dicts."""
    seq = items if isinstance(items, list | tuple) else [items]
    out: list[dict[str, Any]] = []
    for r in seq:
        if hasattr(r, "__dict__"):
            out.append(dict(r.__dict__))     # dataclass / simple object
        elif isinstance(r, Mapping):
            out.append(dict(r))              # mapping-like
        else:
            out.append({"value": r})         # fallback
    return out


@app.get("/health")  # type: ignore[misc]
def health() -> dict[str, Any]:
    return {"ok": True}


@app.post("/ingest")  # type: ignore[misc]
def ingest_endpoint(
    space_keys: str | None = Query(
        None, description="Comma-separated space keys"
    ),
) -> dict[str, Any]:
    ingest_func = _resolve_ingest()
    keys: Sequence[str] | None = (
        [s.strip() for s in (space_keys or "").split(",") if s.strip()] or None
    )
    result = ingest_func(STORE, keys)
    return {"results": _to_jsonable_list(result)}


@app.get("/search")  # type: ignore[misc]
def search(
    q: str,
    k: int = 5,
    preferred_spaces: str = "",
    preferred_labels: str = "",
    preferred_authors: str = "",
) -> dict[str, Any]:
    profile = UserProfile(
        user_id="api",
        preferred_spaces=[
            s.strip() for s in preferred_spaces.split(",") if s.strip()
        ] or None,
        preferred_labels=[
            s.strip() for s in preferred_labels.split(",") if s.strip()
        ] or None,
        preferred_authors=[
            s.strip() for s in preferred_authors.split(",") if s.strip()
        ] or None,
    )
    results = RAG_PIPE.retrieve(q, k=k, user_profile=profile.to_dict())

    out: list[dict[str, Any]] = []
    for r in results:
        score = getattr(r, "score", None)
        meta = getattr(r, "meta", {})
        if isinstance(meta, Mapping):
            out.append({"score": score, **meta})
        else:
            out.append({"score": score, "meta": meta})
    return {"results": out}

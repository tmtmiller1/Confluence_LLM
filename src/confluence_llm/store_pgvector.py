# src/confluence_llm/store_pgvector.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from typing import Any, Protocol, cast

# ---- Optional deps (guarded imports) ---------------------------------------
psycopg: Any | None = None
sql: Any | None = None
SentenceTransformer: Any | None = None

try:
    import psycopg as _psycopg  # psycopg3
    from psycopg import sql as _sql

    psycopg = _psycopg
    sql = _sql
except Exception as _imp_err:
    # Explicitly assign to avoid bare `pass` (Bandit B110)
    psycopg = None
    sql = None

try:
    from sentence_transformers import SentenceTransformer as _ST

    SentenceTransformer = _ST
except Exception as _st_err:
    SentenceTransformer = None


# ---- Types -----------------------------------------------------------------
class Encoder(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = ...,
        show_progress_bar: bool = ...,
    ) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class ScoredDoc:
    score: float
    meta: dict[str, Any]


# ---- Helpers ---------------------------------------------------------------
def _fallback_encode(texts: Sequence[str], dim: int) -> list[list[float]]:
    """
    Deterministic no-deps embedding: hash -> vector in [0,1], then l2-normalize.
    For testing or environments without sentence-transformers.
    """
    import hashlib

    out: list[list[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        raw = (h * ((dim // len(h)) + 1))[:dim]
        vec = [b / 255.0 for b in raw]
        norm = (sum(x * x for x in vec) ** 0.5) or 1.0
        out.append([x / norm for x in vec])
    return out


def _env_str(name: str, default: str) -> str:
    """Get an env var as a concrete str (never Optional)."""
    v = os.getenv(name)
    return v if v is not None else default


def _resolve_dsn() -> str:
    """Return a concrete DSN string; never Optional."""
    v = os.getenv("DATABASE_URL")
    if v:
        return v
    v = os.getenv("PG_DSN")
    if v:
        return v
    return "postgresql://postgres:postgres@localhost:5432/postgres"


class _SBERTEncoder:
    def __init__(self, model_name: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not available")
        self._model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        arr = self._model.encode(
            list(texts),
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )
        return cast("list[list[float]]", arr.tolist())


def _get_encoder(model_name: str, dim: int) -> Encoder:
    if SentenceTransformer is not None:
        return _SBERTEncoder(model_name)

    class _Fallback:
        def encode(
            self,
            texts: Sequence[str],
            *,
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
        ) -> list[list[float]]:
            return _fallback_encode(texts, dim)

    return _Fallback()


# ---- Store -----------------------------------------------------------------
class PGVectorStore:
    """
    pgvector-backed store (psycopg3). Keeps a `.docs` mirror for API parity
    with in-memory store, but truth lives in Postgres.
    """

    def __init__(
        self,
        model_name: str | None = None,
        dim: int | None = None,
        table: str | None = None,
    ) -> None:
        # Avoid assigning Optional[str] to str-typed attrs: use _env_str
        self.model_name: str = (
            model_name if model_name is not None
            else _env_str("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        dim_env = _env_str("EMBED_DIM", "384")
        self.dim: int = int(dim if dim is not None else int(dim_env))
        self.table: str = table if table is not None else _env_str("PGVECTOR_TABLE", "rag_docs")

        self._encoder: Encoder = _get_encoder(self.model_name, self.dim)
        self.docs: list[dict[str, Any]] = []  # mirror (last added batch)

        # Concrete string; no Optional[str] assignment
        self._dsn: str = _resolve_dsn()

        if psycopg is None or sql is None:
            raise RuntimeError("psycopg3 is required for PGVectorStore")

        self._ensure_schema()

    # -- utils ---------------------------------------------------------------
    def _ensure_schema(self) -> None:
        """Create table and index if missing."""
        if psycopg is None or sql is None:
            raise RuntimeError("psycopg3/sql not available")

        create_table = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {tbl} (
                chunk_id TEXT PRIMARY KEY,
                text     TEXT NOT NULL,
                meta     JSONB NOT NULL,
                embedding VECTOR(%s) NOT NULL
            )
            """
        ).format(tbl=sql.Identifier(self.table))

        create_index = sql.SQL(
            "CREATE INDEX IF NOT EXISTS {idx} "
            "ON {tbl} USING ivfflat (embedding vector_cosine_ops)"
        ).format(
            idx=sql.Identifier(f"{self.table}_embedding_idx"),
            tbl=sql.Identifier(self.table),
        )

        # Combine contexts (ruff SIM117)
        with psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(create_table, (self.dim,))
            cur.execute(create_index)
            conn.commit()

    # -- API -----------------------------------------------------------------
    def add(self, items: Sequence[dict[str, Any]]) -> int:
        """
        Insert items (each with keys: chunk_id, text, meta).
        Embeddings are computed here.
        """
        if not items:
            return 0
        if psycopg is None or sql is None:
            raise RuntimeError("psycopg3/sql not available")

        texts = [str(it.get("text", "")) for it in items]
        vecs = self._encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        insert_stmt = sql.SQL(
            "INSERT INTO {tbl}(chunk_id, text, meta, embedding) "
            "VALUES (%s, %s, %s, %s::vector) "
            "ON CONFLICT (chunk_id) DO NOTHING"
        ).format(tbl=sql.Identifier(self.table))

        with psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            for it, vec in zip(items, vecs, strict=False):
                chunk_id = str(it.get("chunk_id", ""))
                meta = cast(dict[str, Any], it.get("meta", {}))
                vec_lit = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
                cur.execute(
                    insert_stmt,
                    (chunk_id, str(it.get("text", "")), json.dumps(meta), vec_lit),
                )
            conn.commit()

        self.docs.extend(items)
        return len(items)

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        user_profile: dict[str, Any] | None = None,  # kept for parity
    ) -> list[ScoredDoc]:
        """Cosine similarity search via pgvector. Returns top-k ScoredDoc."""
        if psycopg is None or sql is None:
            raise RuntimeError("psycopg3/sql not available")

        q_vec = self._encoder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        vec_lit = "[" + ",".join(f"{x:.6f}" for x in q_vec) + "]"

        select_stmt = sql.SQL(
            "SELECT chunk_id, text, meta, "
            "       1 - (embedding <=> %s::vector) AS score "
            "FROM {tbl} "
            "ORDER BY embedding <=> %s::vector "
            "LIMIT %s"
        ).format(tbl=sql.Identifier(self.table))

        results: list[ScoredDoc] = []
        with psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(select_stmt, (vec_lit, vec_lit, int(k)))
            for chunk_id, text, meta, score in cur.fetchall():
                meta_obj = cast(dict[str, Any], meta) if isinstance(meta, dict) else {}
                meta_out = dict(meta_obj)
                meta_out["chunk_id"] = chunk_id
                meta_out["text"] = text
                results.append(ScoredDoc(score=float(score), meta=meta_out))

        return results

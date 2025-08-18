from __future__ import annotations

import collections
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import os
from typing import Any


# -------- Preference-aware rerank (shared) --------
def _recency_decay(ts: float | int | None, half_life_days: float) -> float:
    """Return a multiplier in [0,1] that decays by half each *half_life_days*.
    ts is epoch seconds. If None, return 1.0.
    """
    if not ts or half_life_days <= 0:
        return 1.0
    import time

    age_days = max(0.0, (time.time() - float(ts)) / 86400.0)
    return float(0.5 ** (age_days / float(half_life_days)))


def _apply_post_scores(
    docs: list[dict[str, Any]],
    scores: list[float],
    user_profile: dict[str, Any] | None,
) -> list[float]:
    """Apply recency time-decay and negative de-boosts based on user_profile."""
    if not user_profile:
        return scores
    neg_spaces = {s.lower() for s in (user_profile.get("negative_spaces") or [])}
    neg_labels = {s.lower() for s in (user_profile.get("negative_labels") or [])}
    neg_authors = {s.lower() for s in (user_profile.get("negative_authors") or [])}
    half_life = float(user_profile.get("recency_half_life_days") or 0.0)

    out: list[float] = []
    for i, s in enumerate(scores):
        d = docs[i]
        # recency decay
        if half_life > 0:
            s *= _recency_decay(d.get("updated_ts"), half_life)
        # negative de-boosts
        if neg_spaces and str(d.get("space", "")).lower() in neg_spaces:
            s *= 0.7
        doc_labels = [str(x).lower() for x in (d.get("labels") or [])]
        if neg_labels and any(lbl in neg_labels for lbl in doc_labels):
            s *= 0.8
        if neg_authors and str(d.get("author", "")).lower() in neg_authors:
            s *= 0.85
        out.append(float(s))
    return out


def _tokenize(text: str) -> list[str]:
    import re as _re

    return _re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _apply_preference_boosts(
    docs: list[dict[str, Any]],
    scores: list[float],
    user_profile: dict[str, Any] | None,
) -> list[float]:
    if not user_profile:
        return scores
    spaces = {s.lower() for s in (user_profile.get("preferred_spaces") or [])}
    labels = {lbl.lower() for lbl in (user_profile.get("preferred_labels") or [])}
    authors = {a.lower() for a in (user_profile.get("preferred_authors") or [])}

    out: list[float] = []
    for i, s in enumerate(scores):
        b = 1.0
        if spaces and str(docs[i].get("space", "")).lower() in spaces:
            b *= 1.2
        doc_labels = [str(x).lower() for x in (docs[i].get("labels") or [])]
        if labels and any(lbl in labels for lbl in doc_labels):
            b *= 1.1
        if authors and str(docs[i].get("author", "")).lower() in authors:
            b *= 1.05
        out.append(float(s * b))
    return out


def _acl_allows(doc: dict[str, Any], user_profile: dict[str, Any]) -> bool:
    """Allow if not restricted, or if user/group matches profile (when provided).
    Profile keys: user_id, groups (list of group names). If none provided, allow only
    unrestricted docs.
    """
    if not doc.get("restricted"):
        return True
    user_id = (user_profile or {}).get("user_id")
    groups = {g.lower() for g in (user_profile or {}).get("groups", [])}
    allowed_users = set(doc.get("allow_users") or [])
    allowed_groups = {g.lower() for g in (doc.get("allow_groups") or [])}
    if user_id and user_id in allowed_users:
        return True
    return bool(groups and (groups & allowed_groups))


@dataclass
class ScoredDoc:
    score: float
    meta: dict[str, Any]


# -------- BM25 fallback --------
class BM25Store:
    def put(self, doc: dict[str, Any]) -> None:
        """Insert a single document (adapter for ingest.Store protocol)."""
        self.add([doc])

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []
        self.tfs: list[collections.Counter[str]] = []
        self.df: collections.Counter[str] = collections.Counter()
        self.N: int = 0

    def add(self, items: Sequence[dict[str, Any]]) -> int:
        if not items:
            return 0
        for it in items:
            text = str(it.get("text", ""))
            tf: collections.Counter[str] = collections.Counter(_tokenize(text))
            self.docs.append(it)
            self.tfs.append(tf)
        # recompute df
        self.df = collections.Counter()
        for tf in self.tfs:
            self.df.update(tf.keys())
        self.N = len(self.docs)
        return len(items)

    def _bm25(
        self,
        q: list[str],
        tf: collections.Counter[str],
        avgdl: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        dl = sum(tf.values()) or 1
        score = 0.0
        for term in q:
            f = tf.get(term, 0)
            if f == 0:
                continue
            n = self.df.get(term, 0) or 1
            idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        return float(score)

    def search(
        self,
        query: str,
        k: int = 5,
        user_profile: dict[str, Any] | None = None,
    ) -> list[ScoredDoc]:
        if not self.docs:
            return []
        q = _tokenize(query)
        avgdl = sum(sum(tf.values()) for tf in self.tfs) / max(1, self.N)
        scores = [self._bm25(q, tf, avgdl) for tf in self.tfs]
        scores = _apply_preference_boosts(self.docs, scores, user_profile)
        scores = _apply_post_scores(self.docs, scores, user_profile)
        # ACL post-filter then top-k
        ordering = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        filtered = [i for i in ordering if _acl_allows(self.docs[i], user_profile or {})]
        idx = filtered[:k]
        return [
            ScoredDoc(
                score=float(scores[i]),
                meta={kk: vv for kk, vv in self.docs[i].items() if kk != "text"},
            )
            for i in idx
        ]


# -------- Embedding + FAISS --------
class FAISSStore:
    def put(self, doc: dict[str, Any]) -> None:
        """Insert a single document (adapter for ingest.Store protocol)."""
        self.add([doc])

    def __init__(self, model_name: str | None = None) -> None:
        self.docs: list[dict[str, Any]] = []
        self.vecs: Any = None  # numpy.ndarray | None
        # Typed as Any to keep Pylance clean if faiss stubs are absent
        self.index: Any = None
        self.model_name = model_name or os.getenv(
            "EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self._model: Any = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def _encode(self, texts: Sequence[str]) -> Any:
        self._ensure_model()
        import numpy as np

        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def add(self, items: Sequence[dict[str, Any]]) -> int:
        import faiss
        import numpy as np

        texts = [str(it.get("text", "")) for it in items]
        if not texts:
            return 0
        new_vecs = self._encode(texts)
        if self.vecs is None:
            self.vecs = new_vecs
            dim = new_vecs.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.vecs)
        else:
            self.vecs = np.vstack([self.vecs, new_vecs])
            self.index.add(new_vecs)
        # keep docs
        for it in items:
            self.docs.append(it)
        return len(items)

    def search(
        self,
        query: str,
        k: int = 5,
        user_profile: dict[str, Any] | None = None,
    ) -> list[ScoredDoc]:
        if not self.docs or self.index is None:
            return []
        qv = self._encode([query])
        # oversample before ACL filter
        oversample = min(k * 10, len(self.docs))
        distances, indices = self.index.search(qv, oversample)  # (1, n)
        # initial scores aligned to oversample list
        scores = [float(distances[0][i]) for i in range(indices.shape[1])]
        idxs = [int(indices[0][i]) for i in range(indices.shape[1])]
        docs = [self.docs[i] for i in idxs]
        # preference/post boosts
        scores = _apply_preference_boosts(docs, scores, user_profile)
        scores = _apply_post_scores(docs, scores, user_profile)
        # ACL filter, then take top-k
        ordering = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        filtered_doc_idxs = [
            idxs[j] for j in ordering if _acl_allows(docs[j], user_profile or {})
        ]
        top = filtered_doc_idxs[:k]
        out: list[ScoredDoc] = []
        for pos, doc_idx in enumerate(top):
            j = ordering[pos]
            out.append(
                ScoredDoc(
                    score=float(scores[j]),
                    meta={kk: vv for kk, vv in self.docs[doc_idx].items() if kk != "text"},
                )
            )
        return out


# -------- pgvector (PostgreSQL) --------
class PgvectorStore:
    def put(self, doc: dict[str, Any]) -> None:
        """Insert a single document (adapter for ingest.Store protocol)."""
        self.add([doc])

    """pgvector-backed store (psycopg3). Also exposes a `.docs` attribute for API parity."""

    def __init__(
        self,
        model_name: str | None = None,
        dim: int | None = None,
        table: str | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.dim = int(dim or int(os.getenv("EMBED_DIM", "384")))
        self.table = table or os.getenv("PGVECTOR_TABLE", "rag_docs")
        self._model: Any = None
        self._conn: Any = None  # psycopg.Connection | None
        self._setup_done: bool = False
        # kept for compatibility (last added docs)
        self.docs: list[dict[str, Any]] = []

    # --- internals ---------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def _ensure_conn(self) -> None:
        if self._conn is None:
            import psycopg

            dsn = (
                os.getenv("DATABASE_URL")
                or os.getenv("PG_DSN")
                or "postgresql://postgres:postgres@localhost:5432/postgres"
            )
            self._conn = psycopg.connect(dsn, autocommit=True)
        if not self._setup_done:
            from psycopg import sql

            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {tbl} (
                            id bigserial PRIMARY KEY,
                            chunk_id text,
                            text text,
                            meta jsonb,
                            embedding vector({dim})
                        );
                        """
                    ).format(
                        tbl=sql.Identifier(self.table),
                        dim=sql.Literal(self.dim),
                    )
                )
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {idx} "
                        "ON {tbl} USING ivfflat (embedding vector_cosine_ops);"
                    ).format(
                        idx=sql.Identifier(f"idx_{self.table}_embedding"),
                        tbl=sql.Identifier(self.table),
                    )
                )
            self._setup_done = True

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure_model()
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Return plain Python lists for psycopg param binding
        return [list(map(float, v)) for v in vecs]

    # --- API ----------------------------------------------------------------

    def add(self, items: Sequence[dict[str, Any]]) -> int:
        if not items:
            return 0
        self._ensure_conn()
        from psycopg import sql

        texts = [str(it.get("text", "")) for it in items]
        vecs = self._encode(texts)
        with self._conn.cursor() as cur:
            for it, v in zip(items, vecs, strict=False):
                meta = {k: val for k, val in it.items() if k not in ("text",)}
                chunk_id = str(it.get("chunk_id", ""))
                # Build vector literal safely and use composed SQL for type-safe query object
                vec_lit = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {tbl}(chunk_id, text, meta, embedding) "
                        "VALUES (%s, %s, %s, %s::vector)"
                    ).format(tbl=sql.Identifier(self.table)),
                    (chunk_id, str(it.get("text", "")), json.dumps(meta), vec_lit),
                )
        # Keep a shallow copy for compatibility
        self.docs.extend(items)
        return len(items)

    def search(
        self,
        query: str,
        k: int = 5,
        user_profile: dict[str, Any] | None = None,
    ) -> list[ScoredDoc]:
        self._ensure_conn()
        from psycopg import sql

        qv = self._encode([query])[0]
        vec_lit = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"
        oversample = max(1, k * 10)
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT id, chunk_id, text, meta,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {tbl}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                ).format(tbl=sql.Identifier(self.table)),
                (vec_lit, vec_lit, oversample),
            )
            rows = cur.fetchall()
        # shape rows: (id, chunk_id, text, meta, score)
        docs = [{**(row[3] or {}), "text": row[2]} for row in rows]
        scores = [float(row[4]) for row in rows]
        scores = _apply_preference_boosts(docs, scores, user_profile)
        scores = _apply_post_scores(docs, scores, user_profile)
        ordering = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        filtered = [i for i in ordering if _acl_allows(docs[i], user_profile or {})][:k]
        return [
            ScoredDoc(
                score=float(scores[i]),
                meta={kk: vv for kk, vv in docs[i].items() if kk != "text"},
            )
            for i in filtered
        ]


# -------- Backend selector --------
def get_store() -> object:
    backend = os.getenv("STORE", "bm25").lower()
    if backend == "faiss":
        return FAISSStore()
    if backend == "pgvector":
        return PgvectorStore()
    return BM25Store()

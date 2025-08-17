from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
import os
import re
from typing import Any, Protocol, cast

from .attachments import sniff_text
from .confluence_client import ConfluenceClient
from .settings import (
    CONFLUENCE_API_TOKEN as _CFG_TOKEN,
    CONFLUENCE_BASE_URL as _CFG_BASE,
    CONFLUENCE_EMAIL as _CFG_EMAIL,
)

logger = logging.getLogger(__name__)

# ---- Types / interfaces -----------------------------------------------------


class Store(Protocol):
    """Minimal store interface: anything with a put() method works."""

    def put(self, doc: dict[str, Any]) -> None:
        ...


@dataclass
class IngestResult:
    space_key: str
    pages: int
    attachments: int


# ---- Tiny helpers (inline replacements for text_utils) ----------------------


_HEADING_RX = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)


def html_to_markdown(html: str) -> str:
    """
    Naive HTML→Markdown:
      - Convert <hN>...</hN> to N hashes, keep text.
      - Strip remaining tags.
      - Collapse extra whitespace.
    Replace with markdownify/html2text later if desired.
    """
    def _h_sub(match: re.Match[str]) -> str:
        level = int(match.group(1))
        text = match.group(2).strip()
        return f"{'#' * level} {text}"

    # Convert headings first
    html = re.sub(
        r"<h([1-6])>(.*?)</h\1>",
        _h_sub,
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", html)
    # Normalize whitespace
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def heading_chunks(md: str) -> list[str]:
    """Split Markdown into chunks by headings; keep each heading with its body."""
    if not md.strip():
        return []
    chunks: list[str] = []
    last = 0
    for m in _HEADING_RX.finditer(md):
        start = m.start()
        if start > last:
            block = md[last:start].strip()
            if block:
                chunks.append(block)
        last = start
    tail = md[last:].strip()
    if tail:
        chunks.append(tail)
    return chunks


# ---- Client wiring (settings with env fallbacks) ----------------------------


_BASE = os.getenv("CONFLUENCE_BASE_URL", _CFG_BASE or "")
_EMAIL = os.getenv("CONFLUENCE_EMAIL", _CFG_EMAIL or "")
_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", _CFG_TOKEN or "")

# Construct a client instance; use Any to allow optional methods that may
# be added later (e.g., get_page_content) without tripping Pylance.
_client: Any = ConfluenceClient(_BASE, _EMAIL, _TOKEN)


SPACE_KEYS: list[str] = [
    s.strip() for s in os.getenv("SPACE_KEYS", "").split(",") if s.strip()
]


# ---- Core ingestion ---------------------------------------------------------


def _page_meta(page: dict[str, Any]) -> dict[str, Any]:
    space = cast(dict[str, Any], page.get("space", {}))
    version = cast(dict[str, Any], page.get("version", {}))
    last_when = cast(str, version.get("when", ""))
    last_by = cast(dict[str, Any], version.get("by", {})).get("displayName", "")
    return {
        "id": page.get("id"),
        "title": page.get("title"),
        "space_key": space.get("key"),
        "space_name": space.get("name"),
        "version": version.get("number"),
        "last_updated": last_when,
        "last_updated_by": last_by,
        "url": page.get("_links", {}).get("tinyui")
        or page.get("_links", {}).get("webui"),
    }


def ingest_space(store: Store, space_key: str, *, max_pages: int | None = None) -> IngestResult:
    pages_seen = 0
    attach_seen = 0

    cql = f"space={space_key} and type=page"
    for page in _client.iter_all_pages(
        cql=cql,
        page_size=50,
        max_pages=max_pages or 200,
    ):
        meta = _page_meta(page)
        pid = str(page.get("id", ""))

        # Body → markdown → heading chunks
        try:
            get_page_content = _client.get_page_content
            html: str = get_page_content(pid) or ""
        except Exception as e:
            logger.debug("get_page_content failed for page %s: %s", pid, e)
            html = ""
        md = html_to_markdown(html)
        chunks = heading_chunks(md)

        for i, chunk in enumerate(chunks):
            store.put(
                {
                    "kind": "page",
                    "page_id": pid,
                    "chunk": i,
                    "text": chunk,
                    "meta": meta,
                }
            )

        # Attachments
        try:
            get_attachments = _client.get_attachments
            atts = get_attachments(pid) or []
        except Exception as e:
            logger.debug("get_attachments failed for page %s: %s", pid, e)
            atts = []

        for att in atts:
            name = str(att.get("title") or att.get("filename") or "")
            data: bytes | None = None
            try:
                download_attachment = _client.download_attachment
                data = download_attachment(att)
            except Exception as e:
                logger.warning(
                    "download_attachment failed for page %s attachment %s: %s",
                    pid,
                    att.get("id"),
                    e,
                )

            if not data:
                # Skip this attachment if download failed
                continue

            text = sniff_text(name, data)
            if text:
                store.put(
                    {
                        "kind": "attachment",
                        "page_id": pid,
                        "filename": name,
                        "text": text,
                        "meta": meta,
                    }
                )
                attach_seen += 1

        pages_seen += 1

    return IngestResult(space_key=space_key, pages=pages_seen, attachments=attach_seen)


def ingest(store: Store, space_keys: Sequence[str] | None = None) -> list[IngestResult]:
    if space_keys:
        keys = list(space_keys)
    elif SPACE_KEYS:
        keys = SPACE_KEYS
    else:
        try:
            list_spaces = _client.list_spaces
            spaces = list_spaces() or []
            keys = [
                str(s.get("key", "")).strip()
                for s in spaces
                if str(s.get("key", "")).strip()
            ]
        except Exception as e:
            logger.warning("list_spaces failed: %s", e)
            keys = []

    return [ingest_space(store, k) for k in keys]

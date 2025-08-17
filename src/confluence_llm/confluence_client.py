# src/confluence_llm/confluence_client.py
from __future__ import annotations

from collections.abc import Iterator
import logging
from typing import Any, cast

import requests

logger = logging.getLogger(__name__)


class ConfluenceClient:
    """
    Minimal Confluence REST client for spaces and content search.

    - Uses a shared requests.Session with basic auth (email, API token).
    - Public methods return parsed JSON dicts or iterators of page dicts.
    """

    def __init__(self, base_url: str, email: str, api_token: str) -> None:
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self.base_url: str = base_url
        self.session = requests.Session()
        self.session.auth = (email, api_token)
        self.session.headers.update({"Accept": "application/json"})

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Perform a GET and return parsed JSON as a dict[str, Any].
        Raises requests.HTTPError for non-2xx responses.
        """
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()  # json() is Any; we promise a mapping here.
        return cast(dict[str, Any], data)

    def get_spaces(self, limit: int = 50, start: int = 0) -> dict[str, Any]:
        return self._get("/rest/api/space", {"limit": limit, "start": start})

    def get_pages(
        self,
        cql: str | None = None,
        limit: int = 50,
        start: int = 0,
        expand: str = "version,space,body.storage",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "start": start, "expand": expand}
        if cql:
            params["cql"] = cql
        return self._get("/rest/api/content/search", params)

    def iter_all_pages(
        self,
        cql: str | None = None,
        page_size: int = 50,
        max_pages: int = 200,
        expand: str = "version,space,body.storage",
    ) -> Iterator[dict[str, Any]]:
        """
        Iterate pages from /rest/api/content/search in batches.

        Args:
            cql: Confluence Query Language filter (optional).
            page_size: Items per request (Confluence 'limit').
            max_pages: Maximum number of batches to pull.
            expand: Fields to expand per page.

        Yields:
            Individual page dicts from the API "results" array.
        """
        start = 0
        fetched_batches = 0

        while fetched_batches < max_pages:
            batch = self.get_pages(
                cql=cql,
                limit=page_size,
                start=start,
                expand=expand,
            )
            results = cast(list[dict[str, Any]], batch.get("results", []))
            if not results:
                break

            # Ruff UP028: prefer 'yield from' over a loop that yields each item
            yield from results

            fetched_batches += 1
            start += page_size

            # Stop if the server indicates no further pages (defensive).
            size = cast(int, batch.get("size", len(results)))
            if size < page_size:
                break

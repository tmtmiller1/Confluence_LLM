from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def normalize_pages(items: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for it in items:
        space = (it.get("space") or {})
        version = (it.get("version") or {})
        by = (version.get("by") or {})
        rows.append({
            "id": it.get("id"),
            "title": it.get("title"),
            "space_key": space.get("key"),
            "space_name": space.get("name"),
            "version": version.get("number"),
            "updated_by": by.get("displayName") or by.get("username") or by.get("userKey"),
            "updated_when": version.get("when"),
            "status": it.get("status"),
            "type": it.get("type"),
        })
    df = pd.DataFrame(rows)
    return df

def page_counts_by_space(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["space_key", "count"])
    return (df.groupby("space_key", dropna=False)
              .size()
              .reset_index(name="count")
              .sort_values("count", ascending=False))

def page_counts_by_user(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["updated_by", "count"])
    return (df.groupby("updated_by", dropna=False)
              .size()
              .reset_index(name="count")
              .sort_values("count", ascending=False))

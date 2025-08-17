from __future__ import annotations

import pandas as pd


def summarize(df: pd.DataFrame) -> str:
    if df.empty:
        return "No pages found."
    total = len(df)
    spaces = df["space_key"].nunique(dropna=True)
    users = df["updated_by"].nunique(dropna=True)
    return f"Indexed {total} pages across {spaces} spaces, updated by {users} users."

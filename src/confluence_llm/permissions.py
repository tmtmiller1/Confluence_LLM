from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

"""Preference-aware retrieval profile (no hard ACL enforcement)."""

@dataclass
class UserProfile:
    user_id: str | None = None
    preferred_spaces: list[str] | None = None
    preferred_labels: list[str] | None = None
    preferred_authors: list[str] | None = None
    negative_spaces: list[str] | None = None
    negative_labels: list[str] | None = None
    negative_authors: list[str] | None = None
    groups: list[str] | None = None
    recency_half_life_days: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

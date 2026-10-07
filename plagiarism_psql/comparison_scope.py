"""Comparison pool rules — keep in sync with guard-main/lib/comparison-scope.ts."""

from __future__ import annotations

import re

GRADUATION_TYPES = frozenset({"coursework", "diploma"})


def normalize_category_slug(category: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ_-]", "_", (category or "").strip())
    slug = slug.strip("_") or "uncategorized"
    return slug or "uncategorized"


def comparison_categories(category: str) -> list[str]:
    slug = normalize_category_slug(category)
    if slug in GRADUATION_TYPES:
        return ["coursework", "diploma"]
    return [slug]

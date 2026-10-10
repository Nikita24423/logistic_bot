"""Comparison pool rules — keep in sync with guard-main/lib/comparison-scope.ts."""

from __future__ import annotations

import re
from typing import List, Optional

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

GRADUATION_TYPES = frozenset({"coursework", "diploma"})

# Qdrant MatchAny по огромному списку дорог, а список своих работ у студента
# ограничен практически.
MAX_EXCLUDED_DOCUMENT_IDS = 500


def normalize_category_slug(category: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ_-]", "_", (category or "").strip())
    return slug.strip("_") or "uncategorized"


def comparison_categories(category: str) -> list[str]:
    slug = normalize_category_slug(category)
    if slug in GRADUATION_TYPES:
        return ["coursework", "diploma"]
    return [slug]


def build_comparison_filter(
    document_id: Optional[int],
    filename: str,
    category: str,
    institution_id: Optional[str],
    user_id: Optional[str] = None,
    exclude_document_ids: Optional[List[int]] = None,
) -> Filter:
    """
    Сравниваем только с чужими работами того же вуза и типа.

    Исключаем: сам проверяемый документ, все работы того же user_id и явный
    список exclude_document_ids (для старых точек, у которых в payload нет
    user_id, — иначе студент совпадёт со своей же прошлогодней работой).
    """
    must_not: List[FieldCondition] = []
    if document_id is not None:
        must_not.append(
            FieldCondition(key="document_id", match=MatchValue(value=document_id))
        )
    else:
        must_not.append(
            FieldCondition(key="filename", match=MatchValue(value=filename))
        )

    uid = (user_id or "").strip()
    if uid:
        must_not.append(FieldCondition(key="user_id", match=MatchValue(value=uid)))

    extra_ids = [
        int(x)
        for x in (exclude_document_ids or [])
        if isinstance(x, (int, float)) and not isinstance(x, bool)
        and int(x) > 0 and int(x) != document_id
    ]
    extra_ids = sorted(set(extra_ids))[:MAX_EXCLUDED_DOCUMENT_IDS]
    if extra_ids:
        must_not.append(
            FieldCondition(key="document_id", match=MatchAny(any=extra_ids))
        )

    must: List[FieldCondition] = []
    inst = (institution_id or "").strip()
    if inst:
        must.append(
            FieldCondition(key="institution_id", match=MatchValue(value=inst))
        )

    pool = comparison_categories(category)
    if len(pool) == 1:
        must.append(FieldCondition(key="category", match=MatchValue(value=pool[0])))
    else:
        must.append(FieldCondition(key="category", match=MatchAny(any=pool)))

    return Filter(must=must, must_not=must_not)

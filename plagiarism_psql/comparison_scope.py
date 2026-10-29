"""Comparison pool rules — keep in sync with guard-main/lib/comparison-scope.ts.

Тип задания определяет, с какими работами сравнивается присланная работа.
По решению БГУИР каждый тип сравнивается ТОЛЬКО с работами того же типа:
лабораторная — с лабораторными, курсовой проект — с курсовыми проектами и т.д.
Работа и проект одного уровня (KuP/KuR, DiP/DiR) считаются РАЗНЫМИ типами и в
общий пул не объединяются.

Коды типов заданий (значение поля `category` в запросе):
    LR  — лабораторная работа
    PZ  — практическая работа
    IPR — индивидуальная практическая работа
    KuP — курсовой проект
    KuR — курсовая работа
    DiP — дипломный проект
    DiR — дипломная работа
"""

from __future__ import annotations

import re

# Известные коды типов заданий. Используются для документации и валидации;
# на логику сравнения не влияют — любой тип сравнивается сам с собой.
TASK_TYPE_CODES = frozenset({"LR", "PZ", "IPR", "KuP", "KuR", "DiP", "DiR"})

# Особый режим: SEO-работа сравнивается со всем корпусом вуза, без фильтра по типу.
SEO_CATEGORY = "seo"


def normalize_category_slug(category: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ_-]", "_", (category or "").strip())
    slug = slug.strip("_") or "uncategorized"
    return slug or "uncategorized"


def comparison_categories(category: str) -> list[str] | None:
    """
    Пул типов, с которыми сравнивается работа.

    Возвращает:
      - None  — без фильтра по типу (SEO: сравнение со всем корпусом вуза);
      - [тип] — сравнение только с работами того же типа (поведение по умолчанию
                для всех типов заданий: каждый тип сам с собой).
    """
    slug = normalize_category_slug(category)
    if slug == SEO_CATEGORY:
        return None
    return [slug]

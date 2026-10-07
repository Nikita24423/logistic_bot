"""
Отсечение служебных разделов студенческой работы перед анализом.

Зачем: титульный лист, содержание, список литературы и приложения не являются
авторским текстом и искажают оценку в обе стороны. Титульник одного вуза
идентичен у всех работ, а библиографическая запись по ГОСТу совпадает дословно
у любых двух студентов, цитирующих одну книгу, — лексический сигнал даёт на них
скор около 1.0, и каскад относит источник к тиру `exact`. Одновременно объёмное
приложение (листинги, таблицы данных) раздувает знаменатель `plagiarism_percent`
и занижает реальный процент по содержательной части.

ВАЖНО: отсечение выполняется до нарезки на чанки. Фильтровать готовые чанки
нельзя — они не выровнены по границам разделов: один чанк регулярно захватывает
конец титульника и начало содержания одновременно.

Тот же результат идёт и в поиск заимствований, и в индекс Qdrant, иначе в корпусе
остались бы разделы, которые мы перестали искать.

Разбор строчный и намеренно консервативный: при неуверенности раздел остаётся
в тексте. Ложно удалённый раздел занижает процент молча, а оставленный виден
в отчёте — поэтому вторая ошибка предпочтительнее.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

TITLE_PAGE = "title_page"
TABLE_OF_CONTENTS = "table_of_contents"
REFERENCES = "references"
APPENDIX = "appendix"

# Внутренние виды строк (в отчёт не попадают).
BODY = "body"
BODY_NUMBERED = "body_numbered"

# Титульник отсекается только если первый структурный заголовок найден в пределах
# этого числа символов от начала. Иначе «титульником» стало бы пол-работы.
TITLE_PAGE_MAX_CHARS = 4000

# Предохранитель от переусердствования: если после чистки осталось меньше этой
# доли исходного текста, разбор считается неудачным и текст возвращается целиком.
MIN_KEEP_RATIO = 0.3

# Строка длиннее этого — это абзац, а не заголовок раздела.
HEADING_MAX_LEN = 120

_TOC_HEAD = re.compile(r"^(содержание|оглавление|contents|table\s+of\s+contents)\b")
_INTRO_HEAD = re.compile(r"^(введение|introduction)\b")
_REFS_HEAD = re.compile(
    r"^(список\s+(использованн\w+\s+)?(источник\w+|литератур\w+)"
    r"|библиографическ\w+\s+(список|ссылк\w+)"
    r"|литература|references|bibliography|works\s+cited)\b"
)
_APPENDIX_HEAD = re.compile(r"^(приложение|приложения|appendix|appendices)\b")
_OTHER_HEAD = re.compile(
    r"^(глава|раздел|часть|заключение|выводы|chapter|conclusion|summary)\b"
)
# Нумерованный заголовок: «1 Анализ», «2.1. Постановка задачи».
_NUMBERED_HEAD = re.compile(r"^\d+(\.\d+)*[.)]?\s+\S")

# Отбивка строки содержания: «ВВЕДЕНИЕ ...... 3», «1.1 Задача … 5».
_DOT_LEADER = re.compile(r"(\.{2,}|…+|_{2,})\s*\d{1,4}\s*$")
# Внутри содержания номер страницы может стоять без отбивки: «ВВЕДЕНИЕ 3».
_TRAILING_PAGE = re.compile(r"\s\d{1,4}\s*$")


@dataclass
class StripReport:
    """Что было удалено — для диагностики и отчёта пользователю."""

    removed: List[str] = field(default_factory=list)
    chars_before: int = 0
    chars_after: int = 0
    fallback: bool = False  # разбор отвергнут предохранителем, текст не изменён

    def as_dict(self) -> dict:
        return {
            "removed_sections": list(self.removed),
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "fallback": self.fallback,
        }


def _heading_kind(line: str) -> Optional[str]:
    """Вид структурного заголовка в строке, либо None."""
    stripped = line.strip()
    if not stripped or len(stripped) > HEADING_MAX_LEN:
        return None
    probe = stripped.lower()
    if _TOC_HEAD.match(probe):
        return TABLE_OF_CONTENTS
    if _REFS_HEAD.match(probe):
        return REFERENCES
    if _APPENDIX_HEAD.match(probe):
        return APPENDIX
    if _INTRO_HEAD.match(probe) or _OTHER_HEAD.match(probe):
        return BODY
    if _NUMBERED_HEAD.match(probe):
        # Нумерация неоднозначна: «1 Анализ предметной области» — заголовок,
        # а «1. Маннинг, К. Введение...» — запись в списке источников. Различаем
        # по контексту при обходе, см. BODY_NUMBERED.
        return BODY_NUMBERED
    return None


def _looks_like_toc_entry(line: str, inside_toc: bool) -> bool:
    """
    Строка — элемент содержания, а не заголовок раздела.

    Отбивка точками однозначна в любом месте документа. Номер страницы без
    отбивки проверяется только внутри содержания: вне него «Глава 1» — заголовок.
    """
    stripped = line.strip()
    if _DOT_LEADER.search(stripped):
        return True
    return inside_toc and bool(_TRAILING_PAGE.search(stripped))


def strip_service_sections(text: str) -> tuple[str, StripReport]:
    """
    Удаляет титульный лист, содержание, список источников и приложения.

    Возвращает (очищенный текст, отчёт). При неудачном разборе текст возвращается
    без изменений с `fallback=True` в отчёте.
    """
    report = StripReport(chars_before=len(text or ""))
    if not text or not text.strip():
        report.chars_after = report.chars_before
        return text, report

    lines = text.splitlines()
    drop = [False] * len(lines)

    # Смещение начала каждой строки — нужно для границы титульного листа.
    offsets: List[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1

    kinds: List[Optional[str]] = []
    inside_toc = False
    inside_refs = False
    for line in lines:
        kind = _heading_kind(line)
        if kind is not None and _looks_like_toc_entry(line, inside_toc):
            # Элемент содержания («ПРИЛОЖЕНИЕ А ... 33») заголовком не считается.
            kind = None
        if kind == BODY_NUMBERED and inside_refs:
            kind = None
        if kind == TABLE_OF_CONTENTS:
            inside_toc = True
            inside_refs = False
        elif kind == REFERENCES:
            inside_refs = True
            inside_toc = False
        elif kind is not None:
            inside_toc = False
            inside_refs = False
        kinds.append(kind)

    # --- титульный лист: всё до первого структурного заголовка ---
    first_heading = next((i for i, k in enumerate(kinds) if k is not None), None)
    if first_heading is not None and first_heading > 0:
        if offsets[first_heading] <= TITLE_PAGE_MAX_CHARS:
            for i in range(first_heading):
                drop[i] = True
            if any(lines[i].strip() for i in range(first_heading)):
                report.removed.append(TITLE_PAGE)

    # --- содержание: от заголовка до следующего реального заголовка ---
    for i, kind in enumerate(kinds):
        if kind != TABLE_OF_CONTENTS:
            continue
        end = len(lines)
        for j in range(i + 1, len(lines)):
            if kinds[j] is not None:
                end = j
                break
        for j in range(i, end):
            drop[j] = True
        if TABLE_OF_CONTENTS not in report.removed:
            report.removed.append(TABLE_OF_CONTENTS)

    # --- список источников: до следующего заголовка или до конца ---
    for i, kind in enumerate(kinds):
        if kind != REFERENCES:
            continue
        end = len(lines)
        for j in range(i + 1, len(lines)):
            if kinds[j] is not None:
                end = j
                break
        for j in range(i, end):
            drop[j] = True
        if REFERENCES not in report.removed:
            report.removed.append(REFERENCES)

    # --- приложения: от первого заголовка до конца документа ---
    first_appendix = next((i for i, k in enumerate(kinds) if k == APPENDIX), None)
    if first_appendix is not None:
        for j in range(first_appendix, len(lines)):
            drop[j] = True
        report.removed.append(APPENDIX)

    cleaned = "\n".join(line for i, line in enumerate(lines) if not drop[i]).strip()

    # Предохранитель: подозрительно мало текста — разбор не удался.
    if not cleaned or len(cleaned) < max(200, int(report.chars_before * MIN_KEEP_RATIO)):
        report.fallback = True
        report.removed = []
        report.chars_after = report.chars_before
        return text, report

    report.chars_after = len(cleaned)
    return cleaned, report

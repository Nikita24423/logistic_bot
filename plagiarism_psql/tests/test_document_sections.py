from document_sections import (
    APPENDIX,
    REFERENCES,
    TABLE_OF_CONTENTS,
    TITLE_PAGE,
    strip_service_sections,
)

# Структура типовой курсовой: титульник, содержание с отбивкой, введение,
# список источников по ГОСТу, приложение с листингом.
COURSEWORK = """МИНИСТЕРСТВО ОБРАЗОВАНИЯ РЕСПУБЛИКИ БЕЛАРУСЬ
УЧРЕЖДЕНИЕ ОБРАЗОВАНИЯ "БЕЛОРУССКИЙ ГОСУДАРСТВЕННЫЙ УНИВЕРСИТЕТ
ИНФОРМАТИКИ И РАДИОЭЛЕКТРОНИКИ"
Кафедра программного обеспечения информационных технологий

КУРСОВАЯ РАБОТА
на тему "Разработка системы обнаружения заимствований"

Выполнил: студент группы 950501 Иванов И.И.
Минск 2026

СОДЕРЖАНИЕ

ВВЕДЕНИЕ ... 3
1 АНАЛИЗ ПРЕДМЕТНОЙ ОБЛАСТИ ... 5
1.1 Постановка задачи ... 5
ЗАКЛЮЧЕНИЕ ... 30
СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ ... 31
ПРИЛОЖЕНИЕ А ЛИСТИНГ ПРОГРАММЫ ... 33

ВВЕДЕНИЕ

Актуальность работы обусловлена ростом объёма студенческих работ и
необходимостью автоматизированной проверки их оригинальности. Целью работы
является разработка системы обнаружения текстовых заимствований на основе
векторного поиска по корпусу ранее сданных работ.

1 АНАЛИЗ ПРЕДМЕТНОЙ ОБЛАСТИ

Существующие системы проверки опираются на поиск точных совпадений, что
делает их уязвимыми к перефразированию исходного текста студентом.

СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ

1. Маннинг, К. Введение в информационный поиск / К. Маннинг, П. Рагхаван,
Х. Шютце. - М.: Вильямс, 2011. - 528 с.
2. Кормен, Т. Алгоритмы: построение и анализ / Т. Кормен, Ч. Лейзерсон,
Р. Ривест. - 3-е изд. - М.: Вильямс, 2013. - 1328 с.

ПРИЛОЖЕНИЕ А
ЛИСТИНГ ПРОГРАММЫ

def main():
    index = build_index(corpus)
    return index.search(query, top_k=10)
"""


def test_all_four_service_sections_are_removed():
    cleaned, report = strip_service_sections(COURSEWORK)
    assert set(report.removed) == {TITLE_PAGE, TABLE_OF_CONTENTS, REFERENCES, APPENDIX}
    assert not report.fallback


def test_author_text_survives():
    cleaned, _ = strip_service_sections(COURSEWORK)
    assert "Актуальность работы обусловлена" in cleaned
    assert "уязвимыми к перефразированию" in cleaned


def test_title_page_markers_are_gone():
    cleaned, _ = strip_service_sections(COURSEWORK)
    for marker in ("МИНИСТЕРСТВО", "Кафедра", "Иванов И.И.", "Минск 2026"):
        assert marker not in cleaned


def test_toc_entries_are_gone_but_real_headings_remain():
    cleaned, _ = strip_service_sections(COURSEWORK)
    assert "... 3" not in cleaned
    assert "ПРИЛОЖЕНИЕ А ЛИСТИНГ ПРОГРАММЫ" not in cleaned
    # Настоящий заголовок раздела (без номера страницы) сохраняется.
    assert "1 АНАЛИЗ ПРЕДМЕТНОЙ ОБЛАСТИ" in cleaned


def test_bibliography_entries_are_gone():
    cleaned, _ = strip_service_sections(COURSEWORK)
    for marker in ("Маннинг", "Кормен", "Вильямс", "528 с."):
        assert marker not in cleaned


def test_appendix_listing_is_gone():
    cleaned, _ = strip_service_sections(COURSEWORK)
    assert "build_index" not in cleaned
    assert "def main" not in cleaned


def test_plain_text_without_sections_is_untouched():
    text = (
        "Обычный текст без служебных разделов. " * 20
    ).strip()
    cleaned, report = strip_service_sections(text)
    assert cleaned == text
    assert report.removed == []


def test_lowercase_cross_reference_is_not_treated_as_appendix():
    # «см. приложение А» в середине абзаца не должно обрезать работу до конца.
    text = (
        "Схема алгоритма вынесена отдельно, см. приложение А для деталей. " * 8
        + "\nДалее рассматривается оценка вычислительной сложности метода. " * 8
    )
    cleaned, report = strip_service_sections(text)
    assert APPENDIX not in report.removed
    assert "оценка вычислительной сложности" in cleaned


def test_fallback_when_stripping_would_eat_the_document():
    # Работа, состоящая почти целиком из приложения: разбор отвергается,
    # текст возвращается целиком, чтобы не занизить процент молча.
    text = "ВВЕДЕНИЕ\n\nКороткое введение.\n\nПРИЛОЖЕНИЕ А\n" + "листинг кода\n" * 200
    cleaned, report = strip_service_sections(text)
    assert report.fallback is True
    assert cleaned == text
    assert report.removed == []


def test_empty_input_is_safe():
    cleaned, report = strip_service_sections("")
    assert cleaned == ""
    assert report.removed == []


def test_english_section_headings():
    text = (
        "Title Page Stuff University\n\nCONTENTS\n\nIntroduction ... 3\n\n"
        "INTRODUCTION\n\n"
        + "This thesis studies retrieval of near-duplicate student submissions. " * 6
        + "\n\nREFERENCES\n\n1. Manning C. Introduction to Information Retrieval.\n"
    )
    cleaned, report = strip_service_sections(text)
    assert TABLE_OF_CONTENTS in report.removed
    assert REFERENCES in report.removed
    assert "near-duplicate student submissions" in cleaned
    assert "Manning" not in cleaned

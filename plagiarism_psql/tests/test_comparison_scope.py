from qdrant_client.models import MatchAny, MatchValue

from comparison_scope import (
    MAX_EXCLUDED_DOCUMENT_IDS,
    build_comparison_filter,
    comparison_categories,
    normalize_category_slug,
)


def test_blank_category_gets_a_slug():
    assert normalize_category_slug("   ") == "uncategorized"
    assert normalize_category_slug("!!!") == "uncategorized"


def test_graduation_types_share_one_pool():
    assert comparison_categories("diploma") == ["coursework", "diploma"]
    assert comparison_categories("coursework") == ["coursework", "diploma"]
    assert comparison_categories("essay") == ["essay"]


def _keys(conditions):
    return [c.key for c in conditions]


def test_own_document_is_excluded_by_id():
    f = build_comparison_filter(42, "a.docx", "essay", "bsuir")
    assert "document_id" in _keys(f.must_not)


def test_without_id_the_filename_is_excluded():
    f = build_comparison_filter(None, "a.docx", "essay", "bsuir")
    assert "filename" in _keys(f.must_not)


def test_other_works_of_the_same_student_are_excluded():
    # Приватность и корректность: студент не должен совпадать со своей же
    # прошлой работой.
    f = build_comparison_filter(1, "a.docx", "essay", "bsuir", user_id="ivanov")
    assert "user_id" in _keys(f.must_not)


def test_blank_user_id_adds_no_condition():
    f = build_comparison_filter(1, "a.docx", "essay", "bsuir", user_id="   ")
    assert "user_id" not in _keys(f.must_not)


def test_sibling_ids_are_deduplicated_and_cleaned():
    f = build_comparison_filter(
        7, "a.docx", "essay", "bsuir", exclude_document_ids=[3, 3, 7, -1, 0, True, 5]
    )
    match = [c.match for c in f.must_not if isinstance(c.match, MatchAny)][0]
    # 7 — сам документ, -1 и 0 невалидны, True не номер документа.
    assert match.any == [3, 5]


def test_sibling_ids_are_capped():
    f = build_comparison_filter(
        1, "a.docx", "essay", "bsuir", exclude_document_ids=list(range(2, 2000))
    )
    match = [c.match for c in f.must_not if isinstance(c.match, MatchAny)][0]
    assert len(match.any) == MAX_EXCLUDED_DOCUMENT_IDS


def test_institution_and_category_are_required():
    f = build_comparison_filter(1, "a.docx", "essay", "bsuir")
    assert _keys(f.must) == ["institution_id", "category"]
    category = [c.match for c in f.must if c.key == "category"][0]
    assert isinstance(category, MatchValue)


def test_graduation_category_uses_match_any():
    f = build_comparison_filter(1, "a.docx", "diploma", "bsuir")
    category = [c.match for c in f.must if c.key == "category"][0]
    assert isinstance(category, MatchAny)
    assert sorted(category.any) == ["coursework", "diploma"]


def test_blank_institution_adds_no_condition():
    f = build_comparison_filter(1, "a.docx", "essay", "   ")
    assert "institution_id" not in _keys(f.must)

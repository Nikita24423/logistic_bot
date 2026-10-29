from comparison_scope import (
    TASK_TYPE_CODES,
    comparison_categories,
    normalize_category_slug,
)


def test_each_task_type_compares_with_itself_only():
    # Решение БГУИР: каждый тип сравнивается только с работами того же типа.
    for code in TASK_TYPE_CODES:
        assert comparison_categories(code) == [normalize_category_slug(code)]


def test_coursework_and_diploma_are_no_longer_pooled():
    # Раньше coursework и diploma были в общем пуле — теперь разделены.
    assert comparison_categories("KuR") == ["KuR"]
    assert comparison_categories("DiR") == ["DiR"]
    assert "DiR" not in comparison_categories("KuR")


def test_work_and_project_of_one_level_are_separate_types():
    assert comparison_categories("KuP") == ["KuP"]
    assert comparison_categories("KuR") == ["KuR"]
    assert comparison_categories("KuP") != comparison_categories("KuR")


def test_unknown_category_still_compares_with_itself():
    assert comparison_categories("lab") == ["lab"]


def test_blank_category_gets_a_slug():
    assert comparison_categories("   ") == ["uncategorized"]


def test_seo_has_no_category_filter():
    assert comparison_categories("seo") is None


def test_normalize_strips_punctuation():
    assert normalize_category_slug("Курсовая работа/проект") == "Курсовая_работа_проект"

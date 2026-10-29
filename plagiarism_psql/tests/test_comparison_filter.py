"""
Задача 5: различать повторную сдачу СВОЕЙ работы и использование ЧУЖОЙ.

Работы хранятся в корпусе бессрочно, поэтому через год работа, ранее сданная
другим студентом, должна находиться как источник. При этом собственная
доработанная работа студента не должна «совпадать сама с собой».

Механика — в фильтре сравнения: исключаются текущий документ и все работы того
же user_id (своя доработка), а чужие работы остаются в пуле (находятся).
"""
from qdrant_client.models import MatchValue

import worker


def _filter(**kw):
    w = object.__new__(worker.AntiPlagiarismWorker)
    return w._build_comparison_filter(**kw)


def _must_not_keys(f):
    return [c.key for c in f.must_not]


def test_own_other_works_are_excluded_by_user_id():
    # Своя доработанная работа (тот же user_id) не должна совпадать сама с собой.
    f = _filter(document_id=100, filename="v2.docx", category="KuR",
                institution_id="bsuir", user_id="ivanov")
    assert "user_id" in _must_not_keys(f)
    uid_cond = [c for c in f.must_not if c.key == "user_id"][0]
    assert uid_cond.match == MatchValue(value="ivanov")


def test_current_document_is_excluded():
    f = _filter(document_id=100, filename="v2.docx", category="KuR",
                institution_id="bsuir", user_id="ivanov")
    assert "document_id" in _must_not_keys(f)


def test_other_students_works_stay_in_the_pool():
    # В фильтре нет условия, исключающего чужих: работа другого студента того же
    # вуза и типа остаётся сравнимой (находится даже спустя год).
    f = _filter(document_id=100, filename="v2.docx", category="KuR",
                institution_id="bsuir", user_id="ivanov")
    must_keys = [c.key for c in f.must]
    assert "institution_id" in must_keys
    assert "category" in must_keys  # тот же вуз и тот же тип — чужие работы входят


def test_anonymous_user_still_excludes_current_doc():
    # Без user_id хотя бы текущий документ исключается (своя работа не совпадёт).
    f = _filter(document_id=100, filename="v2.docx", category="KuR",
                institution_id="bsuir", user_id=None)
    assert "document_id" in _must_not_keys(f)
    assert "user_id" not in _must_not_keys(f)


def test_legacy_sibling_ids_excluded_for_points_without_user_id():
    # Старые точки без user_id: СЭО передаёт список своих работ явным списком.
    f = _filter(document_id=100, filename="v2.docx", category="KuR",
                institution_id="bsuir", user_id="ivanov",
                exclude_document_ids=[7, 8, 9])
    # второе условие document_id (MatchAny) со списком своих прошлых работ
    doc_conds = [c for c in f.must_not if c.key == "document_id"]
    assert len(doc_conds) == 2

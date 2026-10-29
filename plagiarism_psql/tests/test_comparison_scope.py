from comparison_scope import comparison_categories


def test_graduation_types_share_a_pool():
    assert comparison_categories("coursework") == ["coursework", "diploma"]
    assert comparison_categories("diploma") == ["coursework", "diploma"]


def test_other_types_stay_in_their_own_pool():
    assert comparison_categories("lab") == ["lab"]


def test_seo_has_no_category_filter():
    assert comparison_categories("seo") is None

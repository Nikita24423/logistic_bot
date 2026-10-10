from cascade import (
    EXACT,
    PARAPHRASE,
    SEMANTIC,
    classify_match,
    content_tokens,
    paraphrase_similarity,
)


def test_content_tokens_drop_stopwords_and_short_words():
    tokens = content_tokens("И в работе рассмотрены методы поиска")
    assert "и" not in tokens
    assert "в" not in tokens
    assert "работе" in tokens
    assert "методы" in tokens


def test_paraphrase_similarity_is_high_for_shared_content_words():
    a = "методы поиска похожих документов в корпусе университета"
    b = "методы поиска похожих документов применяются в корпусе университета"
    assert paraphrase_similarity(a, b) > 0.15


def test_paraphrase_similarity_is_low_for_unrelated_texts():
    assert paraphrase_similarity("сортировка массива пузырьком", "рецепт борща со сметаной") < 0.15


def test_classify_match_cascade_exact_then_paraphrase_then_semantic():
    assert classify_match(0.9, 0.0, exact_threshold=0.6) == EXACT
    assert classify_match(0.2, 0.4, exact_threshold=0.6) == PARAPHRASE
    assert classify_match(0.1, 0.05, exact_threshold=0.6) == SEMANTIC

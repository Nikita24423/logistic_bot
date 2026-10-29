from text_normalize import normalize_text


def test_two_format_versions_collapse_to_same_text():
    # «одна работа» из PDF и из DOCX: разные пробелы/переносы -> один текст.
    pdf_like = "Цель  работы.\r\n\r\n\r\nАнализ сайтов   выполнен."
    docx_like = "Цель работы.\n\nАнализ сайтов выполнен."
    assert normalize_text(pdf_like) == normalize_text(docx_like)


def test_nonbreaking_and_narrow_spaces_become_regular():
    assert normalize_text("а б в г") == "а б в г"


def test_soft_hyphen_and_zero_width_removed():
    assert normalize_text("пере­нос​слова") == "переносслова"


def test_crlf_normalized():
    assert normalize_text("строка1\r\nстрока2\rстрока3") == "строка1\nстрока2\nстрока3"


def test_runs_of_blank_lines_collapse():
    assert normalize_text("а\n\n\n\n\nб") == "а\n\nб"


def test_trailing_spaces_stripped():
    assert normalize_text("слово   \nдругое") == "слово\nдругое"


def test_nfkc_folds_compatibility_forms():
    # римская цифра Ⅳ (U+2163) -> IV при NFKC
    assert normalize_text("глава Ⅳ") == "глава IV"


def test_empty_and_blank_safe():
    assert normalize_text("") == ""
    assert normalize_text("   \n\n  ") == ""


def test_meaningful_content_preserved():
    t = "Краулинговый бюджет это количество страниц, которые робот сканирует."
    assert normalize_text(t) == t


def test_determinism_same_input_same_output():
    t = "Любой   текст\r\n\r\nс разными пробелами."
    assert normalize_text(t) == normalize_text(t)

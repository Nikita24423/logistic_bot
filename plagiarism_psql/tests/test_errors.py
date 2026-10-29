import base64
import io

import pytest

from errors import (
    ERROR_CATALOG,
    AnalysisError,
    AnalysisErrorCode,
    http_status_for,
    message_for,
)

# Точные тексты из таблицы требований (задача 4).
EXPECTED = {
    AnalysisErrorCode.SERVICE_UNAVAILABLE: "Сервис проверки временно недоступен. Повторите попытку позже.",
    AnalysisErrorCode.TIMEOUT: "Превышено время ожидания ответа от системы.",
    AnalysisErrorCode.OVERLOADED: "Система перегружена. Работа ожидает проверки.",
    AnalysisErrorCode.UNSUPPORTED_FORMAT: "Формат файла не поддерживается. Загрузите PDF или DOCX.",
    AnalysisErrorCode.CORRUPT_FILE: "Не удалось обработать файл. Проверьте его содержимое.",
    AnalysisErrorCode.NO_TEXT: "В документе не обнаружен текст для проверки.",
    AnalysisErrorCode.EXTRACTION_FAILED: "Не удалось извлечь текст из документа.",
    AnalysisErrorCode.INTERNAL_ERROR: "Во время проверки произошла ошибка обработки.",
    AnalysisErrorCode.RESULT_DELIVERY_FAILED: "Не удалось передать результаты проверки в СЭО.",
    AnalysisErrorCode.AUTH_FAILED: "Не удалось подключиться к сервису проверки.",
}


def test_catalog_matches_requirements_table():
    assert len(ERROR_CATALOG) == 10
    for code, text in EXPECTED.items():
        assert message_for(code) == text


def test_error_carries_code_and_user_message():
    e = AnalysisError(AnalysisErrorCode.NO_TEXT, detail="internal note")
    assert e.code == "no_text"
    assert e.as_response() == {
        "error_code": "no_text",
        "detail": "В документе не обнаружен текст для проверки.",
    }
    # detail для логов не утекает пользователю
    assert "internal note" not in e.as_response()["detail"]


def test_http_status_mapping():
    assert http_status_for(AnalysisErrorCode.UNSUPPORTED_FORMAT) == 400
    assert http_status_for(AnalysisErrorCode.OVERLOADED) == 503
    assert http_status_for(AnalysisErrorCode.INTERNAL_ERROR) == 500
    assert http_status_for(AnalysisErrorCode.AUTH_FAILED) == 401


def test_unknown_code_falls_back():
    assert http_status_for("made_up") == 500
    assert message_for("made_up") == "Во время проверки произошла ошибка обработки."


# --- коды, которые поднимает extract_real_text (без ML-моделей) ---

def _worker():
    import worker
    return object.__new__(worker.AntiPlagiarismWorker)


def _payload(ext, data: bytes) -> str:
    return "FILE_BASE64|" + ext + "|" + base64.b64encode(data).decode()


def test_unsupported_format():
    with pytest.raises(AnalysisError) as ei:
        _worker().extract_real_text(_payload("exe", b"MZ"))
    assert ei.value.code == AnalysisErrorCode.UNSUPPORTED_FORMAT


def test_corrupt_base64():
    with pytest.raises(AnalysisError) as ei:
        _worker().extract_real_text("FILE_BASE64|pdf|не base64 %%%")
    assert ei.value.code == AnalysisErrorCode.CORRUPT_FILE


def test_corrupt_pdf_is_extraction_failed():
    with pytest.raises(AnalysisError) as ei:
        _worker().extract_real_text(_payload("pdf", "это не pdf".encode("utf-8")))
    assert ei.value.code == AnalysisErrorCode.EXTRACTION_FAILED


def test_empty_docx_is_no_text():
    docx = pytest.importorskip("docx")
    d = docx.Document()
    buf = io.BytesIO(); d.save(buf)
    with pytest.raises(AnalysisError) as ei:
        _worker().extract_real_text(_payload("docx", buf.getvalue()))
    assert ei.value.code == AnalysisErrorCode.NO_TEXT


def test_real_docx_extracts_text():
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Актуальность работы обусловлена ростом объёма работ.")
    buf = io.BytesIO(); d.save(buf)
    text = _worker().extract_real_text(_payload("docx", buf.getvalue()))
    assert "Актуальность работы" in text


def test_plain_text_passes_through():
    assert _worker().extract_real_text("обычный текст") == "обычный текст"

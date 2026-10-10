import base64
import io
import zipfile

import pytest

from document_text import FILE_PREFIX, extract_document_text


def _payload(ext: str, data: bytes) -> str:
    return f"{FILE_PREFIX}{ext}|{base64.b64encode(data).decode()}"


def test_plain_text_passes_through():
    text, was_file = extract_document_text("просто текст работы")
    assert text == "просто текст работы"
    assert was_file is False


def test_malformed_prefix_is_treated_as_text():
    raw = FILE_PREFIX + "нет третьей части"
    text, was_file = extract_document_text(raw)
    assert text == raw
    assert was_file is False


def test_unsupported_extension_is_rejected():
    with pytest.raises(ValueError, match="Неподдерживаемый тип файла"):
        extract_document_text(_payload("exe", b"MZ"))


def test_corrupt_base64_is_rejected():
    with pytest.raises(ValueError):
        extract_document_text(f"{FILE_PREFIX}pdf|не-base64-%%%")


def test_corrupt_pdf_raises_value_error_not_a_stub():
    # Заглушка вместо текста недопустима: раньше она индексировалась как текст
    # документа, и все последующие битые файлы «совпадали» с ней на 100%.
    with pytest.raises(ValueError, match="Не удалось извлечь текст"):
        extract_document_text(_payload("pdf", "это не pdf".encode("utf-8")))


def test_oversized_file_is_rejected_before_decoding(monkeypatch):
    monkeypatch.setenv("ANALYSIS_MAX_FILE_BYTES", "1024")
    with pytest.raises(ValueError, match="слишком большой"):
        extract_document_text(_payload("pdf", b"x" * 4096))


def test_zip_bomb_is_rejected(monkeypatch):
    monkeypatch.setenv("ANALYSIS_MAX_UNCOMPRESSED_BYTES", "1024")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        # Хорошо сжимаемый мусор: мал в архиве, огромен в распакованном виде.
        archive.writestr("word/document.xml", "A" * 200_000)
    with pytest.raises(ValueError, match="Распакованный размер"):
        extract_document_text(_payload("docx", buf.getvalue()))


def test_real_docx_is_extracted():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Актуальность работы обусловлена ростом объёма работ.")
    document.add_paragraph("Целью является разработка системы обнаружения.")
    buf = io.BytesIO()
    document.save(buf)

    text, was_file = extract_document_text(_payload("docx", buf.getvalue()))
    assert was_file is True
    assert "Актуальность работы" in text
    assert "Целью является разработка" in text


def test_extracted_text_length_is_capped(monkeypatch):
    docx = pytest.importorskip("docx")
    monkeypatch.setenv("ANALYSIS_MAX_TEXT_CHARS", "50")
    document = docx.Document()
    document.add_paragraph("слово " * 200)
    buf = io.BytesIO()
    document.save(buf)
    with pytest.raises(ValueError, match="слишком длинный"):
        extract_document_text(_payload("docx", buf.getvalue()))

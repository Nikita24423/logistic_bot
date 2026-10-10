"""
Извлечение текста работы из загруженного файла, с лимитами на размер.

Работа приходит в поле `content` как обычный текст либо как
`FILE_BASE64|<ext>|<base64>`. Содержимое загружается студентом, то есть это
недоверенные данные: лимиты здесь — единственное, что отделяет один большой
файл от исчерпания памяти процесса, в котором анализ сериализован локом
(эффективная конкурентность 1, см. AntiPlagiarismWorker._process_lock).
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import zipfile
from typing import Tuple

FILE_PREFIX = "FILE_BASE64|"


def _limit(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# Размер файла после base64-декодирования.
def max_file_bytes() -> int:
    return _limit("ANALYSIS_MAX_FILE_BYTES", 25 * 1024 * 1024)


# Суммарный несжатый размер членов docx-архива: защита от zip-бомбы, когда
# небольшой .docx распаковывается в гигабайты.
def max_uncompressed_bytes() -> int:
    return _limit("ANALYSIS_MAX_UNCOMPRESSED_BYTES", 200 * 1024 * 1024)


def max_pdf_pages() -> int:
    return _limit("ANALYSIS_MAX_PDF_PAGES", 500)


# Длина итогового текста: дальше он режется на чанки, и каждый чанк — это
# forward LM и два запроса в Qdrant.
def max_text_chars() -> int:
    return _limit("ANALYSIS_MAX_TEXT_CHARS", 2_000_000)


def _decode_payload(b64_data: str) -> bytes:
    limit = max_file_bytes()
    # Оценка размера до декодирования: 4 символа base64 -> 3 байта. Позволяет
    # отвергнуть гигантский вход, не выделяя под него память.
    approx = (len(b64_data) * 3) // 4
    if approx > limit:
        raise ValueError(
            f"Файл слишком большой: ~{approx // (1024 * 1024)} МиБ, "
            f"лимит {limit // (1024 * 1024)} МиБ"
        )
    try:
        file_bytes = base64.b64decode(b64_data, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Повреждённые данные файла (base64): {exc}") from exc
    if len(file_bytes) > limit:
        raise ValueError(
            f"Файл слишком большой: {len(file_bytes) // (1024 * 1024)} МиБ, "
            f"лимит {limit // (1024 * 1024)} МиБ"
        )
    return file_bytes


def _extract_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = reader.pages
    page_limit = max_pdf_pages()
    if len(pages) > page_limit:
        raise ValueError(
            f"В PDF слишком много страниц: {len(pages)}, лимит {page_limit}"
        )
    return "\n".join((page.extract_text() or "") for page in pages).strip()


def _extract_docx(file_bytes: bytes) -> str:
    stream = io.BytesIO(file_bytes)
    # Проверка несжатого объёма до разбора: python-docx читает архив целиком.
    try:
        with zipfile.ZipFile(stream) as archive:
            total = sum(info.file_size for info in archive.infolist())
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Повреждённый docx-архив: {exc}") from exc
    limit = max_uncompressed_bytes()
    if total > limit:
        raise ValueError(
            f"Распакованный размер документа {total // (1024 * 1024)} МиБ "
            f"превышает лимит {limit // (1024 * 1024)} МиБ"
        )
    stream.seek(0)

    import docx

    document = docx.Document(stream)
    return "\n".join(p.text for p in document.paragraphs).strip()


def extract_document_text(text_input: str) -> Tuple[str, bool]:
    """
    Возвращает (текст, был_ли_это_файл).

    Обычная строка возвращается как есть. При нераспознаваемом файле или
    превышении лимита поднимается ValueError — api_server отдаёт на него 400.
    Заглушку вместо текста возвращать нельзя: раньше она индексировалась как
    текст документа, и все последующие битые файлы «совпадали» с ней на 100%.
    """
    if not isinstance(text_input, str) or not text_input.startswith(FILE_PREFIX):
        return text_input, False

    parts = text_input.split("|", 2)
    if len(parts) != 3:
        return text_input, False
    _, ext, b64_data = parts
    ext = ext.strip().lower()

    file_bytes = _decode_payload(b64_data)
    try:
        if ext == "pdf":
            text = _extract_pdf(file_bytes)
        elif ext in ("docx", "doc"):
            text = _extract_docx(file_bytes)
        else:
            raise ValueError(f"Неподдерживаемый тип файла: {ext!r}")
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 — любой сбой парсера это 400, не 500
        raise ValueError(f"Не удалось извлечь текст из файла: {exc}") from exc

    char_limit = max_text_chars()
    if len(text) > char_limit:
        raise ValueError(
            f"Извлечённый текст слишком длинный: {len(text)} символов, "
            f"лимит {char_limit}"
        )
    return text, True

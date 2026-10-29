"""
Каталог ошибок анализа: код + краткое человеческое описание причины.

Задача 4 из требований: в СЭО должно возвращаться краткое описание причины,
а не просто код. Поэтому каждая ошибка несёт и машинный `code` (для логики СЭО),
и готовый текст `message` на русском (для показа пользователю).

Часть категорий из таблицы требований возникает НА СТОРОНЕ СЭО при обращении к
этому сервису (недоступность, таймаут, передача результата, авторизация) — сам
сервис их не поднимает, когда он недоступен. Они включены в каталог, чтобы обе
стороны использовали одинаковые формулировки (единый источник истины).
"""

from __future__ import annotations


class AnalysisErrorCode:
    # --- определяет сам сервис анализа ---
    UNSUPPORTED_FORMAT = "unsupported_format"
    CORRUPT_FILE = "corrupt_file"
    NO_TEXT = "no_text"
    EXTRACTION_FAILED = "extraction_failed"
    OVERLOADED = "overloaded"
    INTERNAL_ERROR = "internal_error"
    # --- показываются СЭО при обращении к сервису (сервис их не поднимает) ---
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout"
    RESULT_DELIVERY_FAILED = "result_delivery_failed"
    AUTH_FAILED = "auth_failed"


# Код -> (сообщение пользователю, HTTP-статус для ответа сервиса).
ERROR_CATALOG: dict[str, tuple[str, int]] = {
    AnalysisErrorCode.SERVICE_UNAVAILABLE: (
        "Сервис проверки временно недоступен. Повторите попытку позже.", 503),
    AnalysisErrorCode.TIMEOUT: (
        "Превышено время ожидания ответа от системы.", 504),
    AnalysisErrorCode.OVERLOADED: (
        "Система перегружена. Работа ожидает проверки.", 503),
    AnalysisErrorCode.UNSUPPORTED_FORMAT: (
        "Формат файла не поддерживается. Загрузите PDF или DOCX.", 400),
    AnalysisErrorCode.CORRUPT_FILE: (
        "Не удалось обработать файл. Проверьте его содержимое.", 400),
    AnalysisErrorCode.NO_TEXT: (
        "В документе не обнаружен текст для проверки.", 400),
    AnalysisErrorCode.EXTRACTION_FAILED: (
        "Не удалось извлечь текст из документа.", 400),
    AnalysisErrorCode.INTERNAL_ERROR: (
        "Во время проверки произошла ошибка обработки.", 500),
    AnalysisErrorCode.RESULT_DELIVERY_FAILED: (
        "Не удалось передать результаты проверки в СЭО.", 502),
    AnalysisErrorCode.AUTH_FAILED: (
        "Не удалось подключиться к сервису проверки.", 401),
}

# Подсказки по умолчанию на случай неизвестного кода.
_FALLBACK = ("Во время проверки произошла ошибка обработки.", 500)


def message_for(code: str) -> str:
    return ERROR_CATALOG.get(code, _FALLBACK)[0]


def http_status_for(code: str) -> int:
    return ERROR_CATALOG.get(code, _FALLBACK)[1]


class AnalysisError(Exception):
    """
    Ошибка анализа с кодом из каталога.

    `message` — краткое описание причины для пользователя (берётся из каталога,
    если не задано явно). `detail` — техническая подробность для логов, не для
    показа пользователю.
    """

    def __init__(self, code: str, detail: str | None = None, message: str | None = None):
        self.code = code
        self.message = message or message_for(code)
        self.detail = detail
        super().__init__(self.detail or self.message)

    @property
    def http_status(self) -> int:
        return http_status_for(self.code)

    def as_response(self) -> dict:
        """Тело ответа сервиса: код + человеческое сообщение."""
        return {"error_code": self.code, "detail": self.message}

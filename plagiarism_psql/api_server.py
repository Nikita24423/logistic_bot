import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from job_store import get_job_store, run_queued_job
from university_queues import DEFAULT_UNIVERSITY, UniversityQueues
from worker import AntiPlagiarismWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Длина работы в символах. Без лимита один большой документ (или base64 PDF)
# исчерпывает память процесса, в котором анализ сериализован локом.
MAX_CONTENT_CHARS = int(os.getenv("ANALYSIS_MAX_CONTENT_CHARS", str(40 * 1024 * 1024)))

# Пауза перед повторной загрузкой моделей, если инициализация провалилась.
WORKER_RETRY_DELAY_SEC = int(os.getenv("ANALYSIS_WORKER_RETRY_SEC", "60"))

# Сильные ссылки на фоновые таски: event loop держит лишь слабую ссылку, и
# задача без собственной ссылки может быть собрана сборщиком мусора до
# завершения — джоба тогда навсегда осталась бы в статусе queued.
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

_worker: Optional[AntiPlagiarismWorker] = None
_worker_error: Optional[str] = None
_worker_loading = False

# Очереди анализа: 1 университет = 1 очередь (создаются динамически).
_queues = UniversityQueues(concurrency=int(os.getenv("ANALYSIS_CONCURRENCY", "1")))


class AnalyzeRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_CHARS)
    filename: str = Field(default="document.txt", max_length=512)
    document_id: Optional[int] = None
    university_id: str = Field(default=DEFAULT_UNIVERSITY, max_length=128)
    category: str = Field(default="uncategorized", max_length=128)
    # Owner username — exclude this user's other works from comparison.
    user_id: Optional[str] = Field(default=None, max_length=128)
    # Legacy Qdrant points may lack user_id; Guard sends sibling doc ids to exclude.
    exclude_document_ids: list[int] = Field(default_factory=list)
    # Отсечение служебных разделов (титульник, содержание, список источников,
    # приложения) до анализа. None — взять значение из env STRIP_SERVICE_SECTIONS.
    strip_sections: Optional[bool] = None
    # Индексировать ли работу в корпус сравнения. None — значение из env
    # INDEX_ANALYZED_DOCUMENTS. false — «пробная» проверка, которая не попадёт
    # в пул сравнения последующих работ.
    index_document: Optional[bool] = None


class SemanticMatch(BaseModel):
    """Совпадение с работой-источником: семантическое (глубокий рерайт, max_score)
    и/или лексическое (дословное копирование, max_lexical_score).

    match_type — каскадная классификация источника (источник истины — cascade.py):
    exact (дословное) → paraphrase (перефраз) → semantic (глубокий рерайт)."""
    document_id: Optional[int] = None
    filename: Optional[str] = None
    matched_chunks: int = 0
    max_score: float = 0.0
    max_lexical_score: float = 0.0
    paraphrase_score: float = 0.0
    match_type: str = "semantic"
    sample: str = ""


class SectionsInfo(BaseModel):
    """Какие служебные разделы были вырезаны до анализа.

    fallback=true — разбор структуры отвергнут предохранителем (вырезание съело
    бы почти весь текст), работа проанализирована целиком."""
    removed_sections: list[str] = []
    chars_before: int = 0
    chars_after: int = 0
    fallback: bool = False


class AnalyzeResponse(BaseModel):
    plagiarism_percent: float
    ai_percent: float
    semantic_matches: list[SemanticMatch] = []
    by_type: dict[str, int] = {}
    sections: SectionsInfo = SectionsInfo()


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None
    document_id: Optional[int] = None
    filename: Optional[str] = None


def _verify_api_key(x_api_key: Optional[str]) -> None:
    expected = os.getenv("ANALYSIS_API_KEY", "").strip()
    if not expected:
        # Аутентификация выключена: сервис принимает работы и ПИШЕТ в корпус
        # от кого угодно, кто может достучаться до порта.
        return
    provided = (x_api_key or "").strip()
    # compare_digest вместо !=: обычное сравнение строк выходит на первом
    # различии и по времени ответа подсказывает подбирающему длину совпавшего
    # префикса.
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _require_worker() -> AntiPlagiarismWorker:
    if _worker is None:
        if _worker_error:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Worker failed to start: {_worker_error}. "
                    f"Retrying every {WORKER_RETRY_DELAY_SEC}s."
                ),
            )
        raise HTTPException(
            status_code=503,
            detail="Worker is still loading models. Wait and retry (first start can take 5–15 minutes on CPU).",
        )
    return _worker


def _to_analyze_response(result: dict[str, Any]) -> AnalyzeResponse:
    return AnalyzeResponse(
        plagiarism_percent=float(result.get("plagiarism_percent", 0.0)),
        ai_percent=float(result.get("ai_percent", 0.0)),
        semantic_matches=[SemanticMatch(**m) for m in result.get("semantic_matches", [])],
        by_type=result.get("by_type", {}),
        sections=SectionsInfo(**(result.get("sections") or {})),
    )


async def _init_worker() -> None:
    """Загружает модели, повторяя попытки: самая частая причина провала —
    Qdrant, который поднимается дольше, чем ждёт клиент, и проходит сама."""
    attempt = 0
    while _worker is None:
        attempt += 1
        await _init_worker_once(attempt)
        if _worker is None:
            await asyncio.sleep(WORKER_RETRY_DELAY_SEC)


async def _init_worker_once(attempt: int) -> None:
    global _worker, _worker_error, _worker_loading
    _worker_loading = True
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "university_docs")
        logger.info(
            "Загрузка ML-моделей, попытка %d (первый старт на CPU — несколько минут)...",
            attempt,
        )
        _worker = await run_in_threadpool(
            lambda: AntiPlagiarismWorker(
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                collection_name=qdrant_collection,
            )
        )
        logger.info("ML-воркер готов")
        _worker_error = None
    except Exception as exc:
        _worker_error = str(exc)
        logger.exception(
            "Попытка %d инициализации ML-воркера не удалась; повтор через %d с",
            attempt,
            WORKER_RETRY_DELAY_SEC,
        )
    finally:
        _worker_loading = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not os.getenv("ANALYSIS_API_KEY", "").strip():
        logger.warning(
            "ANALYSIS_API_KEY не задан: эндпоинты анализа открыты без "
            "аутентификации, и любой, кто достучится до порта, может писать "
            "в корпус сравнения. Задайте ключ в .env для production."
        )
    store = get_job_store()
    if not store.persistent:
        logger.warning(
            "Журнал джоб ведётся в памяти: после рестарта все job_id исчезнут "
            "и клиент получит 404 на валидный идентификатор."
        )
    # Джобы, прерванные прошлым рестартом, уже никто не выполнит.
    await store.recover_interrupted()

    task = _spawn(_init_worker())
    _queues.start()
    yield
    await _queues.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Antiplagiarism Analysis API",
    version="1.1.0",
    lifespan=lifespan,
)

_cors_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
if _cors_origins_raw:
    origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
async def health():
    if _worker is not None:
        return {"ok": True, "ready": True}
    if _worker_error:
        return {"ok": False, "ready": False, "error": _worker_error}
    return {"ok": True, "ready": False, "loading": _worker_loading}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """Синхронный анализ (совместимость). Предпочтительно POST /v1/jobs + poll."""
    _verify_api_key(x_api_key)
    worker = _require_worker()

    try:
        result = await _queues.submit(
            body.university_id,
            lambda: worker.process_text(
                body.content,
                body.filename,
                False,
                body.document_id,
                category=body.category,
                institution_id=body.university_id,
                user_id=body.user_id,
                exclude_document_ids=body.exclude_document_ids,
                strip_sections=body.strip_sections,
                index_document=body.index_document,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _to_analyze_response(result)


@app.post("/v1/jobs", response_model=JobSubmitResponse, status_code=202)
async def submit_job(body: AnalyzeRequest, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """
    Поставить анализ в очередь и сразу вернуть job_id.
    Клиент опрашивает GET /v1/jobs/{job_id} — соединение не держится минутами.
    """
    _verify_api_key(x_api_key)
    worker = _require_worker()
    store = get_job_store()

    rec = await store.create(body.university_id, body.document_id, body.filename)
    content = body.content
    filename = body.filename
    document_id = body.document_id
    university_id = body.university_id
    category = body.category
    user_id = body.user_id
    exclude_document_ids = list(body.exclude_document_ids or [])
    strip_sections = body.strip_sections
    index_document = body.index_document
    job_id = rec.job_id

    def _submit():
        return _queues.submit(
            university_id,
            lambda: worker.process_text(
                content,
                filename,
                False,
                document_id,
                category=category,
                institution_id=university_id,
                user_id=user_id,
                exclude_document_ids=exclude_document_ids,
                strip_sections=strip_sections,
                index_document=index_document,
            ),
        )

    _spawn(run_queued_job(job_id, _submit))

    return JobSubmitResponse(job_id=job_id, status="queued")


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _verify_api_key(x_api_key)
    rec = await get_job_store().get(job_id)
    if rec is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found: unknown id, or the job expired and was purged",
        )

    result = None
    if rec.status == "completed" and rec.result is not None:
        try:
            result = _to_analyze_response(rec.result)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Invalid stored result: {exc}")

    return JobStatusResponse(
        job_id=rec.job_id,
        status=rec.status,
        result=result,
        error=rec.error,
        document_id=rec.document_id,
        filename=rec.filename,
    )


@app.get("/v1/queues")
async def queues(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """Состояние очередей анализа по университетам."""
    _verify_api_key(x_api_key)
    return {
        "queues": _queues.stats(),
        "jobs": await get_job_store().stats(),
    }

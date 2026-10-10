"""
Хранилище async-джоб для POST /v1/jobs + GET /v1/jobs/{id}.

Guard-worker отправляет работу один раз и затем опрашивает статус — HTTP-соединение
не висит минутами. Поэтому состояние джобы обязано переживать рестарт контейнера:
при `restart: unless-stopped` перезапуск неизбежен, а клиент, получивший 404 на
валидный job_id, не может отличить «потеряно» от «не существовало» и опрашивает
его до истечения TTL.

Состояние пишется в SQLite на томе. Postgres был бы уместен (на него намекает имя
репозитория), но потребовал бы отдельного сервиса и дублировал бы БД платформы,
тогда как здесь нужен один небольшой журнал задач рядом с самим анализом.
Подменяется переменной ANALYSIS_JOB_DB; `:memory:` возвращает прежнее поведение
без персистентности.

Все обращения к БД уходят в asyncio.to_thread: sqlite3 блокирующий, а держать
event loop анализа занятым нельзя.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _ttl_seconds() -> int:
    raw = os.getenv("ANALYSIS_JOB_TTL_SEC", "").strip()
    if not raw:
        return 48 * 3600
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ANALYSIS_JOB_TTL_SEC=%r не число; беру 48 часов", raw)
        return 48 * 3600
    return value if value > 0 else 48 * 3600


JOB_TTL_SEC = _ttl_seconds()

DEFAULT_DB_PATH = "/app/data/jobs.db"
IN_MEMORY = ":memory:"

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"

# parsed_text — полный текст работы. В AnalyzeResponse его нет, то есть клиенту
# он всё равно не достаётся, а в журнале джоб он пролежал бы весь TTL.
_DROPPED_RESULT_KEYS = ("parsed_text",)


@dataclass
class JobRecord:
    job_id: str
    status: str
    university_id: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    document_id: Optional[int] = None
    filename: str = "document.txt"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    university_id TEXT NOT NULL,
    created_at    REAL NOT NULL,
    started_at    REAL,
    completed_at  REAL,
    result        TEXT,
    error         TEXT,
    document_id   INTEGER,
    filename      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status);
CREATE INDEX IF NOT EXISTS jobs_expiry_idx ON jobs (completed_at, created_at);
"""


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    raw_result = row["result"]
    result = None
    if raw_result:
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError:
            logger.warning("Повреждённый JSON результата джобы %s", row["job_id"])
    return JobRecord(
        job_id=row["job_id"],
        status=row["status"],
        university_id=row["university_id"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        result=result,
        error=row["error"],
        document_id=row["document_id"],
        filename=row["filename"],
    )


class JobStore:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path if db_path is not None else os.getenv(
            "ANALYSIS_JOB_DB", DEFAULT_DB_PATH
        ).strip() or DEFAULT_DB_PATH
        # :memory: живёт только в одном соединении, поэтому держим его открытым.
        self._shared: Optional[sqlite3.Connection] = None
        if self._db_path != IN_MEMORY:
            directory = os.path.dirname(self._db_path)
            if directory:
                try:
                    os.makedirs(directory, exist_ok=True)
                except OSError as exc:
                    logger.error(
                        "Каталог %s для журнала джоб недоступен (%s); "
                        "перехожу на хранение в памяти — джобы не переживут рестарт",
                        directory,
                        exc,
                    )
                    self._db_path = IN_MEMORY
        self._init_schema()

    # --- низкий уровень ---

    def _connect(self) -> sqlite3.Connection:
        if self._db_path == IN_MEMORY:
            if self._shared is None:
                self._shared = sqlite3.connect(IN_MEMORY, check_same_thread=False)
                self._shared.row_factory = sqlite3.Row
            return self._shared
        conn = sqlite3.connect(self._db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL: чтение статуса не блокируется записью результата.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _run(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        conn = self._connect()
        try:
            with conn:
                return fn(conn)
        finally:
            if conn is not self._shared:
                conn.close()

    def _init_schema(self) -> None:
        try:
            self._run(lambda conn: conn.executescript(_SCHEMA))
        except sqlite3.Error as exc:
            if self._db_path == IN_MEMORY:
                raise
            logger.error(
                "Не удалось открыть журнал джоб %s (%s); перехожу на хранение "
                "в памяти — джобы не переживут рестарт",
                self._db_path,
                exc,
            )
            self._db_path = IN_MEMORY
            self._run(lambda conn: conn.executescript(_SCHEMA))

    @property
    def persistent(self) -> bool:
        return self._db_path != IN_MEMORY

    # --- операции ---

    async def create(
        self, university_id: str, document_id: Optional[int], filename: str
    ) -> JobRecord:
        await self.purge_expired()
        rec = JobRecord(
            job_id=uuid.uuid4().hex,
            status=QUEUED,
            university_id=university_id,
            created_at=time.time(),
            document_id=document_id,
            filename=filename,
        )

        def write(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO jobs (job_id, status, university_id, created_at, "
                "document_id, filename) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    rec.job_id,
                    rec.status,
                    rec.university_id,
                    rec.created_at,
                    rec.document_id,
                    rec.filename,
                ),
            )

        await asyncio.to_thread(self._run, write)
        return rec

    async def get(self, job_id: str) -> Optional[JobRecord]:
        def read(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
            cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            return cur.fetchone()

        row = await asyncio.to_thread(self._run, read)
        return _row_to_record(row) if row is not None else None

    async def mark_processing(self, job_id: str) -> None:
        def write(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE jobs SET status = ?, started_at = ? "
                "WHERE job_id = ? AND status = ?",
                (PROCESSING, time.time(), job_id, QUEUED),
            )

        await asyncio.to_thread(self._run, write)

    async def mark_completed(self, job_id: str, result: dict[str, Any]) -> None:
        stored = {k: v for k, v in (result or {}).items() if k not in _DROPPED_RESULT_KEYS}

        def write(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE jobs SET status = ?, result = ?, completed_at = ?, "
                "error = NULL WHERE job_id = ?",
                (COMPLETED, json.dumps(stored, ensure_ascii=False), time.time(), job_id),
            )

        await asyncio.to_thread(self._run, write)

    async def mark_failed(self, job_id: str, error: str) -> None:
        message = (error or "unknown error")[:2000]

        def write(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, completed_at = ? "
                "WHERE job_id = ?",
                (FAILED, message, time.time(), job_id),
            )

        await asyncio.to_thread(self._run, write)

    async def recover_interrupted(self) -> int:
        """
        Джобы, застигнутые рестартом в queued/processing, уже никто не выполнит:
        планировщик очередей живёт в памяти процесса. Оставить их «processing»
        значит заставить клиента опрашивать их до TTL, поэтому помечаем провалом.
        """

        def write(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                "UPDATE jobs SET status = ?, error = ?, completed_at = ? "
                "WHERE status IN (?, ?)",
                (
                    FAILED,
                    "Analysis service restarted before the job finished; resubmit it",
                    time.time(),
                    QUEUED,
                    PROCESSING,
                ),
            )
            return cur.rowcount or 0

        recovered = await asyncio.to_thread(self._run, write)
        if recovered:
            logger.warning(
                "Помечено провалом %d джоб, прерванных рестартом сервиса", recovered
            )
        return recovered

    async def purge_expired(self) -> None:
        cutoff = time.time() - JOB_TTL_SEC

        def write(conn: sqlite3.Connection) -> None:
            conn.execute(
                "DELETE FROM jobs WHERE status IN (?, ?) "
                "AND COALESCE(completed_at, created_at) < ?",
                (COMPLETED, FAILED, cutoff),
            )

        await asyncio.to_thread(self._run, write)

    async def stats(self) -> dict[str, int]:
        def read(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()

        rows = await asyncio.to_thread(self._run, read)
        counts = {QUEUED: 0, PROCESSING: 0, COMPLETED: 0, FAILED: 0}
        for row in rows:
            counts[row["status"]] = row["n"]
        return {"total": sum(counts.values()), "persistent": int(self.persistent), **counts}


_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store


async def run_queued_job(
    job_id: str,
    submit: Callable[[], "asyncio.Future[Any]"],
) -> None:
    """submit() возвращает Future из UniversityQueues.submit(...)."""
    store = get_job_store()
    await store.mark_processing(job_id)
    try:
        result = await submit()
        await store.mark_completed(job_id, result if isinstance(result, dict) else {})
    except asyncio.CancelledError:
        await store.mark_failed(job_id, "Analysis was cancelled")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Async analysis job %s failed", job_id)
        await store.mark_failed(job_id, str(exc))

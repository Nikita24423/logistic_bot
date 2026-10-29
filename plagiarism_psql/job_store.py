"""
In-process async job store for POST /v1/jobs + GET /v1/jobs/{id}.

Jobs live in the analysis process memory (same lifetime as university queues).
Guard worker submits once, then polls — no long-held HTTP connection.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

JOB_TTL_SEC = int(__import__("os").getenv("ANALYSIS_JOB_TTL_SEC", str(48 * 3600)))


@dataclass
class JobRecord:
    job_id: str
    status: str  # queued | processing | completed | failed
    university_id: str
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    document_id: Optional[int] = None
    filename: str = "document.txt"


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, university_id: str, document_id: Optional[int], filename: str) -> JobRecord:
        await self._purge_expired()
        job_id = uuid.uuid4().hex
        rec = JobRecord(
            job_id=job_id,
            status="queued",
            university_id=university_id,
            document_id=document_id,
            filename=filename,
        )
        async with self._lock:
            self._jobs[job_id] = rec
        return rec

    async def get(self, job_id: str) -> Optional[JobRecord]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def mark_processing(self, job_id: str) -> None:
        async with self._lock:
            rec = self._jobs.get(job_id)
            if rec and rec.status == "queued":
                rec.status = "processing"
                rec.started_at = time.time()

    async def mark_completed(self, job_id: str, result: dict[str, Any]) -> None:
        async with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            rec.status = "completed"
            rec.result = result
            rec.completed_at = time.time()
            rec.error = None

    async def mark_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            rec.status = "failed"
            rec.error = (error or "unknown error")[:2000]
            rec.completed_at = time.time()

    async def _purge_expired(self) -> None:
        cutoff = time.time() - JOB_TTL_SEC
        async with self._lock:
            dead = [
                jid
                for jid, rec in self._jobs.items()
                if (rec.completed_at or rec.created_at) < cutoff
                and rec.status in ("completed", "failed")
            ]
            for jid in dead:
                del self._jobs[jid]

    def stats(self) -> dict[str, int]:
        counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0}
        for rec in self._jobs.values():
            counts[rec.status] = counts.get(rec.status, 0) + 1
        return {"total": len(self._jobs), **counts}


_store = JobStore()


def get_job_store() -> JobStore:
    return _store


async def run_queued_job(
    job_id: str,
    submit: Callable[[], "asyncio.Future[Any]"],
) -> None:
    """
    submit() must return an awaitable Future from UniversityQueues.submit(...).
    """
    store = get_job_store()
    await store.mark_processing(job_id)
    try:
        future = submit()
        result = await future
        await store.mark_completed(job_id, result if isinstance(result, dict) else {})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Async analysis job %s failed", job_id)
        await store.mark_failed(job_id, str(exc))

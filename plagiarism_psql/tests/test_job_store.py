import asyncio
import json
import sqlite3

import pytest

from job_store import COMPLETED, FAILED, PROCESSING, QUEUED, JobStore


@pytest.fixture
def store(tmp_path):
    return JobStore(db_path=str(tmp_path / "jobs.db"))


def test_store_is_persistent_when_given_a_path(store):
    assert store.persistent is True


@pytest.mark.asyncio
async def test_create_then_get_roundtrip(store):
    rec = await store.create("bsuir", 42, "kursovaya.docx")
    loaded = await store.get(rec.job_id)
    assert loaded is not None
    assert loaded.status == QUEUED
    assert loaded.university_id == "bsuir"
    assert loaded.document_id == 42
    assert loaded.filename == "kursovaya.docx"


@pytest.mark.asyncio
async def test_unknown_job_is_none(store):
    assert await store.get("deadbeef") is None


@pytest.mark.asyncio
async def test_status_transitions(store):
    rec = await store.create("bsuir", 1, "a.docx")
    await store.mark_processing(rec.job_id)
    assert (await store.get(rec.job_id)).status == PROCESSING
    await store.mark_completed(rec.job_id, {"plagiarism_percent": 12.3})
    done = await store.get(rec.job_id)
    assert done.status == COMPLETED
    assert done.result["plagiarism_percent"] == 12.3
    assert done.error is None


@pytest.mark.asyncio
async def test_mark_processing_does_not_resurrect_finished_job(store):
    rec = await store.create("bsuir", 1, "a.docx")
    await store.mark_failed(rec.job_id, "boom")
    await store.mark_processing(rec.job_id)
    assert (await store.get(rec.job_id)).status == FAILED


@pytest.mark.asyncio
async def test_parsed_text_is_not_stored(store):
    # Полный текст работы не должен лежать в журнале весь TTL: клиенту он
    # всё равно не отдаётся (в AnalyzeResponse такого поля нет).
    rec = await store.create("bsuir", 1, "a.docx")
    await store.mark_completed(
        rec.job_id, {"plagiarism_percent": 1.0, "parsed_text": "весь текст работы"}
    )
    loaded = await store.get(rec.job_id)
    assert "parsed_text" not in loaded.result
    assert "parsed_text" not in json.dumps(loaded.result, ensure_ascii=False)


@pytest.mark.asyncio
async def test_error_message_is_truncated(store):
    rec = await store.create("bsuir", 1, "a.docx")
    await store.mark_failed(rec.job_id, "x" * 5000)
    assert len((await store.get(rec.job_id)).error) == 2000


@pytest.mark.asyncio
async def test_jobs_survive_a_restart(tmp_path):
    # Главное, ради чего вводилась персистентность: после рестарта контейнера
    # клиент не должен получить 404 на валидный job_id.
    path = str(tmp_path / "jobs.db")
    first = JobStore(db_path=path)
    rec = await first.create("bsuir", 7, "diplom.docx")
    await first.mark_completed(rec.job_id, {"plagiarism_percent": 5.0})

    second = JobStore(db_path=path)  # «новый процесс»
    loaded = await second.get(rec.job_id)
    assert loaded is not None
    assert loaded.status == COMPLETED
    assert loaded.result["plagiarism_percent"] == 5.0


@pytest.mark.asyncio
async def test_recover_interrupted_fails_orphaned_jobs(tmp_path):
    path = str(tmp_path / "jobs.db")
    first = JobStore(db_path=path)
    queued = await first.create("bsuir", 1, "a.docx")
    running = await first.create("bsuir", 2, "b.docx")
    await first.mark_processing(running.job_id)

    second = JobStore(db_path=path)
    assert await second.recover_interrupted() == 2
    for job_id in (queued.job_id, running.job_id):
        rec = await second.get(job_id)
        assert rec.status == FAILED
        assert "restarted" in rec.error


@pytest.mark.asyncio
async def test_purge_removes_only_expired_finished_jobs(store, monkeypatch):
    import job_store as module

    fresh = await store.create("bsuir", 1, "fresh.docx")
    stale = await store.create("bsuir", 2, "stale.docx")
    await store.mark_completed(stale.job_id, {})
    # Сдвигаем срок жизни в прошлое для одной записи.
    conn = sqlite3.connect(store._db_path)
    with conn:
        conn.execute(
            "UPDATE jobs SET completed_at = 0, created_at = 0 WHERE job_id = ?",
            (stale.job_id,),
        )
    conn.close()
    await store.purge_expired()
    assert await store.get(stale.job_id) is None
    assert await store.get(fresh.job_id) is not None


@pytest.mark.asyncio
async def test_stats_counts_by_status(store):
    a = await store.create("bsuir", 1, "a.docx")
    b = await store.create("bsuir", 2, "b.docx")
    await store.mark_completed(b.job_id, {})
    stats = await store.stats()
    assert stats["total"] == 2
    assert stats[QUEUED] == 1
    assert stats[COMPLETED] == 1
    assert stats["persistent"] == 1


@pytest.mark.asyncio
async def test_unwritable_directory_falls_back_to_memory(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("это файл, а не каталог")
    store = JobStore(db_path=str(blocker / "nested" / "jobs.db"))
    assert store.persistent is False
    rec = await store.create("bsuir", 1, "a.docx")
    assert (await store.get(rec.job_id)) is not None

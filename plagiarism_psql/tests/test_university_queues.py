import asyncio

import pytest

from university_queues import DEFAULT_UNIVERSITY, UniversityQueues


@pytest.mark.asyncio
async def test_single_job_runs_and_returns_result():
    queues = UniversityQueues(concurrency=1)
    queues.start()
    try:
        assert await queues.submit("bsuir", lambda: 7) == 7
    finally:
        await queues.stop()


@pytest.mark.asyncio
async def test_exception_reaches_the_caller():
    queues = UniversityQueues(concurrency=1)
    queues.start()
    try:
        def boom():
            raise ValueError("не удалось извлечь текст")

        with pytest.raises(ValueError, match="не удалось извлечь текст"):
            await queues.submit("bsuir", boom)
    finally:
        await queues.stop()


@pytest.mark.asyncio
async def test_blank_university_falls_back_to_default():
    queues = UniversityQueues(concurrency=1)
    queues.start()
    try:
        await queues.submit("   ", lambda: 1)
        assert DEFAULT_UNIVERSITY in queues.stats()["universities"]
    finally:
        await queues.stop()


@pytest.mark.asyncio
async def test_round_robin_does_not_starve_a_quiet_university():
    # Поток документов одного вуза не должен блокировать остальные: за круг
    # берётся по одной задаче из каждой непустой очереди.
    queues = UniversityQueues(concurrency=1)
    order: list[str] = []

    def make(tag: str):
        def run():
            order.append(tag)
            return tag

        return run

    queues.start()
    try:
        futures = [queues.submit("big", make(f"big{i}")) for i in range(3)]
        futures.append(queues.submit("small", make("small0")))
        await asyncio.gather(*futures)
    finally:
        await queues.stop()

    # Работа маленького вуза выполнена не последней, несмотря на то что
    # поставлена в очередь после трёх задач большого.
    assert order.index("small0") < 3


@pytest.mark.asyncio
async def test_jobs_queued_after_a_drain_still_run():
    # Регрессия на потерянный wakeup: флаг пробуждения снимается только когда
    # очереди действительно пусты, поэтому задача, поставленная сразу после
    # опустошения, не остаётся лежать до следующего запроса.
    queues = UniversityQueues(concurrency=1)
    queues.start()
    try:
        assert await queues.submit("bsuir", lambda: "первая") == "первая"
        await asyncio.sleep(0)
        assert await asyncio.wait_for(
            queues.submit("bsuir", lambda: "вторая"), timeout=5
        ) == "вторая"
    finally:
        await queues.stop()


@pytest.mark.asyncio
async def test_stop_rejects_pending_jobs_instead_of_hanging():
    queues = UniversityQueues(concurrency=1)
    release = asyncio.Event()
    loop = asyncio.get_running_loop()

    def blocker():
        # Держим единственный слот, пока тест не разрешит продолжить.
        asyncio.run_coroutine_threadsafe(_wait(release), loop).result(timeout=10)
        return "blocked"

    async def _wait(event: asyncio.Event) -> None:
        await event.wait()

    queues.start()
    running = queues.submit("bsuir", blocker)
    await asyncio.sleep(0.05)
    pending = queues.submit("bsuir", lambda: "никогда")

    release.set()
    await queues.stop()

    assert await running == "blocked"
    with pytest.raises(RuntimeError, match="shutting down"):
        await pending


@pytest.mark.asyncio
async def test_stop_waits_for_running_jobs():
    # Задачу в threadpool прервать нельзя; stop() обязан её дождаться, иначе
    # upsert в Qdrant оборвётся на полпути.
    queues = UniversityQueues(concurrency=1)
    finished: list[str] = []

    def slow():
        import time

        time.sleep(0.2)
        finished.append("done")
        return "done"

    queues.start()
    future = queues.submit("bsuir", slow)
    await asyncio.sleep(0.05)
    await queues.stop()
    assert finished == ["done"]
    assert await future == "done"


@pytest.mark.asyncio
async def test_stats_report_queue_counters():
    queues = UniversityQueues(concurrency=1)
    queues.start()
    try:
        await queues.submit("bsuir", lambda: 1)
        stats = queues.stats()
        assert stats["total_queues"] == 1
        assert stats["universities"]["bsuir"]["processed"] == 1
        assert stats["universities"]["bsuir"]["pending"] == 0
    finally:
        await queues.stop()

"""
Очереди анализа по университетам: 1 университет = 1 очередь.

Очередь создаётся динамически при первом запросе от университета (появился
четвёртый университет — появилась четвёртая очередь). Планировщик обходит
очереди по кругу (round-robin): за один круг берётся по одной задаче из каждой
непустой очереди, поэтому большой поток документов одного университета не
блокирует остальные.

Число одновременно выполняемых задач ограничено ANALYSIS_CONCURRENCY
(по умолчанию 1 — модель одна, инференс CPU/GPU-bound).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSITY = "default"


class UniversityQueues:
    def __init__(self, concurrency: int = 1) -> None:
        self._queues: dict[str, deque] = {}
        self._processed: dict[str, int] = {}
        self._active: dict[str, int] = {}
        self._rr_order: list[str] = []
        self._rr_pos = 0
        self._wake = asyncio.Event()
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._task: Optional[asyncio.Task] = None
        # Сильные ссылки на запущенные задачи: event loop держит лишь слабую, и
        # задача без ссылки может быть собрана сборщиком мусора до завершения.
        self._running: Set[asyncio.Task] = set()

    # --- жизненный цикл ---

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Задачи, оставшиеся в очередях, уже никто не выполнит — разрешаем их
        # Future ошибкой, иначе ожидающие запросы зависнут и заблокируют shutdown.
        for queue in self._queues.values():
            while queue:
                _fn, future, _enqueued = queue.popleft()
                if not future.done():
                    future.set_exception(
                        RuntimeError("Analysis service is shutting down; retry the request")
                    )

        # Уже запущенные задачи выполняются в threadpool и прервать их нельзя:
        # дожидаемся, иначе upsert в Qdrant оборвётся на полпути.
        if self._running:
            logger.info(
                "Ожидание %d выполняющихся задач анализа перед остановкой",
                len(self._running),
            )
            await asyncio.gather(*list(self._running), return_exceptions=True)

    # --- постановка задач ---

    def submit(self, university: str, fn: Callable[[], Any]) -> "asyncio.Future[Any]":
        """Ставит задачу в очередь университета и возвращает Future с результатом."""
        university = (university or DEFAULT_UNIVERSITY).strip() or DEFAULT_UNIVERSITY
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[Any]" = loop.create_future()

        queue = self._queues.get(university)
        if queue is None:
            queue = deque()
            self._queues[university] = queue
            self._processed[university] = 0
            self._active[university] = 0
            self._rr_order.append(university)
            logger.info(
                "Создана очередь анализа для университета %r (всего очередей: %d)",
                university,
                len(self._queues),
            )

        queue.append((fn, future, time.monotonic()))
        self._wake.set()
        return future

    # --- планировщик ---

    def _next_job(self):
        """Round-robin: следующая задача из первой непустой очереди по кругу."""
        total = len(self._rr_order)
        for step in range(total):
            index = (self._rr_pos + step) % total
            university = self._rr_order[index]
            queue = self._queues[university]
            if queue:
                self._rr_pos = (index + 1) % total
                return university, queue.popleft()
        return None

    def _has_pending(self) -> bool:
        return any(self._queues[u] for u in self._rr_order)

    async def _scheduler_loop(self) -> None:
        while True:
            # Сначала свободный слот, потом задача: между извлечением задачи из
            # очереди и её запуском нет точек ожидания, поэтому отмена планировщика
            # (stop) не может потерять уже извлечённую задачу — всё невыполненное
            # остаётся в очередях и корректно отклоняется дренажем в stop().
            await self._sem.acquire()
            try:
                job = self._next_job()
                while job is None:
                    # Флаг снимается ТОЛЬКО когда очереди действительно пусты, и
                    # проверка эта — под тем же тиком event loop, что и clear().
                    # Иначе submit() между проверкой и clear() потерялся бы, и
                    # планировщик спал бы при непустой очереди до следующего
                    # запроса. Условие делает инвариант явным и переживает
                    # появление await выше по коду.
                    if not self._has_pending():
                        self._wake.clear()
                        await self._wake.wait()
                    job = self._next_job()
            except BaseException:
                self._sem.release()
                raise
            university, (fn, future, _enqueued) = job
            if future.cancelled():
                self._sem.release()
                continue
            self._active[university] += 1
            task = asyncio.create_task(self._run_job(university, fn, future))
            self._running.add(task)
            task.add_done_callback(self._running.discard)

    async def _run_job(self, university: str, fn: Callable[[], Any], future) -> None:
        try:
            result = await asyncio.to_thread(fn)
            if not future.done():
                future.set_result(result)
        except BaseException as exc:  # noqa: BLE001 — ошибку получит вызывающий через Future
            if not future.done():
                future.set_exception(exc)
        finally:
            self._active[university] -= 1
            self._processed[university] += 1
            self._sem.release()

    # --- наблюдаемость ---

    def stats(self) -> dict:
        return {
            "total_queues": len(self._queues),
            "running": len(self._running),
            "universities": {
                university: {
                    "pending": len(self._queues[university]),
                    "active": self._active[university],
                    "processed": self._processed[university],
                }
                for university in self._rr_order
            },
        }

#!/usr/bin/env python3
"""
Нагрузочное тестирование сервиса анализа (задача 3).

Запускается ПРОТИВ развёрнутого сервиса (модели должны быть загружены). Шлёт N
работ с заданной параллельностью через async-эндпоинты (POST /v1/jobs + опрос
GET /v1/jobs/{id}) и считает пропускную способность и задержки.

Цель из требований: 1400 работ/сутки сейчас, до 2–2.5 тыс. в сессию.
  1400/сутки  = 1 работа каждые ~62 c
  2500/сутки  = 1 работа каждые ~35 c
То есть одного воркера (ANALYSIS_CONCURRENCY=1) достаточно, только если ОДНА
проверка укладывается в ~35 c. Этот скрипт показывает реальное время проверки.

Пример:
    python scripts/loadtest.py --url http://localhost:8765 --api-key KEY \\
        --total 100 --concurrency 10 --file sample.docx
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import statistics
import time

import urllib.request
import urllib.error
import json


def _payload(file_path: str | None) -> dict:
    if file_path:
        ext = os.path.splitext(file_path)[1].lstrip(".").lower() or "docx"
        with open(file_path, "rb") as fh:
            content = f"FILE_BASE64|{ext}|" + base64.b64encode(fh.read()).decode()
    else:
        content = ("Тестовый текст работы для нагрузочного прогона. " * 80)
    return {"content": content, "filename": "loadtest.docx", "university_id": "loadtest"}


def _post(url: str, api_key: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get(url: str, api_key: str) -> dict:
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


async def _one(sem, base, api_key, body, poll_interval, timeout) -> tuple[bool, float]:
    async with sem:
        start = time.monotonic()
        try:
            sub = await asyncio.to_thread(_post, f"{base}/v1/jobs", api_key, body)
            job_id = sub["job_id"]
            deadline = start + timeout
            while time.monotonic() < deadline:
                st = await asyncio.to_thread(_get, f"{base}/v1/jobs/{job_id}", api_key)
                if st["status"] == "completed":
                    return True, time.monotonic() - start
                if st["status"] == "failed":
                    return False, time.monotonic() - start
                await asyncio.sleep(poll_interval)
            return False, time.monotonic() - start
        except Exception:
            return False, time.monotonic() - start


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--api-key", default=os.getenv("ANALYSIS_API_KEY", ""))
    ap.add_argument("--total", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--file", default=None, help="docx/pdf работы; иначе синтетический текст")
    ap.add_argument("--poll-interval", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    body = _payload(args.file)
    sem = asyncio.Semaphore(args.concurrency)
    print(f"Шлём {args.total} работ, параллельность {args.concurrency}, на {args.url}")

    t0 = time.monotonic()
    results = await asyncio.gather(*[
        _one(sem, args.url, args.api_key, body, args.poll_interval, args.timeout)
        for _ in range(args.total)
    ])
    wall = time.monotonic() - t0

    ok = [d for good, d in results if good]
    fail = [d for good, d in results if not good]
    print(f"\nУспешно: {len(ok)}/{args.total}, провалов/таймаутов: {len(fail)}")
    print(f"Общее время: {wall:.1f} c, пропускная способность: {len(ok)/wall:.2f} работ/c "
          f"({len(ok)/wall*86400:.0f}/сутки при этой параллельности)")
    if ok:
        ok.sort()
        p = lambda q: ok[min(len(ok) - 1, int(len(ok) * q))]
        print(f"Задержка на работу: медиана {statistics.median(ok):.1f} c, "
              f"p90 {p(0.9):.1f} c, p99 {p(0.99):.1f} c, макс {max(ok):.1f} c")
    print("\nСверка с целью: 1400/сутки ≈ 1 каждые 62 c; 2500/сутки ≈ 1 каждые 35 c.")


if __name__ == "__main__":
    asyncio.run(main())

# Guard-main integration contract

This module exposes HTTP API compatible with `guard-main/lib/analysis-client.ts`.

## Endpoint

- `POST /v1/analyze`
- `Content-Type: application/json`
- Optional header: `X-API-Key` (required only if `ANALYSIS_API_KEY` is set)

## Request body

```json
{
  "content": "text to analyze",
  "filename": "document.txt",
  "document_id": 123,
  "university_id": "bsuir"
}
```

- `content` is required, non-empty string
- `filename` optional, defaults to `document.txt`
- `document_id` optional — сохраняется в payload чанков Qdrant и используется
  для атрибуции источника в `semantic_matches`
- `university_id` optional (default: `default`) — очередь анализа: 1 университет =
  1 очередь, создаётся динамически при первом запросе. Планировщик обходит очереди
  по кругу (round-robin), поэтому поток документов одного университета не блокирует
  остальные. Состояние очередей: `GET /v1/queues` (тот же `X-API-Key`).

## Success response (200)

```json
{
  "plagiarism_percent": 12.34,
  "ai_percent": 7.89,
  "semantic_matches": [
    {
      "document_id": 42,
      "filename": "kursovaya_ivanov.docx",
      "matched_chunks": 7,
      "max_score": 0.83,
      "max_lexical_score": 0.95,
      "paraphrase_score": 0.22,
      "match_type": "exact",
      "sample": "первые 300 символов совпавшего чанка…"
    }
  ],
  "by_type": { "exact": 1, "paraphrase": 0, "semantic": 0 }
}
```

- `match_type` — каскадная классификация источника (источник истины — `cascade.py`;
  локальный каскад guard-main — только фолбэк): `exact` (лексический косинус ≥
  `LEXICAL_SCORE_THRESHOLD`, дословное копирование) → `paraphrase` (Жаккар 3-грамм
  смысловых слов ≥ `PARAPHRASE_THRESHOLD`, переформулировка) → `semantic`
  (только dense-сигнал: глубокий рерайт).
- `by_type` — число источников в каждом тире каскада.

- `semantic_matches` — совпадения, агрегированные по работам-источникам. Поиск
  гибридный: `max_score` — максимальная косинусная близость эмбеддингов (ловит
  глубокий рерайт), `max_lexical_score` — максимальный лексический косинус
  BM25-подобных разреженных векторов в [0..1] (ловит дословное копирование;
  1.0 — точная копия чанка). Отсортированы по числу совпавших чанков; `sample` —
  фрагмент проверяемого текста. `document_id` источника может быть `null`,
  если источник индексировался без id.

## Error responses

- `400` failed to extract text from base64 file (broken PDF/DOCX)
- `401` invalid API key
- `422` invalid request schema
- `503` worker not initialized
- `500` internal worker/runtime error

## Environment variables

- `QDRANT_HOST` (default: `localhost`)
- `QDRANT_PORT` (default: `6333`)
- `QDRANT_COLLECTION` (default: `university_docs`)
- `ANALYSIS_API_KEY` (optional; must match `ANALYSIS_SERVICE_API_KEY` on guard-main)
- `DEVICE` (default: `auto`) — `auto` | `cpu` | `cuda` (use `cuda` only with GPU-enabled PyTorch + GPU access)
- `ANALYSIS_CONCURRENCY` (default: `1`) — сколько задач анализа выполняется одновременно (очереди университетов обходятся по кругу). Внутри одного процесса анализ дополнительно сериализуется локом ML-воркера, поэтому значения >1 дают эффект только при нескольких экземплярах сервиса
- `QDRANT_SCORE_THRESHOLD` (default: `0.80`) — порог косинуса эмбеддингов (семантический сигнал)
- `LEXICAL_SCORE_THRESHOLD` (default: `0.60`) — порог лексического косинуса (дословное копирование)
- `PARAPHRASE_THRESHOLD` (default: `0.15`) — порог Жаккара смысловых 3-грамм для тира «перефраз» каскада
- `AI_ENABLED` (default: `true`) — `false` не грузит языковую модель и отдаёт `ai_percent = 0`
- `AI_DETECT_BATCH_SIZE` (default: `8`) — размер батча детектора ИИ (больше — быстрее, но выше пик памяти)
- `AI_DETECT_D0` (default: `1.0`), `AI_DETECT_SCALE` (default: `1.5`) — калибровка сигмоиды
- `AI_CHUNK_SIZE` / `AI_CHUNK_OVERLAP` (default: `500` / `50`) — окно детектора, отдельно от поиска
- `AI_MIN_CHUNK_CHARS` (default: `100`) — окна этой длины и короче пропускаются
- `AI_MAX_TOKENS` (default: `1024`) — обрезка окна
- `AI_AGGREGATION` (default: `mean`) — `mean` | `median` | `max` | `top_share`
- `AI_TOP_SHARE_THRESHOLD` (default: `AI_DETECT_D0`) — порог для `top_share`
- `AI_DEBUG_LOG` / `AI_DEBUG_LOG_CHUNKS` (default: `false`) — сводка и окна в stdout
- `EMBEDDING_MODEL` (default: `BAAI/bge-m3`), `AI_MODEL` (default: `sberbank-ai/rugpt3small_based_on_gpt2`) — замена моделей; `AI_MODEL` требует пересборки образа, остальные `AI_*` — только перезапуск
- `CORS_ALLOW_ORIGINS` (optional) — список origin через запятую для браузерных запросов напрямую к API
- `TORCH_CPU_THREADS` (optional) — OpenMP/torch CPU threads on CPU runs (default: up to 8 or available cores)

## Guard-main pairing

- ML base URL on web: **`ANALYSIS_SERVICE_URL`** (e.g. `http://172.16.251.219:8765`, no trailing slash).
- Shared secret header: **`ANALYSIS_SERVICE_API_KEY`** on Next.js MUST equal **`ANALYSIS_API_KEY`** on this service.

## Run locally

```bash
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8765
```

Или в Docker:

```bash
docker compose up -d --build analysis
```

## Коды типов заданий (`category`)

Каждый тип сравнивается ТОЛЬКО с работами того же типа (решение БГУИР):

| Код | Тип задания |
|-----|-------------|
| `LR`  | лабораторная работа |
| `PZ`  | практическая работа |
| `IPR` | индивидуальная практическая работа |
| `KuP` | курсовой проект |
| `KuR` | курсовая работа |
| `DiP` | дипломный проект |
| `DiR` | дипломная работа |

Работа и проект одного уровня (`KuP`/`KuR`, `DiP`/`DiR`) — разные типы, в общий
пул не объединяются. Особый код `seo` сравнивается со всем корпусом вуза без
фильтра по типу.

## Ошибки

При сбое сервис возвращает `{"error_code": "<код>", "detail": "<краткая причина
на русском>"}` — не только код (задача 4). Коды и сообщения:

| error_code | HTTP | Сообщение пользователю |
|------------|------|------------------------|
| `unsupported_format` | 400 | Формат файла не поддерживается. Загрузите PDF или DOCX. |
| `corrupt_file` | 400 | Не удалось обработать файл. Проверьте его содержимое. |
| `no_text` | 400 | В документе не обнаружен текст для проверки. |
| `extraction_failed` | 400 | Не удалось извлечь текст из документа. |
| `overloaded` | 503 | Система перегружена. Работа ожидает проверки. |
| `internal_error` | 500 | Во время проверки произошла ошибка обработки. |

Для асинхронных задач те же `error_code` и `detail` приходят в
`GET /v1/jobs/{id}` (поля `error_code`, `error`).

Категории, которые возникают на стороне СЭО при обращении к сервису
(`service_unavailable`, `timeout`, `result_delivery_failed`, `auth_failed`),
живут в том же каталоге `errors.py` для единых формулировок, но сам сервис их
не поднимает — их показывает СЭО.

## Полный отчёт: совпавшие фрагменты (задача 1)

Каждый элемент `semantic_matches` помимо агрегатов несёт список `fragments` —
какие фрагменты работы с какими фрагментами источника совпали:

```json
"fragments": [
  {
    "work_text": "фрагмент проверяемой работы",
    "source_text": "фрагмент работы-источника",
    "match_type": "exact | paraphrase | semantic",
    "semantic_score": 0.0,
    "lexical_score": 0.91,
    "paraphrase_score": 0.55
  }
]
```

Один чанк может совпасть с несколькими источниками (скомпилированный текст):
на чанк берётся до `REPORT_SOURCES_PER_CHUNK` источников. Число фрагментов на
источник ограничено `REPORT_MAX_FRAGMENTS_PER_SOURCE`.

Настройки:
- `REPORT_SOURCES_PER_CHUNK` (по умолчанию `5`) — сколько источников искать на чанк.
- `REPORT_MAX_FRAGMENTS_PER_SOURCE` (по умолчанию `50`) — лимит фрагментов на источник.

Задача 2 (полный отчёт — преподавателю, краткий — студенту) решается на стороне
СЭО: сервис отдаёт полные данные, а разграничение показа — за платформой.

## Нормализация текста (задача 10)

Перед нарезкой текст нормализуется (`text_normalize.py`): Unicode NFKC, единый
перевод строки, схлопнутые пробелы, убранные мягкие переносы и символы нулевой
ширины. Это снижает разброс показателей (в т.ч. ИИ) между форматами одной
работы: детектор детерминирован, а расхождение давали тривиальные различия
извлечённого текста (PDF vs DOCX, пересохранение), сдвигавшие границы чанков.

## Хранение и повторные сдачи (задача 5)

Работы хранятся в корпусе Qdrant бессрочно, поэтому работа, сданная год назад
другим студентом, находится как источник. Своя доработанная работа не совпадает
сама с собой: фильтр исключает текущий документ и все работы того же `user_id`
(для старых точек без `user_id` СЭО передаёт список своих работ в
`exclude_document_ids`).

## Нагрузка и ёмкость (задача 3)

Цель: 1400 работ/сутки сейчас, до 2–2.5 тыс. в сессию.

  1400/сутки = 1 проверка каждые ~62 c
  2500/сутки = 1 проверка каждые ~35 c

Архитектура: `ANALYSIS_CONCURRENCY` одновременных проверок (по умолчанию 1 —
модель одна, инференс CPU/GPU-bound), очереди по вузам обходятся round-robin,
поэтому поток одного вуза не блокирует остальные. Асинхронные задачи
(`POST /v1/jobs` + опрос) не держат HTTP-соединение.

Вывод по ёмкости: одного воркера достаточно для 2.5 тыс/сутки ТОЛЬКО если одна
проверка укладывается в ~35 c. На CPU проверка многостраничной работы (эмбеддинг
bge-m3 + форварды детектора ИИ) может занимать десятки секунд, поэтому для
пиковой нагрузки вероятно потребуется либо GPU, либо `ANALYSIS_CONCURRENCY > 1`
на нескольких процессах/узлах.

Измерить реальное время проверки и пропускную способность на развёрнутом
сервисе: `scripts/loadtest.py` (шлёт N работ с заданной параллельностью и
считает задержки и работы/сутки). Запускать против стенда с загруженными
моделями, а не в CI.

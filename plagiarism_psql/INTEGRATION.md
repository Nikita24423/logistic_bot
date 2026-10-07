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
- `strip_sections` optional (bool, default — значение env `STRIP_SERVICE_SECTIONS`,
  по умолчанию включено) — вырезать служебные разделы работы до анализа:
  титульный лист, содержание/оглавление, список использованных источников,
  приложения. Они не являются авторским текстом: титульник одного вуза идентичен
  у всех работ, а библиографическая запись по ГОСТу совпадает дословно у любых двух
  студентов, цитирующих одну книгу, — без отсечения такие совпадения попадают в тир
  `exact`. Объёмное приложение (листинги, таблицы) при этом раздувает знаменатель
  `plagiarism_percent`. Отсечение выполняется до нарезки на чанки, и очищенный текст
  идёт и в поиск, и в индекс Qdrant. Передайте `false`, чтобы проверить работу целиком.
  Что именно вырезано — в поле `sections` ответа.
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
- `sections` — отчёт об отсечении служебных разделов: `removed_sections`
  (`title_page`, `table_of_contents`, `references`, `appendix`), `chars_before`,
  `chars_after` и `fallback`. `fallback: true` означает, что разбор структуры
  отвергнут предохранителем (отсечение съело бы почти весь текст) и работа
  проанализирована целиком — процент в этом случае включает служебные разделы.

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
- `AI_DETECT_BATCH_SIZE` (default: `8`) — размер батча детектора ИИ (больше — быстрее, но выше пик памяти)
- `AI_DETECT_D0` (default: `1.0`), `AI_DETECT_SCALE` (default: `1.5`) — калибровка детектора ИИ
- `EMBEDDING_MODEL` (default: `BAAI/bge-m3`), `AI_MODEL` (default: `sberbank-ai/rugpt3small_based_on_gpt2`) — замена моделей без правки кода
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
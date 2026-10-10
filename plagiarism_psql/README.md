# plagiarism_psql

Docker-образуемый модуль анализа текста: **HTTP API** (`POST /v1/analyze`), совместимый с клиентом `guard-main` (`ANALYSIS_SERVICE_URL`).

Что делает: **гибридный** поиск заимствований — семантический (эмбеддинги **BAAI/bge-m3**, ловит глубокий рерайт) плюс лексический (BM25-подобные разреженные векторы, ловит дословное копирование) в Qdrant, с атрибуцией работ-источников в `semantic_matches` — и определение ИИ-текста методом **Fast-DetectGPT** на открытой LM `rugpt3small`.

Перед анализом из работы вырезаются служебные разделы — титульный лист,
содержание, список использованных источников и приложения (`document_sections.py`):
авторским текстом они не являются и искажают процент в обе стороны. Отключается
через `STRIP_SERVICE_SECTIONS=0` или полем `strip_sections` в запросе.

Подробный контракт: [INTEGRATION.md](INTEGRATION.md).

## Быстрый запуск (Docker, Linux)

```bash
git clone https://github.com/sakura-sak/plagiarism_psql.git
cd plagiarism_psql
cp env.docker.example .env
# задайте ANALYSIS_API_KEY в .env при необходимости
docker compose up -d --build
curl -sS http://localhost:8765/health
```

## Переменные окружения

- **Для Docker Compose:** файл `.env` в корне репозитория (см. `env.docker.example`): `ANALYSIS_API_KEY`.
- **Qdrant:** в `docker-compose.yml` заданы `QDRANT_*` для сервиса `analysis`; при необходимости вынесите в `.env`.
- **Устройство для ML (CPU/GPU):** переменная `DEVICE`:
  - `DEVICE=auto` (по умолчанию) — использовать `cuda`, если доступно, иначе `cpu`
  - `DEVICE=cuda` — требовать GPU (если CUDA недоступна, сервис упадёт при старте)
  - `DEVICE=cpu` — принудительно CPU
- **Отсечение служебных разделов:** `STRIP_SERVICE_SECTIONS` (`1` по умолчанию, `0` — анализировать работу целиком)
- **Индексация проверяемых работ:** `INDEX_ANALYZED_DOCUMENTS` (`1` по умолчанию; `0` — не добавлять работу в корпус сравнения)
- **Порог тира «дословное копирование»:** `EXACT_THRESHOLD` (`0.80`); порог лексического *поиска* — отдельный `LEXICAL_SCORE_THRESHOLD` (`0.60`)
- **Пересоздание коллекции Qdrant:** `ALLOW_COLLECTION_RECREATE` (`0`; при `1` несовместимость схемы приводит к УДАЛЕНИЮ всех проиндексированных работ)
- **Журнал async-джоб:** `ANALYSIS_JOB_DB` (`/app/data/jobs.db`; `:memory:` — без персистентности), TTL — `ANALYSIS_JOB_TTL_SEC`
- **Лимиты входа:** `ANALYSIS_MAX_CONTENT_CHARS`, `ANALYSIS_MAX_FILE_BYTES`, `ANALYSIS_MAX_PDF_PAGES`, `ANALYSIS_MAX_UNCOMPRESSED_BYTES`

Порт Qdrant намеренно **не публикуется** на хост: у него нет аутентификации.
Для отладки раскомментируйте `ports` в `docker-compose.yml`, привязав к `127.0.0.1`.

Для GPU в Docker на Linux требуется NVIDIA runtime (`nvidia-container-toolkit`) и запуск контейнера с доступом к GPU.

## Локальный запуск без Docker

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_COLLECTION=university_docs
uvicorn api_server:app --host 0.0.0.0 --port 8765
```

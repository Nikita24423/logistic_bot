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
git clone https://github.com/Andhanc/plagiarism_psql.git
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

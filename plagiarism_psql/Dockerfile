# syntax=docker/dockerfile:1.7
# ML analysis API — models baked into the image (slow first build, ready after start).
# First build ~25–40 min (PyTorch + HuggingFace). Code-only rebuild: ~1–2 min.
#
# BuildKit cache mounts resume interrupted model downloads on the next build.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence-transformers \
    TORCH_HOME=/opt/models/torch \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    QDRANT_COLLECTION=university_docs

# Layer 1 — Python deps (invalidates only when requirements.txt changes)
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Layer 2 — HuggingFace models baked into image
ARG EMBEDDING_MODEL=BAAI/bge-m3
ARG AI_MODEL=sberbank-ai/rugpt3small_based_on_gpt2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    AI_MODEL=${AI_MODEL}

COPY preload_models.py ./
# Cache mounts: if build is interrupted, next build resumes downloads instead of starting over.
RUN --mount=type=cache,id=analysis-hf,target=/cache/hf \
    --mount=type=cache,id=analysis-st,target=/cache/st \
    HF_HOME=/cache/hf \
    SENTENCE_TRANSFORMERS_HOME=/cache/st \
    python preload_models.py \
 && mkdir -p /opt/models/huggingface /opt/models/sentence-transformers \
 && cp -a /cache/hf/. /opt/models/huggingface/ \
 && cp -a /cache/st/. /opt/models/sentence-transformers/

# Layer 3 — application code (changes often; cheap rebuild)
COPY api_server.py worker.py cascade.py university_queues.py job_store.py \
     comparison_scope.py document_sections.py document_text.py ./

# Каталог журнала джоб (в compose сюда монтируется том analysis_jobs).
RUN mkdir -p /app/data

EXPOSE 8765

# ready=true only after models are loaded into memory (still ~1–5 min on CPU after start).
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=10 \
    CMD python -c "import json, sys, urllib.request; d = json.load(urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=5)); sys.exit(0 if d.get('ready') else 1)"

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8765"]

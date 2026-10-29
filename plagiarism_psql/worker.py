import base64
import hashlib
import io
import math
import os
import re
import threading
import time
import uuid
import warnings
from collections import Counter
from typing import Dict, List, Optional

import urllib3

urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _parse_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


# BLAS/OpenMP: выставляем до import torch, иначе MKL не подхватит.
# На CPU раньше было жёстко 1 поток — из‑за этого LM forward на rugpt3small очень долгий.
_default_cpu = min(8, max(2, os.cpu_count() or 4))
_cpu_threads = _parse_positive_int("TORCH_CPU_THREADS", _default_cpu)
os.environ["OMP_NUM_THREADS"] = str(_cpu_threads)
os.environ["MKL_NUM_THREADS"] = str(_cpu_threads)

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    MatchAny,
    PointStruct,
    QueryRequest,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Стоп-слова и каскадная классификация заимствований — в cascade.py
# (единственный источник истины по типам exact/paraphrase/semantic).
from ai_scoring import (
    ai_percent_from_scores,
    format_startup_line,
    load_ai_config,
    score_stats,
    sigmoid_ai_percent,
)
from comparison_scope import comparison_categories, normalize_category_slug
from errors import AnalysisError, AnalysisErrorCode
from text_normalize import normalize_text
from cascade import (
    DEFAULT_PARAPHRASE_THRESHOLD,
    STOPWORDS as _STOPWORDS,
    classify_match,
    paraphrase_similarity,
)


def _term_id(token: str) -> int:
    # Стабильный 32-битный id термина для разреженного вектора Qdrant.
    # Встроенный hash() не годится: он меняется между запусками процесса.
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")


class AntiPlagiarismWorker:
    # Имена векторов в коллекции Qdrant (гибридный поиск):
    # dense — семантический эмбеддинг (рерайт, перефраз, межъязыковые заимствования),
    # lexical — разреженный BM25-подобный вектор (дословное копирование, редкие
    # термины, ФИО, номера ГОСТов) с объяснимым сигналом «совпали конкретные слова».
    DENSE_VECTOR = "dense"
    LEXICAL_VECTOR = "lexical"

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "university_docs",
    ):
        # Единственный ML-worker разделяется между запросами (api_server → threadpool).
        # Модель PyTorch и клиент Qdrant не рассчитаны на конкурентный доступ, поэтому
        # тяжёлый анализ сериализуется этим локом (эффективная конкурентность = 1).
        self._process_lock = threading.Lock()
        requested_device = os.getenv("DEVICE", "auto").strip().lower()
        if requested_device not in {"auto", "cpu", "cuda"}:
            requested_device = "auto"

        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "DEVICE=cuda requested but CUDA is not available in this runtime. "
                    "Install CUDA-enabled PyTorch and run the container with GPU access."
                )
            self.device = "cuda"
        elif requested_device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            torch.set_num_threads(_cpu_threads)

        # Модель эмбеддингов для семантического поиска заимствований.
        # По умолчанию BAAI/bge-m3: мультиязычная (вкл. русский), контекст до 8192
        # токенов, 1024 dim. Prefix-free — не требует "query:"/"passage:" (в отличие
        # от e5-моделей). Меняется через env без правки кода.
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()

        # Порог косинусной близости, выше которого чанк считается заимствованным.
        # ВНИМАНИЕ: у каждой модели своё распределение косинусов. Значение 0.80 —
        # разумная стартовая точка для bge-m3; перекалибруйте на своих данных.
        self.qdrant_score_threshold = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.80"))

        # Порог лексического косинуса гибридного поиска. Оба вектора L2-нормированы,
        # поэтому скор лежит в [0..1]: 1.0 — дословная копия чанка, ~0.8 — лёгкая
        # правка, <0.3 — несвязанные тексты. Калибруется через env.
        self.lexical_score_threshold = float(os.getenv("LEXICAL_SCORE_THRESHOLD", "0.60"))

        # Максимум фрагментов на источник в полном отчёте (задача 1): ограничивает
        # размер ответа на работах с массой совпадений.
        self.report_max_fragments = _parse_positive_int("REPORT_MAX_FRAGMENTS_PER_SOURCE", 50)

        # Сколько работ-источников искать на каждый чанк. Раньше было 1 — и если
        # один фрагмент скомпилирован из нескольких работ, атрибутировался лишь
        # один источник. Теперь до REPORT_SOURCES_PER_CHUNK источников на чанк.
        self.report_top_k = _parse_positive_int("REPORT_SOURCES_PER_CHUNK", 5)

        # Порог Жаккара 3-грамм смысловых слов для тира «перефраз» каскада.
        self.paraphrase_threshold = float(
            os.getenv("PARAPHRASE_THRESHOLD", str(DEFAULT_PARAPHRASE_THRESHOLD))
        )

        # Детектор ИИ — Fast-DetectGPT. Все пороги, окно и агрегация — из env
        # (ai_scoring.load_ai_config). Поиск заимствований ниже остаётся на 500/50.
        self.ai_config = load_ai_config()
        self.ai_detect_d0 = self.ai_config.d0
        self.ai_detect_scale = self.ai_config.scale
        self.ai_detect_batch_size = self.ai_config.batch_size
        if self.ai_config.debug_log:
            print(format_startup_line(self.ai_config), flush=True)

        self.encoder = SentenceTransformer(
            self.embedding_model_name, device=self.device
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        self.ai_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.ai_config.chunk_size,
            chunk_overlap=self.ai_config.chunk_overlap,
        )

        # Scoring-модель для детектора ИИ (любая открытая causal LM). rugpt3small —
        # компактная и бесплатная; при желании заменяется на более сильную через env.
        # AI_ENABLED=false не грузит модель: процент ИИ тогда всегда 0.
        self.ai_tokenizer = None
        self.ai_model = None
        if self.ai_config.enabled:
            model_name = self.ai_config.model
            self.ai_tokenizer = AutoTokenizer.from_pretrained(model_name)
            # У GPT-2-подобных моделей нет pad-токена, а батчинг требует паддинга.
            # eos безопасен: padding-позиции всё равно исключаются маской внимания.
            if self.ai_tokenizer.pad_token is None:
                self.ai_tokenizer.pad_token = self.ai_tokenizer.eos_token
            # Только правый паддинг: при левом реальные токены получают сдвинутые
            # position_ids, и logits расходятся с по-чанковым вычислением
            # (у rugpt3 в конфиге токенизатора задан именно левый).
            self.ai_tokenizer.padding_side = "right"
            self.ai_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.ai_model.eval()

        self.qdrant = QdrantClient(
            url=f"http://{qdrant_host}:{qdrant_port}", prefer_grpc=False, timeout=30.0
        )
        self.collection_name = collection_name
        self._ensure_qdrant_collection()

    def _ensure_qdrant_collection(self) -> None:
        """
        The API relies on Qdrant for similarity search. If the expected collection
        does not exist, Qdrant returns 404 and the API replies 500.
        Create the collection at startup to make the service self-contained.
        """
        # Qdrant may still be booting when the analysis container starts.
        last_err: Exception | None = None
        vector_size = int(self.encoder.get_sentence_embedding_dimension())
        for _ in range(30):
            try:
                existing = {c.name for c in self.qdrant.get_collections().collections}
                if self.collection_name in existing:
                    # Коллекция есть — проверяем совместимость схемы. Несовместимость
                    # возможна в двух случаях: сменили EMBEDDING_MODEL (другая
                    # размерность dense-вектора) или коллекция создана старой версией
                    # без гибридного поиска (безымянный вектор, нет lexical). В обоих
                    # случаях коллекцию нужно пересоздать (данные в ней будут удалены —
                    # работы придётся переиндексировать заново).
                    # Ошибка чтения схемы уходит в retry-цикл (внешний except),
                    # а не трактуется как несовпадение: иначе транзиентный сбой
                    # get_collection привёл бы к удалению живой коллекции.
                    current_size = None
                    info = self.qdrant.get_collection(self.collection_name)
                    dense_cfg = info.config.params.vectors
                    if isinstance(dense_cfg, dict):
                        dense_params = dense_cfg.get(self.DENSE_VECTOR)
                        current_size = dense_params.size if dense_params else None
                    sparse_cfg = info.config.params.sparse_vectors or {}
                    has_lexical = self.LEXICAL_VECTOR in sparse_cfg
                    if current_size == vector_size and has_lexical:
                        return
                    print(
                        f"Qdrant collection schema mismatch "
                        f"(dense: {current_size} -> {vector_size}, lexical: {has_lexical}); "
                        f"recreating collection."
                    )
                    self.qdrant.delete_collection(self.collection_name)

                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.DENSE_VECTOR: VectorParams(
                            size=vector_size, distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={self.LEXICAL_VECTOR: SparseVectorParams()},
                )
                return
            except Exception as e:  # includes connection errors / Qdrant not ready yet
                last_err = e
                time.sleep(1)

        if last_err:
            raise RuntimeError(f"Qdrant is not ready: {last_err}") from last_err

    def extract_real_text(self, text_input: str) -> str:
        """
        Текст работы из `FILE_BASE64|ext|...` либо строка как есть.

        На каждую категорию сбоя поднимается AnalysisError со своим кодом
        (задача 4: пользователю возвращается краткая причина, а не общий текст).
        Заглушку вместо текста не возвращаем: раньше она индексировалась как
        текст документа, и все битые файлы «совпадали» с ней на 100%.
        """
        if not isinstance(text_input, str) or not text_input.startswith("FILE_BASE64|"):
            return text_input
        parts = text_input.split("|", 2)
        if len(parts) != 3:
            raise AnalysisError(
                AnalysisErrorCode.CORRUPT_FILE,
                detail="FILE_BASE64 payload has fewer than 3 parts",
            )
        _, ext, b64_data = parts
        ext = ext.lower().strip()

        if ext not in ("pdf", "docx", "doc"):
            raise AnalysisError(
                AnalysisErrorCode.UNSUPPORTED_FORMAT,
                detail=f"unsupported extension {ext!r}",
            )

        try:
            file_bytes = base64.b64decode(b64_data)
        except Exception as e:  # noqa: BLE001 — битый base64
            raise AnalysisError(
                AnalysisErrorCode.CORRUPT_FILE, detail=f"base64 decode failed: {e}"
            ) from e

        file_stream = io.BytesIO(file_bytes)
        try:
            if ext == "pdf":
                from pypdf import PdfReader

                reader = PdfReader(file_stream)
                extracted = "\n".join(
                    [page.extract_text() or "" for page in reader.pages]
                ).strip()
            else:  # docx / doc
                import docx

                doc = docx.Document(file_stream)
                extracted = "\n".join([p.text for p in doc.paragraphs]).strip()
        except Exception as e:  # noqa: BLE001 — повреждённый или нечитаемый файл
            raise AnalysisError(
                AnalysisErrorCode.EXTRACTION_FAILED, detail=f"{ext} parse failed: {e}"
            ) from e

        if not extracted:
            raise AnalysisError(
                AnalysisErrorCode.NO_TEXT,
                detail=f"no extractable text in {ext} file",
            )
        return extracted

    def _lexical_sparse(self, text: str) -> Optional[SparseVector]:
        """
        Разреженный лексический вектор чанка для гибридного поиска.

        BM25-подобное взвешивание без корпусной IDF: sublinear TF (1 + ln tf)
        плюс отбрасывание стоп-слов, затем L2-нормировка. Скалярное произведение
        двух таких векторов — косинус в [0..1], интерпретируемый как доля
        лексического пересечения: 1.0 у дословной копии. Возвращает None, если
        в чанке не осталось значимых слов (тогда работает только dense-сигнал).
        """
        tokens = [
            t
            for t in re.findall(r"[a-zа-яё0-9]+", text.lower())
            if len(t) >= 2 and t not in _STOPWORDS
        ]
        if not tokens:
            return None
        weights: Dict[int, float] = {}
        for token, tf in Counter(tokens).items():
            # Коллизии 32-битного хеша на объёме словаря одного чанка пренебрежимы.
            weights[_term_id(token)] = 1.0 + math.log(tf)
        norm = math.sqrt(sum(w * w for w in weights.values()))
        indices = sorted(weights)
        return SparseVector(indices=indices, values=[weights[i] / norm for i in indices])

    def _log_ai(
        self,
        document_id: Optional[int],
        filename: str,
        windows_total: int,
        pairs: List[tuple],
        ai_percent: float,
    ) -> None:
        if not self.ai_config.debug_log:
            return
        stats = score_stats([score for _, score in pairs])

        def fmt(value: Optional[float]) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        print(
            "AI debug: "
            f"document_id={document_id} file={filename!r} model={self.ai_config.model} "
            f"windows={windows_total} scored={len(pairs)} "
            f"mean={fmt(stats['mean'])} median={fmt(stats['median'])} "
            f"min={fmt(stats['min'])} max={fmt(stats['max'])} "
            f"aggregation={self.ai_config.aggregation} ai_percent={ai_percent}",
            flush=True,
        )
        if not self.ai_config.debug_chunks:
            return
        for chunk, score in pairs:
            preview = " ".join(chunk.split())[:80]
            print(f"AI debug chunk: D={score:.4f} {preview}", flush=True)

    def _analyze_ai_chunks(self, texts: List[str]) -> List[Optional[float]]:
        """
        Детектор ИИ методом Fast-DetectGPT (Bao et al., 2024), аналитическая
        sampling-free версия — без обучения, на той же открытой LM.

        Идея: текст, сгенерированный ИИ, лежит в области высокой условной
        вероятности модели. Считаем нормированную «кривизну условной вероятности»:

            D = (Σ logp(x_i) − Σ μ_i) / sqrt(Σ σ²_i),

        где на каждой позиции по распределению модели p(·):
            μ_i  = Σ_v p(v)·logp(v)      — матожидание log-вероятности,
            σ²_i = Σ_v p(v)·logp(v)² − μ² — её дисперсия.
        Чем больше D, тем вероятнее машинная генерация.

        Чанки прогоняются батчами (AI_DETECT_BATCH_SIZE) — один forward вместо
        B последовательных. Паддинг справа до самого длинного чанка в батче;
        padding-позиции исключены из статистики маской внимания, поэтому
        результат совпадает с по-чанковым вычислением.

        Возвращает список статистик D (float или None для слишком коротких
        чанков) в порядке исходных текстов.
        """
        results: List[Optional[float]] = []
        for start in range(0, len(texts), self.ai_detect_batch_size):
            batch = texts[start : start + self.ai_detect_batch_size]
            enc = self.ai_tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.ai_config.max_tokens,
                truncation=True,
                padding=True,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                logits = self.ai_model(
                    input_ids, attention_mask=attention_mask
                ).logits[:, :-1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = log_probs.exp()

                # Матожидание и дисперсия log p по распределению модели на каждой позиции.
                mu = (probs * log_probs).sum(dim=-1)                # E[logp]
                second = (probs * log_probs.pow(2)).sum(dim=-1)     # E[logp²]
                var = (second - mu.pow(2)).clamp(min=0.0)           # Var[logp] ≥ 0

                actual = input_ids[:, 1:]                           # реально следующие токены
                observed = log_probs.gather(2, actual.unsqueeze(-1)).squeeze(-1)

                # Позиция учитывается, только если её целевой токен реальный
                # (не padding); при правом паддинге префикс таких позиций
                # всегда состоит из реальных токенов.
                target_mask = attention_mask[:, 1:].to(log_probs.dtype)
                for row in range(input_ids.shape[0]):
                    mask = target_mask[row]
                    if mask.sum().item() < 1:  # < 2 реальных токенов в чанке
                        results.append(None)
                        continue
                    denom = torch.sqrt((var[row] * mask).sum()) + 1e-6
                    d = (((observed[row] - mu[row]) * mask).sum() / denom).item()
                    results.append(d)
        return results

    def _build_comparison_filter(
        self,
        document_id: Optional[int],
        filename: str,
        category: str,
        institution_id: Optional[str],
        user_id: Optional[str] = None,
        exclude_document_ids: Optional[List[int]] = None,
    ) -> Filter:
        """
        Сравниваем только с чужими работами того же вуза/типа.
        Исключаем: текущий документ, все документы того же user_id,
        и явный список exclude_document_ids (для старых точек без user_id в payload).
        """
        must_not: List[FieldCondition] = []
        if document_id is not None:
            must_not.append(
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            )
        else:
            must_not.append(
                FieldCondition(key="filename", match=MatchValue(value=filename))
            )

        uid = (user_id or "").strip()
        if uid:
            must_not.append(
                FieldCondition(key="user_id", match=MatchValue(value=uid))
            )

        extra_ids = [
            int(x)
            for x in (exclude_document_ids or [])
            if isinstance(x, (int, float)) and int(x) > 0 and int(x) != document_id
        ]
        # Unique, capped — Qdrant MatchAny on huge lists is costly
        extra_ids = sorted(set(extra_ids))[:500]
        if extra_ids:
            must_not.append(
                FieldCondition(key="document_id", match=MatchAny(any=extra_ids))
            )

        must: List[FieldCondition] = []
        inst = (institution_id or "").strip()
        if inst:
            must.append(
                FieldCondition(key="institution_id", match=MatchValue(value=inst))
            )

        pool = comparison_categories(category)
        # None: SEO work, compare with every category of this institution.
        # Site categories never include "seo", so those works stay out of the site pool.
        if pool is None:
            pass
        elif len(pool) == 1:
            must.append(FieldCondition(key="category", match=MatchValue(value=pool[0])))
        else:
            must.append(FieldCondition(key="category", match=MatchAny(any=pool)))

        return Filter(must=must, must_not=must_not)

    def process_text(
        self,
        text: str,
        filename: str,
        verbose: bool = True,
        document_id: Optional[int] = None,
        category: Optional[str] = None,
        institution_id: Optional[str] = None,
        user_id: Optional[str] = None,
        exclude_document_ids: Optional[List[int]] = None,
    ) -> Dict:
        # Сериализуем доступ к общей ML-модели и клиенту Qdrant (см. self._process_lock).
        with self._process_lock:
            return self._process_text_locked(
                text,
                filename,
                verbose,
                document_id,
                category,
                institution_id,
                user_id,
                exclude_document_ids,
            )

    def _process_text_locked(
        self,
        text: str,
        filename: str,
        verbose: bool = True,
        document_id: Optional[int] = None,
        category: Optional[str] = None,
        institution_id: Optional[str] = None,
        user_id: Optional[str] = None,
        exclude_document_ids: Optional[List[int]] = None,
    ) -> Dict:
        norm_category = normalize_category_slug(category or "uncategorized")
        norm_institution = (institution_id or "").strip() or None
        norm_user_id = (user_id or "").strip() or None
        # Нормализация до нарезки (задача 10): тривиальные различия форматов
        # одной работы не должны менять границы чанков и показатели.
        actual_text = normalize_text(self.extract_real_text(text))
        chunks = self.text_splitter.split_text(actual_text)
        if not chunks:
            # Нет текста для проверки (пустой документ, одни пробелы).
            raise AnalysisError(AnalysisErrorCode.NO_TEXT, detail="no chunks after split")
        if verbose:
            print(f"Analysis {filename[:20]}... (chunks: {len(chunks)})")
        ai_windows = self.ai_splitter.split_text(actual_text) if self.ai_config.enabled else []
        ai_texts = [
            chunk
            for chunk in ai_windows
            if len(chunk.strip()) > self.ai_config.min_chunk_chars
        ]
        ai_pairs = []
        if ai_texts and self.ai_model is not None:
            for chunk, score in zip(ai_texts, self._analyze_ai_chunks(ai_texts)):
                if score is not None:
                    ai_pairs.append((chunk, score))
        ai_stats = [score for _, score in ai_pairs]
        # null, а не 0: «детектор выключен» и «нет анализируемого текста» — это
        # отсутствие сигнала, а не «точно человек» (задача 9).
        if not self.ai_config.enabled or not ai_stats:
            ai_percent = None
        else:
            ai_percent = ai_percent_from_scores(ai_stats, self.ai_config)
        # Локализация: по-чанковые оценки, чтобы было видно, КАКИЕ фрагменты
        # выглядят как ИИ, а не только одно число по документу (задача 9).
        ai_fragments = [
            {
                "text": chunk[:300],
                "d_score": round(score, 4),
                "ai_probability": sigmoid_ai_percent(
                    score, self.ai_config.d0, self.ai_config.scale
                ),
            }
            for chunk, score in ai_pairs[: self.report_max_fragments]
        ]
        self._log_ai(document_id, filename, len(ai_windows), ai_pairs, ai_percent)
        vectors = self.encoder.encode(
            chunks, batch_size=16, show_progress_bar=False, normalize_embeddings=True
        ).tolist()
        sparse_vectors = [self._lexical_sparse(chunk) for chunk in chunks]
        search_filter = self._build_comparison_filter(
            document_id,
            filename,
            norm_category,
            norm_institution,
            user_id=norm_user_id,
            exclude_document_ids=exclude_document_ids,
        )
        if document_id is not None:
            self_condition = FieldCondition(
                key="document_id", match=MatchValue(value=document_id)
            )
        else:
            self_condition = FieldCondition(
                key="filename", match=MatchValue(value=filename)
            )
        # Гибридный поиск: на каждый чанк — dense-запрос (семантика, рерайт) и
        # lexical-запрос (дословное совпадение). Чанк считается заимствованным,
        # если сработал хотя бы один сигнал. with_payload=True: нужен payload
        # найденного чанка, чтобы атрибутировать совпадение к работе-источнику.
        requests = []
        request_map = []  # (индекс чанка, тип сигнала) для разбора батч-ответа
        for i, vec in enumerate(vectors):
            requests.append(
                QueryRequest(
                    query=vec,
                    using=self.DENSE_VECTOR,
                    limit=self.report_top_k,
                    score_threshold=self.qdrant_score_threshold,
                    with_payload=True,
                    filter=search_filter,
                )
            )
            request_map.append((i, "semantic"))
        for i, sparse in enumerate(sparse_vectors):
            if sparse is not None:
                requests.append(
                    QueryRequest(
                        query=sparse,
                        using=self.LEXICAL_VECTOR,
                        limit=self.report_top_k,
                        score_threshold=self.lexical_score_threshold,
                        with_payload=True,
                        filter=search_filter,
                    )
                )
                request_map.append((i, "lexical"))
        try:
            batch_search_results = self.qdrant.query_batch_points(
                collection_name=self.collection_name, requests=requests
            )
        except UnexpectedResponse as e:
            # Most common production issue: collection was deleted or never created.
            if getattr(e, "status_code", None) == 404:
                self._ensure_qdrant_collection()
                batch_search_results = self.qdrant.query_batch_points(
                    collection_name=self.collection_name, requests=requests
                )
            else:
                raise
        # Все совпадения каждого чанка по каждому сигналу (до report_top_k).
        chunk_hits: List[Dict[str, list]] = [{} for _ in chunks]
        for (chunk_idx, kind), qdrant_res in zip(request_map, batch_search_results):
            if qdrant_res.points:
                chunk_hits[chunk_idx][kind] = list(qdrant_res.points)
        plagiarized_chunks = 0
        points_to_insert = []
        # Аггрегация совпадений по работам-источникам: dense ловит глубокий рерайт
        # (смысл совпал, слов нет), lexical — дословное копирование; оба указывают,
        # ИЗ КАКОЙ работы заимствовано.
        matches_by_source: Dict[str, dict] = {}
        for chunk_idx, (chunk_text, vector, sparse) in enumerate(
            zip(chunks, vectors, sparse_vectors)
        ):
            hits = chunk_hits[chunk_idx]
            if any(hits.values()):
                plagiarized_chunks += 1
            # Сгруппировать все совпадения этого чанка по источнику: один чанк
            # может совпасть с НЕСКОЛЬКИМИ работами, и по обоим сигналам.
            per_source: Dict[str, dict] = {}
            for kind in ("semantic", "lexical"):
                for top in hits.get(kind, []):
                    payload = top.payload or {}
                    src_key = str(
                        payload.get("document_id")
                        if payload.get("document_id") is not None
                        else payload.get("filename") or "unknown"
                    )
                    entry = per_source.setdefault(
                        src_key,
                        {
                            "document_id": payload.get("document_id"),
                            "filename": payload.get("filename"),
                            "semantic_score": 0.0,
                            "lexical_score": 0.0,
                            "source_text": "",
                        },
                    )
                    score = float(top.score or 0.0)
                    if kind == "semantic":
                        entry["semantic_score"] = max(entry["semantic_score"], score)
                    else:
                        entry["lexical_score"] = max(entry["lexical_score"], score)
                    matched_text = payload.get("text")
                    if isinstance(matched_text, str) and matched_text and not entry["source_text"]:
                        entry["source_text"] = matched_text

            # Записать фрагмент-совпадение в каждый задетый источник (задача 1:
            # какой фрагмент работы с каким фрагментом источника совпал).
            for src_key, entry in per_source.items():
                agg = matches_by_source.setdefault(
                    src_key,
                    {
                        "document_id": entry["document_id"],
                        "filename": entry["filename"],
                        "matched_chunks": 0,
                        "max_score": 0.0,
                        "max_lexical_score": 0.0,
                        "paraphrase_score": 0.0,
                        "sample": "",
                        "fragments": [],
                        "_best_score": 0.0,
                    },
                )
                agg["matched_chunks"] += 1  # один инкремент на пару чанк-источник
                sem = round(entry["semantic_score"], 4)
                lex = round(entry["lexical_score"], 4)
                par = round(paraphrase_similarity(chunk_text, entry["source_text"]), 4) \
                    if entry["source_text"] else 0.0
                agg["max_score"] = max(agg["max_score"], sem)
                agg["max_lexical_score"] = max(agg["max_lexical_score"], lex)
                agg["paraphrase_score"] = max(agg["paraphrase_score"], par)
                frag_type = classify_match(
                    lex, par, self.lexical_score_threshold, self.paraphrase_threshold
                )
                if len(agg["fragments"]) < self.report_max_fragments:
                    agg["fragments"].append({
                        "work_text": chunk_text[:500],
                        "source_text": (entry["source_text"] or "")[:500],
                        "match_type": frag_type,
                        "semantic_score": sem,
                        "lexical_score": lex,
                        "paraphrase_score": par,
                    })
                # Пример источника — самый сильный фрагмент по любому сигналу.
                best = max(sem, lex)
                if best > agg["_best_score"]:
                    agg["_best_score"] = best
                    agg["sample"] = chunk_text[:300]
            point_vector: Dict[str, object] = {self.DENSE_VECTOR: vector}
            if sparse is not None:
                point_vector[self.LEXICAL_VECTOR] = sparse
            points_to_insert.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=point_vector,
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "text": chunk_text,
                        "category": norm_category,
                        "institution_id": norm_institution,
                        "user_id": norm_user_id,
                    },
                )
            )
        if points_to_insert:
            # Переиндексация без дубликатов: старые чанки этого документа
            # удаляются, иначе каждый повторный анализ раздувает коллекцию.
            # Удаляем только по document_id — по filename опасно (несколько
            # разных документов могут разделять имя вроде "document.txt").
            if document_id is not None:
                self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(must=[self_condition])
                    ),
                )
            self.qdrant.upsert(
                collection_name=self.collection_name, points=points_to_insert
            )
        plagiarism_percent = round((plagiarized_chunks / len(chunks)) * 100, 2)
        semantic_matches: List[dict] = sorted(
            matches_by_source.values(),
            key=lambda m: (m["matched_chunks"], max(m["max_score"], m["max_lexical_score"])),
            reverse=True,
        )
        # Каскадная классификация источников (единственный источник истины):
        # exact → paraphrase → semantic, см. cascade.py.
        by_type = {"exact": 0, "paraphrase": 0, "semantic": 0}
        for match in semantic_matches:
            match.pop("_best_score", None)
            match["match_type"] = classify_match(
                match["max_lexical_score"],
                match["paraphrase_score"],
                self.lexical_score_threshold,
                self.paraphrase_threshold,
            )
            by_type[match["match_type"]] += 1
        return {
            "plagiarism_percent": plagiarism_percent,
            "ai_percent": ai_percent,
            "ai_fragments": ai_fragments,
            "parsed_text": actual_text,
            "semantic_matches": semantic_matches,
            "by_type": by_type,
        }

"""Fast-DetectGPT scoring knobs. No PyTorch — safe to unit-test.

Defaults match the previous hardcoded behaviour:
window 500/50, skip chunks of 100 characters or shorter, 1024 tokens,
mean of D, then a sigmoid with d0=1.0 and scale=1.5.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

AGGREGATIONS = ("mean", "median", "max", "top_share")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _warn(message: str) -> None:
    print(f"AI config: {message}", flush=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    _warn(f"{name}={raw!r} is not a boolean, using {default}")
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip().replace(",", "."))
    except ValueError:
        _warn(f"{name}={raw!r} is not a number, using {default}")
        return default


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        _warn(f"{name}={raw!r} is not an integer, using {default}")
        return default
    if value < 1:
        _warn(f"{name}={value} must be >= 1, using {default}")
        return default
    return value


@dataclass(frozen=True)
class AiConfig:
    enabled: bool
    chunk_size: int
    chunk_overlap: int
    min_chunk_chars: int
    max_tokens: int
    aggregation: str
    top_share_threshold: float
    d0: float
    scale: float
    batch_size: int
    model: str
    debug_log: bool
    debug_chunks: bool


def load_ai_config() -> AiConfig:
    d0 = _env_float("AI_DETECT_D0", 1.0)
    scale = _env_float("AI_DETECT_SCALE", 1.5)
    chunk_size = _env_positive_int("AI_CHUNK_SIZE", 500)
    overlap_raw = os.getenv("AI_CHUNK_OVERLAP")
    if overlap_raw is None or overlap_raw.strip() == "":
        chunk_overlap = 50
    elif overlap_raw.strip() == "0":
        chunk_overlap = 0
    else:
        chunk_overlap = _env_positive_int("AI_CHUNK_OVERLAP", 50)
    if chunk_overlap >= chunk_size:
        fallback = max(0, chunk_size // 5)
        _warn(f"AI_CHUNK_OVERLAP={chunk_overlap} must be < AI_CHUNK_SIZE={chunk_size}, using {fallback}")
        chunk_overlap = fallback

    aggregation_raw = os.getenv("AI_AGGREGATION", "").strip().lower()
    if not aggregation_raw:
        aggregation = "mean"
    elif aggregation_raw not in AGGREGATIONS:
        _warn(f"AI_AGGREGATION={aggregation_raw!r} is unknown, using mean")
        aggregation = "mean"
    else:
        aggregation = aggregation_raw

    threshold_raw = os.getenv("AI_TOP_SHARE_THRESHOLD")
    if threshold_raw is None or threshold_raw.strip() == "":
        top_share_threshold = d0
    else:
        top_share_threshold = _env_float("AI_TOP_SHARE_THRESHOLD", d0)

    model = os.getenv("AI_MODEL", "sberbank-ai/rugpt3small_based_on_gpt2").strip()
    if not model:
        model = "sberbank-ai/rugpt3small_based_on_gpt2"

    return AiConfig(
        enabled=_env_bool("AI_ENABLED", True),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=_env_positive_int("AI_MIN_CHUNK_CHARS", 100),
        max_tokens=_env_positive_int("AI_MAX_TOKENS", 1024),
        aggregation=aggregation,
        top_share_threshold=top_share_threshold,
        d0=d0,
        scale=scale,
        batch_size=_env_positive_int("AI_DETECT_BATCH_SIZE", 8),
        model=model,
        debug_log=_env_bool("AI_DEBUG_LOG", False),
        debug_chunks=_env_bool("AI_DEBUG_LOG_CHUNKS", False),
    )


def score_stats(scores: list[float]) -> dict[str, float | None]:
    if not scores:
        return {"mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(scores)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "mean": sum(scores) / n,
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
    }


def sigmoid_ai_percent(summary: float, d0: float, scale: float) -> float:
    x = scale * (summary - d0)
    if x >= 60:
        probability = 1.0
    elif x <= -60:
        probability = 0.0
    else:
        probability = 1.0 / (1.0 + math.exp(-x))
    return round(probability * 100, 2)


def _summary_d(scores: list[float], aggregation: str) -> float:
    stats = score_stats(scores)
    if aggregation == "median":
        return float(stats["median"])
    if aggregation == "max":
        return float(stats["max"])
    return float(stats["mean"])


def ai_percent_from_scores(scores: list[float], config: AiConfig) -> float:
    """Map per-window D values to one percent. Empty input is 0."""
    if not scores:
        return 0.0
    if config.aggregation == "top_share":
        hits = sum(1 for value in scores if value >= config.top_share_threshold)
        return round(hits / len(scores) * 100, 2)
    return sigmoid_ai_percent(_summary_d(scores, config.aggregation), config.d0, config.scale)


def format_startup_line(config: AiConfig) -> str:
    return (
        "AI config: "
        f"enabled={config.enabled} model={config.model} "
        f"chunk={config.chunk_size}/{config.chunk_overlap} "
        f"min_chars={config.min_chunk_chars} max_tokens={config.max_tokens} "
        f"aggregation={config.aggregation} d0={config.d0} scale={config.scale} "
        f"top_share_threshold={config.top_share_threshold} batch={config.batch_size} "
        f"debug={config.debug_log} debug_chunks={config.debug_chunks}"
    )

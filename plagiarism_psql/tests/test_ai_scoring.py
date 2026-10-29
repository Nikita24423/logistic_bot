from ai_scoring import ai_percent_from_scores, load_ai_config, score_stats, sigmoid_ai_percent


def _clear_ai_env(monkeypatch):
    for name in (
        "AI_ENABLED",
        "AI_CHUNK_SIZE",
        "AI_CHUNK_OVERLAP",
        "AI_MIN_CHUNK_CHARS",
        "AI_MAX_TOKENS",
        "AI_AGGREGATION",
        "AI_TOP_SHARE_THRESHOLD",
        "AI_DETECT_D0",
        "AI_DETECT_SCALE",
        "AI_DETECT_BATCH_SIZE",
        "AI_MODEL",
        "AI_DEBUG_LOG",
        "AI_DEBUG_LOG_CHUNKS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_match_previous_behaviour(monkeypatch):
    _clear_ai_env(monkeypatch)
    config = load_ai_config()
    assert config.enabled is True
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50
    assert config.min_chunk_chars == 100
    assert config.max_tokens == 1024
    assert config.aggregation == "mean"
    assert config.d0 == 1.0
    assert config.scale == 1.5
    assert config.top_share_threshold == 1.0
    assert config.batch_size == 8
    assert config.debug_log is False
    assert config.debug_chunks is False


def test_invalid_values_fall_back(monkeypatch, capsys):
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("AI_DETECT_D0", "nope")
    monkeypatch.setenv("AI_AGGREGATION", "average")
    monkeypatch.setenv("AI_CHUNK_OVERLAP", "900")
    monkeypatch.setenv("AI_ENABLED", "maybe")
    config = load_ai_config()
    assert config.d0 == 1.0
    assert config.aggregation == "mean"
    assert config.chunk_overlap < config.chunk_size
    assert config.enabled is True
    logged = capsys.readouterr().out
    assert "AI_DETECT_D0" in logged
    assert "AI_AGGREGATION" in logged
    assert "AI_CHUNK_OVERLAP" in logged


def test_sigmoid_at_neutral_d_is_about_18_percent():
    # scale 1.5, d0 1.0, D = 0 → 1 / (1 + exp(1.5)) ≈ 18.24
    assert sigmoid_ai_percent(0.0, d0=1.0, scale=1.5) == 18.24


def test_mean_aggregation_uses_sigmoid(monkeypatch):
    _clear_ai_env(monkeypatch)
    config = load_ai_config()
    assert ai_percent_from_scores([0.0, 0.0], config) == sigmoid_ai_percent(0.0, 1.0, 1.5)
    assert ai_percent_from_scores([], config) == 0.0


def test_max_median_and_top_share(monkeypatch):
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("AI_AGGREGATION", "max")
    config = load_ai_config()
    assert ai_percent_from_scores([-2.0, 1.0], config) == sigmoid_ai_percent(1.0, 1.0, 1.5)

    monkeypatch.setenv("AI_AGGREGATION", "median")
    config = load_ai_config()
    assert ai_percent_from_scores([-2.0, 0.0, 4.0], config) == sigmoid_ai_percent(0.0, 1.0, 1.5)

    monkeypatch.setenv("AI_AGGREGATION", "top_share")
    monkeypatch.setenv("AI_TOP_SHARE_THRESHOLD", "1.0")
    config = load_ai_config()
    assert ai_percent_from_scores([-1.0, 1.0, 2.0, 0.5], config) == 50.0


def test_score_stats():
    stats = score_stats([1.0, 3.0, 5.0])
    assert stats["mean"] == 3.0
    assert stats["median"] == 3.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0

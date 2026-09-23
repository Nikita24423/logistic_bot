from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("omnot.audit")


class RequestLogger:
    def __init__(self, log_dir: str | Path = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_request(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        response: dict[str, Any] | None = None,
        error: str | None = None,
        latency_ms: float = 0,
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "message_count": len(messages),
            "latency_ms": round(latency_ms, 2),
        }

        if response:
            usage = response.get("usage", {})
            entry["tokens"] = {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
            }

        if error:
            entry["error"] = error

        logger.info(json.dumps(entry))

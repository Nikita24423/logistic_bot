from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProviderConfig:
    name: str
    api_key_env: str
    base_url: str
    models: list[str]
    rpm_limit: int = 0
    rpd_limit: int = 0
    tpm_limit: int = 0
    priority: int = 0
    enabled: bool = True
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    @property
    def is_configured(self) -> bool:
        return self.api_key is not None and self.enabled


@dataclass
class OmnotConfig:
    providers: list[ProviderConfig] = field(default_factory=list)
    fallback_strategy: str = "round_robin"
    max_retries: int = 3
    timeout: float = 30.0
    log_level: str = "INFO"

    @classmethod
    def from_file(cls, path: str | Path) -> OmnotConfig:
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OmnotConfig:
        providers = [
            ProviderConfig(**p) for p in data.get("providers", [])
        ]
        return cls(
            providers=providers,
            fallback_strategy=data.get("fallback_strategy", "round_robin"),
            max_retries=data.get("max_retries", 3),
            timeout=data.get("timeout", 30.0),
            log_level=data.get("log_level", "INFO"),
        )

    def get_active_providers(self) -> list[ProviderConfig]:
        return sorted(
            [p for p in self.providers if p.is_configured],
            key=lambda p: p.priority,
        )

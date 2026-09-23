from __future__ import annotations

from omnot.config import ProviderConfig

PROVIDER_PRESETS: dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        name="groq",
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        models=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ],
        rpm_limit=30,
        rpd_limit=14400,
        tpm_limit=131072,
        priority=1,
    ),
    "together": ProviderConfig(
        name="together",
        api_key_env="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        models=[
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ],
        rpm_limit=60,
        priority=2,
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        models=[
            "meta-llama/llama-3.1-70b-instruct:free",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
        ],
        rpm_limit=20,
        rpd_limit=200,
        priority=3,
    ),
    "cerebras": ProviderConfig(
        name="cerebras",
        api_key_env="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        models=[
            "llama-3.3-70b",
            "llama-3.1-8b",
        ],
        rpm_limit=30,
        tpm_limit=60000,
        priority=4,
    ),
    "sambanova": ProviderConfig(
        name="sambanova",
        api_key_env="SAMBANOVA_API_KEY",
        base_url="https://api.sambanova.ai/v1",
        models=[
            "Meta-Llama-3.1-405B-Instruct",
            "Meta-Llama-3.1-70B-Instruct",
            "Meta-Llama-3.1-8B-Instruct",
        ],
        rpm_limit=10,
        priority=5,
    ),
}


def get_preset(name: str) -> ProviderConfig | None:
    return PROVIDER_PRESETS.get(name.lower())

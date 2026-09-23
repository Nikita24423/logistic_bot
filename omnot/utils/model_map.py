from __future__ import annotations

MODEL_EQUIVALENTS: dict[str, list[str]] = {
    "gpt-4": [
        "llama-3.3-70b-versatile",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "Meta-Llama-3.1-405B-Instruct",
    ],
    "gpt-3.5-turbo": [
        "llama-3.1-8b-instant",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "gemma2-9b-it",
        "mistralai/mistral-7b-instruct:free",
    ],
    "claude-3-opus": [
        "Meta-Llama-3.1-405B-Instruct",
    ],
    "claude-3-sonnet": [
        "llama-3.3-70b-versatile",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    ],
    "claude-3-haiku": [
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
    ],
}


def find_equivalent(model: str, available_models: list[str]) -> str | None:
    for key, equivalents in MODEL_EQUIVALENTS.items():
        if model == key or model in equivalents:
            for eq in equivalents:
                if eq in available_models:
                    return eq
    return None

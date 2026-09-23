from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

import httpx

from omnot.config import OmnotConfig, ProviderConfig
from omnot.rate_limiter import RateLimiter

logger = logging.getLogger("omnot")


class ProviderError(Exception):
    def __init__(self, provider: str, status: int, message: str):
        self.provider = provider
        self.status = status
        super().__init__(f"[{provider}] {status}: {message}")


class AllProvidersExhausted(Exception):
    pass


class OmnotRouter:
    def __init__(self, config: OmnotConfig):
        self.config = config
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._client = httpx.AsyncClient(timeout=config.timeout)

        for provider in config.get_active_providers():
            self._rate_limiters[provider.name] = RateLimiter(
                rpm=provider.rpm_limit,
                rpd=provider.rpd_limit,
                tpm=provider.tpm_limit,
            )

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        providers = self.config.get_active_providers()
        last_error: Exception | None = None

        for provider in providers:
            if model and model not in provider.models:
                continue

            limiter = self._rate_limiters.get(provider.name)
            if limiter and not limiter.allow():
                logger.info("Rate limit hit for %s, trying next", provider.name)
                continue

            try:
                response = await self._call_provider(
                    provider, messages, model, **kwargs
                )
                if limiter:
                    tokens = response.get("usage", {}).get("total_tokens", 0)
                    limiter.record(tokens=tokens)
                return response
            except ProviderError as e:
                last_error = e
                if e.status == 429:
                    logger.warning("429 from %s, rotating", provider.name)
                    if limiter:
                        limiter.mark_exhausted()
                    continue
                logger.error("Error from %s: %s", provider.name, e)
                continue

        raise AllProvidersExhausted(
            f"No providers available. Last error: {last_error}"
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        providers = self.config.get_active_providers()

        for provider in providers:
            if model and model not in provider.models:
                continue

            limiter = self._rate_limiters.get(provider.name)
            if limiter and not limiter.allow():
                continue

            try:
                async for chunk in self._stream_provider(
                    provider, messages, model, **kwargs
                ):
                    yield chunk
                return
            except ProviderError as e:
                if e.status == 429:
                    logger.warning("429 from %s during stream, rotating", provider.name)
                    if limiter:
                        limiter.mark_exhausted()
                    continue
                raise

        raise AllProvidersExhausted("No providers available for streaming")

    async def _call_provider(
        self,
        provider: ProviderConfig,
        messages: list[dict[str, str]],
        model: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        url = f"{provider.base_url}/chat/completions"
        target_model = model or provider.models[0]

        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
            **provider.headers,
        }

        payload = {
            "model": target_model,
            "messages": messages,
            **kwargs,
        }

        response = await self._client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise ProviderError(
                provider.name, response.status_code, response.text
            )

        return response.json()

    async def _stream_provider(
        self,
        provider: ProviderConfig,
        messages: list[dict[str, str]],
        model: str | None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        import json

        url = f"{provider.base_url}/chat/completions"
        target_model = model or provider.models[0]

        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
            **provider.headers,
        }

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        async with self._client.stream(
            "POST", url, json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise ProviderError(
                    provider.name, response.status_code, body.decode()
                )

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                yield json.loads(data)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OmnotRouter:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def status(self) -> dict[str, Any]:
        result = {}
        for provider in self.config.get_active_providers():
            limiter = self._rate_limiters.get(provider.name)
            result[provider.name] = {
                "models": provider.models,
                "priority": provider.priority,
                "rate_limit": limiter.status() if limiter else "unlimited",
            }
        return result

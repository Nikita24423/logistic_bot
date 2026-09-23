from __future__ import annotations

import json
import logging
from typing import Any

from omnot.config import OmnotConfig
from omnot.router import OmnotRouter

logger = logging.getLogger("omnot.server")

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    HAS_STARLETTE = True
except ImportError:
    HAS_STARLETTE = False


def create_app(config: OmnotConfig | None = None) -> Any:
    if not HAS_STARLETTE:
        raise ImportError(
            "starlette is required for the server. "
            "Install with: pip install omnot[server]"
        )

    if config is None:
        config = OmnotConfig.from_file("omnot.json")

    router = OmnotRouter(config)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "providers": router.status()})

    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        messages = body.get("messages", [])
        model = body.get("model")
        stream = body.get("stream", False)

        kwargs = {
            k: v
            for k, v in body.items()
            if k not in ("messages", "model", "stream")
        }

        if stream:
            async def generate():
                async for chunk in router.chat_stream(
                    messages, model=model, **kwargs
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(), media_type="text/event-stream"
            )

        response = await router.chat(messages, model=model, **kwargs)
        return JSONResponse(response)

    async def models(request: Request) -> JSONResponse:
        all_models = []
        for provider in config.get_active_providers():
            for model in provider.models:
                all_models.append({
                    "id": model,
                    "object": "model",
                    "owned_by": provider.name,
                })
        return JSONResponse({"object": "list", "data": all_models})

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/models", models),
        ],
    )

    return app

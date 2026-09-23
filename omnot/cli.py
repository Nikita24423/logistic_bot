from __future__ import annotations

import argparse
import asyncio
import json
import sys

from omnot.config import OmnotConfig
from omnot.providers.registry import PROVIDER_PRESETS
from omnot.router import OmnotRouter


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="omnot",
        description="Omnot — роутер к AI-провайдерам с автоматическим переключением",
    )
    subparsers = parser.add_subparsers(dest="command")

    chat_parser = subparsers.add_parser("chat", help="Отправить сообщение")
    chat_parser.add_argument("message", help="Текст сообщения")
    chat_parser.add_argument("-m", "--model", help="Модель (необязательно)")
    chat_parser.add_argument(
        "-c", "--config", default="omnot.json", help="Путь к конфигурации"
    )

    status_parser = subparsers.add_parser("status", help="Статус провайдеров")
    status_parser.add_argument(
        "-c", "--config", default="omnot.json", help="Путь к конфигурации"
    )

    subparsers.add_parser("providers", help="Список доступных пресетов")

    init_parser = subparsers.add_parser("init", help="Создать конфигурацию")
    init_parser.add_argument(
        "-o", "--output", default="omnot.json", help="Путь для сохранения"
    )
    init_parser.add_argument(
        "-p", "--providers", nargs="+",
        choices=list(PROVIDER_PRESETS.keys()),
        default=list(PROVIDER_PRESETS.keys()),
        help="Провайдеры для включения",
    )

    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(_chat(args))
    elif args.command == "status":
        _status(args)
    elif args.command == "providers":
        _list_providers()
    elif args.command == "init":
        _init_config(args)
    else:
        parser.print_help()


async def _chat(args: argparse.Namespace) -> None:
    config = OmnotConfig.from_file(args.config)
    async with OmnotRouter(config) as router:
        response = await router.chat(
            messages=[{"role": "user", "content": args.message}],
            model=args.model,
        )
        content = response["choices"][0]["message"]["content"]
        print(content)


def _status(args: argparse.Namespace) -> None:
    config = OmnotConfig.from_file(args.config)
    router = OmnotRouter(config)
    status = router.status()
    print(json.dumps(status, indent=2, ensure_ascii=False))


def _list_providers() -> None:
    for name, preset in PROVIDER_PRESETS.items():
        configured = "+" if preset.is_configured else "-"
        models = ", ".join(preset.models[:2])
        if len(preset.models) > 2:
            models += f" (+{len(preset.models) - 2})"
        print(f"  [{configured}] {name:12s}  {models}")


def _init_config(args: argparse.Namespace) -> None:
    providers = []
    for name in args.providers:
        preset = PROVIDER_PRESETS[name]
        providers.append({
            "name": preset.name,
            "api_key_env": preset.api_key_env,
            "base_url": preset.base_url,
            "models": preset.models,
            "rpm_limit": preset.rpm_limit,
            "rpd_limit": preset.rpd_limit,
            "tpm_limit": preset.tpm_limit,
            "priority": preset.priority,
            "enabled": True,
        })

    config = {
        "providers": providers,
        "fallback_strategy": "round_robin",
        "max_retries": 3,
        "timeout": 30.0,
        "log_level": "INFO",
    }

    with open(args.output, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Конфигурация сохранена: {args.output}")


if __name__ == "__main__":
    main()

# Omnot — роутер к AI-провайдерам

Автоматическое переключение между бесплатными AI-провайдерами при достижении лимитов.

## Зачем

Бесплатные тарифы AI-провайдеров имеют жёсткие ограничения: 30 RPM у Groq, 200 RPD у OpenRouter, и т.д.
Omnot объединяет несколько провайдеров в один OpenAI-совместимый эндпоинт и автоматически переключается на следующий при ошибке 429.

## Провайдеры

| Провайдер  | Модели                          | Бесплатный лимит           | Приоритет |
|------------|--------------------------------|---------------------------|-----------|
| Groq       | Llama 3.3 70B, Llama 3.1 8B   | 30 RPM / 14400 RPD        | 1         |
| Together   | Llama 3.1 70B/8B Turbo         | 60 RPM                    | 2         |
| OpenRouter | Llama 3.1 70B, Gemma 2 9B     | 20 RPM / 200 RPD          | 3         |
| Cerebras   | Llama 3.3 70B, 3.1 8B         | 30 RPM / 60K TPM          | 4         |
| SambaNova  | Llama 3.1 405B/70B/8B         | 10 RPM                    | 5         |

## Быстрый старт

```bash
pip install -e .
omnot init                        # создаёт omnot.json
export GROQ_API_KEY=gsk_...
export TOGETHER_API_KEY=...
omnot chat "Привет, мир"          # авто-роутинг
omnot status                      # статус провайдеров
```

## Как сервер (OpenAI-совместимый)

```bash
pip install -e ".[server]"
uvicorn omnot.server:create_app --factory --port 8080
```

Эндпоинты:
- `POST /v1/chat/completions` — чат (stream/non-stream)
- `GET  /v1/models` — список доступных моделей
- `GET  /health` — статус

## Использование в коде

```python
import asyncio
from omnot import OmnotRouter, OmnotConfig

async def main():
    config = OmnotConfig.from_file("omnot.json")
    async with OmnotRouter(config) as router:
        response = await router.chat([
            {"role": "user", "content": "Объясни квантовые вычисления"}
        ])
        print(response["choices"][0]["message"]["content"])

asyncio.run(main())
```

## Конфигурация

Скопируйте `omnot.example.json` → `omnot.json` и заполните API-ключи через переменные окружения.

## Тесты

```bash
pip install -e ".[dev]"
pytest
```

---

## Аналитика инструментов экосистемы

| Инструмент    | Назначение                      | Оценка |
|---------------|--------------------------------|--------|
| **Omnot**     | Роутер к провайдерам            | 🔴 Работу делает не Claude, риск блокировки |
| PROMts        | 10K промптов, MCP-сервер        | 🔴 Неизвестный MCP получает доступ к среде |
| Karpathy Skill| Файл правил-гайдрейлов          | 🟢 Снижает ошибки |
| Граф кодовой базы | Индексация проекта          | 🟢 Здравая концепция |
| Память сессий | Сохранение контекста            | 🟢 Реальная потребность |
| Firecrawl     | Скрейпинг сайтов               | 🟡 Есть ограничения |
| Playwright    | Управление браузером            | 🟡 Хрупкий |
| Perplexity    | Веб-поиск с цитатами            | 🟡 У Claude уже есть поиск |

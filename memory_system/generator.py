"""Локальный генератор ответов поверх ``MemoryContext``.

Веса LLM не тащим в Python-зависимости: используется Ollama с OpenAI-compatible
``/api/chat``. Модель по умолчанию — ``qwen2.5:3b-instruct``: для RTX 4050
Laptop 6GB это лучший баланс качества на русском и умещаемости в VRAM.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from memory_system.context_builder import MemoryContext

SYSTEM_PROMPT = (
    "Ты — персональный ассистент пользователя. Отвечай по-русски, кратко и по делу. "
    "Используй только предоставленный контекст памяти. Если в контексте нет точного "
    "ответа, честно скажи, что пока не знаешь, и не угадывай и не выдумывай факты. "
    "На просьбу пересказать или перечислить "
    "известное выдавай все пункты из контекста списком, а не отвечай «не знаю»."
)


class GeneratorUnavailable(RuntimeError):
    """Локальная LLM недоступна: Ollama не запущен или модель не скачана."""


class AnswerGenerator(Protocol):
    """Минимальный интерфейс генератора: принимает chat-сообщения, отдаёт текст."""

    def respond(self, messages: list[dict[str, str]]) -> str:
        """Возвращает ответ модели на список ``{"role": ..., "content": ...}``."""
        ...


def build_messages(context: MemoryContext, query: str) -> list[dict[str, str]]:
    """Собирает chat-сообщения: системный промпт + память + вопрос."""
    memory = context.text.strip() or "Память пока пуста."
    user = (
        "Контекст памяти о пользователе:\n"
        f"{memory}\n\n"
        "Вопрос пользователя:\n"
        f"{query}\n\n"
        "Дай ответ, опираясь только на этот контекст."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class OllamaGenerator:
    """Генератор через локальный Ollama HTTP API."""

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        host: str = "http://localhost:11434",
        *,
        num_ctx: int = 4096,
        temperature: float = 0.2,
        timeout: int = 180,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout

    def respond(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise GeneratorUnavailable(
                f"Ollama вернул ошибку {exc.code}: {body}. "
                f"Проверьте, что модель «{self.model}» скачана: ollama pull {self.model}"
            ) from exc
        except urllib.error.URLError as exc:
            raise GeneratorUnavailable(
                "Ollama недоступен. Запустите `ollama serve` и убедитесь, что "
                f"модель «{self.model}» скачана (`ollama pull {self.model}`)."
            ) from exc
        content = data.get("message", {}).get("content", "")
        return str(content).strip()


class LlamaCppServerGenerator:
    """Генератор через локальный ``llama-server`` из llama.cpp (OpenAI-compatible)."""

    def __init__(
        self,
        model: str = "local",
        host: str = "http://127.0.0.1:8080",
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: int = 180,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def respond(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise GeneratorUnavailable(
                f"llama-server вернул ошибку {exc.code}: {body}. "
                "Проверьте, что сервер запущен и alias модели совпадает."
            ) from exc
        except urllib.error.URLError as exc:
            raise GeneratorUnavailable(
                "llama-server недоступен. Запустите бинарник llama.cpp с "
                f"моделью GGUF и --alias {self.model}."
            ) from exc
        choices = data.get("choices") or []
        if not choices:
            raise GeneratorUnavailable("llama-server вернул пустой choices")
        content = choices[0].get("message", {}).get("content", "")
        return str(content).strip()


__all__ = [
    "AnswerGenerator",
    "GeneratorUnavailable",
    "LlamaCppServerGenerator",
    "OllamaGenerator",
    "SYSTEM_PROMPT",
    "build_messages",
]

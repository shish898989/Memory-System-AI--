"""Конфигурация Memory System.

Все параметры имеют разумные значения по умолчанию; часть можно переопределить
переменными окружения, чтобы не менять код при экспериментах.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(slots=True)
class DecayConfig:
    """Параметры забывания нерелевантного."""

    # Веса формулы важности: recency + importance + relevance.
    w_recency: float = 0.40
    w_importance: float = 0.35
    w_relevance: float = 0.25
    # Через сколько дней «свежесть» факта падает вдвое.
    half_life_days: float = 14.0
    # Ниже этого score факт уходит в архив (не удаляется).
    archive_threshold: float = 0.18


@dataclass(slots=True)
class ConsolidationConfig:
    """Параметры объединения похожих фактов и разрешения противоречий."""

    # Косинусное сходство, выше которого два значения считаются одним фактом.
    merge_similarity: float = 0.90
    # Сколько приращений confidence даёт каждое повторное упоминание.
    confidence_step: float = 0.1
    # Максимальный confidence, чтобы факт не стал «абсолютной истиной».
    confidence_cap: float = 1.0
    # Политика противоречий: True — побеждает новое значение (recency wins).
    newest_wins: bool = True


@dataclass(slots=True)
class ContextConfig:
    """Параметры сборки контекста для ответа."""

    # Сколько эпизодических фактов максимум подмешивать к профилю.
    max_episodic_facts: int = 8
    # Бюджет контекста в «токенах» (приблизительно: 1 токен ≈ 4 символа).
    token_budget: int = 700
    # Сколько последних реплик диалога добавлять как рабочую память.
    recent_turns: int = 4


@dataclass(slots=True)
class MemoryConfig:
    """Корневая конфигурация памяти."""

    db_path: Path = Path("memory.db")
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    device: str | None = None
    # «sentence-transformers» — реальные эмбеддинги, «hashing» — офлайн-фолбэк.
    embedder: str = "sentence-transformers"
    # Вес важности предиката попадает в score забывания.
    subject: str = "user"
    # Локальная LLM для генерации ответа поверх MemoryContext.
    # backend: "ollama" (HTTP API Ollama) или "llamacpp" (llama-server из llama.cpp).
    llm_backend: str = "ollama"
    llm_model: str = "qwen2.5:3b-instruct"
    ollama_host: str = "http://localhost:11434"
    llamacpp_host: str = "http://127.0.0.1:8080"
    decay: DecayConfig = field(default_factory=DecayConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    context: ContextConfig = field(default_factory=ContextConfig)

    @classmethod
    def from_env(cls) -> MemoryConfig:
        """Собирает конфигурацию из переменных окружения MEMORY_*."""
        return cls(
            db_path=Path(os.getenv("MEMORY_DB", "memory.db")),
            embedding_model=os.getenv("MEMORY_MODEL", DEFAULT_EMBEDDING_MODEL),
            device=os.getenv("MEMORY_DEVICE") or None,
            embedder=os.getenv("MEMORY_EMBEDDER", "sentence-transformers"),
            subject=os.getenv("MEMORY_SUBJECT", "user"),
            llm_backend=os.getenv("MEMORY_LLM_BACKEND", "ollama"),
            llm_model=os.getenv("MEMORY_LLM_MODEL", "qwen2.5:3b-instruct"),
            ollama_host=os.getenv("MEMORY_OLLAMA_HOST")
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434",
            llamacpp_host=os.getenv("MEMORY_LLAMACPP_HOST")
            or os.getenv("LLAMACPP_HOST")
            or "http://127.0.0.1:8080",
        )

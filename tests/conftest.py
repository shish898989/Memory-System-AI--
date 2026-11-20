"""Общие фикстуры тестов.

Тесты используют офлайн-эмбеддер: быстро, детерминированно, без сети и GPU.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from memory_system.assistant import MemoryAssistant
from memory_system.config import MemoryConfig
from memory_system.embeddings import HashingEmbedder
from memory_system.models import utcnow
from memory_system.store import MemoryStore


@pytest.fixture()
def store() -> MemoryStore:
    store = MemoryStore(keep_in_memory=True)
    yield store
    store.close()


@pytest.fixture()
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture()
def assistant() -> MemoryAssistant:
    assistant = MemoryAssistant(
        MemoryConfig(), store=MemoryStore(keep_in_memory=True), embedder=HashingEmbedder()
    )
    yield assistant
    assistant.close()


@pytest.fixture()
def now():
    """Фиксированное «сейчас» — тесты не зависят от реального времени."""
    return utcnow()


@pytest.fixture()
def days_ago(now):
    def _days_ago(days: float):
        return now - timedelta(days=days)

    return _days_ago

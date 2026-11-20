"""Оркестратор памяти: единый цикл ingest → consolidate → recall.

``MemoryAssistant`` — та единственная точка, с которой работает приложение:

    assistant = MemoryAssistant(config)
    assistant.remember("Привет! Меня зовут Алексей, живу в Питере")
    context = assistant.recall("Какой город мне посоветуешь для переезда?")

Генерация ответа — опциональный слой поверх памяти: ``answer()`` отправляет
``MemoryContext`` в локальную LLM через Ollama. Без неё ``recall()`` возвращает
тот же контекст как готовый промпт для любой модели.
"""

from __future__ import annotations

import logging
from datetime import datetime

from pydantic import BaseModel, Field

from memory_system.config import MemoryConfig
from memory_system.consolidation import Consolidator
from memory_system.context_builder import ContextBuilder, MemoryContext
from memory_system.decay import Decayer
from memory_system.embeddings import Embedder, EmbedderUnavailable, build_embedder
from memory_system.extractor import Extractor, RuleBasedExtractor
from memory_system.generator import (
    AnswerGenerator,
    LlamaCppServerGenerator,
    OllamaGenerator,
    build_messages,
)
from memory_system.models import ConsolidationReport, Fact, utcnow
from memory_system.profile import ProfileBuilder, UserProfile
from memory_system.store import MemoryStore

logger = logging.getLogger(__name__)


class IngestResult(BaseModel):
    """Результат обработки одной реплики: что именно запомнилось."""

    episode_id: int
    facts: list[ConsolidationReport] = Field(default_factory=list)
    archived: list[int] = Field(default_factory=list)

    @property
    def actions(self) -> list[str]:
        return [report.action for report in self.facts]


class AnswerResult(BaseModel):
    """Ответ LLM вместе с контекстом памяти, который был ей передан."""

    query: str
    context: MemoryContext
    answer: str


class MemoryAssistant:
    """Память AI-ассистента: помнит, объединяет, забывает, строит профиль."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        store: MemoryStore | None = None,
        embedder: Embedder | None = None,
        extractor: Extractor | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.store = store or MemoryStore(
            None if self.config.db_path is None else self.config.db_path
        )
        self.embedder = embedder or self._make_embedder()
        self.extractor = extractor or RuleBasedExtractor()
        self.generator = generator

        self.consolidator = Consolidator(self.store, self.embedder, self.config.consolidation)
        self.decayer = Decayer(self.store, self.embedder, self.config.decay)
        self.profile_builder = ProfileBuilder(self.store)
        self.context_builder = ContextBuilder(
            self.store,
            self.embedder,
            self.decayer,
            self.profile_builder,
            self.config.context,
        )

    def _make_embedder(self) -> Embedder:
        """Создаёт эмбеддер, откатываясь на офлайн-вариант при недоступности модели."""
        try:
            return build_embedder(
                self.config.embedder, self.config.embedding_model, self.config.device
            )
        except EmbedderUnavailable as exc:
            logger.warning("Модель эмбеддингов недоступна (%s). Использую офлайн-эмбеддер.", exc)
            from memory_system.embeddings import HashingEmbedder

            return HashingEmbedder()

    def _make_generator(self) -> AnswerGenerator:
        """Создаёт генератор ответа по ``config.llm_backend``."""
        if self.config.llm_backend == "llamacpp":
            return LlamaCppServerGenerator(
                model=self.config.llm_model, host=self.config.llamacpp_host
            )
        if self.config.llm_backend == "ollama":
            return OllamaGenerator(model=self.config.llm_model, host=self.config.ollama_host)
        raise ValueError(f"Неизвестный llm_backend: {self.config.llm_backend}")

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------
    def remember(
        self, text: str, *, now: datetime | None = None, run_decay: bool = True
    ) -> IngestResult:
        """Обрабатывает реплику: извлекает факты, консолидирует, забывает лишнее."""
        now = now or utcnow()
        episode = self.store.add_episode(text, created_at=now)

        reports: list[ConsolidationReport] = []
        for extracted in self.extractor.extract(text):
            reports.append(
                self.consolidator.add(
                    extracted,
                    episode_id=episode.id,
                    now=now,
                    subject=self.config.subject,
                )
            )

        archived: list[int] = []
        if run_decay:
            archived = [int(fact.id) for fact in self.decayer.sweep(now=now) if fact.id is not None]

        return IngestResult(episode_id=int(episode.id or 0), facts=reports, archived=archived)

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------
    def recall(self, query: str, *, now: datetime | None = None) -> MemoryContext:
        """Собирает контекст под запрос: профиль + релевантные факты."""
        return self.context_builder.build(query, now=now, subject=self.config.subject)

    def answer(
        self,
        query: str,
        *,
        now: datetime | None = None,
        generator: AnswerGenerator | None = None,
    ) -> AnswerResult:
        """Отвечает на вопрос через локальную LLM, передавая ей контекст памяти."""
        context = self.recall(query, now=now)
        gen = generator or self.generator
        if gen is None:
            gen = self._make_generator()
            self.generator = gen
        answer = gen.respond(build_messages(context, query))
        return AnswerResult(query=query, context=context, answer=answer)

    def profile(self) -> UserProfile:
        """Текущий долгосрочный профиль пользователя."""
        return self.profile_builder.build(subject=self.config.subject)

    def facts(self, *, include_archived: bool = False) -> list[Fact]:
        return self.store.all_facts(include_archived=include_archived, subject=self.config.subject)

    def history(self) -> list[tuple[str, str, str]]:
        """История вытесненных значений: (слот, было, причина)."""
        return [
            (entry.predicate.value, entry.object, entry.reason)
            for entry in self.store.history_for()
        ]

    def stats(self) -> dict[str, int]:
        return {
            "episodes": self.store.count_episodes(),
            "active": self.store.count_facts(include_archived=False),
            "archived": self.store.count_facts(include_archived=True)
            - self.store.count_facts(include_archived=False),
            "history": self.store.count_history(),
        }

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> MemoryAssistant:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["AnswerResult", "IngestResult", "MemoryAssistant"]

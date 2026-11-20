"""Сборка контекста для ответа ассистента.

Контекст собирается в три слоя с разным приоритетом:

1. **профиль** — всегда, это ядро долгосрочной памяти;
2. **эпизодические факты** — top-K по релевантности к запросу с учётом свежести;
3. **последние реплики** — рабочая (кратковременная) память.

Жёсткий ``token_budget`` защищает от распухания промпта: слои добавляются
по убыванию важности, и как только бюджет исчерпан, набор обрезается.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field

from memory_system.config import ContextConfig
from memory_system.decay import Decayer
from memory_system.embeddings import Embedder, cosine
from memory_system.models import PREDICATE_LABELS, Fact, utcnow
from memory_system.profile import ProfileBuilder, UserProfile
from memory_system.store import MemoryStore

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Грубая оценка токенов: для русского ≈ 4 символа на токен."""
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


class MemoryContext(BaseModel):
    """Готовый контекст: то, что показывается модели и пользователю."""

    profile: UserProfile
    facts: list[Fact] = Field(default_factory=list)
    recent_turns: list[str] = Field(default_factory=list)
    text: str = ""
    tokens: int = 0


class ContextBuilder:
    """Собирает контекст ответа из профиля, фактов и последних реплик."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        decayer: Decayer,
        profile_builder: ProfileBuilder,
        config: ContextConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.decayer = decayer
        self.profile_builder = profile_builder
        self.config = config or ContextConfig()

    def build(
        self,
        query: str,
        *,
        now: datetime | None = None,
        subject: str = "user",
    ) -> MemoryContext:
        """Собирает контекст под конкретный запрос пользователя."""
        now = now or utcnow()
        profile = self.profile_builder.build(subject=subject)
        query_embedding = self.embedder.encode_one(query) if query.strip() else None

        ranked = self._rank_episodic_facts(
            query_embedding=query_embedding, now=now, subject=subject
        )
        recent_episodes = self.store.recent_episodes(self.config.recent_turns)
        # В чате текущая реплика уже сохранена как episode: не показываем её как
        # «прошлый ход», иначе модель начинает повторять вопрос.
        if recent_episodes and recent_episodes[-1].text.strip() == query.strip():
            recent_episodes = recent_episodes[:-1]
        recent = [f"Пользователь: {episode.text}" for episode in recent_episodes]

        lines: list[str] = []
        budget = self.config.token_budget

        # Слой 1: профиль — приоритет, но тоже в пределах бюджета.
        if not profile.is_empty():
            profile_lines = ["Долгосрочный профиль пользователя:"]
            profile_lines += [f"- {line}" for line in profile.as_lines()]
            lines = _consume(profile_lines, budget)

        # Слой 2: эпизодические факты по релевантности. Заголовок секции
        # учитывается в бюджете заранее, иначе он добавлял «лишние» токены.
        fact_candidates: list[tuple[str, Fact]] = []
        for fact, _score in ranked:
            if len(fact_candidates) >= self.config.max_episodic_facts:
                break
            label = PREDICATE_LABELS.get(fact.predicate, fact.predicate.value)
            fact_candidates.append((f"- {label}: {fact.object}", fact))

        selected: list[Fact] = []
        if fact_candidates:
            header = "Дополнительные факты:" if lines else "Известные факты о пользователе:"
            fact_lines = _consume(
                [line for line, _fact in fact_candidates], budget, prefix=[*lines, header]
            )
            selected = [fact for _line, fact in fact_candidates[: len(fact_lines)]]
            if fact_lines:
                lines.extend([header, *fact_lines])

        # Слой 3: последние реплики — только если остался бюджет.
        turn_header = "Последние реплики диалога:"
        turn_lines = _consume(recent, budget, prefix=[*lines, turn_header])
        if turn_lines:
            lines.extend([turn_header, *turn_lines])

        text = "\n".join(lines)
        # Отмечаем факты как использованные: свежесть влияет на забывание.
        self.store.touch([int(fact.id) for fact in selected if fact.id is not None], at=now)

        return MemoryContext(
            profile=profile,
            facts=selected,
            recent_turns=turn_lines,
            text=text,
            tokens=estimate_tokens(text),
        )

    def _rank_episodic_facts(
        self,
        *,
        query_embedding: list[float] | None,
        now: datetime,
        subject: str,
    ) -> list[tuple[Fact, float]]:
        """Ранжирует не-профильные факты по релевантности и свежести."""
        target = (
            np.asarray(query_embedding, dtype=np.float32) if query_embedding is not None else None
        )
        scored: list[tuple[Fact, float]] = []
        for fact in self.store.find_facts(subject=subject):
            if fact.predicate.is_profile_slot:
                continue
            relevance = 0.5
            if target is not None and fact.embedding is not None:
                own = np.asarray(fact.embedding, dtype=np.float32)
                if own.shape == target.shape:
                    relevance = max(0.0, cosine(target, own))
            decay_score = self.decayer.score(fact, now=now, query_embedding=query_embedding)
            # Релевантность запросу важнее «просто важности» слота.
            scored.append((fact, 0.6 * relevance + 0.4 * decay_score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


def _consume(candidates: list[str], budget: int, prefix: list[str] | None = None) -> list[str]:
    """Берёт строки, пока итоговый текст с учётом заголовков не превысит бюджет."""
    accepted: list[str] = []
    base = list(prefix or [])
    for line in candidates:
        if estimate_tokens("\n".join([*base, *accepted, line])) > budget:
            break
        accepted.append(line)
    return accepted


__all__ = ["ContextBuilder", "MemoryContext", "estimate_tokens"]

"""Забывание нерелевантного.

Формула взята из подхода Generative Agents и адаптирована под слоты:

    score = w_recency * recency + w_importance * importance + w_relevance * relevance

* ``recency`` — экспоненциальный распад по времени с заданным периодом полураспада;
* ``importance`` — базовая важность слота (имя важнее, чем «люблю кофе»);
* ``relevance`` — сходство факта с текущим запросом.

Факты профиля не забываются никогда: долгосрочный профиль — требование системы.
Забывание — это архивирование, а не удаление: факт остаётся в базе и в истории.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from memory_system.config import DecayConfig
from memory_system.embeddings import Embedder, cosine
from memory_system.models import Fact, utcnow
from memory_system.store import MemoryStore


def recency_score(age_days: float, half_life_days: float) -> float:
    """Экспоненциальный распад: 1.0 сейчас, 0.5 через период полураспада."""
    if half_life_days <= 0:
        return 0.0
    return math.pow(0.5, age_days / half_life_days)


class Decayer:
    """Решает, какие факты пора отправить в архив."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        config: DecayConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config or DecayConfig()

    def score(
        self,
        fact: Fact,
        *,
        now: datetime | None = None,
        query_embedding: list[float] | None = None,
    ) -> float:
        """Итоговый score факта: чем выше, тем нужнее он в активной памяти."""
        now = now or utcnow()
        recency = recency_score(fact.age_days(now), self.config.half_life_days)

        # Без запроса релевантность неизвестна и вклада не даёт. Если бы мы
        # подставляли «нейтральные» 0.5, ни один факт никогда не дошёл бы
        # до порога архивирования — забывание бы просто не работало.
        relevance = 0.0
        if query_embedding is not None and fact.embedding is not None:
            relevance = max(
                0.0,
                cosine(
                    np.asarray(query_embedding, dtype=np.float32),
                    np.asarray(fact.embedding, dtype=np.float32),
                ),
            )

        return (
            self.config.w_recency * recency
            + self.config.w_importance * fact.importance
            + self.config.w_relevance * relevance
        )

    def sweep(self, *, now: datetime | None = None) -> list[Fact]:
        """Архивирует факты ниже порога. Возвращает список забытых фактов."""
        now = now or utcnow()
        forgotten: list[Fact] = []
        for fact in self.store.find_facts(include_archived=False):
            if fact.predicate.is_profile_slot:
                continue  # профиль не забываем
            if self.score(fact, now=now) >= self.config.archive_threshold:
                continue
            archived = fact.model_copy(update={"archived": True, "updated_at": now})
            self.store.update_fact(archived)
            forgotten.append(archived)
        return forgotten


__all__ = ["Decayer", "recency_score"]

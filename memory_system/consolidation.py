"""Консолидация памяти: объединение похожих фактов и разрешение противоречий.

Ключевая идея — факт адресуется не текстом, а слотом ``(subject, predicate)``.
Это сводит три разные операции к простым и полностью детерминированным правилам:

* **дубликат** — тот же слот и то же (каноническое) значение → усиление;
* **похожий факт** — тот же слот, значение близко по эмбеддингу → объединение;
* **противоречие** — тот же однозначный слот, другое значение → вытеснение
  старого значения в ``fact_history``.

Многозначные предикаты (``likes``, ``dislikes``, ``stack``, ``goal``) не создают
противоречий: «люблю кофе» и «люблю чай» — два независимых факта.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from memory_system.config import ConsolidationConfig
from memory_system.embeddings import Embedder, cosine
from memory_system.extractor import ExtractedFact
from memory_system.models import ConsolidationReport, Fact, FactHistoryEntry, canonicalize, utcnow
from memory_system.store import MemoryStore


class Consolidator:
    """Записывает извлечённые факты в память, поддерживая её непротиворечивость."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        config: ConsolidationConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config or ConsolidationConfig()

    def add(
        self,
        extracted: ExtractedFact,
        *,
        episode_id: int | None = None,
        now: datetime | None = None,
        subject: str = "user",
    ) -> ConsolidationReport:
        """Добавляет факт, применяя дедупликацию, слияние и вытеснение."""
        now = now or utcnow()
        predicate = extracted.predicate
        canonical = extracted.canonical
        embedding = self.embedder.encode_one(extracted.object)

        existing = self.store.find_facts(predicate=predicate, subject=subject)

        # 1. Прямой дубликат по каноническому значению — усиливаем доверие.
        for fact in existing:
            if fact.value_key == canonical:
                return self._reinforce(fact, extracted, embedding, now=now)

        # 2. Похожее значение (по смыслу) — объединяем в один факт.
        similar = self._find_similar(existing, embedding)
        if similar is not None:
            return self._merge(similar, extracted, embedding, now=now)

        # 3. Противоречие: однозначный слот уже занят другим значением.
        if not predicate.is_multivalued and existing:
            return self._supersede_all(existing, extracted, embedding, episode_id, now)

        # 4. Новый факт.
        return self._insert(extracted, embedding, episode_id, now, subject)

    def _find_similar(self, facts: list[Fact], embedding: list[float]) -> Fact | None:
        """Возвращает самый близкий факт, если сходство выше порога."""
        best: Fact | None = None
        best_score = 0.0
        target = np.asarray(embedding, dtype=np.float32)
        for fact in facts:
            if fact.embedding is None:
                continue
            own = np.asarray(fact.embedding, dtype=np.float32)
            if own.shape != target.shape:
                continue
            score = cosine(target, own)
            if score >= self.config.merge_similarity and score > best_score:
                best, best_score = fact, score
        return best

    def _bump_confidence(self, fact: Fact) -> float:
        """Округление защищает от накопления ошибки float (0.7+0.1 = 0.7999…)."""
        conf = fact.confidence + self.config.confidence_step
        return round(min(self.config.confidence_cap, conf), 4)

    def _reinforce(
        self,
        fact: Fact,
        extracted: ExtractedFact,
        embedding: list[float],
        *,
        now: datetime,
    ) -> ConsolidationReport:
        """Повторное упоминание: доверие растёт, ничего не теряем."""
        updated = fact.model_copy(
            update={
                "mentions": fact.mentions + 1,
                "confidence": self._bump_confidence(fact),
                "updated_at": now,
                "last_accessed_at": now,
                "importance": max(fact.importance, extracted.importance),
                "embedding": embedding,
            }
        )
        self.store.update_fact(updated)
        return ConsolidationReport(
            action="reinforced",
            fact_id=int(fact.id or 0),
            notes=[f"Повторное упоминание: mentions={updated.mentions}"],
        )

    def _merge(
        self,
        fact: Fact,
        extracted: ExtractedFact,
        embedding: list[float],
        *,
        now: datetime,
    ) -> ConsolidationReport:
        """Похожие значения сливаются — в памяти остаётся один факт."""
        merged_from: list[int] = []
        others = self.store.find_facts(predicate=fact.predicate, subject=fact.subject)
        target = np.asarray(embedding, dtype=np.float32)
        for other in others:
            if other.id == fact.id or other.embedding is None:
                continue
            own = np.asarray(other.embedding, dtype=np.float32)
            if own.shape != target.shape:
                continue
            if cosine(target, own) < self.config.merge_similarity:
                continue
            merged_from.append(int(other.id or 0))
            self.store.add_history(
                FactHistoryEntry(
                    fact_id=int(other.id or 0),
                    subject=other.subject,
                    predicate=other.predicate,
                    object=other.object,
                    canonical=other.canonical,
                    confidence=other.confidence,
                    superseded_by=fact.id,
                    reason="merged",
                    created_at=other.created_at,
                    replaced_at=now,
                )
            )
            self.store.delete_fact(int(other.id or 0))

        # Более информативное (длинное) значение становится активным,
        # второе сохраняется в истории — ничего не теряем.
        new_canonical = canonicalize(extracted.object)
        keep_new = len(new_canonical) > len(fact.value_key)
        if keep_new:
            self.store.add_history(
                FactHistoryEntry(
                    fact_id=int(fact.id or 0),
                    subject=fact.subject,
                    predicate=fact.predicate,
                    object=fact.object,
                    canonical=fact.canonical,
                    confidence=fact.confidence,
                    superseded_by=fact.id,
                    reason="merged",
                    created_at=fact.created_at,
                    replaced_at=now,
                )
            )
        else:
            # Новое значение менее информативно: оно не становится активным,
            # но всё равно сохраняется в истории, чтобы не потерять формулировку.
            self.store.add_history(
                FactHistoryEntry(
                    fact_id=int(fact.id or 0),
                    subject=fact.subject,
                    predicate=fact.predicate,
                    object=extracted.object,
                    canonical=new_canonical,
                    confidence=extracted.confidence,
                    superseded_by=fact.id,
                    reason="merged",
                    created_at=now,
                    replaced_at=now,
                )
            )

        updated = fact.model_copy(
            update={
                "object": extracted.object if keep_new else fact.object,
                "canonical": new_canonical if keep_new else fact.canonical,
                "mentions": fact.mentions + 1,
                "confidence": self._bump_confidence(fact),
                "updated_at": now,
                "last_accessed_at": now,
                "importance": max(fact.importance, extracted.importance),
                "embedding": embedding if keep_new else fact.embedding,
            }
        )
        self.store.update_fact(updated)
        notes = [f"Объединено с похожим значением «{fact.object}»"]
        if merged_from:
            notes.append(f"Убраны дубликаты: {merged_from}")
        return ConsolidationReport(
            action="merged",
            fact_id=int(fact.id or 0),
            merged_from=merged_from,
            notes=notes,
        )

    def _supersede_all(
        self,
        conflicts: list[Fact],
        extracted: ExtractedFact,
        embedding: list[float],
        episode_id: int | None,
        now: datetime,
    ) -> ConsolidationReport:
        """Новое значение вытесняет старое; старое остаётся в истории."""
        new_fact = Fact(
            subject=conflicts[0].subject,
            predicate=extracted.predicate,
            object=extracted.object,
            canonical=extracted.canonical,
            confidence=extracted.confidence,
            importance=max(extracted.importance, extracted.predicate.importance),
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
            source_episode_id=episode_id,
            embedding=embedding,
        )
        created = self.store.add_fact(new_fact)

        replaced_ids: list[int] = []
        for old in conflicts:
            self.store.add_history(
                FactHistoryEntry(
                    fact_id=int(old.id or 0),
                    subject=old.subject,
                    predicate=old.predicate,
                    object=old.object,
                    canonical=old.canonical,
                    confidence=old.confidence,
                    superseded_by=created.id,
                    reason="superseded",
                    created_at=old.created_at,
                    replaced_at=now,
                )
            )
            self.store.delete_fact(int(old.id or 0))
            replaced_ids.append(int(old.id or 0))

        old_values = ", ".join(f"«{old.object}»" for old in conflicts)
        return ConsolidationReport(
            action="superseded",
            fact_id=int(created.id or 0),
            replaced_fact_id=replaced_ids[0] if replaced_ids else None,
            notes=[
                f"Противоречие в слоте {extracted.predicate.value}: "
                f"{old_values} -> «{extracted.object}» (старое сохранено в истории)"
            ],
        )

    def _insert(
        self,
        extracted: ExtractedFact,
        embedding: list[float],
        episode_id: int | None,
        now: datetime,
        subject: str,
    ) -> ConsolidationReport:
        """Новый факт: слот свободен, противоречий нет."""
        fact = Fact(
            subject=subject,
            predicate=extracted.predicate,
            object=extracted.object,
            canonical=extracted.canonical,
            confidence=extracted.confidence,
            importance=max(extracted.importance, extracted.predicate.importance),
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
            source_episode_id=episode_id,
            embedding=embedding,
        )
        created = self.store.add_fact(fact)
        return ConsolidationReport(
            action="inserted",
            fact_id=int(created.id or 0),
            notes=[f"Новый факт: {extracted.predicate.value}={extracted.object}"],
        )


__all__ = ["Consolidator"]

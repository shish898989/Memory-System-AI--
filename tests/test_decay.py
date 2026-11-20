"""Тесты забывания нерелевантного."""

from __future__ import annotations

import pytest

from memory_system.config import DecayConfig
from memory_system.consolidation import Consolidator
from memory_system.decay import Decayer, recency_score
from memory_system.embeddings import HashingEmbedder
from memory_system.extractor import ExtractedFact
from memory_system.models import Predicate
from memory_system.store import MemoryStore


def test_recency_decays_by_half_life() -> None:
    assert recency_score(0, 14) == pytest.approx(1.0)
    assert recency_score(14, 14) == pytest.approx(0.5)
    assert recency_score(28, 14) == pytest.approx(0.25)


def test_recency_never_negative() -> None:
    assert recency_score(1000, 14) > 0.0


def test_stale_low_importance_fact_is_archived(store: MemoryStore, now, days_ago) -> None:
    consolidator = Consolidator(store, HashingEmbedder())
    report = consolidator.add(
        ExtractedFact(predicate=Predicate.LIKES, object="кофе", importance=0.2),
        now=days_ago(200),
    )
    decayer = Decayer(store, HashingEmbedder(), DecayConfig(half_life_days=14))

    forgotten = decayer.sweep(now=now)

    assert [fact.id for fact in forgotten] == [report.fact_id]
    archived = store.get_fact(report.fact_id)
    assert archived is not None and archived.archived is True
    assert store.count_facts(include_archived=False) == 0


def test_fresh_fact_is_not_archived(store: MemoryStore, now) -> None:
    consolidator = Consolidator(store, HashingEmbedder())
    consolidator.add(
        ExtractedFact(predicate=Predicate.LIKES, object="кофе", importance=0.2), now=now
    )
    decayer = Decayer(store, HashingEmbedder())

    assert decayer.sweep(now=now) == []


def test_profile_facts_are_never_archived(store: MemoryStore, now, days_ago) -> None:
    consolidator = Consolidator(store, HashingEmbedder())
    name = consolidator.add(
        ExtractedFact(predicate=Predicate.NAME, object="Алексей", importance=1.0),
        now=days_ago(3650),
    )
    city = consolidator.add(
        ExtractedFact(predicate=Predicate.CITY, object="Москва", importance=0.85),
        now=days_ago(3650),
    )
    decayer = Decayer(store, HashingEmbedder(), DecayConfig(half_life_days=1))

    forgotten = decayer.sweep(now=now)

    assert forgotten == []
    assert store.get_fact(name.fact_id).archived is False
    assert store.get_fact(city.fact_id).archived is False


def test_archived_fact_keeps_full_data(store: MemoryStore, now, days_ago) -> None:
    """Забывание — это архив, а не удаление: факт можно восстановить."""
    consolidator = Consolidator(store, HashingEmbedder())
    report = consolidator.add(
        ExtractedFact(predicate=Predicate.OTHER, object="люблю джаз", importance=0.1),
        now=days_ago(365),
    )
    Decayer(store, HashingEmbedder(), DecayConfig(half_life_days=1)).sweep(now=now)

    archived = store.get_fact(report.fact_id)
    assert archived is not None
    assert archived.object == "люблю джаз"
    assert archived.archived is True


def test_accessing_fact_refreshes_recency(store: MemoryStore, now, days_ago) -> None:
    consolidator = Consolidator(store, HashingEmbedder())
    report = consolidator.add(
        ExtractedFact(predicate=Predicate.OTHER, object="джаз", importance=0.1),
        now=days_ago(100),
    )
    decayer = Decayer(store, HashingEmbedder(), DecayConfig(half_life_days=14))

    before = decayer.score(store.get_fact(report.fact_id), now=now)
    store.touch([report.fact_id], at=now)
    after = decayer.score(store.get_fact(report.fact_id), now=now)

    assert after > before

"""Тесты консолидации: дедупликация, объединение похожих, противоречия."""

from __future__ import annotations

import pytest

from memory_system.config import ConsolidationConfig
from memory_system.consolidation import Consolidator
from memory_system.embeddings import HashingEmbedder
from memory_system.extractor import ExtractedFact
from memory_system.models import Predicate
from memory_system.store import MemoryStore


@pytest.fixture()
def consolidator(store: MemoryStore) -> Consolidator:
    return Consolidator(store, HashingEmbedder(), ConsolidationConfig())


def make(predicate: Predicate, value: str, importance: float = 0.8) -> ExtractedFact:
    return ExtractedFact(predicate=predicate, object=value, importance=importance)


def test_new_fact_is_inserted(consolidator: Consolidator, store: MemoryStore) -> None:
    report = consolidator.add(make(Predicate.CITY, "Москва"))
    assert report.action == "inserted"
    assert store.count_facts() == 1
    assert store.count_history() == 0


def test_repeated_fact_is_reinforced_not_duplicated(
    consolidator: Consolidator, store: MemoryStore
) -> None:
    consolidator.add(make(Predicate.CITY, "Москва"))
    report = consolidator.add(make(Predicate.CITY, "Москва"))

    assert report.action == "reinforced"
    assert store.count_facts() == 1
    facts = store.find_facts(Predicate.CITY)
    assert facts[0].mentions == 2
    assert facts[0].confidence == pytest.approx(0.8)
    assert facts[0].confidence > 0.7  # доверие выросло после повтора


def test_canonical_case_insensitive_dedup(consolidator: Consolidator, store: MemoryStore) -> None:
    consolidator.add(make(Predicate.CITY, "Москва"))
    report = consolidator.add(make(Predicate.CITY, "москва"))

    assert report.action == "reinforced"
    assert store.count_facts() == 1


def test_similar_values_are_merged(store: MemoryStore) -> None:
    # Низкий порог — чтобы тест проверял логику слияния, а не качество эмбеддера.
    consolidator = Consolidator(
        store, HashingEmbedder(), ConsolidationConfig(merge_similarity=0.35)
    )
    consolidator.add(make(Predicate.LIKES, "играть в шахматы"))
    report = consolidator.add(make(Predicate.LIKES, "играть в шахматы вечером"))

    assert report.action == "merged"
    assert store.count_facts() == 1
    assert store.count_history() == 1
    assert store.history_for()[0].reason == "merged"


def test_contradiction_supersedes_old_value(consolidator: Consolidator, store: MemoryStore) -> None:
    first = consolidator.add(make(Predicate.CITY, "Санкт-Петербург"))
    report = consolidator.add(make(Predicate.CITY, "Москва"))

    assert report.action == "superseded"
    assert report.replaced_fact_id == first.fact_id

    active = store.find_facts(Predicate.CITY)
    assert [fact.object for fact in active] == ["Москва"]

    history = store.history_for(predicate=Predicate.CITY)
    assert len(history) == 1
    assert history[0].object == "Санкт-Петербург"
    assert history[0].superseded_by == report.fact_id
    assert history[0].reason == "superseded"


def test_multivalued_predicate_does_not_conflict(
    consolidator: Consolidator, store: MemoryStore
) -> None:
    consolidator.add(make(Predicate.LIKES, "кофе"))
    report = consolidator.add(make(Predicate.LIKES, "чай"))

    assert report.action == "inserted"
    assert store.count_facts() == 2
    assert store.count_history() == 0


def test_superseded_value_is_recoverable_from_history(
    consolidator: Consolidator, store: MemoryStore
) -> None:
    consolidator.add(make(Predicate.JOB, "аналитик"))
    consolidator.add(make(Predicate.JOB, "разработчик"))
    consolidator.add(make(Predicate.JOB, "тимлид"))

    history = store.history_for(predicate=Predicate.JOB)
    assert [entry.object for entry in history] == ["аналитик", "разработчик"]
    assert [fact.object for fact in store.find_facts(Predicate.JOB)] == ["тимлид"]

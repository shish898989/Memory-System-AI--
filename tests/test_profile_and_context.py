"""Тесты долгосрочного профиля и сборки контекста."""

from __future__ import annotations

from memory_system.config import ContextConfig
from memory_system.consolidation import Consolidator
from memory_system.context_builder import ContextBuilder, estimate_tokens
from memory_system.decay import Decayer
from memory_system.embeddings import HashingEmbedder
from memory_system.extractor import ExtractedFact
from memory_system.models import Predicate
from memory_system.profile import ProfileBuilder
from memory_system.store import MemoryStore


def _seed(store: MemoryStore, now) -> None:
    consolidator = Consolidator(store, HashingEmbedder())
    for predicate, value in [
        (Predicate.NAME, "Алексей"),
        (Predicate.CITY, "Санкт-Петербург"),
        (Predicate.JOB, "backend-разработчик"),
        (Predicate.LIKES, "кофе"),
    ]:
        consolidator.add(
            ExtractedFact(predicate=predicate, object=value, importance=predicate.importance),
            now=now,
        )


def test_profile_contains_stable_slots_only(store: MemoryStore, now) -> None:
    _seed(store, now)
    profile = ProfileBuilder(store).build()

    assert profile.get(Predicate.NAME) == "Алексей"
    assert profile.get(Predicate.CITY) == "Санкт-Петербург"
    assert profile.get(Predicate.LIKES) is None  # предпочтения не входят в профиль


def test_profile_order_is_stable(store: MemoryStore, now) -> None:
    _seed(store, now)
    profile = ProfileBuilder(store).build()
    order = [slot.predicate for slot in profile.slots]

    assert order == [Predicate.NAME, Predicate.CITY, Predicate.JOB]


def test_profile_keeps_history_of_changes(store: MemoryStore, now) -> None:
    _seed(store, now)
    Consolidator(store, HashingEmbedder()).add(
        ExtractedFact(predicate=Predicate.CITY, object="Москва", importance=0.85), now=now
    )

    profile = ProfileBuilder(store).build()
    city = next(slot for slot in profile.slots if slot.predicate is Predicate.CITY)

    assert city.value == "Москва"
    assert len(city.history) == 1
    assert "Санкт-Петербург" in city.history[0]


def test_context_includes_profile_and_relevant_facts(store: MemoryStore, now) -> None:
    _seed(store, now)
    embedder = HashingEmbedder()
    builder = ContextBuilder(
        store, embedder, Decayer(store, embedder), ProfileBuilder(store), ContextConfig()
    )

    context = builder.build("Посоветуй кофейню рядом", now=now)

    assert "Долгосрочный профиль" in context.text
    assert "Алексей" in context.text
    assert context.tokens > 0


def test_context_excludes_current_query_from_recent_turns(store: MemoryStore, now) -> None:
    store.add_episode("привет", created_at=now)
    embedder = HashingEmbedder()
    builder = ContextBuilder(
        store, embedder, Decayer(store, embedder), ProfileBuilder(store), ContextConfig()
    )

    context = builder.build("привет", now=now)

    assert context.text == ""
    assert context.recent_turns == []


def test_context_respects_token_budget(store: MemoryStore, now) -> None:
    consolidator = Consolidator(store, HashingEmbedder())
    for index in range(40):
        consolidator.add(
            ExtractedFact(
                predicate=Predicate.LIKES, object=f"увлечение номер {index}", importance=0.4
            ),
            now=now,
        )
    embedder = HashingEmbedder()
    config = ContextConfig(token_budget=120, max_episodic_facts=40)
    builder = ContextBuilder(
        store, embedder, Decayer(store, embedder), ProfileBuilder(store), config
    )

    context = builder.build("увлечение", now=now)

    assert context.tokens <= 120
    assert estimate_tokens(context.text) <= 120


def test_context_marks_used_facts_as_accessed(store: MemoryStore, now, days_ago) -> None:
    consolidator = Consolidator(store, HashingEmbedder())
    report = consolidator.add(
        ExtractedFact(predicate=Predicate.LIKES, object="кофе", importance=0.4),
        now=days_ago(10),
    )
    embedder = HashingEmbedder()
    builder = ContextBuilder(
        store, embedder, Decayer(store, embedder), ProfileBuilder(store), ContextConfig()
    )

    builder.build("кофе", now=now)

    fact = store.get_fact(report.fact_id)
    assert fact is not None
    assert fact.last_accessed_at == now


def test_multivalued_profile_slots_show_all_values(store: MemoryStore, now) -> None:
    consolidator = Consolidator(store, HashingEmbedder())
    for predicate, value in [
        (Predicate.STACK, "Python"),
        (Predicate.STACK, "JavaScript"),
        (Predicate.GOAL, "выучить Rust"),
        (Predicate.GOAL, "стать ML-инженером"),
    ]:
        consolidator.add(
            ExtractedFact(predicate=predicate, object=value, importance=predicate.importance),
            now=now,
        )

    profile = ProfileBuilder(store).build()
    stack = profile.get(Predicate.STACK)
    goal = profile.get(Predicate.GOAL)

    assert stack is not None
    assert {part.strip() for part in stack.split(",")} == {"Python", "JavaScript"}
    assert goal is not None
    assert "выучить Rust" in goal and "стать ML-инженером" in goal

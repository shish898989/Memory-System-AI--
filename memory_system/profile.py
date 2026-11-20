"""Долгосрочный профиль пользователя.

Профиль — это подмножество стабильных слотов (имя, возраст, город, работа,
язык, стек, цели). Отличия от «обычной» памяти:

* профиль попадает в контекст **всегда**, а не по релевантности;
* профиль **не забывается** при decay;
* у каждого слота видна история изменений (переезд, смена работы).

Порядок слотов фиксирован, чтобы контекст был стабильным между вызовами.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from memory_system.models import (
    PREDICATE_LABELS,
    Fact,
    Predicate,
)
from memory_system.store import MemoryStore

# Порядок важен: сначала самые «якорные» атрибуты личности.
PROFILE_ORDER: tuple[Predicate, ...] = (
    Predicate.NAME,
    Predicate.AGE,
    Predicate.CITY,
    Predicate.JOB,
    Predicate.LANGUAGE,
    Predicate.STACK,
    Predicate.GOAL,
)


class ProfileSlot(BaseModel):
    """Один слот профиля с активным значением и историей."""

    predicate: Predicate
    label: str
    value: str
    confidence: float
    updated_at: datetime
    history: list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    """Долгосрочный профиль: упорядоченный набор слотов."""

    slots: list[ProfileSlot] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.slots

    def get(self, predicate: Predicate) -> str | None:
        for slot in self.slots:
            if slot.predicate is predicate:
                return slot.value
        return None

    def as_lines(self) -> list[str]:
        return [f"{slot.label}: {slot.value}" for slot in self.slots]

    def as_text(self) -> str:
        return "\n".join(self.as_lines())


class ProfileBuilder:
    """Собирает профиль из активных фактов памяти."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def build(self, *, subject: str = "user", with_history: bool = True) -> UserProfile:
        active = self.store.find_facts(subject=subject)
        facts_by_predicate: dict[Predicate, list[Fact]] = {}
        for fact in active:
            if fact.predicate.is_profile_slot:
                facts_by_predicate.setdefault(fact.predicate, []).append(fact)

        history_entries = self.store.history_for() if with_history else []
        slots: list[ProfileSlot] = []
        for predicate in PROFILE_ORDER:
            facts = facts_by_predicate.get(predicate)
            if not facts:
                continue
            if predicate.is_multivalued:
                # «Стек» и «цели» могут иметь несколько значений — показываем все.
                ordered = sorted(facts, key=lambda fact: fact.object)
                value = ", ".join(fact.object for fact in ordered)
                confidence = max(fact.confidence for fact in facts)
                updated_at = max(fact.updated_at for fact in facts)
            else:
                # Если значений несколько, показываем самое доверенное и свежее.
                fact = max(facts, key=lambda item: (item.confidence, item.updated_at))
                value = fact.object
                confidence = fact.confidence
                updated_at = fact.updated_at
            previous = [
                f"{entry.object} (до {entry.replaced_at:%d.%m.%Y})"
                for entry in history_entries
                if entry.predicate is predicate and entry.reason == "superseded"
            ]
            slots.append(
                ProfileSlot(
                    predicate=predicate,
                    label=PREDICATE_LABELS[predicate],
                    value=value,
                    confidence=confidence,
                    updated_at=updated_at,
                    history=previous,
                )
            )
        return UserProfile(slots=slots)


__all__ = ["PROFILE_ORDER", "ProfileBuilder", "ProfileSlot", "UserProfile"]

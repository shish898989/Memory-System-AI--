"""Модели данных памяти.

Факт хранится как типизированный слот ``(subject, predicate, object)``.
Именно слот, а не свободный текст, позволяет детерминированно решать
объединение похожих фактов и противоречия.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    """Текущее время в UTC (единая точка для тестируемости)."""
    return datetime.now(UTC)


class Predicate(StrEnum):
    """Поддерживаемые типы фактов (слоты)."""

    NAME = "name"
    CITY = "city"
    JOB = "job"
    AGE = "age"
    LANGUAGE = "language"
    STACK = "stack"
    GOAL = "goal"
    LIKES = "likes"
    DISLIKES = "dislikes"
    PET = "pet"
    OTHER = "other"

    @property
    def is_profile_slot(self) -> bool:
        """Стабильные слоты образуют долгосрочный профиль и не забываются."""
        return self in _PROFILE_PREDICATES

    @property
    def is_multivalued(self) -> bool:
        """Многозначные предикаты не конфликтуют: два хобби не противоречие."""
        return self in _MULTIVALUED_PREDICATES

    @property
    def importance(self) -> float:
        """Базовая важность слота: чем выше, тем медленнее забывается."""
        return _PREDICATE_IMPORTANCE.get(self, 0.5)


_PROFILE_PREDICATES = frozenset(
    {
        Predicate.NAME,
        Predicate.CITY,
        Predicate.JOB,
        Predicate.AGE,
        Predicate.LANGUAGE,
        Predicate.STACK,
        Predicate.GOAL,
    }
)

_MULTIVALUED_PREDICATES = frozenset(
    {Predicate.LIKES, Predicate.DISLIKES, Predicate.STACK, Predicate.GOAL}
)

# Порядок важен для стабильного вывода профиля в контексте.
_PROFILE_ORDER = (
    Predicate.NAME,
    Predicate.AGE,
    Predicate.CITY,
    Predicate.JOB,
    Predicate.LANGUAGE,
    Predicate.STACK,
    Predicate.GOAL,
)

_PREDICATE_IMPORTANCE: dict[Predicate, float] = {
    Predicate.NAME: 1.0,
    Predicate.AGE: 0.9,
    Predicate.CITY: 0.85,
    Predicate.JOB: 0.85,
    Predicate.LANGUAGE: 0.8,
    Predicate.STACK: 0.75,
    Predicate.GOAL: 0.7,
    Predicate.PET: 0.6,
    Predicate.LIKES: 0.45,
    Predicate.DISLIKES: 0.45,
    Predicate.OTHER: 0.4,
}

# Человекочитаемые названия слотов для контекста и профиля.
PREDICATE_LABELS: dict[Predicate, str] = {
    Predicate.NAME: "Имя",
    Predicate.AGE: "Возраст",
    Predicate.CITY: "Город",
    Predicate.JOB: "Работа",
    Predicate.LANGUAGE: "Язык программирования",
    Predicate.STACK: "Технологии",
    Predicate.GOAL: "Цель",
    Predicate.PET: "Питомец",
    Predicate.LIKES: "Нравится",
    Predicate.DISLIKES: "Не нравится",
    Predicate.OTHER: "Прочее",
}


class Episode(BaseModel):
    """Сырая реплика пользователя — источник фактов."""

    model_config = ConfigDict(frozen=True)

    id: int | None = None
    text: str
    created_at: datetime = Field(default_factory=utcnow)


class Fact(BaseModel):
    """Факт памяти: типизированный слот с метаданными доверия."""

    model_config = ConfigDict(use_enum_values=False)

    id: int | None = None
    subject: str = "user"
    predicate: Predicate
    object: str
    # Нормализованное значение: по нему ищутся дубликаты.
    canonical: str = ""
    # confidence растёт при повторных упоминаниях и падает у старых фактов.
    confidence: float = 0.6
    importance: float = 0.5
    mentions: int = 1
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_accessed_at: datetime = Field(default_factory=utcnow)
    archived: bool = False
    source_episode_id: int | None = None
    embedding: list[float] | None = None

    @property
    def slot(self) -> tuple[str, str]:
        """Ключ однозначного слота: (subject, predicate)."""
        return (self.subject, self.predicate.value)

    @property
    def value_key(self) -> str:
        """Ключ значения внутри слота."""
        return self.canonical or self.object

    def text(self) -> str:
        """Текст факта для отображения и эмбеддинга."""
        return f"{PREDICATE_LABELS.get(self.predicate, self.predicate.value)}: {self.object}"

    def age_days(self, now: datetime | None = None) -> float:
        """Сколько дней прошло с последнего обращения к факту."""
        now = now or utcnow()
        return max(0.0, (now - self.last_accessed_at).total_seconds() / 86400.0)


class FactHistoryEntry(BaseModel):
    """Вытесненное значение слота: полная история изменений."""

    model_config = ConfigDict(frozen=True)

    id: int | None = None
    fact_id: int
    subject: str
    predicate: Predicate
    object: str
    canonical: str
    confidence: float
    superseded_by: int | None = None
    reason: str = "superseded"
    created_at: datetime = Field(default_factory=utcnow)
    replaced_at: datetime = Field(default_factory=utcnow)


class ConsolidationReport(BaseModel):
    """Что произошло при записи одного факта — удобно для тестов и демо."""

    action: str  # "inserted" | "reinforced" | "merged" | "superseded"
    fact_id: int
    replaced_fact_id: int | None = None
    merged_from: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def canonicalize(value: str) -> str:
    """Нормализация значения для поиска дубликатов.

    Приводим регистр, убираем пунктуацию и лишние пробелы. Синонимы
    («Питер»/«Санкт-Петербург») склеиваются только через словарь алиасов,
    это осознанное упрощение rule-based подхода.
    """
    cleaned = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in value)
    return " ".join(cleaned.split())


__all__ = [
    "PREDICATE_LABELS",
    "ConsolidationReport",
    "Episode",
    "Fact",
    "FactHistoryEntry",
    "Predicate",
    "canonicalize",
    "utcnow",
]

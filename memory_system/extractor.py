"""Извлечение фактов из русских реплик пользователя.

Реализация на правилах (регулярные выражения + словарь алиасов). Это осознанное
упрощение: полноценное извлечение требует LLM. Правила покрывают типовые
формулировки, которые встречаются в диалоге с ассистентом.

Интерфейс ``Extractor`` намеренно узкий, чтобы позже подменить реализацию
на LLM-экстрактор без изменений в остальном коде.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from memory_system.models import Predicate, canonicalize

# Синонимы и основы названий городов: сводим к одной форме, чтобы «Питер»,
# «Питере» и «Санкт-Петербурге» попали в один слот и считались одним фактом.
# Проверка идёт по вхождению основы, поэтому основа должна быть достаточно
# длинной, чтобы не ловить ложные совпадения.
CITY_STEMS: tuple[tuple[str, str], ...] = (
    ("санкт петербург", "Санкт-Петербург"),
    ("петербург", "Санкт-Петербург"),
    ("петербурге", "Санкт-Петербург"),
    ("питер", "Санкт-Петербург"),
    ("спб", "Санкт-Петербург"),
    ("москв", "Москва"),
    ("мск", "Москва"),
    ("екатеринбург", "Екатеринбург"),
    ("екб", "Екатеринбург"),
    ("новосибирск", "Новосибирск"),
    ("казан", "Казань"),
)

# Синонимы технологий.
STACK_ALIASES: dict[str, str] = {
    "питон": "Python",
    "пайтон": "Python",
    "python": "Python",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "тайпскрипт": "TypeScript",
    "typescript": "TypeScript",
    "с++": "C++",
    "cpp": "C++",
    "golang": "Go",
    "раст": "Rust",
    "rust": "Rust",
}

# Слова, обрывающие значение факта.
_STOPWORDS = {
    "и",
    "а",
    "но",
    "уже",
    "сейчас",
    "теперь",
    "потом",
    "раньше",
    "потому",
    "чтобы",
    "который",
    "которая",
}

_VALUE = r"[A-Za-zА-Яа-яЁё0-9\+\#\.\-\s]"


@dataclass(slots=True)
class ExtractedFact:
    """Промежуточный результат извлечения — ещё не факт памяти."""

    predicate: Predicate
    object: str
    importance: float
    confidence: float = 0.7
    matched_text: str = ""

    @property
    def canonical(self) -> str:
        return canonicalize(self.object)


class Extractor(ABC):
    """Интерфейс экстрактора фактов."""

    @abstractmethod
    def extract(self, text: str) -> list[ExtractedFact]:
        """Возвращает все факты, найденные в реплике."""


def _trim_value(raw: str) -> str:
    """Обрезает значение по стоп-словам и лишней пунктуации."""
    value = raw.strip(" .,!?:;-\t\n")
    kept: list[str] = []
    for word in value.split():
        if word.lower() in _STOPWORDS and kept:
            break
        kept.append(word)
    return " ".join(kept).strip(" .,!?:;-")


class RuleBasedExtractor(Extractor):
    """Извлечение фактов регулярными выражениями.

    Порядок правил важен: специфичные шаблоны («переехал в …») идут раньше
    общих («живу в …»), иначе одно и то же место извлечётся дважды.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[re.Pattern[str], Predicate, float, float]] = [
            (
                re.compile(r"(?:меня зовут|мо[её] имя)\s+([а-яё]+)", re.I),
                Predicate.NAME,
                1.0,
                0.95,
            ),
            (
                re.compile(
                    r"мне\s+(?:уже\s+|сейчас\s+)?(\d{1,2})"
                    r"(?:\s*(?:год|года|лет))?(?=\s*[^а-яёa-z0-9]|$)",
                    re.I,
                ),
                Predicate.AGE,
                0.9,
                0.9,
            ),
            (
                re.compile(rf"(?:я\s+)?переехал[а]?\s+в\s+({_VALUE}{{2,40}})", re.I),
                Predicate.CITY,
                0.9,
                0.8,
            ),
            (
                re.compile(rf"(?:я\s+)?(?:живу|проживаю)\s+в\s+({_VALUE}{{2,40}})", re.I),
                Predicate.CITY,
                0.85,
                0.8,
            ),
            (
                re.compile(rf"(?:я\s+)?(?:родом из|приехал[а]? из)\s+({_VALUE}{{2,40}})", re.I),
                Predicate.CITY,
                0.8,
                0.7,
            ),
            (
                re.compile(rf"(?:я\s+)?(?:теперь\s+)?работаю\s+({_VALUE}{{2,40}})", re.I),
                Predicate.JOB,
                0.85,
                0.8,
            ),
            (
                re.compile(
                    rf"(?:я\s+)?(?:пишу|код[аю]|программирую)\s+(?:на|в)\s+({_VALUE}{{2,30}})",
                    re.I,
                ),
                Predicate.STACK,
                0.75,
                0.8,
            ),
            (
                re.compile(
                    rf"мой\s+(?:основной\s+)?язык\s+(?:—|-|это)?\s*({_VALUE}{{2,30}})",
                    re.I,
                ),
                Predicate.LANGUAGE,
                0.8,
                0.8,
            ),
            (
                re.compile(rf"мо[яи]\s+цель\s+(?:—|-|это)?\s*({_VALUE}{{3,60}})", re.I),
                Predicate.GOAL,
                0.75,
                0.8,
            ),
            (
                re.compile(rf"(?:я\s+)?хочу\s+({_VALUE}{{3,60}})", re.I),
                Predicate.GOAL,
                0.7,
                0.7,
            ),
            (
                re.compile(rf"(?:я\s+)?(?:очень\s+)?(?:люблю|обожаю)\s+({_VALUE}{{2,50}})", re.I),
                Predicate.LIKES,
                0.45,
                0.7,
            ),
            (
                re.compile(
                    rf"(?:я\s+)?(?:не\s+люблю|ненавижу|терпеть не могу)\s+({_VALUE}{{2,50}})",
                    re.I,
                ),
                Predicate.DISLIKES,
                0.45,
                0.7,
            ),
            (
                re.compile(
                    r"(?:у меня есть|у меня живёт)\s+"
                    r"(кот[а]?|кошк[ау]?|пс[а]?|собак[ау]?|попугай|хомяк)",
                    re.I,
                ),
                Predicate.PET,
                0.6,
                0.75,
            ),
        ]

    def extract(self, text: str) -> list[ExtractedFact]:
        """Прогоняет правила и убирает дубликаты внутри одной реплики."""
        # Отрицания вырезаем заранее: иначе «не люблю X» даст и «люблю X».
        negated = re.compile(
            rf"(?:не\s+люблю|ненавижу|терпеть не могу)\s+({_VALUE}{{1,50}})", re.I
        ).findall(text)
        negative_spans = [canonicalize(match) for match in negated]

        found: list[ExtractedFact] = []
        seen: set[tuple[Predicate, str]] = set()
        for pattern, predicate, importance, confidence in self._rules:
            for match in pattern.finditer(text):
                value = _trim_value(match.group(1))
                if not value:
                    continue
                value = self._normalize_value(predicate, value)
                canonical = canonicalize(value)
                if predicate is Predicate.LIKES and any(
                    canonical in negative for negative in negative_spans
                ):
                    continue
                key = (predicate, canonical)
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    ExtractedFact(
                        predicate=predicate,
                        object=value,
                        importance=importance,
                        confidence=confidence,
                        matched_text=match.group(0).strip(),
                    )
                )
        return found

    @staticmethod
    def _normalize_value(predicate: Predicate, value: str) -> str:
        """Приводит синонимы и падежи городов/технологий к канонической форме."""
        key = canonicalize(value)
        if predicate is Predicate.CITY:
            for stem, city in CITY_STEMS:
                if stem in key:
                    return city
            return _titled(value)
        if predicate in {Predicate.STACK, Predicate.LANGUAGE}:
            if key in STACK_ALIASES:
                return STACK_ALIASES[key]
            for alias, tech in STACK_ALIASES.items():
                if alias in key:
                    return tech
            return value
        if predicate is Predicate.JOB:
            return _titled(value)
        if predicate is Predicate.NAME:
            return _titled(value.lower())
        return value


__all__ = ["ExtractedFact", "Extractor", "RuleBasedExtractor", "STACK_ALIASES"]


def _titled(value: str) -> str:
    """Делает первую букву заглавной, сохраняя остальное."""
    return value[:1].upper() + value[1:]

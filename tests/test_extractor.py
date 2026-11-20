"""Тесты извлечения фактов из русских реплик."""

from __future__ import annotations

import pytest

from memory_system.extractor import RuleBasedExtractor
from memory_system.models import Predicate


@pytest.fixture()
def extractor() -> RuleBasedExtractor:
    return RuleBasedExtractor()


def _pairs(extractor: RuleBasedExtractor, text: str) -> dict[Predicate, str]:
    return {fact.predicate: fact.object for fact in extractor.extract(text)}


def test_extracts_name_and_age(extractor: RuleBasedExtractor) -> None:
    facts = _pairs(extractor, "Привет! Меня зовут Алексей, мне 27 лет.")
    assert facts[Predicate.NAME] == "Алексей"
    assert facts[Predicate.AGE] == "27"


def test_extracts_age_without_let_suffix(extractor: RuleBasedExtractor) -> None:
    facts = _pairs(extractor, "Привет, меня зовут Илья, мне 21, я живу в Екатеринбурге")
    assert facts[Predicate.AGE] == "21"
    assert facts[Predicate.CITY] == "Екатеринбург"


def test_extracts_city(extractor: RuleBasedExtractor) -> None:
    assert _pairs(extractor, "Я живу в Питере")[Predicate.CITY] == "Санкт-Петербург"


def test_city_synonyms_share_canonical_form(extractor: RuleBasedExtractor) -> None:
    a = _pairs(extractor, "Я живу в Питере")[Predicate.CITY]
    b = _pairs(extractor, "Я живу в Санкт-Петербурге")[Predicate.CITY]
    assert a == b == "Санкт-Петербург"


def test_relocation_extracts_destination_city(extractor: RuleBasedExtractor) -> None:
    assert _pairs(extractor, "Потом я переехал в Москву")[Predicate.CITY] == "Москва"


def test_extracts_job_and_goal(extractor: RuleBasedExtractor) -> None:
    facts = _pairs(extractor, "Работаю backend-разработчиком, хочу стать ML-инженером")
    assert "backend" in facts[Predicate.JOB].lower()
    assert "ml" in facts[Predicate.GOAL].lower()


def test_extracts_preferences(extractor: RuleBasedExtractor) -> None:
    facts = _pairs(extractor, "Люблю кофе, но не люблю жару")
    assert "кофе" in facts[Predicate.LIKES].lower()
    assert "жару" in facts[Predicate.DISLIKES].lower()


def test_value_is_cut_at_stopword(extractor: RuleBasedExtractor) -> None:
    facts = extractor.extract("Я живу в Москве и работаю в Яндексе")
    cities = [fact.object for fact in facts if fact.predicate is Predicate.CITY]
    assert cities == ["Москва"]


def test_irrelevant_text_yields_nothing(extractor: RuleBasedExtractor) -> None:
    assert extractor.extract("Спасибо, это было полезно!") == []


def test_no_duplicates_within_one_utterance(extractor: RuleBasedExtractor) -> None:
    facts = extractor.extract("Я живу в Питере, живу в Питере")
    assert len([f for f in facts if f.predicate is Predicate.CITY]) == 1


def test_negated_like_is_not_a_like(extractor: RuleBasedExtractor) -> None:
    facts = _pairs(extractor, "Я не люблю жару")
    assert Predicate.LIKES not in facts
    assert "жару" in facts[Predicate.DISLIKES].lower()

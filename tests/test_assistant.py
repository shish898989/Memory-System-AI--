"""End-to-end тесты MemoryAssistant на офлайн-эмбеддере."""

from __future__ import annotations

from memory_system.assistant import MemoryAssistant
from memory_system.models import Predicate


def test_assistant_remembers_profile_and_recalls_it(assistant: MemoryAssistant) -> None:
    result = assistant.remember("Привет! Меня зовут Алексей, мне 27 лет. Я живу в Питере.")

    assert result.episode_id > 0
    assert set(result.actions) <= {"inserted", "reinforced", "merged", "superseded"}

    context = assistant.recall("Кто я и где живу?")
    assert "Алексей" in context.text
    assert "Санкт-Петербург" in context.text

    profile = assistant.profile()
    assert profile.get(Predicate.NAME) == "Алексей"
    assert profile.get(Predicate.CITY) == "Санкт-Петербург"
    assert assistant.stats()["active"] >= 3


def test_assistant_resolves_city_contradiction(assistant: MemoryAssistant) -> None:
    assistant.remember("Я живу в Питере.")
    assistant.remember("Потом я переехал в Москву.")

    assert assistant.profile().get(Predicate.CITY) == "Москва"
    assert any("Санкт-Петербург" in value for _slot, value, _reason in assistant.history())

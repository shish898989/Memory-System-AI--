"""Тесты слоя генерации ответа: память -> сообщения -> LLM."""

from __future__ import annotations

from memory_system.assistant import MemoryAssistant
from memory_system.config import MemoryConfig
from memory_system.context_builder import MemoryContext
from memory_system.embeddings import HashingEmbedder
from memory_system.generator import LlamaCppServerGenerator, build_messages
from memory_system.profile import UserProfile
from memory_system.store import MemoryStore


class FakeGenerator:
    """Offline-генератор: проверяет, что в LLM уходит именно контекст памяти."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def respond(self, messages: list[dict[str, str]]) -> str:
        self.messages = [dict(message) for message in messages]
        return "Ответ по памяти"


def _assistant() -> MemoryAssistant:
    return MemoryAssistant(
        MemoryConfig(), store=MemoryStore(keep_in_memory=True), embedder=HashingEmbedder()
    )


def test_answer_passes_memory_context_to_generator() -> None:
    assistant = _assistant()
    try:
        assistant.remember("Меня зовут Алексей, мне 27 лет. Я живу в Питере.")
        generator = FakeGenerator()

        result = assistant.answer("Как меня зовут?", generator=generator)

        assert result.answer == "Ответ по памяти"
        assert "Алексей" in result.context.text
        assert generator.messages[0]["role"] == "system"
        assert "Как меня зовут?" in generator.messages[1]["content"]
        assert "Алексей" in generator.messages[1]["content"]
    finally:
        assistant.close()


def test_build_messages_marks_empty_memory() -> None:
    context = MemoryContext(profile=UserProfile(), text="", tokens=0)

    messages = build_messages(context, "Привет")

    assert "Память пока пуста." in messages[1]["content"]


def test_make_generator_respects_llamacpp_backend() -> None:
    assistant = _assistant()
    try:
        assistant.config.llm_backend = "llamacpp"
        assert isinstance(assistant._make_generator(), LlamaCppServerGenerator)
    finally:
        assistant.close()

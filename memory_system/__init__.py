"""Memory System для AI-ассистента.

Пакет реализует четыре требования к памяти ассистента:
1. забывание нерелевантного (``decay``);
2. объединение похожих фактов (``consolidation``);
3. разрешение противоречий (``consolidation``);
4. долгосрочный профиль пользователя (``profile``).
"""

from memory_system.assistant import AnswerResult, MemoryAssistant
from memory_system.config import MemoryConfig
from memory_system.embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from memory_system.generator import GeneratorUnavailable, LlamaCppServerGenerator, OllamaGenerator
from memory_system.store import MemoryStore

__all__ = [
    "AnswerResult",
    "MemoryAssistant",
    "MemoryConfig",
    "MemoryStore",
    "Embedder",
    "SentenceTransformerEmbedder",
    "HashingEmbedder",
    "GeneratorUnavailable",
    "LlamaCppServerGenerator",
    "OllamaGenerator",
]

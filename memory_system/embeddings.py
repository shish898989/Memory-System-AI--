"""Эмбеддинги для смыслового сравнения фактов.

Абстракция ``Embedder`` позволяет подменить модель:
* ``SentenceTransformerEmbedder`` — настоящая мультиязычная модель (русский);
* ``HashingEmbedder`` — офлайн-фолбэк на символьных n-граммах, нужен для тестов
  и работы без сети. Смысловое сходство он ловит хуже, но детерминирован.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Интерфейс получения эмбеддингов."""

    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Возвращает матрицу (len(texts), dim) с L2-нормализованными векторами."""

    def encode_one(self, text: str) -> list[float]:
        """Эмбеддинг одной строки в виде списка float (для хранения в SQLite)."""
        return self.encode([text])[0].tolist()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство для уже нормализованных векторов — это скалярное произведение."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SentenceTransformerEmbedder(Embedder):
    """Мультиязычные эмбеддинги через sentence-transformers на GPU/CPU."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str | None = None,
    ) -> None:
        # Импорт внутри конструктора: пакет не нужен для офлайн-тестов.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover - torch есть в requirements
                device = "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder(Embedder):
    """Офлайн-эмбеддинг на хешах символьных n-грамм.

    Быстрый, детерминированный и не требует сети. Ловит частичные совпадения
    подстрок («питер» / «питере»), но не синонимы.
    """

    def __init__(self, dim: int = 512, ngram_range: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self.ngram_range = ngram_range

    def _ngrams(self, text: str) -> list[str]:
        norm = " ".join(text.lower().replace("ё", "е").split())
        padded = f"_{norm}_"
        grams: list[str] = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if len(padded) < n:
                continue
            grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
        return grams

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for gram in self._ngrams(text):
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[row, index] += sign
            norm = float(np.linalg.norm(out[row]))
            if norm > 0.0:
                out[row] /= norm
        return out


def build_embedder(kind: str, model_name: str, device: str | None = None) -> Embedder:
    """Фабрика эмбеддеров с откатом на офлайн-вариант при проблемах с моделью."""
    if kind == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name=model_name, device=device)
    except Exception as exc:  # noqa: BLE001 - нужен любой откат, причина логируется вызывающим
        raise EmbedderUnavailable(str(exc)) from exc


class EmbedderUnavailable(RuntimeError):
    """Модель эмбеддингов недоступна (нет сети/кэша/зависимостей)."""


__all__ = [
    "Embedder",
    "EmbedderUnavailable",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
    "cosine",
]

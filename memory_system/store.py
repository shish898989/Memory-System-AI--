"""Хранилище памяти на SQLite.

Три таблицы:
* ``episodes`` — сырые реплики пользователя (источник фактов);
* ``facts`` — активные факты (типизированные слоты);
* ``fact_history`` — вытесненные значения, полная история изменений.

Факты не удаляются при забывании: оно выставляет флаг ``archived``.
При вытеснении значение сначала переносится в ``fact_history``, после чего
старая строка ``facts`` может быть удалена — история при этом сохраняется.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from memory_system.models import Episode, Fact, FactHistoryEntry, Predicate, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    canonical TEXT NOT NULL,
    confidence REAL NOT NULL,
    importance REAL NOT NULL,
    mentions INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,
    source_episode_id INTEGER,
    embedding TEXT
);

CREATE INDEX IF NOT EXISTS idx_facts_slot ON facts(subject, predicate, archived);
CREATE INDEX IF NOT EXISTS idx_facts_canonical ON facts(canonical);

CREATE TABLE IF NOT EXISTS fact_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id INTEGER NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    canonical TEXT NOT NULL,
    confidence REAL NOT NULL,
    superseded_by INTEGER,
    reason TEXT NOT NULL DEFAULT 'superseded',
    created_at TEXT NOT NULL,
    replaced_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_fact ON fact_history(fact_id);
"""

FACT_COLUMNS = (
    "id, subject, predicate, object, canonical, confidence, importance, mentions, "
    "created_at, updated_at, last_accessed_at, archived, source_episode_id, embedding"
)


class MemoryStore:
    """Тонкий слой доступа к SQLite без ORM."""

    def __init__(self, db_path: Path | str | None = None, *, keep_in_memory: bool = False) -> None:
        self.keep_in_memory = keep_in_memory
        self.db_path: Path | None = None if keep_in_memory else Path(db_path or "memory.db")
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def _connect(self) -> sqlite3.Connection:
        if self.keep_in_memory:
            conn = sqlite3.connect(":memory:")
        else:
            conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # --- episodes -------------------------------------------------------
    def add_episode(self, text: str, created_at: datetime | None = None) -> Episode:
        created_at = created_at or utcnow()
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO episodes (text, created_at) VALUES (?, ?)",
                (text, created_at.isoformat()),
            )
            episode_id = int(cur.lastrowid)
        return Episode(id=episode_id, text=text, created_at=created_at)

    def recent_episodes(self, limit: int = 5) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        episodes = [
            Episode(
                id=row["id"],
                text=row["text"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]
        return list(reversed(episodes))

    def count_episodes(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])

    # --- facts ----------------------------------------------------------
    def add_fact(self, fact: Fact) -> Fact:
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO facts (
                    subject, predicate, object, canonical, confidence, importance, mentions,
                    created_at, updated_at, last_accessed_at, archived,
                    source_episode_id, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.subject,
                    fact.predicate.value,
                    fact.object,
                    fact.canonical or fact.object,
                    fact.confidence,
                    fact.importance,
                    fact.mentions,
                    fact.created_at.isoformat(),
                    fact.updated_at.isoformat(),
                    fact.last_accessed_at.isoformat(),
                    int(fact.archived),
                    fact.source_episode_id,
                    json.dumps(fact.embedding) if fact.embedding is not None else None,
                ),
            )
            fact_id = int(cur.lastrowid)
        return fact.model_copy(update={"id": fact_id})

    def update_fact(self, fact: Fact) -> Fact:
        if fact.id is None:
            raise ValueError("Нельзя обновить факт без id")
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE facts SET
                    object = ?, canonical = ?, confidence = ?, importance = ?, mentions = ?,
                    updated_at = ?, last_accessed_at = ?, archived = ?, embedding = ?
                WHERE id = ?
                """,
                (
                    fact.object,
                    fact.canonical or fact.object,
                    fact.confidence,
                    fact.importance,
                    fact.mentions,
                    fact.updated_at.isoformat(),
                    fact.last_accessed_at.isoformat(),
                    int(fact.archived),
                    json.dumps(fact.embedding) if fact.embedding is not None else None,
                    fact.id,
                ),
            )
        return fact

    def get_fact(self, fact_id: int) -> Fact | None:
        row = self._conn.execute(
            f"SELECT {FACT_COLUMNS} FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return _row_to_fact(row) if row else None

    def find_facts(
        self,
        predicate: Predicate | None = None,
        *,
        subject: str = "user",
        include_archived: bool = False,
    ) -> list[Fact]:
        query = f"SELECT {FACT_COLUMNS} FROM facts WHERE subject = ?"
        params: list[object] = [subject]
        if predicate is not None:
            query += " AND predicate = ?"
            params.append(predicate.value)
        if not include_archived:
            query += " AND archived = 0"
        query += " ORDER BY id"
        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_fact(row) for row in rows]

    def all_facts(self, *, include_archived: bool = True, subject: str = "user") -> list[Fact]:
        query = f"SELECT {FACT_COLUMNS} FROM facts WHERE subject = ?"
        if not include_archived:
            query += " AND archived = 0"
        query += " ORDER BY id"
        rows = self._conn.execute(query, (subject,)).fetchall()
        return [_row_to_fact(row) for row in rows]

    def count_facts(self, *, include_archived: bool = False) -> int:
        query = "SELECT COUNT(*) FROM facts"
        if not include_archived:
            query += " WHERE archived = 0"
        return int(self._conn.execute(query).fetchone()[0])

    def delete_fact(self, fact_id: int) -> None:
        """Физическое удаление — только когда значение уже перенесено в историю."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))

    def touch(self, fact_ids: list[int], at: datetime | None = None) -> None:
        """Отмечает факты как использованные — влияет на recency."""
        if not fact_ids:
            return
        at = at or utcnow()
        with self.transaction() as conn:
            conn.executemany(
                "UPDATE facts SET last_accessed_at = ? WHERE id = ?",
                [(at.isoformat(), fact_id) for fact_id in fact_ids],
            )

    # --- history --------------------------------------------------------
    def add_history(self, entry: FactHistoryEntry) -> FactHistoryEntry:
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO fact_history (
                    fact_id, subject, predicate, object, canonical, confidence,
                    superseded_by, reason, created_at, replaced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.fact_id,
                    entry.subject,
                    entry.predicate.value,
                    entry.object,
                    entry.canonical,
                    entry.confidence,
                    entry.superseded_by,
                    entry.reason,
                    entry.created_at.isoformat(),
                    entry.replaced_at.isoformat(),
                ),
            )
            entry_id = int(cur.lastrowid)
        return entry.model_copy(update={"id": entry_id})

    def history_for(
        self, fact_id: int | None = None, predicate: Predicate | None = None
    ) -> list[FactHistoryEntry]:
        query = "SELECT * FROM fact_history"
        conditions: list[str] = []
        params: list[object] = []
        if fact_id is not None:
            conditions.append("fact_id = ?")
            params.append(fact_id)
        if predicate is not None:
            conditions.append("predicate = ?")
            params.append(predicate.value)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id"
        rows = self._conn.execute(query, params).fetchall()
        return [
            FactHistoryEntry(
                id=row["id"],
                fact_id=row["fact_id"],
                subject=row["subject"],
                predicate=Predicate(row["predicate"]),
                object=row["object"],
                canonical=row["canonical"],
                confidence=row["confidence"],
                superseded_by=row["superseded_by"],
                reason=row["reason"],
                created_at=datetime.fromisoformat(row["created_at"]),
                replaced_at=datetime.fromisoformat(row["replaced_at"]),
            )
            for row in rows
        ]

    def count_history(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM fact_history").fetchone()[0])


def _row_to_fact(row: sqlite3.Row) -> Fact:
    embedding = json.loads(row["embedding"]) if row["embedding"] else None
    return Fact(
        id=row["id"],
        subject=row["subject"],
        predicate=Predicate(row["predicate"]),
        object=row["object"],
        canonical=row["canonical"],
        confidence=row["confidence"],
        importance=row["importance"],
        mentions=row["mentions"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]),
        archived=bool(row["archived"]),
        source_episode_id=row["source_episode_id"],
        embedding=embedding,
    )


__all__ = ["MemoryStore"]

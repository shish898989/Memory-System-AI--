"""CLI для Memory System: запись, recall, ответ через локальную LLM, профиль, история.

Ядро проекта — память: команды показывают, что именно ассистент запомнил, что
забыл и какой контекст собрал бы для ответа. Команда ``answer`` дополнительно
прогоняет этот контекст через локальную LLM (Ollama или llama.cpp), если она установлена.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memory_system import GeneratorUnavailable, HashingEmbedder, MemoryAssistant, MemoryConfig
from memory_system.assistant import IngestResult
from memory_system.context_builder import MemoryContext
from memory_system.models import PREDICATE_LABELS, Predicate
from memory_system.profile import UserProfile

app = typer.Typer(help="Memory System для AI-ассистента", no_args_is_help=True)
console = Console()

DEFAULT_SCENARIO = Path("scenarios/demo_dialog.json")
EMBEDDER_KINDS = {"hashing", "sentence-transformers"}

DB_OPTION = typer.Option(Path("memory.db"), "--db", help="Путь к SQLite-базе")
DEMO_DB_OPTION = typer.Option(Path("demo_memory.db"), "--db", help="Путь к SQLite-базе")
EMBEDDER_OPTION = typer.Option("hashing", "--embedder", help="hashing|sentence-transformers")
SCENARIO_OPTION = typer.Option(DEFAULT_SCENARIO, "--scenario", help="JSON со сценарием")
LLM_MODEL_OPTION = typer.Option(
    None, "--llm-model", help="Модель LLM, например qwen2.5:3b-instruct"
)
LLM_BACKEND_OPTION = typer.Option(None, "--llm-backend", help="Бэкенд LLM: ollama|llamacpp")


def _assistant(db: Path, embedder: str) -> MemoryAssistant:
    if embedder not in EMBEDDER_KINDS:
        raise typer.BadParameter(f"Неизвестный эмбеддер: {embedder}")
    config = MemoryConfig.from_env()
    config.db_path = db
    config.embedder = embedder
    if embedder == "hashing":
        # Явно подставляем офлайн-эмбеддер: torch/sentence-transformers не нужны.
        return MemoryAssistant(config, embedder=HashingEmbedder())
    return MemoryAssistant(config)


def _print_ingest(result: IngestResult) -> None:
    console.print(f"[bold]episode[/] {result.episode_id}")
    if not result.facts:
        console.print("  [dim]фактов не извлечено[/]")
    for report in result.facts:
        line = f"  [cyan]{report.action}[/] fact_id={report.fact_id}"
        if report.replaced_fact_id is not None:
            line += f", replaced={report.replaced_fact_id}"
        if report.merged_from:
            line += f", merged_from={report.merged_from}"
        console.print(line)
        for note in report.notes:
            console.print(f"    [dim]{note}[/]")
    if result.archived:
        console.print(f"  [dim]архивировано фактов: {len(result.archived)}[/]")


def _print_context(context: MemoryContext) -> None:
    if not context.text:
        console.print("[dim]Контекст пуст: память ещё ничего не знает.[/]")
        return
    console.print(Panel(context.text, title=f"Контекст памяти · ~{context.tokens} токенов"))


def _print_profile(profile: UserProfile) -> None:
    if profile.is_empty():
        console.print("[dim]Профиль пока пуст.[/]")
        return
    console.print(Panel(profile.as_text(), title="Долгосрочный профиль"))


def _print_stats(stats: dict[str, int]) -> None:
    table = Table(title="Статистика памяти")
    table.add_column("Метрика")
    table.add_column("Значение", justify="right")
    for key, value in stats.items():
        table.add_row(key, str(value))
    console.print(table)


def _print_history(entries: list[tuple[str, str, str]]) -> None:
    if not entries:
        console.print("[dim]История изменений пуста.[/]")
        return
    table = Table(title="История памяти")
    table.add_column("Слот")
    table.add_column("Было")
    table.add_column("Причина")
    for predicate, value, reason in entries:
        try:
            label = PREDICATE_LABELS.get(Predicate(predicate), predicate)
        except ValueError:
            label = predicate
        table.add_row(label, value, reason)
    console.print(table)


@app.command()
def remember(
    text: str = typer.Argument(..., help="Реплика пользователя"),
    db: Path = DB_OPTION,
    embedder: str = EMBEDDER_OPTION,
) -> None:
    """Запоминает реплику: извлекает факты, консолидирует, забывает лишнее."""
    with _assistant(db, embedder) as assistant:
        _print_ingest(assistant.remember(text))


@app.command()
def ask(
    query: str = typer.Argument(..., help="Вопрос пользователя"),
    db: Path = DB_OPTION,
    embedder: str = EMBEDDER_OPTION,
) -> None:
    """Показывает контекст, который память собрала бы для ответа."""
    with _assistant(db, embedder) as assistant:
        _print_context(assistant.recall(query))


@app.command()
def answer(
    query: str = typer.Argument(..., help="Вопрос пользователя"),
    db: Path = DB_OPTION,
    embedder: str = EMBEDDER_OPTION,
    llm_model: str | None = LLM_MODEL_OPTION,
    llm_backend: str | None = LLM_BACKEND_OPTION,
) -> None:
    """Отвечает на вопрос: память собирает контекст, локальная LLM формулирует ответ."""
    if llm_backend is not None and llm_backend not in {"ollama", "llamacpp"}:
        raise typer.BadParameter("--llm-backend должен быть ollama или llamacpp")
    with _assistant(db, embedder) as assistant:
        if llm_model:
            assistant.config.llm_model = llm_model
        if llm_backend:
            assistant.config.llm_backend = llm_backend
        try:
            result = assistant.answer(query)
        except GeneratorUnavailable as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc
        console.print(Panel(result.answer, title=f"Ответ · {assistant.config.llm_model}"))
        if result.context.text:
            console.print(
                Panel(
                    result.context.text,
                    title=f"Контекст памяти · ~{result.context.tokens} токенов",
                )
            )


@app.command()
def profile(
    db: Path = DB_OPTION,
    embedder: str = EMBEDDER_OPTION,
) -> None:
    """Показывает долгосрочный профиль пользователя."""
    with _assistant(db, embedder) as assistant:
        _print_profile(assistant.profile())


@app.command()
def history(
    db: Path = DB_OPTION,
    embedder: str = EMBEDDER_OPTION,
) -> None:
    """Показывает историю вытесненных и объединённых значений."""
    with _assistant(db, embedder) as assistant:
        _print_history(assistant.history())


@app.command()
def stats(
    db: Path = DB_OPTION,
    embedder: str = EMBEDDER_OPTION,
) -> None:
    """Показывает счётчики памяти."""
    with _assistant(db, embedder) as assistant:
        _print_stats(assistant.stats())


@app.command()
def demo(
    scenario: Path = SCENARIO_OPTION,
    db: Path = DEMO_DB_OPTION,
    embedder: str = EMBEDDER_OPTION,
) -> None:
    """Прогоняет демонстрационный диалог и показывает память в действии."""
    if not scenario.exists():
        raise typer.BadParameter(f"Сценарий не найден: {scenario}")
    data = json.loads(scenario.read_text(encoding="utf-8"))
    dialog = data.get("dialog", [])
    queries = data.get("queries", [])

    with _assistant(db, embedder) as assistant:
        console.rule("[bold]Демонстрация Memory System[/]")
        for item in dialog:
            text = str(item.get("user", "")) if isinstance(item, dict) else str(item)
            if not text:
                continue
            console.print(f"\n[bold]User:[/] {text}")
            _print_ingest(assistant.remember(text))

        for query in queries:
            console.print(f"\n[bold]Query:[/] {query}")
            _print_context(assistant.recall(str(query)))

        console.print()
        _print_profile(assistant.profile())
        _print_stats(assistant.stats())
        _print_history(assistant.history())


if __name__ == "__main__":
    app()

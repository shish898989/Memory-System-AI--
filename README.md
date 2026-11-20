# Memory System для AI-ассистента

NLP-проект: память ассистента на SQLite, которая умеет

1. **забывать нерелевантное** — факты получают score по свежести, важности и
   релевантности запросу и со временем уходят в архив (`memory_system/decay.py`);
2. **объединять похожие факты** — значения сравниваются по эмбеддингам, близкие
   варианты сливаются в один факт (`memory_system/consolidation.py`);
3. **разрешать противоречия** — однозначные слоты (`город`, `работа`, `имя`)
   вытесняют старое значение, а старое сохраняется в `fact_history`;
4. **поддерживать долгосрочный профиль** — стабильные слоты всегда попадают в
   контекст и не забываются (`memory_system/profile.py`).

Поверх памяти есть опциональный слой ответа: `answer()` отправляет собранный
`MemoryContext` в локальную LLM (Ollama или llama.cpp). Ядро проекта работает и без LLM:
`ask()` показывает тот же контекст как готовый промпт.

## Архитектура

```
реплика пользователя
        │
        ▼
 RuleBasedExtractor ──► ExtractedFact(predicate, object, confidence)
        │
        ▼
   Consolidator ──► inserted / reinforced / merged / superseded
        │                    │
        │                    └──► fact_history (старые значения не теряются)
        ▼
   Decayer ──► score = recency + importance + relevance ──► archived
        │
        ▼
 ContextBuilder ──► профиль + релевантные факты + последние реплики
        │                    в пределах token_budget
        ▼
 MemoryContext(text, tokens) ──► ask(): вывод контекста
        │                    └──► answer(): OllamaGenerator -> локальная LLM
        ▼
   AnswerResult(query, context, answer)
```

Ключевое решение: факт хранится не как свободный текст, а как слот
`(subject, predicate, object)`. Это делает дедупликацию, слияние и
противоречия детерминированными и тестируемыми.

## Структура

```
проект/
├── chat.py                     # CLI: remember / ask / answer / profile / history / stats / demo
├── web_chat.py                 # локальный веб-чат: GET / + POST /api/chat
├── scenarios/demo_dialog.json  # демо-сценарий
├── models/                     # локальные веса LLM (Ollama/llama.cpp, игнорируется git)
├── memory_system/
│   ├── assistant.py            # оркестратор: remember → recall/answer → profile/stats
│   ├── config.py               # MemoryConfig + параметры decay/consolidation/context/LLM
│   ├── embeddings.py           # Embedder: sentence-transformers или hashing-фолбэк
│   ├── extractor.py            # rule-based извлечение фактов из русских реплик
│   ├── consolidation.py        # дедупликация, слияние, противоречия
│   ├── decay.py                # забывание нерелевантного
│   ├── profile.py              # долгосрочный профиль пользователя
│   ├── context_builder.py      # сборка контекста с бюджетом токенов
│   ├── generator.py            # OllamaGenerator/LlamaCppServerGenerator: MemoryContext -> ответ LLM
│   ├── models.py               # Predicate, Fact, Episode, FactHistoryEntry
│   └── store.py                # SQLite: episodes, facts, fact_history
├── tests/                      # pytest, офлайн-эмбеддер и fake-генератор
├── pyproject.toml              # pytest/ruff config
└── requirements.txt            # torch cu128 + sentence-transformers + pytest/ruff
```

## Установка Python-окружения

Python 3.11+, Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` ставит `torch==2.9.1+cu128` из индекса PyTorch (GPU-сборка,
проверено под RTX 4050 Laptop). Если GPU не нужен, замените блок torch на
CPU-вариант из комментария в файле.

## Локальная LLM для `answer`

Для ответов используется локальная LLM — **Ollama** или **llama.cpp**. Оба
варианта общаются с проектом по HTTP, поэтому Python-зависимостей для генерации
нет. Веса кладём в папку проекта.

### Какую модель брать под RTX 4050 Laptop 6GB

Рекомендация: **`qwen2.5:3b-instruct`**.

- Q4_K_M, ~1.9 ГБ весов; с `num_ctx=4096` вместе с эмбеддингами умещается в 6 ГБ VRAM.
- Хорошо держит русский и мультиязычность, следует инструкциям «отвечай только по контексту».
- `qwen2.5:7b-instruct` (~4.7 ГБ) уже рискованно: останется мало места под
  KV-cache и sentence-transformers. `qwen2.5:1.5b-instruct` можно взять, если
  нужно быстрее/легче, но качество ниже.

### Установка модели в проект

1. Установите Ollama: <https://ollama.com/download/windows>.
2. Полностью закройте Ollama из трея, если он уже запущен.
3. В PowerShell **из корня проекта** задайте папку моделей и поднимите сервер:

```powershell
$env:OLLAMA_MODELS = Join-Path (Get-Location) "models\ollama"
ollama serve
```

Оставьте это окно открытым. В другом терминале из той же папки проекта:

```powershell
$env:OLLAMA_MODELS = Join-Path (Get-Location) "models\ollama"
ollama pull qwen2.5:3b-instruct
ollama list
ollama run qwen2.5:3b-instruct "Привет! Ответь одним словом."
```

Чтобы не задавать переменную каждый раз, сохраните её на уровне пользователя
(после этого перезапустите Ollama):

```powershell
[Environment]::SetEnvironmentVariable(
  "OLLAMA_MODELS",
  (Join-Path (Get-Location) "models\ollama"),
  "User"
)
```

Проверка из проекта:

```powershell
curl http://localhost:11434/api/tags
.\.venv\Scripts\python.exe chat.py answer "Кто я и где живу?" --db memory.db
```

### Альтернатива: GGUF + llama.cpp server — файл модели прямо в проекте

Если хочешь именно `.gguf` в `models/` без Ollama:

1. Скачай Windows CUDA-сборку llama.cpp со страницы релизов
   <https://github.com/ggml-org/llama.cpp/releases>, распакуй и положи
   `llama-server.exe` в `bin\`.
2. Скачай официальный GGUF в `models/` через уже установленный `huggingface_hub`:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-3B-Instruct-GGUF','qwen2.5-3b-instruct-q4_k_m.gguf',local_dir='models')"
```

3. Запусти сервер из корня проекта: модель уходит на GPU, контекст 4096:

```powershell
.\bin\llama-server.exe -m .\models\qwen2.5-3b-instruct-q4_k_m.gguf --alias qwen2.5:3b-instruct --port 8080 -ngl 99 -c 4096 --jinja
```

`--alias` должен совпадать с `--llm-model` (по умолчанию `qwen2.5:3b-instruct`).

4. В другом терминале отвечай через этот backend:

```powershell
.\.venv\Scripts\python.exe chat.py answer "Кто я и где живу?" --db memory.db --llm-backend llamacpp
```

## Использование

CLI по умолчанию использует быстрый офлайн-эмбеддер `hashing` — без скачивания
модели эмбеддингов. Для настоящих мультиязычных эмбеддингов добавьте
`--embedder sentence-transformers` (модель скачается один раз, ~470 МБ).

```powershell
# демо-сценарий: диалог → профиль → статистика → история
python chat.py demo

# записать реплику в память
python chat.py remember "Привет! Меня зовут Алексей, живу в Питере."

# посмотреть контекст, который соберётся для вопроса
python chat.py ask "Кто я и где живу?" --db memory.db

# получить ответ от локальной LLM поверх памяти
python chat.py answer "Кто я и где живу?" --db memory.db
python chat.py answer "Что мне посоветуешь учить?" --db memory.db --llm-model qwen2.5:3b-instruct

# долгосрочный профиль
python chat.py profile --db memory.db

# история вытесненных значений
python chat.py history --db memory.db

# счётчики
python chat.py stats --db memory.db
```

## Веб-чат

Локальный чат с памятью и LLM в браузере:

```powershell
# терминал 1: llama-server (команда выше)
# терминал 2: веб-интерфейс проекта
python web_chat.py --port 8000
# открыть http://127.0.0.1:8000
```

По умолчанию память веб-чата живёт только в памяти процесса: при перезапуске
`web_chat.py` всё забывается, и контекст прошлой сессии не подмешивается. Если
нужна постоянная SQLite-база между запусками, передай `--db memory.db`.

Каждое сообщение сначала проходит через `remember()` (факты извлекаются и
консолидируются), затем `answer()` использует `MemoryContext`. Веб-чат по
умолчанию берёт `hashing`-эмбеддер и backend `llamacpp`; переопределяется через
`MEMORY_EMBEDDER`, `MEMORY_LLM_BACKEND`, `MEMORY_LLM_MODEL`,
`MEMORY_LLAMACPP_HOST`. Сервер однопоточный, потому что у `MemoryStore` одно
SQLite-соединение — для локального чата этого достаточно.

Пример с реальной моделью эмбеддингов:

```powershell
python chat.py demo --embedder sentence-transformers
python chat.py answer "Кем я работаю?" --embedder sentence-transformers
```

## Конфигурация

Через переменные окружения:

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `MEMORY_DB` | путь к SQLite-базе | `memory.db` |
| `MEMORY_EMBEDDER` | `sentence-transformers` / `hashing` | `sentence-transformers` в API, `hashing` в CLI |
| `MEMORY_MODEL` | имя модели sentence-transformers | `paraphrase-multilingual-MiniLM-L12-v2` |
| `MEMORY_DEVICE` | `cuda` / `cpu` / авто | авто |
| `MEMORY_SUBJECT` | субъект памяти | `user` |
| `MEMORY_LLM_BACKEND` | `ollama` / `llamacpp` | `ollama` |
| `MEMORY_LLM_MODEL` | модель/alias LLM | `qwen2.5:3b-instruct` |
| `MEMORY_OLLAMA_HOST` / `OLLAMA_HOST` | адрес Ollama API | `http://localhost:11434` |
| `MEMORY_LLAMACPP_HOST` / `LLAMACPP_HOST` | адрес llama-server | `http://127.0.0.1:8080` |

## Тесты и проверки

```powershell
python -m pytest tests -q
python -m ruff check .
python -m ruff format --check memory_system/ tests/ chat.py
```

Тесты используют `HashingEmbedder` и fake-генератор: они детерминированны,
не требуют сети, GPU, скачивания embedding-модели и запущенного Ollama.

## Ограничения и развитие

- `RuleBasedExtractor` покрывает типовые русские формулировки (имя, возраст,
  город, работа, стек, цель, likes/dislikes, питомец). Для продакшена его можно
  заменить LLM-экстрактором за тем же интерфейсом `Extractor`.
- `HashingEmbedder` — офлайн-фолбэк для тестов и CLI; смысловые синонимы он
  ловит хуже, чем `sentence-transformers`.
- `answer` зависит от внешнего LLM-сервера (Ollama или llama.cpp): если сервер
  или модель недоступны, команда завершится понятной ошибкой, а `ask` продолжит
  показывать контекст.
- Забывание — архивирование, а не удаление: восстановление возможно через
  `include_archived=True` и `fact_history`.
#   M e m o r y - S y s t e m - A I - -  
 
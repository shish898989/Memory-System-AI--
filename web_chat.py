"""Локальный веб-чат поверх Memory System.

Поднимает один HTTP-сервер на stdlib: ``GET /`` отдаёт чат, ``POST /api/chat``
запоминает реплику, собирает MemoryContext и просит локальную LLM ответить.
SQLite-соединение одно, поэтому сервер намеренно однопоточный (``HTTPServer``).
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from memory_system import (
    GeneratorUnavailable,
    HashingEmbedder,
    MemoryAssistant,
    MemoryConfig,
    MemoryStore,
)

ASSISTANT: MemoryAssistant | None = None

INDEX_HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory Chat</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; font-family: system-ui, Segoe UI, Arial, sans-serif;
  background: #0f172a; color: #e5e7eb; height: 100vh;
  display: flex; flex-direction: column;
}
header {
  padding: 14px 18px; border-bottom: 1px solid #1f2937;
  display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;
}
h1 { font-size: 18px; margin: 0; }
#status { color: #93c5fd; font-size: 13px; }
#chat {
  flex: 1; overflow-y: auto; padding: 18px;
  display: flex; flex-direction: column; gap: 12px;
}
.msg {
  max-width: 78%; padding: 10px 13px; border-radius: 14px;
  line-height: 1.45; white-space: pre-wrap; word-wrap: break-word;
}
.user { align-self: flex-end; background: #2563eb; color: #fff; }
.bot { align-self: flex-start; background: #1f2937; border: 1px solid #374151; }
.err { align-self: flex-start; background: #7f1d1d; border: 1px solid #991b1b; }
details {
  align-self: flex-start; max-width: 78%; color: #9ca3af; font-size: 12px;
}
details pre {
  white-space: pre-wrap; background: #111827; border: 1px solid #374151;
  border-radius: 10px; padding: 10px; overflow-x: auto;
}
form {
  display: flex; gap: 10px; padding: 14px 18px;
  border-top: 1px solid #1f2937; background: #111827;
}
input {
  flex: 1; border-radius: 10px; border: 1px solid #374151;
  background: #0b1220; color: #e5e7eb; padding: 11px 13px; font-size: 15px;
}
button {
  border: 0; border-radius: 10px; padding: 0 18px; font-size: 15px;
  background: #2563eb; color: #fff; cursor: pointer;
}
button:disabled { opacity: .55; cursor: wait; }
</style>
</head>
<body>
<header>
  <h1>Memory Chat</h1>
  <div id="status">подключение…</div>
</header>
<div id="chat"></div>
<form id="form">
  <input id="input" autocomplete="off" placeholder="Напиши сообщение…">
  <button id="send" type="submit">Отправить</button>
</form>
<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}
function addMsg(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}
function addContext(context, tokens) {
  const det = document.createElement('details');
  det.innerHTML = '<summary>контекст памяти · ~' + tokens + ' ток.</summary>';
  const pre = document.createElement('pre');
  pre.textContent = context || '(пусто)';
  det.appendChild(pre);
  chat.appendChild(det);
  chat.scrollTop = chat.scrollHeight;
}
async function loadStatus() {
  try {
    const res = await fetch('/api/config');
    const cfg = await res.json();
    statusEl.textContent = cfg.llm_backend + ' · ' + cfg.llm_model +
      ' · db: ' + cfg.db;
  } catch (err) {
    statusEl.textContent = 'offline';
  }
}
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = '';
  addMsg(message, 'user');
  const pending = addMsg('Думаю…', 'bot');
  sendBtn.disabled = true;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message})
    });
    const data = await res.json();
    pending.remove();
    if (!res.ok) {
      addMsg(data.error || ('HTTP ' + res.status), 'err');
    } else {
      addMsg(data.reply || '(пустой ответ)', 'bot');
      if (data.context) addContext(data.context, data.tokens || 0);
    }
  } catch (err) {
    pending.remove();
    addMsg(String(err), 'err');
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});
loadStatus();
input.focus();
</script>
</body>
</html>
"""


def _make_assistant(db_path: Path | None) -> MemoryAssistant:
    config = MemoryConfig.from_env()
    store = None
    if db_path is None:
        # По умолчанию веб-чат живёт только в рамках процесса: при перезапуске
        # память чистая и контекст прошлой сессии не подмешивается.
        config.db_path = Path("memory (in-memory, session-only)")
        store = MemoryStore(keep_in_memory=True)
    else:
        config.db_path = db_path
    # Для локального чата по умолчанию берём быстрый офлайн-эмбеддер.
    # sentence-transformers включается через MEMORY_EMBEDDER=sentence-transformers.
    config.embedder = os.getenv("MEMORY_EMBEDDER", "hashing")
    config.llm_backend = os.getenv("MEMORY_LLM_BACKEND", "llamacpp")
    config.llm_model = os.getenv("MEMORY_LLM_MODEL", "qwen2.5:3b-instruct")
    config.llamacpp_host = (
        os.getenv("MEMORY_LLAMACPP_HOST") or os.getenv("LLAMACPP_HOST") or "http://127.0.0.1:8080"
    )
    if config.embedder == "hashing":
        return MemoryAssistant(config, store=store, embedder=HashingEmbedder())
    return MemoryAssistant(config, store=store)


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP-обработчик: статика чата + JSON API."""

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - имя требует http.server
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_html()
            return
        if path == "/api/config" and ASSISTANT is not None:
            self._send_json(
                {
                    "db": str(ASSISTANT.config.db_path),
                    "embedder": ASSISTANT.config.embedder,
                    "llm_backend": ASSISTANT.config.llm_backend,
                    "llm_model": ASSISTANT.config.llm_model,
                }
            )
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - имя требует http.server
        path = urlparse(self.path).path
        if path != "/api/chat" or ASSISTANT is None:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            message = str(payload.get("message", "")).strip()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": f"bad json: {exc}"}, status=400)
            return
        if not message:
            self._send_json({"error": "empty message"}, status=400)
            return

        ingest = ASSISTANT.remember(message)
        try:
            result = ASSISTANT.answer(message)
        except GeneratorUnavailable as exc:
            self._send_json({"error": str(exc)}, status=503)
            return
        except Exception as exc:  # noqa: BLE001 - локальный чат: показать ошибку
            self._send_json({"error": str(exc)}, status=500)
            return
        self._send_json(
            {
                "reply": result.answer,
                "context": result.context.text,
                "tokens": result.context.tokens,
                "ingest": {"episode_id": ingest.episode_id, "actions": ingest.actions},
            }
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Веб-чат поверх Memory System")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite-файл для постоянной памяти; без флага память только в сессии",
    )
    args = parser.parse_args()

    global ASSISTANT
    ASSISTANT = _make_assistant(args.db)
    server = HTTPServer((args.host, args.port), ChatHandler)
    print(f"Memory Chat: http://{args.host}:{args.port}")
    print(f"LLM backend: {ASSISTANT.config.llm_backend} / {ASSISTANT.config.llm_model}")
    print(f"SQLite DB:   {ASSISTANT.config.db_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        ASSISTANT.close()


if __name__ == "__main__":
    main()

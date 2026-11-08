"""Local long-term memory for any model: retain exchanges, recall relevant ones.

Stored only on the user's machine (SQLite + FTS5). Optionally mirrors to a
Hindsight server when one is configured. No model internals are involved, so
this works with every backend.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path

_WORD = re.compile(r"[A-Za-z0-9_]{3,}")
_STOP = set("the and for are but not you your with this that have from was were what when where which who will would "
            "can could should about into than then them they their there these those been being just like also very "
            "how why does did doing done its it's our ours yours his her hers him she he".split())


def _fts_query(text: str, limit: int = 24) -> str:
    words = []
    for word in _WORD.findall(text.lower()):
        if word not in _STOP and word not in words:
            words.append(word)
        if len(words) >= limit:
            break
    return " OR ".join(f'"{w}"' for w in words)


class Memory:
    def __init__(self, path: Path, hindsight_url: str | None = None, bank: str = "lens"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hindsight_url = (hindsight_url or "").rstrip("/") or None
        self.bank = bank
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memories(
                id INTEGER PRIMARY KEY, created REAL NOT NULL, model TEXT, user TEXT NOT NULL, assistant TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(user, assistant, content='memories', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, user, assistant) VALUES (new.id, new.user, new.assistant); END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, user, assistant) VALUES ('delete', old.id, old.user, old.assistant); END;
            CREATE TABLE IF NOT EXISTS inner_states(
                memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                space TEXT NOT NULL, vector TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS inner_states_ad AFTER DELETE ON memories BEGIN
                DELETE FROM inner_states WHERE memory_id = old.id; END;
        """)
        self._db.commit()

    # ------------------------------------------------------------- local
    def retain(self, user: str, assistant: str, model: str = "") -> int:
        user, assistant = user.strip(), assistant.strip()
        if not user or not assistant:
            return 0
        with self._lock:
            cur = self._db.execute("INSERT INTO memories(created, model, user, assistant) VALUES (?,?,?,?)",
                                   (time.time(), model, user[:8000], assistant[:8000]))
            self._db.commit()
            row_id = cur.lastrowid
        if self.hindsight_url:
            threading.Thread(target=self._hindsight_retain, args=(user, assistant, row_id), daemon=True).start()
        return row_id

    # ------------------------------------------------------------- inner-state memory
    def retain_state(self, memory_id: int, space: str, vector: list[float]) -> None:
        """Store an answer's internal-state fingerprint (this device's 16-number projections of the running mean).
        `space` names the model + device key, so only fingerprints from the same projection are ever compared."""
        if not memory_id or not vector:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO inner_states VALUES (?,?,?)",
                             (memory_id, space, json.dumps([round(float(x), 6) for x in vector])))
            self._db.commit()

    def similar_states(self, space: str, vector: list[float], exclude: int = 0, limit: int = 3) -> list[dict]:
        """Earlier answers whose internal state was most similar (cosine), same model and device key only."""
        import math
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        with self._lock:
            rows = self._db.execute("SELECT s.memory_id, s.vector, m.user, m.model FROM inner_states s "
                                    "JOIN memories m ON m.id = s.memory_id WHERE s.space = ? AND s.memory_id != ?",
                                    (space, exclude)).fetchall()
        scored = []
        for memory_id, raw, user, model in rows:
            other = json.loads(raw)
            if len(other) != len(vector):
                continue
            dot = sum(a * b for a, b in zip(vector, other))
            other_norm = math.sqrt(sum(x * x for x in other)) or 1.0
            scored.append({"id": memory_id, "user": user, "model": model, "similarity": round(dot / norm / other_norm, 3)})
        return sorted(scored, key=lambda r: -r["similarity"])[:limit]

    def recall(self, query: str, limit: int = 4) -> list[dict]:
        match = _fts_query(query)
        if not match:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT m.id, m.created, m.user, m.assistant FROM memories_fts f JOIN memories m ON m.id = f.rowid "
                "WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT ?", (match, limit * 3)).fetchall()
        # Relevance floor: one shared word (e.g. "name") is not enough to bring a memory back.
        terms = set(_WORD.findall(query.lower())) - _STOP
        need = 1 if len(terms) <= 2 else 2
        found, seen = [], set()
        for r in rows:
            words = set(_WORD.findall((r[2] + " " + r[3]).lower()))
            key = " ".join(r[2].lower().split())
            if key in seen:            # the same thing said twice is one memory
                continue
            if len(terms & words) >= need:
                seen.add(key)
                found.append({"id": r[0], "created": r[1], "user": r[2], "assistant": r[3], "source": "local"})
        found = found[:limit]
        if self.hindsight_url:
            found += self._hindsight_recall(query)
        return found

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT id, created, model, user, assistant FROM memories ORDER BY id DESC LIMIT ?",
                                    (limit,)).fetchall()
        return [{"id": r[0], "created": r[1], "model": r[2], "user": r[3], "assistant": r[4]} for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def forget(self, memory_id: int | None = None) -> int:
        with self._lock:
            if memory_id is None:
                n = self._db.execute("DELETE FROM memories").rowcount
            else:
                n = self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,)).rowcount
            self._db.commit()
        return n

    @staticmethod
    def as_context(memories: list[dict]) -> str:
        if not memories:
            return ""
        lines = ["Background notes from earlier conversations with this user. Use them only if relevant; "
                 "do not repeat or quote them unless asked:"]
        for m in memories:
            note = f"- The user once said: \"{m['user'][:400]}\""
            if len(m.get("assistant", "")) > 40:
                note += f" (you answered about: {m['assistant'][:200]})"
            lines.append(note)
        return "\n".join(lines)

    # ------------------------------------------------------------- Hindsight (optional)
    def _post(self, suffix: str, payload: dict, timeout: float = 8.0):
        req = urllib.request.Request(f"{self.hindsight_url}/v1/default/banks/{self.bank}/{suffix}",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    def _hindsight_retain(self, user: str, assistant: str, row_id: int) -> None:
        try:
            self._post("memories", {"async": True, "items": [{
                "content": f"User: {user}\nAssistant: {assistant}", "document_id": f"lens-{row_id}",
                "context": "Local chat exchange recorded by Proprioceptive Lens."}]})
        except (OSError, ValueError):
            pass  # local copy is authoritative; Hindsight is an optional mirror

    def _hindsight_recall(self, query: str) -> list[dict]:
        try:
            data = self._post("memories/recall", {"query": query[:1400], "budget": "low", "max_tokens": 800})
        except (OSError, ValueError):
            return []
        items = data.get("results") or data.get("memories") or []
        out = []
        for item in items[:3]:
            text = item.get("text") or item.get("content") if isinstance(item, dict) else str(item)
            if text:
                out.append({"id": None, "created": None, "user": str(text)[:400], "assistant": "", "source": "hindsight"})
        return out

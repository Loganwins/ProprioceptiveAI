"""Ollama backend: streamed chat with per-token probabilities.

Every streamed token carries its probability and the top alternatives the
model considered, which powers the live confidence view. Works with any model
an Ollama install serves; no model internals or proprietary components.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Iterator


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, url: str = "http://127.0.0.1:11434"):
        self.url = url.rstrip("/")

    def _request(self, path: str, payload: dict | None = None, timeout: float = 10.0):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"},
                                     method="POST" if payload is not None else "GET")
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.URLError as exc:
            raise OllamaError(f"Ollama is not reachable at {self.url} ({exc.reason if hasattr(exc, 'reason') else exc})")

    def available(self) -> bool:
        try:
            with self._request("/api/version", timeout=2) as response:
                return response.status == 200
        except OllamaError:
            return False

    def models(self) -> list[str]:
        with self._request("/api/tags") as response:
            data = json.load(response)
        names = [m["name"] for m in data.get("models", [])]
        return [n for n in names if "embed" not in n]

    def chat_stream(self, model: str, messages: list[dict], options: dict | None = None,
                    top_alternatives: int = 3) -> Iterator[dict]:
        """Yield {'type': 'token', ...} per token, then {'type': 'done', ...}."""
        payload = {"model": model, "messages": messages, "stream": True, "logprobs": True,
                   "top_logprobs": top_alternatives, "options": options or {}}
        with self._request("/api/chat", payload, timeout=600) as response:
            for raw in response:
                if not raw.strip():
                    continue
                chunk = json.loads(raw)
                if chunk.get("error"):
                    raise OllamaError(chunk["error"])
                text = (chunk.get("message") or {}).get("content", "")
                probs = chunk.get("logprobs") or []
                if probs:
                    for item in probs:
                        yield token_event(item)
                elif text:
                    yield {"type": "token", "text": text, "p": None, "alternatives": []}
                if chunk.get("done"):
                    yield {"type": "done", "tokens": chunk.get("eval_count"),
                           "seconds": (chunk.get("eval_duration") or 0) / 1e9,
                           "prompt_tokens": chunk.get("prompt_eval_count")}
                    return


def token_event(item: dict) -> dict:
    """Convert one Ollama logprob record into a UI token event."""
    p = math.exp(item.get("logprob", 0.0))
    alternatives = [{"text": alt.get("token", ""), "p": math.exp(alt.get("logprob", -99.0))}
                    for alt in item.get("top_logprobs") or []
                    if alt.get("token") != item.get("token") and alt.get("token", "").strip()]
    return {"type": "token", "text": item.get("token", ""), "p": p, "alternatives": alternatives[:3]}

"""`lens mcp`: Lens as an MCP server (stdio), so coding agents can ask local models and see how sure they were.

Add to an MCP client, e.g. Claude Code:   claude mcp add lens -- lens mcp
Tools:
  ask_local_model  - ask a local model; returns the answer, its certainty, and the words where it hesitated
  list_local_models - models Lens can use
  search_memory    - search Lens's private memory of past conversations
Standard library only. Messages are newline-delimited JSON-RPC 2.0 on stdin/stdout.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from . import __version__
from .server import LensApp

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

TOOLS = [
    {"name": "ask_local_model",
     "description": "Ask a model running locally (Ollama, or a Lens deep model) and get its answer together with how "
                    "certain it was: overall certainty, the words it hesitated on and what it almost said instead, "
                    "plus Behaviour probe scores when deep mode is on. Use it for a second opinion from a local model "
                    "or to check where a model is unsure.",
     "inputSchema": {"type": "object", "properties": {
         "prompt": {"type": "string", "description": "The question or instruction for the local model."},
         "model": {"type": "string", "description": "Model name (see list_local_models). Optional."},
         "use_memory": {"type": "boolean", "description": "Recall and save Lens memories (default true)."},
         "max_tokens": {"type": "integer", "description": "Maximum answer length in tokens (default 512)."}},
         "required": ["prompt"]}},
    {"name": "list_local_models",
     "description": "List the local models Lens can use.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "check_claims",
     "description": "Truth check (Lens Deluxe): a local reader model reads each factual claim and Proprioceptive AI's "
                    "internal-state probe returns a fabrication-risk reading (low / elevated / high). Validated for "
                    "factual relations: who/what/where (capitals, authors, discoveries, landmarks, currencies, symbols). "
                    "Catches blatant mismatches, NOT plausible-sounding errors; NOT for numbers or dates. Use it on claims from your own answer before presenting them as fact; "
                    "pass one claim per item.",
     "inputSchema": {"type": "object", "properties": {
         "claims": {"type": "array", "items": {"type": "string"}, "description": "One factual claim per item."},
         "text": {"type": "string", "description": "Alternatively, prose to split into sentence claims."}}}},
    {"name": "search_memory",
     "description": "Search Lens's private memory of earlier conversations with local models.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}, "limit": {"type": "integer", "description": "default 5"}},
         "required": ["query"]}},
]


def default_model(models: list[str]) -> str | None:
    from .terminal import Terminal
    return Terminal._default_model(models)


def ask(app: LensApp, prompt: str, model: str | None, use_memory: bool = True, max_tokens: int = 512) -> dict:
    model = model or default_model(app.status()["models"])
    if not model:
        raise ValueError("No local model found. Start Ollama and `ollama pull qwen3:4b`, or run Lens with --deep.")
    send, recalled, user_text = app.prepare([{"role": "user", "content": prompt}], use_memory)
    tokens, behaviour, info = [], None, {}
    for event in app.chat_events(model, send, recalled, {"num_predict": max_tokens}):
        kind = event["type"]
        if kind == "token" and not event.get("thinking"):
            tokens.append(event)
        elif kind == "behaviour":
            behaviour = event["scores"][-1] if event["scores"] else behaviour
        elif kind == "behaviour_status":
            info = {p["name"]: p.get("label", p["name"]) for p in event.get("probes", [])}
        elif kind == "error":
            raise RuntimeError(event["message"])
    answer = "".join(t["text"] for t in tokens)
    if use_memory and answer:
        app.memory.retain(user_text, answer, model)
    known = [t["p"] for t in tokens if t.get("p") is not None]
    hesitations = sorted((t for t in tokens if t.get("p") is not None and t["text"].strip() and t["p"] < 0.9),
                         key=lambda t: t["p"])[:8]
    return {
        "model": model,
        "answer": answer,
        "certainty": round(sum(known) / len(known), 3) if known else None,
        "uncertain_words": sum(p < 0.5 for p in known),
        "hesitations": [{"word": t["text"].strip(), "p": round(t["p"], 3),
                         "almost": [{"word": a["text"].strip(), "p": round(a["p"], 3)} for a in t.get("alternatives", [])]}
                        for t in hesitations],
        "behaviour": ({info.get(k, k): round(v, 2) for k, v in behaviour.items()} if behaviour else None),
        "memories_used": len(recalled),
    }


def ask_text(result: dict) -> str:
    lines = [result["answer"].strip(), "", f"— {result['model']}"]
    if result["certainty"] is not None:
        lines[-1] += f" · certainty {round(100 * result['certainty'])}% · {result['uncertain_words']} uncertain words"
    if result["hesitations"]:
        lines.append("Hesitated on: " + "; ".join(
            f"\"{h['word']}\" {round(100 * h['p'])}%" + (" (almost: " + ", ".join(
                f"\"{a['word']}\" {round(100 * a['p'])}%" for a in h["almost"][:3]) + ")" if h["almost"] else "")
            for h in result["hesitations"]))
    if result["behaviour"]:
        lines.append("Behaviour (σ above typical): " + ", ".join(
            f"{k} {v:+.1f}" for k, v in sorted(result["behaviour"].items(), key=lambda kv: -kv[1])))
    return "\n".join(lines)


class Server:
    def __init__(self, app: LensApp):
        self.app = app

    def call(self, name: str, args: dict) -> dict:
        if name == "ask_local_model":
            prompt = str(args.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("prompt is required")
            result = ask(self.app, prompt, args.get("model"), bool(args.get("use_memory", True)),
                         int(args.get("max_tokens", 512)))
            return {"content": [{"type": "text", "text": ask_text(result)}], "structuredContent": result}
        if name == "list_local_models":
            models = self.app.status()["models"]
            return {"content": [{"type": "text", "text": "\n".join(models) or "No models found (is Ollama running?)"}],
                    "structuredContent": {"models": models}}
        if name == "check_claims":
            from .server import split_claims
            claims = args.get("claims") or split_claims(str(args.get("text", "")))
            result = self.app.check_claims([str(c) for c in claims])
            lines = [f"[{(r['band'] or '?').upper():8}] {r['claim']}" + (f"  (risk {r['risk']:+.1f})" if r.get('risk') is not None else "")
                     for r in result["claims"]]
            ops = result.get("operating_points") or {}
            lines += ["", "Scope: " + (result.get("scope") or ""),
                      f"Calibration: 'high' flags ~{round(100 * ops.get('high', 0))}% of false relation claims with ~2.5% "
                      f"false alarms; 'elevated' ~{round(100 * ops.get('elevated', 0))}% with ~10%. It catches blatant mismatches, "
                      "not plausible-sounding errors or misconceptions: 'low' is not confirmation."]
            return {"content": [{"type": "text", "text": "\n".join(lines)}], "structuredContent": result}
        if name == "search_memory":
            rows = self.app.memory.recall(str(args.get("query", "")), limit=int(args.get("limit", 5)))
            text = "\n\n".join(f"[{r.get('model', '')}] Q: {r['user']}\nA: {r['assistant']}" for r in rows)
            return {"content": [{"type": "text", "text": text or "No matching memories."}],
                    "structuredContent": {"memories": rows}}
        raise KeyError(name)

    def handle(self, msg: dict) -> dict | None:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:                       # notification (e.g. notifications/initialized): no reply
            return None
        try:
            if method == "initialize":
                asked = (msg.get("params") or {}).get("protocolVersion")
                result = {"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "proprioceptive-lens", "version": __version__},
                          "instructions": "Ask local models and see how certain they were, word by word."}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = msg.get("params") or {}
                try:
                    result = self.call(params.get("name", ""), params.get("arguments") or {})
                except KeyError:
                    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {params.get('name')}"}}
                except Exception as exc:     # tool errors are reported to the model, not as protocol errors
                    result = {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True}
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
            return {"jsonrpc": "2.0", "id": mid, "result": result}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(exc)}}

    def run(self, stdin=sys.stdin, stdout=sys.stdout) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32700, "message": "parse error"}}) + "\n")
                stdout.flush()
                continue
            with contextlib.redirect_stdout(sys.stderr):    # stray prints must never corrupt the protocol stream
                reply = self.handle(msg)
            if reply is not None:
                stdout.write(json.dumps(reply) + "\n")
                stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lens mcp", description="Run Lens as an MCP server on stdio.")
    parser.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--deep", default=os.environ.get("LENS_DEEP_MODEL"))
    parser.add_argument("--hindsight", default=os.environ.get("LENS_HINDSIGHT_URL"))
    parser.add_argument("--data", default=os.environ.get("LENS_DATA", str(Path.home() / ".proprioceptive-lens")))
    parser.add_argument("--cloud", default=os.environ.get("LENS_CLOUD_URL"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    ollama = args.ollama if args.ollama.startswith("http") else "http://" + args.ollama
    with contextlib.redirect_stdout(sys.stderr):
        app = LensApp(Path(args.data), ollama, args.hindsight, deep_model=args.deep, cloud_url=args.cloud)
    Server(app).run()
    return 0

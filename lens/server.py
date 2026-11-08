"""Local Lens server: the app, live chat streaming, memory, OpenAI-compatible proxy.

Standard library only. Binds 127.0.0.1 and rejects foreign Host headers, so
web pages cannot drive it (DNS-rebinding protection).
"""
from __future__ import annotations

import json
import mimetypes
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .cloud import Cloud, CloudError
from .memory import Memory
from .ollama import Ollama, OllamaError

DEEP_PREFIX = "deep:"
SCORE_EVERY = 8

STATIC = Path(__file__).resolve().parent / "static"
SYSTEM_PROMPT = "You are a helpful assistant running locally on the user's computer."


class LensApp:
    def __init__(self, data_dir: Path, ollama_url: str, hindsight_url: str | None = None,
                 deep_model: str | None = None, cloud_url: str | None = None):
        self.data_dir = Path(data_dir)
        self.ollama = Ollama(ollama_url)
        self.memory = Memory(self.data_dir / "memory.sqlite3", hindsight_url=hindsight_url)
        self.cloud = Cloud(self.data_dir, cloud_url)
        self.deep = None
        self.deep_error = None
        self.deep_loading = None            # Hugging Face model being loaded by "See inside"
        self.cloud_error = None
        self.steer_settings: dict[str, tuple[str, float]] = {}   # behaviour -> (direction, amount), Deluxe
        if deep_model:
            try:
                from .hf import DeepModel
                self.deep = DeepModel(deep_model)
            except Exception as exc:  # missing torch/transformers or model download failure
                self.deep_error = f"{type(exc).__name__}: {exc}"
        self.started = time.time()

    def ensure_cloud(self) -> bool:
        """Load this device's scrambled matrices for the deep model, if the user opted in."""
        if not (self.deep and self.cloud.enabled):
            return False
        if self.cloud.config and self.cloud.config.get("model_id") == self.deep.model_id:
            return True
        try:
            self.deep.set_projections(self.cloud.load_config(self.deep.model_id))
            self.cloud_error = None
            self.apply_steering()
            return True
        except CloudError as exc:
            self.cloud_error = str(exc)
            self.deep.set_projections({})
            return False

    def apply_steering(self) -> None:
        """Fetch this device's directives for the chosen behaviours and hand them to the deep model."""
        if not self.deep:
            return
        directives = []
        for behaviour, (direction, amount) in list(self.steer_settings.items()):
            if amount > 0:
                directives.append(self.cloud.steer(behaviour, direction, amount))
        self.deep.set_steering(directives)

    def set_steer(self, behaviour: str, direction: str, amount: float) -> None:
        if direction not in ("more", "less"):
            raise ValueError("direction must be 'more' or 'less'")
        previous = dict(self.steer_settings)
        if amount > 0:
            self.steer_settings[behaviour] = (direction, float(amount))
        else:
            self.steer_settings.pop(behaviour, None)
        try:
            self.apply_steering()
        except CloudError:
            self.steer_settings = previous
            raise

    @staticmethod
    def deep_available() -> bool:
        import importlib.util
        return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))

    def load_deep(self, model_id: str) -> None:
        """'See inside': load a Hugging Face model in the background (first time downloads it), replacing any other."""
        if self.deep_loading:
            raise ValueError(f"already loading {self.deep_loading}")
        if not self.deep_available():
            raise ValueError('needs the deep extra: pip install "proprioceptive-lens[deep]"')

        def work():
            try:
                from .hf import DeepModel
                if self.deep is not None:               # free the previous model's memory first
                    old, self.deep = self.deep, None
                    del old
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                self.cloud.config = None                # the next chat fetches this model's pack
                self.deep = DeepModel(model_id)
                self.deep_error = None
            except Exception as exc:                    # surfaced in status; never crash the app
                self.deep_error = f"{type(exc).__name__}: {exc}"
            finally:
                self.deep_loading = None

        self.deep_loading = model_id
        self.deep_error = None
        threading.Thread(target=work, daemon=True).start()

    def status(self) -> dict:
        ok = self.ollama.available()
        models = []
        if ok:
            try:
                models = self.ollama.models()
            except (OllamaError, OSError, ValueError):
                ok = False
        from .catalog import hf_for
        deep_options = {m: hf_for(m) for m in models if hf_for(m)}
        if self.deep:
            models = [DEEP_PREFIX + self.deep.model_id] + models
        if self.deep and self.cloud.enabled:
            self.ensure_cloud()            # loads the pack config once, so the app can show steering before a chat
        config = self.cloud.config or {}
        cloud = {"enabled": self.cloud.enabled, "url": self.cloud.url, "error": self.cloud_error,
                 "probes": config.get("probes", []), "plan": config.get("plan", "free"),
                 "steerable": config.get("steerable", []), "upgrade_url": config.get("upgrade_url"),
                 "steering": {b: {"direction": d, "amount": a} for b, (d, a) in self.steer_settings.items()},
                 "supported_models": self.cloud.supported_models()}
        return {"app": "proprioceptive-lens", "version": __version__, "backend": "ollama",
                "backend_url": self.ollama.url, "backend_ok": ok or bool(self.deep), "models": models,
                "memory_count": self.memory.count(), "hindsight": bool(self.memory.hindsight_url),
                "deep_model": self.deep.model_id if self.deep else None, "deep_error": self.deep_error,
                "deep_layers": self.deep.n_layers if self.deep else 0, "deep_options": deep_options,
                "deep_loading": self.deep_loading, "deep_available": self.deep_available(), "cloud": cloud}

    def prepare(self, messages: list[dict], use_memory: bool) -> tuple[list[dict], list[dict], str]:
        """Return (messages to send, recalled memories, latest user text)."""
        user_text = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        recalled = self.memory.recall(user_text) if use_memory and user_text else []
        system = SYSTEM_PROMPT
        context = Memory.as_context(recalled)
        if context:
            system += "\n\n" + context
        clean = [{"role": m["role"], "content": str(m.get("content", ""))} for m in messages
                 if m.get("role") in {"user", "assistant", "system"}]
        if not clean or clean[0]["role"] != "system":
            clean.insert(0, {"role": "system", "content": system})
        else:
            clean[0] = {"role": "system", "content": clean[0]["content"] + ("\n\n" + context if context else "")}
        return clean, recalled, user_text


def _chat_events(self, model: str, send: list[dict], recalled: list[dict], options: dict | None = None):
    """Shared chat engine for the web app and the terminal: yields UI events."""
    deep = model.startswith(DEEP_PREFIX)
    scoring = deep and self.ensure_cloud()
    self.last_state = None
    yield {"type": "memory", "items": recalled}
    if deep:
        yield {"type": "behaviour_status", "enabled": scoring, "error": self.cloud_error,
               "probes": (self.cloud.config or {}).get("probes", []) if scoring else []}
    pending = []
    try:
        unsteered = deep and (options or {}).get("steer") is False      # "before" side of a before/after
        saved = self.deep.steering if unsteered else None
        if unsteered:
            self.deep.steering = {}
        stream = (self.deep.chat_stream(send, max_new_tokens=int((options or {}).get("num_predict", 512)))
                  if deep else self.ollama.chat_stream(model, send,
                                                       options={k: v for k, v in (options or {}).items() if k != "steer"}))
        for event in stream:
            if event["type"] == "token" and "sigma" in event:
                step = {"sigma": event.pop("sigma"), "norm": event.pop("norm")}
                pending.append(step)
                self.last_state = step                    # final running-mean fingerprint of this answer
            yield event
            if scoring and (len(pending) >= SCORE_EVERY or (event["type"] == "done" and pending)):
                try:
                    result = self.cloud.score(pending)
                    yield {"type": "behaviour", "scores": result["scores"], "soft": result.get("soft", 2.0),
                           "hard": result.get("hard", 4.0), "used_today": result.get("used_today"),
                           "limit": result.get("limit")}
                except CloudError as exc:
                    yield {"type": "behaviour_error", "message": str(exc)}
                pending = []
    except (OllamaError, OSError, ValueError) as exc:
        yield {"type": "error", "message": str(exc)}
    except Exception as exc:  # never end a stream silently: the user must see what failed
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        if unsteered:
            self.deep.steering = saved


LensApp.chat_events = _chat_events


def make_handler(app: LensApp, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost"}

    class Handler(BaseHTTPRequestHandler):
        server_version = f"ProprioceptiveLens/{__version__}"

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        # ------------------------------------------------------------ helpers
        def _host_ok(self) -> bool:
            if self.headers.get("Host", "") not in allowed_hosts:
                self._json(403, {"error": "forbidden host"})
                return False
            return True

        def _json(self, status: int, data) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 2_000_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def _sse_start(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _sse(self, data: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
            self.wfile.flush()

        # ------------------------------------------------------------ routes
        def do_GET(self):
            if not self._host_ok():
                return
            path = urlsplit(self.path).path
            if path == "/api/status":
                return self._json(200, app.status())
            if path == "/api/memory":
                return self._json(200, {"count": app.memory.count(), "items": app.memory.recent(100)})
            if path == "/v1/models":
                try:
                    names = app.ollama.models()
                except OllamaError as exc:
                    return self._json(502, {"error": str(exc)})
                return self._json(200, {"object": "list", "data": [
                    {"id": n, "object": "model", "owned_by": "local"} for n in names]})
            if path in ("/", "/index.html"):
                path = "/index.html"
            target = (STATIC / path.lstrip("/")).resolve()
            if STATIC in target.parents and target.is_file():
                body = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            self._json(404, {"error": "not found"})

        def do_DELETE(self):
            if not self._host_ok():
                return
            parts = urlsplit(self.path)
            if parts.path == "/api/memory":
                query = parse_qs(parts.query)
                memory_id = int(query["id"][0]) if "id" in query else None
                return self._json(200, {"deleted": app.memory.forget(memory_id)})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok():
                return
            path = urlsplit(self.path).path
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})
            if path == "/api/chat":
                return self._chat(body)
            if path == "/api/cloud/enable":
                try:
                    app.cloud.enable()
                    app.ensure_cloud()
                except CloudError as exc:
                    return self._json(502, {"error": str(exc)})
                return self._json(200, app.status()["cloud"])
            if path == "/api/cloud/disable":
                app.cloud.disable()
                if app.deep:
                    app.deep.set_projections({})
                return self._json(200, app.status()["cloud"])
            if path == "/api/deep/load":
                try:
                    app.load_deep(str(body.get("model", "")).strip())
                except ValueError as exc:
                    return self._json(400, {"error": str(exc)})
                return self._json(202, {"loading": app.deep_loading})
            if path == "/api/deluxe/trial":
                try:
                    result = app.cloud.start_trial(str(body.get("email", "")))
                except CloudError as exc:
                    return self._json(400, {"error": str(exc)})
                return self._json(200, {"message": result.get("message", "Check your email for the activation token.")})
            if path == "/api/deluxe/activate":
                try:
                    result = app.cloud.activate(str(body.get("activation_token", "")))
                except CloudError as exc:
                    return self._json(402, {"error": str(exc)})
                return self._json(200, {**result, "cloud": app.status()["cloud"]})
            if path == "/api/steer":
                if not app.deep or not app.ensure_cloud():
                    return self._json(400, {"error": "steering needs deep mode with Behaviour switched on"})
                try:
                    app.set_steer(str(body.get("behaviour", "")), str(body.get("direction", "more")),
                                  float(body.get("amount", 0)))
                except (CloudError, ValueError) as exc:
                    return self._json(402 if "Deluxe" in str(exc) else 400, {"error": str(exc)})
                return self._json(200, app.status()["cloud"])
            if path == "/v1/chat/completions":
                return self._openai(body)
            self._json(404, {"error": "not found"})

        def _chat(self, body: dict) -> None:
            model = body.get("model")
            messages = body.get("messages") or []
            if not model or not messages:
                return self._json(400, {"error": "model and messages are required"})
            send, recalled, user_text = app.prepare(messages, bool(body.get("use_memory", True)))
            if model.startswith(DEEP_PREFIX) and not app.deep:
                return self._json(400, {"error": "deep model not loaded"})
            self._sse_start()
            reply = []
            for event in app.chat_events(model, send, recalled, body.get("options")):
                if event["type"] == "token":
                    reply.append(event["text"])
                self._sse(event)
                if event["type"] == "error":
                    return
            if body.get("use_memory", True):
                row_id = app.memory.retain(user_text, "".join(reply), model)
                self._sse({"type": "retained", "memory_count": app.memory.count()})
                state, config = getattr(app, "last_state", None), app.cloud.config or {}
                if row_id and state and config and (body.get("options") or {}).get("steer") is not False:
                    space = f"{config.get('model_id')}|{app.cloud.creds.get('device_id')}|{config.get('key_version')}"
                    vector = [x for layer in sorted(state["sigma"], key=int) for x in state["sigma"][layer]]
                    similar = app.memory.similar_states(space, vector, exclude=row_id)
                    app.memory.retain_state(row_id, space, vector)
                    if similar:
                        self._sse({"type": "similar_states", "items": similar})

        def _openai(self, body: dict) -> None:
            """Minimal OpenAI-compatible endpoint so any chat app gains memory."""
            model = body.get("model")
            messages = body.get("messages") or []
            if not model or not messages:
                return self._json(400, {"error": {"message": "model and messages are required"}})
            send, _, user_text = app.prepare(messages, True)
            completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            reply = []
            try:
                if body.get("stream"):
                    self._sse_start()
                    for event in app.ollama.chat_stream(model, send):
                        if event["type"] != "token":
                            continue
                        reply.append(event["text"])
                        self._sse({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                                   "model": model, "choices": [{"index": 0, "delta": {"content": event["text"]},
                                                                "finish_reason": None}]})
                    self._sse({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                               "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    for event in app.ollama.chat_stream(model, send):
                        if event["type"] == "token":
                            reply.append(event["text"])
                    self._json(200, {"id": completion_id, "object": "chat.completion", "created": created,
                                     "model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "".join(reply)}}]})
            except (OllamaError, OSError, ValueError) as exc:
                if not body.get("stream"):
                    return self._json(502, {"error": {"message": str(exc)}})
                return
            app.memory.retain(user_text, "".join(reply), model)

    return Handler


def serve(app: LensApp, port: int = 7777) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, port))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

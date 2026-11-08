import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from lens.memory import Memory
from lens.ollama import token_event
from lens.server import LensApp, serve


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = Memory(Path(self.tmp.name) / "m.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_recalls_relevant_memory_across_conversations(self):
        self.mem.retain("My dog's name is Biscuit and she is a corgi.", "Got it.")
        hits = self.mem.recall("What breed is my dog Biscuit?")
        self.assertEqual(len(hits), 1)
        self.assertIn("corgi", hits[0]["user"])

    def test_single_generic_shared_word_is_not_enough(self):
        self.mem.retain("My dog's name is Biscuit and she is a corgi.", "Got it.")
        self.assertEqual(self.mem.recall("Name a surprising fact about octopuses in two sentences."), [])

    def test_forget_one_and_all(self):
        a = self.mem.retain("alpha beta gamma", "reply one")
        self.mem.retain("delta epsilon zeta", "reply two")
        self.assertEqual(self.mem.forget(a), 1)
        self.assertEqual(self.mem.count(), 1)
        self.mem.forget()
        self.assertEqual(self.mem.count(), 0)

    def test_context_is_framed_as_notes_not_a_script(self):
        text = Memory.as_context([{"user": "I live in Lisbon", "assistant": "OK"}])
        self.assertIn("do not repeat", text)
        self.assertNotIn("You replied", text)


class TokenEventTests(unittest.TestCase):
    def test_probability_and_alternatives(self):
        ev = token_event({"token": "Hi", "logprob": -0.1053605,
                          "top_logprobs": [{"token": "Hi", "logprob": -0.1053605}, {"token": "Hey", "logprob": -2.302585},
                                           {"token": " ", "logprob": -3.0}]})
        self.assertAlmostEqual(ev["p"], 0.9, places=3)
        self.assertEqual([a["text"] for a in ev["alternatives"]], ["Hey"])   # blank alternatives dropped


class FakeOllama:
    url = "http://fake"

    def available(self):
        return True

    def models(self):
        return ["tiny:1b"]

    def chat_stream(self, model, messages, options=None, top_alternatives=3):
        self.last_messages = messages
        yield {"type": "token", "text": "Hello", "p": 0.9, "alternatives": []}
        yield {"type": "token", "text": " there", "p": 0.4, "alternatives": [{"text": " friend", "p": 0.3}]}
        yield {"type": "done", "tokens": 2, "seconds": 0.1}


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.app = LensApp(Path(cls.tmp.name), "http://fake")
        cls.fake = FakeOllama()
        cls.app.ollama = cls.fake
        cls.port = 7799
        cls.server = serve(cls.app, cls.port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.tmp.cleanup()

    def request(self, path, body=None, host=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **({"Host": host} if host else {})})
        return urllib.request.urlopen(req, timeout=10)

    def test_foreign_host_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.request("/api/status", host="evil.example")
        self.assertEqual(ctx.exception.code, 403)

    def test_chat_streams_tokens_and_retains_memory(self):
        body = {"model": "tiny:1b", "messages": [{"role": "user", "content": "remember purple elephants dancing"}]}
        events = [json.loads(l[6:]) for l in self.request("/api/chat", body).read().decode().split("\n\n") if l.startswith("data: ")]
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "memory")
        self.assertEqual(types.count("token"), 2)
        self.assertIn("retained", types)
        events = [json.loads(l[6:]) for l in self.request("/api/chat", {
            "model": "tiny:1b", "messages": [{"role": "user", "content": "what about the purple elephants?"}]}
        ).read().decode().split("\n\n") if l.startswith("data: ")]
        self.assertEqual(len(events[0]["items"]), 1)                       # recalled
        self.assertIn("purple elephants", self.fake.last_messages[0]["content"])  # injected as context

    def test_openai_compatible_endpoint(self):
        data = json.load(self.request("/v1/chat/completions",
                                      {"model": "tiny:1b", "messages": [{"role": "user", "content": "hi"}]}))
        self.assertEqual(data["choices"][0]["message"]["content"], "Hello there")
        models = json.load(self.request("/v1/models"))
        self.assertEqual(models["data"][0]["id"], "tiny:1b")

    def test_app_page_served(self):
        page = self.request("/").read().decode()
        self.assertIn("Inside the model", page)


if __name__ == "__main__":
    unittest.main()


class LayerFinderTests(unittest.TestCase):
    def test_finds_layers_across_architecture_families(self):
        from types import SimpleNamespace as NS
        from lens.hf import find_layers
        blocks = [object(), object()]
        cases = [NS(model=NS(layers=blocks)),                       # Llama / Qwen / Mistral / Gemma / Phi
                 NS(transformer=NS(h=blocks)),                      # GPT-2 / GPT-J / Falcon / BLOOM
                 NS(gpt_neox=NS(layers=blocks)),                    # Pythia / NeoX
                 NS(backbone=NS(layers=blocks)),                    # Mamba / Falcon-Mamba
                 NS(model=NS(decoder=NS(layers=blocks)))]           # OPT
        for model in cases:
            self.assertIs(find_layers(model), blocks)
        with self.assertRaises(ValueError):
            find_layers(NS(foo=1))


class TerminalTests(unittest.TestCase):
    def test_one_shot_json_and_why(self):
        import io
        from lens.terminal import Terminal
        tmp = tempfile.TemporaryDirectory()
        app = LensApp(Path(tmp.name), "http://fake")
        app.ollama = FakeOllama()
        term = Terminal(app, None, True, False, True)
        self.assertEqual(term.model, "tiny:1b")
        out = io.StringIO()
        self.assertEqual(term.ask("hello purple world", out), "Hello there")
        events = [json.loads(l) for l in out.getvalue().splitlines()]
        self.assertEqual([e["type"] for e in events], ["memory", "token", "token", "done"])
        self.assertIn("there", term.why())                  # 40% token listed with its alternative
        self.assertEqual(app.memory.count(), 1)
        tmp.cleanup()


class EngineErrorTests(unittest.TestCase):
    def test_unexpected_backend_error_is_reported_not_silent(self):
        class Broken(FakeOllama):
            def chat_stream(self, *a, **k):
                raise TypeError("boom")
                yield
        tmp = tempfile.TemporaryDirectory()
        app = LensApp(Path(tmp.name), "http://fake")
        app.ollama = Broken()
        events = list(app.chat_events("tiny:1b", [{"role": "user", "content": "hi"}], []))
        self.assertEqual(events[-1], {"type": "error", "message": "TypeError: boom"})
        tmp.cleanup()


class CatalogTests(unittest.TestCase):
    def test_ollama_names_map_to_hugging_face(self):
        from lens.catalog import hf_for
        self.assertEqual(hf_for("qwen2.5:3b-instruct-q4_K_M"), "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(hf_for("llama3.2:3b"), "unsloth/Llama-3.2-3B-Instruct")   # ungated copy: no HF account needed
        self.assertIsNone(hf_for("my-custom-model:latest"))


class MemoryDedupeTests(unittest.TestCase):
    def test_repeated_identical_messages_recalled_once(self):
        tmp = tempfile.TemporaryDirectory()
        mem = Memory(Path(tmp.name) / "m.sqlite3")
        for _ in range(3):
            mem.retain("Write a Python function that reverses a string.", "def r(s): return s[::-1]")
        self.assertEqual(len(mem.recall("python function to reverse a string")), 1)
        tmp.cleanup()


class CatalogTest(unittest.TestCase):
    def test_ollama_names(self):
        from lens.catalog import hf_for
        self.assertEqual(hf_for("qwen3:4b-q4_K_M"), "Qwen/Qwen3-4B")
        self.assertEqual(hf_for("qwen3:4b-instruct-2507-q4_K_M"), "Qwen/Qwen3-4B-Instruct-2507")
        self.assertEqual(hf_for("llama3.2:latest"), "unsloth/Llama-3.2-3B-Instruct")
        self.assertEqual(hf_for("mistral"), "mistralai/Mistral-7B-Instruct-v0.3")
        self.assertIsNone(hf_for("gemma3:latest"))
        self.assertIsNone(hf_for("no-such-model:3b"))


class MCPTest(unittest.TestCase):
    """`lens mcp`: JSON-RPC handshake, tool listing and tool calls, without Ollama."""

    def make(self):
        from lens.mcp import Server

        class FakeMemory:
            def recall(self, query, limit=4):
                return [{"user": "hi", "assistant": "hello", "model": "m"}] if query else []

            def retain(self, *a):
                pass

        class FakeApp:
            memory = FakeMemory()

            def status(self):
                return {"models": ["qwen3:4b"]}

            def prepare(self, messages, use_memory):
                return messages, [], messages[-1]["content"]

            def chat_events(self, model, send, recalled, options=None):
                yield {"type": "token", "text": "Paris", "p": 0.98, "alternatives": []}
                yield {"type": "token", "text": " maybe", "p": 0.4, "alternatives": [{"text": " surely", "p": 0.3}]}
                yield {"type": "done", "tokens": 2}

        return Server(FakeApp())

    def rpc(self, server, lines):
        import io
        out = io.StringIO()
        server.run(io.StringIO("\n".join(json.dumps(l) if isinstance(l, dict) else l for l in lines) + "\n"), out)
        return [json.loads(x) for x in out.getvalue().splitlines()]

    def test_handshake_tools_and_errors(self):
        replies = self.rpc(self.make(), [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "ask_local_model", "arguments": {"prompt": "Capital of France?"}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "search_memory", "arguments": {"query": "hi"}}},
            {"jsonrpc": "2.0", "id": 5, "method": "nope"},
            "not json",
        ])
        by_id = {r.get("id"): r for r in replies}
        self.assertEqual(len(replies), 6)                     # the notification gets no reply
        self.assertEqual(by_id[1]["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual({t["name"] for t in by_id[2]["result"]["tools"]},
                         {"ask_local_model", "list_local_models", "search_memory"})
        result = by_id[3]["result"]["structuredContent"]
        self.assertEqual(result["answer"], "Paris maybe")
        self.assertEqual(result["uncertain_words"], 1)
        self.assertEqual(result["hesitations"][0]["word"], "maybe")
        self.assertIn("hello", by_id[4]["result"]["content"][0]["text"])
        self.assertEqual(by_id[5]["error"]["code"], -32601)
        self.assertEqual(by_id[None]["error"]["code"], -32700)


class InnerStateMemoryTest(unittest.TestCase):
    def test_similar_states_same_space_only(self):
        from lens.memory import Memory
        m = Memory(Path(tempfile.mkdtemp()) / "m.sqlite3")
        a = m.retain("write a formal letter", "Dear Sir")
        b = m.retain("write a poem", "Roses")
        c = m.retain("other model", "x")
        m.retain_state(a, "M|dev|1", [1.0, 0.0, 0.1])
        m.retain_state(b, "M|dev|1", [0.0, 1.0, 0.0])
        m.retain_state(c, "OTHER|dev|1", [1.0, 0.0, 0.1])
        got = m.similar_states("M|dev|1", [0.9, 0.1, 0.1])
        self.assertEqual([r["id"] for r in got], [a, b])          # nearest first, other spaces never compared
        m.forget(a)
        self.assertEqual([r["id"] for r in m.similar_states("M|dev|1", [0.9, 0.1, 0.1])], [b])

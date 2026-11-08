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

            def double_check(self, model, messages, short_answer):
                return {"full_answer": "The capital of France is Paris.", "agrees": True}

            def check_claims(self, claims):
                return {"claims": [{"claim": c, "risk": 4.0 if "Sydney" in c else 0.2,
                                    "band": "high" if "Sydney" in c else "low"} for c in claims],
                        "scope": "Factual relations.", "operating_points": {"high": 0.57, "elevated": 0.69}}

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
                         {"ask_local_model", "list_local_models", "search_memory", "check_claims"})
        result = by_id[3]["result"]["structuredContent"]
        self.assertEqual(result["answer"], "Paris maybe")
        self.assertEqual(result["uncertain_words"], 1)
        self.assertEqual(result["hesitations"][0]["word"], "maybe")
        self.assertEqual(result["double_check"]["low_certainty_word"]["word"], "maybe")   # short + low certainty
        self.assertTrue(result["double_check"]["agrees"])
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


class ClaimToolTest(unittest.TestCase):
    def test_check_claims_tool_and_splitting(self):
        from lens.server import split_claims
        self.assertEqual(split_claims("The capital of Australia is Sydney. Paris is in France! Ok?"),
                         ["The capital of Australia is Sydney.", "Paris is in France!"])   # fragments dropped
        server = MCPTest().make()
        out = server.call("check_claims", {"text": "The capital of Australia is Sydney. Paris is in France."})
        text = out["content"][0]["text"]
        self.assertIn("[HIGH    ] The capital of Australia is Sydney.", text)
        self.assertIn("[LOW     ] Paris is in France.", text)
        self.assertIn("not confirmation", text)


class DoubleCheckTest(unittest.TestCase):
    def test_trigger_and_agreement(self):
        from lens.server import answers_agree, needs_double_check
        short_low = [{"text": "Lagos", "p": 0.45}, {"text": ".", "p": 0.99}]
        self.assertEqual(needs_double_check(short_low)["word"], "Lagos")
        self.assertIsNone(needs_double_check([{"text": "Abuja", "p": 0.95}]))
        long_low = [{"text": "word ", "p": 0.4}] * 40
        self.assertIsNone(needs_double_check(long_low))                       # only short answers (as measured)
        self.assertTrue(answers_agree("Abuja.", "The capital of Nigeria is Abuja."))
        self.assertFalse(answers_agree("Lagos.", "The capital of Nigeria is Abuja."))


class SteeringTest(unittest.TestCase):
    """The shipped steering hook adds amount * |h| * v on known tensors (no model download)."""

    def test_constant_adds_amount_times_norm(self):
        import torch
        from lens.hf import DeepModel
        m = DeepModel.__new__(DeepModel)
        m.torch = torch
        m.steering = {0: 0.5 * torch.tensor([1.0, 0.0, 0.0])}
        h = torch.tensor([[[0.0, 3.0, 4.0]]])                      # |h| = 5
        out = m._steer_hook(0)(None, None, (h,))[0]
        self.assertTrue(torch.allclose(out, torch.tensor([[[2.5, 3.0, 4.0]]])))


class SteerOnlySpaceTest(unittest.TestCase):
    """Language steering lives in its own scrambled space: the client uses the "s:<layer>" matrix and says so."""

    def test_directive_uses_its_own_matrix(self):
        import torch
        from lens.hf import DeepModel
        m = DeepModel.__new__(DeepModel)
        m.torch, m.device = torch, "cpu"
        m.projections = {"3": torch.eye(4)[:2], "s:3": torch.eye(4)[2:]}       # different spaces, same layer
        m.set_steering([{"layer": 3, "coeffs": [1.0, 0.0], "amount": 0.5, "key": "s:3"}])
        self.assertTrue(torch.allclose(m.steering[3], torch.tensor([0.0, 0.0, 0.5, 0.0])))
        m.set_steering([{"layer": 3, "coeffs": [1.0, 0.0], "amount": 0.5}])                 # behaviour space
        self.assertTrue(torch.allclose(m.steering[3], torch.tensor([0.5, 0.0, 0.0, 0.0])))

    def test_client_names_the_space_for_steer_only_behaviours(self):
        from lens.cloud import Cloud
        c = Cloud.__new__(Cloud)
        c.config = {"pack_id": "p", "key_version": 1, "steerable_v2": [{"name": "language_french"}]}
        sent = []
        c._call = lambda method, path, body=None: sent.append(body) or {}
        c.steer("language_french", "more", 0.1)
        c.steer("confidence", "less", 0.1)
        self.assertEqual(sent[0].get("space"), "s")
        self.assertNotIn("space", sent[1])


class StreamDecodeTest(unittest.TestCase):
    """Streamed text keeps word spaces on SentencePiece tokenizers (single-token decode drops them)."""

    def make(self, tok):
        from lens.hf import DeepModel
        m = DeepModel.__new__(DeepModel)
        m.tokenizer = tok
        m._stream_ids, m._stream_text = [], ""
        return m

    def test_spaces_survive_streaming(self):
        words = ["Rivers", "are", "large", "bodies"]

        class SentencePieceLike:                 # decode([i]) drops the word's leading space, like Mistral/Llama 2
            def decode(self, ids):
                return " ".join(words[i] for i in ids)
        m = self.make(SentencePieceLike())
        self.assertEqual("".join(m._piece(i) for i in range(4)), "Rivers are large bodies")

    def test_real_mistral_tokenizer_if_present(self):
        import os
        from pathlib import Path
        path = Path(os.environ.get("LENS_TEST_MISTRAL", "")) if os.environ.get("LENS_TEST_MISTRAL") else None
        if not path or not path.exists():
            self.skipTest("set LENS_TEST_MISTRAL to a local Mistral model folder to run this check")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(path))
        text = "Je suis heureux de vous parler des fleuves. Rivers are large, flowing bodies of water."
        ids = tok.encode(text, add_special_tokens=False)
        m = self.make(tok)
        self.assertEqual("".join(m._piece(i) for i in ids).strip(), text)


class StopTokenTest(unittest.TestCase):
    """Deep mode stops at the model's own end-of-turn token (Phi-3 <|end|>, Gemma <end_of_turn>), not just a fixed list."""

    def test_declared_and_common_end_tokens(self):
        from lens.hf import DeepModel
        m = DeepModel.__new__(DeepModel)

        class Cfg:
            eos_token_id = [7, 9]

        class Model:
            generation_config = Cfg()

        class Tok:
            unk_token_id = 0
            def convert_tokens_to_ids(self, t):
                return {"<|end|>": 32007, "<end_of_turn>": 107}.get(t, 0)
        m.model, m.tokenizer = Model(), Tok()
        self.assertEqual(set(m._stop_ids()), {7, 9, 32007, 107})


class SpelledEndMarkerTest(unittest.TestCase):
    """An end marker spelled out as plain text (Yi-1.5) ends the answer and never reaches the screen."""

    def test_marker_split_across_tokens_is_cut(self):
        from lens.hf import DeepModel
        pieces = ["Hello", " there.", "\n", "<|", "im", "_end", "|>", "junk"]

        class Tok:
            def decode(self, ids):
                return "".join(pieces[i] for i in ids)
        m = DeepModel.__new__(DeepModel)
        m.tokenizer = Tok()
        m._stream_ids, m._stream_text, m._text_stop = [], "", False
        shown = ""
        for i in range(len(pieces)):
            shown += m._piece(i)
            if m._text_stop:
                break
        self.assertTrue(m._text_stop)
        self.assertEqual(shown, "Hello there.\n")

    def test_lone_angle_bracket_is_released(self):
        from lens.hf import DeepModel
        pieces = ["a ", "<", " b"]

        class Tok:
            def decode(self, ids):
                return "".join(pieces[i] for i in ids)
        m = DeepModel.__new__(DeepModel)
        m.tokenizer = Tok()
        m._stream_ids, m._stream_text, m._text_stop = [], "", False
        self.assertEqual("".join(m._piece(i) for i in range(3)), "a < b")


class TokenizerRoundTripTest(unittest.TestCase):
    """A tokenizer that drops word spaces on decode is replaced by the fast one built from tokenizer.json."""

    def test_falls_back_to_fast_tokenizer_when_spaces_are_lost(self):
        import json, tempfile
        from pathlib import Path
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        d = Path(tempfile.mkdtemp())
        tk = Tokenizer(models.BPE(unk_token="[UNK]"))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        tk.train_from_iterator(["Hello! I can help you with tasks."] * 50, trainers.BpeTrainer(vocab_size=400, special_tokens=["[UNK]"]))
        tk.save(str(d / "tokenizer.json"))
        import lens.hf as hf

        class Lossy:                                   # stands in for the slow tokenizer that loses spaces
            chat_template = "T"; eos_token = "</s>"; bos_token = None; pad_token = None; unk_token = None
            def encode(self, s, add_special_tokens=False): return [1, 2]
            def decode(self, ids): return "Hello!Icanhelpyouwithtasks."
        import transformers
        real = transformers.AutoTokenizer.from_pretrained
        transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: Lossy())
        try:
            tok = hf.load_tokenizer(str(d))
        finally:
            transformers.AutoTokenizer.from_pretrained = real
        self.assertEqual(type(tok).__name__ != "Lossy", True)
        self.assertEqual(tok.decode(tok.encode("Hello! I can help you with tasks.", add_special_tokens=False)).strip(),
                         "Hello! I can help you with tasks.")
        self.assertEqual(tok.chat_template, "T")

"""Deep mode: run a Hugging Face model locally and read its internal state.

Needs `pip install proprioceptive-lens[deep]` (torch + transformers).
Per generated token this yields the same events as the Ollama backend plus
  layers: activity (RMS of the hidden state) for every layer, for the deep view
  sigma/norm: for opted-in cloud scoring, M_L @ h and |h| at the layers the
              cloud requested, where M_L is the device's scrambled matrix.
This module contains no proprietary components: it multiplies hidden states by
whatever matrix the cloud sent and nothing else.
"""
from __future__ import annotations

import os
import threading
from typing import Iterator


# Where decoder blocks live, by architecture family. Covers Llama, Qwen 1-3, Mistral, Mixtral, Gemma 1-3, Phi 1-4,
# Granite, OLMo, DeepSeek, InternLM, StableLM, Falcon, GPT-2/J/NeoX, BLOOM, MPT, OPT, Mamba/Falcon-Mamba, RWKV and more.
LAYER_PATHS = ("model.layers", "model.language_model.layers", "language_model.model.layers", "transformer.h",
               "gpt_neox.layers", "model.decoder.layers", "transformer.blocks", "backbone.layers", "rwkv.blocks",
               "transformer.layers", "model.transformer.h")


def find_layers(model):
    """Return the model's list of decoder blocks, or raise if the architecture is unknown."""
    for path in LAYER_PATHS:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
            return obj
    raise ValueError(f"deep mode does not recognise the layer layout of {type(model).__name__}")


def load_tokenizer(source: str):
    """AutoTokenizer, checked by a decode round-trip. Some repos (DeepSeek Coder 1.3B) declare a byte-level fast tokenizer
    in tokenizer.json but transformers 5.x loads the slow LlamaTokenizer for them, which drops every word space
    ("Hello!Icanhelpyou"). When the round-trip loses spaces and a tokenizer.json exists, the fast tokenizer is built from
    it directly and the chat template and special tokens are carried over."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source)
    probe = "Hello! I can help you with tasks."
    try:
        ok = tok.decode(tok.encode(probe, add_special_tokens=False)).strip() == probe
    except Exception:
        ok = True
    if ok:
        return tok
    tokenizer_json = None
    for base in ([source] if os.path.isdir(source) else []):
        cand = os.path.join(base, "tokenizer.json")
        if os.path.exists(cand):
            tokenizer_json = cand
    if tokenizer_json is None:
        try:
            from huggingface_hub import hf_hub_download
            tokenizer_json = hf_hub_download(source, "tokenizer.json")
        except Exception:
            return tok
    from transformers import PreTrainedTokenizerFast
    fast = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
    for attr in ("chat_template", "eos_token", "bos_token", "pad_token", "unk_token"):
        value = getattr(tok, attr, None)
        if value is not None:
            try:
                setattr(fast, attr, value)
            except Exception:
                pass
    if fast.decode(fast.encode(probe, add_special_tokens=False)).strip() == probe:
        return fast
    return tok


class DeepModel:
    def __init__(self, model_id: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        # "Org/Name@/local/path" loads local weights while reporting the public model ID.
        model_id, _, local_path = model_id.partition("@")
        source = local_path or model_id
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # bf16 everywhere by default: on CPU it measured 2.5x faster than fp32 with half the memory and identical
        # greedy text (Qwen2.5-1.5B), and it matches the GPU precision the probes were validated with.
        dtype = torch.float32 if os.environ.get("LENS_DTYPE", "").lower() in ("float32", "fp32") else torch.bfloat16
        self.tokenizer = load_tokenizer(source)
        try:
            model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype)
        except TypeError:   # transformers < 4.56 uses torch_dtype
            model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype)
        self.model = model.to(self.device).eval()
        self.layers = find_layers(self.model)
        self.n_layers = len(self.layers)
        self.lock = threading.Lock()
        self.projections: dict[str, "torch.Tensor"] = {}   # "<layer>" or "p:<layer>" -> (16, d) scrambled matrix
        self.steering: dict[int, "torch.Tensor"] = {}      # layer -> nudge (d), relative to each token's norm

    def set_projections(self, matrices: dict) -> None:
        """Keys: a layer number ("15" or 15) for Behaviour, or "p:15" for the Deluxe premium space at layer 15."""
        torch = self.torch
        self.projections = {str(key): torch.tensor(m, dtype=torch.float32, device=self.device)
                            for key, m in matrices.items()}

    @staticmethod
    def _layer_of(key: str) -> int:
        return int(str(key).split(":")[-1])

    def prompt_ids(self, messages: list[dict]):
        """Token ids for a chat, exactly as Lens sends it. Templates that reject a system role (e.g. Gemma 2) get the
        system text folded into the first user message instead."""
        def encode(msgs):
            out = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                                     return_dict=True)
            return out["input_ids"].to(self.device)      # dict form works on transformers 4.4x and 5.x
        try:
            return encode(messages)
        except Exception:
            if not messages or messages[0].get("role") != "system":
                raise
            system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
            for m in rest:
                if m.get("role") == "user":
                    m["content"] = system + "\n\n" + m["content"]
                    break
            return encode(rest)

    def read(self, text: str, max_tokens: int = 128, projections: dict | None = None) -> dict:
        """Read a piece of text (no generation, no chat template) and return this device's projections of the
        mean hidden state over its tokens: {"sigma": {key: [16]}, "norm": {key: float}}. Used by claim checks,
        which were validated on mean-pooled reading of plain statements."""
        torch = self.torch
        ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"].to(self.device)
        captured = {}
        with self.lock:
            active = dict(self.projections) if projections is None else {
                str(k): torch.as_tensor(v, dtype=torch.float32, device=self.device)
                for k, v in projections.items()}
            layers = sorted({self._layer_of(k) for k in active})
            handles = []
            try:
                for L in layers:
                    handles.append(self.layers[L].register_forward_hook(
                        (lambda L: lambda _m, _i, out: captured.__setitem__(L, (out[0] if isinstance(out, tuple) else out)[0].float()))(L)))
                with torch.no_grad():
                    self.model(input_ids=ids)
            finally:
                for h in handles:
                    h.remove()
        means = {L: captured[L].mean(0) for L in captured}
        return {"sigma": {k: (M @ means[self._layer_of(k)]).tolist() for k, M in active.items()},
                "norm": {k: float(means[self._layer_of(k)].norm()) for k in active}}

    def set_steering(self, directives: list[dict]) -> None:
        """Steering directives from the cloud: [{"layer": L, "coeffs": [16 floats], "amount": a}, ...].

        The nudge is v = M_L^T c (M_L is this device's scrambled matrix); each token's hidden state at layer L
        gets amount * |h| * v added. amount 0 or an empty list switches steering off.
        """
        torch = self.torch
        steering: dict[int, "torch.Tensor"] = {}
        for item in directives:
            layer, amount = int(item["layer"]), float(item.get("amount", 0.0))
            key = str(item.get("key", layer))                  # "s:<layer>" for steer-only spaces
            if not amount or key not in self.projections:
                continue
            coeffs = torch.tensor(item["coeffs"], dtype=torch.float32, device=self.device)
            vector = self.projections[key].T @ coeffs
            steering[layer] = steering.get(layer, 0) + amount * vector / (vector.norm() + 1e-8)
        self.steering = steering

    def steer_raw(self, vectors: dict[int, "torch.Tensor"]) -> None:
        """Set nudges directly (layer -> amount * unit vector). Used by tests and controls."""
        self.steering = dict(vectors)

    def _steer_hook(self, layer: int, vector=None):
        vector = self.steering[layer] if vector is None else vector

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            nudge = vector.to(hidden.dtype) * hidden.float().norm(dim=-1, keepdim=True).to(hidden.dtype)
            hidden = hidden + nudge
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return hook

    def chat_stream(self, messages: list[dict], max_new_tokens: int = 512, temperature: float = 0.6,
                    top_alternatives: int = 3, steering: dict | None = None) -> Iterator[dict]:
        """`steering` overrides self.steering for this call only (the app's before/after path passes {}), so one stream
        never mutates state another stream may be reading."""
        torch = self.torch
        captured: dict[int, "torch.Tensor"] = {}

        def make_hook(index):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[index] = hidden[0, -1, :].detach().float()   # last token, like the validated board
            return hook

        with self.lock:                                     # everything per-stream happens under ownership of the model
            active = dict(self.steering) if steering is None else dict(steering)
            # Running mean of each projected layer's hidden state over the answer so far. The probes were
            # validated on mean-pooled states, so Behaviour scores the mean (generic arithmetic, no IP).
            self._sums = {}
            self._count = 0
            self._stream_ids, self._stream_text = [], ""   # incremental decode: keeps word spaces on SentencePiece models
            self._text_stop = False                          # an end marker appeared as plain text (Yi-1.5 spells <|im_end|>)
            handles = []
            try:
                for L, vector in active.items():
                    if 0 <= L < self.n_layers:
                        handles.append(self.layers[L].register_forward_hook(self._steer_hook(L, vector)))
                for i, layer in enumerate(self.layers):
                    handles.append(layer.register_forward_hook(make_hook(i)))
                input_ids = self.prompt_ids(messages)
                past = None
                generated = 0
                with torch.no_grad():
                    while generated < max_new_tokens:
                        out = self._forward(input_ids, past)
                        past = self._cache(out, past, input_ids)
                        logits = out.logits[0, -1].float()
                        probs = torch.softmax(logits, dim=-1)
                        if temperature > 0:
                            choice = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
                        else:
                            choice = int(torch.argmax(probs))
                        if generated == 0:
                            captured.clear()      # the prompt pass is not a generated token
                        else:
                            yield self._step_event(previous_token, previous_probs, captured, top_alternatives)
                            if self._text_stop:
                                break
                        if choice in (self.tokenizer.eos_token_id, *self._stop_ids()):
                            break
                        previous_token, previous_probs = choice, probs
                        input_ids = torch.tensor([[choice]], device=self.device)
                        generated += 1
                    else:
                        out = self._forward(input_ids, past)
                        yield self._step_event(previous_token, previous_probs, captured, top_alternatives)
            finally:
                for handle in handles:
                    handle.remove()
        yield {"type": "done", "tokens": generated, "seconds": None}

    def _forward(self, input_ids, past):
        """One step. Transformers use past_key_values; state-space models (Mamba) use cache_params."""
        torch = self.torch
        if past is None:
            return self.model(input_ids=input_ids, use_cache=True)
        kind, cache, seen = past
        if kind == "kv":
            return self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        if kind == "ssm":
            position = torch.tensor([seen], device=self.device)
            return self.model(input_ids=input_ids, cache_params=cache, use_cache=True, cache_position=position)
        return self.model(input_ids=torch.cat([cache, input_ids], dim=1))    # no cache: rerun the whole text

    @staticmethod
    def _cache(out, past, input_ids):
        seen = (past[2] if past else 0) + input_ids.shape[1]
        if getattr(out, "past_key_values", None) is not None:
            return ("kv", out.past_key_values, seen)
        if getattr(out, "cache_params", None) is not None:
            return ("ssm", out.cache_params, seen)
        prior = past[1] if past and past[0] == "full" else None
        import torch
        return ("full", input_ids if prior is None else torch.cat([prior, input_ids], dim=1), seen)

    END_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<|end_of_text|>",
                  "<|im_sep|>", "</s>", "<|return|>", "<|end_of_turn|>", "<|eom_id|>",
                  # a new turn starting means the answer is over (seen with sampling on Yi-1.5)
                  "<|im_start|>", "<start_of_turn>", "<|start_header_id|>", "<|user|>")

    def _stop_ids(self) -> tuple:
        """End-of-turn tokens: the model's own declared ones (generation_config) plus the common chat end markers.
        Phi-3 ends with <|end|> and Gemma with <end_of_turn>; missing one makes deep mode run past the answer."""
        cached = getattr(self, "_stop_cache", None)
        if cached is not None:
            return cached
        ids = set()
        declared = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        for tid in (declared if isinstance(declared, (list, tuple)) else [declared]):
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)
        for token in self.END_TOKENS:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(token)
            except Exception:
                continue
            if isinstance(tid, int) and tid >= 0 and tid != getattr(self.tokenizer, "unk_token_id", None):
                ids.add(tid)
        self._stop_cache = tuple(sorted(ids))
        return self._stop_cache

    def _piece(self, token_id: int) -> str:
        """Text this token adds to the answer. Decoding tokens one at a time drops the leading space that SentencePiece
        tokenizers (Mistral, Llama 2, Phi-3, TinyLlama, Yi, Zephyr, ...) attach to a word, so the answer is decoded as a
        whole and the new tail is returned. An incomplete UTF-8 sequence waits for the next token. Text that could be the
        start of an end marker is held back; a spelled-out end marker (Yi-1.5 writes <|im_end|> as plain text) ends the
        answer without being shown."""
        ids = getattr(self, "_stream_ids", None)
        if ids is None:                                   # outside chat_stream: single-token decode
            return self.tokenizer.decode([token_id])
        ids.append(token_id)
        full = self.tokenizer.decode(ids)
        if full.endswith("\ufffd"):
            return ""
        shown = len(self._stream_text)                    # characters already returned to the caller
        if not full.startswith(self._stream_text):        # tokenizer re-normalised earlier text: fall back
            piece = self.tokenizer.decode([token_id])
            self._stream_text += piece
            return piece
        cut = min((full.find(m) for m in self.END_TOKENS if m in full), default=-1)
        if cut >= 0:                                      # end marker complete: stop, never show it
            self._text_stop = True
            out = full[shown:cut] if cut > shown else ""
            self._stream_text = full[:max(cut, shown)]
            return out
        hold = 0                                          # longest tail that could still become an end marker
        for m in self.END_TOKENS:
            for k in range(min(len(m) - 1, len(full)), 0, -1):
                if full.endswith(m[:k]):
                    hold = max(hold, k)
                    break
        upto = max(len(full) - hold, shown)
        out = full[shown:upto]
        self._stream_text = full[:upto]
        return out

    def _step_event(self, token_id: int, probs, captured: dict, top_k: int) -> dict:
        """Event for one generated token; hidden states are those produced while reading that token."""
        torch = self.torch
        top = torch.topk(probs, top_k + 1)
        alternatives = [{"text": self.tokenizer.decode([int(i)]), "p": float(p)}
                        for p, i in zip(top.values, top.indices) if int(i) != token_id][:top_k]
        alternatives = [a for a in alternatives if a["text"].strip()]
        event = {"type": "token", "text": self._piece(token_id), "p": float(probs[token_id]),
                 "alternatives": alternatives,
                 "layers": [round(float(captured[i].pow(2).mean().sqrt()), 4) for i in range(self.n_layers)
                            if i in captured]}
        if self.projections:
            self._count += 1
            for L in {self._layer_of(k) for k in self.projections}:
                if L in captured:
                    self._sums[L] = self._sums.get(L, 0) + captured[L]
            means = {L: self._sums[L] / self._count for L in self._sums}
            event["sigma"] = {k: (M @ means[self._layer_of(k)]).tolist() for k, M in self.projections.items()
                              if self._layer_of(k) in means}
            event["norm"] = {k: float(means[self._layer_of(k)].norm()) for k in self.projections
                             if self._layer_of(k) in means}
        return event

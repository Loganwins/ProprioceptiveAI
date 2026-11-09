"""Verified answers (Lens Deluxe): exact sparse control of a multiple-choice first token, verified then committed or
exactly restored. The model runs here; Proprioceptive Cloud holds the fact base, the admission rule, the controller and
the signed provenance. This module uploads text-free measurements only (hidden-state fields, the write-site state, letter
and top logits) plus an admission key and hashes of the option strings — never the question text, never the options.

Sites (which blocks are probed, read and written) come from the pack config; the numbers behind them live on the server.
"""
from __future__ import annotations

import base64
import hashlib

import numpy as np

TOP_K = 8


def option_hash(text: str) -> str:
    return hashlib.sha256(" ".join(str(text).strip().lower().split()).encode()).hexdigest()


def probe_schedule(seed: str) -> tuple[int, float]:
    h = hashlib.sha256(seed.encode()).hexdigest()
    return (-1 if int(h[:2], 16) % 2 else 1), (.1, .25, .5, .75)[int(h[2:4], 16) % 4]


def probe_direction(seed: str, width: int) -> np.ndarray:
    n = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little")
    d = np.random.default_rng(n).choice([-1., 1.], width).astype(np.float32)
    return d / np.linalg.norm(d)


class ExactControl:
    def __init__(self, deep, cloud, sites: dict, width: int):
        self.deep, self.cloud, self.width = deep, cloud, int(width)
        self.source, self.read, self.write = int(sites["source_block"]), int(sites["read_block"]), int(sites["write_block"])
        if self.deep.model.config.hidden_size != self.width:
            raise ValueError("pack controller width does not match this model")

    # --- measurement primitives (text never leaves the machine) ---------------------------------------------------
    def prefix_ids(self, prompt: str) -> list[int]:
        ids = self.deep.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                      add_generation_prompt=True, return_dict=False)
        return [int(i) for i in (ids["input_ids"] if isinstance(ids, dict) else ids)]

    def letter_ids(self) -> list[int]:
        ids = [self.deep.tokenizer.encode(c, add_special_tokens=False) for c in "ABC"]
        if any(len(i) != 1 for i in ids):
            raise ValueError("option letters are not single tokens for this model")
        return [int(i[0]) for i in ids]

    def capture(self, prefix: list[int], letters: list[int], direction: np.ndarray, dose: float) -> np.ndarray:
        """Read-site fields for the three candidates under source perturbation 0, +dose, -dose → (3, 3, width)."""
        torch, model = self.deep.torch, self.deep.model
        device, pos = self.deep.device, len(prefix) - 1
        tokens = torch.tensor([prefix + [int(c)] for _ in range(3) for c in letters], device=device)
        d = torch.as_tensor(direction, device=device); saved, handles = {}, []

        def perturb(_m, _a, output):
            h = output[0] if isinstance(output, tuple) else output; h = h.clone()
            for arm, sign in enumerate((0, 1, -1)):
                v = h[arm * 3:(arm + 1) * 3, pos]
                h[arm * 3:(arm + 1) * 3, pos] = v + sign * dose * v.norm(dim=1, keepdim=True) * d.to(h.dtype)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        def read(_m, _a, output):
            h = output[0] if isinstance(output, tuple) else output
            saved["fields"] = h[:, -1].detach().float().cpu().numpy().reshape(3, 3, self.width)
        try:
            handles.append(self.deep.layers[self.source - 1].register_forward_hook(perturb))
            handles.append(self.deep.layers[self.read - 1].register_forward_hook(read))
            with torch.inference_mode():
                model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
            return saved["fields"]
        finally:
            for h in handles:
                h.remove()

    def first_token(self, prefix: list[int], write: np.ndarray | None = None, source: tuple | None = None) -> dict:
        """One forward: optional source perturbation (direction, dose), optional raw write at the write site's final
        prompt token; returns letter/top logits, argmax, and the write-site state before the write."""
        torch, model = self.deep.torch, self.deep.model
        device, pos = self.deep.device, len(prefix) - 1
        tokens = torch.tensor([prefix], device=device); saved, handles = {}, []
        w = None if write is None else torch.as_tensor(np.asarray(write, np.float32), device=device)

        def src(_m, _a, output):
            if not source or float(source[1]) == 0.0:
                return output
            h = output[0] if isinstance(output, tuple) else output; h = h.clone(); v = h[:, pos]
            d = torch.as_tensor(source[0], device=device)
            h[:, pos] = v + float(source[1]) * v.norm(dim=1, keepdim=True) * d.to(h.dtype)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        def wr(_m, _a, output):
            h = output[0] if isinstance(output, tuple) else output; h = h.clone()
            saved["before"] = h[0, pos].detach().float().cpu().numpy()
            if w is not None:
                h[:, pos] += w.to(h.dtype)
                saved["realized"] = (h[0, pos].detach().float().cpu().numpy() - saved["before"])
            return (h,) + output[1:] if isinstance(output, tuple) else h
        try:
            handles.append(self.deep.layers[self.source - 1].register_forward_hook(src))
            handles.append(self.deep.layers[self.write - 1].register_forward_hook(wr))
            with torch.inference_mode():
                out = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
                logits = model.lm_head(out.last_hidden_state[:, -1]).float()[0]
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite logits")
            if w is not None:
                err = float(np.linalg.norm(saved["realized"] - np.asarray(write, np.float32)) / max(float(np.linalg.norm(write)), 1e-30))
                if err > .05:
                    raise ValueError("native write realization failed")
            top = torch.topk(logits, TOP_K)
            return {"logits": logits.cpu().numpy(), "top": [[int(i), float(v)] for v, i in zip(top.values.tolist(), top.indices.tolist())],
                    "argmax_id": int(torch.argmax(logits)), "native_state": saved["before"]}
        finally:
            for h in handles:
                h.remove()

    # --- the route ------------------------------------------------------------------------------------------------
    def verified_answer(self, prompt: str, options: list[str], key: dict, probe_seed: str | None = None,
                        force: bool = False, source_dose: float = 0.0) -> dict:
        """Answer a three-option question with the verified-answer route. Returns the released letter, the native letter,
        the decision (abstain/commit/restore) and the server-signed provenance. The model state is exactly restored."""
        if len(options) != 3:
            raise ValueError("three options required")
        prefix, letters = self.prefix_ids(prompt), self.letter_ids()
        prefix_sha = hashlib.sha256(np.asarray(prefix, np.int64).tobytes()).hexdigest()
        seed = probe_seed or prefix_sha
        polarity, dose = probe_schedule(seed)
        d = probe_direction(seed, self.width)
        src = (d, source_dose * polarity) if source_dose else None
        base = self.first_token(prefix, source=src)
        letter_logits = [float(base["logits"][t]) for t in letters]
        native = next((c for c, t in zip("ABC", letters) if t == base["argmax_id"]), None)
        fields = self.capture(prefix, letters, d, dose)
        opened = self.cloud.verify_open({
            "key": {"relation": key["relation"], "subject": key["subject"]}, "option_hashes": [option_hash(o) for o in options],
            "letter_ids": letters, "letter_logits": letter_logits, "top": base["top"], "argmax_id": base["argmax_id"],
            "prefix_sha256": prefix_sha, "probe_seed": seed, "source_dose": source_dose * polarity, "force": force,
            "fields": base64.b64encode(fields.astype("<f4").tobytes()).decode(),
            "native_state": base64.b64encode(np.asarray(base["native_state"], "<f4").tobytes()).decode()})
        if opened.get("decision") == "abstain":
            return {"letter": native, "native": native, "decision": "abstain", "attempts": 0, "write_norm": 0.0,
                    "provenance": opened["provenance"]}
        resp, attempts = opened, 0
        while True:
            write = np.frombuffer(base64.b64decode(resp["write"]), dtype="<f4").astype(np.float32)
            out = self.first_token(prefix, write=write if np.any(write) else None, source=src)
            if resp.get("commit"):
                realised = out["argmax_id"]
                restored = self.first_token(prefix, source=src)
                if restored["argmax_id"] != base["argmax_id"] or not np.array_equal(restored["logits"], base["logits"]):
                    raise RuntimeError("model state was not exactly restored")
                final = self.cloud.verify_commit(resp["session_id"], realised,
                                                 {"letter_logits": [float(restored["logits"][t]) for t in letters],
                                                  "argmax_id": restored["argmax_id"]})
                raw = next((c for c, t in zip("ABC", letters) if t == realised), None)
                return {"letter": final["released_letter"], "native": native, "decision": final["decision"], "attempts": attempts,
                        "raw_under_write": raw, "write_norm": float(np.linalg.norm(write)), "provenance": final["provenance"]}
            attempts += 1
            resp = self.cloud.verify_step(resp["session_id"], [float(out["logits"][t]) for t in letters], out["top"])

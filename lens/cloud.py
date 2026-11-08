"""Proprioceptive Cloud client (opt-in): behaviour scores from internal state.

What is sent: for each generated token, 16 numbers per probe layer (the hidden
state multiplied by this device's scrambled matrix) and the hidden state's
length. No text, no prompts and no raw hidden states leave the machine.
What comes back: behaviour scores. The probes themselves stay in the cloud.
"""
from __future__ import annotations

import base64
import json
import struct
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import __version__

DEFAULT_URL = "https://api.proprioceptiveai.com"


class CloudError(RuntimeError):
    pass


class Cloud:
    def __init__(self, data_dir: Path, url: str | None = None):
        self.url = (url or DEFAULT_URL).rstrip("/")
        self.creds_path = Path(data_dir) / "cloud.json"
        self.creds = json.loads(self.creds_path.read_text()) if self.creds_path.exists() else {}
        self.config: dict | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.creds.get("token")) and self.creds.get("url") == self.url

    def _call(self, method: str, path: str, body: dict | None = None, timeout: float = 15.0) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": f"proprioceptive-lens/{__version__}"}
        if self.creds.get("token"):
            headers["Authorization"] = "Bearer " + self.creds["token"]
        req = urllib.request.Request(self.url + path, method=method, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                message = json.load(exc).get("error", str(exc))
            except (ValueError, AttributeError):
                message = str(exc)
            raise CloudError(message) from None
        except (urllib.error.URLError, OSError) as exc:
            raise CloudError(f"Proprioceptive Cloud unreachable ({exc})") from None

    def enable(self) -> dict:
        import os                                  # LENS_CLIENT_TAG=internal marks test runs (excluded from user counts)
        tag = os.environ.get("LENS_CLIENT_TAG", "").strip()
        data = self._call("POST", "/v1/lens/devices", {"client": f"lens/{__version__}" + (f" {tag}" if tag else "")})
        self.creds = {"url": self.url, "device_id": data["device_id"], "token": data["token"]}
        self.creds_path.parent.mkdir(parents=True, exist_ok=True)
        self.creds_path.write_text(json.dumps(self.creds))
        self.creds_path.chmod(0o600)
        return data

    def disable(self) -> None:
        self.creds = {}
        self.config = None
        self.creds_path.unlink(missing_ok=True)

    def fetch_config(self, model_id: str) -> tuple[dict, dict]:
        """(config, matrices) for a model without touching the chat config (used by the claim reader)."""
        saved = self.config
        try:
            matrices = self.load_config(model_id)
            return self.config, matrices
        finally:
            self.config = saved

    def check_claims(self, config: dict, steps: list[dict]) -> dict:
        """Deluxe: claim-risk readings for text read by the reader model (one step per claim)."""
        return self._call("POST", "/v1/lens/claims", {"pack_id": config["pack_id"], "key_version": config["key_version"],
                                                     "steps": steps}, timeout=30)

    def load_config(self, model_id: str) -> dict:
        """Fetch this device's scrambled matrices for a model. Returns {"<layer>" or "p:<layer>": rows}."""
        data = self._call("GET", "/v1/lens/packs/" + urllib.parse.quote(model_id, safe="") + "/config")
        matrices = {}
        # Behaviour matrices are keyed by layer ("15"); Deluxe premium matrices by "p:15" (a separate space).
        # Steer-only directions (e.g. language) use their own space, keyed "s:15".
        for prefix, entries in (("", data["layers"]), ("p:", data.get("premium_layers") or []),
                                ("s:", data.get("steer_layers") or [])):
            for layer in entries:
                raw = base64.b64decode(layer["matrix_b64"])
                values = struct.unpack(f"<{layer['rows'] * layer['cols']}f", raw)
                cols = layer["cols"]
                matrices[f"{prefix}{int(layer['layer'])}"] = [list(values[r * cols:(r + 1) * cols])
                                                              for r in range(layer["rows"])]
        self.config = data
        return matrices

    def supported_models(self) -> list[str]:
        """Models with a Behaviour pack (public list, cached for 10 minutes)."""
        import time
        cached = getattr(self, "_supported", None)
        if cached and time.time() - cached[0] < 600:
            return cached[1]
        try:
            models = self._call("GET", "/v1/lens/health", timeout=5).get("packs", [])
        except CloudError:
            models = cached[1] if cached else []
        self._supported = (time.time(), models)
        return models

    def score(self, steps: list[dict]) -> dict:
        if not self.config:
            raise CloudError("no behaviour pack loaded")
        return self._call("POST", "/v1/lens/score", {"pack_id": self.config["pack_id"],
                                                    "key_version": self.config["key_version"], "steps": steps})

    # ---------------------------------------------------------------- Deluxe
    def activate(self, activation_token: str) -> dict:
        """Unlock Lens Deluxe with the activation token emailed after checkout or trial sign-up."""
        result = self._call("POST", "/v1/lens/activate", {"activation_token": activation_token.strip()})
        if self.config:
            self.config["plan"] = result.get("plan", "deluxe")
        return result

    def start_trial(self, email: str) -> dict:
        """Ask Proprioceptive AI billing to email a 7-day Deluxe trial activation token (no card needed)."""
        email = email.strip()
        if "@" not in email or len(email) > 254:
            raise CloudError("Enter a valid email address.")
        return self._call("POST", "/v1/trial-signup", {"email": email, "product_code": "lens_deluxe"})

    def steer(self, behaviour: str, direction: str, amount: float) -> dict:
        """Steering directive for this device: {"layer", "coeffs" (16 numbers), "amount"} (Deluxe)."""
        if not self.config:
            raise CloudError("no behaviour pack loaded")
        body = {"pack_id": self.config["pack_id"], "key_version": self.config["key_version"],
                "behaviour": behaviour, "direction": direction, "amount": amount}
        if any(x.get("name") == behaviour for x in self.config.get("steerable_v2") or []):
            body["space"] = "s"                     # steer-only space: this client has its "s:<layer>" matrix
        return self._call("POST", "/v1/lens/steer", body)


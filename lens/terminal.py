"""Proprioceptive Terminal Mode: Lens in the terminal, for people who live there.

Interactive:   lens term                       chat with confidence-coloured output and slash commands
One-shot:      lens term -m qwen3:4b "question"
Scripting:     echo "question" | lens term --json -m qwen3:4b      JSON-lines events on stdout
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from .server import DEEP_PREFIX, LensApp

RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
BLUE, CYAN, GREEN, YELLOW, ORANGE, RED = ("\033[38;5;39m", "\033[38;5;87m", "\033[38;5;78m", "\033[38;5;221m",
                                          "\033[38;5;209m", "\033[38;5;203m")

BANNER = r"""
   ___                       _                 _   _
  | _ \_ _ ___ _ __ _ _ ___ (_)___ __ ___ _ __| |_(_)_ _____
  |  _/ '_/ _ \ '_ \ '_/ _ \| / _ / _/ -_) '_ \  _| \ V / -_)
  |_| |_| \___/ .__/_| \___/|_\___\__\___| .__/\__|_|\_/\___|
              |_|                        |_|
"""

HELP = """Commands
  /models              list available models        /model <name>    switch model
  /why                 where the last answer hesitated, with the alternatives it weighed
  /memory              recent memories               /forget <id|all> forget memories
  /behaviour on|off    internal-state behaviour scores (deep models; sends 16–32 scrambled numbers per word)
  /colors on|off       confidence colouring          /new             start a new conversation
  /json on|off         print raw JSON events         /quit            exit"""


class Terminal:
    def __init__(self, app: LensApp, model: str | None, use_memory: bool, color: bool, json_mode: bool):
        self.app, self.use_memory, self.color, self.json_mode = app, use_memory, color, json_mode
        self.history: list[dict] = []
        self.last_tokens: list[dict] = []
        models = app.status()["models"]
        self.model = model or self._default_model(models)

    @staticmethod
    def _default_model(models: list[str]) -> str | None:
        for m in models:
            if m.startswith(DEEP_PREFIX):
                return m
        for m in models:
            low = m.lower()
            if "instruct" in low and not any(x in low for x in ("-vl", "vision", "120b", "70b", "235b")):
                return m
        return next((m for m in models if "-vl" not in m.lower()), models[0] if models else None)

    def c(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self.color else text

    def tint(self, text: str, p) -> str:
        if not self.color or p is None or p >= 0.9:
            return text
        return self.c(YELLOW if p >= 0.6 else ORANGE if p >= 0.3 else RED, text)

    # ------------------------------------------------------------------ chat
    def ask(self, text: str, out=sys.stdout) -> str:
        self._thinking_shown = False
        self.history.append({"role": "user", "content": text})
        send, recalled, user_text = self.app.prepare(self.history, self.use_memory)
        tokens, behaviour, done = [], [], None
        for event in self.app.chat_events(self.model, send, recalled):
            if self.json_mode:
                out.write(json.dumps(event) + "\n")
                out.flush()
            if event["type"] == "token" and event.get("thinking"):
                if not self.json_mode and not getattr(self, "_thinking_shown", False):
                    out.write(self.c(DIM, "(thinking…) "))
                    out.flush()
                    self._thinking_shown = True
                continue
            if event["type"] == "token":
                tokens.append(event)
                if not self.json_mode:
                    out.write(self.tint(event["text"], event["p"]))
                    out.flush()
            elif event["type"] == "behaviour":
                behaviour += event["scores"]
                soft, hard = event["soft"], event["hard"]
            elif event["type"] == "done":
                done = event
            elif event["type"] == "error":
                if not self.json_mode:
                    out.write(self.c(RED, f"\n✖ {event['message']}\n"))
                self.history.pop()
                return ""
        reply = "".join(t["text"] for t in tokens)
        self.history.append({"role": "assistant", "content": reply})
        self.last_tokens = tokens
        if self.use_memory:
            self.app.memory.retain(user_text, reply, self.model)
        from .server import needs_double_check
        flag = needs_double_check(tokens)
        if flag and not self.json_mode:
            out.write(self.c(DIM, f"\n  ↻ double-check (low certainty on \"{flag['word']}\")…"))
            out.flush()
            check = self.app.double_check(self.model, self.history[:-1], reply)
            out.write(self.c(GREEN, " consistent.") if check["agrees"] else
                      self.c(YELLOW, f"\n  ⚠ asked in full, it says: {check['full_answer']}"))
        if not self.json_mode:
            out.write("\n" + self.summary(tokens, done, recalled) + "\n")
            if behaviour:
                out.write(self.behaviour_lines(behaviour, soft, hard) + "\n")
        return reply

    def summary(self, tokens, done, recalled) -> str:
        known = [t["p"] for t in tokens if t["p"] is not None]
        parts = []
        if known:
            parts.append(f"certainty {round(100 * sum(known) / len(known))}%")
            parts.append(f"{sum(p < 0.5 for p in known)} uncertain")
        if done and done.get("tokens") and done.get("seconds"):
            parts.append(f"{done['tokens'] / done['seconds']:.0f} tok/s")
        if recalled:
            parts.append(f"{len(recalled)} memor{'y' if len(recalled) == 1 else 'ies'} used")
        parts.append(self.model or "")
        return self.c(DIM, "  ⟡ " + " · ".join(parts) + ("   /why for hesitations" if known else ""))

    def behaviour_lines(self, scores: list[dict], soft: float, hard: float) -> str:
        info = {p["name"]: p for p in (self.app.cloud.config or {}).get("probes", [])}
        peak = dict(scores[-1])     # running-mean score at the end of the answer
        lines = [self.c(CYAN, "  behaviour (relative to this model's usual range)")]
        for name, z in sorted(peak.items(), key=lambda kv: -abs(kv[1]))[:6]:
            probe = info.get(name, {})
            label = probe.get("label", name)
            if probe.get("poles"):                      # two-sided: lean toward one end
                low, high = probe["poles"]
                bar = ("◀" * min(10, int(-z * 2)) if z < 0 else "▶" * min(10, int(z * 2))) or "·"
                lean = "neutral" if abs(z) < 0.5 else (high if z > 0 else low)
                lines.append(f"  {label:<28} {self.c(CYAN, bar)} {lean} ({z:+.1f})")
                continue
            z = max(0.0, z)
            colour = RED if z >= hard else YELLOW if z >= soft else GREEN
            bar = "█" * min(20, int(z * 2)) or "·"
            lines.append(f"  {label:<28} {self.c(colour, bar)} {z:.1f}")
        return "\n".join(lines)

    def why(self) -> str:
        worst = sorted((t for t in self.last_tokens if t["p"] is not None and t["text"].strip() and t["p"] < 0.9),
                       key=lambda t: t["p"])[:8]
        if not worst:
            return "No hesitations: every word was chosen with over 90% certainty."
        rows = []
        for t in worst:
            alts = " · ".join(f"{a['text'].strip()} {round(100 * a['p'])}%" for a in t["alternatives"]) or "—"
            rows.append(f"  {self.c(YELLOW, t['text'].strip()):<30} {round(100 * t['p']):>3}%   almost: {alts}")
        return "\n".join(rows)

    # ------------------------------------------------------------------ commands
    def command(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        if cmd in ("/quit", "/exit", "/q"):
            return False
        if cmd == "/help":
            print(HELP)
        elif cmd == "/models":
            for m in self.app.status()["models"]:
                print(("● " if m == self.model else "  ") + m)
        elif cmd == "/model":
            models = self.app.status()["models"]
            if arg in models:
                self.model = arg
                print(f"Model: {arg}")
            else:
                print(f"Unknown model. /models lists {len(models)}.")
        elif cmd == "/why":
            print(self.why())
        elif cmd == "/memory":
            for m in self.app.memory.recent(10):
                print(f"  [{m['id']}] {m['user'][:90]}")
            print(self.c(DIM, f"  {self.app.memory.count()} saved in total"))
        elif cmd == "/forget":
            n = self.app.memory.forget(None if arg == "all" else int(arg)) if arg else 0
            print(f"Forgot {n}.")
        elif cmd == "/new":
            self.history = []
            print("New conversation.")
        elif cmd == "/colors":
            self.color = arg != "off"
        elif cmd == "/json":
            self.json_mode = arg == "on"
        elif cmd in ("/behaviour", "/behavior"):
            if arg == "off":
                self.app.cloud.disable()
                print("Behaviour off.")
            else:
                try:
                    self.app.cloud.enable()
                    ok = self.app.ensure_cloud()
                    print("Behaviour on." + ("" if ok else f" ({self.app.cloud_error or 'needs a deep model'})"))
                except Exception as exc:
                    print(self.c(RED, f"Could not enable behaviour: {exc}"))
        else:
            print("Unknown command. /help lists them.")
        return True

    def banner(self) -> None:
        s = self.app.status()
        print(self.c(BLUE, BANNER))
        print(self.c(BOLD, "  Welcome to Proprioceptive Terminal Mode") + self.c(DIM, f"   v{__version__}"))
        width = min(shutil.get_terminal_size((80, 20)).columns, 78)
        print(self.c(DIM, "  " + "─" * (width - 2)))
        print(f"  model     {self.c(CYAN, self.model or 'none found')}")
        print(f"  backend   {'Ollama ' + s['backend_url'] if s['backend_ok'] else self.c(RED, 'Ollama not found')}"
              + (f" · deep: {s['deep_model']}" if s["deep_model"] else ""))
        print(f"  memory    {s['memory_count']} saved{' · Hindsight mirror on' if s['hindsight'] else ''}")
        print(f"  behaviour {'on' if s['cloud']['enabled'] else 'off (/behaviour on)'}")
        print(self.c(DIM, "  Words are coloured by certainty. /why shows hesitations, /help lists commands."))
        print()

    def repl(self) -> None:
        self.banner()
        while True:
            try:
                line = input(self.c(BLUE, "you › ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            if line.startswith("/"):
                if not self.command(line):
                    return
                continue
            sys.stdout.write(self.c(CYAN, "ai  › "))
            self.ask(line)
            print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lens term", description="Proprioceptive Terminal Mode")
    parser.add_argument("prompt", nargs="*", help="one-shot prompt (otherwise interactive, or read from stdin)")
    parser.add_argument("-m", "--model")
    parser.add_argument("--json", action="store_true", help="emit JSON-lines events (for scripts and pipelines)")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--deep", default=os.environ.get("LENS_DEEP_MODEL"))
    parser.add_argument("--hindsight", default=os.environ.get("LENS_HINDSIGHT_URL"))
    parser.add_argument("--data", default=os.environ.get("LENS_DATA", str(Path.home() / ".proprioceptive-lens")))
    parser.add_argument("--cloud", default=os.environ.get("LENS_CLOUD_URL"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    ollama = args.ollama if args.ollama.startswith("http") else "http://" + args.ollama
    app = LensApp(Path(args.data), ollama, args.hindsight, deep_model=args.deep, cloud_url=args.cloud)
    color = sys.stdout.isatty() and not args.no_color and not args.json and os.environ.get("NO_COLOR") is None
    term = Terminal(app, args.model, not args.no_memory, color, args.json)
    if not term.model:
        print("No model available. Start Ollama (ollama.com) and `ollama pull qwen3:4b`, or use --deep.",
              file=sys.stderr)
        return 1
    prompt = " ".join(args.prompt) or ("" if sys.stdin.isatty() else sys.stdin.read().strip())
    if prompt:
        reply = term.ask(prompt)
        return 0 if reply or args.json else 1
    term.repl()
    return 0

"""`lens` command: start the local app and open it in the browser."""
from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser
from pathlib import Path

from . import __version__
from .server import LensApp, serve


def verify_main(argv) -> int:
    """`lens verify`: a verified answer to a three-option factual question on a supported deep model (Lens Deluxe).
    The model runs locally; only text-free measurements, the admission key and option hashes go to the cloud."""
    import json
    parser = argparse.ArgumentParser(prog="lens verify", description="Verified answer (Lens Deluxe): exact sparse control of the "
                                     "answer, verified then committed or exactly restored. Currently Qwen/Qwen2.5-0.5B-Instruct.")
    parser.add_argument("question"); parser.add_argument("--options", nargs=3, required=True, metavar=("A", "B", "C"))
    parser.add_argument("--relation", help="optional: fact relation, e.g. capital, death_year (otherwise the target is bound from evidence)")
    parser.add_argument("--subject", help="optional: the subject the fact is about, e.g. France")
    parser.add_argument("--model", default=os.environ.get("LENS_DEEP_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--data", default=os.environ.get("LENS_DATA", str(Path.home() / ".proprioceptive-lens")))
    parser.add_argument("--cloud", default=os.environ.get("LENS_CLOUD_URL"), help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true")
    a = parser.parse_args(argv)
    from .cloud import Cloud, CloudError
    from .hf import DeepModel
    from .control import ExactControl
    cloud = Cloud(Path(a.data), a.cloud)
    if not cloud.enabled:
        cloud.enable()
    try:
        cloud.config = cloud._call("GET", f"/v1/lens/packs/{a.model}/config")
    except CloudError as exc:
        print(f"Cloud: {exc}", file=sys.stderr); return 1
    va = cloud.config.get("verified_answer")
    if not va:
        print("Verified answers are not available for this model yet (Lens Deluxe, Qwen/Qwen2.5-0.5B-Instruct).", file=sys.stderr); return 1
    deep = DeepModel(a.model)
    ctl = ExactControl(deep, cloud, va["sites"], va["width"], evidence=va.get("evidence"))
    prompt = a.question.strip() + "\n" + "\n".join(f"{c}) {o}" for c, o in zip("ABC", a.options)) + "\nAnswer with just the letter."
    key = {"relation": a.relation, "subject": a.subject} if a.relation and a.subject else None
    if key is None and not va.get("evidence"):
        print("This server binds targets by key only: pass --relation and --subject.", file=sys.stderr); return 1
    try:
        out = ctl.verified_answer(prompt, list(a.options), key)
    except CloudError as exc:
        print(f"Cloud: {exc}", file=sys.stderr); return 1
    if a.json:
        print(json.dumps(out, indent=1)); return 0
    letter = out["letter"]; chosen = a.options["ABC".index(letter)] if letter in "ABC" else "(no letter)"
    verb = {"commit": "corrected and verified", "restore": "left as the model's own answer (write not verified; state restored)",
            "abstain": "left as the model's own answer (not admitted)"}[out["decision"]]
    print(f"{letter}) {chosen}  — {verb}; model's own first answer: {out['native']}; provenance signed by {out['provenance'].get('signer')}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("term", "terminal"):
        from .terminal import main as terminal_main
        return terminal_main(argv[1:])
    if argv and argv[0] == "install":
        from .desktop import install_launcher
        print(f"Added Proprioceptive Lens to your apps: {install_launcher()}")
        return 0
    if argv and argv[0] == "mcp":
        from .mcp import main as mcp_main
        return mcp_main(argv[1:])
    if argv and argv[0] == "verify":
        return verify_main(argv[1:])
    parser = argparse.ArgumentParser(prog="lens", description="Proprioceptive Lens — see inside your local model. "
                                     "Run `lens term` for Proprioceptive Terminal Mode, `lens mcp` for the MCP server, `lens install` to add it to your apps.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("LENS_PORT", 7777)))
    parser.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
                        help="Ollama URL (default http://127.0.0.1:11434)")
    parser.add_argument("--hindsight", default=os.environ.get("LENS_HINDSIGHT_URL"),
                        help="optional Hindsight server URL to mirror memories into")
    parser.add_argument("--data", default=os.environ.get("LENS_DATA", str(Path.home() / ".proprioceptive-lens")))
    parser.add_argument("--deep", default=os.environ.get("LENS_DEEP_MODEL"),
                        help="Hugging Face model to run in deep mode, e.g. Qwen/Qwen2.5-3B-Instruct "
                             "(needs: pip install proprioceptive-lens[deep])")
    parser.add_argument("--cloud", default=os.environ.get("LENS_CLOUD_URL"), help=argparse.SUPPRESS)
    parser.add_argument("--no-browser", action="store_true", help="don't open a window")
    parser.add_argument("--tab", action="store_true", help="open in a browser tab instead of an app window")
    parser.add_argument("--version", action="version", version=f"lens {__version__}")
    args = parser.parse_args(argv)
    ollama = args.ollama if args.ollama.startswith("http") else "http://" + args.ollama
    if args.deep:
        print(f"Loading deep model {args.deep} (first run downloads it)…")
    app = LensApp(Path(args.data), ollama, args.hindsight, deep_model=args.deep, cloud_url=args.cloud)
    if app.deep_error:
        print(f"  Deep mode unavailable: {app.deep_error}")
    try:
        serve(app, args.port)
    except OSError as exc:
        url = f"http://127.0.0.1:{args.port}/"
        try:                                    # already running (e.g. launcher clicked twice): just show it
            import json
            import urllib.request
            with urllib.request.urlopen(url + "api/status", timeout=3) as r:
                if json.load(r).get("app") == "proprioceptive-lens":
                    print(f"Lens is already running at {url}")
                    if not args.no_browser:
                        from .desktop import open_window
                        webbrowser.open(url) if args.tab else open_window(url, Path(args.data))
                    return 0
        except Exception:
            pass
        print(f"Could not start on port {args.port}: {exc}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{args.port}/"
    status = app.status()
    print(f"Proprioceptive Lens {__version__} running at {url}")
    print(f"  Ollama: {'connected' if status['backend_ok'] else 'NOT FOUND at ' + ollama} · "
          f"{len(status['models'])} models · memory: {status['memory_count']} saved")
    print(f"  Use it from other apps: OpenAI-compatible endpoint {url}v1 (adds memory to any chat app)")
    if not args.no_browser:
        if args.tab:
            webbrowser.open(url)
        else:
            from .desktop import open_window
            if open_window(url, Path(args.data)) == "window":
                print("  Opened in its own window (use --tab for a browser tab). Add it to your apps: lens install")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

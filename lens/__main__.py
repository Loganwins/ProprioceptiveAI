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


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("term", "terminal"):
        from .terminal import main as terminal_main
        return terminal_main(argv[1:])
    if argv and argv[0] == "mcp":
        from .mcp import main as mcp_main
        return mcp_main(argv[1:])
    parser = argparse.ArgumentParser(prog="lens", description="Proprioceptive Lens — see inside your local model. "
                                     "Run `lens term` for Proprioceptive Terminal Mode, `lens mcp` for the MCP server.")
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
    parser.add_argument("--no-browser", action="store_true")
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
        print(f"Could not start on port {args.port}: {exc}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{args.port}/"
    status = app.status()
    print(f"Proprioceptive Lens {__version__} running at {url}")
    print(f"  Ollama: {'connected' if status['backend_ok'] else 'NOT FOUND at ' + ollama} · "
          f"{len(status['models'])} models · memory: {status['memory_count']} saved")
    print(f"  Use it from other apps: OpenAI-compatible endpoint {url}v1 (adds memory to any chat app)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

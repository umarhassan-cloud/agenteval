#!/usr/bin/env python
"""Serve the AgentEval leaderboard web demo (stdlib only, no dependencies).

    python scripts/serve_ui.py              # http://127.0.0.1:8768
    python scripts/serve_ui.py --open       # also open the browser
    python scripts/serve_ui.py --rebuild    # regenerate leaderboard/index.html from
                                            # data/leaderboard_results.json first

Serves the static ``leaderboard/`` directory. ``/`` shows the dashboard and ``/data``
returns the underlying JSON (``data/leaderboard_results.json``).
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_DIR = ROOT / "leaderboard"
DATA_FILE = ROOT / "data" / "leaderboard_results.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8768


class Handler(SimpleHTTPRequestHandler):
    """Static handler for leaderboard/ with a JSON endpoint and no caching."""

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path = self.path.split("?", 1)[0]
        if path in ("/data", "/data.json", "/api/leaderboard"):
            if not DATA_FILE.exists():
                self.send_error(404, "data/leaderboard_results.json not found")
                return
            body = DATA_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        sys.stderr.write("[agenteval] %s\n" % (format % args))


def rebuild_html() -> None:
    """Re-render leaderboard/index.html from the saved JSON results (no re-run of the eval)."""
    sys.path.insert(0, str(ROOT))
    from agenteval.reporting import render_html

    board = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    template = (LEADERBOARD_DIR / "template.html").read_text(encoding="utf-8")
    (LEADERBOARD_DIR / "index.html").write_bytes(render_html(board, template).encode("utf-8"))
    print("Rebuilt leaderboard/index.html from data/leaderboard_results.json")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    p.add_argument("--rebuild", action="store_true", help="re-render index.html from saved JSON before serving")
    args = p.parse_args(argv)

    if args.rebuild:
        rebuild_html()
    if not (LEADERBOARD_DIR / "index.html").exists():
        print("leaderboard/index.html is missing. Run: python scripts/generate_leaderboard.py", file=sys.stderr)
        return 1

    handler = functools.partial(Handler, directory=str(LEADERBOARD_DIR))
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as exc:
        print(f"Cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}/"
    print(f"AgentEval leaderboard demo running at {url}  (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

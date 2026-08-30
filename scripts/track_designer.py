#!/usr/bin/env python3
"""Open a dependency-free browser UI for editing ``configs/track.json``."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from himalaya.track import TrackConfig, load_track_config, save_track_config


HTML_PATH = Path(__file__).with_suffix(".html")


def make_handler(config_path: Path) -> type[BaseHTTPRequestHandler]:
    class TrackDesignerHandler(BaseHTTPRequestHandler):
        def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = HTML_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/config":
                config = load_track_config(config_path)
                self._json({"config": config.to_dict(), "path": str(config_path)})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/config":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65_536:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("config must be a JSON object")
                config = TrackConfig.from_mapping(payload)
                save_track_config(config, config_path)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._json(
                {
                    "config": config.to_dict(),
                    "summary": config.summary(),
                    "path": str(config_path),
                }
            )

        def log_message(self, format: str, *args: object) -> None:
            print(f"track-designer: {format % args}")

    return TrackDesignerHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "track.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    save_track_config(load_track_config(config_path), config_path)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(config_path))
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Track designer: {url}")
    print(f"Saving to: {config_path}")
    print("Press Ctrl+C to stop.")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

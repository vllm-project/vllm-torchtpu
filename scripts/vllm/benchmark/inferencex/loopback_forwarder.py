#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loopback HTTP forwarder for InferenceX eval suites targeting remote URLs."""

from __future__ import annotations

import argparse
import http.client
import http.server
import math
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_TIMEOUT_SEC = 300.0
_HOP_BY_HOP_HEADERS = frozenset({"host", "connection", "transfer-encoding"})


def parse_timeout(raw: str) -> float:
    """Validate a positive finite forwarder timeout in seconds."""
    try:
        value = float(raw)
    except ValueError as err:
        raise ValueError(f"timeout must be a positive number, got {raw!r}") from err
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"timeout must be a positive finite number, got {raw!r}")
    return value


def create_server(
    upstream: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    host: str = "127.0.0.1",
    port: int = 0,
) -> http.server.ThreadingHTTPServer:
    """Create a loopback HTTP server forwarding GET/POST to ``upstream``."""
    base = upstream.rstrip("/")

    class ForwardHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_bad_gateway(self) -> None:
            msg = b"502 Bad Gateway"
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

        def _proxy(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else None
            fwd_headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in _HOP_BY_HOP_HEADERS
            }
            req = urllib.request.Request(
                base + self.path, data=body, headers=fwd_headers, method=method
            )
            headers_sent = False
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    ctype = resp.headers.get("Content-Type", "application/json")
                    clen = resp.headers.get("Content-Length")
                    if clen is not None and not ctype.startswith("text/event-stream"):
                        payload = resp.read()
                        headers_sent = True
                        self.send_response(resp.status)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                    else:
                        headers_sent = True
                        self.close_connection = True
                        self.send_response(resp.status)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        read_chunk = getattr(resp, "read1", resp.read)
                        while True:
                            chunk = read_chunk(4096)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
            except urllib.error.HTTPError as err:
                try:
                    payload = err.read()
                except (OSError, http.client.HTTPException):
                    if headers_sent:
                        self.close_connection = True
                        return
                    self._send_bad_gateway()
                    return
                headers_sent = True
                self.send_response(err.code)
                self.send_header(
                    "Content-Type",
                    err.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (OSError, http.client.HTTPException):
                if headers_sent:
                    self.close_connection = True
                    return
                self._send_bad_gateway()

        def do_GET(self) -> None:
            self._proxy("GET")

        def do_POST(self) -> None:
            self._proxy("POST")

    return http.server.ThreadingHTTPServer((host, port), ForwardHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", help="Normalized upstream API root URL.")
    parser.add_argument(
        "port_file", type=pathlib.Path, help="File to write bound port."
    )
    parser.add_argument(
        "--timeout",
        default=str(DEFAULT_TIMEOUT_SEC),
        help="Upstream request timeout in seconds (default: 300).",
    )
    args = parser.parse_args(argv)
    try:
        timeout = parse_timeout(args.timeout)
    except ValueError as err:
        print(f"loopback_forwarder.py: {err}", file=sys.stderr)
        return 2
    srv = create_server(args.upstream, timeout=timeout)
    args.port_file.write_text(str(srv.server_address[1]))
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

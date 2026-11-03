#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""CUJ-2 Golden Trace Replay Server for offline CUJ-1 joiner and UI integration.

Serves an OpenAI-compatible `/v1/chat/completions` endpoint on CPU and emits
pre-captured golden CUJ-2 `llm_request` OTel spans (linked to the caller's W3C
`traceparent` header) without requiring a TPU slice or XLA compilation.
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time
from typing import Any

DEFAULT_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "cuj2_golden_spans.jsonl"
)


def load_fixtures(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def maybe_emit_otel_span(
    attrs: dict[str, Any], headers: dict[str, str], otlp_endpoint: str | None
) -> None:
    """Emit an OTel `llm_request` SERVER span parented to `traceparent` if OTel is available."""
    if not otlp_endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.propagate import extract
        from opentelemetry.trace import SpanKind

        ctx = extract(headers)
        tracer = trace.get_tracer("vllm.cuj2.replay")
        with tracer.start_as_current_span(
            "llm_request", context=ctx, kind=SpanKind.SERVER
        ) as span:
            for k, v in attrs.items():
                if v is not None:
                    span.set_attribute(k, v)
    except ImportError:
        pass


class ReplayHandler(BaseHTTPRequestHandler):
    fixtures: list[dict[str, Any]] = []
    cursor: int = 0
    span_out_path: str | None = None
    otlp_endpoint: str | None = None

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if not self.path.startswith("/v1/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        lowered_hdrs = {k.lower(): v for k, v in self.headers.items()}
        session_id = lowered_hdrs.get("x-agent-session-id")
        parent_id = lowered_hdrs.get("x-agent-parent-session-id")
        raw_turn = lowered_hdrs.get("x-agent-turn-id")
        turn_id = int(raw_turn) if raw_turn and raw_turn.isdigit() else None

        matched: dict[str, Any] | None = None
        for item in self.fixtures:
            if (
                session_id
                and item.get("gen_ai.session.id") == session_id
                and (turn_id is None or item.get("gen_ai.session.turn_id") == turn_id)
            ):
                matched = dict(item)
                break
        if matched is None:
            matched = dict(
                self.fixtures[ReplayHandler.cursor % len(self.fixtures)]
            )
            ReplayHandler.cursor += 1
            if session_id:
                matched["gen_ai.session.id"] = session_id
            if turn_id is not None:
                matched["gen_ai.session.turn_id"] = turn_id
            if parent_id:
                matched["gen_ai.session.parent_id"] = parent_id

        if "traceparent" in lowered_hdrs:
            matched["traceparent"] = lowered_hdrs["traceparent"]

        if self.span_out_path:
            with open(self.span_out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(matched, sort_keys=True) + "\n")

        maybe_emit_otel_span(matched, lowered_hdrs, self.otlp_endpoint)

        prompt_toks = int(matched.get("gen_ai.usage.prompt_tokens", 64))
        comp_toks = int(matched.get("gen_ai.usage.completion_tokens", 24))
        cached_toks = int(matched.get("gen_ai.usage.input_tokens.cache_hit", 0))
        response = {
            "id": matched.get("gen_ai.request.id", "chatcmpl-cuj2-replay"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get(
                "model",
                matched.get("gen_ai.response.model", "Qwen/Qwen2.5-0.5B-Instruct"),
            ),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            f"[CUJ-2 Replay] session={matched.get('gen_ai.session.id')} "
                            f"turn={matched.get('gen_ai.session.turn_id')} "
                            f"cache_hit={cached_toks}/{prompt_toks}"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "total_tokens": prompt_toks + comp_toks,
                "prompt_tokens_details": {"cached_tokens": cached_toks},
            },
            "cuj2_span_attributes": matched,
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="CUJ-2 Golden Trace Replay Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--span-log", default="/tmp/cuj2_replay_spans.jsonl")
    parser.add_argument("--otlp-endpoint", default=None)
    args = parser.parse_args()

    ReplayHandler.fixtures = load_fixtures(args.fixture)
    ReplayHandler.span_out_path = args.span_log
    ReplayHandler.otlp_endpoint = args.otlp_endpoint

    server = HTTPServer((args.host, args.port), ReplayHandler)
    print(
        f"CUJ-2 Golden Trace Replay Server listening on http://{args.host}:{args.port} "
        f"({len(ReplayHandler.fixtures)} golden spans loaded)"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

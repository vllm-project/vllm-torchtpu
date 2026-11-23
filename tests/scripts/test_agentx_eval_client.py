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
"""Tests for agentx_eval_client.sh remote URLs, forwarder, and normalization."""

from __future__ import annotations

import http.client
import http.server
import importlib.util
import json
import pathlib
import socket
import stat
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_AGENTX_EVAL_CLIENT = (
    _REPO_ROOT
    / "scripts"
    / "vllm"
    / "benchmark"
    / "inferencex"
    / "agentx_eval_client.sh"
)
_FORWARDER_SCRIPT = (
    _REPO_ROOT
    / "scripts"
    / "vllm"
    / "benchmark"
    / "inferencex"
    / "loopback_forwarder.py"
)
_CHECK_REG_SCRIPT = (
    _REPO_ROOT / "scripts" / "vllm" / "benchmarking" / "check_regression.py"
)


def _load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _valid_schema_reply(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a schema-valid JSON object for any json_schema_default.yaml case."""
    rf = payload.get("response_format") or {}
    schema = (rf.get("json_schema") or {}).get("schema") or {}
    props = schema.get("properties", {})
    if "users" in props:
        obj: Any = {
            "users": [
                {
                    "name": "a",
                    "age": 1,
                    "email": "a@b.c",
                    "is_active": True,
                    "role": "user",
                }
                for _ in range(10)
            ]
        }
    elif "expressions" in props:
        obj = {
            "expressions": [
                {"ast": {"op": "+", "left": 1, "right": 2}, "description": "d"}
                for _ in range(10)
            ]
        }
    elif "house" in props:
        obj = {
            "name": "Harry Potter",
            "house": "Gryffindor",
            "blood status": "Half-blood",
            "occupation": "auror",
            "wand": {"wood": "Holly", "core": "Phoenix", "length": 11},
            "alive": "Alive",
            "patronus": "Stag",
            "bogart": "Dementor",
        }
    else:
        obj = {
            "tool_name": "query_logs",
            "arguments": {"service": "gateway", "limit": 10, "severity": "ERROR"},
        }
    return {
        "choices": [{"message": {"content": json.dumps(obj)}, "finish_reason": "stop"}]
    }


class _MockDisaggEndpoint:
    """Local HTTP endpoint serving health, metrics, models, and chat completions."""

    def __init__(self, prefix: str = "", model_id: str = "moonshotai/Kimi-K3"):
        self.prefix = prefix
        self.model_id = model_id
        self.requests: list[tuple[str, str]] = []
        endpoint = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                endpoint.requests.append(("GET", self.path))
                if self.path in (
                    f"{endpoint.prefix}/health",
                    "/health",
                    "/healthcheck",
                ):
                    self._send(200, b'{"status":"ok"}', "application/json")
                elif self.path in (f"{endpoint.prefix}/metrics", "/metrics"):
                    self._send(
                        200,
                        b'vllm:cache_config_info{block_size="1296"} 1\n'
                        b"vllm:num_requests_running 0\n"
                        b"vllm:num_requests_waiting 0\n",
                        "text/plain",
                    )
                elif self.path == f"{endpoint.prefix}/v1/models":
                    card = {
                        "data": [
                            {
                                "id": endpoint.model_id,
                                "max_model_len": 65536,
                            }
                        ]
                    }
                    self._send(
                        200, json.dumps(card).encode("utf-8"), "application/json"
                    )
                else:
                    self._send(404, b"{}", "application/json")

            def do_POST(self) -> None:
                endpoint.requests.append(("POST", self.path))
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if self.path == f"{endpoint.prefix}/v1/chat/completions":
                    if payload.get("stream"):
                        self.close_connection = True
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        chunk = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        return
                    reply = _valid_schema_reply(payload)
                    self._send(
                        200, json.dumps(reply).encode("utf-8"), "application/json"
                    )
                else:
                    self._send(404, b"{}", "application/json")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self._server.server_address[1])
        self.url = f"http://127.0.0.1:{self.port}{prefix}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def make_endpoint():
    endpoints: list[_MockDisaggEndpoint] = []

    def _make(
        prefix: str = "", model_id: str = "moonshotai/Kimi-K3"
    ) -> _MockDisaggEndpoint:
        ep = _MockDisaggEndpoint(prefix=prefix, model_id=model_id)
        endpoints.append(ep)
        return ep

    yield _make
    for ep in endpoints:
        ep.close()


@pytest.fixture
def make_forwarder():
    mod = _load_module("loopback_forwarder_test", _FORWARDER_SCRIPT)
    servers: list[http.server.ThreadingHTTPServer] = []

    def _make(upstream: str, *, timeout: float = 5.0) -> tuple[str, int]:
        srv = mod.create_server(upstream, timeout=timeout)
        servers.append(srv)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return "127.0.0.1", int(srv.server_address[1])

    yield _make
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _make_fake_inferencex(root: pathlib.Path) -> pathlib.Path:
    """Create a minimal InferenceX tree for agentx_eval_client.sh unit tests."""
    ix = root / "InferenceX"
    (ix / "utils" / "aiperf" / "src").mkdir(parents=True)
    (ix / "benchmarks").mkdir(parents=True)
    (ix / "infx" / "evals").mkdir(parents=True)
    (ix / "benchmarks" / "runtime_settings.sh").write_text("#!/bin/bash\n")
    (ix / "infx" / "evals" / "thresholds.yaml").write_text("kimik3: {}\n")
    (ix / "infx" / "evals" / "validate_scores.py").write_text(
        "import sys\nprint('PASS: stub validate_scores')\nsys.exit(0)\n"
    )
    (ix / "benchmarks" / "benchmark_lib.sh").write_text(
        """#!/bin/bash
run_eval() {
    local port=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port) port="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    # Verify that 127.0.0.1:$port answers /v1/models (tests the local forwarder
    # when --server-url points to a prefixed or remote endpoint).
    curl -sf --max-time 10 "http://127.0.0.1:${port}/v1/models" >/dev/null || return 1
    if [[ "${EVAL_FRAMEWORK:-}" == "kimi-vendor" ]]; then
        local sse
        sse=$(curl -sfN --max-time 10 -H "Content-Type: application/json" \
            -d '{"model":"m","messages":[],"stream":true}' \
            "http://127.0.0.1:${port}/v1/chat/completions") || return 1
        [[ "$sse" == *"[DONE]"* ]] || return 1
        cat > "${EVAL_SUITE}.json" << 'JSON'
{
  "schema_accuracy": 0.965,
  "finish_others": 0,
  "successful_ToolCalls": 193,
  "total_ToolCalls": 200
}
JSON
    else
        cat > "results_gsm8k.json" << 'JSON'
{
  "results": {
    "gsm8k": {
      "exact_match,flexible-extract": 0.92,
      "exact_match_stderr,flexible-extract": 0.01
    }
  }
}
JSON
    fi
}
"""
    )
    return ix


def test_agentx_eval_client_remote_url_standalone_defaults_and_schema_normalization(
    tmp_path, make_endpoint
):
    endpoint = make_endpoint(prefix="/gw", model_id="moonshotai/Kimi-K3")
    scratch = tmp_path / "scratch"
    _make_fake_inferencex(scratch)
    results_dir = tmp_path / "eval_results"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    lm_eval_stub = fake_bin / "lm_eval"
    lm_eval_stub.write_text(
        """#!/bin/bash
printf '%s\\n' "$@" >> "$STUB_LOG_DIR/lm_eval.args"
out=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_path) out="$2"; shift 2 ;;
        *) shift ;;
    esac
done
cat > "$out/results_mmlu_pro.json" << 'JSON'
{
  "results": {
    "mmlu_pro": {
      "exact_match,custom-extract": 0.84,
      "exact_match_stderr,custom-extract": 0.01
    }
  }
}
JSON
"""
    )
    lm_eval_stub.chmod(lm_eval_stub.stat().st_mode | stat.S_IXUSR)

    proc = subprocess.run(
        [
            "bash",
            str(_AGENTX_EVAL_CLIENT),
            "--suites",
            "mmlu_pro,kimi_tool_call_schema_full",
            "--server-url",
            endpoint.url,
            "--results-dir",
            str(results_dir),
        ],
        cwd=str(_REPO_ROOT),
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "STUB_LOG_DIR": str(tmp_path),
            "AGENTX_SCRATCH_DIR": str(scratch),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    # 1. mmlu_pro ran with standalone default concurrency 16 against the remote URL.
    lm_args = (tmp_path / "lm_eval.args").read_text().splitlines()
    model_args = [lm_args[i + 1] for i, a in enumerate(lm_args) if a == "--model_args"]
    assert model_args == [
        f"model=moonshotai/Kimi-K3,base_url={endpoint.url}/v1/chat/completions,"
        "num_concurrent=16,max_retries=3,timeout=1800"
    ]

    # 2. kimi_tool_call_schema_full was normalized into results_<task>.json and
    #    can be gated by check_regression.py --mode eval.
    schema_res_path = results_dir / "results_kimi_tool_call_schema_full.json"
    assert schema_res_path.exists()
    schema_res = json.loads(schema_res_path.read_text())
    assert schema_res["results"]["kimi_tool_call_schema_full"] == {
        "acc,none": 0.965,
        "acc_stderr,none": 0.0,
    }

    check_reg = _load_module("check_reg_r2", _CHECK_REG_SCRIPT)
    loaded = check_reg.load_eval_results(results_dir)
    assert loaded["kimi_tool_call_schema_full"]["acc,none"] == 0.965
    assert loaded["mmlu_pro"]["exact_match,custom-extract"] == 0.84
    gate = check_reg.check_eval(
        loaded,
        {
            "kimi_tool_call_schema_full": {"acc,none": 0.95, "acc_stderr,none": 0.0},
            "mmlu_pro": {"exact_match,custom-extract": 0.834},
        },
        tolerance=0.015,
    )
    assert gate.failed_count == 0


def test_agentx_eval_client_unknown_suite_lists_mmlu_pro(tmp_path):
    proc = subprocess.run(
        ["bash", str(_AGENTX_EVAL_CLIENT), "--suites", "bogus"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 2
    assert "mmlu_pro" in proc.stderr


@pytest.mark.parametrize(
    ("bad_url", "must_appear", "must_not_appear"),
    [
        ("ftp://bad.url", "ftp://bad.url", None),
        ("http://bad.url:notaport", "http://bad.url:notaport", None),
        ("http://user:secretpw@bad.url", "<redacted: contains '@'>", "secretpw"),
        ("http://bad.url?q=1", "http://bad.url?q=1", None),
        ("http://bad.url/v1/chat/completions", "/v1/chat/completions", None),
    ],
)
def test_agentx_eval_client_invalid_server_url_uses_shared_rules(
    bad_url, must_appear, must_not_appear
):
    proc = subprocess.run(
        [
            "bash",
            str(_AGENTX_EVAL_CLIENT),
            "--suites",
            "mmlu_pro",
            "--server-url",
            bad_url,
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 2
    assert must_appear in proc.stderr
    if must_not_appear is not None:
        assert must_not_appear not in proc.stderr


def test_agentx_eval_client_mmlu_pro_failure_writes_fail_verdict(
    tmp_path, make_endpoint
):
    endpoint = make_endpoint(model_id="moonshotai/Kimi-K3")
    scratch = tmp_path / "scratch"
    _make_fake_inferencex(scratch)
    results_dir = tmp_path / "eval_results_fail"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    lm_eval_stub = fake_bin / "lm_eval"
    lm_eval_stub.write_text("#!/bin/bash\necho 'boom' >&2\nexit 1\n")
    lm_eval_stub.chmod(lm_eval_stub.stat().st_mode | stat.S_IXUSR)

    proc = subprocess.run(
        [
            "bash",
            str(_AGENTX_EVAL_CLIENT),
            "--suites",
            "mmlu_pro",
            "--server-url",
            endpoint.url,
            "--results-dir",
            str(results_dir),
        ],
        cwd=str(_REPO_ROOT),
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "AGENTX_SCRATCH_DIR": str(scratch),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    assert not (results_dir / "results_mmlu_pro.json").exists()
    summary = (results_dir / "summary.md").read_text()
    assert "### mmlu_pro: FAIL (eval rc=1, validate rc=0)" in summary
    assert "FAIL: mmlu_pro lm_eval failed (rc=1)" in summary
    assert "PASS: mmlu_pro" not in summary


def test_forwarder_returns_502_when_upstream_is_unreachable(make_forwarder):
    closed_port = _free_port()
    host, port = make_forwarder(f"http://127.0.0.1:{closed_port}")
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        body = resp.read()
    finally:
        conn.close()
    assert resp.status == 502
    assert body == b"502 Bad Gateway"


def test_forwarder_returns_502_on_truncated_content_length(make_forwarder):
    """IncompleteRead from a truncated upstream body produces 502, not empty reply."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    up_port = listener.getsockname()[1]

    def serve_once() -> None:
        conn, _ = listener.accept()
        try:
            conn.recv(4096)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 100\r\n"
                b"\r\n"
                b"short"
            )
        finally:
            conn.close()
            listener.close()

    t = threading.Thread(target=serve_once, daemon=True)
    t.start()
    host, port = make_forwarder(f"http://127.0.0.1:{up_port}")
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        body = resp.read()
    finally:
        conn.close()
        t.join(timeout=5)
    assert resp.status == 502
    assert body == b"502 Bad Gateway"


def test_forwarder_does_not_inject_502_after_streaming_headers_sent(make_forwarder):
    """A mid-stream upstream reset closes the connection without a second HTTP reply."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    up_port = listener.getsockname()[1]

    def serve_broken_chunked() -> None:
        conn, _ = listener.accept()
        try:
            conn.recv(4096)
            # Chunked SSE response that stops before the terminating 0\r\n\r\n chunk.
            chunk = b"data: hello\n\n"
            header = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"\r\n"
                + f"{len(chunk):x}\r\n".encode("ascii")
                + chunk
                + b"\r\n"
                + b"50\r\npartial"
            )
            conn.sendall(header)
        finally:
            conn.close()
            listener.close()

    t = threading.Thread(target=serve_broken_chunked, daemon=True)
    t.start()
    host, port = make_forwarder(f"http://127.0.0.1:{up_port}")
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("POST", "/v1/chat/completions", body=b'{"stream":true}')
        resp = conn.getresponse()
        body = resp.read()
    finally:
        conn.close()
        t.join(timeout=5)
    assert resp.status == 200
    assert b"data: hello\n\n" in body
    assert b"HTTP/1.1 502" not in body
    assert b"502 Bad Gateway" not in body


def test_forwarder_streams_sse_chunks_immediately(make_forwarder):
    """read1() flushes the first SSE event before the upstream finishes the stream."""
    first_chunk_received = threading.Event()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    up_port = listener.getsockname()[1]

    def serve_sse() -> None:
        conn, _ = listener.accept()
        try:
            conn.recv(4096)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Connection: close\r\n"
                b"\r\n"
                b"data: first\n\n"
            )
            # Only send the second chunk after the client already received the first.
            # If the forwarder used resp.read(4096) instead of read1(4096), this wait
            # would time out because the first chunk is far smaller than 4096 bytes.
            if first_chunk_received.wait(timeout=3.0):
                conn.sendall(b"data: second\n\n")
        finally:
            conn.close()
            listener.close()

    t = threading.Thread(target=serve_sse, daemon=True)
    t.start()
    host, port = make_forwarder(f"http://127.0.0.1:{up_port}", timeout=5.0)
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("POST", "/v1/chat/completions", body=b'{"stream":true}')
        resp = conn.getresponse()
        assert resp.status == 200
        first = resp.read1(4096)
        assert first == b"data: first\n\n"
        first_chunk_received.set()
        rest = resp.read()
        assert rest == b"data: second\n\n"
    finally:
        conn.close()
        t.join(timeout=5)


def test_forwarder_respects_configurable_timeout(make_forwarder):
    """A hung upstream returns 502 promptly when configured with a short timeout."""
    release = threading.Event()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    up_port = listener.getsockname()[1]

    def hang() -> None:
        conn, _ = listener.accept()
        try:
            conn.recv(4096)
            release.wait(timeout=5.0)
        finally:
            conn.close()
            listener.close()

    t = threading.Thread(target=hang, daemon=True)
    t.start()
    host, port = make_forwarder(f"http://127.0.0.1:{up_port}", timeout=0.3)
    t0 = time.monotonic()
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        body = resp.read()
    finally:
        elapsed = time.monotonic() - t0
        release.set()
        conn.close()
        t.join(timeout=5)
    assert resp.status == 502
    assert body == b"502 Bad Gateway"
    assert elapsed < 3.0

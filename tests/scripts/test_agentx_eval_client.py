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

import gzip
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


class _MockDisaggEndpoint:
    """Local HTTP endpoint serving health, metrics, models, and chat completions."""

    def __init__(
        self,
        prefix: str = "",
        model_id: str = "moonshotai/Kimi-K3",
        *,
        model_ids: list[str] | None = None,
        include_max_model_len: bool = True,
        required_api_key: str | None = None,
        models_raw_body: bytes | None = None,
    ):
        self.prefix = prefix
        self.model_ids = model_ids if model_ids is not None else [model_id]
        self.include_max_model_len = include_max_model_len
        self.required_api_key = required_api_key
        self.models_raw_body = models_raw_body
        self.requests: list[tuple[str, str]] = []
        self.seen_auth_headers: list[str | None] = []
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

            def _check_auth(self) -> bool:
                endpoint.seen_auth_headers.append(self.headers.get("Authorization"))
                key = endpoint.required_api_key
                if key is None or self.headers.get("Authorization") == f"Bearer {key}":
                    return True
                self._send(401, b'{"error":"unauthorized"}', "application/json")
                return False

            def do_GET(self) -> None:
                endpoint.requests.append(("GET", self.path))
                if not self._check_auth():
                    return
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
                    if endpoint.models_raw_body is not None:
                        self._send(200, endpoint.models_raw_body, "application/json")
                        return
                    entries = []
                    for mid in endpoint.model_ids:
                        entry: dict[str, Any] = {"id": mid}
                        if endpoint.include_max_model_len:
                            entry["max_model_len"] = 65536
                        entries.append(entry)
                    card = {"data": entries}
                    self._send(
                        200, json.dumps(card).encode("utf-8"), "application/json"
                    )
                else:
                    self._send(404, b"{}", "application/json")

            def do_POST(self) -> None:
                endpoint.requests.append(("POST", self.path))
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if not self._check_auth():
                    return
                payload = json.loads(raw.decode("utf-8") or "{}")
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
                    reply = {
                        "choices": [
                            {"message": {"content": "ok"}, "finish_reason": "stop"}
                        ]
                    }
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
        prefix: str = "",
        model_id: str = "moonshotai/Kimi-K3",
        *,
        model_ids: list[str] | None = None,
        include_max_model_len: bool = True,
        required_api_key: str | None = None,
        models_raw_body: bytes | None = None,
    ) -> _MockDisaggEndpoint:
        ep = _MockDisaggEndpoint(
            prefix=prefix,
            model_id=model_id,
            model_ids=model_ids,
            include_max_model_len=include_max_model_len,
            required_api_key=required_api_key,
            models_raw_body=models_raw_body,
        )
        endpoints.append(ep)
        return ep

    yield _make
    for ep in endpoints:
        ep.close()


@pytest.fixture
def make_forwarder():
    mod = _load_module("loopback_forwarder_test", _FORWARDER_SCRIPT)
    servers: list[http.server.ThreadingHTTPServer] = []

    def _make(
        upstream: str, *, timeout: float = 5.0, api_key: str | None = None
    ) -> tuple[str, int]:
        srv = mod.create_server(upstream, timeout=timeout, api_key=api_key)
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
    curl -sf --max-time 10 -H "Authorization: Bearer EMPTY" \
        "http://127.0.0.1:${port}/v1/models" >/dev/null || return 1
    mkdir -p "${EVAL_RESULT_DIR:-.}"
    if [[ "${EVAL_FRAMEWORK:-}" == "kimi-vendor" ]]; then
        local sse
        sse=$(curl -sfN --max-time 10 -H "Content-Type: application/json" \
            -H "Authorization: Bearer EMPTY" \
            -d '{"model":"m","messages":[],"stream":true}' \
            "http://127.0.0.1:${port}/v1/chat/completions") || return 1
        [[ "$sse" == *"[DONE]"* ]] || return 1
        cat > "${EVAL_RESULT_DIR:-.}/results_kimi_vendor_20260401_120000.json" << JSON
{
  "results": {
    "${EVAL_SUITE}": {
      "exact_match,strict-match": 0.965,
      "exact_match_stderr,strict-match": 0.0
    }
  }
}
JSON
    else
        cat > "${EVAL_RESULT_DIR:-.}/results_gsm8k.json" << 'JSON'
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

    # 2. kimi_tool_call_schema_full emits exact_match,strict-match (hyphen), which
    #    normalize_suite_results augments with acc,none + acc_stderr,none so
    #    check_regression.load_eval_results() always reads it.
    schema_res_path = results_dir / "results_kimi_tool_call_schema_full.json"
    assert schema_res_path.exists()
    schema_res = json.loads(schema_res_path.read_text())
    assert schema_res["results"]["kimi_tool_call_schema_full"] == {
        "exact_match,strict-match": 0.965,
        "exact_match_stderr,strict-match": 0.0,
        "acc,none": 0.965,
        "acc_stderr,none": 0.0,
    }

    check_reg = _load_module("check_reg_r2", _CHECK_REG_SCRIPT)
    loaded = check_reg.load_eval_results(results_dir)
    assert loaded["kimi_tool_call_schema_full"] == {
        "acc,none": 0.965,
        "acc_stderr,none": 0.0,
    }
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


def test_forwarder_server_config_sets_queue_size_and_daemon_threads():
    mod = _load_module("loopback_forwarder_cfg", _FORWARDER_SCRIPT)
    srv = mod.create_server("http://127.0.0.1:1")
    try:
        assert srv.request_queue_size == 128
        assert srv.daemon_threads is True
    finally:
        srv.server_close()


def test_forwarder_forces_identity_encoding_and_strips_hop_by_hop_headers(
    make_forwarder,
):
    """Upstream receives Accept-Encoding: identity and no hop-by-hop headers."""
    seen_headers: list[dict[str, str]] = []

    class GzipCapableHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:
            seen_headers.append({k.lower(): v for k, v in self.headers.items()})
            raw = b'{"ok":true}'
            if "gzip" in self.headers.get("Accept-Encoding", "").lower():
                payload = gzip.compress(raw)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), GzipCapableHandler)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    try:
        host, port = make_forwarder(f"http://127.0.0.1:{up.server_address[1]}")
        conn = http.client.HTTPConnection(host, port, timeout=5)
        try:
            conn.request(
                "GET",
                "/v1/models",
                headers={
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive, X-Custom-Hop",
                    "Keep-Alive": "timeout=5",
                    "Proxy-Authorization": "Basic abc",
                    "TE": "trailers",
                    "Upgrade": "h2c",
                    "X-Custom-Hop": "drop-me",
                    "X-End-To-End": "keep-me",
                },
            )
            resp = conn.getresponse()
            body = resp.read()
        finally:
            conn.close()
    finally:
        up.shutdown()
        up.server_close()

    assert resp.status == 200
    assert json.loads(body) == {"ok": True}
    assert len(seen_headers) == 1
    hdrs = seen_headers[0]
    assert hdrs.get("accept-encoding") == "identity"
    assert hdrs.get("x-end-to-end") == "keep-me"
    for forbidden in (
        "keep-alive",
        "proxy-authorization",
        "te",
        "upgrade",
        "x-custom-hop",
    ):
        assert forbidden not in hdrs


def test_forwarder_rejects_chunked_post_with_411(make_forwarder, make_endpoint):
    ep = make_endpoint()
    host, port = make_forwarder(ep.url)
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"2\r\n{}\r\n0\r\n\r\n"
        )
        raw = sock.recv(4096)
    assert raw.startswith(b"HTTP/1.1 411 ")
    assert b"Connection: close" in raw
    assert ep.requests == []


@pytest.mark.parametrize("bad_clen", ["abc", "-5", "1.5"])
def test_forwarder_rejects_invalid_content_length_with_400(
    make_forwarder, make_endpoint, bad_clen
):
    ep = make_endpoint()
    host, port = make_forwarder(ep.url)
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(
            f"POST /v1/chat/completions HTTP/1.1\r\n"
            f"Host: 127.0.0.1\r\n"
            f"Content-Length: {bad_clen}\r\n"
            f"\r\n".encode("ascii")
        )
        raw = sock.recv(4096)
    assert raw.startswith(b"HTTP/1.1 400 ")
    assert ep.requests == []


@pytest.mark.parametrize("bad_path", ["//evil.example/v1/models", "http://evil/v1"])
def test_forwarder_rejects_non_origin_path_with_400(
    make_forwarder, make_endpoint, bad_path
):
    ep = make_endpoint()
    host, port = make_forwarder(ep.url)
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(
            f"GET {bad_path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode("ascii")
        )
        raw = sock.recv(4096)
    assert raw.startswith(b"HTTP/1.1 400 ")
    assert ep.requests == []


def test_forwarder_passes_through_http_error_status_and_body(make_forwarder):
    class ErrorHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:
            body = b'{"error":"rate limited"}'
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ErrorHandler)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    try:
        host, port = make_forwarder(f"http://127.0.0.1:{up.server_address[1]}")
        conn = http.client.HTTPConnection(host, port, timeout=5)
        try:
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            body = resp.read()
        finally:
            conn.close()
    finally:
        up.shutdown()
        up.server_close()

    assert resp.status == 429
    assert json.loads(body) == {"error": "rate limited"}


def test_forwarder_substitutes_openai_api_key_when_missing_or_empty(
    make_forwarder, make_endpoint
):
    ep = make_endpoint(prefix="/gw")
    host, port = make_forwarder(ep.url, api_key="real-gw-token")
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()
        conn.request("GET", "/v1/models", headers={"Authorization": "Bearer EMPTY"})
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()
        conn.request(
            "GET", "/v1/models", headers={"Authorization": "Bearer explicit-caller"}
        )
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()
    finally:
        conn.close()

    assert ep.seen_auth_headers == [
        "Bearer real-gw-token",
        "Bearer real-gw-token",
        "Bearer explicit-caller",
    ]


def test_agentx_eval_client_mmlu_pro_only_skips_forwarder_inferencex_and_max_len(
    tmp_path, make_endpoint
):
    """mmlu_pro-only hits the endpoint directly without InferenceX or max_model_len."""
    endpoint = make_endpoint(
        prefix="/gw",
        model_id="org/remote-model",
        include_max_model_len=False,
    )
    scratch = tmp_path / "scratch_unused"
    results_dir = tmp_path / "mmlu_only_results"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    # Failing git stub proves ix_checkout never tries to clone InferenceX.
    git_stub = fake_bin / "git"
    git_stub.write_text("#!/bin/bash\necho 'git must not be called' >&2\nexit 99\n")
    git_stub.chmod(git_stub.stat().st_mode | stat.S_IXUSR)

    lm_eval_stub = fake_bin / "lm_eval"
    lm_eval_stub.write_text(
        """#!/bin/bash
printf '%s\\n' "${OPENAI_API_KEY:-}" > "$STUB_LOG_DIR/lm_eval.api_key"
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
      "exact_match,custom-extract": 0.77,
      "exact_match_stderr,custom-extract": 0.02
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
            "mmlu_pro",
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
    assert not (scratch / "InferenceX").exists()
    assert (tmp_path / "lm_eval.api_key").read_text().strip() == "EMPTY"
    assert (results_dir / "results_mmlu_pro.json").exists()


def test_agentx_eval_client_preserves_openai_api_key_for_auth_gateway(
    tmp_path, make_endpoint
):
    endpoint = make_endpoint(
        prefix="/gw",
        model_id="moonshotai/Kimi-K3",
        required_api_key="gw-secret-key",
    )
    scratch = tmp_path / "scratch"
    _make_fake_inferencex(scratch)
    results_dir = tmp_path / "auth_results"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    lm_eval_stub = fake_bin / "lm_eval"
    lm_eval_stub.write_text(
        """#!/bin/bash
printf '%s\\n' "${OPENAI_API_KEY:-}" > "$STUB_LOG_DIR/lm_eval.api_key"
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
            "OPENAI_API_KEY": "gw-secret-key",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert (tmp_path / "lm_eval.api_key").read_text().strip() == "gw-secret-key"
    assert all(h == "Bearer gw-secret-key" for h in endpoint.seen_auth_headers)


def test_agentx_eval_client_gsm8k_requires_max_model_len_unless_overridden(
    tmp_path, make_endpoint
):
    endpoint = make_endpoint(
        model_ids=["model-a", "model-b"],
        include_max_model_len=False,
    )
    scratch = tmp_path / "scratch"
    _make_fake_inferencex(scratch)
    results_dir = tmp_path / "gsm8k_results"

    fail_proc = subprocess.run(
        [
            "bash",
            str(_AGENTX_EVAL_CLIENT),
            "--suites",
            "gsm8k",
            "--server-url",
            endpoint.url,
            "--results-dir",
            str(results_dir),
        ],
        cwd=str(_REPO_ROOT),
        env={"PATH": "/usr/bin:/bin", "AGENTX_SCRATCH_DIR": str(scratch)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert fail_proc.returncode == 1
    assert "gsm8k requires max_model_len" in fail_proc.stdout

    ok_proc = subprocess.run(
        [
            "bash",
            str(_AGENTX_EVAL_CLIENT),
            "--suites",
            "gsm8k",
            "--server-url",
            endpoint.url,
            "--model",
            "model-b",
            "--max-model-len",
            "32768",
            "--results-dir",
            str(results_dir),
        ],
        cwd=str(_REPO_ROOT),
        env={"PATH": "/usr/bin:/bin", "AGENTX_SCRATCH_DIR": str(scratch)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert ok_proc.returncode == 0, (
        f"STDOUT:\n{ok_proc.stdout}\nSTDERR:\n{ok_proc.stderr}"
    )
    assert "served model: model-b, context 32768" in ok_proc.stdout
    assert (results_dir / "results_gsm8k.json").exists()


def test_agentx_eval_client_fails_on_malformed_models_json(tmp_path, make_endpoint):
    endpoint = make_endpoint(models_raw_body=b"{not json")
    results_dir = tmp_path / "bad_models"

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
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 1
    assert "FATAL: cannot parse /v1/models" in proc.stdout


def test_agentx_eval_client_mmlu_pro_fails_without_lm_eval_results_json(
    tmp_path, make_endpoint
):
    """Empty lm_jsons for mmlu_pro fails instead of scanning stray JSON files."""
    endpoint = make_endpoint()
    results_dir = tmp_path / "no_lm_json"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    lm_eval_stub = fake_bin / "lm_eval"
    lm_eval_stub.write_text(
        """#!/bin/bash
out=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_path) out="$2"; shift 2 ;;
        *) shift ;;
    esac
done
# Write a non-results JSON with a score key; mmlu_pro must not fall through to it.
echo '{"score": 0.99}' > "$out/stray.json"
"""
    )
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
        env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    assert "No lm_eval results*.json found" in proc.stderr
    assert not (results_dir / "results_mmlu_pro.json").exists()


@pytest.mark.parametrize(
    ("vendor_payload", "expected_score"),
    [
        pytest.param(
            {"successful_ToolCalls": 19, "total_ToolCalls": 20},
            0.95,
            id="valid_ratio",
        ),
        pytest.param({"schema_accuracy": True}, None, id="bool_direct_score"),
        pytest.param({"schema_accuracy": 95.0}, None, id="out_of_range_gt_1"),
        pytest.param({"schema_accuracy": -0.1}, None, id="negative_score"),
        pytest.param(
            {"successful_ToolCalls": True, "total_ToolCalls": 2},
            None,
            id="bool_ok_count",
        ),
        pytest.param(
            {"successful_ToolCalls": 25, "total_ToolCalls": 20},
            None,
            id="ok_exceeds_total",
        ),
    ],
)
def test_agentx_eval_client_vendor_normalization_rejects_bool_and_out_of_range_scores(
    tmp_path, make_endpoint, vendor_payload, expected_score
):
    endpoint = make_endpoint()
    scratch = tmp_path / "scratch"
    ix = _make_fake_inferencex(scratch)
    payload_json = json.dumps(vendor_payload)
    (ix / "benchmarks" / "benchmark_lib.sh").write_text(
        "#!/bin/bash\n"
        "run_eval() {\n"
        '  mkdir -p "${EVAL_RESULT_DIR:-.}"\n'
        f"  printf '%s\\n' '{payload_json}' > \"${{EVAL_RESULT_DIR:-.}}/vendor.json\"\n"
        "}\n"
    )
    results_dir = tmp_path / "vendor_norm"
    proc = subprocess.run(
        [
            "bash",
            str(_AGENTX_EVAL_CLIENT),
            "--suites",
            "kimi_tool_call_schema",
            "--server-url",
            endpoint.url,
            "--results-dir",
            str(results_dir),
        ],
        cwd=str(_REPO_ROOT),
        env={"PATH": "/usr/bin:/bin", "AGENTX_SCRATCH_DIR": str(scratch)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    out_file = results_dir / "results_kimi_tool_call_schema.json"
    if expected_score is None:
        assert proc.returncode == 1
        assert not out_file.exists()
    else:
        assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        doc = json.loads(out_file.read_text())
        assert doc["results"]["kimi_tool_call_schema"]["acc,none"] == expected_score

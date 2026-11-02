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

"""vllm-router's prefill/decode mode speaks the toy proxy's two-hop protocol.

The image ships vllm-router (docker/Dockerfile, stage `router-build`) and
k3_agentx_client_pod.sh starts it in place of examples/disagg/toy_proxy_server.py
when a config sets AGENTX_PROXY=router. The engines see the same two hops
either way: the prefill request with max_tokens=1 and no streaming, then the
decode request carrying the kv_transfer_params the prefill answered with.
Two stub engines record what the router sends them.
"""

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

PREFILL_KV_PARAMS = {
    "uuid": 4242,
    "remote_block_ids": [3, 4, 5],
    "remote_host": "10.0.0.7",
    "remote_port": 5555,
}
DECODE_CHUNKS = [
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "4"}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
]


def _router_binary() -> str | None:
    return shutil.which("vllm-router") or (
        "/usr/local/bin/vllm-router"
        if os.access("/usr/local/bin/vllm-router", os.X_OK)
        else None
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StubEngine:
    """One engine's HTTP surface: health, a model card, and a completions route
    whose request bodies are kept for the assertions."""

    def __init__(self, role: str):
        self.role = role
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _json(self, code: int, body: dict | list):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.startswith("/v1/models"):
                    self._json(
                        200,
                        {"object": "list", "data": [{"id": "m", "object": "model"}]},
                    )
                elif self.path.startswith("/metrics"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"vllm:num_requests_running 0\n")
                else:
                    self._json(200, {"status": "ok"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                stub.requests.append(
                    {
                        "path": self.path,
                        "body": body,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                    }
                )
                if stub.role == "prefill":
                    self._json(
                        200,
                        {
                            "id": "chatcmpl-prefill",
                            "object": "chat.completion",
                            "model": "m",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": ""},
                                    "finish_reason": "length",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 12,
                                "completion_tokens": 1,
                                "total_tokens": 13,
                            },
                            "kv_transfer_params": PREFILL_KV_PARAMS,
                        },
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for chunk in DECODE_CHUNKS:
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def pd_router():
    binary = _router_binary()
    if binary is None:
        if os.environ.get("BUILDKITE") == "true":
            pytest.fail(
                "vllm-router is missing from the CI image "
                "(docker/Dockerfile, stage router-build)"
            )
        pytest.skip("vllm-router not installed")

    prefill, decode = _StubEngine("prefill"), _StubEngine("decode")
    port = _free_port()
    cmd = [
        binary,
        "--host", "127.0.0.1", "--port", str(port),
        "--vllm-pd-disaggregation",
        "--prefill", prefill.url,
        "--decode", decode.url,
        "--policy", "consistent_hash",
        "--disable-retries",
        "--worker-startup-timeout-secs", "60",
        "--prometheus-host", "127.0.0.1", "--prometheus-port", str(_free_port()),
    ]  # fmt: skip
    log_path = os.path.join(tempfile.gettempdir(), f"vllm_router_{port}.log")
    with open(log_path, "w") as log:
        router = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 60
        while True:
            if router.poll() is not None:
                pytest.fail(
                    f"vllm-router exited with {router.returncode}; log: {log_path}"
                )
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass
            if time.time() > deadline:
                pytest.fail(f"vllm-router not healthy after 60s; log: {log_path}")
            time.sleep(0.5)
        yield base, prefill, decode
    finally:
        router.terminate()
        try:
            router.wait(timeout=10)
        except subprocess.TimeoutExpired:
            router.kill()
        prefill.close()
        decode.close()


def test_router_pd_two_hops_match_the_toy_proxy(pd_router):
    base, prefill, decode = pd_router
    request = {
        "model": "m",
        "messages": [{"role": "user", "content": "2+2?"}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(request).encode(),
        headers={"Content-Type": "application/json"},
    )
    lines: list[str] = []
    with urllib.request.urlopen(req, timeout=30) as response:
        assert response.status == 200
        assert response.headers.get("Content-Type", "").startswith("text/event-stream")
        for raw in response:
            line = raw.decode().strip()
            if line:
                lines.append(line)

    # Hop 1: prefill, one token, not streamed, told that decode happens elsewhere.
    assert len(prefill.requests) == 1, prefill.requests
    p = prefill.requests[0]
    assert p["path"] == "/v1/chat/completions"
    assert p["body"]["max_tokens"] == 1
    assert p["body"]["stream"] is False
    assert "stream_options" not in p["body"]
    assert p["body"]["messages"] == request["messages"]
    assert p["body"]["kv_transfer_params"]["do_remote_decode"] is True
    assert p["body"]["kv_transfer_params"]["do_remote_prefill"] is False

    # Hop 2: decode, the original request plus what prefill answered with.
    assert len(decode.requests) == 1, decode.requests
    d = decode.requests[0]
    assert d["path"] == "/v1/chat/completions"
    assert d["body"]["max_tokens"] == 32
    assert d["body"]["stream"] is True
    assert d["body"]["stream_options"] == {"include_usage": True}
    assert d["body"]["messages"] == request["messages"]
    assert d["body"]["kv_transfer_params"] == PREFILL_KV_PARAMS

    # Both hops carry the same request id, as the toy proxy's do.
    assert p["headers"].get("x-request-id")
    assert p["headers"]["x-request-id"] == d["headers"].get("x-request-id")

    # The client sees decode's stream, unchanged, through to [DONE].
    data = [line[len("data:") :].strip() for line in lines if line.startswith("data:")]
    assert data[-1] == "[DONE]", lines
    assert [json.loads(x) for x in data[:-1]] == DECODE_CHUNKS

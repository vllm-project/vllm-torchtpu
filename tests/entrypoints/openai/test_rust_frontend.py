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

"""`vllm serve` with VLLM_USE_RUST_FRONTEND=1 serves through the vllm-rs
binary the image ships (docker/Dockerfile, stage `rust-build`): the binary is
found, it is the process answering (/version reports a Rust frontend version),
and the main OpenAI routes work against a TPU engine.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import pytest
import vllm
from tpu_release import wait_for_tpu_release

MODEL = "Qwen/Qwen3-0.6B"
SERVE_ARGS = [
    "--max-model-len",
    "512",
    "--max-num-batched-tokens",
    "512",
    "--max-num-seqs",
    "4",
]
STARTUP_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 300


def _rust_frontend_binary() -> str:
    """Where `vllm serve` will look for vllm-rs: the env override, else the
    package directory the `auto` default resolves to."""
    explicit = os.environ.get("VLLM_RUST_FRONTEND_PATH")
    if explicit and explicit.lower() not in ("auto", "1", "true"):
        return explicit
    return os.path.join(os.path.dirname(vllm.__file__), "vllm-rs")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get(url: str, timeout: float = REQUEST_TIMEOUT_S) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def _post_json(url: str, payload: dict) -> bytes:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        return response.read()


def _tail(path: str, lines: int = 60) -> str:
    with open(path, errors="replace") as fh:
        return "\n".join(fh.read().splitlines()[-lines:])


@pytest.fixture(scope="module")
def rust_served_base():
    """`vllm serve` with the Rust frontend; yields the server's base URL."""
    binary = _rust_frontend_binary()
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        pytest.fail(
            f"no vllm-rs binary at {binary}: the image did not build the Rust "
            "frontend (docker/Dockerfile, stage rust-build) or "
            "VLLM_RUST_FRONTEND_PATH points elsewhere"
        )

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *SERVE_ARGS,
    ]
    env = dict(
        os.environ,
        VLLM_USE_RUST_FRONTEND="1",
        VLLM_ENGINE_READY_TIMEOUT_S=str(STARTUP_TIMEOUT_S),
    )
    # A file, not a pipe: startup logs more than a pipe buffer holds and
    # nothing drains it here.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", prefix="vllm_serve_rust_", delete=False
    ) as log:
        server = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            start_new_session=True,
        )
    try:
        deadline = time.time() + STARTUP_TIMEOUT_S
        while True:
            if server.poll() is not None:
                pytest.fail(
                    f"vllm serve exited with {server.returncode} during "
                    f"startup; log {log.name}:\n{_tail(log.name)}"
                )
            try:
                _get(f"{base}/health", timeout=10)
                break
            except (urllib.error.URLError, OSError):
                pass
            if time.time() > deadline:
                pytest.fail(
                    f"vllm serve not healthy after {STARTUP_TIMEOUT_S}s; "
                    f"log {log.name}:\n{_tail(log.name)}"
                )
            time.sleep(5)
        print(f"vllm serve (Rust frontend) ready at {base}; log: {log.name}")
        yield base
    finally:
        server.terminate()
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            # The server leads its process group, which holds the engine and
            # workers; killing only the server would leave them on the chip.
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=60)
        wait_for_tpu_release()


@pytest.mark.tpu_test
@pytest.mark.timeout(1500)
def test_rust_frontend_answers(rust_served_base: str):
    """/version is the Rust route: the Python server has no
    rust_frontend_version field."""
    body = json.loads(_get(f"{rust_served_base}/version"))
    assert body.get("rust_frontend_version"), body
    assert body["version"] == vllm.__version__, body


@pytest.mark.tpu_test
@pytest.mark.timeout(1500)
def test_rust_frontend_chat_completion(rust_served_base: str):
    body = json.loads(
        _post_json(
            f"{rust_served_base}/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 8,
                "temperature": 0,
            },
        )
    )
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant", body
    assert body["usage"]["completion_tokens"] >= 1, body
    assert body["usage"]["prompt_tokens"] >= 1, body


@pytest.mark.tpu_test
@pytest.mark.timeout(1500)
def test_rust_frontend_streaming_completion(rust_served_base: str):
    request = urllib.request.Request(
        f"{rust_served_base}/v1/completions",
        data=json.dumps(
            {
                "model": MODEL,
                "prompt": "The capital of France is",
                "max_tokens": 8,
                "temperature": 0,
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    chunks: list[dict] = []
    done = False
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        assert response.headers.get("Content-Type", "").startswith("text/event-stream")
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                done = True
                break
            chunks.append(json.loads(data))
    assert done, "stream ended without [DONE]"
    text = "".join(c["choices"][0]["text"] for c in chunks if c.get("choices"))
    assert text, chunks


@pytest.mark.tpu_test
@pytest.mark.timeout(1500)
def test_rust_frontend_metrics(rust_served_base: str):
    """/metrics is served by the frontend, so it must come from vllm-rs."""
    body = _get(f"{rust_served_base}/metrics").decode()
    assert "vllm:" in body, body[:500]

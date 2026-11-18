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
"""Tests for the endpoint and model name run_eval_flow.sh hands its clients.

Every client (`lm_eval`, `evalscope`, `vllm bench serve`) is a stub on PATH
that records its argv, so the tests read the exact URL and model name each one
was given. The JSON schema client is the real one and talks to a local server.
"""

import http.server
import json
import pathlib
import stat
import subprocess
import threading
from typing import Any

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RUN_EVAL_FLOW = _REPO_ROOT / "scripts" / "vllm" / "benchmarking" / "run_eval_flow.sh"
_CONFIG = "qwen3-0.6b-structured-flat-json"
_CONFIG_MODEL = "Qwen/Qwen3-0.6B"

_RECORD_ARGV = """#!/bin/bash
printf '%s\\n' "$@" >> "$STUB_LOG_DIR/$(basename "$0").args"
"""

# evalscope must leave a report behind, or the flow fails the suite.
_EVALSCOPE_STUB = (
    _RECORD_ARGV
    + """
while [ $# -gt 0 ]; do
    if [ "$1" = "--work-dir" ]; then work_dir="$2"; fi
    shift
done
mkdir -p "$work_dir/reports/stub"
echo '{"score": 0.5, "num": 4}' > "$work_dir/reports/stub/mmmu_pro.json"
"""
)


class _Gateway:
    """An OpenAI-compatible endpoint that only answers under `prefix`."""

    def __init__(self, prefix: str, model_ids: list[str]):
        self.prefix = prefix
        self.model_ids = model_ids
        self.chat_requests: list[tuple[str, dict[str, Any]]] = []
        gateway = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _reply(self, status: int, obj: Any) -> None:
                body = json.dumps(obj).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == f"{gateway.prefix}/v1/models":
                    ids = [{"id": model_id} for model_id in gateway.model_ids]
                    self._reply(200, {"data": ids})
                else:
                    self._reply(404, {})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                gateway.chat_requests.append((self.path, payload))
                if self.path != f"{gateway.prefix}/v1/chat/completions":
                    self._reply(404, {})
                    return
                # Matches no schema: these tests read where requests went,
                # not the score (see EVAL_SOFT_FAIL in _Flow.run_unchecked).
                content = json.dumps({"ok": True})
                self._reply(
                    200,
                    {
                        "choices": [
                            {"message": {"content": content}, "finish_reason": "stop"}
                        ]
                    },
                )

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}{prefix}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def make_gateway():
    gateways = []

    def _make(prefix: str, model_ids: list[str]) -> _Gateway:
        gateways.append(_Gateway(prefix, model_ids))
        return gateways[-1]

    yield _make
    for gateway in gateways:
        gateway.close()


class _Flow:
    """Runs run_eval_flow.sh with stub clients and reads back their argv."""

    def __init__(self, tmp_path: pathlib.Path):
        self._tmp_path = tmp_path
        self._stub_logs = tmp_path / "stub_logs"
        self._stub_logs.mkdir()
        self._bin = tmp_path / "bin"
        self._bin.mkdir()
        for name, body in (
            ("lm_eval", _RECORD_ARGV),
            ("vllm", _RECORD_ARGV),
            ("evalscope", _EVALSCOPE_STUB),
        ):
            stub = self._bin / name
            stub.write_text(body)
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
        # run_benchmarks.sh only checks that the vllm package can be found.
        self._site = tmp_path / "site"
        (self._site / "vllm").mkdir(parents=True)
        (self._site / "vllm" / "__init__.py").write_text("")

    def run(
        self, *args: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        proc = self.run_unchecked(*args, env=env)
        assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        return proc

    def run_unchecked(
        self, *args: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "bash",
                str(_RUN_EVAL_FLOW),
                "--config",
                _CONFIG,
                "--results-dir",
                str(self._tmp_path / "results"),
                "--skip-db-upload",
                "--skip-perf-check",
                *args,
            ],
            cwd=str(_REPO_ROOT),
            env={
                "PYTHONPATH": str(self._site),
                "PATH": f"{self._bin}:/usr/bin:/bin",
                "STUB_LOG_DIR": str(self._stub_logs),
                "LM_EVAL_REQUEST_CACHE_ROOT": str(self._tmp_path / "cache"),
                # The gateway's canned reply fails the json_schema baseline.
                "EVAL_SOFT_FAIL": "1",
                **(env or {}),
            },
            capture_output=True,
            text=True,
        )

    def argv(self, client: str) -> list[str]:
        return (self._stub_logs / f"{client}.args").read_text().splitlines()

    def option(self, client: str, flag: str) -> set[str]:
        """Every distinct value `client` was given for `flag`, over all calls."""
        argv = self.argv(client)
        return {argv[i + 1] for i, arg in enumerate(argv) if arg == flag}


@pytest.fixture
def flow(tmp_path):
    return _Flow(tmp_path)


def test_base_url_path_prefix_and_served_model_reach_every_client(flow, make_gateway):
    gateway = make_gateway("/gw", ["router-served-name"])

    flow.run(
        "--base-url",
        gateway.url,
        "--eval-suites",
        "json_schema,mmlu_pro,mmmu_pro",
        "--perf-suites",
        "bench_serve",
    )

    chat_url = f"http://127.0.0.1:{gateway.port}/gw/v1/chat/completions"
    assert flow.option("lm_eval", "--model_args") == {
        f"model=router-served-name,base_url={chat_url},num_concurrent=128"
    }
    assert flow.option("evalscope", "--api-url") == {chat_url}
    assert flow.option("evalscope", "--model") == {"router-served-name"}
    assert flow.option("vllm", "--base-url") == {f"http://127.0.0.1:{gateway.port}/gw"}
    assert flow.option("vllm", "--served-model-name") == {"router-served-name"}
    # The tokenizer still comes from the config's model.
    assert flow.option("vllm", "--model") == {_CONFIG_MODEL}
    assert flow.option("vllm", "--host") == set()
    assert flow.option("vllm", "--port") == set()
    assert gateway.chat_requests
    assert {path for path, _ in gateway.chat_requests} == {"/gw/v1/chat/completions"}
    assert {payload["model"] for _, payload in gateway.chat_requests} == {
        "router-served-name"
    }


@pytest.mark.parametrize(
    ("args", "env"),
    [
        pytest.param(["--served-model-name", "my-served"], {}, id="flag"),
        pytest.param([], {"SERVED_MODEL_NAME": "my-served"}, id="env"),
        pytest.param(
            ["--served-model-name", "my-served"],
            {"SERVED_MODEL_NAME": "from-env"},
            id="flag_beats_env",
        ),
    ],
)
def test_https_base_url_and_explicit_served_model(flow, args, env):
    # No server is listening: an explicit name must not need /v1/models.
    flow.run(
        "--base-url",
        "https://gw.example.com/prefix/v1/",
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
        *args,
        env=env,
    )

    assert flow.option("lm_eval", "--model_args") == {
        "model=my-served,"
        "base_url=https://gw.example.com/prefix/v1/chat/completions,"
        "num_concurrent=128"
    }


def test_explicit_served_model_beats_the_one_the_endpoint_lists(flow, make_gateway):
    gateway = make_gateway("", ["router-served-name"])

    flow.run(
        "--base-url",
        gateway.url,
        "--served-model-name",
        "my-served",
        "--eval-suites",
        "json_schema",
        "--perf-suites",
        "none",
    )

    assert {payload["model"] for _, payload in gateway.chat_requests} == {"my-served"}


def test_base_url_keeps_config_model_when_endpoint_lists_several(flow, make_gateway):
    gateway = make_gateway("", ["model-a", "model-b"])

    proc = flow.run(
        "--base-url",
        gateway.url,
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
    )

    assert flow.option("lm_eval", "--model_args") == {
        f"model={_CONFIG_MODEL},"
        f"base_url=http://127.0.0.1:{gateway.port}/v1/chat/completions,"
        "num_concurrent=128"
    }
    assert f"using the config's model {_CONFIG_MODEL}" in proc.stdout


def test_host_and_port_keep_config_model_and_host_port_flags(flow, make_gateway):
    gateway = make_gateway("", ["router-served-name"])

    flow.run(
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway.port),
        "--eval-suites",
        "mmlu_pro,mmmu_pro",
        "--perf-suites",
        "bench_serve",
        env={"SERVED_MODEL_NAME": "ignored-without-base-url"},
    )

    chat_url = f"http://127.0.0.1:{gateway.port}/v1/chat/completions"
    assert flow.option("lm_eval", "--model_args") == {
        f"model={_CONFIG_MODEL},base_url={chat_url},num_concurrent=128"
    }
    assert flow.option("evalscope", "--api-url") == {chat_url}
    assert flow.option("evalscope", "--model") == {_CONFIG_MODEL}
    assert flow.option("vllm", "--host") == {"127.0.0.1"}
    assert flow.option("vllm", "--port") == {str(gateway.port)}
    assert flow.option("vllm", "--model") == {_CONFIG_MODEL}
    assert flow.option("vllm", "--base-url") == set()
    assert flow.option("vllm", "--served-model-name") == set()


@pytest.mark.parametrize(
    "base_url",
    ["ftp://gw.example.com", "http://", "http://gw.example.com/v1?key=1"],
)
def test_invalid_base_url_fails_before_any_client_runs(flow, base_url):
    proc = flow.run_unchecked(
        "--base-url", base_url, "--eval-suites", "mmlu_pro", "--perf-suites", "none"
    )

    assert proc.returncode == 1
    assert f"Invalid --base-url: {base_url}" in proc.stderr
    assert "Running benchmarks" not in proc.stdout

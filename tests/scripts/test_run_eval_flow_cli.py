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
"""Tests for endpoint_url.py, run_benchmarks.sh, and run_eval_flow.sh flags.

Every test runs against a hermetic copy of ``scripts/vllm/benchmarking`` under
``tmp_path`` with fixture configs, stubbed client CLIs (``lm_eval``,
``evalscope``, ``vllm``, ``curl``), and a stubbed ``cleanup_server.sh``. The
``json_schema`` suite runs the real ``eval_json_schema.py`` against the fixture
gateway. Nothing writes to the repository tree or kills host processes.
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import pathlib
import shutil
import socket
import stat
import subprocess
import threading
from collections.abc import Callable
from typing import Any

import pytest
from jax.experimental.pallas import tpu as pltpu

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BENCHMARKING_SRC = _REPO_ROOT / "scripts" / "vllm" / "benchmarking"
_COPIED_FILES = (
    "endpoint_url.py",
    "eval_json_schema.py",
    "run_eval_flow.sh",
    "run_benchmarks.sh",
    "upload_results.py",
    "bq_utils.py",
    "check_regression.py",
    "publish_perf_dashboard.py",
)
# Data the flow's suites read: lm_eval task definitions, and the json_schema
# suite's task configs with the schemas and the dataset they point at.
_COPIED_DIRS = ("lm_eval_tasks", "task_configs", "structured_schemas", "datasets")
_JSON_SCHEMA_PASS_RULE = "baselines/eval/json_schema.pass_rule.baseline.json"

_FIXTURE_CONFIG = "fixture-model"
_MULTIMODAL_CONFIG = "fixture-multimodal"
_CONFIG_MODEL = "FixtureOrg/Fixture-Model"
_FIXTURE_CONFIG_BODY = f"""MODEL="{_CONFIG_MODEL}"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ISL_OSL_CONFIGS="128:64"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=4
RANDOM_RANGE_RATIO=0.1
EVAL_TOLERANCE=0.05
"""

# Shaped like configs/qwen3-0.6b-ngram.sh: spec_bench configs set no
# RANDOM_RANGE_RATIO or EVAL_TOLERANCE.
_SPEC_BENCH_CONFIG = "fixture-spec-bench"
_SPEC_BENCH_CONFIG_BODY = f"""MODEL="{_CONFIG_MODEL}"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
BENCHMARK_MODE="spec_bench"
SPEC_METHOD="ngram"
SPEC_K_LIST="0 2"
SPEC_BS_LIST="1"
"""

_spec = importlib.util.spec_from_file_location(
    "endpoint_url", _BENCHMARKING_SRC / "endpoint_url.py"
)
assert _spec is not None and _spec.loader is not None
endpoint_url = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(endpoint_url)

_RECORD_ARGV = """#!/bin/bash
printf '%s\\n' "$@" >> "$STUB_LOG_DIR/$(basename "$0").args"
"""

_VLLM_STUB = """#!/bin/bash
printf '%s\\n' "$@" >> "$STUB_LOG_DIR/vllm.args"
if [ "${1:-}" = "serve" ]; then
    echo "$$" > "$STUB_LOG_DIR/vllm_serve.pid"
    touch "$STUB_LOG_DIR/server.ready"
    exec sleep 30
fi
"""

_CURL_STUB = """#!/bin/bash
case "$*" in
    */health*)
        [ -f "$STUB_LOG_DIR/server.ready" ] && exit 0
        exit 7
        ;;
esac
exit 1
"""

_CLEANUP_STUB = """#!/bin/bash
echo "cleanup" >> "$STUB_LOG_DIR/cleanup_server.args"
if [ -f "$STUB_LOG_DIR/vllm_serve.pid" ]; then
    kill "$(cat "$STUB_LOG_DIR/vllm_serve.pid")" 2>/dev/null || true
    rm -f "$STUB_LOG_DIR/vllm_serve.pid" "$STUB_LOG_DIR/server.ready"
fi
"""

# Stands in for a repo Python script, so a test can read the arguments the flow
# gave it and the script never reaches a database or the network.
_PYTHON_RECORD_ARGV = """import os
import pathlib
import sys

log = pathlib.Path(os.environ["STUB_LOG_DIR"]) / (pathlib.Path(__file__).stem + ".args")
with log.open("a") as f:
    f.writelines(arg + "\\n" for arg in sys.argv[1:])
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


# One valid answer for each case of task_configs/json_schema_default.yaml.
_VALID_JSON_SCHEMA_ANSWERS: dict[str, Any] = {
    "flat_json": {
        "users": [
            {
                "name": "Ada",
                "age": 36,
                "email": "ada@example.com",
                "is_active": True,
                "role": "admin",
            }
        ]
        * 10
    },
    "complex_ast": {
        "expressions": [
            {
                "ast": {
                    "op": "+",
                    "left": 1,
                    "right": {"op": "*", "left": "x", "right": 2},
                },
                "description": "one plus twice x",
            }
        ]
        * 10
    },
    "character_schema": {
        "name": "Harry Potter",
        "house": "Gryffindor",
        "blood status": "Half-blood",
        "occupation": "auror",
        "wand": {"wood": "Holly", "core": "Phoenix feather", "length": 11},
        "alive": "Alive",
        "patronus": "Stag",
        "bogart": "Dementor",
    },
    "tool_call_arguments": {
        "tool_name": "query_logs",
        "arguments": {"service": "gateway-router", "limit": 50, "severity": "ERROR"},
    },
}

_ChatReply = tuple[int, dict[str, Any]]


def _chat_completion(content: str, finish_reason: str = "stop") -> _ChatReply:
    """A 200 chat completion with ``content`` as its text."""
    message = {"content": content}
    return 200, {"choices": [{"message": message, "finish_reason": finish_reason}]}


def _valid_json_schema_reply(payload: dict[str, Any]) -> _ChatReply:
    case = payload["response_format"]["json_schema"]["name"]
    return _chat_completion(json.dumps(_VALID_JSON_SCHEMA_ANSWERS[case]))


class _Gateway:
    """Lists ``model_ids`` at ``<prefix>/v1/models`` with optional bearer auth.

    ``models_reply`` replaces the model list with a raw ``(status, body)``.
    Chat completions are answered by ``chat_reply``, which by default gives
    each case of the shipped ``json_schema`` task configs a valid answer.
    """

    def __init__(
        self,
        prefix: str,
        model_ids: list[str],
        required_api_key: str | None = None,
        models_reply: tuple[int, bytes] | None = None,
        chat_reply: Callable[[dict[str, Any]], _ChatReply] = _valid_json_schema_reply,
    ):
        self.prefix = prefix
        self.model_ids = model_ids
        self.required_api_key = required_api_key
        self.models_reply = models_reply
        self.seen_auth_headers: list[str | None] = []
        self.chat_reply = chat_reply
        self.chat_requests: list[dict[str, Any]] = []
        gateway = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _reply(self, status: int, obj: Any) -> None:
                self._reply_raw(status, json.dumps(obj).encode("utf-8"))

            def _reply_raw(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self) -> bool:
                key = gateway.required_api_key
                if key is None or self.headers.get("Authorization") == f"Bearer {key}":
                    return True
                self._reply(401, {"error": "unauthorized"})
                return False

            def do_GET(self) -> None:
                gateway.seen_auth_headers.append(self.headers.get("Authorization"))
                if not self._authorized():
                    return
                if self.path != f"{gateway.prefix}/v1/models":
                    self._reply(404, {})
                elif gateway.models_reply is not None:
                    self._reply_raw(*gateway.models_reply)
                else:
                    ids = [{"id": model_id} for model_id in gateway.model_ids]
                    self._reply(200, {"data": ids})

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if not self._authorized():
                    return
                if self.path != f"{gateway.prefix}/v1/chat/completions":
                    self._reply(404, {})
                    return
                payload = json.loads(body)
                gateway.chat_requests.append(payload)
                self._reply(*gateway.chat_reply(payload))

        class Server(http.server.ThreadingHTTPServer):
            # eval_json_schema.py connects 16 times at once. The default
            # backlog of 5 drops some of those, and each retry takes a second.
            request_queue_size = 32

        self._server = Server(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}{prefix}"
        # A short poll, so closing a gateway after each test does not wait 0.5 s.
        threading.Thread(
            target=self._server.serve_forever, args=(0.05,), daemon=True
        ).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def make_gateway():
    gateways = []

    def _make(
        prefix: str,
        model_ids: list[str],
        required_api_key: str | None = None,
        models_reply: tuple[int, bytes] | None = None,
        chat_reply: Callable[[dict[str, Any]], _ChatReply] = _valid_json_schema_reply,
    ) -> _Gateway:
        gateways.append(
            _Gateway(
                prefix,
                model_ids,
                required_api_key=required_api_key,
                models_reply=models_reply,
                chat_reply=chat_reply,
            )
        )
        return gateways[-1]

    yield _make
    for gateway in gateways:
        gateway.close()


class _Flow:
    """Runs scripts in a hermetic copy of ``scripts/vllm/benchmarking``."""

    def __init__(self, tmp_path: pathlib.Path, server: _Gateway):
        # The already-running server of --host runs: it lists the fixture
        # config's MODEL, which the flow checks before it sends a request.
        self.server = server
        self.host_args = ("--host", "127.0.0.1", "--port", str(server.port))
        self._tmp_path = tmp_path
        self._repo = tmp_path / "repo"
        self._bench_dir = self._repo / "scripts" / "vllm" / "benchmarking"
        self._configs_dir = self._bench_dir / "configs"
        self._configs_dir.mkdir(parents=True)
        (self._bench_dir / "baselines" / "eval").mkdir(parents=True)
        (self._bench_dir / "baselines" / "perf").mkdir(parents=True)
        for name in _COPIED_DIRS:
            shutil.copytree(_BENCHMARKING_SRC / name, self._bench_dir / name)
        for name in (*_COPIED_FILES, _JSON_SCHEMA_PASS_RULE):
            shutil.copy2(_BENCHMARKING_SRC / name, self._bench_dir / name)

        cleanup_script = self._bench_dir / "cleanup_server.sh"
        cleanup_script.write_text(_CLEANUP_STUB)
        cleanup_script.chmod(cleanup_script.stat().st_mode | stat.S_IXUSR)

        self.write_config(_FIXTURE_CONFIG, _FIXTURE_CONFIG_BODY)
        self.write_config(_MULTIMODAL_CONFIG, _FIXTURE_CONFIG_BODY)
        self.write_config(_SPEC_BENCH_CONFIG, _SPEC_BENCH_CONFIG_BODY)
        self.results_dir = tmp_path / "results"

        self._stub_logs = tmp_path / "stub_logs"
        self._stub_logs.mkdir()
        self._bin = tmp_path / "bin"
        self._bin.mkdir()
        for name, body in (
            ("lm_eval", _RECORD_ARGV),
            ("vllm", _VLLM_STUB),
            ("curl", _CURL_STUB),
            ("evalscope", _EVALSCOPE_STUB),
        ):
            stub = self._bin / name
            stub.write_text(body)
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)

        # run_benchmarks.sh checks that the vllm package can be found.
        self._site = tmp_path / "site"
        (self._site / "vllm").mkdir(parents=True)
        (self._site / "vllm" / "__init__.py").write_text("")

    def write_config(self, name: str, body: str) -> str:
        (self._configs_dir / f"{name}.sh").write_text(body)
        return name

    def stub_python_script(self, name: str) -> None:
        """Replaces ``<name>.py`` with a stub; ``argv(name)`` reads its calls."""
        (self._bench_dir / f"{name}.py").write_text(_PYTHON_RECORD_ARGV)

    def write_perf_baseline(self, config: str = _FIXTURE_CONFIG) -> str:
        """Writes an empty v7x perf baseline; returns the path the flow passes on."""
        relative = (
            f"scripts/vllm/benchmarking/baselines/perf/v7x/{config}.baseline.json"
        )
        (self._repo / relative).parent.mkdir()
        (self._repo / relative).write_text("{}")
        return relative

    def _base_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        return {
            "PYTHONPATH": str(self._site),
            "PATH": f"{self._bin}:/usr/bin:/bin",
            "STUB_LOG_DIR": str(self._stub_logs),
            "LM_EVAL_REQUEST_CACHE_ROOT": str(self._tmp_path / "cache"),
            **(extra or {}),
        }

    def run(
        self,
        *args: str,
        env: dict[str, str] | None = None,
        config: str = _FIXTURE_CONFIG,
        skip_upload_and_perf_check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        proc = self.run_unchecked(
            *args,
            env=env,
            config=config,
            skip_upload_and_perf_check=skip_upload_and_perf_check,
        )
        assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        return proc

    def run_unchecked(
        self,
        *args: str,
        env: dict[str, str] | None = None,
        config: str = _FIXTURE_CONFIG,
        skip_upload_and_perf_check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        skip_flags = (
            ["--skip-db-upload", "--skip-perf-check"]
            if skip_upload_and_perf_check
            else []
        )
        return subprocess.run(
            [
                "bash",
                str(self._bench_dir / "run_eval_flow.sh"),
                "--config",
                config,
                "--results-dir",
                str(self.results_dir),
                *skip_flags,
                *args,
            ],
            cwd=str(self._repo),
            env=self._base_env(env),
            capture_output=True,
            text=True,
        )

    def run_benchmarks(
        self,
        *args: str,
        env: dict[str, str] | None = None,
        config: str = _FIXTURE_CONFIG,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self._bench_dir / "run_benchmarks.sh"),
                "--config",
                config,
                "--results-dir",
                str(self._tmp_path / "bench_results"),
                *args,
            ],
            cwd=str(self._repo),
            env=self._base_env(env),
            capture_output=True,
            text=True,
        )

    def argv(self, client: str) -> list[str]:
        """The arguments of every call to ``client``, or [] if it never ran."""
        log = self._stub_logs / f"{client}.args"
        return log.read_text().splitlines() if log.exists() else []

    def values(self, client: str, flag: str) -> list[str]:
        """Each value ``client`` was given for ``flag``, in call order."""
        argv = self.argv(client)
        return [argv[i + 1] for i, arg in enumerate(argv) if arg == flag]

    def option(self, client: str, flag: str) -> set[str]:
        """Every distinct value ``client`` was given for ``flag``, over all calls."""
        return set(self.values(client, flag))


@pytest.fixture
def flow(tmp_path, make_gateway):
    return _Flow(tmp_path, make_gateway("", [_CONFIG_MODEL]))


# --- Shared endpoint URL helper tests ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("gw.example.com", "http://gw.example.com:8000"),
        ("gw.example.com/prefix", "http://gw.example.com:8000/prefix"),
        ("gw.example.com:9000/v1", "http://gw.example.com:9000"),
        ("v1", "http://v1:8000"),
        ("http://v1", "http://v1"),
        ("http://v1/v1", "http://v1"),
        ("http://v1/v1/", "http://v1"),
        ("http://gw.example.com", "http://gw.example.com"),
        ("https://gw.example.com", "https://gw.example.com"),
        ("https://gw.example.com:8443/prefix/", "https://gw.example.com:8443/prefix"),
        ("https://gw.example.com/prefix/v1/", "https://gw.example.com/prefix"),
        ("http://[::1]:8000", "http://[::1]:8000"),
        ("[::1]", "http://[::1]:8000"),
    ],
)
def test_normalize_base_url_valid(raw, expected):
    assert endpoint_url.normalize_base_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "ftp://gw.example.com",
        "http://",
        "http://gw.example.com/v1?key=1",
        "http://gw.example.com#frag",
        "http://gw.example.com:notaport",
        "http://user:secret@gw.example.com",
        "http://gw.example.com /bad",
        "http://gw.example.com/v1/chat/completions",
        "https://gw.example.com/prefix/v1/chat/completions",
        "http://gw.example.com/v1/models",
        "http://gw.example.com/v1/v1",
        "gw.example.com:",
        "http://gw.example.com:",
        "http:/gw.example.com",
    ],
)
def test_normalize_base_url_rejects_invalid_and_route_urls(raw):
    with pytest.raises(ValueError, match="Invalid --base-url"):
        endpoint_url.normalize_base_url(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "http://user:s3cret@gw.example.com",
        "user:s3cret@gw.example.com",
        "http://user:s3cret@gw.example.com /bad",
        "http://user:s3cret@gw.example.com:notaport",
        "http://user:s3cret@gw.example.com/v1/models",
        "ftp://user:s3cret@gw.example.com",
    ],
)
def test_normalize_base_url_error_does_not_show_a_value_with_userinfo(raw):
    with pytest.raises(ValueError, match="Invalid --base-url") as excinfo:
        endpoint_url.normalize_base_url(raw)

    assert "s3cret" not in str(excinfo.value)
    assert "user:" not in str(excinfo.value)


def test_normalize_base_url_error_shows_a_value_without_userinfo():
    with pytest.raises(ValueError, match="Invalid --base-url: ftp://gw.example.com$"):
        endpoint_url.normalize_base_url("ftp://gw.example.com")


def test_list_served_models_sends_bearer_token(make_gateway):
    gateway = make_gateway("/gw", ["authed-model"], required_api_key="secret-token")

    with pytest.raises(endpoint_url.ModelLookupError, match="401"):
        endpoint_url.list_served_models(gateway.url)
    assert endpoint_url.list_served_models(gateway.url, api_key="secret-token") == [
        "authed-model"
    ]
    assert gateway.seen_auth_headers == [None, "Bearer secret-token"]


@pytest.mark.parametrize("model_ids", [[], ["only-model"], ["model-a", "model-b"]])
def test_list_served_models_returns_every_listed_id(make_gateway, model_ids):
    gateway = make_gateway("", model_ids)

    assert endpoint_url.list_served_models(gateway.url) == model_ids


def test_list_served_models_skips_entries_without_a_string_id(make_gateway):
    body = json.dumps({"data": [{"id": 7}, "text", {"id": "only-model"}]}).encode()
    gateway = make_gateway("", [], models_reply=(200, body))

    assert endpoint_url.list_served_models(gateway.url) == ["only-model"]


@pytest.mark.parametrize(
    ("models_reply", "message"),
    [
        pytest.param((500, b"{}"), "HTTP Error 500", id="http_error"),
        pytest.param((200, b"not json"), "Expecting value", id="malformed_json"),
        pytest.param((200, b"[]"), "not a JSON model list", id="json_not_an_object"),
        pytest.param(
            (200, b'{"error": "no models"}'),
            "not a JSON model list",
            id="no_data_list",
        ),
    ],
)
def test_list_served_models_raises_when_the_lookup_fails(
    make_gateway, models_reply, message
):
    gateway = make_gateway("", ["unused"], models_reply=models_reply)

    with pytest.raises(endpoint_url.ModelLookupError, match=message):
        endpoint_url.list_served_models(gateway.url)


def test_list_served_models_raises_when_nothing_is_listening():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

        with pytest.raises(endpoint_url.ModelLookupError, match="Connection refused"):
            endpoint_url.list_served_models(f"http://127.0.0.1:{port}")


@pytest.mark.parametrize(
    ("listed", "served_model_name", "expected"),
    [
        pytest.param(["cfg"], None, "cfg", id="config_model_listed"),
        pytest.param(["a", "cfg"], None, "cfg", id="several_with_config_model"),
        pytest.param(["a", "b"], "b", "b", id="explicit_name_listed"),
        pytest.param(["b"], "b", "b", id="explicit_name_is_the_single_one"),
    ],
)
def test_choose_served_model_returns_a_listed_name(listed, served_model_name, expected):
    name = endpoint_url.choose_served_model(
        listed,
        config_model="cfg",
        served_model_name=served_model_name,
    )

    assert name == expected


@pytest.mark.parametrize(
    ("listed", "served_model_name", "message"),
    [
        pytest.param(
            [],
            None,
            "the config's MODEL 'cfg' is not a model the endpoint serves "
            "(it lists: no models)",
            id="no_ids",
        ),
        pytest.param(
            ["a", "b"],
            None,
            "the config's MODEL 'cfg' is not a model the endpoint serves "
            "(it lists: a, b)",
            id="several_without_config_model",
        ),
        pytest.param(
            ["a"],
            None,
            "the config's MODEL 'cfg' is not a model the endpoint serves (it lists: a)",
            id="single_other_name_not_adopted",
        ),
        pytest.param(
            ["a", "cfg"],
            "x",
            "--served-model-name 'x' is not a model the endpoint serves "
            "(it lists: a, cfg)",
            id="explicit_name_not_listed",
        ),
    ],
)
def test_choose_served_model_raises_for_a_name_the_endpoint_does_not_list(
    listed, served_model_name, message
):
    with pytest.raises(endpoint_url.ModelNotServedError) as excinfo:
        endpoint_url.choose_served_model(
            listed,
            config_model="cfg",
            served_model_name=served_model_name,
        )

    assert str(excinfo.value) == message


# --- Direct run_benchmarks.sh validation tests ---


def test_run_benchmarks_normalizes_base_url_and_keeps_v1_host(flow):
    proc = flow.run_benchmarks("--base-url", "http://v1/v1/")

    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert flow.option("vllm", "--base-url") == {"http://v1"}
    assert flow.option("vllm", "--host") == set()
    assert flow.option("vllm", "--port") == set()


@pytest.mark.parametrize(
    ("args", "expected_stream", "expected_text"),
    [
        (
            ["--base-url", "http://user:s3cret@gw.example.com"],
            "stderr",
            "Invalid --base-url",
        ),
        (
            ["--base-url", "http://gw.example.com/v1/chat/completions"],
            "stderr",
            "Invalid --base-url",
        ),
        (
            ["--base-url", ""],
            "stdout",
            "ERROR: --base-url cannot be empty",
        ),
        (
            ["--base-url", "http://gw.example.com", "--host", "gw.example.com"],
            "stdout",
            "ERROR: --base-url cannot be combined with --host or --port",
        ),
        (
            ["--base-url", "http://gw.example.com", "--port", "9000"],
            "stdout",
            "ERROR: --base-url cannot be combined with --host or --port",
        ),
        (
            ["--served-model-name", "other-model"],
            "stdout",
            "ERROR: --served-model-name needs a running server",
        ),
        (
            ["--host", "gw.example.com", "--served-model-name", ""],
            "stdout",
            "ERROR: --served-model-name cannot be empty",
        ),
    ],
)
def test_run_benchmarks_rejects_bad_endpoint_flags(
    flow, args, expected_stream, expected_text
):
    proc = flow.run_benchmarks(*args)

    assert proc.returncode == 1
    output = proc.stderr if expected_stream == "stderr" else proc.stdout
    assert expected_text in output
    assert "s3cret" not in proc.stdout + proc.stderr
    assert flow.argv("vllm") == []


@pytest.mark.parametrize(
    "args",
    [
        ["--results-dir", "--dry-run"],
        ["--host", "--base-url", "http://gw.example.com"],
        ["--base-url"],
    ],
)
def test_run_benchmarks_rejects_a_flag_in_place_of_a_value(flow, args):
    proc = flow.run_benchmarks(*args)

    assert proc.returncode == 1
    assert f"ERROR: {args[0]} requires a value" in proc.stdout
    assert flow.argv("vllm") == []


def test_run_benchmarks_rejects_skip_bench_cells_in_spec_bench_mode(flow):
    proc = flow.run_benchmarks(
        "--dry-run", "--skip-bench-cells", config=_SPEC_BENCH_CONFIG
    )

    assert proc.returncode == 1
    assert (
        "ERROR: --skip-bench-cells is not supported in spec_bench mode" in proc.stdout
    )
    assert "Spec-Bench Cell" not in proc.stdout


def test_run_benchmarks_with_model_uri_falls_back_to_model_for_tokenizer(flow):
    # The fixture's empty `vllm` package has no runai_utils, so the probe for
    # the pulled tokenizer directory fails, as on a host without that module.
    config = flow.write_config(
        "model-uri",
        _FIXTURE_CONFIG_BODY + 'MODEL_URI="gs://fixture-bucket/fixture-model"\n',
    )

    proc = flow.run_benchmarks("--host", "gw.example.com", config=config)

    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert flow.option("vllm", "--model") == {_CONFIG_MODEL}
    assert flow.option("vllm", "--tokenizer") == set()
    assert f"bench client will resolve {_CONFIG_MODEL} on the hub" in proc.stderr


# --- run_eval_flow.sh endpoint, auth, and served-model tests ---


def test_base_url_path_prefix_and_served_model_reach_every_client(flow, make_gateway):
    gateway = make_gateway("/gw", ["router-served-name"])

    flow.run(
        "--base-url",
        gateway.url,
        "--served-model-name",
        "router-served-name",
        "--eval-suites",
        "mmlu_pro,mmmu_pro",
        "--perf-suites",
        "bench_serve",
    )

    chat_url = f"http://127.0.0.1:{gateway.port}/gw/v1/chat/completions"
    assert flow.option("lm_eval", "--model_args") == {
        f"model=router-served-name,base_url={chat_url},num_concurrent=128"
    }
    assert flow.option("evalscope", "--api-url") == {chat_url}
    assert flow.option("evalscope", "--api-key") == {"EMPTY"}
    assert flow.option("evalscope", "--model") == {"router-served-name"}
    assert flow.option("vllm", "--base-url") == {f"http://127.0.0.1:{gateway.port}/gw"}
    assert flow.option("vllm", "--served-model-name") == {"router-served-name"}
    assert flow.option("vllm", "--model") == {_CONFIG_MODEL}
    assert flow.option("vllm", "--host") == set()
    assert flow.option("vllm", "--port") == set()
    assert flow.argv("cleanup_server") == []


def test_openai_api_key_reaches_models_lookup_and_evalscope(flow, make_gateway):
    gateway = make_gateway("/gw", [_CONFIG_MODEL], required_api_key="gw-key-123")

    flow.run(
        "--base-url",
        gateway.url,
        "--eval-suites",
        "mmmu_pro",
        "--perf-suites",
        "none",
        env={"OPENAI_API_KEY": "gw-key-123"},
    )

    assert gateway.seen_auth_headers == ["Bearer gw-key-123"]
    assert flow.option("evalscope", "--model") == {_CONFIG_MODEL}
    assert flow.option("evalscope", "--api-key") == {"gw-key-123"}


def test_explicit_served_model_among_several_and_no_env_fallback(flow, make_gateway):
    # SERVED_MODEL_NAME in the environment is ignored.
    gateway = make_gateway("/prefix", ["other-model", "my-served"])

    flow.run(
        "--base-url",
        f"{gateway.url}/v1/",
        "--served-model-name",
        "my-served",
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
        env={"SERVED_MODEL_NAME": "from-env"},
    )

    assert flow.option("lm_eval", "--model_args") == {
        f"model=my-served,base_url={gateway.url}/v1/chat/completions,num_concurrent=128"
    }


def test_served_model_name_env_is_ignored_without_cli_flag(flow, make_gateway):
    gateway = make_gateway("", [_CONFIG_MODEL, "ignored-env-name"])

    flow.run(
        "--base-url",
        gateway.url,
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
        env={"SERVED_MODEL_NAME": "ignored-env-name"},
    )

    assert flow.option("lm_eval", "--model_args") == {
        f"model={_CONFIG_MODEL},"
        f"base_url={gateway.url}/v1/chat/completions,"
        "num_concurrent=128"
    }


def test_explicit_served_model_beats_the_config_model_when_both_are_listed(
    flow, make_gateway
):
    gateway = make_gateway("", ["my-served", _CONFIG_MODEL])

    flow.run(
        "--base-url",
        gateway.url,
        "--served-model-name",
        "my-served",
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
    )

    assert flow.option("lm_eval", "--model_args") == {
        f"model=my-served,base_url={gateway.url}/v1/chat/completions,num_concurrent=128"
    }


def test_base_url_keeps_config_model_when_it_is_among_several_listed(
    flow, make_gateway
):
    gateway = make_gateway("", ["model-a", _CONFIG_MODEL])

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
    assert (
        f"Sending requests for the config's model {_CONFIG_MODEL}, "
        f"which {gateway.url}/v1/models lists" in proc.stdout
    )


@pytest.mark.parametrize(
    "models_reply",
    [
        pytest.param(None, id="nothing_listening"),
        pytest.param((503, b"{}"), id="http_error"),
        pytest.param((200, b"not json"), id="malformed_json"),
    ],
)
def test_base_url_stops_when_the_models_lookup_fails(flow, make_gateway, models_reply):
    gateway = make_gateway("", ["unused"], models_reply=models_reply)
    if models_reply is None:
        gateway.close()

    proc = flow.run_unchecked(
        "--base-url",
        gateway.url,
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
    )

    assert proc.returncode == 1
    assert (
        f"ERROR: Could not list the served models at {gateway.url}/v1/models"
        in proc.stderr
    )
    assert "The run sends no request before it has checked" in proc.stderr
    assert "Running benchmarks" not in proc.stdout
    assert flow.argv("lm_eval") == []


# Every request for a name the endpoint does not list would fail, and
# `vllm bench serve` would still exit 0, so the run stops before any client.
@pytest.mark.parametrize(
    ("endpoint_flag", "model_ids", "extra_args", "message"),
    [
        pytest.param(
            "--base-url",
            [],
            [],
            f"the config's MODEL '{_CONFIG_MODEL}' is not a model the endpoint "
            "serves (it lists: no models)",
            id="base_url_lists_no_models",
        ),
        pytest.param(
            "--base-url",
            ["router-served-name"],
            [],
            f"the config's MODEL '{_CONFIG_MODEL}' is not a model the endpoint "
            "serves (it lists: router-served-name)",
            id="base_url_lists_single_other_model",
        ),
        pytest.param(
            "--base-url",
            ["model-a", "model-b"],
            [],
            f"the config's MODEL '{_CONFIG_MODEL}' is not a model the endpoint "
            "serves (it lists: model-a, model-b)",
            id="base_url_lists_several_without_config_model",
        ),
        pytest.param(
            "--base-url",
            ["router-served-name", _CONFIG_MODEL],
            ["--served-model-name", "my-served"],
            "--served-model-name 'my-served' is not a model the endpoint serves "
            f"(it lists: router-served-name, {_CONFIG_MODEL})",
            id="base_url_explicit_name_not_listed",
        ),
        pytest.param(
            "--host",
            ["router-served-name"],
            [],
            f"the config's MODEL '{_CONFIG_MODEL}' is not a model the endpoint "
            "serves (it lists: router-served-name)",
            id="host_serves_another_name",
        ),
        pytest.param(
            "--host",
            [_CONFIG_MODEL],
            ["--served-model-name", "my-served"],
            "--served-model-name 'my-served' is not a model the endpoint serves "
            f"(it lists: {_CONFIG_MODEL})",
            id="host_explicit_name_not_listed",
        ),
    ],
)
def test_run_stops_when_the_endpoint_does_not_list_the_model_to_send(
    flow, make_gateway, endpoint_flag, model_ids, extra_args, message
):
    gateway = make_gateway("", model_ids)
    endpoint_args = (
        ["--base-url", gateway.url]
        if endpoint_flag == "--base-url"
        else ["--host", "127.0.0.1", "--port", str(gateway.port)]
    )

    proc = flow.run_unchecked(*endpoint_args, *extra_args, "--eval-suites", "mmlu_pro")

    assert proc.returncode == 1
    assert f"ERROR: {gateway.url}/v1/models: {message}." in proc.stderr
    assert "Pass --served-model-name with a name the endpoint serves" in proc.stderr
    assert "Running benchmarks" not in proc.stdout
    assert flow.argv("vllm") == []
    assert flow.argv("lm_eval") == []


# The lookup checks the MODEL run_eval_flow.sh reads from the config, and the
# clients send the one run_benchmarks.sh reads after setting more defaults.
def test_run_stops_when_the_clients_would_send_a_model_the_lookup_did_not_check(
    flow, make_gateway
):
    config = flow.write_config(
        "model-from-a-default",
        _FIXTURE_CONFIG_BODY + 'MODEL="$MODEL-${ENABLE_EP:-no-default}"\n',
    )
    gateway = make_gateway("", [f"{_CONFIG_MODEL}-no-default"])

    proc = flow.run_unchecked(
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway.port),
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
        config=config,
    )

    assert proc.returncode == 1
    assert (
        f"ERROR: model-from-a-default sets MODEL to '{_CONFIG_MODEL}-no-default' "
        f"when run_eval_flow.sh reads it and to '{_CONFIG_MODEL}-false' when "
        f"run_benchmarks.sh does. {gateway.url}/v1/models was checked for the "
        "first name and the clients send the second." in proc.stderr
    )
    assert flow.argv("lm_eval") == []


def test_skip_benchmark_without_host_checks_the_model_on_localhost(flow, make_gateway):
    gateway = make_gateway("", ["another-model"])

    proc = flow.run_unchecked(
        "--skip-benchmark", "--run-lm-eval", "--port", str(gateway.port)
    )

    assert proc.returncode == 1
    assert (
        f"ERROR: http://localhost:{gateway.port}/v1/models: the config's MODEL "
        f"'{_CONFIG_MODEL}' is not a model the endpoint serves" in proc.stderr
    )
    assert flow.argv("lm_eval") == []


def test_skip_benchmark_sends_served_model_name_to_the_server_on_localhost(
    flow, make_gateway
):
    gateway = make_gateway("", ["my-served"])

    flow.run(
        "--skip-benchmark",
        "--run-lm-eval",
        "--port",
        str(gateway.port),
        "--served-model-name",
        "my-served",
    )

    assert flow.option("lm_eval", "--model_args") == {
        "model=my-served,"
        f"base_url=http://localhost:{gateway.port}/v1/chat/completions,"
        "num_concurrent=128"
    }
    assert flow.argv("cleanup_server") == []
    assert flow.argv("vllm") == []


def test_host_and_port_keep_config_model_and_host_port_flags(flow, make_gateway):
    gateway = make_gateway("", [_CONFIG_MODEL])

    flow.run(
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway.port),
        "--eval-suites",
        "mmlu_pro,mmmu_pro",
        "--perf-suites",
        "bench_serve",
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


def test_served_model_name_flag_applies_with_host(flow, make_gateway):
    gateway = make_gateway("", ["my-served"])

    flow.run(
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway.port),
        "--served-model-name",
        "my-served",
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
    )

    assert flow.option("lm_eval", "--model_args") == {
        f"model=my-served,base_url={gateway.url}/v1/chat/completions,num_concurrent=128"
    }


_SERVED_MODEL_NAME_NEEDS_A_RUNNING_SERVER = (
    "ERROR: --served-model-name needs a running server (--base-url, --host and "
    "--port, or --skip-benchmark for the one already on localhost:8000)"
)


@pytest.mark.parametrize(
    ("args", "expected_stream", "message"),
    [
        pytest.param(
            ["--base-url", ""],
            "stdout",
            "ERROR: --base-url cannot be empty",
            id="empty_base_url",
        ),
        pytest.param(
            ["--host", "gw.example.com", "--served-model-name", ""],
            "stdout",
            "ERROR: --served-model-name cannot be empty",
            id="empty_served_model_name",
        ),
        pytest.param(
            ["--host", "gw.example.com", "--eval-suites", ""],
            "stdout",
            "ERROR: --eval-suites cannot be empty",
            id="empty_eval_suites",
        ),
        pytest.param(
            ["--host", "gw.example.com", "--perf-suites", ""],
            "stdout",
            "ERROR: --perf-suites cannot be empty",
            id="empty_perf_suites",
        ),
        pytest.param(
            ["--host", "gw.example.com", "--eval-suites", ","],
            "stdout",
            "ERROR: Empty item in --eval-suites",
            id="comma_only_eval_suites",
        ),
        pytest.param(
            ["--base-url", "http://gw.example.com/v1/chat/completions"],
            "stderr",
            "Invalid --base-url",
            id="route_in_base_url",
        ),
        pytest.param(
            ["--base-url", "http://gw.example.com", "--host", "gw.example.com"],
            "stdout",
            "ERROR: --base-url cannot be combined with --host or --port",
            id="base_url_with_host",
        ),
        pytest.param(
            ["--base-url", "http://gw.example.com", "--port", "9000"],
            "stdout",
            "ERROR: --base-url cannot be combined with --host or --port",
            id="base_url_with_port",
        ),
        pytest.param(
            ["--served-model-name", "my-served"],
            "stdout",
            _SERVED_MODEL_NAME_NEEDS_A_RUNNING_SERVER,
            id="served_model_name_with_self_started_server",
        ),
        pytest.param(
            [
                "--served-model-name",
                "my-served",
                "--skip-benchmark",
                "--perf-suites",
                "bench_serve",
            ],
            "stdout",
            _SERVED_MODEL_NAME_NEEDS_A_RUNNING_SERVER,
            id="served_model_name_with_skip_benchmark_that_still_boots",
        ),
        pytest.param(
            ["--base-url", "--eval-suites", "mmlu_pro"],
            "stdout",
            "ERROR: --base-url requires a value",
            id="flag_in_place_of_a_value",
        ),
        pytest.param(
            ["--base-url", "http://user:s3cret@gw.example.com"],
            "stderr",
            "Invalid --base-url",
            id="credentials_in_base_url",
        ),
        pytest.param(
            ["--eval-suites"],
            "stdout",
            "ERROR: --eval-suites requires a value",
            id="missing_last_value",
        ),
    ],
)
def test_invalid_or_conflicting_flags_fail_without_running_cleanup(
    flow, args, expected_stream, message
):
    proc = flow.run_unchecked(*args)

    assert proc.returncode == 1
    output = proc.stderr if expected_stream == "stderr" else proc.stdout
    assert message in output
    assert "s3cret" not in proc.stdout + proc.stderr
    assert "Running benchmarks" not in proc.stdout
    assert flow.argv("cleanup_server") == []


# --- Suite selection, single-table `all`, and precedence tests ---


def test_eval_suites_run_in_the_order_given(flow):
    flow.run(*flow.host_args, "--eval-suites", "gsm8k, mmlu_pro")

    assert flow.values("lm_eval", "--tasks") == ["gsm8k", "mmlu_pro"]
    assert flow.option("vllm", "--host") == {"127.0.0.1"}


def test_eval_suites_all_covers_text_suites_and_only_adds_mmmu_for_multimodal(flow):
    flow.run(
        *flow.host_args,
        "--eval-suites",
        "all",
        "--perf-suites",
        "none",
        config=_FIXTURE_CONFIG,
    )

    assert flow.values("lm_eval", "--tasks") == [
        "mmlu_llama",
        "mmlu_pro",
        "gsm8k",
        "humaneval_plus_tpu",
        "mbpp_plus_tpu",
    ]
    assert flow.argv("evalscope") == []


def test_eval_suites_all_includes_mmmu_pro_for_multimodal_config(flow):
    flow.run(
        *flow.host_args,
        "--eval-suites",
        "all",
        "--perf-suites",
        "none",
        config=_MULTIMODAL_CONFIG,
    )

    assert flow.values("lm_eval", "--tasks") == [
        "mmlu_llama",
        "mmlu_pro",
        "gsm8k",
        "humaneval_plus_tpu",
        "mbpp_plus_tpu",
    ]
    assert flow.values("evalscope", "--datasets") == ["mmmu_pro"]


def test_eval_suites_all_follows_a_config_that_narrows_lm_eval_tasks(flow):
    config = flow.write_config(
        "mmlu-pro-only",
        _FIXTURE_CONFIG_BODY + 'LM_EVAL_TASKS="mmlu_pro mmlu_llama_custom"\n',
    )

    flow.run(
        *flow.host_args, "--eval-suites", "all", "--perf-suites", "none", config=config
    )

    assert flow.values("lm_eval", "--tasks") == [
        "mmlu_pro",
        "gsm8k",
        "humaneval_plus_tpu",
        "mbpp_plus_tpu",
    ]


def test_run_lm_eval_flag_runs_config_task_list_including_custom_tasks(flow):
    config = flow.write_config(
        "custom-lm-eval",
        _FIXTURE_CONFIG_BODY + 'LM_EVAL_TASKS="mmlu_pro arc_easy"\n',
    )

    flow.run(
        *flow.host_args,
        "--run-lm-eval",
        "--perf-suites",
        "none",
        config=config,
    )

    assert flow.values("lm_eval", "--tasks") == ["mmlu_pro", "arc_easy"]
    assert flow.argv("vllm") == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ["--eval-suites", "mmlu_pro,nope"],
            "ERROR: Unknown --eval-suites item 'nope'",
        ),
        (
            ["--eval-suites", "gpqa_diamond"],
            "ERROR: Unknown --eval-suites item 'gpqa_diamond'",
        ),
        (
            ["--eval-suites", "aime2026"],
            "ERROR: Unknown --eval-suites item 'aime2026'",
        ),
        (
            ["--eval-suites", "bfcl"],
            "ERROR: Unknown --eval-suites item 'bfcl'",
        ),
        (
            ["--eval-suites", "ruler"],
            "ERROR: Unknown --eval-suites item 'ruler'",
        ),
        (
            ["--perf-suites", "nope"],
            "ERROR: Unknown --perf-suites item 'nope' "
            "(expected bench_serve, all, or none)",
        ),
    ],
)
def test_unknown_suite_fails_before_any_client_or_cleanup_runs(flow, args, message):
    proc = flow.run_unchecked(*args)

    assert proc.returncode == 1
    assert message in proc.stdout
    assert "Running benchmarks" not in proc.stdout
    assert flow.argv("cleanup_server") == []


def test_config_suite_defaults_apply_and_env_does_not_leak(flow):
    config = flow.write_config(
        "config-defaults",
        _FIXTURE_CONFIG_BODY + 'EVAL_SUITES="gsm8k"\nPERF_SUITES="none"\n',
    )

    flow.run(
        *flow.host_args,
        config=config,
        env={"EVAL_SUITES": "mmlu_pro", "PERF_SUITES": "bench_serve"},
    )

    assert flow.values("lm_eval", "--tasks") == ["gsm8k"]
    assert flow.argv("vllm") == []


def test_env_suite_vars_are_ignored_without_config_suites(flow):
    flow.run(
        *flow.host_args,
        env={"EVAL_SUITES": "gsm8k", "PERF_SUITES": "none"},
    )

    assert flow.argv("lm_eval") == []
    assert flow.option("vllm", "--host") == {"127.0.0.1"}


def test_skip_benchmark_removes_bench_serve_from_config_perf_suites(flow):
    config = flow.write_config(
        "perf-default-bench",
        _FIXTURE_CONFIG_BODY + 'PERF_SUITES="bench_serve"\n',
    )

    flow.run(*flow.host_args, "--skip-benchmark", config=config)

    assert flow.argv("vllm") == []


def test_explicit_perf_suites_cli_overrides_skip_benchmark_and_config(flow):
    config = flow.write_config(
        "perf-default-none",
        _FIXTURE_CONFIG_BODY + 'PERF_SUITES="none"\n',
    )

    flow.run(
        *flow.host_args,
        "--skip-benchmark",
        "--perf-suites",
        "bench_serve",
        config=config,
    )

    assert flow.option("vllm", "--host") == {"127.0.0.1"}


def test_legacy_eval_flag_overrides_config_eval_suites(flow):
    config = flow.write_config(
        "eval-default-gsm8k",
        _FIXTURE_CONFIG_BODY + 'EVAL_SUITES="gsm8k"\nLM_EVAL_TASKS="mmlu_pro"\n',
    )

    flow.run(
        *flow.host_args,
        "--run-lm-eval",
        "--perf-suites",
        "none",
        config=config,
    )

    assert flow.values("lm_eval", "--tasks") == ["mmlu_pro"]


def test_explicit_eval_suites_cli_overrides_legacy_flag_and_config(flow):
    config = flow.write_config(
        "eval-override-all",
        _FIXTURE_CONFIG_BODY + 'EVAL_SUITES="gsm8k"\nLM_EVAL_TASKS="mmlu_llama"\n',
    )

    flow.run(
        *flow.host_args,
        "--run-lm-eval",
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
        config=config,
    )

    assert flow.values("lm_eval", "--tasks") == ["mmlu_pro"]


def test_config_echoing_stdout_does_not_break_default_reading(flow):
    config = flow.write_config(
        "chatty-config",
        'echo "loading fixture config"\n'
        + _FIXTURE_CONFIG_BODY
        + 'EVAL_SUITES="gsm8k"\n',
    )

    flow.run(*flow.host_args, "--perf-suites", "none", config=config)

    assert flow.values("lm_eval", "--tasks") == ["gsm8k"]


def test_failed_config_source_exits_with_error_and_no_cleanup(flow):
    config = flow.write_config("broken-config", "false\n")

    proc = flow.run_unchecked("--perf-suites", "none", config=config)

    assert proc.returncode == 1
    assert "Failed to read config defaults" in proc.stderr
    assert "Running benchmarks" not in proc.stdout
    assert flow.argv("cleanup_server") == []


# --- Server boot, cleanup, and --skip-bench-cells lifecycle tests ---


def test_skip_benchmark_without_host_does_not_kill_running_server(flow):
    proc = flow.run(
        "--skip-benchmark", "--run-lm-eval", "--port", str(flow.server.port)
    )

    assert "[DRY RUN] Skipping server launch" in proc.stdout
    assert flow.argv("cleanup_server") == []
    assert flow.argv("vllm") == []
    assert flow.option("lm_eval", "--model_args") == {
        f"model={_CONFIG_MODEL},"
        f"base_url=http://localhost:{flow.server.port}/v1/chat/completions,"
        "num_concurrent=128"
    }


# --skip-benchmark decides that no server boots even next to --perf-suites
# none, which on its own boots one (the eval-only local boot test below).
def test_skip_benchmark_with_perf_suites_none_uses_the_running_server(flow):
    proc = flow.run(
        "--skip-benchmark",
        "--perf-suites",
        "none",
        "--eval-suites",
        "mmlu_pro",
        "--port",
        str(flow.server.port),
    )

    assert "[DRY RUN] Skipping server launch" in proc.stdout
    assert flow.argv("cleanup_server") == []
    assert flow.argv("vllm") == []
    assert flow.option("lm_eval", "--model_args") == {
        f"model={_CONFIG_MODEL},"
        f"base_url=http://localhost:{flow.server.port}/v1/chat/completions,"
        "num_concurrent=128"
    }


# The other half of that rule: --perf-suites bench_serve boots a server even
# next to --skip-benchmark. The boot is the one every run_eval_flow.sh call in
# .buildkite/pipeline_perf.yml makes: cleanup, a server kept alive, the bench
# cells against it, then the evals.
@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--run-lm-eval"], id="run_lm_eval"),
        pytest.param(
            ["--skip-benchmark", "--perf-suites", "bench_serve", "--run-lm-eval"],
            id="skip_benchmark_with_perf_suites_bench_serve",
        ),
    ],
)
def test_local_boot_keeps_the_server_alive_for_bench_cells_and_evals(flow, args):
    port = str(flow.server.port)

    proc = flow.run(*args, "--port", port)

    assert "Keeping vLLM server alive in background (--keep-alive)." in proc.stdout
    assert flow.argv("vllm")[:2] == ["serve", _CONFIG_MODEL]
    assert flow.values("vllm", "--port") == [port, port]
    assert flow.option("vllm", "--host") == {"localhost"}
    assert flow.values("vllm", "--result-filename") == [
        str(flow.results_dir / "isl128_osl64_c1.json")
    ]
    assert flow.values("lm_eval", "--tasks") == ["mmlu_llama", "mmlu_pro"]
    assert flow.option("lm_eval", "--model_args") == {
        f"model={_CONFIG_MODEL},"
        f"base_url=http://localhost:{port}/v1/chat/completions,"
        "num_concurrent=128"
    }
    # Initial cleanup before booting + EXIT trap cleanup on finish.
    assert flow.argv("cleanup_server") == ["cleanup", "cleanup"]


def test_no_suites_selected_runs_dry_run_without_server_or_cleanup(flow):
    proc = flow.run("--perf-suites", "none", "--eval-suites", "none")

    assert "[DRY RUN] Skipping server launch" in proc.stdout
    assert flow.argv("cleanup_server") == []
    assert flow.argv("vllm") == []
    assert flow.argv("lm_eval") == []
    assert flow.argv("evalscope") == []


# A run that sends run_benchmarks.sh no request only needs its config.json, so
# it must pass on a spec_bench config, which rejects an external endpoint.
@pytest.mark.parametrize(
    ("server", "args", "expected_tasks"),
    [
        pytest.param(
            "localhost",
            ["--skip-benchmark", "--run-lm-eval"],
            ["mmlu_llama", "mmlu_pro"],
            id="skip_benchmark_run_lm_eval",
        ),
        pytest.param(
            None,
            ["--perf-suites", "none", "--eval-suites", "none"],
            [],
            id="no_suites",
        ),
        pytest.param(
            "host",
            ["--perf-suites", "none", "--eval-suites", "mmlu_pro"],
            ["mmlu_pro"],
            id="host_eval_only",
        ),
    ],
)
def test_spec_bench_config_runs_without_bench_serve(flow, server, args, expected_tasks):
    server_args = {
        None: (),
        "localhost": ("--port", str(flow.server.port)),
        "host": flow.host_args,
    }[server]

    flow.run(*server_args, *args, config=_SPEC_BENCH_CONFIG)

    config = json.loads((flow.results_dir / "config.json").read_text())
    assert config["benchmark_mode"] == "spec_bench"
    assert flow.values("lm_eval", "--tasks") == expected_tasks
    assert flow.argv("vllm") == []
    assert flow.argv("cleanup_server") == []


def test_spec_bench_config_base_url_eval_only_uses_served_model(flow, make_gateway):
    gateway = make_gateway("/gw", ["router-served-name"])

    flow.run(
        "--base-url",
        gateway.url,
        "--served-model-name",
        "router-served-name",
        "--perf-suites",
        "none",
        "--eval-suites",
        "mmlu_pro",
        config=_SPEC_BENCH_CONFIG,
    )

    assert flow.option("lm_eval", "--model_args") == {
        "model=router-served-name,"
        f"base_url={gateway.url}/v1/chat/completions,"
        "num_concurrent=128"
    }
    assert flow.argv("vllm") == []


def test_spec_bench_config_rejects_bench_serve_on_an_external_server(flow):
    proc = flow.run_unchecked(
        *flow.host_args, "--perf-suites", "bench_serve", config=_SPEC_BENCH_CONFIG
    )

    assert proc.returncode == 1
    assert (
        "ERROR: --host and --base-url are not supported in spec_bench mode"
        in proc.stdout
    )
    assert flow.argv("vllm") == []


def test_spec_bench_config_rejects_eval_only_boot_before_cleanup(flow):
    proc = flow.run_unchecked(
        "--perf-suites", "none", "--eval-suites", "mmlu_pro", config=_SPEC_BENCH_CONFIG
    )

    assert proc.returncode == 1
    assert (
        f"ERROR: {_SPEC_BENCH_CONFIG} is a spec_bench config and cannot run eval "
        "suites on a server this script starts." in proc.stdout
    )
    assert "Running benchmarks" not in proc.stdout
    assert flow.argv("cleanup_server") == []
    assert flow.argv("vllm") == []
    assert flow.argv("lm_eval") == []


# --- Perf regression check and dashboard publish gates ---

_PUBLISH_ENV = {
    "UPLOAD_TO_VLLM_CI": "1",
    "SHAPE": "fixture-machine/2x2",
    "TPU_ACCELERATOR_TYPE": "tpu7x",
    "WORKLOAD_IMAGE": "fixture-registry/image:tag",
}


def _run_with_stubbed_check_and_publish(flow, *args: str) -> str:
    """Runs the flow with its upload and perf check on; returns the baseline."""
    for name in ("check_regression", "upload_results", "publish_perf_dashboard"):
        flow.stub_python_script(name)
    baseline = flow.write_perf_baseline()
    flow.run(
        *flow.host_args,
        *args,
        env=_PUBLISH_ENV,
        skip_upload_and_perf_check=False,
    )
    assert flow.argv("upload_results") == [
        "--results-dir",
        str(flow.results_dir),
        "--skip-spanner",
    ]
    return baseline


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--perf-suites", "none"], id="perf_suites_none"),
        pytest.param(["--skip-benchmark"], id="skip_benchmark"),
    ],
)
def test_run_without_bench_serve_skips_perf_check_and_dashboard_publish(flow, args):
    _run_with_stubbed_check_and_publish(flow, *args)

    assert flow.argv("vllm") == []
    assert flow.argv("check_regression") == []
    assert flow.argv("publish_perf_dashboard") == []


def test_perf_suites_bench_serve_beats_skip_benchmark_for_check_and_publish(flow):
    baseline = _run_with_stubbed_check_and_publish(
        flow, "--skip-benchmark", "--perf-suites", "bench_serve"
    )

    assert flow.option("vllm", "--host") == {"127.0.0.1"}
    assert flow.argv("check_regression") == [
        "--mode",
        "perf",
        "--tolerance",
        "0.05",
        "--results-dir",
        str(flow.results_dir),
        "--baseline",
        baseline,
    ]
    assert flow.argv("publish_perf_dashboard") == [
        str(flow.results_dir),
        "--device",
        "tpu7x",
        "--chips",
        "4",
        "--image",
        "fixture-registry/image:tag",
    ]


def test_perf_baselines_are_filed_under_a_chip():
    # run_eval_flow.sh reads baselines/perf/<chip>/ alone, so a baseline filed
    # anywhere else gates nothing.
    perf_baselines = _BENCHMARKING_SRC / "baselines" / "perf"
    assert not list(perf_baselines.glob("*.baseline.json"))
    chips = {"v" + chip.value.removeprefix("v") for chip in pltpu.ChipVersion}
    assert {path.name for path in perf_baselines.iterdir() if path.is_dir()} <= chips


def _run_with_perf_check(
    flow, accelerator_type: str, *args: str
) -> subprocess.CompletedProcess[str]:
    """Runs the flow on an ``accelerator_type`` chip with its perf check on."""
    flow.stub_python_script("check_regression")
    return flow.run_unchecked(
        "--skip-db-upload",
        *args,
        env={"TPU_ACCELERATOR_TYPE": accelerator_type},
        skip_upload_and_perf_check=False,
    )


def test_chip_without_baseline_reports_without_perf_gate(flow):
    flow.write_perf_baseline()

    proc = _run_with_perf_check(flow, "v6e-8", *flow.host_args)

    assert proc.returncode == 0, proc.stderr
    missing = (
        f"scripts/vllm/benchmarking/baselines/perf/v6e/{_FIXTURE_CONFIG}.baseline.json"
    )
    # A ::warning:: line, so the run is not mistaken for a gated one.
    expected = (
        f"::warning::No perf baseline for v6e at {missing}; "
        "reporting without the perf gate."
    )
    assert any(line.startswith(expected) for line in proc.stdout.splitlines())
    assert flow.argv("check_regression") == []


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--skip-perf-check"], id="skip_perf_check"),
        pytest.param(["--perf-suites", "none"], id="perf_suites_none"),
    ],
)
def test_run_without_perf_gate_needs_no_chip(flow, args):
    proc = _run_with_perf_check(flow, "notachip-8", *flow.host_args, *args)

    assert proc.returncode == 0, proc.stderr


def test_unknown_chip_stops_before_cleanup_and_benchmark(flow):
    proc = _run_with_perf_check(flow, "notachip-8")

    assert proc.returncode == 1
    assert "Unrecognized TPU accelerator type 'notachip-8'" in proc.stderr
    assert "ERROR: could not read the TPU chip" in proc.stderr
    assert flow.argv("cleanup_server") == []
    assert flow.argv("vllm") == []


def test_eval_only_local_boot_uses_skip_bench_cells_and_runs_cleanup(flow):
    proc = flow.run("--perf-suites", "none", "--eval-suites", "mmlu_pro")

    assert "Skipping vllm bench serve cells (--skip-bench-cells)." in proc.stdout
    vllm_argv = flow.argv("vllm")
    assert vllm_argv[:2] == ["serve", _CONFIG_MODEL]
    assert "bench" not in vllm_argv
    assert flow.values("lm_eval", "--tasks") == ["mmlu_pro"]
    # Initial cleanup before booting + EXIT trap cleanup on finish.
    assert flow.argv("cleanup_server") == ["cleanup", "cleanup"]


def test_gsm8k_uses_plain_chat_gen_kwargs_without_continue_final_message(flow):
    flow.run(
        *flow.host_args,
        "--eval-suites",
        "gsm8k,mmlu_llama",
        "--perf-suites",
        "none",
    )

    gen_kwargs = flow.values("lm_eval", "--gen_kwargs")
    assert gen_kwargs == [
        '{"chat_template_kwargs": {"enable_thinking": false}}',
        '{"continue_final_message": true, "add_generation_prompt": false, '
        '"chat_template_kwargs": {"enable_thinking": false}}',
    ]
    assert flow.values("lm_eval", "--fewshot_as_multiturn") == ["true"]


# --- json_schema suite ---

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _write_answer_task_config(tmp_path: pathlib.Path, prompts: list[str]) -> str:
    """Writes a one-case task config; a valid answer is ``{"answer": <int>}``."""
    case = {
        "name": "answer",
        "max_tokens": 64,
        "schema": _ANSWER_SCHEMA,
        "prompts": prompts,
    }
    path = tmp_path / "answer_task.yaml"
    path.write_text(json.dumps({"cases": [case]}))
    return str(path)


def _run_json_schema(flow, gateway, **kwargs) -> subprocess.CompletedProcess[str]:
    return flow.run_unchecked(
        "--base-url",
        gateway.url,
        "--eval-suites",
        "json_schema",
        "--perf-suites",
        "none",
        **kwargs,
    )


def _json_schema_results(flow) -> dict[str, Any]:
    return json.loads((flow.results_dir / "results_json_schema.json").read_text())


def test_json_schema_suite_passes_when_every_response_is_valid(flow, make_gateway):
    gateway = make_gateway("/gw", ["router-served-name"])

    proc = _run_json_schema(flow, gateway)

    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert _json_schema_results(flow) == {
        "results": {"json_schema": {"acc,none": 1.0, "acc_stderr,none": 0.0}},
        "meta": {
            "model": "router-served-name",
            "total_trials": 28,
            "successful_trials": 28,
            "failed_trials": [],
        },
    }
    assert (
        "| `json_schema` | acc,none | 1.0000 | 1.0000 | 0.0000 | 0.0000 | +0.0000 "
        "| PASS |" in (flow.results_dir / "eval_check_json_schema.md").read_text()
    )
    assert {request["model"] for request in gateway.chat_requests} == {
        "router-served-name"
    }


def test_json_schema_suite_fails_and_reports_each_invalid_response(
    flow, make_gateway, tmp_path
):
    replies = {
        "valid": _chat_completion('{"answer": 42}'),
        "wrong type": _chat_completion('{"answer": "forty-two"}'),
        "truncated": _chat_completion('{"answer": 42}', finish_reason="length"),
        "not json": _chat_completion("The answer is 42."),
        "refused": (400, {"error": "schema not supported"}),
    }
    task_config = _write_answer_task_config(tmp_path, list(replies))
    config = flow.write_config(
        "answer-task",
        _FIXTURE_CONFIG_BODY + f'JSON_SCHEMA_TASK_CONFIG="{task_config}"\n',
    )
    gateway = make_gateway(
        "",
        [_CONFIG_MODEL],
        chat_reply=lambda payload: replies[payload["messages"][0]["content"]],
    )

    proc = _run_json_schema(flow, gateway, config=config)

    assert proc.returncode == 1
    assert _json_schema_results(flow) == {
        "results": {"json_schema": {"acc,none": 0.2, "acc_stderr,none": 0.0}},
        "meta": {
            "model": _CONFIG_MODEL,
            "total_trials": 5,
            "successful_trials": 1,
            "failed_trials": [
                {
                    "case_name": "answer",
                    "sample_id": "1",
                    "finish_reason": "stop",
                    "error": "Property 'answer': Expected JSON type integer, got str",
                },
                {
                    "case_name": "answer",
                    "sample_id": "2",
                    "finish_reason": "length",
                    "error": "Completion hit max_tokens (finish_reason=length)",
                },
                {
                    "case_name": "answer",
                    "sample_id": "3",
                    "finish_reason": "stop",
                    "error": "JSONDecodeError: Expecting value: line 1 column 1 "
                    "(char 0)",
                },
                {
                    "case_name": "answer",
                    "sample_id": "4",
                    "finish_reason": None,
                    "error": 'HTTP 400: {"error": "schema not supported"}',
                },
            ],
        },
    }
    assert "| `json_schema` | acc,none | 0.2000 | 1.0000 |" in proc.stdout
    assert "::error::Eval regression check failed for json_schema!" in proc.stdout


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        pytest.param(
            (503, {"error": "overloaded"}),
            "ERROR: HTTP 503 for case 'flat_json' sample '0': "
            '{"error": "overloaded"}',
            id="http_503",
        ),
        pytest.param(
            (200, {"object": "error"}),
            "ERROR: Request for case 'flat_json' sample '0' failed: "
            "KeyError('choices')",
            id="not_a_chat_completion",
        ),
    ],
)
def test_json_schema_suite_has_no_result_when_the_endpoint_cannot_be_evaluated(
    flow, make_gateway, reply, error
):
    gateway = make_gateway("", [_CONFIG_MODEL], chat_reply=lambda payload: reply)

    proc = _run_json_schema(flow, gateway)

    assert proc.returncode == 1
    assert error in proc.stderr
    assert "::error::eval_json_schema.py could not evaluate json_schema." in proc.stdout
    assert not (flow.results_dir / "results_json_schema.json").exists()
    assert not (flow.results_dir / "eval_check_json_schema.md").exists()


def test_openai_api_key_reaches_the_json_schema_client(flow, make_gateway):
    gateway = make_gateway("", [_CONFIG_MODEL], required_api_key="gw-key-123")

    proc = _run_json_schema(flow, gateway, env={"OPENAI_API_KEY": "gw-key-123"})

    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    meta = _json_schema_results(flow)["meta"]
    assert (meta["total_trials"], meta["successful_trials"]) == (28, 28)


def test_config_names_the_json_schema_task_config(flow, make_gateway, tmp_path):
    task_config = _write_answer_task_config(tmp_path, ["What is 6 times 7?"])
    config = flow.write_config(
        "answer-task",
        _FIXTURE_CONFIG_BODY + f'JSON_SCHEMA_TASK_CONFIG="{task_config}"\n',
    )
    gateway = make_gateway(
        "",
        [_CONFIG_MODEL],
        chat_reply=lambda payload: _chat_completion('{"answer": 42}'),
    )

    proc = _run_json_schema(flow, gateway, config=config)

    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert gateway.chat_requests == [
        {
            "model": _CONFIG_MODEL,
            "messages": [{"role": "user", "content": "What is 6 times 7?"}],
            "temperature": 0.0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": _ANSWER_SCHEMA,
                },
            },
        }
    ]


def test_json_schema_task_config_in_the_environment_is_ignored(
    flow, make_gateway, tmp_path
):
    task_config = _write_answer_task_config(tmp_path, ["What is 6 times 7?"])
    gateway = make_gateway("", [_CONFIG_MODEL])

    proc = _run_json_schema(flow, gateway, env={"JSON_SCHEMA_TASK_CONFIG": task_config})

    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert _json_schema_results(flow)["meta"]["total_trials"] == 28


def test_missing_json_schema_task_config_fails_before_cleanup(flow):
    config = flow.write_config(
        "missing-task",
        _FIXTURE_CONFIG_BODY + 'JSON_SCHEMA_TASK_CONFIG="/nonexistent/task.yaml"\n',
    )

    # No endpoint flag: this run would boot a server for the suite.
    proc = flow.run_unchecked(
        "--eval-suites", "mmlu_pro,json_schema", "--perf-suites", "none", config=config
    )

    assert proc.returncode == 1
    assert (
        "ERROR: JSON_SCHEMA_TASK_CONFIG of missing-task is not a file: "
        "/nonexistent/task.yaml" in proc.stdout
    )
    assert "Running benchmarks" not in proc.stdout
    assert flow.argv("cleanup_server") == []
    assert flow.argv("vllm") == []
    assert flow.argv("lm_eval") == []


def test_eval_suites_all_runs_json_schema_after_the_other_suites(flow):
    proc = flow.run(*flow.host_args, "--eval-suites", "all", "--perf-suites", "none")

    assert _json_schema_results(flow)["meta"]["total_trials"] == 28
    assert len(flow.server.chat_requests) == 28
    assert proc.stdout.index("Running lm_eval for task: mbpp_plus_tpu") < (
        proc.stdout.index("Running eval_json_schema.py for task: json_schema")
    )


def test_shipped_flat_json_config_gates_its_own_task_config(flow, make_gateway):
    name = "qwen3-0.6b-structured-flat-json"
    config = flow.write_config(
        name, (_BENCHMARKING_SRC / "configs" / f"{name}.sh").read_text()
    )
    gateway = make_gateway("", ["Qwen/Qwen3-0.6B"])

    # No --eval-suites: the config's EVAL_SUITES selects json_schema.
    flow.run("--base-url", gateway.url, "--perf-suites", "none", config=config)

    meta = _json_schema_results(flow)["meta"]
    assert (meta["total_trials"], meta["successful_trials"]) == (24, 24)
    assert {
        request["response_format"]["json_schema"]["name"]
        for request in gateway.chat_requests
    } == {"flat_json", "character_schema", "tool_call_arguments"}

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
``evalscope``, ``vllm``, ``curl``), and a stubbed ``cleanup_server.sh``. Nothing
writes to the repository tree or kills host processes.
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import pathlib
import shutil
import stat
import subprocess
import threading
from typing import Any

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BENCHMARKING_SRC = _REPO_ROOT / "scripts" / "vllm" / "benchmarking"
_COPIED_FILES = (
    "endpoint_url.py",
    "run_eval_flow.sh",
    "run_benchmarks.sh",
    "upload_results.py",
    "bq_utils.py",
    "check_regression.py",
    "publish_perf_dashboard.py",
)

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
    sleep 30
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


class _Gateway:
    """Lists ``model_ids`` at ``<prefix>/v1/models`` with optional bearer auth.

    ``models_reply`` replaces the model list with a raw ``(status, body)``.
    """

    def __init__(
        self,
        prefix: str,
        model_ids: list[str],
        required_api_key: str | None = None,
        models_reply: tuple[int, bytes] | None = None,
    ):
        self.prefix = prefix
        self.model_ids = model_ids
        self.required_api_key = required_api_key
        self.models_reply = models_reply
        self.seen_auth_headers: list[str | None] = []
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

            def do_GET(self) -> None:
                auth = self.headers.get("Authorization")
                gateway.seen_auth_headers.append(auth)
                if (
                    gateway.required_api_key is not None
                    and auth != f"Bearer {gateway.required_api_key}"
                ):
                    self._reply(401, {"error": "unauthorized"})
                    return
                if self.path != f"{gateway.prefix}/v1/models":
                    self._reply(404, {})
                elif gateway.models_reply is not None:
                    self._reply_raw(*gateway.models_reply)
                else:
                    ids = [{"id": model_id} for model_id in gateway.model_ids]
                    self._reply(200, {"data": ids})

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

    def _make(
        prefix: str,
        model_ids: list[str],
        required_api_key: str | None = None,
        models_reply: tuple[int, bytes] | None = None,
    ) -> _Gateway:
        gateways.append(
            _Gateway(
                prefix,
                model_ids,
                required_api_key=required_api_key,
                models_reply=models_reply,
            )
        )
        return gateways[-1]

    yield _make
    for gateway in gateways:
        gateway.close()


class _Flow:
    """Runs scripts in a hermetic copy of ``scripts/vllm/benchmarking``."""

    def __init__(self, tmp_path: pathlib.Path):
        self._tmp_path = tmp_path
        self._repo = tmp_path / "repo"
        self._bench_dir = self._repo / "scripts" / "vllm" / "benchmarking"
        self._configs_dir = self._bench_dir / "configs"
        self._configs_dir.mkdir(parents=True)
        (self._bench_dir / "baselines" / "eval").mkdir(parents=True)
        (self._bench_dir / "baselines" / "perf").mkdir(parents=True)
        shutil.copytree(
            _BENCHMARKING_SRC / "lm_eval_tasks",
            self._bench_dir / "lm_eval_tasks",
        )
        for name in _COPIED_FILES:
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
        """Writes an empty perf baseline; returns the path the flow passes on."""
        relative = f"scripts/vllm/benchmarking/baselines/perf/{config}.baseline.json"
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
def flow(tmp_path):
    return _Flow(tmp_path)


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


def test_discover_served_model_sends_bearer_token(make_gateway):
    gateway = make_gateway("/gw", ["authed-model"], required_api_key="secret-token")

    with pytest.raises(endpoint_url.ModelLookupError, match="401"):
        endpoint_url.discover_served_model(gateway.url)
    assert (
        endpoint_url.discover_served_model(gateway.url, api_key="secret-token")
        == "authed-model"
    )
    assert gateway.seen_auth_headers == [None, "Bearer secret-token"]


@pytest.mark.parametrize(
    ("model_ids", "expected"),
    [
        ([], None),
        (["only-model"], "only-model"),
        (["model-a", "model-b"], None),
    ],
)
def test_discover_served_model_returns_the_id_only_when_one_is_listed(
    make_gateway, model_ids, expected
):
    gateway = make_gateway("", model_ids)

    assert endpoint_url.discover_served_model(gateway.url) == expected


def test_discover_served_model_skips_entries_without_a_string_id(make_gateway):
    body = json.dumps({"data": [{"id": 7}, "text", {"id": "only-model"}]}).encode()
    gateway = make_gateway("", [], models_reply=(200, body))

    assert endpoint_url.discover_served_model(gateway.url) == "only-model"


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
def test_discover_served_model_raises_when_the_lookup_fails(
    make_gateway, models_reply, message
):
    gateway = make_gateway("", ["unused"], models_reply=models_reply)

    with pytest.raises(endpoint_url.ModelLookupError, match=message):
        endpoint_url.discover_served_model(gateway.url)


def test_discover_served_model_raises_when_nothing_is_listening(make_gateway):
    gateway = make_gateway("", ["unused"])
    gateway.close()

    with pytest.raises(endpoint_url.ModelLookupError, match="Connection refused"):
        endpoint_url.discover_served_model(gateway.url)


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
            ["--base-url", "http://user:pw@gw.example.com"],
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
    gateway = make_gateway("/gw", ["authed-served-name"], required_api_key="gw-key-123")

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
    assert flow.option("evalscope", "--model") == {"authed-served-name"}
    assert flow.option("evalscope", "--api-key") == {"gw-key-123"}


def test_https_base_url_explicit_served_model_and_no_env_fallback(flow):
    # No server is listening: --served-model-name skips /v1/models, and
    # SERVED_MODEL_NAME in the environment is ignored.
    flow.run(
        "--base-url",
        "https://gw.example.com/prefix/v1/",
        "--served-model-name",
        "my-served",
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
        env={"SERVED_MODEL_NAME": "from-env"},
    )

    assert flow.option("lm_eval", "--model_args") == {
        "model=my-served,"
        "base_url=https://gw.example.com/prefix/v1/chat/completions,"
        "num_concurrent=128"
    }


def test_served_model_name_env_is_ignored_without_cli_flag(flow, make_gateway):
    gateway = make_gateway("", ["router-served-name"])

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
        f"model=router-served-name,"
        f"base_url={gateway.url}/v1/chat/completions,"
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
        "mmlu_pro",
        "--perf-suites",
        "none",
    )

    assert flow.option("lm_eval", "--model_args") == {
        f"model=my-served,base_url={gateway.url}/v1/chat/completions,num_concurrent=128"
    }


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
    assert "Pass --served-model-name" in proc.stderr
    assert "Running benchmarks" not in proc.stdout
    assert flow.argv("lm_eval") == []


def test_base_url_keeps_config_model_when_endpoint_lists_none(flow, make_gateway):
    gateway = make_gateway("", [])

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
        f"base_url={gateway.url}/v1/chat/completions,"
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


def test_served_model_name_flag_applies_with_host(flow):
    flow.run(
        "--host",
        "gw.example.com",
        "--served-model-name",
        "my-served",
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
    )

    assert flow.option("lm_eval", "--model_args") == {
        "model=my-served,"
        "base_url=http://gw.example.com:8000/v1/chat/completions,"
        "num_concurrent=128"
    }


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
            "ERROR: --served-model-name needs a running server",
            id="served_model_name_with_self_started_server",
        ),
        pytest.param(
            ["--base-url", "--eval-suites", "mmlu_pro"],
            "stdout",
            "ERROR: --base-url requires a value",
            id="flag_in_place_of_a_value",
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
    assert "Running benchmarks" not in proc.stdout
    assert flow.argv("cleanup_server") == []


# --- Suite selection, single-table `all`, and precedence tests ---


def test_eval_suites_run_in_the_order_given(flow):
    flow.run("--host", "gw.example.com", "--eval-suites", "gsm8k, mmlu_pro")

    assert flow.values("lm_eval", "--tasks") == ["gsm8k", "mmlu_pro"]
    assert flow.option("vllm", "--host") == {"gw.example.com"}


def test_eval_suites_all_covers_text_suites_and_only_adds_mmmu_for_multimodal(flow):
    flow.run(
        "--host",
        "gw.example.com",
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
        "--host",
        "gw.example.com",
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


def test_run_lm_eval_flag_runs_config_task_list_including_custom_tasks(flow):
    config = flow.write_config(
        "custom-lm-eval",
        _FIXTURE_CONFIG_BODY + 'LM_EVAL_TASKS="mmlu_pro arc_easy"\n',
    )

    flow.run(
        "--host",
        "gw.example.com",
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
        "--host",
        "gw.example.com",
        config=config,
        env={"EVAL_SUITES": "mmlu_pro", "PERF_SUITES": "bench_serve"},
    )

    assert flow.values("lm_eval", "--tasks") == ["gsm8k"]
    assert flow.argv("vllm") == []


def test_env_suite_vars_are_ignored_without_config_suites(flow):
    flow.run(
        "--host",
        "gw.example.com",
        env={"EVAL_SUITES": "gsm8k", "PERF_SUITES": "none"},
    )

    assert flow.argv("lm_eval") == []
    assert flow.option("vllm", "--host") == {"gw.example.com"}


def test_skip_benchmark_removes_bench_serve_from_config_perf_suites(flow):
    config = flow.write_config(
        "perf-default-bench",
        _FIXTURE_CONFIG_BODY + 'PERF_SUITES="bench_serve"\n',
    )

    flow.run("--host", "gw.example.com", "--skip-benchmark", config=config)

    assert flow.argv("vllm") == []


def test_explicit_perf_suites_cli_overrides_skip_benchmark_and_config(flow):
    config = flow.write_config(
        "perf-default-none",
        _FIXTURE_CONFIG_BODY + 'PERF_SUITES="none"\n',
    )

    flow.run(
        "--host",
        "gw.example.com",
        "--skip-benchmark",
        "--perf-suites",
        "bench_serve",
        config=config,
    )

    assert flow.option("vllm", "--host") == {"gw.example.com"}


def test_legacy_eval_flag_overrides_config_eval_suites(flow):
    config = flow.write_config(
        "eval-default-gsm8k",
        _FIXTURE_CONFIG_BODY + 'EVAL_SUITES="gsm8k"\nLM_EVAL_TASKS="mmlu_pro"\n',
    )

    flow.run(
        "--host",
        "gw.example.com",
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
        "--host",
        "gw.example.com",
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

    flow.run("--host", "gw.example.com", "--perf-suites", "none", config=config)

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
    proc = flow.run("--skip-benchmark", "--run-lm-eval")

    assert "[DRY RUN] Skipping server launch" in proc.stdout
    assert flow.argv("cleanup_server") == []
    assert flow.argv("vllm") == []
    assert flow.option("lm_eval", "--model_args") == {
        f"model={_CONFIG_MODEL},"
        "base_url=http://localhost:8000/v1/chat/completions,"
        "num_concurrent=128"
    }


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
    ("args", "expected_tasks"),
    [
        pytest.param(
            ["--skip-benchmark", "--run-lm-eval"],
            ["mmlu_llama", "mmlu_pro"],
            id="skip_benchmark_run_lm_eval",
        ),
        pytest.param(
            ["--perf-suites", "none", "--eval-suites", "none"],
            [],
            id="no_suites",
        ),
        pytest.param(
            ["--host", "gw.example.com", "--perf-suites", "none"]
            + ["--eval-suites", "mmlu_pro"],
            ["mmlu_pro"],
            id="host_eval_only",
        ),
    ],
)
def test_spec_bench_config_runs_without_bench_serve(flow, args, expected_tasks):
    flow.run(*args, config=_SPEC_BENCH_CONFIG)

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


@pytest.mark.parametrize(
    ("args", "message"),
    [
        pytest.param(
            ["--host", "gw.example.com", "--perf-suites", "bench_serve"],
            "ERROR: --host and --base-url are not supported in spec_bench mode",
            id="bench_serve_on_external_server",
        ),
        pytest.param(
            ["--perf-suites", "none", "--eval-suites", "mmlu_pro"],
            "ERROR: --skip-bench-cells is not supported in spec_bench mode",
            id="eval_only_on_self_started_server",
        ),
    ],
)
def test_spec_bench_config_rejects_runs_it_cannot_serve(flow, args, message):
    proc = flow.run_unchecked(*args, config=_SPEC_BENCH_CONFIG)

    assert proc.returncode == 1
    assert message in proc.stdout
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
        "--host",
        "gw.example.com",
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

    assert flow.option("vllm", "--host") == {"gw.example.com"}
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
        "--host",
        "gw.example.com",
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

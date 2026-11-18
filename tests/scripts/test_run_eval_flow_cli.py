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
"""Tests for run_eval_flow.sh's flags: suites, endpoint, and served model name.

Every client (`lm_eval`, `evalscope`, `vllm bench serve`) is a stub on PATH
that records its argv, so the tests read the exact URL and model name each one
was given.

No test here lets the script start a server itself: that path runs
cleanup_server.sh, which kills processes and removes lock files on the host.
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

_SAFE_RM_STUB = """#!/bin/bash
for arg in "$@"; do
    case "$arg" in
        /tmp/libtpu_lockfile*) exit 0 ;;
    esac
done
exec /bin/rm "$@"
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
    """Lists `model_ids` at `<prefix>/v1/models` and nothing anywhere else."""

    def __init__(self, prefix: str, model_ids: list[str]):
        self.prefix = prefix
        self.model_ids = model_ids
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
            ("pkill", _RECORD_ARGV),
            ("lsof", _RECORD_ARGV),
            ("fuser", _RECORD_ARGV),
            ("sleep", _RECORD_ARGV),
            ("rm", _SAFE_RM_STUB),
        ):
            stub = self._bin / name
            stub.write_text(body)
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
        # run_benchmarks.sh only checks that the vllm package can be found.
        self._site = tmp_path / "site"
        (self._site / "vllm").mkdir(parents=True)
        (self._site / "vllm" / "__init__.py").write_text("")

    def run(
        self, *args: str, env: dict[str, str] | None = None, config: str = _CONFIG
    ) -> subprocess.CompletedProcess:
        proc = self.run_unchecked(*args, env=env, config=config)
        assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        return proc

    def run_unchecked(
        self, *args: str, env: dict[str, str] | None = None, config: str = _CONFIG
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "bash",
                str(_RUN_EVAL_FLOW),
                "--config",
                config,
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
                **(env or {}),
            },
            capture_output=True,
            text=True,
        )

    def argv(self, client: str) -> list[str]:
        """The arguments of every call to `client`, or [] if it never ran."""
        log = self._stub_logs / f"{client}.args"
        return log.read_text().splitlines() if log.exists() else []

    def values(self, client: str, flag: str) -> list[str]:
        """Each value `client` was given for `flag`, in call order."""
        argv = self.argv(client)
        return [argv[i + 1] for i, arg in enumerate(argv) if arg == flag]

    def option(self, client: str, flag: str) -> set[str]:
        """Every distinct value `client` was given for `flag`, over all calls."""
        return set(self.values(client, flag))


@pytest.fixture
def flow(tmp_path):
    return _Flow(tmp_path)


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
    assert flow.option("evalscope", "--model") == {"router-served-name"}
    assert flow.option("vllm", "--base-url") == {f"http://127.0.0.1:{gateway.port}/gw"}
    assert flow.option("vllm", "--served-model-name") == {"router-served-name"}
    # The tokenizer still comes from the config's model.
    assert flow.option("vllm", "--model") == {_CONFIG_MODEL}
    assert flow.option("vllm", "--host") == set()
    assert flow.option("vllm", "--port") == set()


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
    ("base_url", "expected"),
    [
        ("gw.example.com", "http://gw.example.com:8000"),
        ("gw.example.com/prefix", "http://gw.example.com:8000/prefix"),
        ("gw.example.com:9000/v1", "http://gw.example.com:9000"),
        ("http://gw.example.com", "http://gw.example.com"),
        ("https://gw.example.com", "https://gw.example.com"),
        ("https://gw.example.com:8443/prefix/", "https://gw.example.com:8443/prefix"),
        ("http://[::1]:8000", "http://[::1]:8000"),
    ],
)
def test_base_url_forms(flow, base_url, expected):
    flow.run(
        "--base-url",
        base_url,
        "--served-model-name",
        "my-served",
        "--eval-suites",
        "mmlu_pro",
        "--perf-suites",
        "none",
    )

    assert flow.option("lm_eval", "--model_args") == {
        f"model=my-served,base_url={expected}/v1/chat/completions,num_concurrent=128"
    }


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://gw.example.com",
        "http://",
        "http://gw.example.com/v1?key=1",
        "http://gw.example.com:notaport",
        "http://user:secret@gw.example.com",
    ],
)
def test_invalid_base_url_fails_before_any_client_runs(flow, base_url):
    proc = flow.run_unchecked(
        "--base-url", base_url, "--eval-suites", "mmlu_pro", "--perf-suites", "none"
    )

    assert proc.returncode == 1
    assert f"Invalid --base-url: {base_url}" in proc.stderr
    assert "Running benchmarks" not in proc.stdout


@pytest.mark.parametrize(
    ("args", "message"),
    [
        pytest.param(
            ["--base-url", "http://gw.example.com", "--host", "gw.example.com"],
            "ERROR: --base-url cannot be combined with --host or --port",
            id="base_url_with_host",
        ),
        pytest.param(
            ["--base-url", "http://gw.example.com", "--port", "9000"],
            "ERROR: --base-url cannot be combined with --host or --port",
            id="base_url_with_port",
        ),
        pytest.param(
            ["--served-model-name", "my-served"],
            "ERROR: --served-model-name needs a running server "
            "(--base-url, or --host and --port)",
            id="served_model_name_with_self_started_server",
        ),
    ],
)
def test_conflicting_flags_fail_before_any_client_runs(flow, args, message):
    proc = flow.run_unchecked(
        *args, "--eval-suites", "mmlu_pro", "--perf-suites", "none"
    )

    assert proc.returncode == 1
    assert message in proc.stdout
    assert "Running benchmarks" not in proc.stdout


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


def test_eval_suites_run_in_the_order_given(flow):
    flow.run("--host", "gw.example.com", "--eval-suites", "gsm8k, mmlu_pro")

    assert flow.values("lm_eval", "--tasks") == ["gsm8k", "mmlu_pro"]
    # No --perf-suites: the benchmark still runs, as it did before the flag.
    assert flow.option("vllm", "--host") == {"gw.example.com"}


def test_run_lm_eval_flag_still_runs_the_config_task_list(flow):
    flow.run("--host", "gw.example.com", "--run-lm-eval", "--perf-suites", "none")

    assert flow.values("lm_eval", "--tasks") == ["mmlu_llama", "mmlu_pro"]
    assert flow.argv("vllm") == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ["--eval-suites", "mmlu_pro,nope"],
            "ERROR: Unknown --eval-suites item 'nope'",
        ),
        (
            ["--perf-suites", "nope"],
            "ERROR: Unknown --perf-suites item 'nope' "
            "(expected bench_serve, agentx, all, or none)",
        ),
    ],
)
def test_unknown_suite_fails_before_any_client_runs(flow, args, message):
    proc = flow.run_unchecked("--host", "gw.example.com", *args)

    assert proc.returncode == 1
    assert message in proc.stdout
    assert "Running benchmarks" not in proc.stdout


def test_agentx_suite_requires_physical_chip_count(flow):
    # A 1P1D config, loaded client-only: sourcing it must not need a role.
    proc = flow.run_unchecked(
        "--base-url",
        "http://gw.example.com",
        "--perf-suites",
        "agentx",
        "--eval-suites",
        "none",
        config="kimi-k3-tp8-dp4-ep-agentx",
    )

    assert proc.returncode == 1
    assert "agentx suite requires AGENTX_CHIPS=<physical_chips>" in proc.stdout
    assert "Running benchmarks" not in proc.stdout


@pytest.fixture
def config_with_custom_lm_eval_task():
    """A config whose LM_EVAL_TASKS names a task --eval-suites does not list."""
    configs_dir = _RUN_EVAL_FLOW.parent / "configs"
    name = "tmp-test-run-eval-flow-cli-custom-task"
    config = configs_dir / f"{name}.sh"
    config.write_text(
        (configs_dir / f"{_CONFIG}.sh").read_text()
        + '\nLM_EVAL_TASKS="mmlu_pro arc_easy"\n'
    )
    try:
        yield name
    finally:
        config.unlink()


def test_run_lm_eval_runs_config_tasks_the_suite_list_does_not_name(
    flow, config_with_custom_lm_eval_task
):
    flow.run(
        "--host",
        "gw.example.com",
        "--run-lm-eval",
        "--perf-suites",
        "none",
        config=config_with_custom_lm_eval_task,
    )

    assert flow.values("lm_eval", "--tasks") == ["mmlu_pro", "arc_easy"]


def test_skip_benchmark_without_host_keeps_legacy_dry_run_behavior(flow):
    proc = flow.run("--skip-benchmark", "--run-lm-eval")

    assert "[DRY RUN] Skipping server launch" in proc.stdout
    assert flow.argv("vllm") == []
    assert flow.option("lm_eval", "--model_args") == {
        f"model={_CONFIG_MODEL},"
        "base_url=http://localhost:8000/v1/chat/completions,"
        "num_concurrent=128"
    }


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
    # Only mmlu_llama passed --fewshot_as_multiturn true (once across the 2 calls).
    assert flow.values("lm_eval", "--fewshot_as_multiturn") == ["true"]


def test_failed_config_source_exits_with_error(flow):
    configs_dir = _RUN_EVAL_FLOW.parent / "configs"
    name = "tmp-test-run-eval-flow-cli-broken-config"
    config = configs_dir / f"{name}.sh"
    config.write_text("false\n")
    try:
        proc = flow.run_unchecked(
            "--host", "gw.example.com", "--perf-suites", "none", config=name
        )

        assert proc.returncode == 1
        assert "Failed to read config defaults" in proc.stderr
        assert "Running benchmarks" not in proc.stdout
    finally:
        config.unlink()

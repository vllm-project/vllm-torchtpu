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
"""Tests for `experimental/rust/bin/rvllm.bin` and its argument split.

The split uses vLLM's real `vllm serve` parser. The script runs with a fake
rvllm binary and a fake export step, and records what each one received.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "src/vllm_torchtpu/experimental/rust/bin/rvllm.bin"

try:
    from vllm_torchtpu.experimental.export import cli as export_cli
    from vllm_torchtpu.experimental.rust import launcher_args

    PARSER = export_cli.build_arg_parser()
    if "--max-model-len" not in PARSER._option_string_actions:
        raise RuntimeError("the vllm serve parser is not available")
except Exception as e:  # noqa: BLE001
    pytest.skip(f"vLLM's serve parser is unavailable: {e}", allow_module_level=True)


def split(*argv):
    return launcher_args.split_args(list(argv), PARSER)


def test_engine_arguments_go_to_the_exporter_and_deployment_flags_to_rvllm():
    s = split(
        "Qwen/Qwen3-0.6B",
        "--max_model_len",
        "2048",
        "-tp=4",
        "--enforce-eager",
        "--speculative-config",
        '{"method": "ngram"}',
        "--port",
        "8001",
        "--served-model-name",
        "qwen",
        "--no-async-scheduling",
        "--load-format",
        "dummy",
        "--role=decode",
        "--kv-producers",
        "tcp:prefill:7100",
    )
    assert s.model == "Qwen/Qwen3-0.6B"
    assert s.export_args == [
        "--max_model_len",
        "2048",
        "-tp=4",
        "--enforce-eager",
        "--speculative-config",
        '{"method": "ngram"}',
        "--load-format",
        "dummy",
    ]
    assert s.serve_args == [
        "--port",
        "8001",
        "--served-model-name",
        "qwen",
        "--no-async-scheduling",
        "--load-format",
        "dummy",
        "--role=decode",
        "--kv-producers",
        "tcp:prefill:7100",
    ]
    assert s.warnings == []


def test_the_model_can_be_given_with_model():
    assert split("--model=m", "--max-model-len", "8").model == "m"
    assert split("--model", "m").model == "m"


@pytest.mark.parametrize(
    "argv, message",
    [
        (["m", "--api-key", "k"], "--api-key is a `vllm serve` frontend option"),
        (["m", "--chat-template", "t.jinja"], "--chat-template is a `vllm serve`"),
        (["m", "--enable-auto-tool-choice"], "frontend option that rvllm does not"),
        (["m", "--kv-transfer-config", "{}"], "--role prefill|decode"),
        (["m", "--no-such-option"], "neither a `vllm serve` option nor"),
        (["m", "--output-dir", "/tmp/x"], "RVLLM_ARTIFACT_DIR"),
        (["m", "--served-model-name", "a", "b"], "one --served-model-name"),
        (["m", "extra"], "unexpected argument 'extra'"),
        (["m", "--role"], "--role needs a value"),
    ],
)
def test_options_rvllm_cannot_honor_fail(argv, message):
    with pytest.raises(launcher_args.LauncherArgsError, match=None) as e:
        split(*argv)
    assert message in str(e.value)


def test_log_only_frontend_options_warn():
    s = split("m", "--uvicorn-log-level", "debug", "--disable-fastapi-docs")
    assert s.export_args == [] and s.serve_args == []
    assert len(s.warnings) == 2
    assert all("only configures vLLM's own API server" in w for w in s.warnings)


FAKE_RVLLM = """#!/bin/bash
printf '%s\\n' "$@" > "$FAKE_LOG_DIR/rvllm.args"
"""

FAKE_EXPORT = """
import json, os, sys
with open(os.path.join(os.environ["FAKE_LOG_DIR"], "export.calls"), "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
out = sys.argv[sys.argv.index("--output-dir") + 1]
os.makedirs(out)
open(os.path.join(out, "metadata.json"), "w").write("{}")
"""

# Runs the real interpreter, except for the export step and `vllm bench`.
FAKE_PYTHON = """#!/bin/bash
if [ "$1" = -m ] && [ "$2" = vllm_torchtpu.experimental.export ]; then
    shift 2
    exec "$REAL_PYTHON" "$FAKE_EXPORT" "$@"
fi
if [ "$1" = -m ] && [ "$2" = vllm.entrypoints.cli.main ]; then
    shift 2
    printf '%s\\n' "$@" > "$FAKE_LOG_DIR/vllm.args"
    exit 0
fi
exec "$REAL_PYTHON" "$@"
"""


@pytest.fixture
def env(tmp_path):
    def script(name, text):
        path = tmp_path / name
        path.write_text(text)
        path.chmod(0o755)
        return path

    logs = tmp_path / "logs"
    logs.mkdir()
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ),
        "REAL_PYTHON": sys.executable,
        "FAKE_EXPORT": str(script("fake_export.py", FAKE_EXPORT)),
        "PYTHON": str(script("python", FAKE_PYTHON)),
        "RVLLM_BINARY": str(script("rvllm", FAKE_RVLLM)),
        "RVLLM_CACHE_DIR": str(tmp_path / "cache"),
        "FAKE_LOG_DIR": str(logs),
    }


def run(env, *args, check=True):
    return subprocess.run(
        [str(LAUNCHER), *args], env=env, capture_output=True, text=True, check=check
    )


def logged(env, name):
    path = Path(env["FAKE_LOG_DIR"]) / name
    return path.read_text().splitlines() if path.exists() else []


def export_calls(env):
    return [json.loads(line) for line in logged(env, "export.calls")]


def test_serve_exports_then_serves_and_prints_both_commands(env):
    result = run(env, "serve", "m", "--max-model-len", "2048", "--port", "8001")
    (call,) = export_calls(env)
    artifact = call[call.index("--output-dir") + 1].removesuffix(".partial")
    assert call == [
        "m",
        "--output-dir",
        artifact + ".partial",
        "--export-mode",
        "all",
        "--max-model-len",
        "2048",
    ]
    assert logged(env, "rvllm.args") == ["serve", artifact, "--port", "8001"]
    assert (Path(artifact) / ".complete").exists()
    assert "export command:" in result.stderr
    assert (
        f"serve command: {env['RVLLM_BINARY']} serve {artifact} --port 8001"
        in result.stderr
    )


def test_serve_reuses_the_export_for_the_same_model_arguments(env):
    run(env, "serve", "m", "--max-model-len", "2048", "--port", "1")
    run(env, "serve", "--model=m", "--max-model-len", "2048", "--port", "2")
    assert len(export_calls(env)) == 1, "deployment flags do not change the export"
    first = logged(env, "rvllm.args")[1]
    run(env, "serve", "m", "--max-model-len", "4096")
    assert len(export_calls(env)) == 2
    assert logged(env, "rvllm.args")[1] != first


def test_serve_fails_on_an_unsupported_option_before_exporting(env):
    result = run(env, "serve", "m", "--api-key", "secret", check=False)
    assert result.returncode != 0
    assert "--api-key" in result.stderr
    assert export_calls(env) == []
    assert logged(env, "rvllm.args") == []


def test_serve_warns_about_log_only_options(env):
    result = run(env, "serve", "m", "--uvicorn-log-level", "debug")
    assert "warning: ignoring --uvicorn-log-level" in result.stderr
    assert logged(env, "rvllm.args")[2:] == []


def test_serve_uses_a_given_artifact_without_exporting(env, tmp_path):
    env["RVLLM_ARTIFACT_DIR"] = str(tmp_path / "artifact")
    run(env, "serve", "--port", "9")
    assert export_calls(env) == []
    assert logged(env, "rvllm.args") == [
        "serve",
        str(tmp_path / "artifact"),
        "--port",
        "9",
    ]


def test_bench_runs_vllm_bench(env):
    del env["RVLLM_BINARY"]  # vllm bench needs no rvllm binary
    run(env, "bench", "serve", "--model", "m", "--max-concurrency", "8")
    assert logged(env, "vllm.args") == [
        "bench",
        "serve",
        "--model",
        "m",
        "--max-concurrency",
        "8",
    ]


def test_other_subcommands_run_the_binary_directly(env):
    run(env, "proxy", "--port", "8000")
    assert logged(env, "rvllm.args") == ["proxy", "--port", "8000"]


def test_a_binary_must_be_named(env):
    del env["RVLLM_BINARY"]
    result = run(env, "serve", "m", check=False)
    assert result.returncode != 0
    assert "RVLLM_BINARY" in result.stderr


def test_a_binary_location_that_does_not_exist_is_rejected(env, tmp_path):
    env["RVLLM_BINARY"] = str(tmp_path / "missing")
    result = run(env, "proxy", check=False)
    assert result.returncode != 0
    assert "neither a file, a directory nor a URL" in result.stderr


def _publish(env, tmp_path, sha=None):
    """Lays out a download location: the fake binary and its metadata.json."""
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    data = Path(env["RVLLM_BINARY"]).read_bytes()
    (site / "rvllm").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (site / "metadata.json").write_text(json.dumps({"sha256": sha or digest}))
    env["RVLLM_BINARY"] = str(site)
    return digest


def test_downloaded_binary_is_verified_and_cached(env, tmp_path):
    digest = _publish(env, tmp_path)
    run(env, "proxy")
    assert (Path(env["RVLLM_CACHE_DIR"]) / "bin" / digest / "rvllm").exists()
    assert logged(env, "rvllm.args") == ["proxy"]
    # Served from the cache once the download location is gone.
    (tmp_path / "site" / "rvllm").unlink()
    run(env, "proxy", "again")
    assert logged(env, "rvllm.args") == ["proxy", "again"]


def test_downloaded_binary_with_the_wrong_sha256_is_rejected(env, tmp_path):
    _publish(env, tmp_path, sha="0" * 64)
    result = run(env, "proxy", check=False)
    assert result.returncode != 0
    assert "expected " + "0" * 64 in result.stderr
    assert not list((Path(env["RVLLM_CACHE_DIR"]) / "bin").glob("*/rvllm"))

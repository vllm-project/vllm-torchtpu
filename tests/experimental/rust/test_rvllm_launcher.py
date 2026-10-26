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
"""Tests for `experimental/rust/bin/rvllm.bin`, with a fake rvllm and exporter."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCHER = (
    Path(__file__).resolve().parents[3]
    / "src/vllm_torchtpu/experimental/rust/bin/rvllm.bin"
)

FAKE_RVLLM = """#!/bin/bash
printf '%s\\n' "$@" > "$FAKE_LOG_DIR/rvllm.args"
"""

FAKE_EXPORT_MAIN = """
import json, os, sys
log = os.path.join(os.environ["FAKE_LOG_DIR"], "export.calls")
with open(log, "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
out = sys.argv[sys.argv.index("--output-dir") + 1]
os.makedirs(out)
open(os.path.join(out, "metadata.json"), "w").write("{}")
"""


@pytest.fixture
def env(tmp_path):
    """Environment with a fake rvllm binary and a fake exporter package."""
    pkg = tmp_path / "pkg" / "vllm_torchtpu" / "experimental" / "export"
    pkg.mkdir(parents=True)
    for d in [pkg.parent.parent, pkg.parent, pkg]:
        (d / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(FAKE_EXPORT_MAIN)
    rvllm = tmp_path / "rvllm"
    rvllm.write_text(FAKE_RVLLM)
    rvllm.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    return {
        **os.environ,
        "PYTHON": sys.executable,
        "PYTHONPATH": str(tmp_path / "pkg"),
        "RVLLM_BINARY": str(rvllm),
        "RVLLM_CACHE_DIR": str(tmp_path / "cache"),
        "FAKE_LOG_DIR": str(logs),
    }


def run(env, *args, check=True):
    return subprocess.run(
        [str(LAUNCHER), *args], env=env, capture_output=True, text=True, check=check
    )


def rvllm_args(env):
    return (Path(env["FAKE_LOG_DIR"]) / "rvllm.args").read_text().splitlines()


def export_calls(env):
    log = Path(env["FAKE_LOG_DIR"]) / "export.calls"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_serve_splits_deployment_flags_from_model_arguments(env):
    run(
        env,
        "serve",
        "Qwen/Qwen3-0.6B",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--port",
        "8001",
        "--role=decode",
        "--kv-producers",
        "tcp:prefill:7100",
        "--load-format",
        "dummy",
        "--tensor-parallel-size",
        "4",
        "--no-async-scheduling",
    )
    (call,) = export_calls(env)
    artifact = call[call.index("--output-dir") + 1].removesuffix(".partial")
    assert call == [
        "Qwen/Qwen3-0.6B",
        "--output-dir",
        artifact + ".partial",
        "--export-mode",
        "all",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--tensor-parallel-size",
        "4",
    ]
    assert rvllm_args(env) == [
        "serve",
        artifact,
        "--port",
        "8001",
        "--role=decode",
        "--kv-producers",
        "tcp:prefill:7100",
        "--load-format",
        "dummy",
        "--no-async-scheduling",
    ]
    assert (Path(artifact) / ".complete").exists()


def test_serve_reuses_the_export_for_the_same_model_arguments(env):
    run(env, "serve", "--model", "m", "--max-model-len", "2048", "--port", "1")
    run(env, "serve", "--model=m", "--max-model-len", "2048", "--port", "2")
    assert len(export_calls(env)) == 1, "deployment flags do not change the export"
    first = rvllm_args(env)[1]
    run(env, "serve", "m", "--max-model-len", "4096")
    assert len(export_calls(env)) == 2
    assert rvllm_args(env)[1] != first


def test_serve_uses_a_given_artifact_without_exporting(env, tmp_path):
    env["RVLLM_ARTIFACT_DIR"] = str(tmp_path / "artifact")
    run(env, "serve", "--port", "9")
    assert export_calls(env) == []
    assert rvllm_args(env) == ["serve", str(tmp_path / "artifact"), "--port", "9"]


def test_other_subcommands_run_the_binary_directly(env):
    run(env, "bench", "--url", "http://x")
    assert rvllm_args(env) == ["bench", "--url", "http://x"]


def test_serve_rejects_arguments_the_launcher_chooses(env):
    result = run(env, "serve", "m", "--output-dir", "/tmp/x", check=False)
    assert result.returncode != 0
    assert "RVLLM_ARTIFACT_DIR" in result.stderr


def test_a_binary_must_be_named(env):
    del env["RVLLM_BINARY"]
    result = run(env, "serve", "m", check=False)
    assert result.returncode != 0
    assert "RVLLM_BINARY" in result.stderr


def _publish(env, tmp_path, sha=None):
    """Lays out a download location: the fake binary and its metadata.json."""
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    data = Path(env.pop("RVLLM_BINARY")).read_bytes()
    (site / "rvllm").write_bytes(data)
    digest = sha or hashlib.sha256(data).hexdigest()
    (site / "metadata.json").write_text(json.dumps({"sha256": digest}))
    env["RVLLM_BINARY"] = str(site)
    return hashlib.sha256(data).hexdigest()


def test_downloaded_binary_is_verified_and_cached(env, tmp_path):
    digest = _publish(env, tmp_path)
    run(env, "bench")
    cached = Path(env["RVLLM_CACHE_DIR"]) / "bin" / digest / "rvllm"
    assert cached.exists()
    assert rvllm_args(env) == ["bench"]
    # Served from the cache once the download location is gone.
    (tmp_path / "site" / "rvllm").unlink()
    run(env, "bench", "again")
    assert rvllm_args(env) == ["bench", "again"]


def test_downloaded_binary_with_the_wrong_sha256_is_rejected(env, tmp_path):
    _publish(env, tmp_path, sha="0" * 64)
    result = run(env, "bench", check=False)
    assert result.returncode != 0
    assert "expected " + "0" * 64 in result.stderr
    assert not list((Path(env["RVLLM_CACHE_DIR"]) / "bin").glob("*/rvllm"))


def test_a_binary_location_that_does_not_exist_is_rejected(env, tmp_path):
    env["RVLLM_BINARY"] = str(tmp_path / "missing")
    result = run(env, "bench", check=False)
    assert result.returncode != 0
    assert "neither a file, a directory nor a URL" in result.stderr

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
"""The prefill head of .buildkite/kubernetes/qwen35_pd.sh, run for real
against fake engines: its exit status is the correctness stages', and inside a
Buildkite job (only there) a failed artifact upload fails a passing run."""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import signal
import stat
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_CONFIG = "qwen3.5-397b-fp8-pd-1p1d"
_COPIED = [
    ".buildkite/kubernetes/qwen35_pd.sh",
    ".buildkite/kubernetes/multihost_lib.sh",
    "scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-pd.sh",
    f"scripts/vllm/benchmarking/configs/{_CONFIG}.sh",
    "scripts/vllm/integration/qwen35_pd_checks.py",
]
# The ports the recipe and the launcher serve on: prefill 8000, decode's four
# DP API servers 7200-7203, the proxy 8100.
_ENGINE_PORTS = (8000, 7200, 7201, 7202, 7203, 8100)


def _executable(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class _Ok(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b'vllm:cache_config_info{block_size="1280"} 1.0\n'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def engines() -> Iterator[None]:
    """Stand-ins for the engines' and the proxy's HTTP endpoints."""
    servers = []
    for port in _ENGINE_PORTS:
        try:
            servers.append(ThreadingHTTPServer(("127.0.0.1", port), _Ok))
        except OSError as error:
            pytest.fail(
                f"port {port} is taken ({error}); this test serves the ports "
                "the recipe and the launcher use"
            )
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    yield
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture
def sandbox(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "repo"
    for rel in _COPIED:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO / rel, root / rel)
    # The correctness stages, reduced to their exit status and what they got.
    _executable(
        root / "scripts/vllm/integration/run_qwen35_397b_pd_correctness.sh",
        '#!/bin/bash\nenv > "$STUB_ENV_OUT"\nexit "${STUB_RC:-0}"\n',
    )
    bin_dir = tmp_path / "bin"
    # The engine and the proxy processes: alive until the launcher kills them.
    _executable(bin_dir / "vllm", "#!/bin/bash\nexec sleep 600\n")
    _executable(bin_dir / "vllm-router", "#!/bin/bash\nexec sleep 600\n")
    # An agent whose upload fails unless told otherwise, as one without a job
    # token does.
    _executable(
        bin_dir / "buildkite-agent",
        '#!/bin/bash\necho "$*" >> "$AGENT_CALLS"\nexit "${AGENT_RC:-1}"\n',
    )
    return root


def _launch(root: pathlib.Path, **env: str) -> tuple[int, str, dict[str, str]]:
    tmp = root.parent
    run_env = {
        "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}",
        "HOME": str(tmp),
        "REPO_DIR": str(root),
        "QWEN35_PD_CONFIG": _CONFIG,
        "POD_IP": "127.0.0.1",
        "PREFILL_HOSTS": "127.0.0.1",
        "DECODE_HOSTS": "127.0.0.1",
        "PD_REPLICA_INDEX": "0",
        "ARTIFACTS_DIR": str(tmp / "artifacts"),
        "STUB_ENV_OUT": str(tmp / "stages_env.txt"),
        "AGENT_CALLS": str(tmp / "agent_calls.txt"),
        **env,
    }
    proc = subprocess.Popen(
        ["bash", str(root / ".buildkite/kubernetes/qwen35_pd.sh"), "prefill"],
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=240)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    stages_env = {}
    env_file = tmp / "stages_env.txt"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            stages_env[key] = value
    return proc.returncode, out, stages_env


@pytest.mark.usefixtures("engines")
@pytest.mark.parametrize("stages_rc", [0, 1])
def test_the_launcher_exits_with_the_stages_status(sandbox, stages_rc):
    rc, out, stages_env = _launch(sandbox, STUB_RC=str(stages_rc))
    assert rc == stages_rc, out[-3000:]
    assert stages_env["QWEN35_PD_CONFIG"] == _CONFIG
    assert stages_env["DECODE_URLS"].split() == [
        f"http://127.0.0.1:{port}" for port in (7200, 7201, 7202, 7203)
    ]
    # Outside a Buildkite job nothing is uploaded, even with the agent on PATH.
    assert not (sandbox.parent / "agent_calls.txt").exists()


@pytest.mark.usefixtures("engines")
@pytest.mark.parametrize("agent_rc,expected", [(1, 1), (0, 0)])
def test_inside_buildkite_the_upload_decides_a_passing_run(sandbox, agent_rc, expected):
    rc, out, _ = _launch(
        sandbox, STUB_RC="0", BUILDKITE_JOB_ID="0190-test", AGENT_RC=str(agent_rc)
    )
    assert rc == expected, out[-3000:]
    calls = (sandbox.parent / "agent_calls.txt").read_text()
    assert calls.startswith("artifact upload")

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
"""Tests for the head bring-up in .buildkite/scripts/run_multihost_mp.sh."""

import os
import pathlib
import shutil
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SLEEP = shutil.which("sleep")

# Takes PULL_S real seconds to pull before the container exists, then serves
# /health until torn down, or exits DIE_S seconds later. PULL_ERROR fails it.
_FAKE_DOCKER = """#!/bin/bash
state="$FAKE_DIR/node"
case "$1" in
  run)
    SLEEP "$PULL_S"
    if [ -n "$PULL_ERROR" ]; then
      exit 125
    fi
    echo "fake head output" >"$FAKE_DIR/node.log"
    if [ -n "$DIE_S" ]; then
      echo booting >"$state"
      SLEEP "$DIE_S"
      exit 1
    fi
    echo running >"$state"
    while [ ! -f "$FAKE_DIR/down" ]; do SLEEP 0.05; done
    ;;
  exec)
    grep -qx running "$state" 2>/dev/null
    ;;
  logs)
    [ -f "$state" ] || exit 1
    cat "$FAKE_DIR/node.log"
    while [ "$2" = -f ] && grep -qxE "running|booting" "$state" 2>/dev/null; do
      SLEEP 0.05
    done
    ;;
  stop | rm)
    rm -f "$state"
    ;;
esac
""".replace("SLEEP", _SLEEP)

# Sources the backend the way run_multihost.sh does, on a single host.
_DRIVER = """
set -euo pipefail
REPO_DIR="$1"
RUN_CLUSTER_MP="$REPO_DIR/scripts/multihost/run_cluster_mp.sh"
IMAGE_TAG=fake/vllm-torchtpu:new-commit
HOST_HF_HOME="$FAKE_DIR/hf_home"
HEAD_INTERNAL_IP=10.0.0.1
WORKER_IPS_ARRAY=()
NUM_HOSTS=1
CONTAINER_ENV_HEAD=()
MULTIHOST_RUN_LOG="$FAKE_DIR/multihost.log"
source "$REPO_DIR/.buildkite/scripts/run_multihost_mp.sh"
EXIT_CODE=0
run_mp_multihost glm-5.2-fp8-dp16-ep || EXIT_CODE=$?
touch "$FAKE_DIR/down"
wait
exit "$EXIT_CODE"
"""


def _run_head(tmp_path, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "docker": _FAKE_DOCKER,
        # Scripted polls sleep 50 ms; the fake pull and death take real seconds.
        "sleep": f"#!/bin/bash\nexec {_SLEEP} 0.05\n",
    }
    for name, body in stubs.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_DIR": str(tmp_path),
        "DATA_PARALLELISM_LOCAL": "16",
        **env,
    }
    return subprocess.run(
        ["bash", "-c", _DRIVER, "driver", str(_REPO)],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_head_waits_out_a_slow_image_pull(tmp_path):
    result = _run_head(tmp_path, PULL_S="2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Head server is healthy." in result.stdout


@pytest.mark.parametrize(
    "head_env",
    [
        {"PULL_S": "6", "DIE_S": "1"},
        {"PULL_S": "1", "PULL_ERROR": "1"},
    ],
    ids=["dies-after-slow-pull", "pull-fails"],
)
def test_head_that_exits_fails_fast(tmp_path, head_env):
    result = _run_head(tmp_path, **head_env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "the head container exited before the server became healthy" in (
        result.stdout
    )
    if "DIE_S" in head_env:
        head_log = tmp_path / "perf_eval_results" / "head_server.log"
        assert "fake head output" in head_log.read_text()

# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for DP16 coordination and peer-log validation."""

import datetime
import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dsv41_bench/server_steps"))

from audit_multihost import audit  # noqa: E402
from bench_decode import summarize  # noqa: E402
from test_dsv41_decode_accounting import multihost_records  # noqa: E402

pytestmark = pytest.mark.cpu_test


def write_hosts(root):
    records = multihost_records(10**15)
    for rank in records:
        # The peer's wall clock is deliberately one day ahead as well.
        low = int(
            datetime.datetime(2026, 11, 2 + rank["rank"] // 8, 12).timestamp() * 1e9
        )
        rank["audit_wall_interval_ns"] = [low, low + 10_000_000_000]
    for index, path in enumerate([root / "benchmark", root / "worker/benchmark"]):
        path.mkdir(parents=True)
        (path / "server.log").write_text("Server ready\n")
        (path / "startup.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "formal_status": "passed" if index == 0 else "pending",
                    "shutdown_requested": index == 1,
                }
            )
        )
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "scenario": "decode_dp16",
                    "host_rank": index,
                    "hostname": f"host-{index}",
                    "compile_sizes": [288, 2048],
                    "cache_mode": "disabled",
                    "engram_hbm": "first",
                    "engram_dp_shared_memory": False,
                    "moe_kernel": "push",
                    "moe_push_sc_prepush": 2,
                    "moe_push_tc_exchange": True,
                    "moe_preprocess": "topology",
                    "allocator": "glibc",
                    "reclaim_compile_memory": True,
                    "identity": {
                        "torchtpu": {"commit": "abc"},
                        "vllm": {"commit": "def"},
                        "model_weights": "gs://model",
                        "model_config_sha256": "123",
                        "packages": {"torch-tpu": "pinned"},
                    },
                }
            )
        )
    fixed = root / "benchmark/fixed"
    fixed.mkdir()
    (fixed / "manifest.json").write_text(json.dumps({"rounds": 1}))
    (fixed / "round-1-steps.json").write_text(json.dumps({"results": records}))
    report = summarize(records, "decode_dp16")
    report["warmup"] = False
    (fixed / "round-1-summary.json").write_text(json.dumps(report))


def test_both_host_audit_uses_local_clocks(tmp_path):
    write_hosts(tmp_path)
    # Outside this peer's formal window, even though inside the head's window.
    (tmp_path / "worker/benchmark/server.log").write_text(
        "INFO 11-02 12:00:05 Compiling FX graph\n"
    )
    assert audit(tmp_path)["ranks"] == list(range(16))


@pytest.mark.parametrize(
    "invalid", ["compile", "source", "shutdown", "config", "logs", "rank"]
)
def test_peer_failure_cannot_publish_valid_metrics(tmp_path, invalid):
    write_hosts(tmp_path)
    peer = tmp_path / "worker/benchmark"
    if invalid == "compile":
        (peer / "server.log").write_text("INFO 11-03 12:00:05 Compiling FX graph\n")
    elif invalid == "logs":
        (peer / "server.log").unlink()
    elif invalid == "rank":
        path = tmp_path / "benchmark/fixed/round-1-steps.json"
        data = json.loads(path.read_text())
        data["results"].pop()
        path.write_text(json.dumps(data))
    else:
        path = peer / ("startup.json" if invalid == "shutdown" else "manifest.json")
        data = json.loads(path.read_text())
        if invalid == "source":
            data["identity"]["torchtpu"]["commit"] = "different"
        elif invalid == "config":
            data["engram_hbm"] = "off"
        else:
            data["shutdown_requested"] = False
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)):
        audit(tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [("moe_kernel", "v2"), ("moe_push_sc_prepush", 1), ("moe_push_tc_exchange", False)],
)
def test_host_audit_rejects_a_different_push_recipe(tmp_path, key, value):
    write_hosts(tmp_path)
    path = tmp_path / "worker/benchmark/manifest.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=key):
        audit(tmp_path)


def test_pipeline_moves_only_decode_and_preserves_kernel_coverage():
    pipeline = yaml.safe_load((ROOT / ".buildkite/pipeline_dsv41.yml").read_text())
    steps = {
        step["key"]: step
        for group in pipeline["steps"]
        for step in group.get("steps", [group])
    }
    decode = steps["dsv41_flash_dp16_c768_decode"]
    assert decode["agents"]["queue"] == "tpu_v7x_16_queue"
    assert decode["env"]["CONTAINER_SHM_SIZE"] == "256g"
    assert "--backend mp --dsv41-decode" in decode["commands"][0]
    assert decode["depends_on"] == "dsv41_decode_kernel_parity"
    for key in (
        "dsv41_flash_dp8_ep_gsm8k",
        "dsv41_flash_dp8_c8_cached_extend",
        "dsv41_decode_kernel_parity",
    ):
        assert steps[key]["agents"]["queue"] == "tpu_v7x_8_queue"


@pytest.mark.parametrize(
    "failure",
    ["", "head", "worker", "audit", "probe_once", "probe_always", "probe_error"],
)
def test_shell_coordinator_propagates_failure_and_stops_both_hosts(tmp_path, failure):
    binary = tmp_path / "bin"
    binary.mkdir()
    (tmp_path / "perf_eval_results").mkdir()
    out = tmp_path / "persist/perf_eval_results/result/benchmark"
    out.mkdir(parents=True)
    out.parent.chmod(0o777)
    (out.parent / "task.log").write_text("head log")
    peer = tmp_path / "peer/persist/perf_eval_results/result"
    (peer / "worker/benchmark").mkdir(parents=True)
    peer.chmod(0o755)
    (peer / "task.log").write_text("must not overwrite the head log")
    (peer / "worker/benchmark/server.log").write_text("worker log")
    # Simulate the remote worker and Docker control plane; no SSH or containers.
    fake = binary / "fake"
    fake.write_text("""#!/usr/bin/env python3
import os, pathlib, subprocess, sys, time
root = pathlib.Path(os.environ["TEST_ROOT"])
failure = os.environ["TEST_FAILURE"]
name = pathlib.Path(sys.argv[0]).name
command = sys.argv[-1]
if name == "sleep":
    time.sleep(0.01)
elif name == "docker":
    if sys.argv[1] == "stop":
        (root / "head-stop").touch()
    else:
        (root / "audited").touch()
        sys.exit(1 if failure == "audit" else 0)
elif "bash ~/multihost/run_cluster_mp.sh" in command:
    (root / "worker-command").write_text(command)
    (root / "worker-start").touch()
    if failure == "worker":
        sys.exit(17)
    while not (root / "worker-stop").exists():
        time.sleep(0.01)
    (root / "worker-finished").touch()
elif "docker exec" in command or "docker stop" in command:
    (root / "worker-stop").touch()
elif "test -f" in command:
    probes = root / "probes"
    count = int(probes.read_text()) + 1 if probes.exists() else 1
    probes.write_text(str(count))
    if failure == "probe_always" or (failure == "probe_once" and count == 1):
        sys.exit(255)
    if failure == "probe_error":
        sys.exit(2)
    sys.exit(0 if (root / "worker-start").exists() else 1)
elif "launch-permitted" in command:
    (root / "worker-launch-permitted").touch()
elif "tar -C" in command:
    sys.exit(subprocess.run(
        ["bash", "-c", command], env={**os.environ, "HOME": str(root / "peer")}
    ).returncode)
elif "cat >" in command or "bash -s" in command:
    sys.stdin.read()
""")
    fake.chmod(0o755)
    for name in ("ssh", "docker", "sleep"):
        (binary / name).symlink_to(fake)
    head = tmp_path / "run_cluster_mp.sh"
    head.write_text("""#!/bin/bash
out="$TEST_ROOT/persist/perf_eval_results/result/benchmark"
printf '%s\\n' "$@" > "$TEST_ROOT/head-command"
if [[ "$TEST_FAILURE" == head ]]; then exit 23; fi
while [[ ! -f "$TEST_ROOT/worker-start" ]]; do sleep 0.01; done
touch "${out%/benchmark}/ready-to-launch"
while [[ ! -f "${out%/benchmark}/launch-permitted" && ! -f "$TEST_ROOT/head-stop" ]]; do
  sleep 0.01
done
if [[ -f "$TEST_ROOT/head-stop" ]]; then exit 1; fi
test -f "$TEST_ROOT/worker-launch-permitted" || exit 41
if [[ "$TEST_FAILURE" != worker ]]; then touch "$out/formal-complete"; fi
while [[ ! -f "$out/shutdown-permitted" && ! -f "$TEST_ROOT/head-stop" ]]; do
  sleep 0.01
done
if [[ -f "$out/shutdown-permitted" ]]; then
  test -f "$TEST_ROOT/worker-finished" || exit 42
fi
""")
    script = """
set -euo pipefail
NUM_HOSTS=2
WORKER_IPS_ARRAY=(worker)
CONTAINER_ENV_COMMON=(-e TPU_MULTIHOST_BACKEND=mp)
CONTAINER_ENV_HEAD=(-e TPU_MULTIHOST_BACKEND=mp)
SSH_OPTS=(-o BatchMode=yes)
SSH_USER=ci
HEAD_INTERNAL_IP=10.0.0.1
IMAGE_TAG=test-image
HOST_HF_HOME="$TEST_ROOT/hf"
PERSIST_ROOT="$TEST_ROOT/persist"
RUN_CLUSTER_MP="$TEST_ROOT/run_cluster_mp.sh"
SCRIPT_DIR="$REPO/.buildkite/scripts"
MULTIHOST_RUN_LOG=perf_eval_results/head.log
ssh_retry() { ssh "$@"; }
source "$SCRIPT_DIR/run_dsv41_dp16_multihost.sh"
status=0
run_dsv41_dp16_multihost gs://test/model /perf_eval_results/result || status=$?
exit "$status"
"""
    process = subprocess.Popen(
        ["bash", "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "REPO": str(ROOT),
            "TEST_ROOT": str(tmp_path),
            "TEST_FAILURE": failure,
            "BUILDKITE_COMMIT": "a" * 40,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=20)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    assert (process.returncode == 0) == (failure in ("", "probe_once")), stdout + stderr
    assert (tmp_path / "audited").exists() == (failure in ("", "probe_once", "audit"))
    assert (out.parent / "task.log").read_text() == "head log"
    assert out.parent.stat().st_mode & 0o777 == 0o777
    assert (out.parent / "worker/benchmark/server.log").read_text() == "worker log"
    if failure in ("probe_once", "probe_always", "probe_error"):
        assert (
            int((tmp_path / "probes").read_text())
            == {"probe_once": 2, "probe_always": 5, "probe_error": 1}[failure]
        )
    if failure != "worker":
        assert (tmp_path / "worker-finished").exists()
    for path in ("head-command", "worker-command"):
        command = (tmp_path / path).read_text()
        assert "--shm-size=256g" in command
        assert "DSV41_HEAD_ADDRESS=10.0.0.1" in command

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
"""CPU regressions for DP16 coordination and peer-log validation."""

import datetime
import json
import os
import signal
import socket
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dsv41_bench/server_steps"))

import coordinate_multihost as coordinator  # noqa: E402
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
                    "moe_eplb_dynamic": True,
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
    [
        ("moe_kernel", "v2"),
        ("moe_push_sc_prepush", 1),
        ("moe_push_tc_exchange", False),
        ("moe_eplb_dynamic", False),
    ],
)
def test_host_audit_rejects_a_different_decode_recipe(tmp_path, key, value):
    write_hosts(tmp_path)
    path = tmp_path / "worker/benchmark/manifest.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=key):
        audit(tmp_path)


def _pipeline_steps():
    pipeline = yaml.safe_load((ROOT / ".buildkite/pipeline_dsv41.yml").read_text())
    return {
        step["key"]: step
        for group in pipeline["steps"]
        for step in group.get("steps", [group])
    }


_TPU_8 = "tpu7x-standard-4t/2x2x1"
_TPU_2 = "tpu7x-standard-1t/1x1x1"
_SHAPES = {
    "dsv41_layout_parity": _TPU_2,
    "dsv41_greedy_head": _TPU_2,
    "dsv41_output_projection": _TPU_2,
    "dsv41_projection_rope": _TPU_2,
    "dsv41_swa_parity": _TPU_2,
    "dsv41_benchmark_scripts_cpu": "cpu",
    "dsv41_kernel_config_cpu": "cpu",
    "dsv41_stage_model": "cpu",
    "dsv41_flash_dp8_ep_gsm8k": _TPU_8,
    "dsv41_decode_kernel_parity": _TPU_8,
    "dsv41_flash_dp8_c8_cached_extend": _TPU_8,
    # The manifest names its own hardware.
    "dsv41_flash_dp16_c768_decode": None,
}


def test_pipeline_runs_every_step_on_the_kube_fleet():
    text = (ROOT / ".buildkite/pipeline_dsv41.yml").read_text()
    for gone in ("run_in_docker", "tpu_v7x_", "CPU_ONLY", "CONTAINER_SHM_SIZE"):
        assert gone not in text, gone
    assert "/perf_eval_results" not in text
    steps = _pipeline_steps()
    assert set(steps) == set(_SHAPES)
    for key, step in steps.items():
        assert step["agents"]["queue"] == "kube", key
        assert step["plugins"] == [{"kubernetes": {"podTemplate": "tpu-launcher"}}]
        assert step["timeout_in_minutes"] >= 120, key
        assert "artifact_paths" not in step, key
        env = step.get("env", {})
        assert env.get("SHAPE") == _SHAPES[key], key
        if _SHAPES[key] == "cpu":
            assert env["JAX_PLATFORMS"] == "cpu"
        launcher = (
            ".buildkite/kubernetes/run_multihost.sh"
            if _SHAPES[key] is None
            else ".buildkite/kubernetes/run.sh"
        )
        assert all(c.startswith(launcher) for c in step["commands"]), key


def test_pipeline_keeps_checks_and_order():
    steps = _pipeline_steps()
    checks = {
        key: step["notify"][0]["github_check"]["name"]
        for key, step in steps.items()
        if "notify" in step
    }
    assert checks == {
        "dsv41_flash_dp8_ep_gsm8k": "DeepSeek-V4.1 DP8 GSM8K eval",
        "dsv41_flash_dp16_c768_decode": "DeepSeek-V4.1 DP16 Decode throughput",
        "dsv41_flash_dp8_c8_cached_extend": "DeepSeek-V4.1 DP8 Prefill throughput",
    }
    decode = steps["dsv41_flash_dp16_c768_decode"]
    assert sorted(decode["depends_on"]) == [
        "dsv41_decode_kernel_parity",
        "dsv41_stage_model",
    ]
    # Every serving benchmark streams the staged checkpoint.
    for key in ("dsv41_flash_dp8_ep_gsm8k", "dsv41_flash_dp8_c8_cached_extend"):
        assert steps[key]["depends_on"] == "dsv41_stage_model", key
    for key in checks:
        command = " ".join(steps[key]["commands"])
        assert "wdhongtw-hf-flat" not in command, key
    assert decode["commands"] == [
        ".buildkite/kubernetes/run_multihost.sh --manifest dsv41-dp16-decode.yaml"
    ]
    extend = " ".join(steps["dsv41_flash_dp8_c8_cached_extend"]["commands"])
    # The throughput check is published from the pod, around the benchmark.
    assert "dsv41_publish_check.sh" in extend
    assert extend.index("dsv41_publish_check.sh") < extend.index("encoder_extend")


def _manifest():
    path = ROOT / ".buildkite/kubernetes/manifests/workloads/dsv41-dp16-decode.yaml"
    return yaml.safe_load(path.read_text())


def test_manifest_is_one_two_host_slice_with_a_large_shm():
    manifest = _manifest()
    assert manifest["metadata"]["annotations"]["tpu-ci.google.com/defaults"] == (
        "standard"
    )
    assert manifest["spec"]["network"]["enableDNSHostnames"] is True
    [job] = manifest["spec"]["replicatedJobs"]
    spec = job["template"]["spec"]
    assert (spec["parallelism"], spec["completions"]) == (2, 2)
    assert spec["completionMode"] == "Indexed"
    assert spec["successPolicy"]["rules"] == [{"succeededIndexes": "0"}]
    assert spec["backoffLimit"] == 0
    pod = spec["template"]["spec"]
    assert pod["nodeSelector"] == {
        "cloud.google.com/gke-tpu-topology": "2x2x2",
        "cloud.google.com/gke-tpu-accelerator": "tpu7x",
    }
    [container] = pod["containers"]
    assert container["resources"]["limits"] == {"google.com/tpu": "4"}
    assert coordinator.DEFAULT_PORT in [p["containerPort"] for p in container["ports"]]
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert env["HEAD_HOST"] == "${WORKLOAD_NAME}-slice-0-0.${WORKLOAD_NAME}"
    assert env["ARTIFACTS_DIR"] == "perf_eval_results"
    assert env["TMPDIR"] == "/dev/shm"
    [mount] = container["volumeMounts"]
    assert mount["mountPath"] == "/dev/shm"
    [volume] = [v for v in pod["volumes"] if v["name"] == mount["name"]]
    assert volume["emptyDir"] == {"medium": "Memory", "sizeLimit": "256Gi"}
    command = container["args"][0].split()
    script = command[command.index("exec") + 2]
    assert script.startswith("/workspace/vllm-torchtpu/")
    assert (ROOT / script.removeprefix("/workspace/vllm-torchtpu/")).is_file()
    # Results land under ARTIFACTS_DIR, where the head uploads them.
    assert command[-1].startswith("perf_eval_results/")


@pytest.mark.parametrize("index", ["0", "1", "2"])
def test_pod_entry_picks_its_role(tmp_path, index):
    binary = tmp_path / "bin"
    binary.mkdir()
    # Stubs for what a pod has and a Mac does not.
    (binary / "getent").write_text("#!/bin/sh\necho '10.0.0.1 STREAM head'\n")
    (binary / "python3").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" >> "$TEST_ROOT/python3-calls"\n'
    )
    for name in ("getent", "python3"):
        (binary / name).chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key != "ARTIFACTS_DIR"}
    result = subprocess.run(
        [
            "bash",
            str(ROOT / ".buildkite/scripts/run_dsv41_dp16_pod.sh"),
            "gs://bucket/model",
            "perf_eval_results/dp16",
        ],
        env={
            **env,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "TEST_ROOT": str(tmp_path),
            "HEAD_HOST": "head.example",
            "JOB_COMPLETION_INDEX": index,
        },
        capture_output=True,
        text=True,
    )
    if index == "2":
        assert result.returncode == 2
        return
    assert result.returncode == 0, result.stdout + result.stderr
    calls = (tmp_path / "python3-calls").read_text().splitlines()
    assert calls[0].endswith("coordinate_multihost.py")
    start = calls.index("--") + 1
    bench = calls[start : start + 5]
    assert bench[0] == "bash" and bench[1].endswith("run_dsv41_benchmark.sh")
    assert bench[2:] == ["decode_dp16", "gs://bucket/model", "perf_eval_results/dp16"]
    if index == "0":
        assert calls[1:4] == ["head", "--results", "perf_eval_results/dp16"]
        # The head publishes the check after its coordinator returns.
        assert calls[-5].endswith("dsv41_check_summary.py")
        assert calls[-4:-2] == ["perf_eval_results/dp16/benchmark", "0"]
    else:
        assert calls[1:6] == [
            "worker",
            "--results",
            "perf_eval_results/dp16",
            "--head",
            "10.0.0.1",
        ]
        assert not any(c.endswith("dsv41_check_summary.py") for c in calls)


# Stands in for run_dsv41_benchmark.sh decode_dp16 on either host: the same
# handshake files, and run.py's SIGTERM-as-orderly-stop on the worker.
_FAKE_BENCH = r"""
import os, pathlib, signal, subprocess, sys, time
role, results, failure = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
if role == "worker":
    results = results / "worker"
out = results / "benchmark"
out.mkdir(parents=True)
(out / "server.log").write_text(role + " log")
(out / "bench.pid").write_text(str(os.getpid()))
if role == "head" and failure == "head":
    sys.exit(3)
(results / "ready-to-launch").touch()
while not (results / "launch-permitted").exists():
    time.sleep(0.02)
(out / "launched").touch()
if role == "head":
    (out / "formal-complete").touch()
    while not (out / "shutdown-permitted").exists():
        time.sleep(0.02)
    sys.exit(0)
if failure == "worker":
    sys.exit(17)
stopped = []
signal.signal(signal.SIGTERM, lambda *_: stopped.append(1))
pid = os.getpid()
if failure == "stuck":
    # The launcher stops on SIGTERM but its benchmark wrapper never exits.
    pid = subprocess.Popen(["sleep", "600"]).pid
(out / "launcher.pid").write_text(str(pid))
while not stopped:
    time.sleep(0.02)
(out / "shutdown_requested").touch()
"""


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("failure", ["", "head", "worker", "stuck", "audit", "lost"])
def test_coordinator_fails_closed_and_collects_the_worker(tmp_path, failure):
    results = tmp_path / "results"
    bench = tmp_path / "bench.py"
    bench.write_text(_FAKE_BENCH)
    audit_stub = tmp_path / "audit.py"
    audit_stub.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1], 'audited').touch()\n"
        f"sys.exit({1 if failure == 'audit' else 0})\n"
    )
    port = str(_free_port())
    script = str(ROOT / "scripts/dsv41_bench/server_steps/coordinate_multihost.py")
    common = ["--results", str(results), "--port", port, "--poll", "0.05"]
    head = subprocess.Popen(
        [sys.executable, script, "head", *common, "--bind", "127.0.0.1",
         "--worker-stop-timeout", "6", "--collect-timeout", "10",
         "--worker-lost-after", "2", "--audit", str(audit_stub),
         "--", sys.executable, str(bench), "head", str(results), failure],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )  # fmt: skip
    worker = subprocess.Popen(
        [sys.executable, script, "worker", *common, "--head", "127.0.0.1",
         "--stop-timeout", "2", "--head-lost-after", "3",
         "--connect-timeout", "10",
         "--", sys.executable, str(bench), "worker", str(results), failure],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )  # fmt: skip
    try:
        if failure == "lost":
            # The worker pod vanishes mid-run, without a word to the head.
            launched = results / "worker/benchmark/launched"
            deadline = 100
            while not launched.exists() and deadline:
                deadline -= 1
                with suppress(subprocess.TimeoutExpired):
                    worker.wait(timeout=0.1)
            os.killpg(worker.pid, signal.SIGKILL)
            # A pod's death takes its benchmark with it; here the benchmark
            # has a session of its own and would keep the worker's stdout open.
            pid = int((results / "worker/benchmark/bench.pid").read_text())
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)
        head_out, _ = head.communicate(timeout=40)
        worker_out, _ = worker.communicate(timeout=40)
    finally:
        for process in (head, worker):
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    log = head_out + (worker_out or "")
    assert (head.returncode == 0) == (failure == ""), log
    if failure != "lost":
        # The worker never fails the Job; the head owns the verdict.
        assert worker.returncode == 0, log
        assert (results / "worker/benchmark/server.log").read_text() == "worker log"
    assert (results / "audited").exists() == (failure in ("", "audit")), log
    if failure in ("", "audit"):
        assert (results / "worker/benchmark/shutdown_requested").exists()
        assert (results / "benchmark/shutdown-permitted").exists()
    else:
        assert not (results / "benchmark/shutdown-permitted").exists()
    if failure == "head":
        # Neither host was allowed to start serving.
        assert not (results / "worker/benchmark/launched").exists()


def test_head_rejects_a_second_worker_after_launch(tmp_path):
    state = coordinator.HeadState(tmp_path)
    state.check_worker("a")
    state.check_worker("b")
    # Before launch a recreated worker just replaces the first and re-registers.
    assert state.worker_id == "b" and not state.abort
    state.launch = True
    state.check_worker("b")
    assert not state.abort
    state.check_worker("c")
    assert state.abort and "second worker" in (state.error or "")

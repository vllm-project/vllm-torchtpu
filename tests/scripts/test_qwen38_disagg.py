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
"""Execute PD role scripts with CPU stubs and check topology/role isolation."""

import importlib.util
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
KUBE = ROOT / ".buildkite/kubernetes"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROFILE = load("disagg_qwen38", KUBE / "disagg_qwen38.py")
SMOKE = load("check_pd_completions", ROOT / "examples/disagg/check_pd_completions.py")


def flag(args, name):
    for index, arg in enumerate(args):
        if arg == name:
            return args[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    raise AssertionError(name)


@pytest.mark.parametrize("dp", [8, 4])
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_role_command_and_model_admission(dp, role, monkeypatch):
    from types import SimpleNamespace as NS

    from tests.models.vllm.test_qwen4_exp_parallel import config
    from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config

    profile = PROFILE.role_profile(role, dp, "/tmpfs/model", "127.0.0.1")
    args, env = profile["argv"], profile["env"]
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert profile["tp"] * dp == profile["ep"] == 8
    assert env["TPU_RAIDEN_TRANSFER_PARALLELISM"] == str(8 // dp)
    assert env["TPU_PREMAPPED_BUFFER_SIZE"] == str(4 << 30)
    assert flag(args, "--model") == "/tmpfs/model"
    assert flag(args, "--block-size") == "128"
    assert "--async-scheduling" in args and "--enable-chunked-prefill" in args
    assert "--no-enable-prefix-caching" in args
    assert not any(
        "speculative" in arg or "context-parallel-size=8" in arg for arg in args
    )
    additional = json.loads(flag(args, "--additional-config"))
    qwen = additional["qwen4_exp"]
    prefill = role == "prefill"
    assert qwen["ple_host_backend"] == ("torch" if prefill else "native_dma")
    assert qwen["engram_config"] == dict(
        cpu_offload=True, dp_shared_memory=prefill, embedding_across_dp=not prefill
    )
    assert json.loads(flag(args, "--compilation-config"))["compile_sizes"] == (
        [8192] if prefill else [8, 8192]
    )
    cfg = config(8 // dp, dp)
    cfg.additional_config = additional
    cfg.cache_config.cache_dtype = "fp8_e4m3"
    cfg.cache_config.mamba_cache_dtype = "float32"
    cfg.cache_config.mamba_ssm_cache_dtype = "bfloat16"
    cfg.cache_config.mamba_cache_mode = "none"
    cfg.kv_transfer_config = NS(**json.loads(flag(args, "--kv-transfer-config")))
    validate_config(cfg)


STUB = """#!{python}
import json, os, pathlib, resource, sys
name = pathlib.Path(sys.argv[0]).name
root = pathlib.Path(os.environ["STUB_ROOT"])
if name == "python3":
    script = pathlib.Path(sys.argv[1]).name
    if script == "toy_proxy_server.py":
        (root / "proxy.pid").write_text(str(os.getpid()))
        if "STUB_PROXY_RC" in os.environ:
            sys.exit(int(os.environ["STUB_PROXY_RC"]))
        os.execv("/bin/sleep", ["sleep", "600"])
    if script == "check_pd_completions.py":
        out = pathlib.Path(sys.argv[sys.argv.index("--output-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(dict(
            passed=os.environ.get("STUB_NUMERIC_FAILURE") != "1",
            performance_checks_passed=True,
            numeric_reference_passed=os.environ.get("STUB_NUMERIC_FAILURE") != "1",
            diagnostic_failed=os.environ.get("STUB_NUMERIC_FAILURE") == "1")))
        sys.exit(int(os.environ.get("STUB_SANITY_RC", "0")))
    if script == "check_pd_apc.py":
        out = pathlib.Path(sys.argv[sys.argv.index("--output-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        rc = int(os.environ.get("STUB_APC_RC", "0"))
        (out / "summary.json").write_text(json.dumps(dict(passed=rc == 0)))
        (root / "apc-client.json").write_text(json.dumps(sys.argv[2:]))
        sys.exit(rc)
    if script == "parse_gke_results.py":
        with (root / "bq-called").open("a") as f:
            f.write(json.dumps(sys.argv[2:]) + "\\n")
        sys.exit(0)
    if script == "disagg_cpu_bench.py":
        name = "vllm"
        sys.argv = sys.argv[1:]
    else:
        os.execv("{python}", ["{python}", *sys.argv[1:]])
if name == "vllm":
    with (root / "vllm.jsonl").open("a") as f:
        prefixes = ("TPU_", "VLLM_TPU_", "LIBTPU_", "MOE_", "USE_MOE_")
        selected = {{k: v for k, v in os.environ.items() if k.startswith(prefixes)}}
        f.write(json.dumps(dict(argv=sys.argv[1:], env=selected,
            core_limit=resource.getrlimit(resource.RLIMIT_CORE)[0])) + "\\n")
    for arg in sys.argv:
        if arg.startswith("--result-file="):
            result = pathlib.Path(arg.split("=", 1)[1])
            mode = os.environ.get("STUB_RESULT_MODE", "normal")
            expected = int(next(
                x.split("=", 1)[1] for x in sys.argv
                if x.startswith("--num-prompts=")))
            if mode != "missing":
                result.write_text(json.dumps(dict(
                    completed=expected-1 if mode == "short" else expected,
                    failed=1 if mode == "failed" else 0)))
    sys.exit(int(os.environ.get("STUB_VLLM_RC", "0")))
if name == "pip":
    (root / "pip-called").touch()
    sys.exit(99)
if name == "buildkite-agent":
    with (root / "artifacts.jsonl").open("a") as f:
        f.write(json.dumps(sys.argv[1:]) + "\\n")
    sys.exit(int(os.environ.get("STUB_ARTIFACT_RC", "0")))
if name == "curl":
    import time
    time.sleep(0.05)
sys.exit(0)
"""


def stubs(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    # Braces in the JSON literal are doubled; the empty summary is literal too.
    script = (
        STUB.replace("{python}", sys.executable).replace("{{", "{").replace("}}", "}")
    )
    for name in ["vllm", "python3", "curl", "pip", "sleep", "buildkite-agent"]:
        target = binaries / name
        target.write_text(script)
        target.chmod(0o755)
    return dict(
        os.environ,
        PATH=str(binaries) + ":/usr/bin:/bin",
        STUB_ROOT=str(tmp_path),
        DISAGG_PROFILE="qwen3.8-flash",
        POD_IP="127.0.0.1",
        WORKLOAD_NAME="cpu-test",
        DISAGG_RESULTS_DIR=str(tmp_path / "results"),
        BQ_UPLOAD="0",
    )


@pytest.mark.parametrize("dp", [4, 8])
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_actual_engine_script_overrides_ambient_backend_and_keeps_bootstrap(
    dp, role, tmp_path
):
    env = stubs(tmp_path)
    env.update(
        DISAGG_DP_SIZE=str(dp),
        VLLM_TPU_QWEN38_FLASH_PLE_HOST_BACKEND="native_dma",
        MOE_FUSED_EP_KERNEL_IMPL="adaptive",
    )
    factors = dict(
        TPU_ACCELERATOR_TYPE="operator-type",
        TPU_WORKER_HOSTNAMES="operator-host",
        TPU_CHIPS_PER_HOST_BOUNDS="2,2,1",
        TPU_HOST_BOUNDS="1,1,1",
        TPU_SKIP_MDS_QUERY="1",
        LIBTPU_INIT_ARGS="--operator_flag=value",
    )
    env.update(factors)
    result = subprocess.run(
        ["bash", str(KUBE / "disagg_engine.sh"), role],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        preexec_fn=lambda: resource.setrlimit(
            resource.RLIMIT_CORE, (1024, resource.getrlimit(resource.RLIMIT_CORE)[1])
        ),
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "vllm.jsonl").read_text())
    assert receipt["core_limit"] == 0
    effective = receipt["env"]
    assert effective["TPU_RAIDEN_ENGINE_ID"] == f"cpu-test-{role}"
    assert effective["TPU_KV_TRANSFER_NAMESPACE"] == f"cpu-test-{role}"
    assert all(effective[name] == value for name, value in factors.items())
    assert effective["VLLM_TPU_QWEN38_FLASH_PLE_HOST_BACKEND"] == (
        "torch" if role == "prefill" else "native_dma"
    )
    if role == "decode":
        assert "MOE_FUSED_EP_KERNEL_IMPL" not in effective
        assert effective["USE_MOE_FUSED_EP_KERNEL"] == "0"
        assert effective["TPU_MOE_DENSE_EXPERT_THRESHOLD"] == "128"
    else:
        assert effective["MOE_FUSED_EP_KERNEL_IMPL"] == "v2"
        assert effective["MOE_FUSED_EP_V2_SHARDED_PLAN"] == "1"


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_original_qwen35_engine_recipe_is_preserved(role, tmp_path):
    env = stubs(tmp_path)
    env.pop("DISAGG_PROFILE")
    subprocess.run(
        ["bash", str(KUBE / "disagg_engine.sh"), role],
        env=env,
        capture_output=True,
        check=True,
    )
    receipt = json.loads((tmp_path / "vllm.jsonl").read_text())
    args, effective = receipt["argv"], receipt["env"]
    assert flag(args, "--model") == "Qwen/Qwen3.5-397B-A17B-FP8"
    assert flag(args, "--block-size") == "8192"
    assert flag(args, "--max-model-len") == "66560"
    assert "--no-async-scheduling" in args
    assert effective["TPU_RAIDEN_QWEN35_ADMISSION"] == "1"
    assert effective["TPU_RAIDEN_TRANSFER_PARALLELISM"] == "8"
    transfer = json.loads(flag(args, "--kv-transfer-config"))
    assert transfer["kv_connector"] == "TPURaidenConnector"
    assert json.loads(flag(args, "--compilation-config"))["compile_sizes"] == (
        [1024, 2048, 4096, 5120] if role == "prefill" else [8, 16, 32, 4352, 4384]
    )
    assert flag(args, "--prefill-context-parallel-size") == (
        "8" if role == "prefill" else "1"
    )


@pytest.mark.parametrize("benchmark_rc", [0, 7])
def test_actual_benchmark_accounting_and_client_failure_cleanup(benchmark_rc, tmp_path):
    env = stubs(tmp_path)
    env.update(BQ_UPLOAD="1", BUILDKITE_BRANCH="main", BUILDKITE_PULL_REQUEST="false")
    env["STUB_VLLM_RC"] = str(benchmark_rc)
    env["DISAGG_DP_SIZE"] = "4"
    result = subprocess.run(
        ["bash", str(KUBE / "disagg_benchmark.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == benchmark_rc, (result.stdout, result.stderr)
    assert not (tmp_path / "pip-called").exists()
    pid = int((tmp_path / "proxy.pid").read_text())
    assert not Path(f"/proc/{pid}").exists(), "Owned proxy child leaked"
    if benchmark_rc == 0:
        assert len((tmp_path / "bq-called").read_text().splitlines()) == 4
        assert len((tmp_path / "artifacts.jsonl").read_text().splitlines()) == 4
        rows = [
            json.loads(line)
            for line in (tmp_path / "vllm.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 4
        assert flag(rows[0]["argv"], "--max-concurrency") == "8"
        assert flag(rows[0]["argv"], "--random-input-len") == "8192"
        assert flag(rows[0]["argv"], "--random-output-len") == "32"
        assert flag(rows[0]["argv"], "--num-prompts") == "8"
        for row in rows[1:]:
            assert flag(row["argv"], "--max-concurrency") == "32"
            assert flag(row["argv"], "--random-input-len") == "32768"
            assert flag(row["argv"], "--random-output-len") == "1024"
    else:
        assert not (tmp_path / "bq-called").exists()
        assert len((tmp_path / "vllm.jsonl").read_text().splitlines()) == 1
        assert len((tmp_path / "artifacts.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("profile", ["qwen3.5-397b", "qwen3.8-flash"])
def test_failed_benchmark_preserves_raw_artifact_and_original_status(profile, tmp_path):
    env = stubs(tmp_path)
    env.update(
        DISAGG_PROFILE=profile,
        STUB_VLLM_RC="7",
        STUB_ARTIFACT_RC="9",
        STUB_RESULT_MODE="failed",
        BQ_UPLOAD="1",
        BUILDKITE_BRANCH="main",
        BUILDKITE_PULL_REQUEST="false",
    )
    result = subprocess.run(
        ["bash", str(KUBE / "disagg_benchmark.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 7, (result.stdout, result.stderr)
    assert len((tmp_path / "artifacts.jsonl").read_text().splitlines()) == 1
    assert not (tmp_path / "bq-called").exists()
    raw = list((tmp_path / "results").glob("*.json"))
    assert len(raw) == 1 and json.loads(raw[0].read_text())["failed"] == 1
    pid = int((tmp_path / "proxy.pid").read_text())
    assert not Path(f"/proc/{pid}").exists()


def test_manifest_and_matrix_use_same_eight_rank_roles():
    manifest = yaml.safe_load(
        (KUBE / "manifests/workloads/qwen3.8-flash-1p1d.yaml").read_text()
    )
    roles = manifest["spec"]["replicatedJobs"]
    assert [role["name"] for role in roles] == ["prefill", "decode", "benchmark"]
    assert manifest["spec"]["successPolicy"]["targetReplicatedJobs"] == ["benchmark"]
    for role in roles[:2]:
        pod = role["template"]["spec"]["template"]["spec"]
        container = pod["containers"][0]
        assert container["command"][-1] == role["name"]
        assert container["resources"]["limits"]["google.com/tpu"] == "4"
        assert (
            dict((entry["name"], entry.get("value")) for entry in container["env"])[
                "DISAGG_PROFILE"
            ]
            == "qwen3.8-flash"
        )
    assert {item["name"] for item in roles[2]["dependsOn"]} == {"prefill", "decode"}
    pipeline = yaml.safe_load((ROOT / ".buildkite/pipeline_perf.yml").read_text())

    def steps(items):
        for item in items:
            if isinstance(item, dict):
                yield item
                yield from steps(item.get("steps", []))

    lane = [
        step
        for step in steps(pipeline["steps"])
        if "Qwen3.8-Flash 1P1D" in step.get("label", "")
    ]
    assert len(lane) == 1 and lane[0]["matrix"]["setup"]["dp"] == ["8"]
    assert "DISAGG_DP_SIZE" in (KUBE / "common.sh").read_text()


@pytest.mark.parametrize("dp", [1, 3, 16])
def test_invalid_topology_never_falls_back(dp):
    with pytest.raises(ValueError):
        PROFILE.role_profile("prefill", dp, "model", "host")


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_engine_child_failure_is_not_retried_or_hidden(role, tmp_path):
    env = stubs(tmp_path)
    env["STUB_VLLM_RC"] = "29"
    result = subprocess.run(
        ["bash", str(KUBE / "disagg_engine.sh"), role],
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 29
    assert len((tmp_path / "vllm.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "failure,expected", [("STUB_VLLM_RC", 29), ("STUB_PROXY_RC", 1)]
)
def test_benchmark_or_proxy_failure_cleans_up_owned_child(failure, expected, tmp_path):
    env = stubs(tmp_path)
    env[failure] = "29"
    result = subprocess.run(
        ["bash", str(KUBE / "disagg_benchmark.sh")],
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == expected
    pid = int((tmp_path / "proxy.pid").read_text())
    assert not Path(f"/proc/{pid}").exists()


def test_unknown_profile_is_rejected_before_any_child(tmp_path):
    env = stubs(tmp_path)
    env["DISAGG_PROFILE"] = "unknown"
    result = subprocess.run(
        ["bash", str(KUBE / "disagg_engine.sh"), "prefill"],
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert not (tmp_path / "vllm.jsonl").exists()


def test_partial_request_failure_keeps_all_completed_and_failed_receipts(
    tmp_path, monkeypatch
):
    def post(url, body, rank):
        if url.endswith("/tokenize"):
            return dict(tokens=[41, rank + 42])
        if rank == 1:
            raise ValueError("rank request failed")
        return completion(len(body["prompt"]))

    monkeypatch.setattr(SMOKE, "post", post)
    with pytest.raises(ValueError, match="rank request failed"):
        SMOKE.run_checks("proxy", "prefill", "decode", "model", 4, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed"] == 3 and summary["failed"] == 1
    assert summary["unattempted"] == summary["expected"] - 4
    assert len(list(tmp_path.glob("*-reference.json"))) == 4


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_actual_config_hook_resolves_role_environment_before_admission(
    role, monkeypatch
):
    from types import SimpleNamespace as NS

    from tests.models.vllm.test_qwen4_exp_parallel import config
    from vllm_torchtpu.models.vllm.qwen4_exp.config import (
        Qwen4ExpEngramConfig,
        Qwen4ExpForCausalLMConfig,
        TPUQwen4ExpConfig,
    )

    profile = PROFILE.role_profile(role, 8, "model", "host")
    cfg = config()
    cfg.additional_config = json.loads(flag(profile["argv"], "--additional-config"))
    cfg.additional_config["qwen4_exp"]["ple_host_backend"] = (
        "native_dma" if role == "prefill" else "torch"
    )
    cfg.compilation_config = NS(splitting_ops=None)
    cfg.engram_config = None
    cfg.instance_id = "cpu-recipe"
    cfg.model_config.model = "model"
    cfg.model_config.revision = None
    monkeypatch.setenv(
        "VLLM_TPU_QWEN38_FLASH_PLE_HOST_BACKEND",
        profile["env"]["VLLM_TPU_QWEN38_FLASH_PLE_HOST_BACKEND"],
    )
    monkeypatch.setattr(
        Qwen4ExpForCausalLMConfig, "verify_and_update_config", lambda _: None
    )
    TPUQwen4ExpConfig.verify_and_update_config(cfg)
    assert cfg.additional_config["qwen4_exp"]["ple_host_backend"] == (
        "torch" if role == "prefill" else "native_dma"
    )
    assert cfg.compilation_config.splitting_ops == ["qwen4_exp::host_lookup"]
    if role == "prefill":
        engram = cfg.engram_config
        assert isinstance(engram, Qwen4ExpEngramConfig)
        assert engram.dp_shared_memory
        assert engram.host_launch is not None
        assert engram.host_launch.participants == 8


def completion(length=129, token=11):
    return dict(
        choices=[dict(token_ids=[token] * 16, finish_reason="length")],
        usage=dict(
            prompt_tokens=length,
            completion_tokens=16,
            total_tokens=length + 16,
            prompt_tokens_details=dict(cached_tokens=length - 1),
        ),
    )


def test_exact_repeat_and_reference_gates_reject_mutation_and_bad_accounting():
    import copy

    reference = completion()
    SMOKE.check_triplet(reference, completion(), completion(), 129)
    with pytest.raises(ValueError, match="Repeated"):
        SMOKE.check_triplet(reference, completion(), completion(token=12), 129)
    with pytest.raises(ValueError, match="reference"):
        SMOKE.check_triplet(reference, completion(token=12), completion(token=12), 129)
    broken = copy.deepcopy(reference)
    broken["usage"]["prompt_tokens"] = 128
    with pytest.raises(ValueError, match="accounting"):
        SMOKE.completion_ids(broken, 129)
    broken = copy.deepcopy(reference)
    broken["choices"][0].pop("token_ids")
    with pytest.raises(ValueError, match="every generated"):
        SMOKE.completion_ids(broken, 129)


@pytest.mark.parametrize("dp", [4, 8])
def test_actual_smoke_runner_preserves_prompts_routing_and_failure_receipts(
    dp, tmp_path, monkeypatch
):
    calls = []

    def post(url, body, rank):
        calls.append((url, body.copy(), rank))
        if url.endswith("/tokenize"):
            return dict(tokens=[41, 42 + rank, 43])
        return completion(len(body["prompt"]))

    monkeypatch.setattr(SMOKE, "post", post)
    summary = SMOKE.run_checks("proxy", "prefill", "decode", "model", dp, tmp_path)
    assert summary["passed"] and summary["completed"] == dp * len(SMOKE.LENGTHS) * 3
    rows = [row for row in calls if row[0].endswith("/v1/completions")]
    assert {row[2] for row in rows} == set(range(dp))
    for rank in range(dp):
        for length in SMOKE.LENGTHS:
            group = [
                body
                for _, body, worker in rows
                if worker == rank and len(body["prompt"]) == length
            ]
            assert len(group) == 3
            assert group[0]["prompt"] == group[1]["prompt"] == group[2]["prompt"]
            assert length + group[0]["max_tokens"] <= 33792
    monkeypatch.setattr(
        SMOKE, "post", lambda *_: (_ for _ in ()).throw(ValueError("transport failed"))
    )
    with pytest.raises(ValueError, match="transport failed"):
        SMOKE.run_checks(
            "proxy", "prefill", "decode", "model", dp, tmp_path / "failure"
        )
    failed = json.loads((tmp_path / "failure/summary.json").read_text())
    assert not failed["passed"] and failed["errors"] and failed["completed"] == 0


def test_proxy_rejects_missing_handle_before_decode_dispatch(monkeypatch):
    import asyncio
    from types import SimpleNamespace as NS

    proxy = load("pd_proxy_gate", ROOT / "examples/disagg/toy_proxy_server.py")
    monkeypatch.setattr(
        proxy, "global_args", NS(require_kv_transfer=True), raising=False
    )
    monkeypatch.setattr(proxy, "get_next_client", lambda *_: {})

    async def prefill(*_):
        return NS(json=lambda: dict(choices=[{}]))

    async def body():
        return dict(prompt=[41], model="model")

    monkeypatch.setattr(proxy, "send_request_to_prefill", prefill)
    request = NS(json=body, headers={}, app=NS(state=NS()))
    with pytest.raises(ValueError, match="transfer handle"):
        asyncio.run(proxy._handle_completions("/v1/completions", request))
    proxy.require_transfer_handle({}, False)
    proxy.require_transfer_handle(dict(uuid=123), True)


@pytest.mark.parametrize("mode", ["missing", "short", "failed"])
def test_qwen38_benchmark_requires_complete_result_even_when_child_exits_zero(
    mode, tmp_path
):
    env = stubs(tmp_path)
    env["STUB_RESULT_MODE"] = mode
    env.update(BQ_UPLOAD="1", BUILDKITE_BRANCH="main", BUILDKITE_PULL_REQUEST="false")
    result = subprocess.run(
        ["bash", str(KUBE / "disagg_benchmark.sh")],
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert len((tmp_path / "vllm.jsonl").read_text().splitlines()) == 1
    pid = int((tmp_path / "proxy.pid").read_text())
    assert not Path(f"/proc/{pid}").exists()
    assert not (tmp_path / "bq-called").exists()
    if mode != "missing":
        assert len(list((tmp_path / "results").glob("*.json"))) == 1
        uploads = [
            json.loads(line)
            for line in (tmp_path / "artifacts.jsonl").read_text().splitlines()
        ]
        assert len(uploads) == 1  # The invalid raw smoke cell.


def test_nonempty_transfer_handle_with_local_recompute_cannot_pass():
    proxy = load("pd_proxy_gate", ROOT / "examples/disagg/toy_proxy_server.py")
    proxy.require_transfer_handle(dict(uuid=123), True)
    local = completion()
    local["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
    with pytest.raises(ValueError, match="complete remote prefix"):
        SMOKE.check_triplet(local, local, local, 129)
    # The reference is deliberately decode-local; only the PD outputs must consume KV.
    SMOKE.check_triplet(local, completion(), completion(), 129)


@pytest.mark.parametrize("numeric_mode", ["report", "strict"])
@pytest.mark.parametrize("difference", ["reference", "repeat"])
def test_numeric_policy_preserves_failure_separately(
    numeric_mode, difference, tmp_path, monkeypatch
):
    monkeypatch.setattr(SMOKE, "LENGTHS", (129,))

    def post(url, body, rank):
        if url.endswith("/tokenize"):
            return dict(tokens=[41, 42 + rank])
        token = 11 if url.startswith("decode") else 12
        if difference == "repeat" and body["request_id"].endswith("repeat"):
            token = 13
        return completion(len(body["prompt"]), token=token)

    monkeypatch.setattr(SMOKE, "post", post)
    if numeric_mode == "strict":
        with pytest.raises(ValueError, match="reference|Repeated"):
            SMOKE.run_checks(
                "proxy", "prefill", "decode", "model", 4, tmp_path, numeric_mode
            )
    else:
        summary = SMOKE.run_checks("proxy", "prefill", "decode", "model", 4, tmp_path)
        assert summary["performance_checks_passed"]
        assert summary["completed"] == summary["expected"] == 12
        assert len(summary["numeric_diagnostics"]) == 4
    recorded = json.loads((tmp_path / "summary.json").read_text())
    assert not recorded["passed"]
    assert recorded["numeric_reference_passed"] is False
    assert recorded["numeric_repeat_passed"] == (difference != "repeat")
    assert recorded["diagnostic_failed"]
    assert recorded["performance_checks_passed"] == (numeric_mode == "report")
    assert len(json.loads((tmp_path / "results.json").read_text())["requests"]) == 12


@pytest.mark.parametrize("numeric_mode", ["report", "strict"])
@pytest.mark.parametrize("failure", ["HTTP", "tokens", "usage", "remote"])
def test_transport_gates_fail_in_both_numeric_modes(
    numeric_mode, failure, tmp_path, monkeypatch
):
    monkeypatch.setattr(SMOKE, "LENGTHS", (129,))

    def post(url, body, rank):
        if url.endswith("/tokenize"):
            return dict(tokens=[41, 42 + rank])
        if failure == "HTTP":
            raise ValueError("HTTP failed")
        result = completion(len(body["prompt"]))
        if failure == "tokens":
            result["choices"][0].pop("token_ids")
        elif failure == "usage":
            result["usage"]["total_tokens"] += 1
        elif failure == "remote" and url.startswith("proxy"):
            result["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
        return result

    monkeypatch.setattr(SMOKE, "post", post)
    with pytest.raises(ValueError):
        SMOKE.run_checks(
            "proxy", "prefill", "decode", "model", 4, tmp_path, numeric_mode
        )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert not summary["performance_checks_passed"] and summary["errors"]


def test_default_performance_lane_skips_optional_numeric_matrix(tmp_path):
    env = stubs(tmp_path)
    env["STUB_NUMERIC_FAILURE"] = "1"
    env["STUB_SANITY_RC"] = "7"
    result = subprocess.run(
        ["bash", str(KUBE / "disagg_benchmark.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert not (tmp_path / "results/sanity").exists()
    rows = [json.loads(x) for x in (tmp_path / "vllm.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert flag(rows[0]["argv"], "--num-prompts") == "8"
    assert all(flag(row["argv"], "--num-prompts") == "512" for row in rows[1:])
    assert not Path(f"/proc/{(tmp_path / 'proxy.pid').read_text()}").exists()


@pytest.mark.parametrize("dp", [8, 4])
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_profile_destination_geometry_reaches_producer_packet(dp, role, monkeypatch):
    from unittest.mock import patch

    from vllm import SamplingParams
    from vllm.v1.request import Request, RequestStatus

    from tests.distributed.test_tpu_connector import _make_raiden_scheduler

    profile = PROFILE.role_profile(role, dp, "model", "127.0.0.1")
    monkeypatch.delenv("TPU_RAIDEN_DST_SHARDS", raising=False)
    for key, value in profile["env"].items():
        monkeypatch.setenv(key, value)
    producer = _make_raiden_scheduler(
        is_producer=True, block_size=128, dp_size=dp, tp_size=profile["tp"]
    )
    request = Request(
        "profile-destination", [1] * 8193, SamplingParams(max_tokens=1), None
    )
    request.num_computed_tokens = 8193
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    with patch(
        "vllm_torchtpu.distributed.kv_transfer.tpu_connector."
        "tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS",
        "127.0.0.1:27000",
    ):
        delayed, params = producer.request_finished(request, list(range(64)))
    assert delayed
    assert params is not None
    assert params["num_tokens"] == 8192
    assert params["dst_tp_size"] == profile["tp"]

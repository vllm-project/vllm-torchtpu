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
"""The Qwen3.5-397B C3 recipe renders exactly the validation entries' legs.

configs/qwen3.5-397b-fp8-pd.sh claims each of its legs is a leg of a measured
record in models/qwen3_5/validation/ (the 1P1D gates, derived from points p01
and p08). These tests render every role, decode layout and trade-off arm
through the per-pod launcher's dry run and compare the serve arguments, the
--additional-config sections and the environment with the entries, allowing
only the variables listed here with the reason each is not in the entries.
They also hold the CI wiring to the recipe: every gated eval has a baseline
file, a pending baseline is visible on the CI step (which then lets only the
eval verdict warn), and the manifest asks for
the chips the instantiation serves.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
from typing import Any

import pytest
import yaml

from vllm_torchtpu import envs

REPO = pathlib.Path(__file__).resolve().parents[3]
VALIDATION = REPO / "src" / "vllm_torchtpu" / "models" / "qwen3_5" / "validation"
CONFIGS = REPO / "scripts" / "vllm" / "benchmarking" / "configs"
BASELINES = REPO / "scripts" / "vllm" / "benchmarking" / "baselines"
LAUNCHER = REPO / ".buildkite" / "kubernetes" / "qwen35_pd.sh"
MANIFESTS = REPO / ".buildkite" / "kubernetes" / "manifests" / "workloads"
PIPELINE = REPO / ".buildkite" / "pipeline_integration.yml"

RECIPE = "qwen3.5-397b-fp8-pd"
# Each CI instantiation of the recipe, with the manifest that serves it.
INSTANTIATIONS = {"qwen3.5-397b-fp8-pd-1p1d": "qwen3.5-397b-pd-1p1d.yaml"}
GATES = {"dp4tp2": "gate-dp4tp2-1p1d", "tp8dcp4": "gate-tp8-dcp4-1p1d"}
EXPERT_TABLES = {
    "dp4tp2": "expert_placement_dp8_ep8.json",
    "tp8dcp4": "expert_placement_tp8_ep8_agentx_corr_c32.json",
}
ARMS = {"on": "tradeoffs_on", "off": "tradeoffs_off"}
# The entries' $MODEL and $SERVED_MODEL: the checkpoint they were measured on.
MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
SERVED_MODEL = "Qwen3.5-397B-A17B-FP8"
CHIPS_PER_ENGINE = 4
POD_IP = "10.0.0.7"

_LIBTPU = (
    "--xla_tpu_use_dynamic_smem_negotiation=true --xla_tpu_dvfs_p_state=7 "
    "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864 "
    "--xla_tpu_use_minor_sharding_for_major_trivial_input=true "
    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true "
    "--xla_tpu_ars_combiner_threshold_in_bytes=0 "
    "--xla_tpu_enable_async_collective_merger=false --xla_enable_async_all_gather=true "
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false "
    "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=8388608"
)

# Variables the records' runs set that the entries cannot carry, because they
# are not this package's (envs.py) variables. Value per (role, layout); None
# means the leg does not set it. The config's header names each one.
RECIPE_ONLY: dict[str, Any] = {
    # The seq-along-lane KV layout the shared 128-token kernel page applies to.
    "VLLM_KV_CACHE_LAYOUT": "HND",
    # PP8 over 60 layers, as the records partitioned them.
    "VLLM_PP_LAYER_PARTITION": {"prefill": "8,8,8,8,8,8,8,4", "decode": None},
    "LIBTPU_INIT_ARGS": {
        ("prefill", "dp4tp2"): _LIBTPU
        + " --xla_tpu_rematerialization_min_size_in_bytes=1099511627776",
        ("prefill", "tp8dcp4"): _LIBTPU,
        "decode": _LIBTPU,
    },
    # The offloading connector's Raiden store (prefill legs only).
    "RAIDEN_EXPECTED_WORKERS_TIMEOUT_S": {"prefill": "3600", "decode": None},
    "RAIDEN_NUMA_POLICY": {"prefill": "preferred", "decode": None},
    "RUNAI_STREAMER_MEMORY_LIMIT": "5368709120",
    # torch_tpu's; the TP8 + DCP4 decode legs set it off.
    "TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS": {
        ("decode", "tp8dcp4"): "0",
        "default": None,
    },
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "300",
    "VLLM_TARGET_DEVICE": "tpu",
    "VLLM_DP_MASTER_IP": {"prefill": "127.0.0.1", "decode": None},
    "OMP_NUM_THREADS": "8",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "VLLM_LOGGING_LEVEL": "INFO",
    "TPU_PREMAPPED_BUFFER_SIZE": "17179869184",
    # An envs.py variable (default zmq) the entries leave to the deployment
    # with the Raiden ports: their generator classed it as per-pod wiring.
    "TPU_KV_RESHARD_TRANSPORT": "raiden",
    # A departure: CI boots both legs from a cold compile cache.
    "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
}
# RECIPE_ONLY names that are envs.py variables all the same, with the reason
# the entries do not carry them.
PACKAGE_VARIABLES_LEFT_TO_THE_DEPLOYMENT = {"TPU_KV_RESHARD_TRANSPORT"}

# Per pod, so the entries do not record them: the one-host pod shape, the
# Raiden ports and identities, and the address this pod advertises.
WIRING: dict[str, Any] = {
    "JAX_PLATFORMS": "tpu,cpu",
    "PJRT_DEVICE": "TPU",
    "TPU_ACCELERATOR_TYPE": "tpu7x",
    "TPU_BACKEND_TYPE": "jax",
    "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
    "TPU_HOST_BOUNDS": "1,1,1",
    "TPU_TOPOLOGY": "2x2x1",
    "MEGASCALE_COORDINATOR_ADDRESS": "",
    "MEGASCALE_NUM_SLICES": "",
    "MEGASCALE_PORT": "",
    "MEGASCALE_SLICE_ID": "",
    "TPU_PROCESS_ADDRESSES": "",
    "TPU_PROCESS_PORT": "",
    "TPU_WORKER_HOSTNAMES": "",
    "TPU_KV_TRANSFER_PORT": "9100",
    "TPU_SIDE_CHANNEL_PORT": "9600",
    "TPU_KV_TRANSFER_NAMESPACE": {"prefill": "prefill", "decode": "decode"},
    "TPU_RAIDEN_ENGINE_ID": {"prefill": "prefill-engine", "decode": "decode-engine"},
    "TPU_RAIDEN_RESHARD_PORT_BASE": {"prefill": "27000", "decode": "28000"},
    "TPU_RAIDEN_STORE_DISPATCH_PORT_BASE": {"prefill": "27100", "decode": "28100"},
    "VLLM_HOST_IP": POD_IP,
    "TPU_RAIDEN_ADVERTISE_HOST": POD_IP,
    "PD_ROLE": {"prefill": "prefill", "decode": "decode"},
    "PYTHONUNBUFFERED": "1",
}


def _expected(table: dict[str, Any], role: str, layout: str) -> dict[str, str]:
    out = {}
    for name, value in table.items():
        if isinstance(value, dict):
            value = value.get((role, layout), value.get(role, value.get("default")))
        if value is not None:
            out[name] = value
    return out


def _load(path: pathlib.Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _point(number: int) -> dict:
    for path in sorted(VALIDATION.glob("p[0-9][0-9]-*.yaml")):
        entry = _load(path)
        if entry["point"] == number:
            return entry
    raise AssertionError(f"no validation entry for point {number}")


def _merge(base: dict, over: dict) -> dict:
    """The entries' deep merge of an arm's sections over a leg's (README)."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict):
            out[key] = _merge(out.get(key, {}), value)
        else:
            out[key] = value
    return out


def _value(text: str) -> Any:
    return json.loads(text) if text[:1] in "{[" else text


def _flags(args: list[str]) -> tuple[list[str], dict[str, list[Any]]]:
    """Positional arguments and {flag: values}, JSON values parsed, refusing a
    flag given twice (the last would win silently)."""
    positional: list[str] = []
    flags: dict[str, list[Any]] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("--"):
            positional.append(arg)
            i += 1
            continue
        if "=" in arg:
            name, value = arg.split("=", 1)
            values = [value]
            i += 1
        else:
            name, values = arg, []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                values.append(args[i])
                i += 1
        assert name not in flags, f"{name} given twice"
        flags[name] = [_value(v) for v in values]
    return positional, flags


def _render(
    role: str, config: str = RECIPE, tmp_path: pathlib.Path | None = None, **params: str
) -> dict[str, Any]:
    """The launcher's dry run for one role: serve arguments, the environment
    the recipe and the launcher set, and (prefill head) proxy and checks."""
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path or "/tmp"),
        "QWEN35_PD_DRY_RUN": "1",
        "QWEN35_PD_CONFIG": config,
        "POD_IP": POD_IP,
        "PREFILL_HOSTS": "prefill-0",
        "DECODE_HOSTS": "decode-0",
        "REPO_DIR": str(REPO),
        **params,
    }
    proc = subprocess.run(
        ["bash", str(LAUNCHER), role],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        tag, _, rest = line.partition(" ")
        if tag == "[serve-args]":
            out["serve_args"] = json.loads(rest)
        elif tag == "[env]":
            out["env"] = json.loads(rest)
        elif tag == "[proxy]":
            out["proxy"] = shlex.split(rest)
        elif tag == "[correctness]":
            out["correctness"] = shlex.split(rest)
    return out


def _expected_leg(layout: str, tradeoffs: str, role: str):
    gate = _load(VALIDATION / f"{GATES[layout]}.yaml")
    leg = _point(gate["base_point"])["legs"][role]
    table = str(VALIDATION / EXPERT_TABLES[layout])

    def sub(text: str) -> str:
        return (
            text.replace("$MODEL", MODEL)
            .replace("$SERVED_MODEL", SERVED_MODEL)
            .replace("$EXPERT_PLACEMENT_TABLE", table)
        )

    positional, flags = _flags([sub(a) for a in leg["serve_args"]])
    for name, value in gate["legs"][role].get("serve_args_override", {}).items():
        flags[name] = [_value(value)]
    arm = ARMS[tradeoffs]
    for name, value in gate.get("arm_serve_args_override", {}).get(arm, {}).items():
        flags[name] = [_value(value)]
    sections = _merge(leg["additional_config"], gate["arms"][arm]["additional_config"])
    env = {name: sub(value) for name, value in leg["env"].items()}
    return positional, flags, sections, env


# (config, role, layout, trade-off arm, parameters passed): the recipe with
# each parameter value, and each CI instantiation as the nightly serves it,
# with nothing passed, against the gate entry and arm it claims.
LEGS = [
    (RECIPE, role, layout, tradeoffs, True)
    for layout in GATES
    for tradeoffs in ARMS
    for role in ("prefill", "decode")
] + [
    ("qwen3.5-397b-fp8-pd-1p1d", role, "dp4tp2", "on", False)
    for role in ("prefill", "decode")
]


@pytest.mark.parametrize(
    "config,role,layout,tradeoffs,pass_params",
    LEGS,
    ids=[f"{c}-{r}-{lay}-{t}" for c, r, lay, t, _ in LEGS],
)
def test_recipe_renders_the_gate_entries_legs(
    config, role, layout, tradeoffs, pass_params, tmp_path
):
    params = {"PD_DECODE_LAYOUT": layout, "PD_TRADEOFFS": tradeoffs}
    rendered = _render(role, config, tmp_path, **(params if pass_params else {}))
    positional, flags = _flags(rendered["serve_args"])
    sections = flags.pop("--additional-config")
    exp_positional, exp_flags, exp_sections, exp_env = _expected_leg(
        layout, tradeoffs, role
    )

    assert positional == exp_positional
    assert flags == exp_flags
    assert sections == [exp_sections]

    env = rendered["env"]
    wrong = {k: (env.get(k), v) for k, v in exp_env.items() if env.get(k) != v}
    assert not wrong, f"entry env not rendered as recorded: {wrong}"
    extra = {k: v for k, v in env.items() if k not in exp_env}
    allowed = {
        **_expected(RECIPE_ONLY, role, layout),
        **_expected(WIRING, role, layout),
    }
    unexplained = {k: v for k, v in extra.items() if k not in allowed}
    missing = {k: v for k, v in allowed.items() if k not in extra}
    changed = {
        k: (extra[k], v) for k, v in allowed.items() if k in extra and extra[k] != v
    }
    assert not unexplained, (
        f"set but not in the entries or the lists here: {unexplained}"
    )
    assert not missing, f"listed here but not set: {missing}"
    assert not changed, f"(rendered, expected): {changed}"


def test_benchmark_mtp_renders_the_points_synthetic_acceptance(tmp_path):
    """PD_MTP_SYNTHETIC_AL gives back the points' own decode speculative
    config (the records' benchmark setting); the gates override it."""
    for layout, point in (("dp4tp2", 1), ("tp8dcp4", 8)):
        rendered = _render(
            "decode",
            tmp_path=tmp_path,
            PD_DECODE_LAYOUT=layout,
            PD_MTP_SYNTHETIC_AL="3.39",
        )
        _, flags = _flags(rendered["serve_args"])
        _, point_flags = _flags(_point(point)["legs"]["decode"]["serve_args"])
        assert flags["--speculative-config"] == point_flags["--speculative-config"]


def test_recipe_only_variables_are_not_package_variables_and_are_documented():
    """The entries list only envs.py variables, so a variable the recipe adds
    beyond them must be one the entries could not carry, and the config's
    header names it."""
    text = (CONFIGS / f"{RECIPE}.sh").read_text()
    header = text[: text.index("\nPD_ROLE=")]
    for name in RECIPE_ONLY:
        declared = name in envs.environment_variables
        assert declared == (name in PACKAGE_VARIABLES_LEFT_TO_THE_DEPLOYMENT), name
        assert name in header, f"{name} is not in the config's header"


@pytest.mark.parametrize(
    "params,message",
    [
        ({"PD_DECODE_LAYOUT": "dp8"}, "PD_DECODE_LAYOUT must be dp4tp2 or tp8dcp4"),
        ({"PD_TRADEOFFS": "maybe"}, "PD_TRADEOFFS must be on or off"),
        ({"PD_PROXY": "llmd"}, "PD_PROXY must be router or toy"),
    ],
)
def test_unknown_parameter_values_are_refused(params, message, tmp_path):
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "QWEN35_PD_DRY_RUN": "1",
        "QWEN35_PD_CONFIG": RECIPE,
        "POD_IP": POD_IP,
        "PREFILL_HOSTS": "prefill-0",
        "DECODE_HOSTS": "decode-0",
        "REPO_DIR": str(REPO),
        **params,
    }
    proc = subprocess.run(
        ["bash", str(LAUNCHER), "prefill"], env=env, capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert message in proc.stderr


def _config_values(config: str, *names: str) -> dict[str, str]:
    script = f'PD_ROLE=prefill; source "{CONFIGS / config}.sh" >&2 || exit 9; '
    script += " ".join(f'printf "%s\\0" "${{{n}[*]}}";' for n in names)
    proc = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return dict(zip(names, proc.stdout.split("\0")))


def _check_regression():
    path = REPO / "scripts" / "vllm" / "benchmarking" / "check_regression.py"
    spec = importlib.util.spec_from_file_location("recipe_check_regression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pipeline_step(manifest: str) -> dict:
    doc = _load(PIPELINE)
    for group in doc["steps"]:
        for step in group.get("steps", []):
            if f"--manifest {manifest}" in " ".join(step.get("commands", [])):
                return step
    raise AssertionError(f"no pipeline_integration.yml step runs {manifest}")


@pytest.mark.parametrize("config,manifest", INSTANTIATIONS.items())
def test_gated_evals_have_baselines_and_pending_ones_show_on_the_step(config, manifest):
    values = _config_values(
        config,
        "LM_EVAL_TASKS",
        "PD_EVAL_FLOW_ARGS",
        "LM_EVAL_GEN_KWARGS",
        "PD_DECODE_LAYOUT",
        "EXTRA_LM_EVAL_MODEL_ARGS",
    )
    flow_args = values["PD_EVAL_FLOW_ARGS"].split()
    # One attempt per request: the proxy does not retry, so a failed hop must
    # fail its request rather than be resent by the client (last key wins).
    model_args = dict(
        a.split("=", 1) for a in values["EXTRA_LM_EVAL_MODEL_ARGS"].split(",")
    )
    assert model_args["max_retries"] == "1"
    # The stages ask for lm-eval themselves; the config cannot turn it off.
    stages = (
        REPO / "scripts" / "vllm" / "integration" / "run_qwen35_397b_pd_correctness.sh"
    ).read_text()
    assert re.search(
        r"run_eval_flow\.sh \\\n\s+--config \"\$\{CONFIG\}\" --run-lm-eval ", stages
    )
    # The gate's whole split, and no silently skipped baseline.
    assert flow_args[flow_args.index("--lm-eval-limit") + 1] == "none"
    assert "--require-baselines" in flow_args
    gate = _load(VALIDATION / f"{GATES[values['PD_DECODE_LAYOUT']]}.yaml")
    gsm8k = gate["suite"]["gsm8k"]
    assert "gsm8k" in values["LM_EVAL_TASKS"].split()
    gen_kwargs = json.loads(values["LM_EVAL_GEN_KWARGS"])
    assert gen_kwargs["chat_template_kwargs"] == gsm8k["chat_template_kwargs"]

    check_regression = _check_regression()
    baselines = [
        BASELINES / "eval" / f"{config}.{t}.baseline.json"
        for t in values["LM_EVAL_TASKS"].split()
    ]
    perf = [
        BASELINES / "perf" / f"{config}.baseline.json",
        BASELINES / "perf" / "v7x" / f"{config}.baseline.json",
    ]
    assert sum(p.exists() for p in perf) == 1, f"one perf baseline for {config}: {perf}"
    baselines += [p for p in perf if p.exists()]

    pending = []
    for path in baselines:
        assert path.exists(), (
            f"{path.name} is missing: the gate would have nothing to check"
        )
        body = json.loads(path.read_text())
        refusal = check_regression.baseline_refusal(body)
        if "_pending" in body:
            assert isinstance(body["_pending"], str) and body["_pending"].strip()
            assert refusal and "pending" in refusal
            pending.append(path.name)
            continue
        assert refusal is None, f"{path.name}: {refusal}"
        source = body.get("_source", {})
        assert source.get("url") and source.get("description"), path.name
        if path.parent.name == "eval" or path.parent.parent.name == "eval":
            task = path.name.split(".")[-3]
            metric = check_regression.primary_metric(body[task])
            assert metric is not None
            assert check_regression.primary_stderr(body[task], metric.name) is not None

    step = _pipeline_step(manifest)
    # Never the whole step: boot, page, transfer and MTP stay hard.
    assert "soft_fail" not in step
    if pending:
        assert step.get("env", {}).get("EVAL_SOFT_FAIL") == "1", (
            f"pending baselines {pending}: the step must set EVAL_SOFT_FAIL=1"
        )
        assert "(baselines pending)" in step["label"]
    else:
        assert "pending" not in step["label"]


@pytest.mark.parametrize("config,manifest", INSTANTIATIONS.items())
def test_manifest_serves_the_instantiation_on_its_chips(config, manifest, tmp_path):
    values = _config_values(
        config, "PD_PREFILL_REPLICAS", "PD_DECODE_REPLICAS", "PD_DECODE_LAYOUT"
    )
    replicas = {
        "prefill": int(values["PD_PREFILL_REPLICAS"]),
        "decode": int(values["PD_DECODE_REPLICAS"]),
    }
    doc = _load(MANIFESTS / manifest)
    jobs = {job["name"]: job for job in doc["spec"]["replicatedJobs"]}
    assert set(jobs) == {"prefill", "decode"}
    chips = 0
    for role, job in jobs.items():
        spec = job["template"]["spec"]
        (container,) = spec["template"]["spec"]["containers"]
        assert container["name"] == "workload"
        assert container["command"][-2:] == [
            "/workspace/vllm-torchtpu/.buildkite/kubernetes/qwen35_pd.sh",
            role,
        ]
        env = {e["name"]: e.get("value") for e in container["env"]}
        assert env["QWEN35_PD_CONFIG"] == config
        engines = job["replicas"] * spec["parallelism"]
        assert engines == replicas[role]
        assert len(env["PREFILL_HOSTS"].split()) == replicas["prefill"]
        assert len(env["DECODE_HOSTS"].split()) == replicas["decode"]
        tpus = int(container["resources"]["requests"]["google.com/tpu"])
        assert tpus == CHIPS_PER_ENGINE
        chips += engines * tpus
        # Kubernetes waits on the port the role's engine serves.
        _, flags = _flags(_render(role, config, tmp_path)["serve_args"])
        assert container["readinessProbe"]["httpGet"]["port"] == int(flags["--port"][0])
    gate = _load(VALIDATION / f"{GATES[values['PD_DECODE_LAYOUT']]}.yaml")
    assert chips == CHIPS_PER_ENGINE * sum(replicas.values()) == gate["chips"]
    assert doc["spec"]["successPolicy"]["targetReplicatedJobs"] == ["prefill"]
    step = _pipeline_step(manifest)
    assert step["if"] == 'build.env("NIGHTLY") == "1"'


@pytest.mark.parametrize("proxy", ["router", "toy"])
def test_prefill_head_fronts_every_decode_server_and_runs_the_checks(proxy, tmp_path):
    rendered = _render("prefill", "qwen3.5-397b-fp8-pd-1p1d", tmp_path, PD_PROXY=proxy)
    command = rendered["proxy"]
    decode = [f"decode-0:{port}" for port in (7200, 7201, 7202, 7203)]
    if proxy == "router":
        assert command[:2] == ["vllm-router", "--host"]
        assert "--vllm-pd-disaggregation" in command
        assert "--disable-retries" in command
        targets = [command[i + 1] for i, a in enumerate(command) if a == "--decode"]
        assert targets == [f"http://{d}" for d in decode]
    else:
        assert command[1] == "examples/disagg/toy_proxy_server.py"
        hosts = command[
            command.index("--decoder-hosts") + 1 : command.index("--decoder-ports")
        ]
        ports = command[command.index("--decoder-ports") + 1 :]
        assert [f"{h}:{p}" for h, p in zip(hosts, ports)] == decode
    correctness = rendered["correctness"]
    assert (
        correctness[-1] == "scripts/vllm/integration/run_qwen35_397b_pd_correctness.sh"
    )
    env = dict(
        a.split("=", 1) for a in correctness if "=" in a and not a.startswith("scripts")
    )
    assert env["QWEN35_PD_CONFIG"] == "qwen3.5-397b-fp8-pd-1p1d"
    assert env["DECODE_URLS"].split() == [f"http://{d}" for d in decode]
    assert env["DECODE_LEG_STATUS"].split() == ["http://decode-0:8300/leg_status.json"]

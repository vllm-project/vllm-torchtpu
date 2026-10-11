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
"""Derived plumbing: the closed list P-01..P-22 (R-PLAN-2).

The tool derives a value only when it has one correct answer: addresses,
ports, DP ranks, KV and Raiden wiring, the backend's env. It never adds an
engine flag or env var because of some other value; a choice belongs in
scenarios/defaults.yaml. Each item is one function, registered with its
id and rule; the reference lists the registry.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from scenario.model.server import Hardware, Kv, ModelSpec, RoleView, Router
from scenario.model.tasks import AgentxTask, EvalTask, InferencexIdentity
from scenario.plan.argv import compact_json, render_args
from scenario.plan.types import (
    API_PORT,
    DP_RPC_PORT,
    HEALTH_PATH,
    KV_PORT,
    KV_TRANSFER_PORT,
    RAY_PORT,
    READINESS_MARGIN_S,
    ROUTER_PORT,
    SIDE_CHANNEL_PORT,
    Backend,
    Command,
    KvPlan,
    Readiness,
    Ref,
    RouterPlan,
    Value,
    concat,
)
from scenario.reserved import (
    AGENTX_CLIENT_ENV,
    INFERENCEX_EVAL_ENV,
    INFERENCEX_IDENTITY_ENV,
    INFERENCEX_PERF_ENV,
    READY_TIMEOUT_ENV,
    custom_task_env,
    role_env_suffix,
)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class Item:
    id: str
    title: str
    rule: str
    env: tuple[str, ...]


ITEMS: dict[str, Item] = {}


def plumbing(
    item_id: str, title: str, rule: str, env: Sequence[str] = ()
) -> Callable[[F], F]:
    """Register a derivation under its P-id; `env` lists the names it sets."""

    def register(function: F) -> F:
        ITEMS[item_id] = Item(item_id, title, rule, tuple(env))
        return function

    return register


@plumbing(
    "P-01",
    "hardware",
    "The role's accelerator, topology, hosts and chips_per_host as written (on "
    "server for one role); the tool looks nothing up.",
)
def role_hardware(role: RoleView) -> Hardware:
    return role.hardware


@plumbing(
    "P-02",
    "backend",
    "single when the role has one host, else server.multihost (mp or ray).",
)
def backend(hosts: int, multihost: str | None) -> Backend:
    if hosts == 1:
        return "single"
    if multihost == "mp":
        return "mp"
    if multihost == "ray":
        return "ray"
    raise ValueError(f"a role has {hosts} hosts and no server.multihost")


SERVE_PROGRAM = ("vllm", "serve")


@plumbing(
    "P-03",
    "serve argv",
    "vllm serve <model.path> --served-model-name <model.name>, the role's args "
    "in order, then the wiring flags, and --port 8000 on the role's first host, "
    "which serves the API.",
)
def serve_argv(
    model: ModelSpec, args: Mapping[str, Any], wiring: Sequence[Value], api: bool
) -> tuple[Value, ...]:
    port = ("--port", str(API_PORT)) if api else ()
    return (
        *SERVE_PROGRAM,
        model.load_path,
        "--served-model-name",
        model.name,
        *render_args(args),
        *wiring,
        *port,
    )


@plumbing(
    "P-04",
    "mp DP wiring",
    "Every host of an mp role with several hosts: --data-parallel-size-local "
    "<DP / hosts> --data-parallel-address <head_ip> --data-parallel-rpc-port 29500.",
)
def mp_dp_wiring(dp: int, hosts: int) -> tuple[int, tuple[Value, ...]]:
    local = dp // hosts
    flags: tuple[Value, ...] = (
        "--data-parallel-size-local",
        str(local),
        "--data-parallel-address",
        Ref("head_ip"),
        "--data-parallel-rpc-port",
        str(DP_RPC_PORT),
    )
    return local, flags


# Flags of the API frontend, which a headless host does not run; vLLM rejects
# them together with --headless.
FRONTEND_ONLY_ARGS = ("api-server-count",)


@plumbing(
    "P-05",
    "mp headless hosts",
    "Hosts after the first of an mp role add --headless "
    "--data-parallel-start-rank <index x DP local>, and leave off --port and "
    "the frontend-only flags (api-server-count), which vLLM rejects there.",
)
def headless(
    index: int, dp_local: int, args: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Args without the frontend flags, the headless flags, and what was dropped."""
    kept = {k: v for k, v in args.items() if k not in FRONTEND_ONLY_ARGS}
    dropped = tuple(f"--{k}" for k in args if k in FRONTEND_ONLY_ARGS)
    flags = ("--headless", "--data-parallel-start-rank", str(index * dp_local))
    return kept, flags, (*dropped, "--port")


# What GKE sets on every pod of a TPU slice. A run that builds its own slice
# (mp) or pins one host (a single-host role among several) must not see it.
EMPTIED_SLICE_ENV = (
    "MEGASCALE_COORDINATOR_ADDRESS",
    "MEGASCALE_NUM_SLICES",
    "MEGASCALE_PORT",
    "MEGASCALE_SLICE_ID",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "TPU_WORKER_HOSTNAMES",
)


@plumbing(
    "P-06",
    "backend and host env",
    "An mp role over several hosts: the seven slice names GKE injects set "
    "empty, and TPU_MULTIHOST_BACKEND=mp. A ray role over several hosts: "
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1, TPU_MULTIHOST_BACKEND=ray and "
    "JAX_PLATFORMS empty. A single-host role of a multi-role plan: the slice "
    "names set empty, TPU_HOST_BOUNDS=1,1,1 and TPU_TOPOLOGY=<topology>. Every "
    "host of a role with several hosts: TPU_SKIP_MDS_QUERY=1. Every host, and "
    "the driver's tasks: PYTHONUNBUFFERED=1. The driver's tasks also get "
    "TPU_MULTIHOST_BACKEND of the driver's role.",
    env=(
        *EMPTIED_SLICE_ENV,
        "TPU_MULTIHOST_BACKEND",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "JAX_PLATFORMS",
        "TPU_HOST_BOUNDS",
        "TPU_TOPOLOGY",
        "TPU_SKIP_MDS_QUERY",
        "PYTHONUNBUFFERED",
    ),
)
def backend_env(
    backend: Backend, hosts: int, topology: str, multi_role: bool
) -> dict[str, Value]:
    env: dict[str, Value] = {"PYTHONUNBUFFERED": "1"}
    if backend == "mp":
        env |= dict.fromkeys(EMPTIED_SLICE_ENV, "")
        env["TPU_MULTIHOST_BACKEND"] = "mp"
    elif backend == "ray":
        env["RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS"] = "1"
        env["TPU_MULTIHOST_BACKEND"] = "ray"
        env["JAX_PLATFORMS"] = ""
    elif multi_role:
        env |= dict.fromkeys(EMPTIED_SLICE_ENV, "")
        env["TPU_HOST_BOUNDS"] = "1,1,1"
        env["TPU_TOPOLOGY"] = topology
    if hosts > 1:
        env["TPU_SKIP_MDS_QUERY"] = "1"
    return env


@plumbing(
    "P-07",
    "ray bring-up",
    "The first host of a ray role runs ray start --head --port=6379 before "
    "serve; the others run ray start --address=<head_ip>:6379 and serve nothing.",
)
def ray_start(index: int) -> Command:
    if index == 0:
        return Command(("ray", "start", "--head", f"--port={RAY_PORT}"))
    return Command(
        ("ray", "start", concat("--address=", Ref("head_ip"), f":{RAY_PORT}"))
    )


@plumbing(
    "P-08",
    "advertise address",
    "VLLM_HOST_IP=<self_ip> on every host of a role with several hosts or with kv.",
    env=("VLLM_HOST_IP",),
)
def advertise_env(hosts: int, kv: bool) -> dict[str, Value]:
    return {"VLLM_HOST_IP": Ref("self_ip")} if hosts > 1 or kv else {}


@plumbing(
    "P-09",
    "startup timeout",
    f"{READY_TIMEOUT_ENV}=<server.startup_timeout> on every host; the readiness "
    f"wait, GET {HEALTH_PATH} on port {API_PORT}, lasts startup_timeout + "
    f"{READINESS_MARGIN_S} s from the server's start.",
    env=(READY_TIMEOUT_ENV,),
)
def startup(timeout_s: int) -> tuple[dict[str, Value], Readiness]:
    readiness = Readiness(HEALTH_PATH, API_PORT, timeout_s + READINESS_MARGIN_S)
    return {READY_TIMEOUT_ENV: str(timeout_s)}, readiness


@plumbing(
    "P-10",
    "accelerator env",
    "TPU_ACCELERATOR_TYPE=<accelerator> on every TPU host.",
    env=("TPU_ACCELERATOR_TYPE",),
)
def accelerator_env(accelerator: str) -> dict[str, Value]:
    return {"TPU_ACCELERATOR_TYPE": accelerator}


KV_CONNECTOR_MODULES = {
    "TPURaidenConnector": "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
}
KvRole = Literal["kv_producer", "kv_consumer", "kv_both"]
KV_ROLES: Mapping[str, KvRole] = {
    "producer": "kv_producer",
    "consumer": "kv_consumer",
    "both": "kv_both",
}


@plumbing(
    "P-11",
    "KV wiring",
    "A role with kv: --kv-transfer-config {kv_connector, "
    f"kv_connector_module_path, kv_role, kv_port: {KV_PORT}}}, "
    f"TPU_KV_TRANSFER_PORT={KV_TRANSFER_PORT} and "
    f"TPU_SIDE_CHANNEL_PORT={SIDE_CHANNEL_PORT}.",
    env=("TPU_KV_TRANSFER_PORT", "TPU_SIDE_CHANNEL_PORT"),
)
def kv_wiring(kv: Kv) -> tuple[KvPlan, tuple[Value, ...], dict[str, Value]]:
    kv_role = KV_ROLES[kv.role]
    config = {
        "kv_connector": kv.connector,
        "kv_connector_module_path": KV_CONNECTOR_MODULES[kv.connector],
        "kv_role": kv_role,
        "kv_port": KV_PORT,
    }
    env: dict[str, Value] = {
        "TPU_KV_TRANSFER_PORT": str(KV_TRANSFER_PORT),
        "TPU_SIDE_CHANNEL_PORT": str(SIDE_CHANNEL_PORT),
    }
    plan = KvPlan(kv.connector, kv_role, KV_PORT)
    return plan, ("--kv-transfer-config", compact_json(config)), env


RAIDEN_PORT_BASE = 27000
RAIDEN_DISPATCH_PORT_BASE = 27100
RAIDEN_PORT_STRIDE = 1000


@plumbing(
    "P-12",
    "Raiden identity",
    "A role with kv at position r among the roles: "
    "TPU_KV_TRANSFER_NAMESPACE=<role>, TPU_RAIDEN_ENGINE_ID=<role>-engine, "
    "TPU_RAIDEN_JOB_NAME=<role>, TPU_RAIDEN_RESHARD_PORT_BASE=27000+1000r, "
    "TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100+1000r and "
    "TPU_RAIDEN_ADVERTISE_HOST=<role_ips:role>, every host's IPv4 in DP-rank "
    "order (one address on a single host; the store-mode connector requires it).",
    env=(
        "TPU_KV_TRANSFER_NAMESPACE",
        "TPU_RAIDEN_ENGINE_ID",
        "TPU_RAIDEN_JOB_NAME",
        "TPU_RAIDEN_RESHARD_PORT_BASE",
        "TPU_RAIDEN_STORE_DISPATCH_PORT_BASE",
        "TPU_RAIDEN_ADVERTISE_HOST",
    ),
)
def raiden_identity(role: str, position: int) -> dict[str, Value]:
    offset = RAIDEN_PORT_STRIDE * position
    return {
        "TPU_KV_TRANSFER_NAMESPACE": role,
        "TPU_RAIDEN_ENGINE_ID": f"{role}-engine",
        "TPU_RAIDEN_JOB_NAME": role,
        "TPU_RAIDEN_RESHARD_PORT_BASE": str(RAIDEN_PORT_BASE + offset),
        "TPU_RAIDEN_STORE_DISPATCH_PORT_BASE": str(RAIDEN_DISPATCH_PORT_BASE + offset),
        "TPU_RAIDEN_ADVERTISE_HOST": Ref("role_ips", role=role),
    }


ROUTER_HEALTH_PATH = "/healthcheck"


@plumbing(
    "P-13",
    "router",
    "On the driver host: <router.command> --host 0.0.0.0 --port 8100 "
    "--prefiller-hosts <role_head:producer> --prefiller-ports 8000 "
    "--decoder-hosts <role_head:consumer> --decoder-ports 8000, then "
    f"router.args; ready on GET {ROUTER_HEALTH_PATH}.",
)
def router_plan(router: Router, producer: str, consumer: str) -> RouterPlan:
    argv = (
        *router.command,
        "--host",
        "0.0.0.0",
        "--port",
        str(ROUTER_PORT),
        "--prefiller-hosts",
        Ref("role_head", role=producer),
        "--prefiller-ports",
        str(API_PORT),
        "--decoder-hosts",
        Ref("role_head", role=consumer),
        "--decoder-ports",
        str(API_PORT),
        *render_args(router.args),
    )
    return RouterPlan(argv, ROUTER_PORT, ROUTER_HEALTH_PATH)


LOOPBACK = "127.0.0.1"


@plumbing(
    "P-14",
    "driver and endpoint",
    "One role: its first host drives. Several roles: tasks.driver, a role's "
    "first host or a chipless client host. Tasks connect to the router "
    "(127.0.0.1:8100) when there is one, else to the driver role's own API "
    "(127.0.0.1:8000), or from a client host to the first role's head.",
)
def endpoint(
    roles: Sequence[str], driver_role: str | None, router: bool
) -> tuple[Value, int]:
    if router:
        return LOOPBACK, ROUTER_PORT
    if driver_role is not None:
        return LOOPBACK, API_PORT
    return Ref("role_head", role=roles[0]), API_PORT


@plumbing(
    "P-15",
    "client tokenizer",
    "--tokenizer <tasks.perf.tokenizer> when set; else, for an object-storage "
    "model.path, --tokenizer <pulled_model_dir>, which falls back to "
    "model.name when the pulled directory holds no config.json; else none.",
)
def client_tokenizer(tokenizer: str | None, model: ModelSpec) -> tuple[Value, ...]:
    if tokenizer is not None:
        return ("--tokenizer", tokenizer)
    if model.in_object_storage:
        return ("--tokenizer", Ref("pulled_model_dir", fallback=model.name))
    return ()


@plumbing(
    "P-16",
    "result paths",
    "Under <results_dir>: <cell>.json per perf cell (the label in cells mode), "
    "<cell>.warmup<i>.json per warmup pass, lm_eval --output_path "
    "<results_dir> (<results_dir>/<task>_c<N> in a concurrency sweep), "
    "agentx_c<N> for AgentX, and <results_dir>/inferencex_eval for InferenceX's "
    "eval client.",
)
def result_path(name: str) -> Value:
    return concat(Ref("results_dir"), f"/{name}")


@plumbing(
    "P-17",
    "custom-task env",
    "Every custom task gets SCENARIO_SERVED_MODEL, SCENARIO_RESULTS_DIR "
    "(<results_dir>) and SCENARIO_API_PORT=8000; with one producer and one "
    "consumer role PREFILL_URL and DECODE_URL (http://<role_head>:8000); with a "
    "router PROXY_URL=http://127.0.0.1:8100 and SCENARIO_ROUTER_PORT=8100; with "
    "several roles SCENARIO_HOST_<ROLE>=<role_head:role>; on a role's host "
    "SCENARIO_SERVER_LOG. Its env values may use them as ${NAME}; $$ is a "
    "literal $.",
    env=(
        "SCENARIO_SERVED_MODEL",
        "SCENARIO_RESULTS_DIR",
        "SCENARIO_API_PORT",
        "PREFILL_URL",
        "DECODE_URL",
        "PROXY_URL",
        "SCENARIO_ROUTER_PORT",
        "SCENARIO_SERVER_LOG",
    ),
)
def custom_env(
    model: ModelSpec,
    roles: Sequence[str],
    kv_roles: Mapping[str, str],
    router: bool,
    driver_on_role: bool,
) -> dict[str, Value]:
    """The names custom tasks get; the same set V-22 lets ${NAME} use."""
    names = custom_task_env(
        roles=roles, kv_roles=kv_roles, router=router, driver_on_role=driver_on_role
    )
    producers = [role for role, kv in kv_roles.items() if kv == "producer"]
    consumers = [role for role, kv in kv_roles.items() if kv == "consumer"]
    values: dict[str, Value] = {
        "SCENARIO_SERVED_MODEL": model.name,
        "SCENARIO_RESULTS_DIR": Ref("results_dir"),
        "SCENARIO_API_PORT": str(API_PORT),
        "PROXY_URL": f"http://{LOOPBACK}:{ROUTER_PORT}",
        "SCENARIO_ROUTER_PORT": str(ROUTER_PORT),
        "SCENARIO_SERVER_LOG": Ref("server_log"),
    }
    for name, found in (("PREFILL_URL", producers), ("DECODE_URL", consumers)):
        if len(found) == 1:
            values[name] = concat(
                "http://", Ref("role_head", role=found[0]), f":{API_PORT}"
            )
    for role in roles:
        values[f"SCENARIO_HOST_{role_env_suffix(role)}"] = Ref("role_head", role=role)
    return {name: values[name] for name in sorted(names)}


@plumbing(
    "P-18",
    "chipless driver env",
    "A chipless client driver's tasks and router get VLLM_PLUGINS empty and "
    "TORCH_DEVICE_BACKEND_AUTOLOAD=0, so vLLM's CLI from the TPU image does not "
    "start a TPU platform on a host without chips.",
    env=("VLLM_PLUGINS", "TORCH_DEVICE_BACKEND_AUTOLOAD"),
)
def chipless_env() -> dict[str, Value]:
    return {"VLLM_PLUGINS": "", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}


WARM_MODULE = "scenario.runtime.warm"
WARM_TIMEOUT_S = 600


@plumbing(
    "P-19",
    "remote-code warm",
    f"Before serve on mp hosts after the first, when trust-remote-code is set "
    f"and model.path is object storage: python3 -m {WARM_MODULE} <model.path>, "
    f"best effort, at most {WARM_TIMEOUT_S} s, so the hosts do not race to write "
    "the model's remote code.",
)
def remote_code_warm(
    backend: Backend, index: int, args: Mapping[str, Any], model: ModelSpec
) -> tuple[Command, ...]:
    trusted = args.get("trust-remote-code") is True
    if backend != "mp" or index == 0 or not trusted or not model.in_object_storage:
        return ()
    argv = ("python3", "-m", WARM_MODULE, model.load_path)
    return (Command(argv, timeout_s=WARM_TIMEOUT_S, best_effort=True),)


@plumbing(
    "P-20",
    "chips",
    "The chips of the server roles, the sum of hosts x chips_per_host (never "
    "TensorCores); InferenceX divides AgentX throughput by them (TP, or "
    "PREFILL_TP and DECODE_TP per worker for P/D).",
)
def chips(roles: Sequence[Hardware]) -> int:
    return sum(role.chips for role in roles)


# Where the image bakes the pinned InferenceX checkout and the AgentX client's
# own interpreter. AgentX must run under that interpreter: any other makes it
# delete the baked environment and reinstall it from the network. The eval
# client runs under it too.
INFERENCEX_DIR = "/opt/infx/InferenceX/inferencex-e2e"
AIPERF_RUNTIME_DIR = "/opt/infx/runtime"
AGENTX_PYTHON = f"{AIPERF_RUNTIME_DIR}/venv/bin/python"

# AgentX inputs with one value for a vLLM server: Dynamo's routing switches
# off. IS_AGENTIC makes a bad value fail before the replay instead of after it.
AGENTX_FIXED_ENV: Mapping[str, str] = {
    "EVAL_ONLY": "false",
    "IS_AGENTIC": "1",
    "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": "3600",
    "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "0",
    "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "false",
}

# Eval inputs with one value: the eval runs alone on its deployment, and the
# servers take no API key (InferenceX's workflows send EMPTY too).
INFERENCEX_EVAL_FIXED_ENV: Mapping[str, str] = {
    "EVAL_ONLY": "true",
    "OPENAI_API_KEY": "EMPTY",
}
INFERENCEX_EVAL_DIR = "inferencex_eval"


@plumbing(
    "P-21",
    "InferenceX client env",
    "The InferenceX perf client gets PYTHONPATH=<baked checkout>. The AgentX "
    "client gets its baked runtime (AIPERF_RUNTIME_DIR, PYTHONPATH, "
    "PYTHONSAFEPATH), the endpoint (AIPERF_SERVER_URL), "
    "SERVED_MODEL_NAME=<model.name>, CONC and DURATION, RESULT_DIR and "
    "AGENTIC_OUTPUT_DIR (<results_dir>), RESULT_FILENAME=agentx_c<N>, "
    "AIPERF_EXPERIMENTAL_FAST from tasks.agentx.fast (1 or 0), EVAL_ONLY=false, "
    "IS_AGENTIC=1 and the three Dynamo switches off. For several roles it also "
    "gets every role head's /metrics in AIPERF_SERVER_METRICS_URLS, since a "
    "router re-exports no engine metrics, and power measurement off. "
    "InferenceX's eval client gets PYTHONPATH and "
    "PYTHONSAFEPATH, MODEL_NAME=<model.name>, EVAL_SUITE from tasks.eval.suite, "
    "EVAL_ONLY=true and OPENAI_API_KEY=EMPTY. Both get the identity of P-22.",
    env=tuple(
        name
        for name in dict.fromkeys(
            [*AGENTX_CLIENT_ENV, *INFERENCEX_EVAL_ENV, *INFERENCEX_PERF_ENV]
        )
        # --framework carries the framework; EVAL_FRAMEWORK is only reserved.
        if name not in INFERENCEX_IDENTITY_ENV and name != "EVAL_FRAMEWORK"
    ),
)
def agentx_env(
    task: AgentxTask,
    identity: InferencexIdentity,
    model: ModelSpec,
    roles: Sequence[RoleView],
    url: Value,
) -> dict[str, Value]:
    env: dict[str, Value] = {
        "AIPERF_RUNTIME_DIR": AIPERF_RUNTIME_DIR,
        "PYTHONPATH": INFERENCEX_DIR,
        "PYTHONSAFEPATH": "1",
        "AIPERF_SERVER_URL": url,
        "SERVED_MODEL_NAME": model.name,
        "CONC": str(task.concurrency),
        "DURATION": str(task.duration),
        "RESULT_DIR": Ref("results_dir"),
        "AGENTIC_OUTPUT_DIR": Ref("results_dir"),
        "RESULT_FILENAME": f"agentx_c{task.concurrency}",
        "AIPERF_EXPERIMENTAL_FAST": "1" if task.fast else "0",
        **AGENTX_FIXED_ENV,
        **inferencex_identity(identity, model, roles),
    }
    if len(roles) > 1:
        metrics: list[Value] = []
        for role in roles:
            if metrics:
                metrics.append(",")
            metrics += ["http://", Ref("role_head", role.name), f":{API_PORT}/metrics"]
        env |= {
            "AIPERF_SERVER_METRICS_URLS": concat(*metrics),
            "ENABLE_AGENTX_POWER": "0",
            "REQUIRE_POWER": "0",
        }
    return env


def inferencex_eval_env(
    task: EvalTask,
    identity: InferencexIdentity,
    model: ModelSpec,
    roles: Sequence[RoleView],
) -> dict[str, Value]:
    """InferenceX's eval client env (P-21), with the identity of P-22."""
    env: dict[str, Value] = {
        "PYTHONPATH": INFERENCEX_DIR,
        "PYTHONSAFEPATH": "1",
        "MODEL_NAME": model.name,
        **INFERENCEX_EVAL_FIXED_ENV,
        **inferencex_identity(identity, model, roles),
    }
    if task.suite is not None:
        env["EVAL_SUITE"] = task.suite
    return env


def inferencex_perf_env() -> dict[str, Value]:
    return {"PYTHONPATH": INFERENCEX_DIR}


def inferencex_scores_env() -> dict[str, Value]:
    """The env of InferenceX's score validation: its own checkout."""
    return {"PYTHONPATH": INFERENCEX_DIR, "PYTHONSAFEPATH": "1"}


def _expert_parallel_size(role: RoleView) -> str:
    """EP in chips: the role's chips with enable-expert-parallel, else 1."""
    if role.args.get("enable-expert-parallel") is True:
        return str(role.hardware.chips)
    return "1"


def _dp_attention(role: RoleView) -> str:
    """InferenceX's DP attention flag for vLLM: data-parallel-size above 1."""
    size = role.args.get("data-parallel-size", 1)
    above = isinstance(size, int) and not isinstance(size, bool) and size > 1
    return "true" if above else "false"


@plumbing(
    "P-22",
    "InferenceX identity",
    "What InferenceX files an AgentX row and the evals of the same deployment "
    "under, set alike on both clients so they join: MODEL=<model.name>, "
    "tasks.inferencex (MODEL_PREFIX, PRECISION, FRAMEWORK, SPEC_DECODING, "
    "KV_OFFLOADING, and when set KV_OFFLOAD_BACKEND with its JSON metadata, "
    "TOTAL_CPU_DRAM_GB, ROUTER_METADATA as JSON and KV_P2P_TRANSFER), "
    "RUNNER_TYPE=<accelerator> and IMAGE=<the image the pods run>. Sizes are "
    "chips (P-20). DP attention is data-parallel-size above 1, as InferenceX "
    "reads vLLM. One role: IS_MULTINODE=false, TP=<chips>, EP_SIZE (<chips> "
    "with enable-expert-parallel, else 1), PP_SIZE, DCP_SIZE and PCP_SIZE 1, "
    "DP_ATTENTION. P/D: IS_MULTINODE=true, DISAGG=true, and for the producer "
    "and the consumer roles PREFILL_ and DECODE_ NUM_WORKERS, TP (one worker's "
    "chips), EP (likewise) and DP_ATTN.",
    env=INFERENCEX_IDENTITY_ENV,
)
def inferencex_identity(
    identity: InferencexIdentity, model: ModelSpec, roles: Sequence[RoleView]
) -> dict[str, Value]:
    env: dict[str, Value] = {
        "MODEL": model.name,
        "MODEL_PREFIX": identity.model_prefix,
        "PRECISION": identity.precision,
        "FRAMEWORK": identity.framework,
        "SPEC_DECODING": identity.spec_decoding,
        "KV_OFFLOADING": identity.kv_offloading,
        "RUNNER_TYPE": roles[0].hardware.accelerator,
        "IMAGE": Ref("image"),
    }
    backend = identity.kv_offload_backend
    if backend is not None:
        env["KV_OFFLOAD_BACKEND"] = backend.name
        env["KV_OFFLOAD_BACKEND_METADATA"] = compact_json(
            backend.model_dump(exclude_none=True)
        )
    if identity.total_cpu_dram_gb is not None:
        env["TOTAL_CPU_DRAM_GB"] = str(identity.total_cpu_dram_gb)
    if identity.router is not None:
        env["ROUTER_METADATA"] = compact_json(identity.router.model_dump())
    if identity.kv_p2p_transfer is not None:
        env["KV_P2P_TRANSFER"] = identity.kv_p2p_transfer
    if len(roles) == 1:
        env |= {
            "IS_MULTINODE": "false",
            "TP": str(chips([role.hardware for role in roles])),
            "EP_SIZE": _expert_parallel_size(roles[0]),
            "PP_SIZE": "1",
            "DCP_SIZE": "1",
            "PCP_SIZE": "1",
            "DP_ATTENTION": _dp_attention(roles[0]),
        }
        return env
    env |= {"IS_MULTINODE": "true", "DISAGG": "true"}
    for prefix, kv_role in (("PREFILL", "producer"), ("DECODE", "consumer")):
        workers = [role for role in roles if role.kv and role.kv.role == kv_role]
        shapes = {
            (role.hardware.chips, _expert_parallel_size(role), _dp_attention(role))
            for role in workers
        }
        if len(shapes) != 1:
            found = ", ".join(f"{r.name} ({r.hardware.chips} chips)" for r in workers)
            raise ValueError(
                f"InferenceX's identity of several roles needs {kv_role} roles of "
                f"one shape; found {found or 'none'}"
            )
        ((tp, ep, dp_attention),) = shapes
        env[f"{prefix}_NUM_WORKERS"] = str(len(workers))
        env[f"{prefix}_TP"] = str(tp)
        env[f"{prefix}_EP"] = ep
        env[f"{prefix}_DP_ATTN"] = dp_attention
    return env

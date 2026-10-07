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
"""Env names and flags the tool owns, so a scenario may not set them.

The tool sets these from other fields. With two owners of one name,
precedence would decide the value, so a scenario setting one fails instead.
Each name maps to what the tool sets it from; that is what the error says.

The models check scenarios against these names, plan() sets them, and
targets check their own env against them, so this module imports nothing
from the rest of the package.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

READY_TIMEOUT_ENV = "VLLM_ENGINE_READY_TIMEOUT_S"

TOOL_ENV_PREFIXES = ("SCENARIO_",)

_WIRING = "multi-host and backend wiring"
_KV = "KV transfer wiring"
_CUSTOM = "the custom-task env contract"

TOOL_ENV: Mapping[str, str] = MappingProxyType(
    {
        READY_TIMEOUT_ENV: "server.startup_timeout",
        "VLLM_HOST_IP": "the host's own address",
        "TPU_ACCELERATOR_TYPE": "the slice's accelerator",
        "TPU_MULTIHOST_BACKEND": _WIRING,
        "MEGASCALE_COORDINATOR_ADDRESS": _WIRING,
        "MEGASCALE_NUM_SLICES": _WIRING,
        "MEGASCALE_PORT": _WIRING,
        "MEGASCALE_SLICE_ID": _WIRING,
        "TPU_PROCESS_ADDRESSES": _WIRING,
        "TPU_PROCESS_PORT": _WIRING,
        "TPU_WORKER_HOSTNAMES": _WIRING,
        "TPU_HOST_BOUNDS": _WIRING,
        "TPU_TOPOLOGY": _WIRING,
        "TPU_SKIP_MDS_QUERY": _WIRING,
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": _WIRING,
        "PYTHONUNBUFFERED": "the runtime, which streams child output",
        "TPU_KV_TRANSFER_NAMESPACE": _KV,
        "TPU_RAIDEN_ENGINE_ID": _KV,
        "TPU_RAIDEN_JOB_NAME": _KV,
        "TPU_RAIDEN_RESHARD_PORT_BASE": _KV,
        "TPU_RAIDEN_STORE_DISPATCH_PORT_BASE": _KV,
        "TPU_RAIDEN_ADVERTISE_HOST": _KV,
        "TPU_KV_TRANSFER_PORT": _KV,
        "TPU_SIDE_CHANNEL_PORT": _KV,
        "PREFILL_URL": _CUSTOM,
        "DECODE_URL": _CUSTOM,
        "PROXY_URL": _CUSTOM,
    }
)

# Set on the engines of a role that runs on the ray backend.
RAY_ENV: Mapping[str, str] = MappingProxyType({"JAX_PLATFORMS": "the ray backend"})

# Set on tasks and the router when the driver is a chipless host, where vLLM's
# CLI from the TPU image must not initialize a TPU.
CHIPLESS_DRIVER_ENV: Mapping[str, str] = MappingProxyType(
    dict.fromkeys(
        ("VLLM_PLUGINS", "TORCH_DEVICE_BACKEND_AUTOLOAD", "PYTHONPATH"),
        "the chipless driver setup",
    )
)

_AGENTX_RUN = "the point and the results layout"
_AGENTX_IMAGE = "the InferenceX checkout baked into the image"
_AGENTX_ROLES = "the server roles and their chip counts"
_AGENTX_FIXED = "the tool, fixed for a vLLM server"

# Inputs of the InferenceX AgentX client that the tool sets per point; every
# other client setting is the scenario's pass-through env.
AGENTX_CLIENT_ENV: Mapping[str, str] = MappingProxyType(
    {
        "AIPERF_RUNTIME_DIR": _AGENTX_IMAGE,
        "PYTHONPATH": _AGENTX_IMAGE,
        "PYTHONSAFEPATH": _AGENTX_IMAGE,
        "AIPERF_SERVER_URL": "the tool (endpoint)",
        "SERVED_MODEL_NAME": "server.model.name",
        "MODEL": "server.model.name",
        "MODEL_PREFIX": "tasks.agentx.labels.model_prefix",
        "PRECISION": "tasks.agentx.labels.precision",
        "FRAMEWORK": "tasks.agentx.labels.framework",
        "SPEC_DECODING": "tasks.agentx.labels.spec_decoding",
        "CONC": "tasks.agentx.concurrency",
        "DURATION": "tasks.agentx.duration",
        "RESULT_DIR": _AGENTX_RUN,
        "RESULT_FILENAME": _AGENTX_RUN,
        "AGENTIC_OUTPUT_DIR": _AGENTX_RUN,
        "RUNNER_TYPE": "the slice's accelerator",
        "IS_MULTINODE": _AGENTX_ROLES,
        "TP": _AGENTX_ROLES,
        "PP_SIZE": _AGENTX_ROLES,
        "DCP_SIZE": _AGENTX_ROLES,
        "PCP_SIZE": _AGENTX_ROLES,
        "DISAGG": _AGENTX_ROLES,
        "PREFILL_NUM_WORKERS": _AGENTX_ROLES,
        "PREFILL_TP": _AGENTX_ROLES,
        "DECODE_NUM_WORKERS": _AGENTX_ROLES,
        "DECODE_TP": _AGENTX_ROLES,
        "ENABLE_AGENTX_POWER": _AGENTX_FIXED,
        "REQUIRE_POWER": _AGENTX_FIXED,
        "IS_AGENTIC": _AGENTX_FIXED,
        "EVAL_ONLY": _AGENTX_FIXED,
        "KV_OFFLOADING": _AGENTX_FIXED,
        "AIPERF_EXPERIMENTAL_FAST": _AGENTX_FIXED,
        "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": _AGENTX_FIXED,
        "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": _AGENTX_FIXED,
        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": _AGENTX_FIXED,
    }
)

# Env of the InferenceX perf client that the tool sets.
INFERENCEX_PERF_ENV: Mapping[str, str] = MappingProxyType({"PYTHONPATH": _AGENTX_IMAGE})


def tool_env_owner(name: str) -> str | None:
    """What the tool sets `name` from, or None when a scenario may set it."""
    if name in TOOL_ENV:
        return TOOL_ENV[name]
    if name.startswith(TOOL_ENV_PREFIXES):
        return "SCENARIO_* names are reserved"
    return None


def role_env_suffix(role: str) -> str:
    return role.upper().replace("-", "_")


def custom_task_env(
    *,
    roles: Sequence[str],
    kv_roles: Mapping[str, str],
    router: bool,
    driver_on_role: bool,
) -> frozenset[str]:
    """Names the tool provides to custom tasks, usable as ${NAME} in their env.

    Args:
        roles: role names; just `server` for a scenario without roles.
        kv_roles: role name -> kv role (producer, consumer, both).
        router: whether a router fronts the roles.
        driver_on_role: whether tasks run on a role's head, not a chipless host.
    """
    names = {"SCENARIO_SERVED_MODEL", "SCENARIO_RESULTS_DIR", "SCENARIO_API_PORT"}
    producers = [role for role, kv in kv_roles.items() if kv == "producer"]
    consumers = [role for role, kv in kv_roles.items() if kv == "consumer"]
    if len(producers) == 1 and len(consumers) == 1:
        names |= {"PREFILL_URL", "DECODE_URL"}
    if router:
        names |= {"PROXY_URL", "SCENARIO_ROUTER_PORT"}
    if len(roles) > 1:
        names |= {f"SCENARIO_HOST_{role_env_suffix(role)}" for role in roles}
    if driver_on_role:
        names.add("SCENARIO_SERVER_LOG")
    return frozenset(names)


SERVE_ARGS: Mapping[str, str] = MappingProxyType(
    {
        "model": "server.model.path",
        "served-model-name": "server.model.name",
        "port": "the tool (API port)",
        "host": "the tool (address binding)",
        "headless": "the tool (mp hosts after the first)",
        "data-parallel-size-local": "the tool (data-parallel-size over hosts)",
        "data-parallel-address": "the tool (head address)",
        "data-parallel-rpc-port": "the tool (DP wiring)",
        "data-parallel-start-rank": "the tool (host index)",
        "kv-transfer-config": "server.roles.<role>.kv",
    }
)

ROUTER_ARGS: Mapping[str, str] = MappingProxyType(
    dict.fromkeys(
        (
            "host",
            "port",
            "prefiller-hosts",
            "prefiller-host",
            "prefiller-ports",
            "prefiller-port",
            "decoder-hosts",
            "decoder-host",
            "decoder-ports",
            "decoder-port",
        ),
        "the tool (router wiring)",
    )
)

# Perf client flags that typed tasks.perf fields render.
PERF_CLIENT_ARGS: Mapping[str, str] = MappingProxyType(
    {
        "model": "server.model.name",
        "tokenizer": "tasks.perf.tokenizer",
        "host": "the tool (endpoint)",
        "port": "the tool (endpoint)",
        "base-url": "the tool (endpoint)",
        "dataset-name": "tasks.perf.dataset",
        "dataset-path": "tasks.perf.dataset",
        "random-input-len": "tasks.perf.isl_osl or cells",
        "random-output-len": "tasks.perf.isl_osl or cells",
        "sharegpt-output-len": "tasks.perf.isl_osl or cells",
        "random-range-ratio": "tasks.perf.random_range_ratio",
        "num-prompts": "tasks.perf.num_prompts",
        "num-warmups": "tasks.perf.num_warmups",
        "max-concurrency": "tasks.perf.concurrency or cells",
        "percentile-metrics": "tasks.perf.percentile_metrics",
        "temperature": "tasks.perf.temperature",
        "extra-body": "tasks.perf.extra_body",
        "label": "the label of tasks.perf.cells",
        "save-result": "the tool (result files)",
        "append-result": "the tool (result files)",
        "result-dir": "the tool (result files)",
        "result-filename": "the tool (result files)",
    }
)

# lm_eval --model_args keys the tool sets.
LM_EVAL_MODEL_ARGS: Mapping[str, str] = MappingProxyType(
    {
        "model": "server.model.name",
        "base_url": "the tool (the server or router URL)",
    }
)

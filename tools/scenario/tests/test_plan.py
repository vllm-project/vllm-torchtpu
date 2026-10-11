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
"""plan(): one test per plumbing item (P-01..P-22), tasks, gates, --only."""

import pathlib
import re
import textwrap
from typing import Any

import pytest
from plan_fixtures import Data, build
from scenario.errors import DiagnosticError
from scenario.plan.build import PlanError, plan, select_point
from scenario.plan.codec import to_json
from scenario.plan.env import env_rule
from scenario.plan.plumbing import ITEMS
from scenario.plan.types import (
    AgentxTaskPlan,
    Concat,
    CustomTaskPlan,
    EvalTaskPlan,
    PerfTaskPlan,
    Plan,
    Ref,
    concat,
)
from scenario.reserved import (
    AGENTX_CLIENT_ENV,
    CHIPLESS_DRIVER_ENV,
    INFERENCEX_EVAL_ENV,
    INFERENCEX_IDENTITY_ENV,
    INFERENCEX_PERF_ENV,
    RAY_ENV,
    custom_task_env,
    tool_env_owner,
)
from scenario.resolve import resolve

HEAD_IP = Ref("head_ip")
SELF_IP = Ref("self_ip")
RESULTS = Ref("results_dir")
GLM_PATH = "gs://tpu-commons-ci/glm/GLM-5.2-FP8"
GLM = "zai-org/GLM-5.2-FP8"


def env(items: Any) -> dict[str, Any]:
    return dict(items)


def task(plan_: Plan, kind: type) -> Any:
    (found,) = [t for t in plan_.driver.tasks if isinstance(t, kind)]
    return found


def test_registry_is_complete() -> None:
    assert sorted(ITEMS) == [f"P-{n:02d}" for n in range(1, 23)]
    assert all(item.title and item.rule for item in ITEMS.values())


def test_p01_hardware(variants: dict[str, Data]) -> None:
    (role,) = build(variants["single"]).roles
    assert (role.accelerator, role.topology) == ("tpu7x", "2x2x2")
    assert (role.hosts, role.chips_per_host, role.chips) == (2, 4, 8)
    assert len(role.host_plans) == 2


def test_p02_backend(variants: dict[str, Data]) -> None:
    assert build(variants["single"]).roles[0].backend == "ray"
    assert build(variants["mp"]).roles[0].backend == "mp"
    assert build(variants["single_host"]).roles[0].backend == "single"
    assert {r.backend for r in build(variants["pd"]).roles} == {"single"}


def test_p03_serve_argv(variants: dict[str, Data]) -> None:
    head = build(variants["single"]).roles[0].host_plans[0]
    assert head.serve is not None
    assert head.serve.argv == (
        "vllm",
        "serve",
        GLM_PATH,
        "--served-model-name",
        GLM,
        "--load-format",
        "runai_streamer",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "16",
        "--enable-expert-parallel",
        "--max-model-len",
        "9216",
        "--max-num-seqs",
        "64",
        "--api-server-count",
        "1",
        "--port",
        "8000",
    )


def test_p03_role_args_over_server_args(variants: dict[str, Data]) -> None:
    variants["pd"]["server"]["roles"]["decode"]["args"]["enable-expert-parallel"] = None
    prefill, decode = build(variants["pd"]).roles
    assert "--enable-expert-parallel" in prefill.host_plans[0].serve.argv
    assert "--enable-expert-parallel" not in decode.host_plans[0].serve.argv


def test_p04_mp_dp_wiring(variants: dict[str, Data]) -> None:
    wiring = (
        "--data-parallel-size-local",
        "8",
        "--data-parallel-address",
        HEAD_IP,
        "--data-parallel-rpc-port",
        "29500",
    )
    for host in build(variants["mp"]).roles[0].host_plans:
        argv = host.serve.argv
        start = argv.index("--data-parallel-size-local")
        assert argv[start : start + 6] == wiring


def test_p05_headless_hosts(variants: dict[str, Data]) -> None:
    head, worker = build(variants["mp"]).roles[0].host_plans
    assert head.serve.argv[-2:] == ("--port", "8000")
    assert "--api-server-count" in head.serve.argv
    assert worker.serve.argv[-3:] == ("--headless", "--data-parallel-start-rank", "8")
    assert "--port" not in worker.serve.argv
    assert "--api-server-count" not in worker.serve.argv
    assert worker.dropped_flags == ("--api-server-count", "--port")
    assert not worker.head and head.head


def test_p05_start_rank_per_host(variants: dict[str, Data]) -> None:
    prefill = build(variants["k3_pd"]).roles[0]
    ranks = [h.serve.argv[-1] for h in prefill.host_plans[1:]]
    assert ranks == ["1", "2", "3"]


def test_p06_backend_env(variants: dict[str, Data]) -> None:
    emptied = dict.fromkeys(
        [
            "MEGASCALE_COORDINATOR_ADDRESS",
            "MEGASCALE_NUM_SLICES",
            "MEGASCALE_PORT",
            "MEGASCALE_SLICE_ID",
            "TPU_PROCESS_ADDRESSES",
            "TPU_PROCESS_PORT",
            "TPU_WORKER_HOSTNAMES",
        ],
        "",
    )
    mp_env = env(build(variants["mp"]).roles[0].host_plans[1].tool_env)
    assert emptied.items() <= mp_env.items()
    assert mp_env["TPU_MULTIHOST_BACKEND"] == "mp"
    assert mp_env["TPU_SKIP_MDS_QUERY"] == "1"
    ray_env = env(build(variants["single"]).roles[0].host_plans[0].tool_env)
    assert {
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
        "TPU_MULTIHOST_BACKEND": "ray",
        "JAX_PLATFORMS": "",
        "TPU_SKIP_MDS_QUERY": "1",
    }.items() <= ray_env.items()
    pd_env = env(build(variants["pd"]).roles[0].host_plans[0].tool_env)
    assert emptied.items() <= pd_env.items()
    assert (pd_env["TPU_HOST_BOUNDS"], pd_env["TPU_TOPOLOGY"]) == ("1,1,1", "2x2x1")
    assert "TPU_SKIP_MDS_QUERY" not in pd_env
    one_host = env(build(variants["single_host"]).roles[0].host_plans[0].tool_env)
    assert one_host == {
        "PYTHONUNBUFFERED": "1",
        "TPU_ACCELERATOR_TYPE": "tpu7x",
        "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
    }


def test_p07_ray_bring_up(variants: dict[str, Data]) -> None:
    head, worker = build(variants["single"]).roles[0].host_plans
    assert [c.argv for c in head.pre] == [("ray", "start", "--head", "--port=6379")]
    assert [c.argv for c in worker.pre] == [
        ("ray", "start", Concat(("--address=", HEAD_IP, ":6379")))
    ]
    assert worker.serve is None


def test_p08_advertise_address(variants: dict[str, Data]) -> None:
    for name in ("single", "mp", "pd", "k3_pd"):
        for role in build(variants[name]).roles:
            for host in role.host_plans:
                assert env(host.tool_env)["VLLM_HOST_IP"] == SELF_IP
    host = build(variants["single_host"]).roles[0].host_plans[0]
    assert "VLLM_HOST_IP" not in env(host.tool_env)


def test_p09_startup_timeout(variants: dict[str, Data]) -> None:
    variants["single"]["server"]["startup_timeout"] = "4h"
    (role,) = build(variants["single"]).roles
    assert env(role.host_plans[1].tool_env)["VLLM_ENGINE_READY_TIMEOUT_S"] == "14400"
    assert (role.readiness.path, role.readiness.port) == ("/health", 8000)
    assert role.readiness.timeout_s == 14400 + 600


def test_p10_accelerator_env(variants: dict[str, Data]) -> None:
    for role in build(variants["pd"]).roles:
        assert env(role.host_plans[0].tool_env)["TPU_ACCELERATOR_TYPE"] == "tpu7x"


def test_p11_kv_wiring(variants: dict[str, Data]) -> None:
    prefill, decode = build(variants["pd"]).roles
    # The exact text of today's disagg_engine.sh.
    expected = (
        '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":'
        '"vllm_torchtpu.distributed.kv_transfer.tpu_connector",'
        '"kv_role":"kv_producer","kv_port":14579}'
    )
    argv = prefill.host_plans[0].serve.argv
    assert argv[argv.index("--kv-transfer-config") + 1] == expected
    assert decode.kv is not None and decode.kv.kv_role == "kv_consumer"
    for role in (prefill, decode):
        tool_env = env(role.host_plans[0].tool_env)
        assert tool_env["TPU_KV_TRANSFER_PORT"] == "9100"
        assert tool_env["TPU_SIDE_CHANNEL_PORT"] == "9600"
    host = build(variants["single"]).roles[0].host_plans[0]
    assert "--kv-transfer-config" not in host.serve.argv


def test_p12_raiden_identity(variants: dict[str, Data]) -> None:
    prefill, decode = build(variants["pd"]).roles
    assert {
        "TPU_KV_TRANSFER_NAMESPACE": "decode",
        "TPU_RAIDEN_ENGINE_ID": "decode-engine",
        "TPU_RAIDEN_JOB_NAME": "decode",
        "TPU_RAIDEN_RESHARD_PORT_BASE": "28000",
        "TPU_RAIDEN_STORE_DISPATCH_PORT_BASE": "28100",
    }.items() <= env(decode.host_plans[0].tool_env).items()
    assert (
        env(prefill.host_plans[0].tool_env)["TPU_RAIDEN_RESHARD_PORT_BASE"] == "27000"
    )
    for role in [prefill, decode, *build(variants["k3_pd"]).roles]:
        for host in role.host_plans:
            advertise = env(host.tool_env)["TPU_RAIDEN_ADVERTISE_HOST"]
            assert advertise == Ref("role_ips", role=role.name)


def test_p13_router(variants: dict[str, Data]) -> None:
    variants["pd"]["server"]["router"]["args"] = {"timeout": 60}
    router = build(variants["pd"]).driver.router
    assert router is not None
    assert router.argv == (
        "python3",
        "examples/disagg/toy_proxy_server.py",
        "--host",
        "0.0.0.0",
        "--port",
        "8100",
        "--prefiller-hosts",
        Ref("role_head", role="prefill"),
        "--prefiller-ports",
        "8000",
        "--decoder-hosts",
        Ref("role_head", role="decode"),
        "--decoder-ports",
        "8000",
        "--timeout",
        "60",
    )
    assert (router.port, router.health_path) == (8100, "/healthcheck")


def test_p14_driver_and_endpoint(variants: dict[str, Data]) -> None:
    single = build(variants["single"]).driver
    assert (single.placement, single.role, single.chipless) == ("role", "server", False)
    assert (single.endpoint_host, single.endpoint_port) == ("127.0.0.1", 8000)
    assert env(single.task_env) == {
        "PYTHONUNBUFFERED": "1",
        "TPU_MULTIHOST_BACKEND": "ray",
    }
    pd = build(variants["pd"]).driver
    assert (pd.placement, pd.role, pd.chipless) == ("client", None, True)
    assert (pd.endpoint_host, pd.endpoint_port) == ("127.0.0.1", 8100)
    k3 = build(variants["k3_pd"]).driver
    assert (k3.placement, k3.role) == ("role", "prefill")
    del variants["pd"]["server"]["router"]
    no_router = build(variants["pd"]).driver
    assert no_router.endpoint_host == Ref("role_head", role="prefill")
    assert no_router.endpoint_port == 8000


def test_p15_client_tokenizer(variants: dict[str, Data]) -> None:
    cell = task(build(variants["single"]), PerfTaskPlan).cells[0]
    argv = cell.measured.argv
    pulled = Ref("pulled_model_dir", fallback=GLM)
    assert argv[argv.index("--tokenizer") + 1] == pulled
    variants["single"]["tasks"]["perf"]["tokenizer"] = "zai-org/GLM-5.2"
    argv = task(build(variants["single"]), PerfTaskPlan).cells[0].measured.argv
    assert argv[argv.index("--tokenizer") + 1] == "zai-org/GLM-5.2"
    argv = task(build(variants["pd"]), PerfTaskPlan).cells[0].measured.argv
    assert "--tokenizer" not in argv


def test_p16_result_paths(variants: dict[str, Data]) -> None:
    (cell,) = task(build(variants["single"]), PerfTaskPlan).cells
    assert cell.result_file == Concat((RESULTS, "/isl8192_osl1024_c1024.json"))
    (warmup,) = cell.warmups
    assert warmup.argv[-1] == Concat((RESULTS, "/isl8192_osl1024_c1024.warmup1.json"))
    pd_cells = task(build(variants["pd"]), PerfTaskPlan).cells
    assert len(pd_cells) == 17
    assert pd_cells[-1].result_file == Concat(
        (RESULTS, "/pd_disagg_decode_daily_round3_c256_i65536_o1024.json")
    )
    (run,) = task(build(variants["single"]), EvalTaskPlan).runs
    argv = run.command.argv
    assert argv[argv.index("--output_path") + 1] == RESULTS


def test_p17_custom_task_env(variants: dict[str, Data]) -> None:
    custom = task(build(variants["pd"]), CustomTaskPlan)
    assert env(custom.command.env) == {
        "COST": "$5",
        "DECODE_URL": Concat(("http://", Ref("role_head", role="decode"), ":8000")),
        "LOG_DIR": Concat((RESULTS, "/check")),
        "PREFILL_URL": Concat(("http://", Ref("role_head", role="prefill"), ":8000")),
        "PROXY_URL": "http://127.0.0.1:8100",
        "SCENARIO_API_PORT": "8000",
        "SCENARIO_HOST_DECODE": Ref("role_head", role="decode"),
        "SCENARIO_HOST_PREFILL": Ref("role_head", role="prefill"),
        "SCENARIO_RESULTS_DIR": RESULTS,
        "SCENARIO_ROUTER_PORT": "8100",
        "SCENARIO_SERVED_MODEL": "Qwen/Qwen3.5-397B-A17B-FP8",
    }
    assert custom.command.argv == ("bash", "scripts/check.sh")
    assert (custom.timeout_s, custom.skip_rest_if_exit_ge) == (4 * 3600, 2)
    on_role = task(build(variants["k3_pd"]), CustomTaskPlan)
    assert env(on_role.command.env)["SCENARIO_SERVER_LOG"] == Ref("server_log")


def test_p18_chipless_driver_env(variants: dict[str, Data]) -> None:
    assert {"VLLM_PLUGINS": "", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}.items() <= env(
        build(variants["pd"]).driver.task_env
    ).items()
    assert "VLLM_PLUGINS" not in env(build(variants["k3_pd"]).driver.task_env)


def test_p19_remote_code_warm(variants: dict[str, Data]) -> None:
    head, worker = build(variants["mp"]).roles[0].host_plans
    assert head.pre == ()
    (warm,) = worker.pre
    assert warm.argv == ("python3", "-m", "scenario.runtime.warm", GLM_PATH)
    assert (warm.timeout_s, warm.best_effort) == (600, True)
    variants["single"]["server"]["args"]["trust-remote-code"] = True
    ray_worker = build(variants["single"]).roles[0].host_plans[1]
    assert [c.argv[0] for c in ray_worker.pre] == ["ray"]


def test_p20_p21_agentx_env(variants: dict[str, Data]) -> None:
    agentx = task(build(variants["agentx"]), AgentxTaskPlan)
    assert agentx.command.argv == (
        "/opt/infx/runtime/venv/bin/python",
        "-m",
        "infx.bench",
        "agentic",
    )
    assert agentx.timeout_s == agentx.command.timeout_s == 3 * 3600
    assert env(agentx.command.env) == {
        "AGENTIC_OUTPUT_DIR": RESULTS,
        "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": "3600",
        "AIPERF_EXPERIMENTAL_FAST": "0",
        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "false",
        "AGENTIC_WARMUP_GRACE_PERIOD": "3600",
        "AIPERF_RUNTIME_DIR": "/opt/infx/runtime",
        "AIPERF_SERVER_URL": "http://127.0.0.1:8000",
        "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "0",
        "CONC": "8",
        "DCP_SIZE": "1",
        # tensor-parallel-size 32 alone: no data parallelism.
        "DP_ATTENTION": "false",
        "DURATION": "3600",
        # No enable-expert-parallel in this scenario.
        "EP_SIZE": "1",
        "EVAL_ONLY": "false",
        "FRAMEWORK": "vllm",
        "IMAGE": Ref("image"),
        "IS_AGENTIC": "1",
        "IS_MULTINODE": "false",
        "KV_OFFLOADING": "none",
        "MODEL": "moonshotai/Kimi-K3",
        "MODEL_PREFIX": "kimik3",
        "PCP_SIZE": "1",
        "PP_SIZE": "1",
        "PRECISION": "fp4",
        "PYTHONPATH": "/opt/infx/InferenceX/inferencex-e2e",
        "PYTHONSAFEPATH": "1",
        "RESULT_DIR": RESULTS,
        "RESULT_FILENAME": "agentx_c8",
        "RUNNER_TYPE": "tpu7x",
        "SERVED_MODEL_NAME": "moonshotai/Kimi-K3",
        "SPEC_DECODING": "none",
        # Chips, not TensorCores: 4 hosts x 4 chips.
        "TP": "16",
    }


def test_p21_agentx_fast_mode(variants: dict[str, Data]) -> None:
    data = variants["agentx"]
    data["tasks"]["agentx"] |= {"fast": True, "duration": "20m"}
    agentx_env = env(task(build(data), AgentxTaskPlan).command.env)
    assert (agentx_env["AIPERF_EXPERIMENTAL_FAST"], agentx_env["DURATION"]) == (
        "1",
        "1200",
    )


def test_p20_p21_agentx_against_pd(variants: dict[str, Data], agentx: Data) -> None:
    data = variants["pd"]
    data["tasks"] = {"driver": "client", **agentx["tasks"]}
    data["tasks"]["agentx"]["concurrency"] = 4
    del data["server"]["args"]["max-model-len"]
    data["server"]["roles"]["decode"]["args"]["max-num-seqs"] = 8
    agentx_env = env(task(build(data), AgentxTaskPlan).command.env)
    assert {
        "IS_MULTINODE": "true",
        "DISAGG": "true",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "4",
        "PREFILL_EP": "4",
        "PREFILL_DP_ATTN": "false",
        "DECODE_NUM_WORKERS": "1",
        "DECODE_TP": "4",
        "DECODE_EP": "4",
        # decode runs data-parallel-size 8.
        "DECODE_DP_ATTN": "true",
        "ENABLE_AGENTX_POWER": "0",
        "REQUIRE_POWER": "0",
        "AIPERF_SERVER_URL": "http://127.0.0.1:8100",
    }.items() <= agentx_env.items()
    # The router re-exports no engine metrics, so the client reads each head's.
    assert agentx_env["AIPERF_SERVER_METRICS_URLS"] == concat(
        "http://",
        Ref("role_head", "prefill"),
        ":8000/metrics,http://",
        Ref("role_head", "decode"),
        ":8000/metrics",
    )
    assert "TP" not in agentx_env and "EP_SIZE" not in agentx_env


def test_p22_identity_fields(variants: dict[str, Data]) -> None:
    """KV offload, router and transfer reach the row as InferenceX reads them."""
    data = variants["agentx"]
    data["tasks"]["inferencex"] |= {
        "kv_offloading": "dram",
        "kv_offload_backend": {"name": "mooncake", "version": "0.3.11"},
        "total_cpu_dram_gb": 512,
        "router": {"name": "dynamo-router", "version": "1.0"},
        "kv_p2p_transfer": "nixl",
    }
    data["server"]["args"] |= {"tensor-parallel-size": 4, "data-parallel-size": 8}
    agentx_env = env(task(build(data), AgentxTaskPlan).command.env)
    assert {
        "KV_OFFLOADING": "dram",
        "KV_OFFLOAD_BACKEND": "mooncake",
        "KV_OFFLOAD_BACKEND_METADATA": '{"name":"mooncake","version":"0.3.11"}',
        "TOTAL_CPU_DRAM_GB": "512",
        "ROUTER_METADATA": '{"name":"dynamo-router","version":"1.0"}',
        "KV_P2P_TRANSFER": "nixl",
        "DP_ATTENTION": "true",
    }.items() <= agentx_env.items()


def test_p22_identity_is_shared(variants: dict[str, Data], agentx: Data) -> None:
    """An AgentX row and the eval of the same deployment get one identity."""
    pair = variants["pd"]
    pair["tasks"] = {
        "driver": "client",
        "inferencex": variants["ix_eval"]["tasks"]["inferencex"],
        "agentx": agentx["tasks"]["agentx"],
    }
    pair["tasks"]["agentx"]["concurrency"] = 4
    del pair["server"]["args"]["max-model-len"]
    pair["server"]["roles"]["decode"]["args"]["max-num-seqs"] = 8
    agentx_env = env(task(build(pair), AgentxTaskPlan).command.env)
    (run,) = task(build(variants["ix_eval"]), EvalTaskPlan).runs
    eval_env = env(run.command.env)
    identity = {
        name: agentx_env[name] for name in INFERENCEX_IDENTITY_ENV if name in agentx_env
    }
    assert identity == {
        name: eval_env[name] for name in INFERENCEX_IDENTITY_ENV if name in eval_env
    }
    assert identity == {
        "MODEL": "Qwen/Qwen3.5-397B-A17B-FP8",
        "MODEL_PREFIX": "qwen3.5",
        "PRECISION": "fp8",
        "FRAMEWORK": "vllm",
        "SPEC_DECODING": "none",
        "KV_OFFLOADING": "none",
        "RUNNER_TYPE": "tpu7x",
        "IMAGE": Ref("image"),
        "IS_MULTINODE": "true",
        "DISAGG": "true",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "4",
        "PREFILL_EP": "4",
        "PREFILL_DP_ATTN": "false",
        "DECODE_NUM_WORKERS": "1",
        "DECODE_TP": "4",
        "DECODE_EP": "4",
        # decode runs data-parallel-size 8.
        "DECODE_DP_ATTN": "true",
    }
    variants["ix_eval"]["server"]["args"]["enable-expert-parallel"] = False
    (run,) = task(build(variants["ix_eval"]), EvalTaskPlan).runs
    assert (env(run.command.env)["PREFILL_EP"], env(run.command.env)["DECODE_EP"]) == (
        "1",
        "1",
    )


def test_p22_needs_roles_of_one_shape(variants: dict[str, Data]) -> None:
    data = variants["ix_eval"]
    del data["server"]["router"]
    roles = data["server"]["roles"]
    roles["spare"] = {**roles["decode"], "args": {"enable-expert-parallel": False}}
    with pytest.raises(PlanError, match="needs consumer roles of one shape"):
        build(data)


def test_p21_inferencex_eval(variants: dict[str, Data]) -> None:
    evals = task(build(variants["ix_eval"]), EvalTaskPlan)
    assert evals.client == "inferencex"
    (run,) = evals.runs
    assert (run.key, run.task, run.tool, run.gate) == (
        "lm-eval",
        "lm-eval",
        "inferencex",
        None,
    )
    stage = Concat((RESULTS, "/inferencex_eval"))
    # infx.bench eval: the server root behind the router, one concurrency.
    assert run.command.argv == (
        "/opt/infx/runtime/venv/bin/python",
        "-m",
        "infx.bench",
        "eval",
        "--endpoint",
        "http://127.0.0.1:8100",
        "--concurrency",
        "4",
        "--stage-to",
        stage,
        "--framework",
        "lm-eval",
    )
    assert run.command.timeout_s == 8 * 3600
    assert {
        "EVAL_ONLY": "true",
        "OPENAI_API_KEY": "EMPTY",
        "MODEL_NAME": "Qwen/Qwen3.5-397B-A17B-FP8",
        "PYTHONPATH": "/opt/infx/InferenceX/inferencex-e2e",
        "PYTHONSAFEPATH": "1",
        "EVAL_LIMIT": "20",
    }.items() <= env(run.command.env).items()
    assert "EVAL_SUITE" not in env(run.command.env)
    # validate_scores against InferenceX's thresholds; its exit code is the gate.
    assert run.floor is not None
    assert run.floor.argv == (
        "/opt/infx/runtime/venv/bin/python",
        "-m",
        "infx.evals.validate_scores",
        "--meta-env",
        Concat((RESULTS, "/inferencex_eval/meta_env.json")),
        "--results-glob",
        Concat((RESULTS, "/inferencex_eval/results*.json")),
        "--expected-concs",
        "4",
    )
    assert env(run.floor.env) == {
        "PYTHONPATH": "/opt/infx/InferenceX/inferencex-e2e",
        "PYTHONSAFEPATH": "1",
    }
    data = variants["ix_eval"]
    data["tasks"]["eval"] |= {
        "framework": "kimi-vendor",
        "suite": "kimi_tool_call_schema_full",
    }
    (run,) = task(build(data), EvalTaskPlan).runs
    assert run.key == "kimi_tool_call_schema_full"
    assert run.command.argv[-1] == "kimi-vendor"
    assert env(run.command.env)["EVAL_SUITE"] == "kimi_tool_call_schema_full"


def test_vllm_bench_argv(variants: dict[str, Data]) -> None:
    (cell,) = task(build(variants["single"]), PerfTaskPlan).cells
    assert cell.measured.argv == (
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--request-rate",
        "inf",
        "--model",
        GLM,
        "--tokenizer",
        Ref("pulled_model_dir", fallback=GLM),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dataset-name",
        "random",
        "--random-input-len",
        "8192",
        "--random-output-len",
        "1024",
        "--random-range-ratio",
        "0.0",
        "--num-prompts",
        "2048",
        "--max-concurrency",
        "1024",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--temperature",
        "0.0",
        "--save-result",
        "--result-filename",
        Concat((RESULTS, "/isl8192_osl1024_c1024.json")),
    )


def test_min_range_ratio_is_translated(variants: dict[str, Data]) -> None:
    perf = variants["single"]["tasks"]["perf"]
    perf |= {"range_ratio_style": "min", "random_range_ratio": 0.8}
    perf["isl_osl"] = [[8192, 1024]]
    argv = task(build(variants["single"]), PerfTaskPlan).cells[0].measured.argv
    # run_benchmarks.sh: len x (1+r)/2 as %.0f, (1-r)/(1+r) as %.6f.
    start = argv.index("--random-input-len")
    assert argv[start : start + 6] == (
        "--random-input-len",
        "7373",
        "--random-output-len",
        "922",
        "--random-range-ratio",
        "0.111111",
    )


def test_cells_mode_argv(variants: dict[str, Data]) -> None:
    perf = task(build(variants["pd"]), PerfTaskPlan)
    assert perf.mode == "cells"
    prefill = perf.cells[1]
    argv = prefill.measured.argv
    assert prefill.key == "pd_disagg_prefill_c8_i8192_o1"
    assert argv[argv.index("--label") + 1] == prefill.key
    assert argv[argv.index("--percentile-metrics") + 1] == "ttft,e2el"
    assert argv[argv.index("--num-prompts") + 1] == "512"
    smoke = perf.cells[0].measured.argv
    assert smoke[smoke.index("--percentile-metrics") + 1] == "ttft,tpot,itl,e2el"
    assert smoke[3:7] == ("--backend", "openai", "--endpoint", "/v1/completions")


def test_inferencex_argv(variants: dict[str, Data]) -> None:
    perf = variants["single_host"]["tasks"]["perf"]
    perf |= {
        "client": "inferencex",
        "random_range_ratio": 0.8,
        "num_prompts": {"per_concurrency": 10},
        "num_warmups": {"per_concurrency": 2},
        "warmup_runs": 0,
        "client_args": {
            "backend": "vllm",
            "request-rate": "inf",
            "ignore-eos": True,
            "use-chat-template": True,
        },
        "concurrency": [64],
    }
    del perf["range_ratio_style"], perf["temperature"], perf["baseline"]
    variants["single_host"]["server"]["model"] = {"name": "Qwen/Qwen3.5-397B-A17B-FP8"}
    variants["single_host"]["server"]["args"]["load-format"] = "auto"
    variants["single_host"]["tasks"].pop("eval")
    (cell,) = task(build(variants["single_host"]), PerfTaskPlan).cells
    assert cell.warmups == ()
    assert env(cell.measured.env) == {
        "PYTHONPATH": "/opt/infx/InferenceX/inferencex-e2e"
    }
    # The flags and values of InferenceX's qwen3.5/bench.sh.
    assert cell.measured.argv == (
        "python3",
        "-m",
        "infx.bench_serving.benchmark_serving",
        "--model",
        "Qwen/Qwen3.5-397B-A17B-FP8",
        "--backend",
        "vllm",
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--use-chat-template",
        "--base-url",
        "http://127.0.0.1:8000",
        "--dataset-name",
        "random",
        "--random-input-len",
        "8192",
        "--random-output-len",
        "1024",
        "--random-range-ratio",
        "0.8",
        "--num-prompts",
        "640",
        "--max-concurrency",
        "64",
        "--num-warmups",
        "128",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--save-result",
        "--result-dir",
        RESULTS,
        "--result-filename",
        "isl8192_osl1024_c64.json",
    )


def test_lm_eval_argv(variants: dict[str, Data]) -> None:
    (run,) = task(build(variants["single"]), EvalTaskPlan).runs
    assert (run.key, run.task, run.tool) == ("mmlu_pro", "mmlu_pro", "lm_eval")
    # run_eval_flow.sh's order, with the config's extras after the defaults.
    assert run.command.argv == (
        "lm_eval",
        "--tasks",
        "mmlu_pro",
        "--seed",
        "0,1234,None,1234",
        "--output_path",
        RESULTS,
        "--limit",
        "100",
        "--model",
        "local-chat-completions",
        "--model_args",
        "model=zai-org/GLM-5.2-FP8,base_url=http://127.0.0.1:8000/v1/chat/completions,"
        "num_concurrent=32,timeout=7200",
        "--apply_chat_template",
        "--gen_kwargs",
        "until=['<|im_end|>']",
    )


def test_lm_eval_task_items(variants: dict[str, Data]) -> None:
    lm_eval = variants["single"]["tasks"]["eval"]["lm_eval"]
    lm_eval["fewshot_as_multiturn"] = True
    lm_eval["gen_kwargs"] = {"chat_template_kwargs": {"enable_thinking": False}}
    lm_eval["tasks"] = [
        "mmlu_pro",
        {
            "name": "humaneval_plus_tpu",
            "limit": "all",
            "include_path": "scripts/vllm/benchmarking/lm_eval_tasks",
            "confirm_run_unsafe_code": True,
            "fewshot_as_multiturn": False,
            "env": {"HF_ALLOW_CODE_EVAL": "1"},
        },
    ]
    variants["single"]["tasks"]["eval"]["baseline"]["humaneval_plus_tpu"] = {
        "pass@1,create_test": 0.8
    }
    mmlu, code = task(build(variants["single"]), EvalTaskPlan).runs
    assert mmlu.command.argv[-4:] == (
        "--fewshot_as_multiturn",
        "true",
        "--gen_kwargs",
        '{"chat_template_kwargs":{"enable_thinking":false}}',
    )
    argv = code.command.argv
    assert "--limit" not in argv and "--fewshot_as_multiturn" not in argv
    assert argv[argv.index("--include_path") + 1] == (
        "scripts/vllm/benchmarking/lm_eval_tasks"
    )
    assert argv[-1] == "--confirm_run_unsafe_code"
    assert env(code.command.env) == {"HF_ALLOW_CODE_EVAL": "1"}


def test_lm_eval_concurrency_sweep(variants: dict[str, Data]) -> None:
    evals = variants["single"]["tasks"]["eval"]
    evals["lm_eval"] |= {
        "concurrency": [1, 64],
        "model_args": {"max_retries": 3},
        "limit": 10,
        "tasks": [
            {"name": "gsm8k", "limit": 128, "limit_per_concurrency": 2},
            {"name": "mmlu_pro", "limit_per_concurrency": 2, "subtasks": 14},
        ],
    }
    evals["baseline"] = {"gsm8k_c64": {"exact_match,flexible-extract": 0.9}}
    runs = task(build(variants["single"]), EvalTaskPlan).runs
    assert [r.key for r in runs] == [
        "gsm8k_c1",
        "mmlu_pro_c1",
        "gsm8k_c64",
        "mmlu_pro_c64",
    ]

    def flag(run: Any, name: str) -> Any:
        return run.command.argv[run.command.argv.index(name) + 1]

    # eval_conc_sweep.sh: max(limit, ceil(2 x conc / subtasks)).
    assert [flag(r, "--limit") for r in runs] == ["128", "10", "128", "10"]
    assert flag(runs[2], "--model_args") == (
        "model=zai-org/GLM-5.2-FP8,base_url=http://127.0.0.1:8000/v1/chat/completions,"
        "num_concurrent=64,max_retries=3"
    )
    assert flag(runs[3], "--output_path") == Concat((RESULTS, "/mmlu_pro_c64"))
    assert [r.gate is not None for r in runs] == [False, False, True, False]
    evals["lm_eval"]["concurrency"] = [128]
    del evals["baseline"]
    mmlu = task(build(variants["single"]), EvalTaskPlan).runs[1]
    assert flag(mmlu, "--limit") == "19"


def test_gates(variants: dict[str, Data]) -> None:
    single = build(variants["single"])
    gate = task(single, PerfTaskPlan).gate
    assert gate is not None and (gate.tolerance, gate.source) == (0.05, "inline")
    assert gate.as_dict() == {
        "isl8192_osl1024_c1024": {"completed": 2048, "median_tpot_ms": 276.99}
    }
    (run,) = task(single, EvalTaskPlan).runs
    assert run.gate is not None
    assert run.gate.as_dict() == {"mmlu_pro": {"exact_match,custom-extract": 0.7629}}
    assert run.gate.tolerance == 0.025
    experiment = build(variants["single"], overrides=("server.args.max-num-seqs=32",))
    assert experiment.experiment == ("server.args.max-num-seqs=32",)
    assert task(experiment, PerfTaskPlan).gate is None
    assert task(experiment, EvalTaskPlan).runs[0].gate is None


def test_task_order_and_only(variants: dict[str, Data]) -> None:
    data = variants["single"]
    data["tasks"]["custom"] = [{"name": "probe", "command": ["true"], "timeout": 60}]
    kinds = [t.kind for t in build(data).driver.tasks]
    assert kinds == ["perf", "custom", "eval"]
    only = build(data, only=["eval", "probe"])
    assert [t.kind for t in only.driver.tasks] == ["custom", "eval"]
    assert only.only == ("eval", "probe")
    with pytest.raises(PlanError, match="--only names agentx, which this scenario"):
        build(data, only=["agentx"])


def test_plan_identity(variants: dict[str, Data]) -> None:
    single = build(variants["single"])
    assert (single.format, single.scenario_name, single.point) == (
        1,
        "glm-5.2-fp8-dp16-ep",
        None,
    )
    assert (single.image, single.model_name, single.model_path) == (None, GLM, GLM_PATH)
    assert single.experiment == () and single.only is None


def test_shape(variants: dict[str, Data]) -> None:
    shape = build(variants["pd"]).shape()
    assert [r.name for r in shape.roles] == ["prefill", "decode", "client"]
    client = shape.roles[-1]
    assert (client.chipless, client.driver, client.ports) == (True, True, (8100, 8090))
    assert shape.roles[0].ports == (8000,)
    one = build(variants["single_host"]).shape().roles[0]
    assert (one.driver, one.ports, one.backend) == (True, (8000,), "single")
    k3 = build(variants["k3_pd"]).shape().roles
    assert k3[0].ports == (8000, 8100, 8090) and k3[0].scratch_bytes == 64 * 2**30


def test_every_tool_env_name_is_reserved(variants: dict[str, Data]) -> None:
    """A scenario can never set a name the plan sets (V-11, V-13, V-22)."""
    for name, data in variants.items():
        plan_ = build(data)
        for role in plan_.roles:
            for host in role.host_plans:
                for key, _ in host.tool_env:
                    assert tool_env_owner(key) or key in RAY_ENV, (name, key)
                    assert env_rule(key), key
        for key, _ in plan_.driver.task_env:
            assert tool_env_owner(key) or key in CHIPLESS_DRIVER_ENV, (name, key)
            assert env_rule(key), key
        names = [r.name for r in plan_.roles]
        kv = {r.name: r.kv.kv_role.removeprefix("kv_") for r in plan_.roles if r.kv}
        provided = custom_task_env(
            roles=names,
            kv_roles=kv,
            router=plan_.driver.router is not None,
            driver_on_role=not plan_.driver.chipless,
        )
        for item in plan_.driver.tasks:
            if isinstance(item, CustomTaskPlan):
                tool_names = {k for k, _ in item.command.env} - set(
                    data["tasks"]["custom"][0]["env"]
                )
                assert tool_names <= provided
            if isinstance(item, AgentxTaskPlan):
                tool_names = {k for k, _ in item.command.env} - set(
                    data["tasks"]["agentx"]["env"]
                )
                assert tool_names <= set(AGENTX_CLIENT_ENV)
            if isinstance(item, PerfTaskPlan):
                for cell in item.cells:
                    tool_names = {k for k, _ in cell.measured.env}
                    assert tool_names <= set(INFERENCEX_PERF_ENV)
            if isinstance(item, EvalTaskPlan) and item.client == "inferencex":
                for run in item.runs:
                    assert run.floor is not None
                    for command in (run.command, run.floor):
                        tool_names = {k for k, _ in command.env} - set(
                            data["tasks"]["eval"]["env"]
                        )
                        assert tool_names <= set(INFERENCEX_EVAL_ENV)
                        assert all(env_rule(k) for k in tool_names), tool_names


_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def test_no_runtime_values_in_a_plan(variants: dict[str, Data]) -> None:
    for data in variants.values():
        text = to_json(build(data))
        assert set(_IPV4.findall(text)) <= {"127.0.0.1", "0.0.0.0"}
        for banned in ("HF_TOKEN", "POD_IP", "JOB_COMPLETION_INDEX", "-0-0."):
            assert banned not in text


DEFAULTS = """\
schema: 1
server: {startup_timeout: 2h, args: {api-server-count: 1}}
"""


def write(root: pathlib.Path, ref: str, text: str) -> pathlib.Path:
    path = root / "scenarios" / f"{ref}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


def test_plan_from_files_and_points(tmp_path: pathlib.Path) -> None:
    write(tmp_path, "defaults", DEFAULTS)
    ladder = write(
        tmp_path,
        "ladder",
        """\
        schema: 1
        name: ladder
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
          args: {max-num-seqs: 8}
        sweep:
          name: s{server.args.max-num-seqs}
          zip: {server.args.max-num-seqs: [8, 16]}
        """,
    )
    points = resolve(ladder)
    with pytest.raises(PlanError, match="sweep of 2 points.*--point: s8, s16"):
        select_point(points)
    with pytest.raises(PlanError, match="has no point s4"):
        select_point(points, "s4")
    planned = plan(select_point(points, "s16"))
    assert planned.point == "s16"
    assert "16" in planned.roles[0].host_plans[0].serve.argv


def test_dp_errors_come_from_validation(tmp_path: pathlib.Path) -> None:
    write(tmp_path, "defaults", DEFAULTS)
    bad = write(
        tmp_path,
        "bad",
        """\
        schema: 1
        name: bad
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x4
          hosts: 4
          chips_per_host: 4
          multihost: mp
          args: {data-parallel-size: 6, data-parallel-size-local: 2}
        """,
    )
    with pytest.raises(DiagnosticError) as info:
        resolve(bad)
    messages = [d.render() for d in info.value.diagnostics]
    assert messages == [
        "scenarios/bad.yaml:10: server.args.data-parallel-size-local: srt-slurm "
        "style: do not set data-parallel-size-local; the tool derives it from "
        "data-parallel-size and hosts",
        "scenarios/bad.yaml:10: server.args.data-parallel-size: multihost mp needs "
        "data-parallel-size divisible by 4 hosts; got 6",
    ]

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
"""Valid scenarios as plain data, fresh for every test.

They carry the values scenarios/defaults.yaml would add, so each validates
as a Scenario on its own.
"""

import copy
from typing import Any

import pytest

PERCENTILES = ["ttft", "tpot", "itl", "e2el"]


def tpu7x(topology: str, hosts: int) -> dict[str, Any]:
    """The hardware fields of a tpu7x role with four chips per host."""
    return {
        "accelerator": "tpu7x",
        "topology": topology,
        "hosts": hosts,
        "chips_per_host": 4,
    }


@pytest.fixture
def single() -> dict[str, Any]:
    """One role on two hosts over Ray, with perf and eval (GLM-5.2)."""
    return {
        "schema": 1,
        "name": "glm-5.2-fp8-dp16-ep",
        "server": {
            "model": {
                "name": "zai-org/GLM-5.2-FP8",
                "path": "gs://tpu-commons-ci/glm/GLM-5.2-FP8",
            },
            **tpu7x("2x2x2", 2),
            "multihost": "ray",
            "startup_timeout": "2h",
            "args": {
                "load-format": "runai_streamer",
                "tensor-parallel-size": 1,
                "data-parallel-size": 16,
                "enable-expert-parallel": True,
                "max-model-len": 9216,
                "max-num-seqs": 64,
                "api-server-count": 1,
            },
        },
        "tasks": {
            "perf": {
                "dataset": {"name": "random"},
                "isl_osl": [[8192, 1024]],
                "concurrency": [1024],
                "num_prompts": 2048,
                "random_range_ratio": 0.0,
                "range_ratio_style": "symmetric",
                "warmup_runs": 1,
                "temperature": 0,
                "percentile_metrics": PERCENTILES,
                "client_args": {"backend": "vllm", "request-rate": "inf"},
                "tolerance": 0.05,
                "baseline": {
                    "isl8192_osl1024_c1024": {
                        "completed": 2048,
                        "median_tpot_ms": 276.99,
                    }
                },
            },
            "eval": {
                "lm_eval": {
                    "tasks": ["mmlu_pro"],
                    "model": "local-chat-completions",
                    "model_args": {"num_concurrent": 32, "timeout": 7200},
                    "gen_kwargs": "until=['<|im_end|>']",
                    "limit": 100,
                    "seed": "0,1234,None,1234",
                    "apply_chat_template": True,
                    "fewshot_as_multiturn": False,
                },
                "tolerance": 0.025,
                "baseline": {"mmlu_pro": {"exact_match,custom-extract": 0.7629}},
            },
        },
    }


@pytest.fixture
def pd() -> dict[str, Any]:
    """Prefill and decode on 2x2x1 each, a router, a chipless driver (P14)."""
    kv = {"connector": "TPURaidenConnector"}
    return {
        "schema": 1,
        "name": "qwen3.5-397b-fp8-pcp8-dp8-1p1d",
        "server": {
            "model": {"name": "Qwen/Qwen3.5-397B-A17B-FP8"},
            "startup_timeout": "2h",
            "args": {"max-model-len": 66560, "enable-expert-parallel": True},
            "env": {"JAX_PLATFORMS": "tpu,cpu", "TPU_KV_SHM_POOL_GB": 8},
            "roles": {
                "prefill": {
                    **tpu7x("2x2x1", 1),
                    "kv": {**kv, "role": "producer"},
                    "args": {"max-num-seqs": 8, "prefill-context-parallel-size": 8},
                },
                "decode": {
                    **tpu7x("2x2x1", 1),
                    "kv": {**kv, "role": "consumer"},
                    "args": {"max-num-seqs": 32, "data-parallel-size": 8},
                },
            },
            "router": {
                "type": "toy-proxy",
                "command": ["python3", "examples/disagg/toy_proxy_server.py"],
            },
        },
        "tasks": {
            "driver": "client",
            "perf": {
                "dataset": {"name": "random"},
                "num_prompts": 320,
                "random_range_ratio": 0,
                "range_ratio_style": "symmetric",
                "warmup_runs": 0,
                "temperature": 0,
                "percentile_metrics": ["ttft", "e2el"],
                "client_args": {"backend": "openai", "endpoint": "/v1/completions"},
                "tolerance": 0.05,
                "cells": [
                    {
                        "label": "pd_disagg_decode_smoke_c8_i8192_o32",
                        "isl": 8192,
                        "osl": 32,
                        "concurrency": 8,
                        "num_prompts": 8,
                        "percentile_metrics": PERCENTILES,
                    },
                    {
                        "label": "pd_disagg_prefill_c{concurrency}_i8192_o1",
                        "isl": 8192,
                        "osl": 1,
                        "concurrency": [8, 16, 32, 64, 128, 256],
                        "num_prompts": 512,
                    },
                    {
                        "label": "pd_disagg_ttft_c1_i{isl}_o1",
                        "isl": [8192, 16384, 32768, 65536],
                        "osl": 1,
                        "concurrency": 1,
                        "num_prompts": 16,
                    },
                    {
                        "label": "pd_disagg_decode_round{repeat}_c8_i8192_o1024",
                        "isl": 8192,
                        "osl": 1024,
                        "concurrency": 8,
                        "num_prompts": 8,
                        "repeat": 3,
                    },
                    {
                        "label": (
                            "pd_disagg_decode_daily_round{repeat}_c256_i65536_o1024"
                        ),
                        "isl": 65536,
                        "osl": 1024,
                        "concurrency": 256,
                        "num_prompts": 256,
                        "repeat": 3,
                    },
                ],
            },
            "custom": [
                {
                    "name": "correctness",
                    "command": ["bash", "scripts/check.sh"],
                    "env": {"LOG_DIR": "${SCENARIO_RESULTS_DIR}/check", "COST": "$$5"},
                    "timeout": "4h",
                    "skip_rest_if_exit_ge": 2,
                }
            ],
        },
    }


@pytest.fixture
def ix_eval(pd: dict[str, Any]) -> dict[str, Any]:
    """InferenceX's GSM8K eval alone on the P/D pair, from a chipless driver."""
    data = copy.deepcopy(pd)
    data["name"] = "qwen3.5-397b-gsm8k-1p1d"
    data["tasks"] = {
        "driver": "client",
        "inferencex": {
            "model_prefix": "qwen3.5",
            "precision": "fp8",
            "framework": "vllm",
            "spec_decoding": "none",
            "kv_offloading": "none",
        },
        "eval": {
            "client": "inferencex",
            "framework": "lm-eval",
            "concurrency": 4,
            "timeout": "8h",
            "env": {"EVAL_LIMIT": "20"},
        },
    }
    return data


@pytest.fixture
def agentx() -> dict[str, Any]:
    """An AgentX ladder: one fresh server per concurrency (Kimi-K3)."""
    return {
        "schema": 1,
        "name": "kimi-k3-agentx",
        "server": {
            "model": {
                "name": "moonshotai/Kimi-K3",
                "path": "gs://tpu-commons-ci/moonshootai/kimi/k3",
            },
            **tpu7x("2x2x4", 4),
            "multihost": "ray",
            "startup_timeout": "4h",
            "args": {
                "load-format": "runai_streamer",
                "tensor-parallel-size": 32,
                "max-num-seqs": 16,
            },
        },
        "tasks": {
            "inferencex": {
                "model_prefix": "kimik3",
                "precision": "fp4",
                "framework": "vllm",
                "spec_decoding": "none",
                "kv_offloading": "none",
            },
            "agentx": {
                "concurrency": 8,
                "duration": "1h",
                "timeout": "3h",
                "env": {"AGENTIC_WARMUP_GRACE_PERIOD": "3600"},
            },
        },
        "sweep": {
            "name": "c{tasks.agentx.concurrency}",
            "zip": {
                "tasks.agentx.concurrency": [1, 2, 4, 8],
                "server.args.max-num-seqs": [16, 16, 16, 32],
            },
        },
    }


@pytest.fixture
def variants(
    single: dict[str, Any],
    pd: dict[str, Any],
    agentx: dict[str, Any],
    ix_eval: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Every plan_fixtures variant, built from fresh copies of the fixtures."""
    from plan_fixtures import VARIANTS

    base = {"single": single, "pd": pd, "agentx": agentx, "ix_eval": ix_eval}
    return {name: make(base) for name, make in VARIANTS.items()}

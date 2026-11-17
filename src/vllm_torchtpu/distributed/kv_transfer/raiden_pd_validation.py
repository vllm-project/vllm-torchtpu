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
"""Shared startup validation for Raiden Stage-3 P/D connector configs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer.connector_names import (
    RAIDEN_PD_CONNECTOR_NAMES,
)

# TODO: derive Stage-3 admission directly from model_config and retire these
# four flags in a follow-up PR.
RAIDEN_PD_ADMISSION_FLAGS: tuple[str, ...] = (
    "TPU_RAIDEN_QWEN35_ADMISSION",
    "TPU_RAIDEN_KIMIK3_ADMISSION",
    "TPU_RAIDEN_GLM_ADMISSION",
    "TPU_RAIDEN_DSV4_ADMISSION",
)

KIMI_ARCHITECTURES: tuple[str, ...] = (
    "KimiK3ForConditionalGeneration",
    "KimiLinearForCausalLM",
)

USE_RAIDEN_CONNECTOR_REMOVED_MESSAGE = (
    "kv_connector_extra_config['use_raiden_connector'] has been removed; "
    "TPURaidenConnector always uses the Raiden Stage-3 backend (remove "
    "'use_raiden_connector' from kv_connector_extra_config)."
)

RAIDEN_PD_ADMISSION_REQUIRED_MESSAGE = (
    "TPURaidenConnector requires TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 and one of "
    + ", ".join(f"{flag}=1" for flag in RAIDEN_PD_ADMISSION_FLAGS[:-1])
    + f", or {RAIDEN_PD_ADMISSION_FLAGS[-1]}=1 (the legacy ZMQ transport has "
    "been removed)."
)

# Legacy KV-transfer env vars removed with the ZMQ/SHM backend. Set, they
# would otherwise be silently ignored at startup.
_LEGACY_ZMQ_REMOVED_REMEDY = (
    "the legacy ZMQ/SHM KV transfer backend has been removed. Unset it."
)
_REMOVED_KV_TRANSFER_ENVS: dict[str, str] = {
    "TPU_USE_RAIDEN_CONNECTOR": (
        "TPURaidenConnector always uses the Raiden Stage-3 backend. Unset it."
    ),
    "TPU_SIDE_CHANNEL_PORT": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_TRANSFER_CHANNEL_NUMBER": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_TRANSFER_NAMESPACE": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_SHM_POOL_GB": (
        "the legacy ZMQ/SHM KV transfer backend has been removed; Stage-3 "
        "uses TPU_RAIDEN_POOL_STAGING_LEASES and "
        "TPU_RAIDEN_POOL_STAGING_BUDGET_MB instead. Unset it."
    ),
    "TPU_KV_COORD_EXECUTOR_MAX_WORKERS": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_STAGE_WAITER_POOL_SIZE": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_STAGE_WAIT_TIMEOUT_SECS": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_IPC_SOCKET_DIR": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_WARMUP_ENABLED": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_LATENCY_LOG_INTERVAL": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_PIN_SHM": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_RAIDEN_TRANSFER_NUM_SLOTS": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_D2H_CONCURRENCY": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_H2D_CONCURRENCY": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_PULL_CONCURRENCY": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_ZMQ_RCVHWM": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_ZMQ_SNDHWM": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_PULL_LINGER_MS": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_ENABLE_SHM_SNAPSHOT": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_SNAPSHOT_DIR": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_KV_PULL_WAIT_LOG_INTERVAL_S": _LEGACY_ZMQ_REMOVED_REMEDY,
    "VLLM_TORCHTPU_IPC_KEY": _LEGACY_ZMQ_REMOVED_REMEDY,
    "TPU_NODE_ID": _LEGACY_ZMQ_REMOVED_REMEDY,
}


def is_kimi_architecture(vllm_config: Any) -> bool:
    """Whether ``vllm_config`` names a Kimi architecture."""
    model_config = getattr(vllm_config, "model_config", None)
    return getattr(model_config, "architecture", None) in KIMI_ARCHITECTURES


def _raiden_seq_on_lane_layout() -> bool:
    from vllm import envs as vllm_envs

    return vllm_envs.VLLM_KV_CACHE_LAYOUT == "HND"


def raiden_qwen35_admission_topology(
    vllm_config: Any,
    *,
    is_producer: bool,
    tp_size_fallback: int = 1,
    seq_on_lane: bool | None = None,
) -> str:
    """Validate and return the Qwen3.5 Raiden Stage-3 admission topology name."""
    parallel_config = vllm_config.parallel_config
    pcp_size = int(parallel_config.prefill_context_parallel_size or 1)
    tp_size = int(parallel_config.tensor_parallel_size or tp_size_fallback)
    dp_size = int(parallel_config.data_parallel_size or 1)
    pp_size = int(parallel_config.pipeline_parallel_size or 1)
    use_seq_on_lane = (
        _raiden_seq_on_lane_layout() if seq_on_lane is None else seq_on_lane
    )
    if use_seq_on_lane and dp_size == pp_size == 1:
        dcp_size = int(parallel_config.decode_context_parallel_size or 1)
        if dcp_size <= 0 or tp_size % dcp_size:
            raise ValueError("DCP size must divide TP size")
        if is_producer and tp_size > 1 and dcp_size == 1:
            return f"pcp{pcp_size}tp{tp_size}_prefill"
        if not is_producer and pcp_size == 1 and tp_size > 1:
            return f"tp{tp_size}dcp{dcp_size}_decode"
    if is_producer and pp_size > 1 and pcp_size > 1:
        # PP x PCP prefill: every stage-local PCP rank is a source unit
        # (TPU_RAIDEN_TRANSFER_PARALLELISM = PP * PCP) lowering its own
        # tokens and GDN heads of its stage's layers.
        # TODO(yinlin): reachable once pp_validation admits context-parallel
        # stages, i.e. once the pinned torch_tpu builds a stage group's meshes
        # (google-pytorch/torch_tpu#4387, for a group made standalone).
        dcp_size = int(parallel_config.decode_context_parallel_size or 1)
        if (
            use_seq_on_lane
            and tp_size == 1
            and dp_size == 1
            and dcp_size == 1
            and pp_size in (2, 4, 8)
        ):
            return f"pp{pp_size}pcp{pcp_size}_prefill"
        raise ValueError(
            "Raiden Qwen3.5 admission topology pp{2,4,8}pcp{N}_prefill "
            "requires the seq-along-lane FA layout "
            "(VLLM_KV_CACHE_LAYOUT=HND) and kv_producer with "
            "tensor_parallel_size=1, data_parallel_size=1, "
            "decode_context_parallel_size=1, pipeline_parallel_size in "
            f"(2, 4, 8); got prefill_context_parallel_size={pcp_size}, "
            f"tensor_parallel_size={tp_size}, "
            f"data_parallel_size={dp_size}, "
            f"decode_context_parallel_size={dcp_size}, "
            f"pipeline_parallel_size={pp_size}"
        )
    if is_producer:
        if tp_size == 1 and pcp_size == 1 and dp_size == 1 and pp_size in (2, 4, 8):
            return f"pp{pp_size}_prefill"
        if tp_size == 1 and pcp_size in (4, 8) and dp_size == 1:
            return f"pcp{pcp_size}_prefill"
        if tp_size == 1 and pcp_size == 1 and dp_size in (4, 8, 16):
            return f"dp{dp_size}_prefill"
        raise ValueError(
            "Raiden Qwen3.5 admission topology pcp8_prefill, "
            "pcp4_prefill, dp8_prefill, dp4_prefill, dp16_prefill, or "
            "pp{2,4,8}_prefill requires kv_producer with "
            "tensor_parallel_size=1 and either "
            "prefill_context_parallel_size in (4, 8), "
            "data_parallel_size=1; or "
            "prefill_context_parallel_size=1, "
            "data_parallel_size in (4, 8, 16); or "
            "pipeline_parallel_size in (2, 4, 8) alone; got "
            f"prefill_context_parallel_size={pcp_size}, "
            f"tensor_parallel_size={tp_size}, "
            f"data_parallel_size={dp_size}, "
            f"pipeline_parallel_size={pp_size}"
        )
    if tp_size == 1 and pcp_size == 1 and dp_size == 1 and pp_size in (2, 4, 8):
        return f"pp{pp_size}_decode"
    if pcp_size != 1 or tp_size not in (1, 2, 4) or dp_size not in (2, 4, 8, 16):
        raise ValueError(
            "Raiden Qwen3.5 admission topology dp{N}_decode or "
            "dp{N}tp{2,4}_decode requires kv_consumer with "
            "prefill_context_parallel_size=1, tensor_parallel_size in "
            "(1, 2, 4), data_parallel_size in (2, 4, 8, 16); got "
            f"prefill_context_parallel_size={pcp_size}, "
            f"tensor_parallel_size={tp_size}, "
            f"data_parallel_size={dp_size}"
        )
    if tp_size > 1:
        # Head-sharded decode engine: every TP worker is its own
        # destination unit (transfer_rank = tp_rank) and the producer
        # routes each KV head shard to it (TPU_RAIDEN_DST_SHARDS).
        return f"dp{dp_size}tp{tp_size}_decode"
    return f"dp{dp_size}_decode"


def validate_raiden_pd_config(
    vllm_config: Any,
    *,
    tp_size_fallback: int = 1,
    seq_on_lane: bool | None = None,
) -> str:
    """Validate Raiden Stage-3 P/D config and return the active admission flag.

    Shared by ``TpuPlatform.check_and_update_config``,
    ``TPURaidenConnector.__init__``, and
    ``TPURaidenConnectorWorker.register_runner`` so all three enforce identical
    rules, precedence, and exception types (``ValueError``).
    """
    for name, remedy in _REMOVED_KV_TRANSFER_ENVS.items():
        if name in os.environ:
            raise ValueError(f"{name} has been removed; {remedy}")

    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is not None:
        extra = getattr(kv_transfer_config, "kv_connector_extra_config", None)
        extras = [extra] if isinstance(extra, Mapping) else []
        if isinstance(extra, Mapping):
            for child in extra.get("connectors", []):
                if (
                    isinstance(child, Mapping)
                    and child.get("kv_connector") in RAIDEN_PD_CONNECTOR_NAMES
                    and isinstance(child.get("kv_connector_extra_config"), Mapping)
                ):
                    extras.append(child["kv_connector_extra_config"])
        if any("use_raiden_connector" in e for e in extras):
            raise ValueError(USE_RAIDEN_CONNECTOR_REMOVED_MESSAGE)

    # env_with_choices raises ValueError on anything but "raiden".
    _ = tpu_envs.TPU_KV_RESHARD_TRANSPORT

    if not tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER:
        raise ValueError(RAIDEN_PD_ADMISSION_REQUIRED_MESSAGE)
    selected = [name for name in RAIDEN_PD_ADMISSION_FLAGS if getattr(tpu_envs, name)]
    if len(selected) > 1:
        raise ValueError(f"{' and '.join(selected)} are mutually exclusive")
    if not selected:
        raise ValueError(RAIDEN_PD_ADMISSION_REQUIRED_MESSAGE)

    admission = selected[0]
    is_kimi = is_kimi_architecture(vllm_config)
    if admission == "TPU_RAIDEN_KIMIK3_ADMISSION" and not is_kimi:
        raise ValueError(
            "TPU_RAIDEN_KIMIK3_ADMISSION requires "
            "KimiK3ForConditionalGeneration or "
            "KimiLinearForCausalLM"
        )
    if (
        admission == "TPU_RAIDEN_KIMIK3_ADMISSION"
        and tpu_envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER
    ):
        raise ValueError("Kimi K3 does not support TPU_RAIDEN_POOL_TAGS_PER_LAYER")
    if is_kimi and admission != "TPU_RAIDEN_KIMIK3_ADMISSION":
        raise ValueError(
            "Kimi K3 requires TPU_RAIDEN_KIMIK3_ADMISSION=1; "
            "Qwen, GLM and DSv4 admission cannot admit Kimi"
        )

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if (
        admission == "TPU_RAIDEN_QWEN35_ADMISSION"
        and parallel_config is not None
        and kv_transfer_config is not None
    ):
        raiden_qwen35_admission_topology(
            vllm_config,
            is_producer=bool(kv_transfer_config.is_kv_producer),
            tp_size_fallback=tp_size_fallback,
            seq_on_lane=seq_on_lane,
        )

    return admission

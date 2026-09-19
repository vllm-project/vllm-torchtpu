# SPDX-License-Identifier: Apache-2.0
"""Configuration knobs and address helpers for the Raiden KV connector.

Small predicates over ``kv_connector_extra_config`` and ``vllm_torchtpu.envs``
that pick the transport (zmq/shm or Raiden, legacy or Stage-3), resolve the
reshard service addresses, and hold the Stage-3 tuning constants. The
connector, scheduler and worker halves all read these, so they live in a leaf
module that imports none of them.
"""

from functools import lru_cache
from typing import Any

from vllm.config import VllmConfig

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_STAGE3_FINISH_DEDUP_LIMIT = 4096
_STAGE3_D5_REGISTRATION_WAIT_S = 5.0
# tpu_sync GetRequestBlockStatusResponse.Status values reported by the
# reshard client's get_request_block_status probe.
_STAGE3_REGISTRY_STATUS_UNKNOWN = 1
_STAGE3_REGISTRY_STATUS_CANCELLED = 4
# Deferred-submit re-attempts ride cheap no-forward steps, which can cycle at
# kHz while a load waits; rate-limit the controller RPCs per parked request.
_STAGE3_REGISTRATION_REATTEMPT_MIN_INTERVAL_S = 0.05
_STAGE3_REGISTRATION_CANCELLED_ERROR = (
    "Request block registration was cancelled")
# Pool selection is vLLM policy expressed as request data: the connector names
# the manifest tags to move and raiden resolves them against both peers'
# registered manifests. Each GDN state class rides a sibling transfer with a
# T3.1: one transfer carries the FA payload plus every GDN state class
# under the base req_id/uuid; the tags below define the canonical order
# (FA first = H2D order rank 0).
_STAGE3_STATE_CLASS_TAGS = ("gdn.conv", "gdn.ssm")
_STAGE3_GLM_TRANSFER_POOL_TAGS = ("mla.nope", "mla.rope", "dsa.idx")
# Concurrent Stage-3 coordination RPCs per worker. The source controller
# handles concurrent coordinations (plan claims are per request), so a small
# pool keeps an admission burst from queueing behind one slow peer.
_STAGE3_SUBMIT_WORKERS = 4


def _get_extra_config(vllm_config: VllmConfig) -> dict[str, Any]:
    config = vllm_config.kv_transfer_config
    if config is None:
        return {}
    return config.kv_connector_extra_config or {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off"):
            return False
    return bool(value)


def stage3_fa_raiden_id_fields(
    *,
    job_name: str,
    engine_id: str,
    dp_rank: int,
    transfer_rank: int = 0,
    is_producer: bool = True,
    per_rank_unit: bool | None = None,
) -> dict[str, Any]:
    """Build the canonical Stage-3 FA work-unit identity.

    `per_rank_unit` names one unit per transfer rank (`<engine>-rank<k>`);
    it defaults to `is_producer`, and a pipeline-parallel consumer sets it
    because each of its stages is its own destination unit.
    """
    job_name = str(job_name).strip()
    engine_id = str(engine_id).strip()
    dp_rank = int(dp_rank)
    transfer_rank = int(transfer_rank)
    if not job_name:
        raise ValueError("Stage-3 Raiden job_name must not be empty")
    if not engine_id:
        raise ValueError("Stage-3 Raiden engine_id must not be empty")
    if dp_rank < 0:
        raise ValueError("Stage-3 Raiden dp_rank must be non-negative")
    if transfer_rank < 0:
        raise ValueError("Stage-3 Raiden transfer_rank must be non-negative")
    if per_rank_unit is None:
        per_rank_unit = is_producer
    replica_id = (f"{engine_id}-rank{transfer_rank}"
                  if per_rank_unit else engine_id)
    return {
        "job_name": job_name,
        "job_replica_id": replica_id,
        "data_name": "kv.fa",
        "data_replica_idx": dp_rank,
    }


def _use_raiden_stage3_transport() -> bool:
    return str(tpu_envs.TPU_KV_RESHARD_TRANSPORT).strip().lower() == "raiden"


def _use_per_layer_pool_tags() -> bool:
    return bool(tpu_envs.TPU_RAIDEN_POOL_TAGS_PER_LAYER)


def _pipeline_parallel_size(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    if parallel_config is None:
        return 1
    return int(parallel_config.pipeline_parallel_size or 1)


def _engine_is_pipeline(vllm_config: VllmConfig) -> bool:
    """A pipeline-parallel engine: each stage transfers its own layers,
    every token of them, so the transfer ranks are the pipeline stages."""
    return _pipeline_parallel_size(vllm_config) > 1


def _reshard_store_mode() -> bool:
    """In-engine hosting: the reshard service lives in each engine's rank-0
    worker (an in-process KVCacheStore), not an external controller."""
    return str(tpu_envs.TPU_RAIDEN_RESHARD_IMPL).strip().lower() == "store"


def _reshard_advertise_host() -> str:
    host = str(tpu_envs.TPU_RAIDEN_ADVERTISE_HOST).strip()
    if not host:
        raise ValueError("TPU_RAIDEN_ADVERTISE_HOST is required when "
                         "TPU_RAIDEN_RESHARD_IMPL=store")
    return host


def _reshard_service_port(dp_rank: int) -> int:
    return int(tpu_envs.TPU_RAIDEN_RESHARD_PORT_BASE) + int(dp_rank)


def _reshard_service_address(dp_rank: int) -> str:
    return f"{_reshard_advertise_host()}:{_reshard_service_port(dp_rank)}"


def _reshard_dispatch_address(dp_rank: int) -> str:
    port = int(tpu_envs.TPU_RAIDEN_STORE_DISPATCH_PORT_BASE) + int(dp_rank)
    return f"{_reshard_advertise_host()}:{port}"


def _resolved_reshard_controller_address(dp_rank: int) -> str:
    """The address the facade dials and the handoff advertises: the
    per-engine in-process store under store mode, else the role-level
    external controller from the environment."""
    if _reshard_store_mode():
        return _reshard_service_address(dp_rank)
    return str(tpu_envs.TPU_RAIDEN_CONTROLLER_ADDRESS).strip()


def _use_raiden_glm_admission() -> bool:
    return bool(tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER
                and tpu_envs.TPU_RAIDEN_GLM_ADMISSION)


def _prefix_aware_load_requested() -> bool:
    return bool(tpu_envs.TPU_RAIDEN_PREFIX_AWARE_LOAD)


@lru_cache(maxsize=1)
def _prefix_aware_load_supported() -> bool:
    """Whether the installed reshard client can encode suffix clips.

    Capability/version skew is a performance concern, not a serving-fatal
    condition: callers fall back to decode-local computation on partial hits.
    """
    try:
        from tpu_sync.api.torch import reshard_client as _reshard_client

        return bool(getattr(_reshard_client, "SUPPORTS_DST_SKIP_BYTES", False))
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Could not probe tpu-sync dst_skip_bytes support; prefix-aware "
            "loads will fall back to decode-local computation: %s", exc)
        return False


def _prefix_aware_load_enabled() -> bool:
    """Whether suffix-only Stage-3 loads are requested and supported."""
    return (_prefix_aware_load_requested() and _prefix_aware_load_supported())


def _use_raiden_connector(vllm_config: VllmConfig) -> bool:
    if _use_raiden_stage3_transport():
        return True
    extra_config = _get_extra_config(vllm_config)
    if "use_raiden_connector" in extra_config:
        return _as_bool(extra_config["use_raiden_connector"])
    return dist_utils.get_use_raiden_connector()

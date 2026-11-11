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
import dataclasses
from typing import Any

import torch.distributed.distributed_c10d as c10d
from vllm.distributed import parallel_state
from vllm.utils.network_utils import get_ip

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def get_host_ip() -> str:
    """Use `VLLM_HOST_IP` if set, otherwise use default network interface IP."""
    return get_ip()


def get_kv_transfer_port() -> str:
    return envs.TPU_KV_TRANSFER_PORT


def get_p2p_wait_pull_timeout() -> int:
    """KV-cache transfer timeout in seconds."""
    return envs.TPU_P2P_WAIT_PULL_TIMEOUT


# Min delta added on top of the producer lease when computing the reshard store's
# request-block registry TTL.
_RAIDEN_REGISTRY_TTL_MIN_DELTA_S = 60.0


def get_raiden_registry_ttl_s() -> float:
    """TTL, in seconds, for the raiden reshard store's request-block
    registry: the prefill lease (TPU_P2P_WAIT_PULL_TIMEOUT) plus a delta
    of max(60s, 10%). The registry must outlive the lease."""
    lease = float(get_p2p_wait_pull_timeout())
    return lease + max(_RAIDEN_REGISTRY_TTL_MIN_DELTA_S, 0.1 * lease)


def get_stage3_status_probe_s() -> float:
    """Interval, in seconds, at which a Stage-3 producer worker probes the
    source store's request-block registry for its registrations that have
    not reached a native terminal. A consumer's release-only cancellation
    retires only the registry row, so without the probe the registration
    (and vLLM's delayed block free waiting on it) sits until
    p2p_wait_pull_timeout. 0 disables the probe."""
    return max(envs.TPU_RAIDEN_STAGE3_STATUS_PROBE_S, 0.0)


def get_stage3_register_ahead_enabled() -> bool:
    """Whether a PCP, TP or DP Stage-3 producer publishes a request's send
    in the scheduler step that completes its prompt, so its workers
    register the KV blocks while that step runs instead of in the step
    after the finish response, where the registration first waits for the
    next batch's forward (default on). Pipeline producers always register
    ahead."""
    return bool(envs.TPU_RAIDEN_STAGE3_REGISTER_AHEAD)


def get_stage3_deferred_submit_enabled() -> bool:
    """Per-step deferred re-attempt for Stage-3 loads whose producer
    request-block registrations have not arrived yet (default on; the
    registration is out-of-band and rides a later producer scheduler step,
    so a consumer submit can legitimately race ahead of it), instead of the
    legacy
    bounded inline poll that blocks the worker step while waiting.

    Inline-load mode keeps the legacy poll: the worker blocks in-step for
    completion there, so nothing would re-drive a parked submit.
    """
    if get_raiden_inline_load():
        return False
    return envs.TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT


def get_stage3_registration_wait_s() -> float:
    """Total budget for deferred Stage-3 submit re-attempts, bounded by the
    overall pull timeout. The producer's registration rides a later producer
    scheduler step, so under load this must cover multiple step times."""
    val = envs.TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S
    if val <= 0:
        val = 30.0
    return min(val, float(get_p2p_wait_pull_timeout()))


def get_stage3_test_registration_delay_s() -> float:
    """TEST-ONLY fault injection: delay the producer's Stage-3 request-block
    registration by this many seconds to widen the out-of-band
    registration race (0 = disabled)."""
    return max(envs.TPU_RAIDEN_TEST_REGISTRATION_DELAY_S, 0.0)


def get_raiden_pool_staging_leases() -> int:
    """Concurrent pool-reshard transfers to provision bounded host staging for.

    The Raiden pool-reshard path used to shadow each rank's whole device KV
    pool in host memory; with an auto-sized pool that shadow (~25-30 GB/rank)
    overflows libtpu's 16 GiB premapped pool and D2H/H2D silently fall back to
    the staged-copy slow path (2-4x slower transfers, see
    RESHARD_BOUNDED_STAGING_DESIGN.md). With N > 0 the connector asks tpu-sync
    for a bounded staging arena per KV storage of N x the distinct blocks one
    transfer touches on that storage (pages per max-length request plus one
    state block per GDN group sharing it), leased per transfer. 0 keeps the
    full host mirror.
    """
    return envs.TPU_RAIDEN_POOL_STAGING_LEASES


def get_raiden_inline_load() -> bool:
    """Load remote KV before the first forward instead of a no-forward step."""
    return envs.TPU_RAIDEN_INLINE_LOAD


_RAIDEN_TELEMETRY_MODULE = None
_RAIDEN_TELEMETRY_CONFIGURED = False
_RAIDEN_TELEMETRY_IMPORT_ATTEMPTED = False


def get_raiden_telemetry_module():
    """Gets and caches the TPU Raiden C++ extension module."""
    global _RAIDEN_TELEMETRY_MODULE, _RAIDEN_TELEMETRY_IMPORT_ATTEMPTED
    if not _RAIDEN_TELEMETRY_IMPORT_ATTEMPTED:
        # Try to load raiden module only once and ignore other attempts
        _RAIDEN_TELEMETRY_IMPORT_ATTEMPTED = True
        try:
            from tpu_sync.api.torch import kv_cache_manager as kcm

            _RAIDEN_TELEMETRY_MODULE = kcm._torch_impl()
        except Exception as e:
            logger.warning("Failed to import TPU Raiden telemetry module: %s", e)
            return None
    return _RAIDEN_TELEMETRY_MODULE


def configure_raiden_telemetry(backends: list[str] | set[str] | None = None) -> None:
    """Configures TPU Raiden C++ telemetry backends if available."""
    global _RAIDEN_TELEMETRY_CONFIGURED
    try:
        telemetry = get_raiden_telemetry_module()
        if telemetry is not None and hasattr(telemetry, "configure_telemetry"):
            telemetry.configure_telemetry(backends)
            _RAIDEN_TELEMETRY_CONFIGURED = True
            logger.info("Configured TPU Raiden C++ telemetry backends")
    except Exception as e:
        logger.warning("Failed to configure TPU Raiden C++ telemetry: %s", e)


@dataclasses.dataclass(frozen=True)
class _GroupMetadata:
    kind: str | None = None
    role: str | None = None
    size: int | None = None
    ranks: tuple[int, ...] | None = None
    group_ranks: tuple[tuple[int, ...], ...] | None = None


def _find_vllm_group(group_name: str) -> Any | None:
    """Find a registered vLLM `GroupCoordinator` by `unique_name` or `device_group`."""
    group_ref = parallel_state._groups.get(group_name)
    if group_ref is not None:
        group = group_ref()
        if group is not None:
            return group
    for group_ref in parallel_state._groups.values():
        group = group_ref()
        if (
            group is not None
            and group.device_group is not None
            and group.device_group.group_name == group_name
        ):
            return group
    return None


def _resolve_group_metadata(group_name: str) -> _GroupMetadata:
    """Resolve `GroupCoordinator` or `c10d` metadata for `group_name`."""
    group = _find_vllm_group(group_name)
    if group is not None:
        role = group.unique_name.partition(":")[0]
        ranks = tuple(int(r) for r in group.ranks)
        group_ranks = (
            tuple(tuple(int(r) for r in sg) for sg in group.group_ranks)
            if group.group_ranks
            else None
        )
        return _GroupMetadata(
            kind="vllm_pg",
            role=role,
            size=int(group.world_size),
            ranks=ranks,
            group_ranks=group_ranks,
        )

    try:
        pg = c10d._resolve_process_group(c10d.GroupName(group_name))
        ranks = tuple(int(r) for r in c10d.get_process_group_ranks(pg))
    except (KeyError, ValueError, RuntimeError):
        return _GroupMetadata()
    if ranks:
        return _GroupMetadata(
            kind="c10d_pg",
            size=len(ranks),
            ranks=ranks,
        )
    return _GroupMetadata()


def _canonicalize_process_group_name(group_name: str, ordinal: int) -> str:
    """Map a rank-local process-group handle to a canonical token.

    TorchTPU tiles every symmetric subgroup collective's `replica_groups`
    across all `world_size` partitions via `GatherAllSubgroups`, so ranks in
    different subgroups compile identical all-device SPMD executables. When a
    vLLM `GroupCoordinator` is registered for `group_name` (either by
    `unique_name` or `device_group.group_name`), its full `group_ranks`
    partition is encoded directly so wrap-around subgroup layouts (such as
    `tpu7x` topology-aware PCP/TP meshes) match across all subgroups. Raw
    `c10d` groups fall back to subgroup size and rank-stride geometry.
    """
    meta = _resolve_group_metadata(group_name)
    strides = (
        tuple(meta.ranks[i + 1] - meta.ranks[i] for i in range(len(meta.ranks) - 1))
        if meta.ranks is not None and len(meta.ranks) > 1
        else ()
    )
    if meta.kind == "vllm_pg":
        if meta.group_ranks is not None:
            return (
                f"vllm_pg:{ordinal}:{meta.role}:size={meta.size}"
                f":group_ranks={meta.group_ranks}"
            )
        return f"vllm_pg:{ordinal}:{meta.role}:size={meta.size}:strides={strides}"
    if meta.kind == "c10d_pg":
        return f"c10d_pg:{ordinal}:size={meta.size}:strides={strides}"
    raise ValueError(f"Cannot resolve process group {group_name!r}")

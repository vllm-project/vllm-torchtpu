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
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch.distributed.distributed_c10d as c10d
from vllm.distributed import parallel_state

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.utils import RAIDEN_TELEMETRY_BACKENDS_CONFIG_KEY

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


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


def get_raiden_pool_staging_budget_mb() -> int:
    """Consumer-side host staging budget for the pool-reshard arena, in MiB
    per rank across every bounded KV storage.

    With N > 0 the consumer sizes each storage's arena to its share of the
    budget and asks the source controller to pull any request that does not
    fit one lease as a chain of sub-transfers (page-aligned windows of the
    request's full-attention pages, one engine plan each), so the arena no
    longer scales with max_model_len. 0 keeps whole-request leases.
    """
    return envs.TPU_RAIDEN_POOL_STAGING_BUDGET_MB


def get_raiden_pool_staging_window_blocks() -> int:
    """Explicit consumer-side sub-transfer window in full-attention
    destination pages; overrides the budget-derived window. 0 = none."""
    return envs.TPU_RAIDEN_POOL_STAGING_WINDOW_BLOCKS


def get_raiden_inline_load() -> bool:
    """Load remote KV before the first forward instead of a no-forward step."""
    return envs.TPU_RAIDEN_INLINE_LOAD


_RAIDEN_TELEMETRY_MODULE = None
_RAIDEN_TELEMETRY_CONFIGURED = False
_RAIDEN_TELEMETRY_CONFIGURED_BACKENDS: frozenset[str] | None = None
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
    global _RAIDEN_TELEMETRY_CONFIGURED, _RAIDEN_TELEMETRY_CONFIGURED_BACKENDS
    try:
        requested_backends = (
            None
            if backends is None
            else frozenset(b.strip().lower() for b in backends if b.strip())
        )
        if _RAIDEN_TELEMETRY_CONFIGURED and (
            requested_backends is None
            or requested_backends == _RAIDEN_TELEMETRY_CONFIGURED_BACKENDS
        ):
            return
        telemetry = get_raiden_telemetry_module()
        if telemetry is not None and hasattr(telemetry, "configure_telemetry"):
            telemetry.configure_telemetry(backends)
            _RAIDEN_TELEMETRY_CONFIGURED = True
            _RAIDEN_TELEMETRY_CONFIGURED_BACKENDS = requested_backends
            logger.info("Configured TPU Raiden C++ telemetry backends")
    except Exception as e:
        logger.warning("Failed to configure TPU Raiden C++ telemetry: %s", e)


DEFAULT_RAIDEN_TELEMETRY_BACKENDS: tuple[str, ...] = ("buffered",)
_RAIDEN_METRIC_PREFIX = "tpu_raiden_"
_RANK_LABEL_PATTERN = re.compile(r"(?:^|,)\s*(?:local_rank|rank)\s*=")


def get_raiden_worker_telemetry_rank(vllm_config: "VllmConfig | None") -> int | None:
    """Returns the worker rank for Raiden gauge tagging, or ``None``.

    On single-worker engines (uniproc executor, ``world_size <= 1``), the worker
    and scheduler connectors share one process and one C++ sample buffer;
    returning ``None`` keeps worker and scheduler drains on the same gauge
    series regardless of which role reads first. On multi-worker engines
    (TP, PP, or PCP > 1), returns ``parallel_config.rank`` so each worker
    exports a distinct ``local_rank`` series.
    """
    pc = getattr(vllm_config, "parallel_config", None)
    if pc is None:
        return None
    world_size = getattr(pc, "world_size", None)
    if not isinstance(world_size, int):
        tp = getattr(pc, "tensor_parallel_size", 1)
        pp = getattr(pc, "pipeline_parallel_size", 1)
        pcp = getattr(pc, "prefill_context_parallel_size", 1)
        if all(isinstance(v, int) for v in (tp, pp, pcp)):
            world_size = tp * pp * pcp
        else:
            return None
    if world_size <= 1:
        return None
    rank = getattr(pc, "rank", 0)
    return rank if isinstance(rank, int) else 0


def get_raiden_telemetry_backends(vllm_config: "VllmConfig | None") -> list[str]:
    """Returns the configured Raiden C++ telemetry backend list.

    Reads ``additional_config['raiden_telemetry_backends']`` when present,
    defaulting to ``['buffered']``. An explicit empty list ``[]`` disables all
    Raiden telemetry backends.
    """
    config = getattr(vllm_config, "additional_config", None)
    if isinstance(config, Mapping) and RAIDEN_TELEMETRY_BACKENDS_CONFIG_KEY in config:
        raw = config[RAIDEN_TELEMETRY_BACKENDS_CONFIG_KEY]
        if raw is None:
            return list(DEFAULT_RAIDEN_TELEMETRY_BACKENDS)
        if not isinstance(raw, list) or not all(isinstance(b, str) for b in raw):
            raise ValueError(
                f"additional_config.{RAIDEN_TELEMETRY_BACKENDS_CONFIG_KEY} "
                "must be a list of strings"
            )
        backends: list[str] = []
        for entry in raw:
            token = entry.strip().lower()
            if not token:
                raise ValueError(
                    f"additional_config.{RAIDEN_TELEMETRY_BACKENDS_CONFIG_KEY} "
                    "entries must be non-empty strings"
                )
            backends.append(token)
        return backends
    return list(DEFAULT_RAIDEN_TELEMETRY_BACKENDS)


def is_raiden_buffered_telemetry_enabled(
    backends: Sequence[str] | None = None,
) -> bool:
    """Whether vLLM should drain the Raiden C++ buffered exporter.

    ``"cloud_logging"`` shares the single C++ ``BufferedMetricsExporter`` with
    ``"buffered"``, and ``FlushToCloudLogging`` drains the same buffer. vLLM
    therefore drains only when ``"buffered"`` is configured without
    ``"cloud_logging"``.
    """
    resolved = DEFAULT_RAIDEN_TELEMETRY_BACKENDS if backends is None else backends
    tokens = {b.strip().lower() for b in resolved if b.strip()}
    return "buffered" in tokens and "cloud_logging" not in tokens


def _get_raiden_gauge_names(telemetry: Any) -> set[str]:
    """Returns normalized gauge metric names from Raiden metadata."""
    if not hasattr(telemetry, "get_metric_metadata"):
        return set()
    gauges: set[str] = set()
    try:
        for desc in telemetry.get_metric_metadata() or ():
            mtype = getattr(desc, "type", None)
            mtype_str = getattr(mtype, "name", str(mtype)).split(".")[-1].lower()
            if mtype_str == "gauge":
                name = str(desc.name)
                gauges.add(name)
                if not name.startswith(_RAIDEN_METRIC_PREFIX):
                    gauges.add(f"{_RAIDEN_METRIC_PREFIX}{name}")
    except Exception as e:
        logger.warning("Failed to inspect TPU Raiden metric metadata: %s", e)
    return gauges


def _add_worker_rank_to_gauge_key(raw_key: str, worker_rank: int) -> str:
    """Appends ``local_rank="<worker_rank>"`` if no rank label is present."""
    key = raw_key.strip()
    if key.endswith("}") and "{" in key:
        base, rest = key.split("{", 1)
        inner = rest[:-1].strip()
        if _RANK_LABEL_PATTERN.search(inner):
            return key
        rank_pair = f'local_rank="{worker_rank}"'
        return f"{base}{{{inner},{rank_pair}}}" if inner else f"{base}{{{rank_pair}}}"
    return f'{key}{{local_rank="{worker_rank}"}}'


def drain_raiden_metric_samples(
    backends: Sequence[str] | None = None,
    worker_rank: int | None = None,
) -> dict[str, list[float]]:
    """Drains buffered C++ Raiden telemetry samples for the current process.

    When ``worker_rank`` is provided, worker-side gauge samples that do not
    already carry a ``local_rank`` or ``rank`` label are tagged with
    ``local_rank="<worker_rank>"`` so gauges from multiple TP ranks do not
    overwrite each other during Prometheus observation.
    """
    if not is_raiden_buffered_telemetry_enabled(backends):
        return {}
    try:
        telemetry = get_raiden_telemetry_module()
        if telemetry is not None and hasattr(telemetry, "get_and_reset_metric_samples"):
            raiden_samples = telemetry.get_and_reset_metric_samples()
            if raiden_samples:
                gauge_names = (
                    _get_raiden_gauge_names(telemetry)
                    if worker_rank is not None
                    else set()
                )
                result: dict[str, list[float]] = {}
                for raw_name, values in raiden_samples.items():
                    if not values:
                        continue
                    key = str(raw_name)
                    if worker_rank is not None and gauge_names:
                        base_name = key.split("{", 1)[0].strip()
                        if base_name in gauge_names:
                            key = _add_worker_rank_to_gauge_key(key, worker_rank)
                    result.setdefault(key, []).extend(values)
                return result
    except Exception as e:
        logger.warning("Failed to collect TPU Raiden C++ telemetry: %s", e)
    return {}


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

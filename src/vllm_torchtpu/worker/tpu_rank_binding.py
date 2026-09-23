# SPDX-License-Identifier: Apache-2.0
"""Rank -> TPU device binding for vLLM workers.

``LOCAL_RANK`` selects which physical chip a worker opens: torch_tpu copies it
verbatim into ``TPU_VISIBLE_CHIPS``, libtpu turns that into
``--deepsea_hal_included_devs``, and resolves it against its device list. It is
a device-binding index and nothing else.

This module deliberately does **not** permute it. An earlier design remapped
``LOCAL_RANK`` so consecutive PCP ranks would land on ICI-adjacent chips, but
that was wrong in two ways:

  * It solved a data-placement problem in the device-binding layer. The PCP
    ring order is the mesh axis order built by ``get_or_create_pcp_mesh``,
    which follows PCP-group rank order. Reordering data is expressible there;
    moving chips underneath the ranks is not the same thing.
  * ``LOCAL_RANK`` restarts at 0 on every host, so a remap can permute chips
    within a host but can never move a rank across one -- and the
    configurations where ring ordering actually matters are multi-host.

Measured on tpu7x (ICI hops between consecutive PCP ranks, lower is better):

    2x2x1, pcp=8     native 4 (max 1)    legacy remap 6 (max 2)
    2x2x2, pcp=16    native 10 (max 2)   create_device_mesh 10 (max 2)

The remap never beat the native binding, and on a single host it made it
strictly worse. Removing it is both a simplification and a fix.

The native binding is not *guaranteed* optimal -- it follows PCI bus
enumeration order, which nothing promises will track ICI adjacency. It is used
because it measures at least as well as every alternative on every topology
tested. ``RANK_DEVICE_MAP`` in tpu_worker logs the resulting placement so a bad
ordering is visible rather than silent; if one ever appears, the fix belongs in
the PCP group's rank order, not here.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Multihost DP: the Ray executor owns the placement group, so it is the only
# component that knows which host each slice-global rank landed on. It hands
# every worker its own placement through these env vars instead of letting the
# worker re-derive it from the single-host "one contiguous chip block per DP
# engine" layout, which does not hold once the slice spans hosts.
SLICE_RANK_ENV = "TORCH_TPU_SLICE_RANK"
SLICE_LOCAL_RANK_ENV = "TORCH_TPU_SLICE_LOCAL_RANK"
SLICE_WORLD_SIZE_ENV = "TORCH_TPU_SLICE_WORLD_SIZE"
SLICE_LOCAL_WORLD_SIZE_ENV = "TORCH_TPU_SLICE_LOCAL_WORLD_SIZE"

_SLICE_BINDING_ENV_VARS = (
    SLICE_RANK_ENV,
    SLICE_LOCAL_RANK_ENV,
    SLICE_WORLD_SIZE_ENV,
    SLICE_LOCAL_WORLD_SIZE_ENV,
)


def _get_int_env(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class TpuWorkerBinding:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    init_rank: int
    init_world_size: int
    init_local_rank: int
    native_local_rank: int
    dp_rank: int = 0
    dp_size: int = 1
    local_rank_offset: int = 0

    def as_env(self) -> dict[str, str]:
        return {
            "RANK": str(self.rank),
            "LOCAL_RANK": str(self.local_rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
        }


def _get_spawned_pcp_local_rank(
    *,
    env: Mapping[str, str],
    local_rank_offset: int,
    local_world: int,
) -> int:
    """LOCAL_RANK a spawned PCP worker was handed by its parent."""
    local_rank_env = _get_int_env(env, "LOCAL_RANK", local_rank_offset)
    local_rank = local_rank_env - local_rank_offset
    if not 0 <= local_rank < local_world:
        raise ValueError(
            f"LOCAL_RANK={local_rank_env} is out of range for PCP local "
            f"world {local_world} with offset {local_rank_offset}."
        )
    return local_rank


def slice_binding_env(
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
) -> dict[str, str]:
    """Return env describing an executor-assigned slice-global placement.

    `rank`/`world_size` address the whole DP*TP TorchTPU slice, while
    `local_rank`/`local_world_size` address the chips of that slice on the
    worker's own host.
    """
    return {
        SLICE_RANK_ENV: str(rank),
        SLICE_LOCAL_RANK_ENV: str(local_rank),
        SLICE_WORLD_SIZE_ENV: str(world_size),
        SLICE_LOCAL_WORLD_SIZE_ENV: str(local_world_size),
    }


def _slice_binding_from_env(
    env: Mapping[str, str],
    *,
    parallel_config,
    rank: int,
    dp_rank: int,
    dp_size: int,
    pcp_size: int,
    local_rank_offset: int,
) -> TpuWorkerBinding | None:
    """Return the executor-assigned binding, or None when none was supplied."""
    present = [name for name in _SLICE_BINDING_ENV_VARS if env.get(name)]
    if not present:
        return None
    if len(present) != len(_SLICE_BINDING_ENV_VARS):
        missing = [name for name in _SLICE_BINDING_ENV_VARS if not env.get(name)]
        raise ValueError(
            "Executor-assigned TPU slice binding is incomplete: missing "
            f"{missing}. Set all of {list(_SLICE_BINDING_ENV_VARS)} or none."
        )
    if pcp_size > 1:
        raise NotImplementedError(
            "Prefill context parallelism is not supported together with "
            "multihost data parallelism."
        )

    world_size = parallel_config.world_size
    slice_rank = int(env[SLICE_RANK_ENV])
    slice_local_rank = int(env[SLICE_LOCAL_RANK_ENV])
    slice_world = int(env[SLICE_WORLD_SIZE_ENV])
    slice_local_world = int(env[SLICE_LOCAL_WORLD_SIZE_ENV])

    expected_world = world_size * dp_size
    if slice_world != expected_world:
        raise ValueError(
            f"{SLICE_WORLD_SIZE_ENV}={slice_world} does not match "
            f"world_size ({world_size}) x data_parallel_size ({dp_size}) "
            f"= {expected_world}."
        )
    if not 0 <= slice_rank < slice_world:
        raise ValueError(
            f"{SLICE_RANK_ENV}={slice_rank} is out of range for "
            f"slice world size {slice_world}."
        )
    if not 0 < slice_local_world <= slice_world:
        raise ValueError(
            f"{SLICE_LOCAL_WORLD_SIZE_ENV}={slice_local_world} is out of "
            f"range for slice world size {slice_world}."
        )
    if not 0 <= slice_local_rank < slice_local_world:
        raise ValueError(
            f"{SLICE_LOCAL_RANK_ENV}={slice_local_rank} is out of range for "
            f"slice local world size {slice_local_world}."
        )

    return TpuWorkerBinding(
        rank=slice_rank,
        local_rank=local_rank_offset + slice_local_rank,
        world_size=slice_world,
        local_world_size=local_rank_offset + slice_local_world,
        init_rank=rank,
        init_world_size=world_size,
        init_local_rank=slice_local_rank,
        native_local_rank=slice_local_rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        local_rank_offset=local_rank_offset,
    )


def get_tpu_worker_binding(
    parallel_config,
    rank: int,
    local_rank: int,
    *,
    env: Mapping[str, str],
    use_spawned_pcp_local_rank: bool = False,
) -> TpuWorkerBinding:
    world_size = parallel_config.world_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_size = (
        _get_int_env(env, "TORCH_TPU_DP_SIZE", 0) or parallel_config.data_parallel_size
    )
    local_rank_offset = _get_int_env(env, "TPU_LOCAL_RANK_OFFSET", 0)

    rank = int(rank)
    local_rank = int(local_rank)
    if dp_size > 1 and parallel_config.enable_expert_parallel:
        dp_rank = parallel_config.data_parallel_index
        assert dp_rank is not None, (
            "ParallelConfig.data_parallel_index must be resolved when "
            "data_parallel_size > 1"
        )

        executor_binding = _slice_binding_from_env(
            env,
            parallel_config=parallel_config,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            pcp_size=pcp_size,
            local_rank_offset=local_rank_offset,
        )
        if executor_binding is not None:
            return executor_binding

        # dp_rank is the *global* DP replica index (0..dp_size-1), spanning
        # every host under multi-host DP (vLLM's --data-parallel-start-rank
        # hybrid-LB launch). native_local_rank pins this worker to a
        # physical chip (see tpu_worker.py's TPU_VISIBLE_CHIPS), which is
        # numbered locally on *this* host (0..data_parallel_size_local-1), so
        # it must use this host's local DP index, not the raw global one --
        # otherwise a second host's ranks (e.g. 8-15) get pinned to chip
        # indices that don't exist there. data_parallel_rank_local is set by
        # CoreEngineProcManager/run_engine_core (vllm/v1/engine/{utils,core}
        # .py) directly from the process-spawn-time-known local slot -- the
        # same field GPU's worker.py reads for the identical purpose -- so no
        # TPU-specific plumbing is needed to get it here. The modulo is only
        # a defensive fallback for contexts where that never ran; it assumes
        # a contiguous host-major rank layout (true for --data-parallel-
        # start-rank, e.g. host 0 = ranks 0-7 and host 1 = ranks 8-15), unlike
        # the authoritative field.
        dp_rank_local = parallel_config.data_parallel_rank_local
        if dp_rank_local is None:
            dp_size_local = parallel_config.data_parallel_size_local or dp_size
            dp_rank_local = dp_rank % dp_size_local
        native_local_rank = dp_rank_local * world_size + local_rank
        global_rank = dp_rank * world_size + rank
        global_world = world_size * dp_size
        if pcp_size > 1 and use_spawned_pcp_local_rank:
            tpu_local_rank = _get_spawned_pcp_local_rank(
                env=env,
                local_rank_offset=local_rank_offset,
                local_world=global_world,
            )
        else:
            tpu_local_rank = native_local_rank
        return TpuWorkerBinding(
            rank=global_rank,
            local_rank=local_rank_offset + tpu_local_rank,
            world_size=global_world,
            local_world_size=local_rank_offset + global_world,
            init_rank=rank,
            init_world_size=world_size,
            init_local_rank=tpu_local_rank,
            native_local_rank=native_local_rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            local_rank_offset=local_rank_offset,
        )

    dp_rank = parallel_config.data_parallel_index or 0
    native_local_rank = dp_rank * world_size + local_rank
    if pcp_size > 1 and use_spawned_pcp_local_rank:
        tpu_local_rank = _get_spawned_pcp_local_rank(
            env=env,
            local_rank_offset=local_rank_offset,
            local_world=world_size,
        )
    else:
        # LOCAL_RANK stays replica-local; native_local_rank carries the
        # dp offset because it is the physical chip ordinal, and two dense
        # DP replicas must not resolve to the same chip.
        tpu_local_rank = local_rank
    # LOCAL_WORLD_SIZE must be the per-host chip count, not the global TP*PP
    # world size: on a single host they're equal, but across --nnodes hosts
    # each host only has world_size // nnodes chips, and PjRt bootstrap fails
    # ("PjRtClient is not initialized") if told to expect more local devices
    # than physically exist on this host.
    local_chip_count = parallel_config.local_world_size
    return TpuWorkerBinding(
        rank=rank,
        local_rank=local_rank_offset + tpu_local_rank,
        world_size=world_size,
        local_world_size=local_rank_offset + local_chip_count,
        init_rank=rank,
        init_world_size=world_size,
        init_local_rank=tpu_local_rank,
        native_local_rank=native_local_rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        local_rank_offset=local_rank_offset,
    )

# SPDX-License-Identifier: Apache-2.0
"""Topology-aware ordering of the PCP group's ranks.

The PCP attention kernel is a ring: rank *r* passes its KV block to rank
*r+1*, ``pcp_size - 1`` times (``next_rank = (my_id + 1) % pcp_size``). The
ring order is the mesh axis order, and that mesh is built from the PCP group's
rank list -- so reordering the group reorders the ring, with no kernel change.

Placement is *not* decided here. Workers bind to chips naturally
(``LOCAL_RANK == rank``); this only changes which position in the ring each
rank occupies. That distinction matters: ``LOCAL_RANK`` restarts at 0 on every
host and so can never express a cross-host ordering, whereas a rank is a
logical label and a group may span hosts.

Why this runs after startup rather than before it: choosing chips requires
knowing which PCI index yields which coordinates, and that is only observable
by opening a chip -- which is the choice itself. ``topology_aware_mesh`` needs
open chips and an initialised process group, so it can only answer once
placement is already fixed. Reordering data is the one lever still available
at that point.

Both the PCP and TP groups come from the same mesh, and they replace vLLM's
arithmetic layout rather than being reconciled with it. ``mesh[d, :, t]`` is a
PCP ring, ``mesh[d, p, :]`` is a TP lane, so one tensor gives every rank a
single ``(dp, pcp, tp)`` position: a rank's TP partners are by construction the
ranks sharing its ``pcp`` coordinate, and they therefore hold the same sequence
chunk. Ordering PCP alone would leave TP on vLLM's arange layout, and where the
two differ TP partners land at different ring positions -- holding different
sequence chunks while the TP all-reduce combines partials computed over
different tokens. Taking both axes from one mesh makes that unrepresentable.

Only ``tp``, ``pcp`` and (degenerately) ``dp``/``pp`` exist at the supported
scope, and at ``dp == 1, pp == 1`` the DP and PP groups are singletons and the
EP group is the whole world, so no other group constrains this layout.

Scoped to a single host with no KV transfer. ``pcp_rank`` is load-bearing
outside topology:

  * the P/D reshard planner reconstructs which prefill rank holds a shard as
    ``pcp_rank * tp_size + tp_rank`` (kv_transfer/v2/common.py), and
  * Raiden work units are named ``f"{engine_id}-rank{transfer_rank}"`` from
    the same composite, with the consumer re-deriving the name.

Both assume an ordering this module changes, so it is disabled whenever a KV
transfer config is present. Lifting that needs the real rank map to travel in
the handshake instead of being recomputed from sizes.
"""

import contextlib
from collections.abc import Iterator

import numpy

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def resolve_pcp_topology_order(
        vllm_config) -> dict[str, list[list[int]]] | None:
    """Groups to build for each axis, or ``None`` if PCP does not apply.

    Returns ``{"pcp": rings, "tp": lanes}`` -- the ``group_ranks`` vLLM
    should build for each ``group_name``, taken from the topology mesh.

    ``None`` means only one thing: this configuration does not use topology
    ordering at all. Anything that goes *wrong* raises, because a silent
    fallback to the arithmetic layout would hide a real problem behind a small
    and unmeasurable slowdown.
    """
    parallel_config = vllm_config.parallel_config
    pcp_size = int(
        getattr(parallel_config, "prefill_context_parallel_size", 1) or 1)

    # Not applicable: no ring exists.
    if pcp_size <= 1:
        return None

    # Not applicable: the P/D reshard planner and the Raiden work-unit names
    # both recompute a rank from pcp_rank rather than being told it, so they
    # would mis-address against any layout but the arithmetic one.
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        logger.info("Topology aware mesh not applied. Topology aware mesh is "
                    "currently not supported for disaggregated hosts.")
        return None

    # Not applicable: a ring crossing hosts needs multi-slice ordering over
    # DCN, not the single-slice mesh this asks for. PcpStaticSupportValidator
    # already rejects PCP with TPU_MULTIHOST_BACKEND, so only nnodes is
    # checked here.
    nnodes = int(getattr(parallel_config, "nnodes", 1) or 1)
    if nnodes > 1:
        logger.info(
            "Topology aware mesh not applied: nnodes=%d. Topology aware "
            "mesh is currently only supported for single host", nnodes)
        return None

    dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
    pp_size = int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1)

    # PcpStaticSupportValidator rejects PCP with pipeline parallelism before
    # any worker spawns, so reaching here with pp > 1 means that gate moved or
    # broke. Fail rather than build a layout whose pp axis we never modelled.
    if pp_size != 1:
        raise RuntimeError(
            f"Topology aware mesh assignment reached with "
            f"pipeline_parallel_size={pp_size}. "
            f"PcpStaticSupportValidator should have rejected PCP "
            f"with pipeline parallelism before worker startup.")

    # dp > 1 is only visible here under expert parallelism -- TPUWorker forces
    # data_parallel_size to 1 otherwise. That combination cannot build its
    # collectives (the DP replicas never join one torch world), so it would
    # fail a few lines later with a far less obvious error.
    if dp_size != 1:
        raise RuntimeError(
            f"Topology aware mesh assignment reached with "
            f"data_parallel_size={dp_size}. PCP with expert parallelism "
            f"across DP replicas is not supported: the replicas do not share "
            f"a torch distributed world, so the model-parallel groups cannot "
            f"be built.")

    # Ordered by increasing network intensity, matching vLLM's own
    # (dp, pp, pcp, tp) decomposition. jax maps the last axis onto the
    # highest-bandwidth link, which on v7 is the intra-chip core axis.
    mesh_shape = (dp_size, pcp_size, tp_size)

    import torch
    mesh = torch.tpu.topology_aware_mesh(mesh_shape)

    # Keyed by vLLM group_name so the caller can look up the groups for the
    # axis it is building. A ring is mesh[d, :, t] -- reading down the pcp
    # axis, hence the swap. A lane is mesh[d, p, :] -- along it, already
    # memory order. int() because these reach torch.distributed.new_group.
    group_ranks_by_name = {
        "pcp": [[int(r) for r in ring]
                for ring in numpy.swapaxes(mesh, 1, 2).reshape(-1, pcp_size)],
        "tp": [[int(r) for r in lane]
               for lane in numpy.asarray(mesh).reshape(-1, tp_size)],
    }

    # topology_aware_mesh promises a permutation of the global ranks. If it is
    # not one, some rank is missing or duplicated and the groups we build from
    # it would silently drop or double a worker.
    expected = list(range(dp_size * pcp_size * tp_size))
    for axis, groups in group_ranks_by_name.items():
        flat = sorted(r for group in groups for r in group)
        if flat != expected:
            raise RuntimeError(
                f"topology_aware_mesh({mesh_shape}) produced {axis} groups "
                f"{groups} whose ranks {flat} are not a permutation of "
                f"{expected}.")

    logger.info("TOPO_GROUPS | mesh_shape=%s -> %s", mesh_shape,
                group_ranks_by_name)
    return group_ranks_by_name


@contextlib.contextmanager
def pcp_topology_order(
        group_ranks_by_name: dict[str, list[list[int]]] | None
) -> Iterator[None]:
    """Install the mesh's groups while model-parallel groups are built.

    Patches ``init_model_parallel_group`` for the duration rather than
    permanently, so nothing is left installed for later callers to trip over.
    Only the axes present in ``group_ranks_by_name`` are replaced -- today
    ``pcp`` and ``tp``. Every other group, including ``dp``, ``pp`` and
    ``ep``, goes through the original function untouched; at the supported
    scope those are singletons or the whole world, so they cannot conflict
    with this layout.

    ``group_ranks_by_name`` is ``None`` only when topology aware mesh does
    not apply to this configuration. The patch is still installed in that
    case, because it also records what vLLM builds on its own: without it a
    baseline run logs no groups at all and there is nothing to diff a
    topology run against.
    """
    from vllm.distributed import parallel_state

    original = parallel_state.init_model_parallel_group
    replacements = group_ranks_by_name or {}

    def patched(group_ranks, local_rank, backend, *args, **kwargs):
        axis = kwargs.get("group_name")
        # Logged for every axis, whether or not it is being replaced, so the
        # log shows vLLM's own layout even when the mesh does not apply.
        logger.info("TOPO_GROUPS | %s: vLLM built %s", axis,
                    [list(g) for g in group_ranks])
        replacement = replacements.get(axis) if axis else None
        if replacement is not None:
            logger.info("TOPO_GROUPS | %s: replaced with %s", axis,
                        replacement)
            group_ranks = replacement
        return original(group_ranks, local_rank, backend, *args, **kwargs)

    parallel_state.init_model_parallel_group = patched
    try:
        yield
    finally:
        parallel_state.init_model_parallel_group = original

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

Deliberately narrow for now. ``pcp_rank`` is load-bearing outside topology:

  * the P/D reshard planner reconstructs which prefill rank holds a shard as
    ``pcp_rank * tp_size + tp_rank`` (kv_transfer/v2/common.py), and
  * Raiden work units are named ``f"{engine_id}-rank{transfer_rank}"`` from
    the same composite, with the consumer re-deriving the name.

Both assume an ordering this module changes, so it is disabled whenever a KV
transfer config is present. Lifting that needs the real rank map to travel in
the handshake instead of being recomputed from sizes.
"""

import contextlib
from collections.abc import Iterator, Sequence

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def reorder_groups(group_ranks: Sequence[Sequence[int]],
                   order: Sequence[int]) -> list[list[int]]:
    """Sort each group's ranks by their position in ``order``.

    Membership is preserved by construction -- we sort within a group, never
    move a rank between groups. That is what keeps TP/DP/PP grouping intact:
    taking ``mesh[d, :, t]`` directly would instead reassign which rank sits
    in which TP group whenever ``tp > 1``.

    Ranks missing from ``order`` keep their own value as the sort key, so a
    partial ordering degrades to "leave those where they are" rather than
    collapsing them together.
    """
    position = {int(rank): index for index, rank in enumerate(order)}
    return [
        sorted((int(r) for r in group), key=lambda r: position.get(r, r))
        for group in group_ranks
    ]


def resolve_pcp_topology_order(vllm_config) -> list[int] | None:
    """Global ranks in topology order, or None to leave the ring alone.

    Returns None rather than raising on every rejection path. A missing
    ordering costs at most a few ICI hops; on the topologies measured so far
    the natural order is already as good as the topology-aware one, so it is
    not worth failing startup over.
    """
    parallel_config = vllm_config.parallel_config
    pcp_size = int(
        getattr(parallel_config, "prefill_context_parallel_size", 1) or 1)
    if pcp_size <= 1:
        return None

    # Scoped off for P/D: the reshard planner and the Raiden work-unit names
    # both recompute a rank from pcp_rank rather than being told it.
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        logger.info(
            "PCP_ORDER | skipped: KV transfer config present, and the reshard "
            "protocol derives prefill ranks from pcp_rank arithmetic.")
        return None

    # Scoped off for multihost: PCP multihost does not run yet, so this path
    # would be untestable rather than merely unexercised.
    if envs.TPU_MULTIHOST_BACKEND:
        logger.info("PCP_ORDER | skipped: multihost backend.")
        return None

    dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
    # Ordered by increasing network intensity, matching vLLM's own
    # (dp, pp, pcp, tp) decomposition so a flat rank is a flat mesh position.
    mesh_shape = (dp_size, pcp_size, tp_size)

    try:
        import torch
        rank_mesh = torch.tpu.topology_aware_mesh(mesh_shape)
        order = [int(value) for value in rank_mesh.flatten().tolist()]
    except Exception as exc:  # noqa: BLE001 - startup must not fail here
        logger.warning(
            "PCP_ORDER | topology_aware_mesh failed, keeping natural ring "
            "order (mesh_shape=%s): %s", mesh_shape, exc)
        return None

    expected = dp_size * pcp_size * tp_size
    if sorted(order) != list(range(expected)):
        logger.warning(
            "PCP_ORDER | ignoring topology_aware_mesh result %s: not a "
            "permutation of 0..%d.", order, expected - 1)
        return None

    logger.info("PCP_ORDER | topology order for mesh_shape=%s: %s", mesh_shape,
                order)
    return order


@contextlib.contextmanager
def pcp_topology_order(order: Sequence[int] | None) -> Iterator[None]:
    """Reorder the PCP group's ranks while model-parallel groups are built.

    Patches ``init_model_parallel_group`` for the duration rather than
    permanently, so nothing is left installed for later callers to trip over.
    The ``group_name == "pcp"`` filter is what keeps TP/DP/PP untouched: they
    go through the original function with their rank lists unmodified.
    """
    if not order:
        yield
        return

    from vllm.distributed import parallel_state

    original = parallel_state.init_model_parallel_group

    def patched(group_ranks, local_rank, backend, *args, **kwargs):
        if kwargs.get("group_name") == "pcp":
            reordered = reorder_groups(group_ranks, order)
            if reordered != [list(g) for g in group_ranks]:
                logger.info("PCP_ORDER | applied %s -> %s",
                            [list(g) for g in group_ranks], reordered)
            group_ranks = reordered
        return original(group_ranks, local_rank, backend, *args, **kwargs)

    parallel_state.init_model_parallel_group = patched
    try:
        yield
    finally:
        parallel_state.init_model_parallel_group = original

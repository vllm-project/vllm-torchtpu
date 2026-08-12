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

Both the PCP and TP orders come from the same mesh. That is deliberate:
``mesh[d, :, t]`` is a PCP ring and ``mesh[d, p, :]`` is a TP lane, so taking
both from one tensor gives every rank a single ``(dp, pcp, tp)`` position and
makes it impossible for the two axes to disagree. Ordering PCP alone would
leave TP grouped by vLLM's arange layout, and where those differ TP partners
land at different ring positions -- they would hold different sequence chunks
and the TP all-reduce would combine partials computed over different tokens.

Scoped to a single host with no KV transfer. ``pcp_rank`` is load-bearing
outside topology:

  * the P/D reshard planner reconstructs which prefill rank holds a shard as
    ``pcp_rank * tp_size + tp_rank`` (kv_transfer/v2/common.py), and
  * Raiden work units are named ``f"{engine_id}-rank{transfer_rank}"`` from
    the same composite, with the consumer re-deriving the name.

Both assume an ordering this module changes, so it is disabled whenever a KV
transfer config is present. Lifting that needs the real rank map to travel in
the handshake instead of being recomputed from sizes.

Only ``tp == 1`` has run on hardware. ``tp > 1`` is correct by construction
here rather than by measurement, and the mesh grouping is checked against
vLLM's before anything is applied, so a disagreement abandons the reorder
rather than silently regrouping.
"""

import contextlib
from collections.abc import Iterator, Sequence

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def expected_groups(dp_size: int, pp_size: int, pcp_size: int,
                    tp_size: int) -> dict[str, list[frozenset[int]]]:
    """Membership of the groups vLLM will build, as sets.

    Mirrors ``all_ranks = arange(world).reshape(-1, dp, pp, pcp, tp)`` and the
    two slices taken from it in ``initialize_model_parallel``: TP is
    ``all_ranks.view(-1, tp)`` and PCP is ``transpose(3, 4).reshape(-1, pcp)``.
    Only membership is reproduced, never order -- order is what the mesh
    supplies. Named rather than cited by line so a vLLM bump does not silently
    stale the reference.

    This exists so both axes can be validated *before* anything is patched.
    vLLM builds TP first and PCP later, so a per-call decision could apply one
    axis and then reject the other, which is precisely the split we must not
    create.

    If vLLM ever changes its layout this replica stops matching and the whole
    reorder is abandoned, which is the safe direction to fail in.
    """
    world = dp_size * pp_size * pcp_size * tp_size
    ranks = list(range(world))

    def at(d, pp_i, p, t):
        return ranks[((d * pp_size + pp_i) * pcp_size + p) * tp_size + t]

    tp_groups = [
        frozenset(at(d, i, p, t) for t in range(tp_size))
        for d in range(dp_size) for i in range(pp_size)
        for p in range(pcp_size)
    ]
    pcp_groups = [
        frozenset(at(d, i, p, t) for p in range(pcp_size))
        for d in range(dp_size) for i in range(pp_size) for t in range(tp_size)
    ]
    return {"tp": tp_groups, "pcp": pcp_groups}


def mesh_group_orders(mesh, dp_size: int, pcp_size: int,
                      tp_size: int) -> dict[str, list[list[int]]]:
    """Slice a ``(dp, pcp, tp)`` rank mesh into per-axis group lists.

    ``mesh[d, :, t]`` is one PCP ring; ``mesh[d, p, :]`` is one TP lane. Both
    are read from the same tensor, so the two axes describe one coherent
    coordinate system: every rank has exactly one ``(dp, pcp, tp)`` position,
    and a rank's TP partners are by construction the ranks sharing its
    ``pcp`` coordinate.

    That joint derivation is the point. Reordering PCP alone would leave TP
    grouped by vLLM's arange layout, and the two can disagree -- enumerating
    all 40320 meshes for a world of 8 shows only 1.0% agree at pcp=4 tp=2.
    Where they disagree, TP partners land at different ring positions, so
    they hold different sequence chunks and the TP all-reduce combines
    partials computed over different tokens. Taking both axes from one mesh
    makes that unrepresentable.
    """
    pcp_rows = []
    for d in range(dp_size):
        for t in range(tp_size):
            pcp_rows.append([int(r) for r in mesh[d, :, t].tolist()])

    tp_rows = []
    for d in range(dp_size):
        for p in range(pcp_size):
            tp_rows.append([int(r) for r in mesh[d, p, :].tolist()])

    return {"pcp": pcp_rows, "tp": tp_rows}


def match_groups(group_ranks: Sequence[Sequence[int]],
                 rows: Sequence[Sequence[int]]) -> list[list[int]] | None:
    """Pair each of vLLM's groups with the mesh row of identical membership.

    Returns ``None`` if any group is unmatched, which means the mesh split the
    ranks differently from vLLM. Adopting a partial mapping would silently
    regroup, so the caller keeps the natural order instead.
    """
    by_members = {
        frozenset(int(r) for r in row): [int(r) for r in row]
        for row in rows
    }
    out: list[list[int]] = []
    for group in group_ranks:
        row = by_members.get(frozenset(int(r) for r in group))
        if row is None:
            return None
        out.append(row)
    return out


def resolve_pcp_topology_order(
        vllm_config) -> dict[str, list[list[int]]] | None:
    """Per-axis group orders from the topology mesh, or None to do nothing.

    Returns ``{"pcp": rows, "tp": rows}``, both sliced from one mesh so the
    two axes cannot disagree.

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

    # Same reason, checked independently of the backend flag: a ring crossing
    # hosts would need multi-slice ordering over DCN rather than the
    # single-slice default this asks for.
    nnodes = int(getattr(parallel_config, "nnodes", 1) or 1)
    if nnodes > 1:
        logger.info("PCP_ORDER | skipped: nnodes=%d, single host only.",
                    nnodes)
        return None

    dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)

    # No dp gate on purpose. Without EP the worker has already forced
    # data_parallel_size to 1 before this runs, and with EP the group
    # construction below fails regardless of what we return, so a gate here
    # would hide work rather than prevent anything.

    # Ordered by increasing network intensity, matching vLLM's own
    # (dp, pp, pcp, tp) decomposition so a flat rank is a flat mesh position.
    mesh_shape = (dp_size, pcp_size, tp_size)

    try:
        import torch
        rank_mesh = torch.tpu.topology_aware_mesh(mesh_shape)
        orders = mesh_group_orders(rank_mesh, dp_size, pcp_size, tp_size)
    except Exception as exc:  # noqa: BLE001 - startup must not fail here
        logger.warning(
            "PCP_ORDER | topology_aware_mesh failed, keeping natural ring "
            "order (mesh_shape=%s): %s", mesh_shape, exc)
        return None

    expected = list(range(dp_size * pcp_size * tp_size))
    for axis, rows in orders.items():
        if sorted(r for row in rows for r in row) != expected:
            logger.warning(
                "PCP_ORDER | ignoring topology_aware_mesh result: %s rows %s "
                "are not a partition of 0..%d.", axis, rows, expected[-1])
            return None

    # Both axes or neither: vLLM builds TP before PCP, so validating at
    # apply time could accept one and reject the other, leaving TP grouped by
    # arange while PCP follows the mesh -- the exact split that desynchronises
    # TP partners.
    pp_size = int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1)
    wanted = expected_groups(dp_size, pp_size, pcp_size, tp_size)
    for axis, rows in orders.items():
        got = sorted(sorted(row) for row in rows)
        exp = sorted(sorted(group) for group in wanted[axis])
        if got != exp:
            logger.warning(
                "PCP_ORDER | mesh %s grouping %s does not match vLLM's %s; "
                "keeping natural order on every axis.", axis, got, exp)
            return None

    logger.info("PCP_ORDER | topology orders for mesh_shape=%s: %s",
                mesh_shape, orders)
    return orders


@contextlib.contextmanager
def pcp_topology_order(
        orders: dict[str, list[list[int]]] | None) -> Iterator[None]:
    """Apply the mesh's group orders while model-parallel groups are built.

    Patches ``init_model_parallel_group`` for the duration rather than
    permanently, so nothing is left installed for later callers to trip over.
    Only the axes present in ``orders`` are touched -- today ``pcp`` and
    ``tp``. Everything else, including ``dp``, ``pp`` and ``ep``, goes through
    the original function with its rank list unmodified.

    Both axes are applied together or not at all. Applying one without the
    other is what would let TP partners drift onto different ring positions,
    so a membership mismatch on either axis abandons the whole reorder.
    """
    if not orders:
        yield
        return

    from vllm.distributed import parallel_state

    original = parallel_state.init_model_parallel_group

    def patched(group_ranks, local_rank, backend, *args, **kwargs):
        axis = kwargs.get("group_name")
        rows = orders.get(axis) if axis else None
        if rows is not None:
            current = [list(g) for g in group_ranks]
            matched = match_groups(group_ranks, rows)
            if matched is None:
                logger.warning(
                    "PCP_ORDER | %s: mesh rows %s do not match the groups %s; "
                    "keeping natural order.", axis, [list(r) for r in rows],
                    current)
            else:
                if matched != current:
                    logger.info("PCP_ORDER | %s applied %s -> %s", axis,
                                current, matched)
                group_ranks = matched
        return original(group_ranks, local_rank, backend, *args, **kwargs)

    parallel_state.init_model_parallel_group = patched
    try:
        yield
    finally:
        parallel_state.init_model_parallel_group = original

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram tables in HBM, looked up on device across the node-local DP group.

With ``TPU_DSV41_ENGRAM_HBM=first`` (default: only the first Engram layer,
which only layer 0 can hide; later layers keep the host path, overlapped with
the layers before them) or ``all``, each DP rank uploads the head shard of the
node-local Engram table that the host-offload path already loads into its host
memory (``ParallelEngramEmbedding`` with ``dp_size`` = node-local Engram DP
size) into HBM once, and every step's lookup becomes one SPMD op over the EP
mesh:

1. all-gather of the node's n-gram hashes (``axis_index_groups`` = node),
2. each rank gathers its heads' FP8 bytes / E8M0 scale bytes for every token
   of its node,
3. all-to-all back, so each rank holds every head of its own tokens in head
   shard order, dequantized to bf16 -- the rows ``Engram._ready_rows`` returns.

The host path's D2H of the hashes, its two gloo collectives (hash all-gather,
row all-to-all), the host gather and the H2D upload all leave the step; the
device lookup costs ~0.1 ms per Engram layer at 288 decode tokens per rank.
The tables take HBM away from the KV cache (~11.8 GiB per DeepSeek-V4.1-Flash
Engram layer per rank with an 8-rank node), so recipes raise
``--gpu-memory-utilization`` accordingly.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Single character: torch_tpu's local-shape helper iterates a spec entry.
_AXIS = "d"
# E8M0 byte 255 never occurs in a real table; it marks rows outside a shard.
_INVALID_SCALE = 255


def should_use_hbm_lookup(
    *,
    staged_raw: torch.Tensor | None,
    dp_size: int,
    layer_hash_index: int,
) -> bool:
    """Whether an Engram layer should serve its rows from HBM."""
    mode = (envs.TPU_DSV41_ENGRAM_HBM or "first").lower()
    if (
        mode == "off"
        or staged_raw is None
        or staged_raw.device.type != "tpu"
        or dp_size <= 1
    ):
        return False
    from vllm_torchtpu.distributed.ep_mesh import get_ep_group

    if get_ep_group() is None:
        return False
    return mode == "all" or layer_hash_index == 0


def gather_shard_meta(
    head_start: int, vocab_start: int, vocab_end: int, rows: int
) -> tuple[tuple[int, int, int, int], ...]:
    """(head_start, vocab_start, vocab_end, rows) of every EP rank, rank-ordered."""
    from vllm_torchtpu.distributed.ep_mesh import get_ep_group

    group = get_ep_group()
    if group is None:
        raise RuntimeError("Engram HBM lookup needs an expert-parallel group")
    gathered: list = [None] * int(group.world_size)
    torch.distributed.all_gather_object(
        gathered,
        (int(group.rank_in_group), head_start, vocab_start, vocab_end, rows),
        group=group.cpu_group,
    )
    return tuple(tuple(g[1:]) for g in sorted(gathered))


def upload_table(
    weight: torch.Tensor, scales: torch.Tensor, device: torch.device, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy this rank's host head shard to HBM, padded to ``rows`` rows.

    Every rank's shard must have one shape for the SPMD lookup; padded rows
    carry the invalid scale. Raw bytes cross in bounded chunks.
    """
    src = weight.view(torch.uint8)
    n, dim = src.shape
    table = torch.zeros((rows, dim), dtype=torch.uint8, device=device)
    table_scales = torch.full(
        (rows, scales.shape[1]), _INVALID_SCALE, dtype=torch.uint8, device=device
    )
    table_scales[:n].copy_(scales.to(device))
    step = max(1, (256 << 20) // dim)
    for start in range(0, n, step):
        end = min(start + step, n)
        table[start:end].copy_(src[start:end].to(device))
    return table, table_scales


def dequant(raw: jax.Array, scale: jax.Array, block_size: int) -> jax.Array:
    """JAX twin of ``engram.dequant_fp8_rows``, bit for bit."""
    values = jax.lax.bitcast_convert_type(raw, jnp.float8_e4m3fn).astype(jnp.float32)
    power = jax.lax.bitcast_convert_type(scale.astype(jnp.int32) << 23, jnp.float32)
    power = jnp.repeat(power, block_size, axis=-1)
    invalid = jnp.repeat(scale == _INVALID_SCALE, block_size, axis=-1)
    return jnp.where(invalid, 0.0, values * power).astype(jnp.bfloat16)


def make_lookup(
    mesh,
    num_tokens: int,
    n_cols: int,
    part_cols: int,
    rows: int,
    dim: int,
    block_size: int,
    node_size: int,
    meta: tuple[tuple[int, int, int, int], ...],
):
    """The SPMD lookup over ``mesh``: (hashes [W*T, C] i32, tables
    [W*rows, dim] u8, scales [W*rows, B] u8, ranks [W, 1] i32), all sharded on
    axis 0, -> bf16 rows [W*T, part_cols * node_size, dim]."""
    world = len(meta)
    if world % node_size:
        raise ValueError(f"EP size {world} is not a multiple of node size {node_size}")
    blocks = dim // block_size
    cols = part_cols * node_size
    head_start = jnp.asarray(np.array([m[0] for m in meta], np.int32))
    vocab_start = jnp.asarray(np.array([m[1] for m in meta], np.int32))
    vocab_end = jnp.asarray(np.array([m[2] for m in meta], np.int32))
    groups = (
        [list(range(g, g + node_size)) for g in range(0, world, node_size)]
        if world > node_size
        else None
    )

    def local(h, table, scales, rank):
        # A rank operand rather than lax.axis_index, which lowers to
        # partition-id and is rejected on this SPMD path.
        me = rank[0, 0]
        hashes = jax.lax.all_gather(
            h, _AXIS, axis=0, tiled=False, axis_index_groups=groups
        )  # [node, T, C]
        heads = head_start[me] + jnp.arange(part_cols, dtype=jnp.int32)
        ids = jnp.take(hashes, jnp.clip(heads, 0, n_cols - 1), axis=2)
        valid = (
            (heads < n_cols)[None, None, :]
            & (ids >= vocab_start[me])
            & (ids < vocab_end[me])
        )
        row = jnp.clip(ids - vocab_start[me], 0, rows - 1).reshape(-1)
        raw = jnp.take(table, row, axis=0).reshape(
            node_size, num_tokens, part_cols, dim
        )
        scale = jnp.take(scales, row, axis=0).reshape(
            node_size, num_tokens, part_cols, blocks
        )
        scale = jnp.where(valid[..., None], scale, jnp.uint8(_INVALID_SCALE))
        raw = jax.lax.all_to_all(raw, _AXIS, 0, 0, tiled=True, axis_index_groups=groups)
        scale = jax.lax.all_to_all(
            scale, _AXIS, 0, 0, tiled=True, axis_index_groups=groups
        )
        # [source rank, T, heads, ...] -> [T, source rank * heads, ...]: the
        # head shard order of the host exchange.
        raw = jnp.transpose(raw, (1, 0, 2, 3)).reshape(num_tokens, cols, dim)
        scale = jnp.transpose(scale, (1, 0, 2, 3)).reshape(num_tokens, cols, blocks)
        return dequant(raw, scale, block_size)

    def lookup(h: jax.Array, w: jax.Array, s: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(_AXIS, None),) * 4,
            out_specs=P(_AXIS, None, None),
            check_vma=False,
        )(h, w, s, rank)

    return lookup


@lru_cache(maxsize=16)
def build_lookup_op(
    layer: int,
    num_tokens: int,
    n_cols: int,
    part_cols: int,
    rows: int,
    dim: int,
    block_size: int,
    node_size: int,
    meta: tuple[tuple[int, int, int, int], ...],
):
    """Torch op for ``make_lookup`` over the EP mesh, one per layer and shape."""
    from torch_tpu._internal.pallas import jax_op

    from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh

    mesh = build_ep_mesh(axis_name=_AXIS)
    if mesh is None:
        raise RuntimeError("Engram HBM lookup needs an expert-parallel mesh")
    lookup = make_lookup(
        mesh, num_tokens, n_cols, part_cols, rows, dim, block_size, node_size, meta
    )
    cols = part_cols * node_size
    op = jax_op(
        f"pallas::dsv41_engram_hbm_l{layer}_t{num_tokens}",
        lookup,
        mesh=mesh,
        input_partition_specs=(P(_AXIS, None),) * 4,
        output_partition_specs=P(_AXIS, None, None),
    )

    def fake(h, w, s, rank):
        return h.new_empty((num_tokens, cols, dim), dtype=torch.bfloat16)

    op.register_fake(fake)
    logger.info(
        "[DSV41] Engram HBM lookup: layer=%d tokens=%d heads=%d rows/rank=%d",
        layer,
        num_tokens,
        cols,
        rows,
    )
    return op

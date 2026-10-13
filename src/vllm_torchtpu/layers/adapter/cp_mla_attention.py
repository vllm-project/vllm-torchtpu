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
"""DCP (Decode Context Parallelism) attention for MLA: the head gather and
the log-sum-exp merge that bracket the sharded attention call.

The MLA counterpart of `cp_attention.py`.

Every collective runs on tensors already in torch's hands, outside
the exported op.

Each DCP rank stores a shard of the KV cache, so it can only produce a
*partial* attention result.

  1. `all_gather_heads` widens this rank's query head slice to the whole
     group's heads, so every rank's heads are attended against every rank's
     KV shard.
  2. the kernel runs against this rank's shard alone, returning a partial
     output and its log-sum-exp.
  3. `merge_lse_partials_scatter_heads` folds the partials into the exact
     whole-context result and narrows the head axis back to this rank's own.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs


def all_gather_heads(q: torch.Tensor) -> torch.Tensor:
    """Widen a rank's query head slice to the whole DCP group's heads.

    DCP doesn't add a new parallel dimension—it borrows from TP. It
    reallocates dcp ways from heads to sequence positions, reducing
    head sharding from tp to tp // dcp.

    Args:
      q: `[num_tokens, tp_local_num_heads, dim]`, this rank's head slice.

    Returns:
      `[num_tokens, tp_local_num_heads * dcp_size, dim]`, identical
      on every rank. Returned unchanged without a DCP group.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return q
    return group.all_gather(q.contiguous(), dim=1)


def all_gather_q_nope_pe_heads(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused all-gather of `(ql_nope, q_pe)` across DCP ranks in a single call."""
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return ql_nope, q_pe
    q_dcp = group.all_gather(torch.cat([ql_nope, q_pe], dim=-1), dim=1)
    return (
        q_dcp[..., :kv_lora_rank].contiguous(),
        q_dcp[..., kv_lora_rank:].contiguous(),
    )


def merge_lse_partials_scatter_heads(
    output: torch.Tensor,
    lse: torch.Tensor,
    is_backbone: bool = True,
    query_start_loc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine this rank's partial attention output with its peers'.

    With per-rank normalized outputs `o_r` and `lse_r = m_r + log(l_r)`, the
    exact whole-context result is `o = sum_r softmax_r(lse) * o_r`. Correctness
    does not depend on the order the ranks are combined in.

    Two collectives:

      1. all-gather the log-sum-exps. They are `[num_tokens, num_heads]` f32 --
         Leaves every rank holding every `lse_r`.
      2. reduce-scatter the weighted outputs

    With TPU_DCP_LSE_MERGE_A2A set, the backbone's steps of at most
    TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS tokens instead take
    `merge_lse_partials_all_to_all`, which gives the same result. With
    TPU_DCP_MERGE_CHUNKED=n, the backbone's steps of at least
    TPU_DCP_MERGE_CHUNKED_MIN_TOKENS tokens run the reduce-scatter merge as n
    token chunks (`_merge_chunked`), which gives the same per-row result.

    Args:
      output: `[num_tokens, tp_local_num_heads * dcp_size, kv_lora_rank]`,
        this rank's partial.
      lse: `[num_tokens, tp_local_num_heads * dcp_size]`, the matching log-sum-exps.
        Entries are `-inf` for tokens this rank owns none of the top-k for;
        such a rank was still fed a dummy key (the kernel requires
        `kv_len >= 1`) and its output must not be weighted in.
      is_backbone: True if the caller is the backbone, the only model that is
        traced separately per token bucket. Only then may the merge compare
        the token count to TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS (or
        TPU_DCP_MERGE_CHUNKED_MIN_TOKENS), since that comparison is fixed at
        trace time. Pass False from the MTP drafter,
        which is traced once for every bucket: it always takes the
        reduce-scatter, exactly as with TPU_DCP_LSE_MERGE_A2A unset.
      query_start_loc: the step's `[num_reqs + 1]` query starts, padded with
        the real token count (so its last entry is that count). With
        TPU_DCP_LSE_MERGE_A2A_REAL_ROWS the packed all-to-all sends only the
        real tokens' rows; None keeps every padded row.

    Returns:
      `[num_tokens, tp_local_num_heads, kv_lora_rank]` in `output.dtype` --
      this rank's head slice of the whole-context result.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    # This `if` runs while tracing, so its answer is baked into the graph.
    # Only the backbone is traced once per bucket group, so only it can pick
    # by size: the all-to-all wins at decode sizes, the reduce-scatter above
    # because it sums in the network. The MTP drafter has one graph for every
    # bucket; comparing its token count would pin that graph to the first
    # size traced. `is_backbone` is checked before the token count, so the
    # drafter never compares it.
    if (
        envs.TPU_DCP_LSE_MERGE_A2A
        and is_backbone
        and output.shape[0] <= envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS
    ):
        return merge_lse_partials_all_to_all(
            output, lse, group, query_start_loc=query_start_loc
        )

    # The drafter cannot compare the token count in torch, but JAX exports
    # each of its buckets at a concrete shape: one op picks the exchange
    # there (TPU_MTP_DCP_MERGE_SIZED).
    if (
        envs.TPU_MTP_DCP_MERGE_SIZED
        and not is_backbone
        and output.element_size() == 2
        and int(group.world_size) in _SIZED_MERGE_OPS
    ):
        return _SIZED_MERGE_OPS[int(group.world_size)](output, lse)

    # Same trace-time rule as above: only the backbone may split by size.
    chunks = _merge_chunks(output.shape[0]) if is_backbone else 1
    if chunks > 1:
        return _retain_rank(_merge_chunked(output, lse, group, chunks))
    return _retain_rank(cp_lse_ag_out_rs(output, lse, group))


def _merge_chunks(num_tokens: int) -> int:
    """How many token chunks `TPU_DCP_MERGE_CHUNKED` splits this bucket's
    merge into: 1 (unsplit) unless the bucket reaches
    `TPU_DCP_MERGE_CHUNKED_MIN_TOKENS` and splits into equal chunks whose
    length is a multiple of 8 (the f32 sublane tile)."""
    chunks = envs.TPU_DCP_MERGE_CHUNKED
    if (
        chunks <= 1
        or num_tokens < envs.TPU_DCP_MERGE_CHUNKED_MIN_TOKENS
        or num_tokens % (8 * chunks)
    ):
        return 1
    return chunks


def _merge_chunked(
    output: torch.Tensor, lse: torch.Tensor, group: Any, chunks: int
) -> torch.Tensor:
    """The reduce-scatter merge (`cp_lse_ag_out_rs`) over `chunks`
    independent token chunks.

    Every token's merge depends on its own row alone (the lse all-gather,
    logsumexp, weight and reduce-scatter sum all run per token), so splitting
    the token axis changes no per-row arithmetic: each chunk gathers its own
    rows' lse, weights them and reduce-scatters them exactly as the unsplit
    merge does. What changes is the schedule. The f32 reduce-scatter is a
    SparseCore-offloaded async collective while the lse gather and the weight
    multiply are TensorCore work, so chunk `k`'s reduce-scatter can run under
    chunk `k + 1`'s gather and multiply instead of after all of them.
    """
    step = output.shape[0] // chunks
    if _lse_gathered_once(output.shape[0]):
        return _merge_chunked_lse_gathered(output, lse, group, chunks)
    if envs.TPU_DCP_MERGE_RS_BF16:
        # `cp_lse_ag_out_rs` op for op, except the scatter-sum's wire dtype.
        lse_f32 = lse.to(torch.float32)
        pieces = [
            _weighted_reduce_scatter(
                output[c * step : (c + 1) * step],
                lse_f32[c * step : (c + 1) * step],
                _gather_lse(lse_f32[c * step : (c + 1) * step], group),
                group,
            )
            for c in range(chunks)
        ]
        return torch.cat(pieces, dim=0)
    pieces = [
        cp_lse_ag_out_rs(
            output[c * step : (c + 1) * step], lse[c * step : (c + 1) * step], group
        )
        for c in range(chunks)
    ]
    return torch.cat(pieces, dim=0)


def _lse_gathered_once(num_tokens: int) -> bool:
    """Whether `TPU_DCP_MERGE_CHUNKED_LSE_AG` applies to this chunked bucket
    (fixed at trace time; only the backbone chunks)."""
    cap = envs.TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS
    return bool(envs.TPU_DCP_MERGE_CHUNKED_LSE_AG) and (cap <= 0 or num_tokens <= cap)


def _gather_lse(lse_f32: torch.Tensor, group: Any) -> torch.Tensor:
    """`[T, H]` f32 -> `[world, T, H]`: every rank's lse rows (the gather
    `cp_lse_ag_out_rs` runs)."""
    return group.all_gather(lse_f32.unsqueeze(0).contiguous(), dim=0)


def _weighted_reduce_scatter(
    output: torch.Tensor, lse_f32: torch.Tensor, lses: torch.Tensor, group: Any
) -> torch.Tensor:
    """`cp_lse_ag_out_rs` after its lse gather, op for op: `lses` is
    `[world, T, H]` for exactly `output`'s `T` rows."""
    merged_lse = torch.logsumexp(lses, dim=0)
    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    weight = torch.exp(lse_f32 - merged_lse).unsqueeze(-1)
    weighted = output.to(torch.float32) * weight
    weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))
    return _scatter_sum(weighted, group).to(output.dtype)


def _scatter_sum(weighted: torch.Tensor, group: Any) -> torch.Tensor:
    """Sum `[T, H, V]` f32 `weighted` over the ranks and keep this rank's
    head block `[T, H / world, V]`: the merge's reduce-scatter.

    `TPU_DCP_MERGE_RS_BF16` (not bitwise) sends the weighted partial in bf16:
    1 reduce-scatters it in bf16 (the collective accumulates in bf16); 2
    exchanges the same bf16 bytes by all-to-all and sums the received ranks
    locally in f32. Either halves the bytes of the f32 reduce-scatter.
    """
    mode = envs.TPU_DCP_MERGE_RS_BF16
    if mode == 1:
        return group.reduce_scatter(weighted.to(torch.bfloat16).contiguous(), dim=1)
    if mode == 2:
        world = int(group.world_size)
        tokens, heads, vdim = weighted.shape
        send = (
            weighted.to(torch.bfloat16)
            .reshape(tokens, world, heads // world, vdim)
            .transpose(0, 1)
        )
        return _all_to_all(send, group).to(torch.float32).sum(dim=0)
    return group.reduce_scatter(weighted.contiguous(), dim=1)


def _merge_chunked_lse_gathered(
    output: torch.Tensor, lse: torch.Tensor, group: Any, chunks: int
) -> torch.Tensor:
    """`_merge_chunked` with one lse all-gather for every chunk
    (`TPU_DCP_MERGE_CHUNKED_LSE_AG`).

    The per-chunk gathers are synchronous TensorCore collectives with no data
    edge to the earlier chunks, so the scheduler issues most of them before
    chunk 0's weight multiply and each one's fixed cost delays the first
    SparseCore reduce-scatter, the head of the SC-bound chain. One gather pays
    that cost once. It only moves rows, so each chunk's rows of it are bit for
    bit what the chunk's own gather returned, and the per-row merge arithmetic
    is unchanged.
    """
    step = output.shape[0] // chunks
    lse_f32 = lse.to(torch.float32)
    gathered = _gather_lse(lse_f32, group)
    pieces = [
        _weighted_reduce_scatter(
            output[c * step : (c + 1) * step],
            lse_f32[c * step : (c + 1) * step],
            gathered[:, c * step : (c + 1) * step],
            group,
        )
        for c in range(chunks)
    ]
    return torch.cat(pieces, dim=0)


def _all_to_all(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Block `d` of `x`'s leading axis goes to rank `d`; block `r` of the
    result came from rank `r`."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    return funcol.wait_tensor(out)


def merge_lse_partials_all_to_all(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any,
    query_start_loc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same result as the reduce-scatter merge, via all-to-all.

    Each rank sends head block `d` of its partial and lse to rank `d`, then
    merges its own heads locally in f32. Faster for small token counts.
    """
    world = int(group.world_size)
    num_tokens, num_heads, vdim = output.shape
    local = num_heads // world
    # Destination-major: [world, T, local, V] and [world, T, local].
    o_send = output.reshape(num_tokens, world, local, vdim).transpose(0, 1)
    l_send = lse.to(torch.float32).reshape(num_tokens, world, local).transpose(0, 1)
    if envs.TPU_DCP_LSE_MERGE_A2A_PACK and output.element_size() == 2:
        real_tokens = None
        if envs.TPU_DCP_LSE_MERGE_A2A_REAL_ROWS and query_start_loc is not None:
            # Padded with the real count, so the last entry is that count.
            real_tokens = query_start_loc[-1:].to(torch.int32)
        o_recv, l_recv = _all_to_all_packed(o_send, l_send, group, real_tokens)
    else:
        o_recv = _all_to_all(o_send, group)
        l_recv = _all_to_all(l_send, group)
    return _merge_received(o_recv, l_recv).to(output.dtype)


def _all_to_all_packed(
    o_send: torch.Tensor,
    l_send: torch.Tensor,
    group: Any,
    real_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(_all_to_all(o_send), _all_to_all(l_send))` in ONE all-to-all
    (`TPU_DCP_LSE_MERGE_A2A_PACK`).

    `o_send` is `[world, T, local, V]` in a 2-byte dtype and `l_send` the f32
    `[world, T, local]`. The exchange carries int16 words: the partial
    reinterpreted as is, then each destination's `T * local` lse bit patterns
    split into a block of high halves and a block of low halves (integer
    shifts and masks, so no lane interleave), zero padded to whole `V`-wide
    rows after its `T * local` partial rows. The words stay integers until
    they are unpacked: a float concat may be lowered to a select/max that
    canonicalizes the NaN bit patterns half an f32 can take. The exchange moves
    bits, so both halves come back bit for bit what the two separate exchanges
    return; only the f32 lse's own collective (latency-bound at a few KB) is
    gone, for `ceil(2 * T * local / V)` extra rows on the partial's.

    With `real_tokens` (`TPU_DCP_LSE_MERGE_A2A_REAL_ROWS`, an int32 `[1]` count
    of real tokens, the rest of the `T` rows padding), the lse rows lead each
    destination block and a ragged all-to-all sends only them plus the real
    tokens' `real_tokens * local` partial rows: at decode `b1` a 16-row bucket
    holds 4 real rows, so three quarters of the payload stays home. Received
    rows past that are zero. Real rows come back bit for bit; a padding row's
    partial is zero instead of whatever the kernel left there (its lse still
    arrives whole).
    """
    world, tokens, local, vdim = o_send.shape
    rows = tokens * local
    words = rows * 2
    extra = -(-words // vdim)
    o_words = o_send.reshape(world, rows, vdim).view(torch.int16)
    bits = l_send.contiguous().view(torch.int32).reshape(world, rows)
    hi = (bits >> 16).to(torch.int16)
    # The low half as a signed 16-bit value, so the narrowing is exact.
    lo = (((bits & 0xFFFF) ^ 0x8000) - 0x8000).to(torch.int16)
    l_words = torch.nn.functional.pad(
        torch.cat([hi, lo], dim=1), (0, extra * vdim - words)
    )
    if real_tokens is not None:
        send = torch.cat([l_words.reshape(world, extra, vdim), o_words], dim=1)
        recv = _ragged_all_to_all(send, extra + real_tokens * local, group)
        o_recv = recv[:, extra:].reshape(world, tokens, local, vdim)
        o_recv = o_recv.view(o_send.dtype)
        l_recv_words = recv[:, :extra].reshape(world, extra * vdim)
    else:
        send = torch.cat([o_words, l_words.reshape(world, extra, vdim)], dim=1)
        recv = _all_to_all(send, group)
        o_recv = recv[:, :rows].reshape(world, tokens, local, vdim).view(o_send.dtype)
        l_recv_words = recv[:, rows:].reshape(world, extra * vdim)
    hi = l_recv_words[:, :rows].to(torch.int32)
    lo = l_recv_words[:, rows:words].to(torch.int32)
    l_recv = (
        ((hi << 16) | (lo & 0xFFFF)).view(torch.float32).reshape(world, tokens, local)
    )
    return o_recv, l_recv


_RAGGED_A2A_OPS: dict = {}
# This rank's DCP index as an int32 [1] device tensor: the exchange takes it
# as data (XLA's SPMD export refuses `axis_index`'s partition-id).
_DCP_RANK: torch.Tensor | None = None


def _ragged_all_to_all(x: torch.Tensor, rows: torch.Tensor, group: Any) -> torch.Tensor:
    """`_all_to_all(x)` for `x` `[world, R, V]`, moving only the first `rows`
    (int32 `[1]`, the same on every rank) rows of each block; the rest of each
    received block is zero. One `jax.lax.ragged_all_to_all` over the DCP mesh
    axis (`TPU_DCP_LSE_MERGE_A2A_REAL_ROWS`)."""
    world, block, vdim = x.shape
    prebuild_ragged_all_to_all(world)
    op = _RAGGED_A2A_OPS[world]
    return op(x.reshape(world * block, vdim).contiguous(), rows, _DCP_RANK).reshape(
        world, block, vdim
    )


def _ragged_fake(
    x: torch.Tensor, rows: torch.Tensor, rank: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(x)


def prebuild_ragged_all_to_all(world: int) -> None:
    """Build (register) the real-rows exchange op for a DCP group of `world`
    ranks at model build, not inside a trace."""
    global _DCP_RANK
    if world not in _RAGGED_A2A_OPS:
        from vllm_torchtpu.distributed.dcp import dcp_rank_operand

        # The same per-device int32 [1] rank buffer TPU_DCP_RANK_OPERAND hands
        # the DCP attention and indexer ops: one buffer for every consumer.
        _DCP_RANK = dcp_rank_operand(torch.device("tpu"))
        _RAGGED_A2A_OPS[world] = _build_ragged_all_to_all(world)


def _retain_rank(out: torch.Tensor) -> torch.Tensor:
    """`out` plus a zero read from the rank buffer, on the merge branch that
    does not run the ragged exchange: the backbone warmup hands a bucket over
    only between traces that lift the same buffers (as
    `small_all_reduce_op.retain`). A dead read would be eliminated, so the
    zero rides on `out`."""
    from vllm_torchtpu.compilation import shape_variants

    if _DCP_RANK is None or not shape_variants.in_warmup():
        return out
    value = _DCP_RANK[0].to(out.dtype)
    return out + (value - value)


def _build_ragged_all_to_all(world: int):
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, get_or_create_dcp_mesh
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

    mesh = get_or_create_dcp_mesh()
    assert mesh.shape[DCP_AXIS_NAME] == world, (mesh.shape, world)
    # Each rank's own block, as the DCP attention op lays out its per-rank
    # partials (`_SPARSE_MLA_DCP_OUTPUT_PARTITION_SPECS`); the exchange runs
    # within each lane's DCP group.
    rank_spec = P(DCP_AXIS_NAME)

    def body(x, rows, rank):
        block = x.shape[0] // world
        me = rank[0].astype(jnp.int32)
        n = jnp.clip(rows[0], 0, block).astype(jnp.int32)
        sizes = jnp.full((world,), n, jnp.int32)
        return jax.lax.ragged_all_to_all(
            x,
            jnp.zeros_like(x),
            jnp.arange(world, dtype=jnp.int32) * block,  # block d -> rank d
            sizes,
            jnp.full((world,), me * block, jnp.int32),  # lands at block `me`
            sizes,  # every rank sends the same count
            axis_name=DCP_AXIS_NAME,
        )

    def fn(x: jax.Array, rows: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(rank_spec, P(), rank_spec),
            out_specs=rank_spec,
            check_vma=False,
        )(x, rows, rank)

    return sharded_jax_op(
        f"pallas::dcp_merge_ragged_all_to_all_{world}",
        fn,
        mesh=mesh,
        input_partition_specs=(rank_spec, P(), rank_spec),
        output_partition_specs=rank_spec,
        fake=_ragged_fake,
    )


def _zero_nan(o_f32: torch.Tensor) -> torch.Tensor:
    """Zero NaN entries of the received partials.

    A rank that owns no position of a row ran its kernel on a dummy key and may
    return NaN there; the merge gives it weight 0, but NaN * 0 = NaN. Selecting
    on the partial's own value (not on a weight or lse mask) keeps the select
    inside the merge's existing fusion: the decode HLO gains no op. Finite
    partials are untouched, bit for bit. A NaN partial of nonzero weight (never
    produced by an owning rank) would also be zeroed rather than propagated.
    """
    return torch.where(torch.isnan(o_f32), torch.zeros_like(o_f32), o_f32)


def _merge_received(o_recv: torch.Tensor, l_recv: torch.Tensor) -> torch.Tensor:
    """`[R, T, h, V]` partials and `[R, T, h]` lse -> the f32 `[T, h, V]` merge."""
    finite = torch.isfinite(l_recv)
    shift = torch.where(finite, l_recv, torch.full_like(l_recv, float("-inf"))).amax(
        dim=0
    )
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    weights = torch.where(
        finite, torch.exp(l_recv - shift.unsqueeze(0)), torch.zeros_like(l_recv)
    )
    denom = weights.sum(dim=0)
    weights = weights / torch.where(denom > 0, denom, torch.ones_like(denom))
    # The compiler lowers multiply-then-sum to a dot that rounds f32 operands to
    # bf16. Splitting the weights into a bf16 part and a residual keeps it exact.
    o_f32 = _zero_nan(o_recv.to(torch.float32))
    w_hi = weights.to(torch.bfloat16).to(torch.float32)
    w_lo = weights - w_hi
    return (o_f32 * w_hi.unsqueeze(-1)).sum(dim=0) + (o_f32 * w_lo.unsqueeze(-1)).sum(
        dim=0
    )


# ===================================================================== #
# The MTP drafter's sized merge (TPU_MTP_DCP_MERGE_SIZED).
# ===================================================================== #

# world -> the torch op over the DCP mesh; built at model build
# (`prebuild_sized_merge`), never inside a trace.
_SIZED_MERGE_OPS: dict = {}


def _rs_weighted(o: jax.Array, own_lse: jax.Array, lses: jax.Array) -> jax.Array:
    """`cp_lse_ag_out_rs`'s per-element arithmetic before its sum: `o` `[..., V]`
    a partial, `own_lse` `[...]` its lse, `lses` `[world, ...]` every rank's
    lse for the same rows. f32 `o * exp(own_lse - logsumexp(lses))`, zero where
    the weight is not positive (a zero-weight partial may be NaN)."""
    merged = jax.nn.logsumexp(lses, axis=0)
    weight = jnp.exp(own_lse - merged)[..., None]
    weighted = o.astype(jnp.float32) * weight
    return jnp.where(weight > 0, weighted, jnp.zeros_like(weighted))


def _pack_words(o_send: jax.Array, l_send: jax.Array) -> tuple[jax.Array, int]:
    """`[world, T, local, V]` 2-byte partial and f32 `[world, T, local]` lse ->
    int16 `[world, T*local + extra, V]` (`_all_to_all_packed`'s layout): the
    partial's words, then each destination's lse high halves and low halves,
    zero padded to whole V-wide rows. Integer ops only, so every bit moves."""
    world, tokens, local, vdim = o_send.shape
    rows = tokens * local
    words = rows * 2
    extra = -(-words // vdim)
    o_words = jax.lax.bitcast_convert_type(o_send, jnp.int16).reshape(world, rows, vdim)
    bits = jax.lax.bitcast_convert_type(l_send, jnp.int32).reshape(world, rows)
    hi = (bits >> 16).astype(jnp.int16)
    lo = (((bits & 0xFFFF) ^ 0x8000) - 0x8000).astype(jnp.int16)
    l_words = jnp.pad(
        jnp.concatenate([hi, lo], axis=1), ((0, 0), (0, extra * vdim - words))
    )
    return jnp.concatenate(
        [o_words, l_words.reshape(world, extra, vdim)], axis=1
    ), extra


def _unpack_words(
    recv: jax.Array, tokens: int, local: int, dtype
) -> tuple[jax.Array, jax.Array]:
    """Inverse of `_pack_words` on the received blocks: `[world, T, local, V]`
    partials in `dtype` and the f32 `[world, T, local]` lse, bit for bit."""
    world, _, vdim = recv.shape
    rows = tokens * local
    o_recv = jax.lax.bitcast_convert_type(recv[:, :rows], dtype).reshape(
        world, tokens, local, vdim
    )
    l_words = recv[:, rows:].reshape(world, -1)
    hi = l_words[:, :rows].astype(jnp.int32)
    lo = l_words[:, rows : 2 * rows].astype(jnp.int32)
    l_recv = jax.lax.bitcast_convert_type((hi << 16) | (lo & 0xFFFF), jnp.float32)
    return o_recv, l_recv.reshape(world, tokens, local)


def sized_merge_local(
    o: jax.Array, lse: jax.Array, *, world: int, a2a_max_tokens: int, axis_name: str
) -> jax.Array:
    """One rank's merge inside a `shard_map` over the DCP axis.

    `o` `[T, H, V]` (2-byte dtype) and `lse` `[T, H]` are this rank's partial
    and lse over all `H = world * local` heads; returns this rank's merged head
    block `[T, local, V]` in `o.dtype`. `T` is concrete here, so the exchange is
    chosen per bucket:

    - `T <= a2a_max_tokens`: head block `d` of the partial and lse goes to rank
      `d` in one packed all-to-all; each rank weights the received partials as
      the reduce-scatter merge would (same lse gather values, logsumexp, exp,
      multiply, zero select) and adds them in rank order 0..world-1.
    - otherwise: the reduce-scatter merge, op for op (`cp_lse_ag_out_rs`): lse
      all-gather, weight, f32 reduce-scatter over heads.

    The two differ only in the order the `world` weighted f32 terms are summed.
    """
    tokens, heads, vdim = o.shape
    local = heads // world
    lse = lse.astype(jnp.float32)
    if tokens > a2a_max_tokens:
        lses = jax.lax.all_gather(lse, axis_name, axis=0)
        weighted = _rs_weighted(o, lse, lses)
        merged = jax.lax.psum_scatter(
            weighted, axis_name, scatter_dimension=1, tiled=True
        )
        return merged.astype(o.dtype)
    o_send = o.reshape(tokens, world, local, vdim).transpose(1, 0, 2, 3)
    l_send = lse.reshape(tokens, world, local).transpose(1, 0, 2)
    send, _ = _pack_words(o_send, l_send)
    recv = jax.lax.all_to_all(send, axis_name, 0, 0, tiled=True)
    o_recv, l_recv = _unpack_words(recv, tokens, local, o.dtype)
    # Rank r's weight reads every rank's lse for this rank's heads, which is
    # exactly l_recv `[world, T, local]`.
    terms = _rs_weighted(o_recv, l_recv, l_recv)
    # An explicit chain of adds (a reduce over the rank axis may be lowered to
    # a default-precision dot that rounds its f32 operands to bf16).
    acc = terms[0]
    for r in range(1, world):
        acc = acc + terms[r]
    return acc.astype(o.dtype)


def prebuild_sized_merge(world: int) -> None:
    """Register the drafter's sized merge op for a DCP group of `world` ranks
    (TPU_MTP_DCP_MERGE_SIZED). Call at model build, outside any trace."""
    if not envs.TPU_MTP_DCP_MERGE_SIZED or world <= 1 or world in _SIZED_MERGE_OPS:
        return
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, get_or_create_dcp_mesh
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

    mesh = get_or_create_dcp_mesh()
    assert mesh.shape[DCP_AXIS_NAME] == world, (mesh.shape, world)
    cap = int(envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS)
    # Each rank's own rows, as the DCP attention op lays out its partials
    # (`_SPARSE_MLA_DCP_OUTPUT_PARTITION_SPECS`).
    spec = P(DCP_AXIS_NAME)

    def body(o, lse):
        return sized_merge_local(
            o, lse, world=world, a2a_max_tokens=cap, axis_name=DCP_AXIS_NAME
        )

    def fn(output: jax.Array, lse: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )(output, lse)

    def fake(output: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        tokens, heads, vdim = output.shape
        return output.new_empty((tokens, heads // world, vdim))

    _SIZED_MERGE_OPS[world] = sharded_jax_op(
        f"pallas::mtp_dcp_merge_sized_{world}",
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec),
        output_partition_specs=spec,
        fake=fake,
    )

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
"""The attention half of the MSA DCP forward over one KV head as one SPMD
program (additional_config minimax_dcp_attend_merge, from the r10 line).

Inside a ``shard_map`` over the DCP group: the query heads and candidates are
gathered (by collective permutes on a DCP 2 pair at decode sizes, whose XLA
all-gathers block the TensorCore for their latency), `msa_dcp.msa_attend_dcp`
attends over the group's heads, each rank's head block of the partials and
its LSEs go to their owner by all-to-all (permutes on a pair at decode
sizes), and `merge_partials` merges them token-major in one Pallas pass (in
f32, rounded once), unrotating a 4-bit cache's partials. The torch path
(`ops.msa.merge_head_partials`) instead all-gathers the LSEs and
reduce-scatters weighted partials. Not bitwise against it: the merge sums in
f32 and rounds once.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels.msa_dcp import msa_attend_dcp, rank_array

# Decode-sized steps of a DCP pair move their candidates and partials by
# collective permutes (latency-bound), larger ones by XLA collectives.
PERMUTE_MAX_TOKENS = 1024


_PARTIALS_BLOCK_TOKENS = 256


def _merge_partials_kernel(lse_ref, parts_ref, *rest, rotated):
    """One token block of `merge_partials`: ``lse_ref [dcp, Bt, Hl]``,
    ``parts_ref [dcp, Hl, Bt, D]``, (``hadamard [D, D]``), ``o_ref [Bt,
    Hl * D]``."""
    o_ref = rest[-1]
    dcp, num_heads, _, head_dim = parts_ref.shape
    lse = lse_ref[...]
    # torch.logsumexp over the ranks; a token no rank attended (every lse
    # -inf) gets NaN weights, which the where below turns into zeros.
    peak = jnp.max(lse, axis=0)
    peak = jnp.where(jnp.isinf(peak), 0.0, peak)
    total = peak + jnp.log(jnp.sum(jnp.exp(lse - peak[None]), axis=0))
    weight = jnp.exp(lse - total[None])
    weight = jnp.where(weight > 0, weight, 0.0)  # NaN (no rank attended) -> 0
    for h in range(num_heads):
        merged = None
        for r in range(dcp):
            w = jnp.broadcast_to(weight[r, :, h : h + 1], parts_ref.shape[2:])
            # A plain product: a rank that attended nothing for a token has
            # a zero partial (`msa_sparse_attention_lse`), never NaN, so no
            # select guards it (a compare and a select per element, ~20% of
            # the kernel's bundles).
            term = parts_ref[r, h].astype(jnp.float32) * w
            merged = term if merged is None else merged + term
        merged = merged.astype(o_ref.dtype)
        if rotated:
            merged = jnp.dot(merged, rest[0][...], preferred_element_type=jnp.float32)
            merged = (merged * (1.0 / head_dim)).astype(o_ref.dtype)
        o_ref[:, h * head_dim : (h + 1) * head_dim] = merged


@functools.partial(jax.jit, static_argnames=("block_tokens", "interpret"))
def merge_partials(
    parts: jax.Array,
    lse: jax.Array,
    hadamard: jax.Array | None = None,
    *,
    block_tokens: int = _PARTIALS_BLOCK_TOKENS,
    interpret: bool = False,
) -> jax.Array:
    """The log-sum-exp merge of every DCP rank's partial attention output of
    this rank's heads, token-major: ``parts [dcp, Hl, T, D]`` (rank ``r``'s
    head-major partial), ``lse [dcp, T, Hl]`` f32 -> ``[T, Hl * D]``.

    Weights and sum in f32, rounded once; a rank's partial of a token it
    attended nothing for must be zero (as `msa_sparse_attention_lse` writes
    it), not NaN. A 4-bit cache's partials (in the rotated basis) are
    unrotated by its bf16 `hadamard` (``x @ H / D``, as
    `ops.msa._rotate_heads`). One pass that writes the layout the output
    projection reads: the reduce-scatter merge weighted the whole ``[dcp *
    Hl, T, D]`` partial and XLA relaid it out for the token-major output
    (~43 + 24 us per layer at 8K tokens, c4 twin j-db4e365d)."""
    dcp, num_heads, num_tokens, head_dim = parts.shape
    block_t = max(1, min(num_tokens, block_tokens))
    rotated = hadamard is not None
    return pl.pallas_call(
        functools.partial(_merge_partials_kernel, rotated=rotated),
        grid=(pl.cdiv(num_tokens, block_t),),
        in_specs=[
            pl.BlockSpec((dcp, block_t, num_heads), lambda i: (0, i, 0)),
            pl.BlockSpec((dcp, num_heads, block_t, head_dim), lambda i: (0, 0, i, 0)),
            *(
                [pl.BlockSpec((head_dim, head_dim), lambda i: (0, 0))]
                if rotated
                else []
            ),
        ],
        out_specs=pl.BlockSpec((block_t, num_heads * head_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((num_tokens, num_heads * head_dim), parts.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_merge_partials",
    )(lse.astype(jnp.float32), parts, *([hadamard] if rotated else []))


def gather_by_permute(x: jax.Array, rank: jax.Array, axis: str, dcp: int) -> jax.Array:
    """Every rank's `x` in rank order, ``[dcp, *x.shape]``, by ``dcp - 1``
    collective permutes (rank ``r`` gets rank ``r + s``'s `x` from shift
    ``s``). For decode-sized operands of a DCP pair (two cores of one
    chip): each XLA all-gather there blocks the TensorCore for its latency
    (3.2 us per layer at c4 b3, j-c71e5160). Across chips (DCP 8) the
    permutes' starts cost more than the all-gather (c64 twin j-f92f5540)."""
    out = jnp.zeros((dcp, *x.shape), x.dtype)
    out = jax.lax.dynamic_update_index_in_dim(out, x, rank, 0)
    for shift in range(1, dcp):
        perm = [(src, (src - shift) % dcp) for src in range(dcp)]
        received = jax.lax.ppermute(x, axis, perm)
        out = jax.lax.dynamic_update_index_in_dim(
            out, received, (rank + shift) % dcp, 0
        )
    return out


def all_to_all_by_permute(
    x: jax.Array, rank: jax.Array, axis: str, dcp: int
) -> jax.Array:
    """`jax.lax.all_to_all` of ``x [dcp, ...]`` along its leading axis (block
    ``d`` goes to rank ``d``; block ``s`` of the result came from rank
    ``s``), by `gather_by_permute`'s permutes. For decode-sized blocks: each
    XLA all-to-all costs its latency on the TensorCore (~1.3 us per layer at
    c4 decode, j-c71e5160)."""
    out = jnp.zeros_like(x)
    own = jax.lax.dynamic_index_in_dim(x, rank, 0, False)
    out = jax.lax.dynamic_update_index_in_dim(out, own, rank, 0)
    for shift in range(1, dcp):
        perm = [(src, (src - shift) % dcp) for src in range(dcp)]
        send = jax.lax.dynamic_index_in_dim(x, (rank - shift) % dcp, 0, False)
        received = jax.lax.ppermute(send, axis, perm)
        out = jax.lax.dynamic_update_index_in_dim(
            out, received, (rank + shift) % dcp, 0
        )
    return out


def _exchange_partials(
    parts: jax.Array, lses: jax.Array, rank: jax.Array, axis: str, dcp: int
) -> tuple[jax.Array, jax.Array]:
    """`all_to_all_by_permute` of ``parts [dcp, Hl, T, D]`` and ``lses [dcp,
    T, Hl]`` f32 in one set of permutes: the LSEs' bits ride as one more uint16
    head row (a permute copies bytes; every permute start costs ~1 us on the
    TensorCore at decode, c4 twin j-6bbe97e2)."""
    num_ranks, num_heads, num_tokens, head_dim = parts.shape
    if parts.dtype.itemsize != 2 or 2 * num_heads > head_dim:
        raise ValueError(
            f"16-bit partials with 2 * heads <= head_dim, got {parts.dtype} "
            f"{parts.shape}"
        )
    # As 16-bit integers: float copies may canonicalize NaN or flush
    # subnormal bit patterns (the LSE halves are arbitrary bits).
    words = jax.lax.bitcast_convert_type(parts, jnp.uint16)
    bits = jax.lax.bitcast_convert_type(lses, jnp.uint16)  # [dcp, T, Hl, 2]
    bits = bits.reshape(num_ranks, 1, num_tokens, 2 * num_heads)
    bits = jnp.pad(bits, ((0, 0), (0, 0), (0, 0), (0, head_dim - 2 * num_heads)))
    both = all_to_all_by_permute(
        jnp.concatenate([words, bits], axis=1), rank, axis, dcp
    )
    bits = both[:, num_heads, :, : 2 * num_heads].reshape(
        num_ranks, num_tokens, num_heads, 2
    )
    return (
        jax.lax.bitcast_convert_type(both[:, :num_heads], parts.dtype),
        jax.lax.bitcast_convert_type(bits, lses.dtype),
    )


def _gather_heads(q: jax.Array, rank: jax.Array, axis: str, dcp: int) -> jax.Array:
    """The group's head-major query heads ``[dcp * Hl, T, D]`` from this
    rank's ``q [Hl, T, D]``. A pair swaps its halves by one collective
    permute: XLA's all-gather of the 16 MB c4 / c16 queries took 158 us per
    layer in the twin (j-094494a7), a permute plus the concatenation ~51 us
    on the same core pair in a standalone bench (j-a7bf7869; all-gather ~90)."""
    if dcp != 2:
        return jax.lax.all_gather(q, axis, axis=0, tiled=True)
    other = jax.lax.ppermute(q, axis, [(0, 1), (1, 0)])
    # [q, other, q] from row `rank * Hl` is the pair's heads in rank order.
    both = jnp.concatenate([q, other, q])
    return jax.lax.dynamic_slice_in_dim(both, rank * q.shape[0], 2 * q.shape[0])


def attend_and_merge(
    rank: jax.Array,
    q: jax.Array,
    kv_cache: jax.Array,
    candidates: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    scale_cache: jax.Array | None = None,
    hadamard: jax.Array | None = None,
    *,
    axis: str,
    dcp: int,
    sm_scale: float,
    topk: int,
    interpret: bool = False,
) -> jax.Array:
    """The attention half of the DCP forward over one KV head, inside a
    ``shard_map`` over `axis` (the DCP group): this rank's query heads ``q
    [T, Hl, D]`` (rotated for a 4-bit cache, then pass its `scale_cache` and
    `hadamard`) and its `msa_dcp.to_slots` candidates ``[2 * topk, T]`` ->
    its heads' merged output ``[T, Hl * D]``.

    The query heads and candidates are gathered, `msa_attend_dcp` runs over
    the group's heads, and each rank's head block of the partial and its LSEs
    go to their owner by all-to-all for `merge_partials`. As XLA collectives
    in one program they move the head-major buffers as laid out; torch's
    all-to-all of them cost ~890 us per layer at 8K tokens in the r10 line's
    c4 twin."""
    num_tokens, num_heads, head_dim = q.shape
    rank = rank_array(rank)[0]
    q_all = _gather_heads(q.transpose(1, 0, 2), rank, axis, dcp)
    # Decode-sized operands of a pair move by permutes (latency); a prefill
    # step's partials by XLA's all-to-all, which writes them in place.
    small = dcp == 2 and num_tokens <= PERMUTE_MAX_TOKENS
    if small:
        cand_all = gather_by_permute(candidates, rank, axis, dcp)
    else:
        cand_all = jax.lax.all_gather(candidates, axis, axis=0)
    out, lse = msa_attend_dcp(
        q_all.transpose(1, 0, 2),
        kv_cache,
        cand_all,
        seq_lens,
        block_tables,
        cu_q_lens,
        sm_scale=sm_scale,
        topk=topk,
        rank=rank,
        dcp=dcp,
        scale_cache=scale_cache,
        interpret=interpret,
    )
    out, lse = out[:, :num_tokens], lse[:, :num_tokens]
    # Block r of each result came from rank r: its partial of this rank's heads.
    parts = out.reshape(dcp, num_heads, num_tokens, head_dim)
    lses = lse.reshape(dcp, num_heads, num_tokens).transpose(0, 2, 1)
    if small:
        parts, lses = _exchange_partials(parts, lses, rank, axis, dcp)
    else:
        parts = jax.lax.all_to_all(parts, axis, 0, 0, tiled=True)
        lses = jax.lax.all_to_all(lses, axis, 0, 0, tiled=True)
    return merge_partials(parts, lses, hadamard, interpret=interpret)

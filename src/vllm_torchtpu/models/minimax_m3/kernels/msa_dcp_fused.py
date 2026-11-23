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
"""The attention half of a small DCP-2 decode step in one Pallas call.

`msa_dcp.attend_and_merge` over one KV head (c4 / c16 decode, fp8 caches;
also 4-bit caches) runs as
~25 XLA ops and 4 Pallas calls per layer: collective permutes of the query
heads and the candidates with their dynamic-update-slice glue, the candidate
merge (two compare-reduce fusions), the work-list kernel, the sparse
attention kernel, the permute of the partials (bitcasts, slices, an update)
and the LSE merge kernel. At 8-32 decode rows every one of them is latency
bound: a fixed launch and pipeline cost on a few vregs or a few KB.

`fused_attend_and_merge` runs the same chain in one call per rank, inside a
``shard_map`` over the DCP pair:

1. barrier with the partner, then remote DMAs of this rank's query heads
   and packed candidates into the partner's VMEM (and local copies of its
   own), so both ranks hold the group's ``[2 * Hl, T, D]`` heads and
   ``[2, T, 2 * topk]`` candidates;
2. the candidate merge (`msa_dcp._merge_kernel`: the same selection as
   `msa_dcp.merge_candidates`), this rank's page keys of the selected blocks,
   and the decode work list (`msa_attention_lse._work_list_kernel`) in VMEM,
   copied to SMEM;
3. the sparse attention as `msa_attention_lse._kernel` itself (the served
   kernel's body, called over every tile of the step at once);
4. a remote DMA of the partner's heads' partial and LSE into its VMEM, and
   the LSE merge (plus, 4-bit, the inverse rotation) of this rank's heads
   (`msa_dcp._merge_partials_kernel`'s arithmetic, in the same order).

Every arithmetic step is the unfused path's, so the output is bitwise equal
(selection and work list are integer; the attention body is shared; the
merge repeats `merge_partials`' f32 ops element by element).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import tight_vmem
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_attention,
    msa_attention_lse,
    msa_dcp,
    msa_fp4,
)

FUSED_ENV = "TPU_MINIMAX_M3_DCP_FUSED_DECODE_MAX_TOKENS"
# The fused call's barrier semaphore (the DCP pair's exchange kernels and the
# TP all-reduce use 0 too: every use waits for data its partner sends after
# its own barrier, so the counts cannot run ahead).
_COLLECTIVE_ID = 0
_BLOCK_Q = msa_attention.SMALL_BATCH_BLOCK_Q
_PAGES_PER_STEP = msa_attention.DEFAULT_PAGES_PER_STEP
# The largest step the fused call takes (VMEM and the unrolled key stores).
MAX_TOKENS = 128
# Smallest scoped-VMEM limit candidate (`tight_vmem`): larger limits cost
# ~2.5 us per call on v7x.
_VMEM_LIMIT_BYTES = 32 * 1024 * 1024
# CPU interpret mode (tests); the TPU-semantic interpreter lacks ref bitcasts.
_INTERPRET: object = True


def eligible(num_tokens: int, max_tokens: int, *, dcp: int, kv_heads: int) -> bool:
    """Whether a step of `num_tokens` rows takes the fused call (one KV head
    per DCP pair, whole 8-row tiles, at most `max_tokens` rows)."""
    return (
        dcp == 2
        and kv_heads == 1
        and 0 < num_tokens <= min(max_tokens, MAX_TOKENS)
        and num_tokens % _BLOCK_Q == 0
    )


def _layout(num_tokens: int, topk: int):
    """Tiles, groups per tile, work-list offsets and length of a step."""
    tiles = num_tokens // _BLOCK_Q
    groups_per_tile = msa_attention._groups_per_tile(_BLOCK_Q, topk, _PAGES_PER_STEP)
    offsets, length = msa_attention_lse._work_offsets(
        tiles, _BLOCK_Q, topk, _PAGES_PER_STEP, tiles
    )
    return tiles, groups_per_tile, offsets, length


def _remote(src, dst, send_sem, recv_sem, axis, device, interpret):
    return pltpu.make_async_remote_copy(
        src,
        dst,
        send_sem,
        recv_sem,
        device_id=_device(axis, device, interpret),
        device_id_type=pl.DeviceIdType.MESH,
    )


def _wait(ref, sem):
    """Wait for a copy of `ref`'s size on `sem` (a local descriptor: one form
    for TPU and the CPU interpreter, whose discharge of a remote one's wait
    does not trace)."""
    pltpu.make_async_copy(ref, ref, sem).wait()


def _device(axis, device, interpret):
    # The CPU interpreter's discharge of a remote DMA takes a bare index on a
    # 1-D mesh (its dict form gathers a tuple); TPU takes the named axis.
    return device if interpret else {axis: device}


def _kernel(*refs, fp4: bool, **kwargs):
    """`_body` with the 4-bit-only refs (scale pages, hadamard, scale page
    buffers) absent for 8/16-bit caches."""
    if fp4:
        return _body(*refs, **kwargs)
    prefetch, inputs, out, scratch = refs[:4], refs[4:10], refs[10], refs[11:]
    return _body(
        *prefetch,
        *inputs,
        None,
        None,
        out,
        *scratch[:16],
        None,
        None,
        *scratch[16:],
        **kwargs,
    )


def _body(
    # scalar prefetch
    rank_ref,  # [1] this rank
    page_table_ref,  # page key -> physical page
    head_ref,  # [2] zeros: cache-row KV head 0, query-head block 0
    tile_offset_ref,  # [1] zero (read by no index map here)
    # in
    q_ref,  # [Hl, T, D] this rank's query heads (rotated for a 4-bit cache)
    cand_ref,  # [T, 2 * topk] f32 this rank's packed candidates
    local_page_ref,  # [T, 1] page key of each token's own block on this rank, -1 none
    local_limit_ref,  # [T, 1] visible slots of that page
    base_ref,  # [T, 1] seq_id * pages_per_seq, -1 for padding tokens
    kv_hbm_ref,  # the KV cache, HBM
    scale_hbm_ref,  # 4-bit: its scale pages, HBM (else None)
    hadamard_ref,  # 4-bit: [D, D] bf16 (else None)
    # out
    o_ref,  # [T, Hl * D]
    # scratch
    q_all,  # [2 Hl, T, D]
    cand_all,  # [2, T, 2 * topk] f32
    sel_ref,  # [T, topk] i32 merged selection
    keys_ref,  # [T, topk] i32 this rank's page keys
    rows_ref,  # [tiles, Bq * topk] i32 the keys per tile
    work_v,  # [1, L] i32
    work_s,  # SMEM [1, L] i32
    part_o,  # [2 Hl, T, D] f32 partial of every group head (unrounded)
    part_lse,  # [1, tiles, 1, 2 Hl * Bq] f32
    recv_o,  # [Hl, T, D] f32 the partner's partial of this rank's heads
    recv_lse,  # [1, tiles, 1, 2 Hl * Bq] f32 the partner's lse (all heads)
    m_ref,
    l_ref,
    acc_ref,
    buf_ref,
    sem_ref,
    scale_buf_ref,
    scale_sem_ref,
    send_sem,  # DMA [4]
    recv_sem,  # DMA [4]
    local_sem,  # DMA [1]
    *,
    axis: str,
    sm_scale: float,
    topk: int,
    interpret: bool,
    scale_row0: int = 0,
):
    num_heads, num_tokens, head_dim = q_ref.shape
    tiles, groups_per_tile, offsets, _ = _layout(num_tokens, topk)
    max_groups = tiles * groups_per_tile
    union_len = max_groups * _PAGES_PER_STEP
    me = rank_ref[0]
    other = 1 - me

    # 1. Both ranks are in the call (their scratch is live); exchange the
    # query heads and the candidates.
    if not interpret:  # the CPU interpreter runs the ranks in lockstep
        barrier = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(
            barrier,
            inc=1,
            device_id={axis: other},
            device_id_type=pl.DeviceIdType.MESH,
        )
        pl.semaphore_wait(barrier, 1)
    mine_heads = pl.ds(pl.multiple_of(me * num_heads, num_heads), num_heads)
    theirs = pl.ds(pl.multiple_of(other * num_heads, num_heads), num_heads)
    _remote(
        q_ref,
        q_all.at[mine_heads],
        send_sem.at[0],
        recv_sem.at[0],
        axis,
        other,
        interpret,
    ).start()
    _remote(
        cand_ref,
        cand_all.at[me],
        send_sem.at[1],
        recv_sem.at[1],
        axis,
        other,
        interpret,
    ).start()
    q_all[mine_heads] = q_ref[...]
    cand_all[me] = cand_ref[...]
    _wait(q_all.at[theirs], recv_sem.at[0])
    _wait(cand_all.at[other], recv_sem.at[1])

    # 2. The global selection, this rank's page keys of it (as
    # `msa_dcp._attend_selected`), and the work list.
    msa_dcp._merge_kernel(cand_all, sel_ref, topk=topk)
    sel = sel_ref[...]
    base = base_ref[...]
    mine = (sel >= 0) & ((sel & 1) == me) & (base >= 0)
    keys = jnp.where(mine, base + (sel >> 1), -1)
    keys_ref[...] = keys
    for t in range(num_tokens):
        tile, slot = divmod(t, _BLOCK_Q)
        rows_ref[tile : tile + 1, slot * topk : (slot + 1) * topk] = keys_ref[
            t : t + 1, :
        ]
    msa_attention_lse._work_list_kernel(
        rows_ref,
        work_v,
        block_q=_BLOCK_Q,
        pages_per_step=_PAGES_PER_STEP,
        tiles_per_step=tiles,
        offsets=offsets,
    )
    to_smem = pltpu.make_async_copy(work_v, work_s, local_sem.at[0])
    to_smem.start()
    to_smem.wait()

    # 3. The served attention body over all the step's tiles in one grid step.
    work = work_s.at[0]
    fp4 = scale_hbm_ref is not None
    fp4_in = (scale_hbm_ref,) if fp4 else ()
    fp4_scratch = (scale_buf_ref, scale_sem_ref) if fp4 else ()
    fp4_rows = {"fp4_rows": (scale_row0, scale_row0 + 1)} if fp4 else {}
    msa_attention_lse._kernel(
        work.at[pl.ds(offsets[0], union_len)],
        work.at[pl.ds(offsets[1], max_groups)],
        work.at[pl.ds(offsets[2], max_groups)],
        page_table_ref,
        head_ref,
        tile_offset_ref,
        q_all,
        keys_ref,
        local_page_ref,
        local_limit_ref,
        kv_hbm_ref,
        *fp4_in,
        part_o,
        part_lse,
        m_ref,
        l_ref,
        acc_ref,
        buf_ref,
        sem_ref,
        *fp4_scratch,
        sm_scale=sm_scale,
        **fp4_rows,
        block_start_ref=work.at[pl.ds(offsets[3], 2)],
        tiles_per_step=tiles,
        groups_per_tile=groups_per_tile,
        transposed=False,
        pages_per_step=_PAGES_PER_STEP,
        precision=msa_attention_lse.max_free_levers(),
    )

    # 4. The partner's heads' partial and every LSE go to the partner.
    _remote(
        part_o.at[theirs],
        recv_o,
        send_sem.at[2],
        recv_sem.at[2],
        axis,
        other,
        interpret,
    ).start()
    _remote(
        part_lse, recv_lse, send_sem.at[3], recv_sem.at[3], axis, other, interpret
    ).start()
    _wait(recv_o, recv_sem.at[2])
    _wait(recv_lse, recv_sem.at[3])

    # The LSE merge (`msa_dcp._merge_partials_kernel`), ranks in order. The
    # lse rows are lane-dense per tile (lane h * Bq + t over the group's
    # heads); this rank's heads are lanes [me * Hl * Bq, (me + 1) * Hl * Bq).
    lanes = num_heads * _BLOCK_Q
    own_lse = part_lse[0, :, 0, :]  # [tiles, 2 Hl Bq]
    peer_lse = recv_lse[0, :, 0, :]
    first = me == 0
    lse0 = jnp.where(first, own_lse, peer_lse)
    lse1 = jnp.where(first, peer_lse, own_lse)
    peak = jnp.maximum(lse0, lse1)
    peak = jnp.where(jnp.isinf(peak), 0.0, peak)
    total = peak + jnp.log(jnp.exp(lse0 - peak) + jnp.exp(lse1 - peak))
    weights = []
    for lse in (lse0, lse1):
        w = jnp.exp(lse - total)
        w = jnp.where(w > 0, w, 0.0)
        w = jnp.where(first, w[:, :lanes], w[:, lanes:])  # [tiles, Hl Bq]
        rows = max(8, -(-tiles // 8) * 8)
        if rows > tiles:
            w = jnp.concatenate([w, jnp.zeros((rows - tiles, lanes), w.dtype)], axis=0)
        weights.append(w.T)  # [Hl Bq, rows]: row h * Bq + t, column tile
    own_part = part_o.at[mine_heads]
    for h in range(num_heads):
        cols = []
        for w_t in weights:
            col = jnp.concatenate(
                [
                    w_t[h * _BLOCK_Q : (h + 1) * _BLOCK_Q, j : j + 1]
                    for j in range(tiles)
                ],
                axis=0,
            )  # [T, 1]
            cols.append(jnp.broadcast_to(col, (num_tokens, head_dim)))
        own = own_part[h].astype(o_ref.dtype).astype(jnp.float32)
        peer = recv_o[h].astype(o_ref.dtype).astype(jnp.float32)
        p0 = jnp.where(first, own, peer)
        p1 = jnp.where(first, peer, own)
        merged = (p0 * cols[0] + p1 * cols[1]).astype(o_ref.dtype)
        if hadamard_ref is not None:  # back from the rotated basis
            merged = jnp.dot(
                merged, hadamard_ref[...], preferred_element_type=jnp.float32
            )
            merged = (merged * (1.0 / head_dim)).astype(o_ref.dtype)
        o_ref[:, h * head_dim : (h + 1) * head_dim] = merged
    _wait(q_ref, send_sem.at[0])
    _wait(cand_ref, send_sem.at[1])
    _wait(recv_o, send_sem.at[2])
    _wait(part_lse, send_sem.at[3])


def fused_attend_and_merge(
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
    """`msa_dcp.attend_and_merge` (selection "candidates", DCP 2, one KV
    head) in one Pallas call: the same arguments inside a ``shard_map`` over
    `axis`, this rank's head-major ``q [Hl, T, D]`` (rotated, with the
    `scale_cache` and `hadamard` of a 4-bit cache) and packed ``candidates
    [T, 2 * topk]`` -> its heads' merged (unrotated) output ``[T, Hl * D]``,
    bitwise equal to the unfused program's."""
    if dcp != 2:
        raise ValueError(f"the fused DCP attention pairs two ranks, got dcp={dcp}")
    num_heads, num_tokens, head_dim = q.shape
    if num_tokens % _BLOCK_Q or num_tokens > MAX_TOKENS:
        raise ValueError(
            f"{num_tokens} rows: need a multiple of {_BLOCK_Q} up to {MAX_TOKENS}"
        )
    scale_cache, scale_row0 = msa_fp4.split_scales(scale_cache)
    msa_attention.check_cache(kv_cache, scale_cache)
    msa_attention_lse._check_rows(kv_cache, head_dim, 0)
    rank = msa_dcp.rank_array(rank)
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    pages_per_seq = block_tables.reshape(max_num_seqs, -1).shape[1]
    seq_id, pos, valid = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    own_block = pos // block_size
    local_page = jnp.where(
        valid & (own_block % dcp == rank[0]),
        seq_id * pages_per_seq + own_block // dcp,
        -1,
    )
    local_limit = pos % block_size + 1
    base = jnp.where(valid, seq_id * pages_per_seq, -1)
    tiles, _, _, length = _layout(num_tokens, topk)
    _, rows_per_page, packing, _ = kv_cache.shape
    row_width = 2 * head_dim
    group_heads = 2 * num_heads
    rows = group_heads * _BLOCK_Q
    ring = 1 + (msa_attention_lse._FP4_PREFETCH if scale_cache is not None else 1)
    fp4 = scale_cache is not None
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    fp4_scratch = (
        [
            pltpu.VMEM((ring, _PAGES_PER_STEP, 1) + scale_cache.shape[1:], jnp.float32),
            pltpu.SemaphoreType.DMA((ring, 1)),
        ]
        if fp4
        else []
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(1,),
        in_specs=[vmem] * 5 + ([hbm, hbm, vmem] if fp4 else [hbm]),
        out_specs=vmem,
        scratch_shapes=[
            pltpu.VMEM((group_heads, num_tokens, head_dim), q.dtype),
            pltpu.VMEM((2, num_tokens, 2 * topk), jnp.float32),
            pltpu.VMEM((num_tokens, topk), jnp.int32),
            pltpu.VMEM((num_tokens, topk), jnp.int32),
            pltpu.VMEM((tiles, _BLOCK_Q * topk), jnp.int32),
            pltpu.VMEM((1, length), jnp.int32),
            pltpu.SMEM((1, length), jnp.int32),
            # The partials in f32 (8-row tiles store at 8-row offsets; a
            # bf16 buffer's 16-row tiles do not lower), rounded to q's dtype
            # where the unfused path stores them.
            pltpu.VMEM((group_heads, num_tokens, head_dim), jnp.float32),
            pltpu.VMEM((1, tiles, 1, rows), jnp.float32),
            pltpu.VMEM((num_heads, num_tokens, head_dim), jnp.float32),
            pltpu.VMEM((1, tiles, 1, rows), jnp.float32),
            # m, l, acc and the page ring of `msa_attention_lse._attention_call`
            pltpu.VMEM((rows, 1), jnp.float32),
            pltpu.VMEM((rows, 1), jnp.float32),
            pltpu.VMEM((rows, head_dim), jnp.float32),
            pltpu.VMEM(
                (ring, _PAGES_PER_STEP, 1, rows_per_page, packing, row_width),
                kv_cache.dtype,
            ),
            pltpu.SemaphoreType.DMA((ring, 1)),
            *fp4_scratch,
            pltpu.SemaphoreType.DMA((4,)),
            pltpu.SemaphoreType.DMA((4,)),
            pltpu.SemaphoreType.DMA((1,)),
        ],
    )
    return tight_vmem.pallas_call(
        functools.partial(
            _kernel,
            fp4=fp4,
            axis=axis,
            sm_scale=sm_scale,
            topk=topk,
            interpret=interpret,
            **({"scale_row0": scale_row0} if scale_row0 else {}),
        ),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_tokens, num_heads * head_dim), q.dtype),
        compiler_params=pltpu.CompilerParams(
            collective_id=_COLLECTIVE_ID,
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        interpret=_INTERPRET if interpret else False,
        name="minimax_m3_msa_dcp_fused_decode",
    )(
        rank,
        block_tables.reshape(-1),
        jnp.zeros((2,), jnp.int32),
        jnp.zeros((1,), jnp.int32),
        q,
        candidates.astype(jnp.float32),
        local_page[:, None],
        local_limit[:, None],
        base[:, None],
        kv_cache,
        *((scale_cache, hadamard.astype(q.dtype)) if fp4 else ()),
    )

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Causal attention of one core's query rows over the step's gathered keys and
values, as one kernel per core.

Every core holds every new token's key and value (FP8, flat token order, after
an all-gather), so each query chunk walks only the key chunks it sees, read
from the core's own HBM: no key is sent twice. A query row sees the keys of
its own request (flat positions from the request's start) up to its own.

Scores are base-2 (queries scaled by ``sm_scale * log2(e)``). A key block that
every row of the query chunk sees whole is exponentiated against the running
maximum of the earlier blocks, so the state needs no rescale; a block whose
scores pass that maximum by more than ``STALE_LIMIT``, and every block that
needs masks, takes the exact online-softmax update.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.qwen38.reference import BF16, F32

CHUNK = 256  # query rows per chunk (the stage's flat chunk)
BLOCK = 512  # keys per block (two chunks)
STATE_LANES = 128  # the running maximum and sum, lane-replicated
STALE_LIMIT = 64.0  # in log2 units: exponents up to 2^64 stay exact in F32
INITIAL_MAX = -0.7 * float(jnp.finfo(jnp.float32).max)
ROPE_LANES = 128  # the rotary dims sit within each head's first lanes


def _lanes(state, shape):
    """A lane-replicated state ``[..., STATE_LANES]`` tiled to ``shape``."""
    return jnp.tile(state, (1,) * (state.ndim - 1) + (shape[-1] // STATE_LANES,))


def chunk_bounds(q_flat, q_start, tokens):
    """Per query chunk: the first and last key block it sees and the range of
    blocks every row sees whole (``[whole_lo, whole_hi]``, empty if
    ``whole_lo > whole_hi``), from each row's flat position ``q_flat`` and its
    request's first flat position ``q_start`` (``[rows]``; a padding row has
    ``q_start > q_flat``) among ``tokens`` keys."""
    flat = q_flat.reshape(-1, CHUNK)
    start = q_start.reshape(-1, CHUNK)
    live = (start <= flat) & (start >= 0) & (flat < tokens)
    big = jnp.int32(2**30)
    first = jnp.min(jnp.where(live, start, big), axis=1)
    last = jnp.max(jnp.where(live, flat, -1), axis=1)
    lo, hi = first // CHUNK, last // CHUNK
    whole_lo = -(-jnp.max(jnp.where(live, start, 0), axis=1) // CHUNK)
    # Every row live, and the block's last key before the chunk's first row.
    whole_hi = jnp.where(
        jnp.all(live, axis=1), (jnp.min(flat, axis=1) + 1) // CHUNK - 1, -1
    )
    return lo, hi, whole_lo, whole_hi


def rope_tables(positions, rotary_dim, theta):
    """``reference.rope``'s cos and sin (BF16) of each row's position, as
    the first ``rotary_dim`` of :data:`ROPE_LANES` lanes (cos 1 and sin 0
    past them)."""
    inv = theta ** (-2 * jnp.arange(rotary_dim // 2, dtype=F32) / rotary_dim)
    angles = positions[:, None].astype(F32) * inv
    pad = ((0, 0), (0, ROPE_LANES - rotary_dim))
    cos = jnp.concatenate((jnp.cos(angles), jnp.cos(angles)), -1).astype(BF16)
    sin = jnp.concatenate((jnp.sin(angles), jnp.sin(angles)), -1).astype(BF16)
    return jnp.pad(cos, pad, constant_values=1), jnp.pad(sin, pad)


def _norm_and_rotate(query, norm_ref, cos, sin, rotary_dim, eps):
    """``reference.rope(reference.rms(q))`` of F32 queries ``[rows, heads,
    D]`` with the rows' :func:`rope_tables` ``cos`` / ``sin`` ``[rows,
    ROPE_LANES]`` (each step's BF16 rounding); the rotation's halves meet
    by lane rotations within the first :data:`ROPE_LANES` lanes."""
    query = query * lax.rsqrt(jnp.mean(query * query, axis=2, keepdims=True) + eps)
    query = (query * (1 + norm_ref[...])).astype(BF16).astype(F32)
    rotating = query[..., :ROPE_LANES]
    half = rotary_dim // 2
    lane = lax.broadcasted_iota(jnp.int32, rotating.shape, 2)
    rotation = jnp.where(
        lane < half,
        -pltpu.roll(rotating, ROPE_LANES - half, 2),
        pltpu.roll(rotating, half, 2),
    )
    rotated = rotating * cos.astype(F32)[:, None] + rotation * sin.astype(F32)[:, None]
    return jnp.concatenate(
        [rotated.astype(BF16).astype(F32), query[..., ROPE_LANES:]], axis=2
    )


def local_attention(
    q,
    k,
    v,
    q_flat,
    q_start,
    *,
    sm_scale,
    k_scale,
    v_scale,
    cores=1,
    interleave=CHUNK,
    q_norm=None,
    rope=None,
    rotary_dim=0,
    eps=0.0,
    interpret=False,
):
    """``[rows, Hq, D]`` BF16: causal attention of ``q`` ``[rows, Hq, D]``
    (BF16, ``rows`` a multiple of :data:`CHUNK`) over ``k`` / ``v``
    ``[tokens, Hkv * D]`` (FP8 with static scales ``k_scale`` / ``v_scale``,
    ``tokens`` a multiple of :data:`CHUNK`), ``q_flat`` / ``q_start`` as
    :func:`chunk_bounds` takes them. The keys are ``cores`` cores' blocks of
    ``tokens / cores`` rows (core-major, as gathered), dealt in runs of
    ``interleave`` keys: flat run ``j`` is run ``j // cores`` of core
    ``j % cores``. Rows that see no key are zero.
    ``interpret`` runs the kernel on the host.

    With ``q_norm`` (``[D]``) and ``rope`` (:func:`rope_tables`' ``(cos,
    sin)``), ``q`` is the query projection's output (``[rows, Hq, W]``, each
    head's query in its first ``D`` lanes) and each head's query is
    normalized (``reference.rms``, ``eps``) and rotated (``reference.rope``
    over the first ``rotary_dim`` dims) as it loads."""
    rows, q_heads, _ = q.shape
    dim = q.shape[2] if q_norm is None else q_norm.shape[0]
    tokens, width = k.shape
    kv_heads = width // dim
    group = q_heads // kv_heads
    chunks = rows // CHUNK
    per_core = tokens // cores
    assert BLOCK % CHUNK == 0 and tokens % BLOCK == 0
    assert BLOCK % interleave == 0 and per_core % interleave == 0
    lo, hi, whole_lo, whole_hi = chunk_bounds(q_flat, q_start, tokens)
    q_scale = sm_scale * math.log2(math.e)
    fused = q_norm is not None
    fused_inputs = ()
    if q_norm is not None:
        assert rope is not None
        fused_inputs = (q_norm.reshape(1, dim).astype(F32), *rope)

    def kernel(
        lo_ref,
        hi_ref,
        whole_lo_ref,
        whole_hi_ref,
        q_hbm,
        k_hbm,
        v_hbm,
        flat_ref,
        start_ref,
        *refs,
    ):
        norm_ref, cos_ref, sin_ref, refs = (
            (*refs[:3], refs[3:]) if fused else (None, None, None, refs)
        )
        o_hbm, q_vmem, k_vmem, v_vmem, m_ref, l_ref, acc_ref, q_sem, kv_sems = refs
        c = pl.program_id(0)
        rows_c = pl.ds(pl.multiple_of(c * CHUNK, CHUNK), CHUNK)
        per = BLOCK // CHUNK
        first, last = lo_ref[c] // per, hi_ref[c] // per
        flat = flat_ref[pl.ds(c, 1), :].reshape(CHUNK, 1, 1)
        start = start_ref[pl.ds(c, 1), :].reshape(CHUNK, 1, 1)

        def kv_copies(h, j, slot):
            """Head ``h``'s keys and values of block ``j``: its runs, each
            from the core holding it."""
            lanes = pl.ds(pl.multiple_of(h * dim, dim), dim)
            runs = BLOCK // interleave
            copies = []
            for part in range(runs):
                run = j * runs + part
                offset = lax.rem(run, cores) * per_core + run // cores * interleave
                keys = pl.ds(pl.multiple_of(offset, interleave), interleave)
                for i, (src, dst) in enumerate(((k_hbm, k_vmem), (v_hbm, v_vmem))):
                    copies.append(
                        pltpu.make_async_copy(
                            src.at[keys, lanes],
                            dst.at[slot, pl.ds(part * interleave, interleave)],
                            kv_sems.at[i, slot],
                        )
                    )
            return copies

        def head(h, carry):
            heads = pl.ds(pl.multiple_of(h * group, group), group)
            source = q_hbm.at[rows_c, heads]
            if fused:
                source = q_hbm.at[rows_c, heads, pl.ds(0, dim)]
            load_q = pltpu.make_async_copy(source, q_vmem, q_sem)
            load_q.start()
            m_ref[...] = jnp.full(m_ref.shape, INITIAL_MAX, F32)
            l_ref[...] = jnp.zeros(l_ref.shape, F32)
            acc_ref[...] = jnp.zeros(acc_ref.shape, F32)

            @pl.when(first <= last)
            def _():
                for copy in kv_copies(h, first, 0):
                    copy.start()

            load_q.wait()
            query = q_vmem[...].astype(F32)
            if fused:
                assert cos_ref is not None and sin_ref is not None
                query = _norm_and_rotate(
                    query, norm_ref, cos_ref[rows_c], sin_ref[rows_c], rotary_dim, eps
                )
            q_vmem[...] = (query * q_scale).astype(BF16)

            def block(j, carry):
                slot = (j - first) % 2
                for copy in kv_copies(h, j, slot):
                    copy.wait()

                @pl.when(j + 1 <= last)
                def _():
                    for copy in kv_copies(h, j + 1, 1 - slot):
                        copy.start()

                # Whole iff each of its chunks is.
                whole = jnp.logical_and(
                    whole_lo_ref[c] <= per * j, per * j + per - 1 <= whole_hi_ref[c]
                )
                key = (k_vmem[slot].astype(F32) * k_scale).astype(BF16)
                value = (v_vmem[slot].astype(F32) * v_scale).astype(BF16)

                def exact(masked):
                    s = lax.dot_general(
                        q_vmem[...],
                        key,
                        (((2,), (1,)), ((), ())),
                        preferred_element_type=F32,
                    )
                    if masked:
                        key_flat = j * BLOCK + lax.broadcasted_iota(
                            jnp.int32, (1, 1, BLOCK), 2
                        )
                        seen = jnp.logical_and(key_flat >= start, key_flat <= flat)
                        s = jnp.where(seen, s, -jnp.inf)
                    m_old = m_ref[...]
                    m_new = jnp.maximum(m_old, jnp.max(s, axis=2, keepdims=True))
                    p = jnp.exp2(s - _lanes(m_new, s.shape))
                    alpha = jnp.exp2(m_old - m_new)
                    l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=2, keepdims=True)
                    pv = lax.dot_general(
                        p.astype(BF16),
                        value,
                        (((2,), (0,)), ((), ())),
                        preferred_element_type=F32,
                    )
                    acc_ref[...] = _lanes(alpha, pv.shape) * acc_ref[...] + pv
                    m_ref[...] = m_new

                def stale():
                    s = lax.dot_general(
                        q_vmem[...],
                        key,
                        (((2,), (1,)), ((), ())),
                        preferred_element_type=F32,
                    )
                    shifted = s - _lanes(m_ref[...], s.shape)
                    p = jnp.exp2(shifted)
                    row_sum = jnp.sum(p, axis=2, keepdims=True)
                    p = p.astype(BF16)
                    over = jnp.max(shifted) > STALE_LIMIT

                    @pl.when(jnp.logical_not(over))
                    def _():
                        l_ref[...] = l_ref[...] + row_sum
                        acc_ref[...] = acc_ref[...] + lax.dot_general(
                            p,
                            value,
                            (((2,), (0,)), ((), ())),
                            preferred_element_type=F32,
                        )

                    pl.when(over)(functools.partial(exact, False))

                # The first block sets the running maximum the later ones use.
                pl.when(whole & (j > first))(stale)
                pl.when(whole & (j == first))(functools.partial(exact, False))
                pl.when(jnp.logical_not(whole))(functools.partial(exact, True))
                return carry

            lax.fori_loop(first, last + 1, block, 0)
            l_h = _lanes(l_ref[...], acc_ref.shape)
            q_vmem[...] = jnp.where(l_h > 0, acc_ref[...] / l_h, 0.0).astype(BF16)
            store = pltpu.make_async_copy(q_vmem, o_hbm.at[rows_c, heads], q_sem)
            store.start()
            store.wait()
            return carry

        lax.fori_loop(0, kv_heads, head, 0)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    rows_spec = pl.BlockSpec(memory_space=pltpu.VMEM)  # every chunk's, [chunks, CHUNK]
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(chunks,),
            in_specs=[hbm, hbm, hbm, rows_spec, rows_spec]
            + [pl.BlockSpec(memory_space=pltpu.VMEM)] * (3 if fused else 0),
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((CHUNK, group, dim), BF16),
                pltpu.VMEM((2, BLOCK, dim), k.dtype),
                pltpu.VMEM((2, BLOCK, dim), v.dtype),
                pltpu.VMEM((CHUNK, group, STATE_LANES), F32),
                pltpu.VMEM((CHUNK, group, STATE_LANES), F32),
                pltpu.VMEM((CHUNK, group, dim), F32),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2, 2)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((rows, q_heads, dim), BF16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes
            if not interpret
            else None,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="qwen38_local_attention",
    )(
        lo,
        hi,
        whole_lo,
        whole_hi,
        q,
        k,
        v,
        q_flat.reshape(chunks, CHUNK),
        q_start.reshape(chunks, CHUNK),
        *fused_inputs,
    )


def reference_attention(q, k, v, q_flat, q_start, *, sm_scale, k_scale, v_scale):
    """:func:`local_attention`'s math in FP32, unfused."""
    dim = q.shape[2]
    kf = k.astype(F32).reshape(k.shape[0], -1, dim) * k_scale
    vf = v.astype(F32).reshape(v.shape[0], -1, dim) * v_scale
    group = q.shape[1] // kf.shape[1]
    kf = jnp.repeat(kf, group, axis=1)
    vf = jnp.repeat(vf, group, axis=1)
    s = jnp.einsum("qhd,khd->hqk", q.astype(F32), kf) * sm_scale
    key = jnp.arange(k.shape[0])[None, None, :]
    seen = (key >= q_start[None, :, None]) & (key <= q_flat[None, :, None])
    s = jnp.where(seen, s, -jnp.inf)
    w = jnp.exp(s - jnp.max(s, axis=-1, keepdims=True))
    w = jnp.where(seen, w, 0.0)
    total = jnp.sum(w, axis=-1, keepdims=True)
    o = jnp.einsum("hqk,khd->qhd", jnp.where(total > 0, w / total, 0.0), vf)
    return o.astype(BF16)


def attention_plan(
    mesh,
    axis,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    positions,
    *,
    rows,
    interleave,
    kv_layout,
    cache_shape,
    rotary_dim,
    theta,
):
    """What every attention layer of a step without cached prefixes shares,
    from the metadata as the streaming kernel's wrapper takes it
    (``sharded_pcp_ragged_paged_attention``), ``rows`` per core and the
    core's cache shape ``cache_shape``:
    each core's query rows' flat positions and their requests' first ones
    (:func:`chunk_bounds`), the runs of gathered rows its cache pages take
    (``[1, 3, runs]`` descriptors and their count, as the streaming kernel's
    writer takes them), the :func:`rope_tables` of its rows' ``positions``
    (sharded along ``axis``), and whether any request has a cached prefix
    (:func:`has_history`)."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import wrapper
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
        _reshape_metadata_page_indices_jax,
    )

    cores = mesh.shape[axis]
    cache = jax.ShapeDtypeStruct(cache_shape, jnp.float8_e4m3fn)
    page_size = wrapper._kv_cache_page_size(cache, kv_layout)

    def local(kv_lens, page_indices, cu_q_lens, distribution, positions):
        all_rows = wrapper._compute_pcp_rank_major_query_row_mapping_from_metadata(
            kv_lens,
            cu_q_lens,
            distribution,
            local_padded_tokens=rows,
            pcp_size=cores,
            interleave_size=interleave,
        )
        mine = wrapper._select_replicated_query_row_mapping_for_pcp_rank(
            all_rows, axis, cores
        )
        cache_rank = mine.source_ranks[0]
        writeback = wrapper._compute_pcp_kv_writeback_mapping(
            all_rows,
            _reshape_metadata_page_indices_jax(page_indices, kv_lens.shape[0]),
            local_kv_cache_num_blocks=cache_shape[0],
            page_size=page_size,
            pcp_size=cores,
            interleave_size=interleave,
        )
        # The cache slot of each gathered row this core's pages hold, so the
        # writer copies runs straight from the gathered rows.
        owned = all_rows.is_valid & (writeback.cache_owner_ranks == cache_rank)
        slots = jnp.where(owned, writeback.cache_local_slot_ids, -1)
        segments, num_segments = wrapper._build_writeback_segment_descriptors(
            slots, slots.shape[0], kv_layout=kv_layout, page_size=page_size
        )
        start = cu_q_lens[jnp.clip(mine.request_indices, 0, cu_q_lens.shape[0] - 1)]
        q_flat = start + mine.absolute_positions
        q_start = jnp.where(mine.is_valid, start, q_flat + 1)
        cos, sin = rope_tables(positions, rotary_dim, theta)
        return q_flat, q_start, segments[None], num_segments[None], cos, sin

    plan = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(),) * 4 + (P(axis),),
        out_specs=(P(axis),) * 6,
        check_vma=False,
    )(kv_lens, page_indices, cu_q_lens, distribution, positions)
    history = has_history(kv_lens, cu_q_lens, distribution).astype(jnp.int32)
    return (*plan, history[None])


def stage_attention(
    mesh,
    ring,
    axis,
    q,
    k,
    v,
    kv_cache,
    plan,
    *,
    sm_scale,
    kv_layout,
    interleave,
    q_norm,
    rotary_dim,
    eps,
):
    """Attention of a prefill step whose requests have no cached prefix, and
    the KV cache with the new tokens' rows: the global query projection ``q``
    ``[tokens, heads, W]`` (each head's query in its first ``D`` lanes, not
    yet normalized by ``q_norm`` or rotated: :func:`local_attention` does it),
    ``k`` / ``v`` ``[tokens, heads, D]`` (BF16, the stage's rank-major rows
    along ``axis``), ``kv_cache`` as
    ``pcp_streaming_rpa.wrapper.sharded_pcp_ragged_paged_attention`` takes it
    (unit KV scales) and the step's :func:`attention_plan`; the stage deals
    its rows in runs of ``interleave`` tokens. Every core gathers every new
    key and value (``prologue.all_gather`` over ``ring``) and runs
    :func:`local_attention` on its rows; each core writes the rows of the
    cache pages it owns, as the streaming kernel captures them."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel.qwen38.prologue import all_gather

    cores = mesh.shape[axis]

    def local(q, k, v, kv_cache, q_flat, q_start, segments, num_segments, cos, sin):
        rows, kv_heads, dim = k.shape
        width = kv_heads * dim
        fresh = jnp.concatenate([k.reshape(rows, width), v.reshape(rows, width)], 1)
        gathered = all_gather(fresh.astype(jnp.float8_e4m3fn), ring, axis)
        out = local_attention(
            q,
            gathered[:, :width],
            gathered[:, width:],
            q_flat,
            q_start,
            sm_scale=sm_scale,
            k_scale=1.0,
            v_scale=1.0,
            cores=cores,
            interleave=interleave,
            q_norm=q_norm,
            rope=(cos, sin),
            rotary_dim=rotary_dim,
            eps=eps,
        )
        kv_cache = _write_gathered(
            gathered, kv_heads, kv_cache, segments, num_segments, kv_layout
        )
        return out, kv_cache

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(axis),) * 10,
        out_specs=(P(axis), P(axis)),
        check_vma=False,
    )(q, k, v, kv_cache, *plan[:-1])


def write_kv(mesh, ring, axis, k, v, kv_cache, plan, *, kv_layout):
    """The KV cache with a step's new key and value rows only (``k`` / ``v``
    ``[tokens, heads, D]`` BF16, the stage's rank-major rows along ``axis``),
    each core writing the rows of the pages it owns as :func:`stage_attention`
    does (unit KV scales; ``plan``: the step's :func:`attention_plan`)."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel.qwen38.prologue import all_gather

    def local(k, v, kv_cache, segments, num_segments):
        rows, kv_heads, dim = k.shape
        width = kv_heads * dim
        fresh = jnp.concatenate([k.reshape(rows, width), v.reshape(rows, width)], 1)
        gathered = all_gather(fresh.astype(jnp.float8_e4m3fn), ring, axis)
        return _write_gathered(
            gathered, kv_heads, kv_cache, segments, num_segments, kv_layout
        )

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(axis),) * 5,
        out_specs=P(axis),
        check_vma=False,
    )(k, v, kv_cache, plan[2], plan[3])


def _write_gathered(gathered, kv_heads, kv_cache, segments, num_segments, kv_layout):
    """``kv_cache`` with this core's pages' rows of the step's gathered FP8
    ``[tokens, 2 * heads * D]`` keys and values."""
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import kernel as pcp
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import wrapper

    tokens = gathered.shape[0]
    dim = gathered.shape[1] // (2 * kv_heads)
    packed = gathered.astype(BF16).reshape(tokens, 2, kv_heads, dim)
    packed = wrapper._prepare_packed_kv_for_cache(
        packed[:, 0], packed[:, 1], kv_cache, kv_layout=kv_layout
    )
    return pcp.write_captured_kv_to_local_cache(
        wrapper._reshape_packed_kv_cache_for_attention(kv_cache, kv_layout),
        packed,
        segments[0],
        num_segments[0],
        kv_layout=kv_layout,
    )


def has_history(kv_lens, cu_q_lens, distribution):
    """Whether any of the step's requests attends KV cached before the step."""
    requests = kv_lens.shape[0]
    active = jnp.arange(requests) < distribution[-1]
    return jnp.any(active & (kv_lens > jnp.diff(cu_q_lens[: requests + 1])))

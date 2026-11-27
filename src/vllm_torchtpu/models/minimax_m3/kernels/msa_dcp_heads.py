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
"""The attention half of the MSA DCP forward for a DCP group that spans
several KV heads (`msa_dcp`, ``G > 1``), as one SPMD program over the DCP
mesh axis: the per-rank body of a ``shard_map``.

The DCP attention gathers every query head of the group to every rank and
reduce-scatters the ranks' partial outputs back: ``2 * T * G * Hg * D``
bf16 per rank and layer, whatever the context. A prefill chunk's rows are
many and its context comparatively short (c64, DCP 8: 8K rows over a
149K-token context move 235 MB of queries and partials per rank, the
context's pages of one KV head 19 MB). So a long step attends prefill
requests head-parallel: each rank brings its own KV head's pages of those
requests from the other ranks (an all-to-all of whole pages) and attends
its own query heads over them, which needs no query gather and no merge.
Decode-sized requests (at most ``decode_rows`` query rows) keep the DCP
attention on their rows only, compacted to a static ``max_num_seqs *
decode_rows`` rows.

Both paths attend the same selection (`msa_dcp.merge_head_candidates` of
the ranks' candidates) over the same 4-bit pages; the head-parallel one
normalizes in one pass instead of merging bf16 partials.
"""

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_attention,
    msa_attention_lse,
    msa_dcp,
    msa_fp4,
)

# Steps of at least this many tokens attend their prefill requests
# head-parallel. Each round moves a static page buffer (~35 MB per rank at
# max_model_len 256K), which a step's query gather and merge exceed from
# about 2K tokens (~0.16 us per token at c64 DCP 8, j-0a66fa2e).
HEAD_PARALLEL_MIN_TOKENS = 2048
# Tiles and pages per kernel step of the head-parallel calls (8 query heads
# each, every rank's pages): wide groups amortize the kernel's per-step cost
# (`msa_attention.PREFILL_BLOCK_Q_4BIT`). Standalone 8192-token call, 8 query
# heads over all pages: 3.91 -> 2.49 ms at served-c48-like dispersion, 2.40 ->
# 1.90 ms at lower (16-token tiles of 16 pages before; j-f7057d29).
_BLOCK_Q = 8
_PAGES_PER_STEP = 64

# TPU_MINIMAX_M3_DCP_EARLY_Q_GATHER=1: on steps that take the DCP attention
# for every row (`uses_dcp_attention`), the caller gathers the group's query
# heads unrotated before the scoring op (`attend_and_merge(q_gathered=True)`)
# instead of this program gathering the rotated heads after it, so the
# gather (~7 MB per rank at c64 decode, DCP 8) can overlap the indexer and
# the KV write. The rotation is per row and per head, so rotating after the
# gather gives the same queries.
EARLY_Q_GATHER_ENV = "TPU_MINIMAX_M3_DCP_EARLY_Q_GATHER"


def early_q_gather_enabled() -> bool:
    return os.environ.get(EARLY_Q_GATHER_ENV, "0") == "1"


if early_q_gather_enabled():
    init_logger(__name__).info(
        "%s=1: MSA DCP query heads gathered ahead of the scoring op.",
        EARLY_Q_GATHER_ENV,
    )


HP_PREFILL_CONFIG_ENV = "TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG"
_hp_config_logged = False


def hp_prefill_kernel_kwargs() -> dict:
    """Kernel config of the head-parallel prefill attention from
    TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG ("<block_q>:<pages per step>:
    <widths w/w/...>:<page cache slots>[:<page cache tile pages>]"); empty ->
    `_BLOCK_Q` / `_PAGES_PER_STEP` and the kernel's widths, no page cache.
    The optional fifth field gives the head-parallel calls their own page
    cache tile limit, so the global
    TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_TILE_PAGES can be tuned for the
    other prefill calls (c64 DCP 4: steps below 2048 tokens prefer 64, the
    head-parallel calls 192)."""
    global _hp_config_logged
    from vllm_torchtpu import envs

    spec = envs.TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG
    if not spec:
        return dict(block_q=_BLOCK_Q, pages_per_step=_PAGES_PER_STEP)
    parts = spec.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(
            f"{HP_PREFILL_CONFIG_ENV}={spec!r}: want bq:pps:widths:slots[:tile_pages]"
        )
    block_q, pages_per_step, widths, slots = parts[:4]
    tile_pages = int(parts[4] or 0) if len(parts) == 5 else 0
    kw = dict(
        block_q=int(block_q or _BLOCK_Q),
        pages_per_step=int(pages_per_step or _PAGES_PER_STEP),
    )
    if widths:
        kw["step_widths"] = tuple(sorted(int(w) for w in widths.split("/")))
    if int(slots or 0):
        kw["page_cache_slots"] = int(slots)
        kw["page_cache_bf16"] = msa_attention_lse.PREFILL_PAGE_CACHE_BF16
        if tile_pages:
            kw["page_cache_tile_pages"] = tile_pages
    if not _hp_config_logged:
        _hp_config_logged = True
        init_logger(__name__).warning(
            "%s=%s: head-parallel MSA prefill attention kernel %s "
            "(page cache tile pages %d).",
            HP_PREFILL_CONFIG_ENV,
            spec,
            kw,
            tile_pages or msa_attention_lse.PAGE_CACHE_TILE_PAGES,
        )
    return kw


def uses_dcp_attention(
    num_tokens: int,
    max_num_seqs: int,
    decode_rows: int,
    head_parallel_min_tokens: int = HEAD_PARALLEL_MIN_TOKENS,
) -> bool:
    """Whether a step of `num_tokens` rows attends every row with the DCP
    attention (query gather + merge) rather than head-parallel."""
    return num_tokens < max(head_parallel_min_tokens, 2 * max_num_seqs * decode_rows)


def decode_row_plan(
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    max_num_seqs: int,
    decode_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """``(dcp_req, hp_req, cu_d, src_row)`` of a head-parallel step: the
    decode-sized requests, the prefill ones, the decode requests' compacted
    ``cu_q_lens`` and the step row of each of the static
    ``max_num_seqs * decode_rows`` compacted rows."""
    seq_idx = jnp.arange(max_num_seqs, dtype=jnp.int32)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    live = (seq_idx < distribution[2]) & (q_lens > 0)
    dcp_req = live & (q_lens <= decode_rows)
    hp_req = live & (q_lens > decode_rows)
    cu_d = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(jnp.where(dcp_req, q_lens, 0))]
    ).astype(jnp.int32)
    p = jnp.arange(max_num_seqs * decode_rows, dtype=jnp.int32)
    # compare_all: the default scan lowers to a loop of small ops.
    req = jnp.searchsorted(cu_d[1:], p, side="right", method="compare_all")
    req = jnp.minimum(req, max_num_seqs - 1)
    src_row = jnp.where(p < cu_d[-1], cu_q_lens[req] + p - cu_d[req], 0)
    return dcp_req, hp_req, cu_d, src_row


def merge_partials(
    out: jax.Array,
    lse: jax.Array,
    axis: str,
    *,
    rs_out_precision: bool,
    a2a: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """`ops.msa.merge_head_partials` inside the SPMD program: head-major
    partials ``out [H, T, D]``, ``lse [H, T]`` -> this rank's heads,
    unrotated, ``[T, H / dcp, D]``. `a2a`: merge by `_merge_partials_a2a`
    whatever the row count."""
    if a2a or out.shape[1] <= _merge_a2a_max_tokens():
        return _merge_partials_a2a(out, lse, axis, interpret=interpret)
    lse = lse.astype(jnp.float32).T  # token-major across the gather
    lses = jax.lax.all_gather(lse, axis, axis=0)
    weight = jnp.exp(lse - jax.nn.logsumexp(lses, axis=0)).T[..., None]
    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    weighted = jnp.where(weight > 0, out.astype(jnp.float32) * weight, 0.0)
    if rs_out_precision:
        weighted = weighted.astype(out.dtype)
    merged = jax.lax.psum_scatter(weighted, axis, scatter_dimension=0, tiled=True)
    return msa_fp4.unrotate_output(merged.astype(out.dtype)).transpose(1, 0, 2)


_merge_a2a_logged = False


def _merge_a2a_max_tokens() -> int:
    """`TPU_MINIMAX_M3_DCP_HEADS_MERGE_A2A_MAX_TOKENS` (0 = off)."""
    global _merge_a2a_logged
    from vllm_torchtpu import envs

    limit = envs.TPU_MINIMAX_M3_DCP_HEADS_MERGE_A2A_MAX_TOKENS
    if limit and not _merge_a2a_logged:
        _merge_a2a_logged = True
        init_logger(__name__).warning(
            "TPU_MINIMAX_M3_DCP_HEADS_MERGE_A2A_MAX_TOKENS=%d: MSA DCP partials "
            "of steps up to that many rows merged by all-to-all + f32 kernel.",
            limit,
        )
    return limit


def _merge_partials_a2a(
    out: jax.Array, lse: jax.Array, axis: str, *, interpret: bool = False
) -> jax.Array:
    """`merge_partials` by one all-to-all of the partials' head blocks (and
    one of their LSEs) and `msa_dcp.merge_partials`, which weights and sums
    them in f32 and unrotates in one pass. The reduce-scatter path rounds
    each weighted partial to bf16 and sums in bf16, so the results differ
    in the last bits. The rank attended nothing for a token -> its partial
    is zero (`msa_sparse_attention_lse`), so the plain product is safe."""
    num_heads, num_tokens, head_dim = out.shape
    dcp = jax.lax.axis_size(axis)
    local = num_heads // dcp
    parts = out.reshape(dcp, local, num_tokens, head_dim)
    lses = lse.astype(jnp.float32).reshape(dcp, local, num_tokens).transpose(0, 2, 1)
    parts = jax.lax.all_to_all(parts, axis, 0, 0, tiled=True)
    lses = jax.lax.all_to_all(lses, axis, 0, 0, tiled=True)
    hadamard = jnp.asarray(msa_fp4.hadamard(head_dim), jnp.bfloat16)
    merged = msa_dcp.merge_partials(parts, lses, hadamard, interpret=interpret)
    return merged.reshape(num_tokens, local, head_dim)


def _dcp_attend(
    q,
    kv_cache,
    scale_cache,
    candidates,
    seq_lens,
    block_tables,
    cu,
    *,
    axis,
    rank,
    dcp,
    sm_scale,
    topk,
    rs_out_precision,
    interpret,
    q_gathered=False,
    merge_a2a=False,
):
    """The DCP attention of rows ``q [R, Hl, D]`` (this rank's rotated query
    heads) with this rank's packed ``candidates [R, G * 2 * topk]``.
    `q_gathered`: ``q`` is already the group's heads ``[R, dcp * Hl, D]``,
    unrotated."""
    if q_gathered:
        q_all = msa_fp4.rotate_queries(q)
    else:
        q_all = jax.lax.all_gather(q, axis, axis=1, tiled=True)
    cand_all = jax.lax.all_gather(candidates, axis, axis=0)
    out, lse = msa_dcp.msa_attend_dcp(
        q_all,
        kv_cache,
        cand_all,
        seq_lens,
        block_tables,
        cu,
        sm_scale=sm_scale,
        topk=topk,
        rank=rank,
        dcp=dcp,
        scale_cache=scale_cache,
        interpret=interpret,
    )
    return merge_partials(
        out,
        lse,
        axis,
        rs_out_precision=rs_out_precision,
        a2a=merge_a2a,
        interpret=interpret,
    )


def _per_token(seq_id: jax.Array, table: jax.Array) -> jax.Array:
    """``table[seq_id]`` for a per-request table, by a one-hot over the
    requests: a gather of 8K tokens from a 64-entry table cost ~70 us."""
    onehot = seq_id[:, None] == jnp.arange(table.shape[0], dtype=jnp.int32)
    return jnp.sum(jnp.where(onehot, table[None, :].astype(jnp.int32), 0), axis=1)


# Pages whose head copies to every rank an exchange keeps in flight.
_EXCHANGE_PAGES_IN_FLIGHT = 8
# The exchange's barrier semaphore.
_EXCHANGE_COLLECTIVE_ID = 0
_exchange_logged = False


def live_part(live: jax.Array, replica, replicas: int, part: int) -> jax.Array:
    """Live pages of replica `replica`'s part when the first `live` logical
    blocks of a buffer are live and block ``l`` sits in part ``l % replicas``
    at index ``l // replicas``."""
    return jnp.clip((live - replica + replicas - 1) // replicas, 0, part)


def exchange_head_pages(
    kv_cache: jax.Array,
    pages: jax.Array,
    live: jax.Array,
    rank: jax.Array,
    *,
    axis: str,
    dcp: int,
    replicas: int,
    head_width: int,
    interpret: bool = False,
    in_flight: int = _EXCHANGE_PAGES_IN_FLIGHT,
    direct: bool = False,
    scale_cache: jax.Array | None = None,
    scale_row0: int = 0,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Inside a ``shard_map`` over `axis`: every rank's pages of the KV head
    this rank (``rank``, ``int32[1]``) owns, ``[replicas, dcp, L / replicas,
    R, P, head_width]``: ``[j, src, i]`` is rank ``src``'s page ``pages[j *
    L / replicas + i]``, for the `live_part` of ``live`` (``int32[1]``)
    pages of each part ``j``; the others are left unset.

    Rank ``src`` copies each live page's head-``h`` columns straight from its
    cache into rank ``h * replicas + j``'s output by remote DMA (one copy
    per page and destination, all destinations in flight at once), and
    every rank forwards what it received to the other replicas of its head
    (one chip's cores at c64). A static XLA gather, all-to-all and
    all-gather moved the whole buffer whatever the context, relaid the pages
    out around each collective and ran the all-to-all at one link's
    bandwidth.

    `in_flight`: pages whose copies a rank keeps in flight. `direct`: rank
    ``src`` copies every part straight to every replica of the head (no
    forwarding stage, ``replicas`` times the source's sends).

    `scale_cache` (``[pages, SCALE_ROWS, lanes]`` f32, the K / V rows of
    head ``h`` at ``scale_row0 + 2h`` / ``+ 1``; TPU_MINIMAX_M3_DCP_EXCHANGE_SCALES):
    each page's copy is joined by a copy of those two rows of the
    destination's head into rows 0 / 1 of a second output ``[replicas, dcp,
    L / replicas, SCALE_ROWS, lanes]`` (same slots, same live pages; its
    other rows and slots are left unset). Returns ``(pages, scales)``
    then."""
    _, rows, packing, _ = kv_cache.shape
    with_scales = scale_cache is not None
    num_pages = pages.shape[0]
    part = num_pages // replicas
    # Slots past the live pages stay unset: harmless while no key reaches
    # them and while any byte pattern of the 4-bit cache dequantizes finite.
    if kv_cache.dtype != jnp.uint8:
        raise ValueError(f"4-bit (uint8) caches only, got {kv_cache.dtype}")

    def remote(src, dst, send_sem, recv_sem, device):
        return pltpu.make_async_remote_copy(
            src,
            dst,
            send_sem,
            recv_sem,
            device_id={axis: device},
            device_id_type=pl.DeviceIdType.MESH,
        )

    def kernel(pages_ref, live_ref, rank_ref, kv_ref, *refs):
        if with_scales:
            sc_ref, out_ref, sc_out_ref, send_sem, recv_sem, fwd_sem = refs[:6]
            sc_send_sem, sc_recv_sem, sc_fwd_sem = refs[6:]
        else:
            out_ref, send_sem, recv_sem, fwd_sem = refs
        me = rank_ref[0]
        replica = me % replicas
        counts = [live_part(live_ref[0], j, replicas, part) for j in range(replicas)]
        barrier = pltpu.get_barrier_semaphore()
        for d in range(dcp):
            pl.semaphore_signal(
                barrier,
                inc=1,
                device_id={axis: jnp.int32(d)},
                device_id_type=pl.DeviceIdType.MESH,
            )
        # Every rank has entered the call: its output buffer is live.
        pl.semaphore_wait(barrier, dcp)

        def sends(i, op, j=None):
            for d in range(dcp):
                head, jd = divmod(d, replicas)
                jj = jd if j is None else j

                def send(d=d, head=head, jj=jj):
                    columns = pl.ds(head * head_width, head_width)
                    page = pages_ref[jj * part + i]
                    src = kv_ref.at[page, :, :, columns]
                    op(remote(src, out_ref.at[jj, me, i], send_sem, recv_sem, d))
                    if with_scales:
                        # Equal-sized copies on their own semaphores.
                        kv_rows = pl.ds(scale_row0 + 2 * head, 2)
                        op(
                            remote(
                                sc_ref.at[page, kv_rows],
                                sc_out_ref.at[jj, me, i, pl.ds(0, 2)],
                                sc_send_sem,
                                sc_recv_sem,
                                d,
                            )
                        )

                pl.when(i < counts[jj])(send)

        def pipelined(n, copies):
            """Start ``copies(i, op)`` for ``i < n``, a bounded number in
            flight (equal-sized copies on one semaphore: any wait retires
            one)."""
            ahead = in_flight

            def step(i, carry):
                copies(i, lambda c: c.start())
                pl.when(i >= ahead)(lambda: copies(i - ahead, lambda c: c.wait_send()))
                return carry

            def drain(i, carry):
                copies(i, lambda c: c.wait_send())
                return carry

            jax.lax.fori_loop(0, n, step, 0)
            jax.lax.fori_loop(jnp.maximum(n - ahead, 0), n, drain, 0)

        def page_wait(sem, sc_sem=None):
            def wait(i, carry):
                page = out_ref.at[0, 0, 0]
                pltpu.make_async_copy(page, page, sem).wait()
                if sc_sem is not None:
                    rows = sc_out_ref.at[0, 0, 0, pl.ds(0, 2)]
                    pltpu.make_async_copy(rows, rows, sc_sem).wait()
                return carry

            return wait

        sc_recv = sc_recv_sem if with_scales else None
        if direct:
            # Every part to every replica of its head: no forwarding stage.
            for j in range(replicas):
                pipelined(counts[j], functools.partial(sends, j=j))
            jax.lax.fori_loop(0, sum(counts) * dcp, page_wait(recv_sem, sc_recv), 0)
            return
        pipelined(counts[0], sends)

        mine = jnp.where(replica == 0, counts[0], counts[-1])
        for j in range(1, replicas - 1):
            mine = jnp.where(replica == j, counts[j], mine)
        jax.lax.fori_loop(0, mine * dcp, page_wait(recv_sem, sc_recv), 0)

        # Forward the received pages to the head's other replicas.
        def forwards(i, op):
            for k in range(1, replicas):
                peer = me - replica + (replica + k) % replicas
                for src in range(dcp):
                    page = out_ref.at[replica, src, i]
                    op(remote(page, page, send_sem, fwd_sem, peer))
                    if with_scales:
                        rows = sc_out_ref.at[replica, src, i, pl.ds(0, 2)]
                        op(remote(rows, rows, sc_send_sem, sc_fwd_sem, peer))

        pipelined(mine, forwards)
        theirs = sum(counts) - mine
        sc_fwd = sc_fwd_sem if with_scales else None
        jax.lax.fori_loop(0, theirs * dcp, page_wait(fwd_sem, sc_fwd), 0)

    out_shape = jax.ShapeDtypeStruct(
        (replicas, dcp, part, rows, packing, head_width), kv_cache.dtype
    )
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    if not with_scales:
        return pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                grid=(1,),
                in_specs=[any_spec],
                out_specs=any_spec,
                scratch_shapes=[pltpu.SemaphoreType.DMA(())] * 3,
            ),
            out_shape=out_shape,
            compiler_params=pltpu.CompilerParams(collective_id=_EXCHANGE_COLLECTIVE_ID),
            interpret=pltpu.InterpretParams() if interpret else False,
            name="minimax_m3_msa_dcp_exchange_head_pages",
        )(pages, live, rank, kv_cache)
    if scale_cache.dtype != jnp.float32 or scale_cache.shape[1] < scale_row0 + 2 * (
        dcp // replicas
    ):
        raise ValueError(f"scale pages {scale_cache.shape} miss K/V rows")
    sc_shape = jax.ShapeDtypeStruct(
        (replicas, dcp, part, *scale_cache.shape[1:]), jnp.float32
    )
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=[any_spec, any_spec],
            out_specs=(any_spec, any_spec),
            scratch_shapes=[pltpu.SemaphoreType.DMA(())] * 6,
        ),
        out_shape=(out_shape, sc_shape),
        compiler_params=pltpu.CompilerParams(collective_id=_EXCHANGE_COLLECTIVE_ID),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="minimax_m3_msa_dcp_exchange_head_pages_scales",
    )(pages, live, rank, kv_cache, scale_cache)


EXCHANGE_ASYNC_ENV = "TPU_MINIMAX_M3_DCP_EXCHANGE_ASYNC"
_async_logged = False


def exchange_async() -> bool:
    """TPU_MINIMAX_M3_DCP_EXCHANGE_ASYNC, read at trace time via `envs`."""
    global _async_logged
    from vllm_torchtpu import envs

    on = bool(envs.TPU_MINIMAX_M3_DCP_EXCHANGE_ASYNC)
    if on and not _async_logged:
        _async_logged = True
        init_logger(__name__).warning(
            "%s=1: head-parallel MSA page exchange started ahead of the "
            "decode-row attention and the selection merge, retired after them.",
            EXCHANGE_ASYNC_ENV,
        )
    return on


_exchange_scales_logged = False


def exchange_scales() -> bool:
    """TPU_MINIMAX_M3_DCP_EXCHANGE_SCALES, read at trace time via `envs`
    (the synchronous exchange only)."""
    global _exchange_scales_logged
    from vllm_torchtpu import envs

    on = bool(envs.TPU_MINIMAX_M3_DCP_EXCHANGE_SCALES)
    if on and not _exchange_scales_logged:
        _exchange_scales_logged = True
        init_logger(__name__).warning(
            "TPU_MINIMAX_M3_DCP_EXCHANGE_SCALES=1: the MSA DCP page exchange "
            "copies the pages' K/V scale rows by remote DMA (no scale "
            "gather / all-to-all / all-gather)."
        )
    return on


def exchange_direct(replicas: int) -> bool:
    from vllm_torchtpu import envs

    return bool(envs.TPU_MINIMAX_M3_DCP_EXCHANGE_DIRECT) and replicas > 1


def exchange_head_pages_start(
    kv_cache: jax.Array,
    pages: jax.Array,
    live: jax.Array,
    rank: jax.Array,
    *,
    axis: str,
    dcp: int,
    replicas: int,
    head_width: int,
    direct: bool = True,
    interpret: bool = False,
):
    """`exchange_head_pages` as a start / `exchange_head_pages_wait` pair:
    starts every page copy of this rank (`direct`: every part to every
    replica of its head; else part ``j`` to replica ``j``, forwarded by the
    wait) and returns the future ``(out, send_sem, recv_sem, fwd_sem)`` with the
    copies in flight, so the caller's work in between overlaps them. The
    pages are read from `kv_cache` until the wait: the caller keeps it
    unchanged (an operand of the program, written before the start)."""
    _, rows, packing, _ = kv_cache.shape
    num_pages = pages.shape[0]
    part = num_pages // replicas
    if kv_cache.dtype != jnp.uint8:
        raise ValueError(f"4-bit (uint8) caches only, got {kv_cache.dtype}")

    def kernel(
        pages_ref, live_ref, rank_ref, kv_ref, out_ref, send_sem, recv_sem, fwd_sem
    ):
        del fwd_sem  # the wait's forwarding semaphore, live from here on
        me = rank_ref[0]
        counts = [live_part(live_ref[0], j, replicas, part) for j in range(replicas)]
        barrier = pltpu.get_barrier_semaphore()
        for d in range(dcp):
            pl.semaphore_signal(
                barrier,
                inc=1,
                device_id={axis: jnp.int32(d)},
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(barrier, dcp)

        def start_part(j, only=None):
            def body(i, carry):
                for d in range(dcp):
                    head, jd = divmod(d, replicas)
                    if only is not None and jd != j:
                        continue
                    columns = pl.ds(head * head_width, head_width)
                    src = kv_ref.at[pages_ref[j * part + i], :, :, columns]
                    pltpu.make_async_remote_copy(
                        src,
                        out_ref.at[j, me, i],
                        send_sem,
                        recv_sem,
                        device_id={axis: d},
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()
                return carry

            jax.lax.fori_loop(0, counts[j], body, 0)

        for j in range(replicas):
            start_part(j, only=None if direct else True)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=(
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(
                (replicas, dcp, part, rows, packing, head_width), kv_cache.dtype
            ),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=_EXCHANGE_COLLECTIVE_ID, has_side_effects=True
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="minimax_m3_msa_dcp_exchange_start",
    )(pages, live, rank, kv_cache)


def exchange_head_pages_wait(
    future,
    live: jax.Array,
    rank: jax.Array,
    *,
    axis: str,
    dcp: int,
    replicas: int,
    direct: bool = True,
    in_flight: int = _EXCHANGE_PAGES_IN_FLIGHT,
    interpret: bool = False,
) -> jax.Array:
    """Retires `exchange_head_pages_start`'s copies (and, unless `direct`,
    forwards the received part to the head's other replicas): the same
    buffer `exchange_head_pages` returns."""
    # fwd_sem comes from the start: a peer may forward pages before this
    # rank enters the wait, when a scratch semaphore would not exist yet.
    out, send_sem, recv_sem, fwd_sem = future
    part = out.shape[2]
    out_shape = jax.ShapeDtypeStruct(out.shape, out.dtype)

    def kernel(live_ref, rank_ref, buf_ref, send_sem, recv_sem, fwd_sem, out_ref):
        del buf_ref  # aliased to out_ref
        me = rank_ref[0]
        replica = me % replicas
        counts = [live_part(live_ref[0], j, replicas, part) for j in range(replicas)]
        mine = jnp.where(replica == 0, counts[0], counts[-1])
        for j in range(1, replicas - 1):
            mine = jnp.where(replica == j, counts[j], mine)

        def page_wait(sem):
            def wait(i, carry):
                page = out_ref.at[0, 0, 0]
                pltpu.make_async_copy(page, page, sem).wait()
                return carry

            return wait

        # Our sends: one per page and destination (direct: every rank; else
        # the ranks of the page's part, one per head).
        issued = sum(counts) * (dcp if direct else dcp // replicas)
        jax.lax.fori_loop(0, issued, page_wait(send_sem), 0)
        recv_n = sum(counts) if direct else mine
        jax.lax.fori_loop(0, recv_n * dcp, page_wait(recv_sem), 0)
        if direct:
            return

        def forwards(i, op):
            for k in range(1, replicas):
                peer = me - replica + (replica + k) % replicas
                for src in range(dcp):
                    page = out_ref.at[replica, src, i]
                    op(
                        pltpu.make_async_remote_copy(
                            page,
                            page,
                            send_sem,
                            fwd_sem,
                            device_id={axis: peer},
                            device_id_type=pl.DeviceIdType.MESH,
                        )
                    )

        ahead = in_flight

        def step(i, carry):
            forwards(i, lambda c: c.start())
            pl.when(i >= ahead)(lambda: forwards(i - ahead, lambda c: c.wait_send()))
            return carry

        def drain(i, carry):
            forwards(i, lambda c: c.wait_send())
            return carry

        jax.lax.fori_loop(0, mine, step, 0)
        jax.lax.fori_loop(jnp.maximum(mine - ahead, 0), mine, drain, 0)
        jax.lax.fori_loop(0, (sum(counts) - mine) * dcp, page_wait(fwd_sem), 0)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            ],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
        ),
        out_shape=out_shape,
        input_output_aliases={2: 0},
        compiler_params=pltpu.CompilerParams(has_side_effects=True),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="minimax_m3_msa_dcp_exchange_wait",
    )(live, rank, out, send_sem, recv_sem, fwd_sem)


def _exchange_pages(
    kv_cache,
    scale_cache,
    pages,
    live,
    rank,
    *,
    dcp,
    replicas,
    head_width,
    axis,
    interpret,
    future=None,
    deps=(),
):
    """Every rank's ``pages [L]`` of the KV head each rank owns:
    ``(kv [L * dcp, S / 8, 4, 2 * D], scales [L * dcp, SCALE_ROWS, 128])``,
    rank ``src``'s page ``pages[j * L / replicas + i]`` at slot ``(j * dcp +
    src) * L / replicas + i``; only the pages `exchange_head_pages` calls
    live (`live` logical blocks) are set.

    The pages move by `exchange_head_pages`. The ``replicas`` ranks of a KV
    head each receive a ``1 / replicas`` part of the scales over an
    all-to-all and share them with an all-gather inside the replica group
    (one chip's cores at c64).

    `future`: the pages' copies were started by `exchange_head_pages_start`
    (direct = `exchange_direct()`); they are retired here, after `deps`
    (an optimization barrier orders the wait after them)."""
    num_pages = pages.shape[0]
    part = num_pages // replicas
    _, rows, packing, _ = kv_cache.shape
    heads = dcp // replicas
    if kv_cache.shape[3] != heads * head_width:
        raise ValueError(
            f"cache rows {kv_cache.shape} do not hold {heads} KV heads of "
            f"{head_width} columns"
        )
    from vllm_torchtpu import envs

    in_flight = max(1, int(envs.TPU_MINIMAX_M3_DCP_EXCHANGE_IN_FLIGHT))
    direct = exchange_direct(replicas)
    global _exchange_logged
    if (direct or in_flight != _EXCHANGE_PAGES_IN_FLIGHT) and not _exchange_logged:
        _exchange_logged = True
        init_logger(__name__).warning(
            "MSA DCP page exchange: %d pages in flight, direct=%s "
            "(TPU_MINIMAX_M3_DCP_EXCHANGE_IN_FLIGHT / _DIRECT).",
            in_flight,
            direct,
        )
    if future is not None:
        future, _ = jax.lax.optimization_barrier((future, deps))
        recv = exchange_head_pages_wait(
            future,
            live.reshape(1),
            rank.reshape(1),
            axis=axis,
            dcp=dcp,
            replicas=replicas,
            direct=direct,
            in_flight=in_flight,
            interpret=interpret,
        )
        # The scales' XLA collectives run after the page copies retire.
        recv, scale_cache = jax.lax.optimization_barrier((recv, scale_cache))
    elif exchange_scales():
        # The scale rows ride the page copies: no gather, collectives or pad.
        pages_of, row0 = msa_fp4.split_scales(scale_cache)
        recv, recv_sc = exchange_head_pages(
            kv_cache,
            pages,
            live.reshape(1),
            rank.reshape(1),
            axis=axis,
            dcp=dcp,
            replicas=replicas,
            head_width=head_width,
            interpret=interpret,
            in_flight=in_flight,
            direct=direct,
            scale_cache=pages_of,
            scale_row0=row0,
        )
        recv = recv.reshape(num_pages * dcp, rows, packing, head_width)
        return recv, recv_sc.reshape(num_pages * dcp, *recv_sc.shape[3:])
    else:
        recv = exchange_head_pages(
            kv_cache,
            pages,
            live.reshape(1),
            rank.reshape(1),
            axis=axis,
            dcp=dcp,
            replicas=replicas,
            head_width=head_width,
            interpret=interpret,
            in_flight=in_flight,
            direct=direct,
        )
    # Destination rank d = h * replicas + j gets part j of head h.
    scale_cache, row0 = msa_fp4.split_scales(scale_cache)
    sc = scale_cache[pages][:, row0 : row0 + 2 * heads]  # K/V rows of every head
    sc = sc.reshape(replicas, part, heads, 2, sc.shape[-1])
    # [dcp, part * 2, lanes]: a [.., 2, lanes] block pads its (8, 128) tiles
    # four times over.
    send_sc = jnp.moveaxis(sc, 2, 0).reshape(dcp, part * 2, sc.shape[-1])
    recv_sc = jax.lax.all_to_all(send_sc, axis, 0, 0, tiled=True)
    if replicas > 1:
        groups = [list(range(h * replicas, (h + 1) * replicas)) for h in range(heads)]
        recv_sc = jax.lax.all_gather(
            recv_sc, axis, axis_index_groups=groups, tiled=True
        )
    recv = recv.reshape(num_pages * dcp, rows, packing, head_width)
    recv_sc = recv_sc.reshape(num_pages * dcp, 2, recv_sc.shape[-1])
    recv_sc = jnp.pad(recv_sc, ((0, 0), (0, msa_fp4.SCALE_ROWS - 2), (0, 0)))
    return recv, recv_sc


def _own_selection(candidates, *, dcp, replicas, topk, axis, interpret):
    """The selection ``[T, topk]`` of this rank's KV head, from every rank's
    packed ``candidates [T, G * 2 * topk]``. Each replica of the head merges
    the candidates of a ``1 / replicas`` share of the tokens (an all-to-all
    of that share) and the replicas all-gather the merged selections, which
    are 1/8 the candidates' bytes."""
    num_tokens = candidates.shape[0]
    heads = dcp // replicas
    split = replicas if num_tokens % replicas == 0 else 1
    part = num_tokens // split
    per_head = candidates.reshape(num_tokens, heads, 2 * topk).transpose(1, 0, 2)
    # Destination rank d = h * replicas + j gets head h's tokens of share j
    # (every share, when the tokens do not split).
    send = per_head.reshape(heads, split, part, 2 * topk)
    send = jnp.repeat(send, replicas // split, axis=1).reshape(dcp, part, 2 * topk)
    own = jax.lax.all_to_all(send, axis, 0, 0, tiled=True)  # [dcp, T / split, 2K]
    selected = msa_dcp.merge_head_candidates(own, topk, interpret)[:, 0]
    if split > 1:
        groups = [list(range(h * replicas, (h + 1) * replicas)) for h in range(heads)]
        selected = jax.lax.all_gather(
            selected, axis, axis_index_groups=groups, tiled=True
        )
    return selected


HP_DECODE_MERGE_A2A_ENV = "TPU_MINIMAX_M3_DCP_HP_DECODE_MERGE_A2A"
_hp_decode_merge_a2a_logged = False


def hp_decode_merge_a2a() -> bool:
    """TPU_MINIMAX_M3_DCP_HP_DECODE_MERGE_A2A, read at trace time through
    `envs`: head-parallel steps merge their decode rows' partials by
    `_merge_partials_a2a`."""
    global _hp_decode_merge_a2a_logged
    from vllm_torchtpu import envs

    on = bool(envs.TPU_MINIMAX_M3_DCP_HP_DECODE_MERGE_A2A)
    if on and not _hp_decode_merge_a2a_logged:
        _hp_decode_merge_a2a_logged = True
        init_logger(__name__).warning(
            "%s=1: head-parallel steps merge the decode rows' DCP partials "
            "by all-to-all + f32 kernel.",
            HP_DECODE_MERGE_A2A_ENV,
        )
    return on


HP_ROUND_FACTOR_ENV = "TPU_MINIMAX_M3_DCP_HP_ROUND_FACTOR"
_round_factor_logged = False


def hp_round_factor() -> int:
    """Max-length requests' blocks per head-parallel round
    (TPU_MINIMAX_M3_DCP_HP_ROUND_FACTOR, read at trace time through `envs`)."""
    global _round_factor_logged
    from vllm_torchtpu import envs

    factor = max(1, int(envs.TPU_MINIMAX_M3_DCP_HP_ROUND_FACTOR))
    if factor > 1 and not _round_factor_logged:
        _round_factor_logged = True
        init_logger(__name__).warning(
            "%s=%d: head-parallel MSA prefill rounds of %d max-length "
            "requests' blocks.",
            HP_ROUND_FACTOR_ENV,
            factor,
            factor,
        )
    return factor


HP_ROUND_LADDER_ENV = "TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER"
_round_ladder_logged = False


def hp_round_ladder(block_size: int, dcp: int, replicas: int, capacity: int) -> list[int]:
    """Smaller head-parallel round buffers, in logical blocks per rank
    (multiples of ``replicas``, below ``capacity``), from
    TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER ("<tokens>/<tokens>/...": the step's
    prefill requests' summed context each rung holds; read at trace time
    through `envs`). Empty -> none."""
    global _round_ladder_logged
    from vllm_torchtpu import envs

    spec = envs.TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER.replace(",", "/")
    rungs = set()
    for tok in filter(None, (x.strip() for x in spec.split("/"))):
        blocks = -(-int(tok) // (block_size * dcp))
        blocks = -(-blocks // replicas) * replicas
        if 0 < blocks < capacity:
            rungs.add(blocks)
    rungs = sorted(rungs)
    if rungs and not _round_ladder_logged:
        _round_ladder_logged = True
        init_logger(__name__).warning(
            "%s=%s: head-parallel MSA prefill steps whose prefill requests fit "
            "one round of %s logical blocks per rank (of %d) exchange and attend "
            "in that smaller buffer.",
            HP_ROUND_LADDER_ENV,
            spec,
            rungs,
            capacity,
        )
    return rungs


HP_ROW_SPLIT_ENV = "TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"
_row_split_logged = False


def hp_row_split() -> bool:
    """TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT, read at trace time through `envs`."""
    global _row_split_logged
    from vllm_torchtpu import envs

    on = bool(envs.TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT)
    if on and not _row_split_logged:
        _row_split_logged = True
        init_logger(__name__).warning(
            "%s=1: head-parallel MSA prefill replicas split query tiles, "
            "each tile with all its KV head's query heads.",
            HP_ROW_SPLIT_ENV,
        )
    return on


def _attend_row_split(attend, q, kv, keys, local, limit, page_table, *, rank,
                      replicas, block_q, axis, dcp):
    """``attend`` (head-major out ``[Hl, T, D]``) with the rows split across
    the ``replicas`` ranks of this rank's KV head instead of the query
    heads: replica ``j`` takes the query tiles ``i`` with ``i % replicas ==
    j`` with every replica's query heads, ``[T / replicas, replicas * Hl,
    D]``. A tile then carries ``replicas`` times the rows over the same
    selected-page union (the pages are the tile's tokens' selections,
    whatever the heads). Queries and outputs move by an all-to-all inside
    the replica group; each (row, head) is attended over the same pages in
    the same order, so the outputs match the head split bit for bit."""
    num_tokens, num_q_heads, head_dim = q.shape
    tiles = num_tokens // (replicas * block_q)
    groups = [list(range(h * replicas, (h + 1) * replicas))
              for h in range(dcp // replicas)]
    j = jnp.reshape(rank, (-1,))[0] % replicas

    def mine(x):
        # Rows of tiles i % replicas == j: [tiles * block_q, ...].
        x = x.reshape(tiles, replicas, block_q, *x.shape[1:])
        x = jax.lax.dynamic_index_in_dim(x, j, axis=1, keepdims=False)
        return x.reshape(tiles * block_q, *x.shape[2:])

    # [dst replica, tiles, block_q, Hl, D] -> [src replica, ...].
    send = q.reshape(tiles, replicas, block_q, num_q_heads, head_dim)
    send = jnp.moveaxis(send, 1, 0)
    recv = jax.lax.all_to_all(send, axis, 0, 0, axis_index_groups=groups)
    # Query heads replica-major: [tiles * block_q, replicas * Hl, D].
    q_rows = jnp.moveaxis(recv, 0, 2).reshape(
        tiles * block_q, replicas * num_q_heads, head_dim
    )
    out, _ = attend(q_rows, kv, mine(keys), mine(local), mine(limit), page_table)
    # out [replicas * Hl, tiles * block_q, D] -> back to the heads' owners.
    send = out.reshape(replicas, num_q_heads, tiles, block_q, head_dim)
    recv = jax.lax.all_to_all(send, axis, 0, 0, axis_index_groups=groups)
    # recv [src replica (= tile phase), Hl, tiles, block_q, D] -> [Hl, T, D].
    return jnp.moveaxis(recv, 0, 2).reshape(num_q_heads, num_tokens, head_dim)


HP_BALANCE_ENV = "TPU_MINIMAX_M3_DCP_HP_BALANCE"
# (max fraction of a replica's tiles that may move, a priced tile's fixed
# cost, the cost of moving any tiles), in pages of kernel work: at 8:64 a
# tile costs ~300 bundles plus ~66 per distinct page (K1 LLO counts); a page
# is ~0.06 us of the served call and a move (partner pages, helper call)
# ~200 us (k4 mb2).
_BALANCE_DEFAULTS = (0.25, 4.0, 3000.0)
_BALANCE_SYNC = "sync"
_balance_logged = False


def hp_balance() -> tuple[float, float, float, bool] | None:
    """TPU_MINIMAX_M3_DCP_HP_BALANCE (read at trace time through `envs`):
    ``(max tile fraction, tile cost, call cost, asynchronous partner
    exchange)``, or None when off. A fourth field ``sync`` runs the partner
    exchange as one kernel after the rank's own call."""
    global _balance_logged
    from vllm_torchtpu import envs

    spec = envs.TPU_MINIMAX_M3_DCP_HP_BALANCE.strip()
    if spec in ("", "0"):
        return None
    parts = [] if spec == "1" else spec.split(":")
    if len(parts) not in (0, 3, 4) or parts[3:] not in ([], [_BALANCE_SYNC]):
        raise ValueError(
            f"{HP_BALANCE_ENV}={spec!r}: want 1 or frac:tile_cost:call_cost[:sync]"
        )
    cfg = tuple(
        float(p) if p else d for p, d in zip(parts[:3], _BALANCE_DEFAULTS)
    ) if parts else _BALANCE_DEFAULTS
    cfg = (*cfg, parts[3:] != [_BALANCE_SYNC])
    if not _balance_logged:
        _balance_logged = True
        init_logger(__name__).warning(
            "%s=%s: head-parallel MSA prefill hands up to %.0f%% of a heavy KV "
            "head's tiles to its partner head (tile cost %.1f, call cost %.1f).",
            HP_BALANCE_ENV,
            spec,
            100 * cfg[0],
            cfg[1],
            cfg[2],
        )
    return cfg


def tile_costs(keys: jax.Array, block_q: int, tile_cost: float) -> jax.Array:
    """Kernel work of each ``block_q``-row tile of ``keys [T, K]`` (page
    slots, -1 for none), 0 for a tile without any slot, else `tile_cost`
    plus the tile's first row's slots and each further row's slots that
    the row before it does not hold (a cheap stand-in for the distinct
    slots: no sort, ``block_q * K * K`` compares per tile)."""
    num_tokens, topk = keys.shape
    x = keys.reshape(num_tokens // block_q, block_q, topk)
    prev = jnp.concatenate([jnp.full_like(x[:, :1], -1), x[:, :-1]], axis=1)
    seen = jnp.any(x[..., :, None] == prev[..., None, :], axis=-1)
    fresh = jnp.sum((x >= 0) & ~seen, axis=(1, 2)).astype(jnp.float32)
    return jnp.where(fresh > 0, fresh + tile_cost, 0.0)


def balance_plan(
    costs: jax.Array, *, replicas: int, max_tiles: int, call_cost: float
) -> tuple[jax.Array, jax.Array]:
    """``(steal, end)``, each ``[heads, replicas]`` int32, from ``costs
    [heads, tiles]`` (head ``h``'s tile prices; tile ``i`` runs on replica
    ``i % replicas``, as its ``i // replicas``-th tile). ``end[h, j]``: one
    past head ``h``'s replica ``j``'s last priced tile. ``steal[h, j]``:
    the tiles before ``end`` that replica hands to replica ``j`` of the
    partner head ``h ^ 1``, as many (at most `max_tiles`) as keep it at
    least as loaded as the partner plus `call_cost`; 0 unless ``h`` is the
    heavier of the pair on that replica."""
    heads, tiles = costs.shape
    per = tiles // replicas
    c = costs.reshape(heads, per, replicas).transpose(0, 2, 1)
    work = jnp.sum(c, axis=-1)  # [heads, replicas]
    idx = jnp.arange(per, dtype=jnp.int32)
    end = jnp.max(jnp.where(c > 0, idx + 1, 0), axis=-1)
    partner = jnp.arange(heads) ^ 1
    budget = (work - work[partner] - call_cost) / 2
    # tail[.., k]: the k + 1 tiles before `end` (prices >= 0: non-decreasing);
    # the reversed prices start with `per - end` zeros.
    csum = jnp.cumsum(c[..., ::-1], axis=-1)
    at = (per - end)[..., None] + jnp.arange(max_tiles, dtype=jnp.int32)
    tail = jnp.take_along_axis(csum, jnp.minimum(at, per - 1), axis=-1)
    tail = jnp.where(at < per, tail, jnp.inf)
    n = jnp.sum(tail <= budget[..., None], axis=-1)
    return jnp.where(budget > 0, n, 0).astype(jnp.int32), end.astype(jnp.int32)


def _partner_of(me, replicas):
    return ((me // replicas) ^ 1) * replicas + me % replicas


def _partner_sends(pages_ref, live_ref, rank_ref, need_ref, xfer_ref, kv_ref,
                   sc_ref, q_ref, k_ref, out_ref, sc_out_ref, q_out_ref,
                   k_out_ref, sems, *, axis, dcp, replicas, part, head_width,
                   scale_row0, block_q):
    """`exchange_partner_pages_start`'s kernel body: the barrier, then
    every copy started."""
    send_sem, recv_sem, sc_send_sem, sc_recv_sem, q_send, _, k_send, _ = sems
    q_recv, k_recv = sems[5], sems[7]
    me = rank_ref[0]
    counts = [live_part(live_ref[0], j, replicas, part) for j in range(replicas)]
    barrier = pltpu.get_barrier_semaphore()
    for d in range(dcp):
        pl.semaphore_signal(
            barrier,
            inc=1,
            device_id={axis: jnp.int32(d)},
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, dcp)

    def remote(src, dst, send, recv, d):
        return pltpu.make_async_remote_copy(
            src, dst, send, recv,
            device_id={axis: d},
            device_id_type=pl.DeviceIdType.MESH,
        )

    # The tiles this rank gives: queries and keys to the partner's window.
    partner = _partner_of(me, replicas)
    give, src0, dst0 = xfer_ref[0], xfer_ref[1], xfer_ref[2]

    def tile(k, carry):
        src = pl.ds((src0 + k) * block_q, block_q)
        dst = pl.ds((dst0 + k) * block_q, block_q)
        remote(q_ref.at[src], q_out_ref.at[dst], q_send, q_recv, partner).start()
        remote(k_ref.at[src], k_out_ref.at[dst], k_send, k_recv, partner).start()
        return carry

    jax.lax.fori_loop(0, give, tile, 0)

    for j in range(replicas):

        def body(i, carry, j=j):
            page = pages_ref[j * part + i]
            for d in range(dcp):
                other = (d // replicas) ^ 1

                def send(d=d, other=other):
                    columns = pl.ds(other * head_width, head_width)
                    remote(
                        kv_ref.at[page, :, :, columns],
                        out_ref.at[j, me, i],
                        send_sem, recv_sem, d,
                    ).start()
                    remote(
                        sc_ref.at[page, pl.ds(scale_row0 + 2 * other, 2)],
                        sc_out_ref.at[j, me, i, pl.ds(0, 2)],
                        sc_send_sem, sc_recv_sem, d,
                    ).start()

                pl.when(need_ref[d] > 0)(send)
            return carry

        jax.lax.fori_loop(0, counts[j], body, 0)


def _partner_waits(live_ref, rank_ref, need_ref, xfer_ref, out_ref, sc_out_ref,
                   q_out_ref, k_out_ref, sems, *, dcp, replicas, part, block_q):
    """`exchange_partner_pages_wait`'s kernel body: this rank's sends and,
    if it takes tiles, its receives retired."""
    send_sem, recv_sem, sc_send_sem, sc_recv_sem = sems[:4]
    q_send, q_recv, k_send, k_recv = sems[4:]
    me = rank_ref[0]
    total = sum(live_part(live_ref[0], j, replicas, part) for j in range(replicas))
    takers = sum((need_ref[d] > 0).astype(jnp.int32) for d in range(dcp))
    mine = jnp.where(need_ref[me] > 0, total * dcp, 0)

    def waits(sem, sc_sem):
        def wait(i, carry):
            page = out_ref.at[0, 0, 0]
            pltpu.make_async_copy(page, page, sem).wait()
            rows = sc_out_ref.at[0, 0, 0, pl.ds(0, 2)]
            pltpu.make_async_copy(rows, rows, sc_sem).wait()
            return carry

        return wait

    def tile_waits(q_sem, k_sem):
        def wait(i, carry):
            rows = pl.ds(0, block_q)
            pltpu.make_async_copy(q_out_ref.at[rows], q_out_ref.at[rows], q_sem).wait()
            pltpu.make_async_copy(k_out_ref.at[rows], k_out_ref.at[rows], k_sem).wait()
            return carry

        return wait

    jax.lax.fori_loop(0, xfer_ref[0], tile_waits(q_send, k_send), 0)
    jax.lax.fori_loop(0, total * takers, waits(send_sem, sc_send_sem), 0)
    jax.lax.fori_loop(0, xfer_ref[3], tile_waits(q_recv, k_recv), 0)
    jax.lax.fori_loop(0, mine, waits(recv_sem, sc_recv_sem), 0)


_PARTNER_SEMS = 8  # pages, scales, queries, keys: send / receive each


def _partner_shapes(kv_cache, scale_cache, num_pages, q_rows, keys, window, *,
                    dcp, replicas, head_width):
    _, rows, packing, _ = kv_cache.shape
    if kv_cache.dtype != jnp.uint8:
        raise ValueError(f"4-bit (uint8) caches only, got {kv_cache.dtype}")
    part = num_pages // replicas
    return part, (
        jax.ShapeDtypeStruct(
            (replicas, dcp, part, rows, packing, head_width), kv_cache.dtype
        ),
        jax.ShapeDtypeStruct((replicas, dcp, part, *scale_cache.shape[1:]), jnp.float32),
        jax.ShapeDtypeStruct((window, *q_rows.shape[1:]), q_rows.dtype),
        jax.ShapeDtypeStruct((window, *keys.shape[1:]), keys.dtype),
    )


def _return_shape(q_rows, window):
    return jax.ShapeDtypeStruct((window, *q_rows.shape[1:]), q_rows.dtype)


def exchange_partner_pages_start(
    kv_cache: jax.Array,
    scale_cache: jax.Array,
    scale_row0: int,
    pages: jax.Array,
    live: jax.Array,
    rank: jax.Array,
    need: jax.Array,
    xfer: jax.Array,
    q_rows: jax.Array,
    keys: jax.Array,
    *,
    window: int,
    axis: str,
    dcp: int,
    replicas: int,
    head_width: int,
    block_q: int,
    interpret: bool = False,
):
    """Starts copying, to every rank ``d`` with ``need[d] > 0``, this rank's
    live pages (every part) of ``d``'s partner KV head ``d // replicas ^
    1`` and their K/V scale rows, into buffers laid out as
    `exchange_head_pages` lays out the partner head's own (``[replicas, dcp,
    L / replicas, ...]``, scale rows at 0 / 1): a key's slot is the same on
    both heads' ranks. Also copies the ``xfer[0]`` tiles of `q_rows` /
    `keys` from tile ``xfer[1]`` into the partner rank's ``[window, ...]``
    buffers from tile ``xfer[2]`` (``xfer[3]``: the tiles this rank
    receives). Returns the future (four buffers, the return buffer, eight
    semaphores, the return's two) for `exchange_partner_pages_wait` and
    `return_partner_tiles`; the inputs stay unchanged until then."""
    part, shapes = _partner_shapes(
        kv_cache, scale_cache, pages.shape[0], q_rows, keys, window, dcp=dcp,
        replicas=replicas, head_width=head_width,
    )

    def kernel(pages_ref, live_ref, rank_ref, need_ref, xfer_ref, kv_ref, sc_ref,
               q_ref, k_ref, out_ref, sc_out_ref, q_out_ref, k_out_ref, ret_ref,
               *sems):
        del ret_ref  # `return_partner_tiles`'s, live from here on
        _partner_sends(
            pages_ref, live_ref, rank_ref, need_ref, xfer_ref, kv_ref, sc_ref,
            q_ref, k_ref, out_ref, sc_out_ref, q_out_ref, k_out_ref,
            sems[:_PARTNER_SEMS], axis=axis, dcp=dcp, replicas=replicas,
            part=part, head_width=head_width, scale_row0=scale_row0,
            block_q=block_q,
        )

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    sem_spec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1,),
            in_specs=[any_spec] * 4,
            out_specs=(any_spec,) * 5 + (sem_spec,) * (_PARTNER_SEMS + 2),
        ),
        out_shape=shapes
        + (_return_shape(q_rows, window),)
        + (pltpu.SemaphoreType.DMA(()),) * (_PARTNER_SEMS + 2),
        compiler_params=pltpu.CompilerParams(
            collective_id=_EXCHANGE_COLLECTIVE_ID, has_side_effects=True
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="minimax_m3_msa_dcp_exchange_partner_start",
    )(pages, live, rank, need, xfer, kv_cache, scale_cache, q_rows, keys)


def exchange_partner_pages_wait(
    future,
    live: jax.Array,
    rank: jax.Array,
    need: jax.Array,
    xfer: jax.Array,
    *,
    dcp: int,
    replicas: int,
    block_q: int,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """Retires `exchange_partner_pages_start`'s copies: this rank's sends
    and its receives (the partner head's pages if ``need[rank] > 0``, the
    ``xfer[3]`` tiles). Returns ``(pages, scales, queries, keys)``."""
    bufs = future[:4]
    sems = future[5 : 5 + _PARTNER_SEMS]
    part = bufs[0].shape[2]

    def kernel(live_ref, rank_ref, need_ref, xfer_ref, *refs):
        sems = refs[4 : 4 + _PARTNER_SEMS]
        out_ref, sc_out_ref, q_out_ref, k_out_ref = refs[4 + _PARTNER_SEMS :]
        _partner_waits(
            live_ref, rank_ref, need_ref, xfer_ref, out_ref, sc_out_ref,
            q_out_ref, k_out_ref, sems, dcp=dcp, replicas=replicas, part=part,
            block_q=block_q,
        )

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    sem_spec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(1,),
            in_specs=[any_spec] * 4 + [sem_spec] * _PARTNER_SEMS,
            out_specs=(any_spec,) * 4,
        ),
        out_shape=tuple(jax.ShapeDtypeStruct(b.shape, b.dtype) for b in bufs),
        input_output_aliases={4: 0, 5: 1, 6: 2, 7: 3},
        compiler_params=pltpu.CompilerParams(has_side_effects=True),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="minimax_m3_msa_dcp_exchange_partner_wait",
    )(live, rank, need, xfer, *bufs, *sems)


def exchange_partner_pages(
    kv_cache, scale_cache, scale_row0, pages, live, rank, need, xfer, q_rows,
    keys, *, window, axis, dcp, replicas, head_width, block_q, interpret=False,
) -> tuple[jax.Array, ...]:
    """`exchange_partner_pages_start` and `_wait` as one synchronous kernel
    (interpret mode does not model DMA semaphores that outlive a kernel)."""
    part, shapes = _partner_shapes(
        kv_cache, scale_cache, pages.shape[0], q_rows, keys, window, dcp=dcp,
        replicas=replicas, head_width=head_width,
    )

    def kernel(pages_ref, live_ref, rank_ref, need_ref, xfer_ref, kv_ref, sc_ref,
               q_ref, k_ref, out_ref, sc_out_ref, q_out_ref, k_out_ref, *sems):
        _partner_sends(
            pages_ref, live_ref, rank_ref, need_ref, xfer_ref, kv_ref, sc_ref,
            q_ref, k_ref, out_ref, sc_out_ref, q_out_ref, k_out_ref, sems,
            axis=axis, dcp=dcp, replicas=replicas, part=part,
            head_width=head_width, scale_row0=scale_row0, block_q=block_q,
        )
        _partner_waits(
            live_ref, rank_ref, need_ref, xfer_ref, out_ref, sc_out_ref,
            q_out_ref, k_out_ref, sems, dcp=dcp, replicas=replicas, part=part,
            block_q=block_q,
        )

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1,),
            in_specs=[any_spec] * 4,
            out_specs=(any_spec,) * 4,
            scratch_shapes=[pltpu.SemaphoreType.DMA(())] * _PARTNER_SEMS,
        ),
        out_shape=shapes,
        compiler_params=pltpu.CompilerParams(collective_id=_EXCHANGE_COLLECTIVE_ID),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="minimax_m3_msa_dcp_exchange_partner",
    )(pages, live, rank, need, xfer, kv_cache, scale_cache, q_rows, keys)


def return_partner_tiles(
    helped: jax.Array,
    rank: jax.Array,
    xfer: jax.Array,
    future=None,
    *,
    axis: str,
    replicas: int,
    block_q: int,
    interpret: bool = False,
) -> jax.Array:
    """The partner rank's outputs of the tiles this rank gave, ``[window,
    ...]`` like `helped` (only those tiles set): this rank copies the
    ``xfer[0]`` tiles of its `helped` from tile ``xfer[1]`` into the
    partner's output at the same tiles and receives ``xfer[2]`` tiles.
    With `future` (`exchange_partner_pages_start`'s) the output is the
    start's return buffer, live on both partners since then; without, the
    two partners meet at a barrier first when either moves tiles."""
    shape = jax.ShapeDtypeStruct(helped.shape, helped.dtype)

    def kernel(rank_ref, xfer_ref, helped_ref, *refs):
        if future is None:
            out_ref, send_sem, recv_sem = refs
        else:
            _, send_sem, recv_sem, out_ref = refs
        me = rank_ref[0]
        partner = _partner_of(me, replicas)
        take, first, give = xfer_ref[0], xfer_ref[1], xfer_ref[2]

        if future is None:

            @pl.when(take + give > 0)
            def _():
                barrier = pltpu.get_barrier_semaphore()
                pl.semaphore_signal(
                    barrier, inc=1, device_id={axis: partner},
                    device_id_type=pl.DeviceIdType.MESH,
                )
                pl.semaphore_wait(barrier, 1)

        def copy(k):
            rows = pl.ds((first + k) * block_q, block_q)
            return pltpu.make_async_remote_copy(
                helped_ref.at[rows], out_ref.at[rows], send_sem, recv_sem,
                device_id={axis: partner}, device_id_type=pl.DeviceIdType.MESH,
            )

        def start(k, carry):
            copy(k).start()
            return carry

        def wait_send(k, carry):
            copy(k).wait_send()
            return carry

        def wait_recv(k, carry):
            rows = pl.ds(0, block_q)
            pltpu.make_async_copy(out_ref.at[rows], out_ref.at[rows], recv_sem).wait()
            return carry

        jax.lax.fori_loop(0, take, start, 0)
        jax.lax.fori_loop(0, take, wait_send, 0)
        jax.lax.fori_loop(0, give, wait_recv, 0)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    sem_spec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    if future is None:
        return pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                grid=(1,),
                in_specs=[any_spec],
                out_specs=any_spec,
                scratch_shapes=[pltpu.SemaphoreType.DMA(())] * 2,
            ),
            out_shape=shape,
            compiler_params=pltpu.CompilerParams(
                collective_id=_EXCHANGE_COLLECTIVE_ID, has_side_effects=True
            ),
            interpret=pltpu.InterpretParams() if interpret else False,
            name="minimax_m3_msa_dcp_return_partner_tiles",
        )(rank, xfer, helped)
    ret, send_sem, recv_sem = future[4], future[-2], future[-1]
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[any_spec, any_spec, sem_spec, sem_spec],
            out_specs=any_spec,
        ),
        out_shape=shape,
        input_output_aliases={3: 0},
        compiler_params=pltpu.CompilerParams(has_side_effects=True),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="minimax_m3_msa_dcp_return_partner_tiles_async",
    )(rank, xfer, helped, ret, send_sem, recv_sem)


def _attend_row_split_balanced(attend, q, kv, keys, local, limit, page_table, *,
                               steal, end, max_tiles, partner_start,
                               partner_wait, rank, replicas, block_q, axis, dcp,
                               interpret, asynchronous):
    """`_attend_row_split` with `balance_plan`'s ``(steal, end)``: this rank
    (head ``h``, replica ``j``) leaves its ``steal[h, j]`` tiles before
    ``end[h, j]`` to replica ``j`` of the partner head, which attends them
    after its own tiles over the partner head's pages and sends the outputs
    back (`return_partner_tiles`). The given tiles' queries and keys ride
    the partner page exchange (`partner_start` before this rank's own call,
    `partner_wait` after it, which gives ``(attend over the partner pages,
    queries, keys)``) into a window of `max_tiles` tiles ending at ``end``
    where it fits. Every tile is the one the plain row split computes (same
    rows, heads, pages, slots and order), so the outputs match it bit for
    bit."""
    num_tokens, num_q_heads, head_dim = q.shape
    tiles = num_tokens // (replicas * block_q)
    groups = [list(range(h * replicas, (h + 1) * replicas))
              for h in range(dcp // replicas)]
    me = jnp.reshape(rank, (-1,))[0]
    j = me % replicas
    head = me // replicas
    give, my_end = steal[head, j], end[head, j]
    take, its_end = steal[head ^ 1, j], end[head ^ 1, j]

    def mine(x):
        x = x.reshape(tiles, replicas, block_q, *x.shape[1:])
        x = jax.lax.dynamic_index_in_dim(x, j, axis=1, keepdims=False)
        return x.reshape(tiles * block_q, *x.shape[2:])

    send = q.reshape(tiles, replicas, block_q, num_q_heads, head_dim)
    send = jnp.moveaxis(send, 1, 0)
    recv = jax.lax.all_to_all(send, axis, 0, 0, axis_index_groups=groups)
    q_rows = jnp.moveaxis(recv, 0, 2).reshape(
        tiles * block_q, replicas * num_q_heads, head_dim
    )
    keys, local, limit = mine(keys), mine(local), mine(limit)
    window = max_tiles * block_q

    def start_of(e):  # first tile of the window ending at tile e
        return jnp.clip(e - max_tiles, 0, tiles - max_tiles)

    def win(x, first):
        return jax.lax.dynamic_slice_in_dim(x, first * block_q, window, 0)

    my_first, its_first = start_of(my_end), start_of(its_end)
    row_tile = jnp.arange(tiles * block_q, dtype=jnp.int32) // block_q
    given = (row_tile >= my_end - give) & (row_tile < my_end)
    own_keys = jnp.where(given[:, None], -1, keys)
    own_local = jnp.where(given, -1, local)
    xfer = jnp.stack([give, my_end - give, my_end - give - my_first, take])
    xfer = xfer.astype(jnp.int32)
    win_tile = jnp.arange(window, dtype=jnp.int32) // block_q
    its_tile = its_first + win_tile
    taken = (its_tile >= its_end - take) & (its_tile < its_end)
    my_tile = my_first + win_tile
    mask = (my_tile >= my_end - give) & (my_tile < my_end)

    def moved(q_rows):
        future = partner_start(q_rows, keys, xfer)
        future, q_rows, own_keys_ = jax.lax.optimization_barrier(
            (future, q_rows, own_keys)
        )
        out, _ = attend(q_rows, kv, own_keys_, own_local, limit, page_table)
        attend_partner, q_win, k_win = partner_wait(future, out, xfer)

        def helper(args):
            q_win, k_win = args
            return attend_partner(
                jnp.where(taken[:, None, None], q_win, jnp.zeros((), q_win.dtype)),
                jnp.where(taken[:, None], k_win, -1),
                jnp.where(taken, win(local, its_first), -1),
                win(limit, its_first),
            )[0].transpose(1, 0, 2)

        def idle(args):
            return jnp.zeros((window, replicas * num_q_heads, head_dim), out.dtype)

        helped = jax.lax.cond(take > 0, helper, idle, (q_win, k_win))
        back = return_partner_tiles(
            helped,
            jnp.reshape(me, (1,)),
            jnp.stack([take, its_end - take - its_first, give]).astype(jnp.int32),
            future if asynchronous else None,
            axis=axis, replicas=replicas, block_q=block_q, interpret=interpret,
        )
        old = jax.lax.dynamic_slice_in_dim(out, my_first * block_q, window, 1)
        tail = jnp.where(mask[None, :, None], back.transpose(1, 0, 2), old)
        return jax.lax.dynamic_update_slice_in_dim(out, tail, my_first * block_q, 1)

    def stay(q_rows):
        return attend(q_rows, kv, keys, local, limit, page_table)[0]

    # Group-uniform (every rank holds the same plan): the group's partner
    # kernels and barriers run in the same branch on all its ranks.
    out = jax.lax.cond(jnp.sum(steal) > 0, moved, stay, q_rows)
    send = out.reshape(replicas, num_q_heads, tiles, block_q, head_dim)
    recv = jax.lax.all_to_all(send, axis, 0, 0, axis_index_groups=groups)
    return jnp.moveaxis(recv, 0, 2).reshape(num_q_heads, num_tokens, head_dim)


def _head_parallel_attend(
    q,
    kv_cache,
    scale_cache,
    candidates,
    seq_lens,
    block_tables,
    cu,
    hp_req,
    tokens,
    *,
    axis,
    rank,
    dcp,
    kv_head_replicas,
    sm_scale,
    topk,
    interpret,
    between=None,
):
    """Head-major ``[Hl, T, D]`` attention (rotated basis) of the rows of the
    requests in ``hp_req`` over their pages of this rank's KV head; other
    rows are zero. ``tokens`` is `msa.token_positions` of the step.

    `between` (``(q, candidates) -> x``, with `exchange_async()`): work the
    first round's page copies overlap (they start before it and are retired
    after it and the selection); returns ``(out, x)`` then.

    Requests are attended in rounds of at most ``pages_per_seq`` logical
    blocks (one max-length request): each round exchanges the live pages
    of one page buffer and attends the round's rows in windows of one
    kernel call."""
    num_tokens, num_q_heads, head_dim = q.shape
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    width = block_tables_2d.shape[1]  # a max-length request's blocks
    capacity = width * hp_round_factor()  # logical blocks per round
    replicas = kv_head_replicas
    part = -(-capacity // replicas)  # each replica's share of the buffer

    seq_id, pos, valid = tokens
    seq_blocks = -(-seq_lens // (block_size * dcp))
    blocks = jnp.where(hp_req, seq_blocks, 0)
    seq_idx = jnp.arange(max_num_seqs, dtype=jnp.int32)
    # Rows per kernel call (the caller pads tokens to a block_q multiple).
    window = min(num_tokens, msa_attention.MAX_TOKENS_PER_CALL)
    own_block = pos // block_size
    limit = pos % block_size + 1
    head_width = 2 * head_dim

    def plan_round(first, capacity=capacity):
        csum = jnp.cumsum(jnp.where(seq_idx >= first, blocks, 0))
        # A request never exceeds the buffer (the block table's width); always
        # admitting `first` keeps the rounds moving if that ever broke.
        in_round = (
            hp_req & (seq_idx >= first) & ((csum <= capacity) | (seq_idx == first))
        )
        base = csum - blocks  # the request's first logical block in the buffer
        total = jnp.max(jnp.where(in_round, csum, 0))
        nxt = jnp.min(
            jnp.where((seq_idx >= first) & ~in_round & hp_req, seq_idx, max_num_seqs)
        )
        return csum, in_round, base, total, nxt

    def round_pages(plan, part=part):
        csum, _, base, total, _ = plan
        num_pages = part * replicas
        # Buffer page b holds logical block l = b % part * replicas + b //
        # part: logical block s - base[q] of request q.
        buf = jnp.arange(num_pages, dtype=jnp.int32)
        logical = buf % part * replicas + buf // part
        q_of = jnp.searchsorted(csum, logical, side="right", method="compare_all")
        q_of = jnp.minimum(q_of, max_num_seqs - 1)
        j = jnp.clip(logical - base[q_of], 0, width - 1)
        return jnp.where(logical < total, block_tables_2d[q_of, j], 0)

    def attend_round(plan, acc, merge, selected, future=None, deps=(), part=part):
        csum, in_round, base, total, _ = plan
        num_pages = part * replicas
        pages = round_pages(plan, part)
        sel_block = jnp.maximum(selected, 0)

        def slot(base, block):
            # Logical block l = base + block // dcp of rank block % dcp.
            logical = base + block // dcp
            j = logical % replicas
            return (j * dcp + block % dcp) * part + logical // replicas

        mine = valid & (_per_token(seq_id, in_round) > 0)
        tok_base = _per_token(seq_id, base)
        keys = jnp.where(
            mine[:, None] & (selected >= 0), slot(tok_base[:, None], sel_block), -1
        )
        local = jnp.where(mine, slot(tok_base, own_block), -1)
        kv, sc = _exchange_pages(
            kv_cache,
            scale_cache,
            pages,
            total,
            rank,
            dcp=dcp,
            replicas=replicas,
            head_width=head_width,
            axis=axis,
            interpret=interpret,
            future=future,
            deps=(keys, local, *deps),
        )
        lo = cu[jnp.min(jnp.where(in_round, seq_idx, max_num_seqs - 1))]
        hi = cu[jnp.max(jnp.where(in_round, seq_idx + 1, 0))]
        page_table = jnp.arange(dcp * num_pages, dtype=jnp.int32)

        attend = functools.partial(
            msa_attention_lse.msa_sparse_attention_lse,
            sm_scale=sm_scale,
            **hp_prefill_kernel_kwargs(),
            head=0,
            scale_cache=sc,
            k_scale_row=0,
            v_scale_row=1,
            head_major=True,
            interpret=interpret,
        )
        block_q = attend.keywords["block_q"]
        split = replicas > 1 and hp_row_split()

        def call(q, keys, local, limit):
            if split and q.shape[0] % (replicas * block_q) == 0:
                return _attend_row_split(
                    attend,
                    q,
                    kv,
                    keys,
                    local,
                    limit,
                    page_table,
                    rank=rank,
                    replicas=replicas,
                    block_q=block_q,
                    axis=axis,
                    dcp=dcp,
                )
            return attend(q, kv, keys, local, limit, page_table)[0]

        bal = hp_balance() if not merge and future is None else None
        if (
            bal is not None
            and split
            and (dcp // replicas) % 2 == 0
            and q.shape[0] % (replicas * block_q) == 0
        ):
            # TPU_MINIMAX_M3_DCP_HP_BALANCE: the heavier KV head of each pair
            # hands its last tiles to the lighter one (`balance_plan`).
            tiles = q.shape[0] // (replicas * block_q)
            max_tiles = max(1, min(tiles, int(bal[0] * tiles)))
            costs = tile_costs(keys, block_q, bal[1])
            costs = jax.lax.all_gather(costs, axis)[::replicas]
            steal, end = balance_plan(
                costs, replicas=replicas, max_tiles=max_tiles, call_cost=bal[2]
            )
            d = jnp.arange(dcp, dtype=jnp.int32)
            need = steal[(d // replicas) ^ 1, d % replicas]
            # Start after the first exchange (they share a barrier semaphore).
            need = need + (kv.reshape(-1)[0] & 0).astype(jnp.int32)
            sc_pages, sc_row0 = msa_fp4.split_scales(scale_cache)
            rank1 = jnp.reshape(rank, (1,))
            live1 = jnp.reshape(total, (1,))
            _, kv_rows, kv_packing, _ = kv_cache.shape

            # Interpret mode does not model DMA semaphores that outlive a
            # kernel: the synchronous variant, after this rank's own call.
            asynchronous = bal[3] and not interpret

            def partner(need, q_rows, keys_m, xfer):
                return (
                    exchange_partner_pages_start
                    if asynchronous
                    else exchange_partner_pages
                )(
                    kv_cache, sc_pages, sc_row0, pages, live1, rank1, need,
                    xfer, q_rows, keys_m, window=max_tiles * block_q, axis=axis, dcp=dcp,
                    replicas=replicas, head_width=head_width, block_q=block_q,
                    interpret=interpret,
                )

            def partner_start(*args):
                return partner(need, *args) if asynchronous else (need, args)

            def partner_wait(fut, deps, xfer):
                fut, _ = jax.lax.optimization_barrier((fut, deps))
                if asynchronous:
                    kv2, sc2, q_in, k_in = exchange_partner_pages_wait(
                        fut, live1, rank1, need, xfer, dcp=dcp,
                        replicas=replicas, block_q=block_q, interpret=interpret,
                    )
                else:
                    kv2, sc2, q_in, k_in = partner(fut[0], *fut[1])
                kv2 = kv2.reshape(num_pages * dcp, kv_rows, kv_packing, head_width)
                sc2 = sc2.reshape(num_pages * dcp, *sc2.shape[3:])

                def attend_partner(q, keys, local, limit):
                    return attend(q, kv2, keys, local, limit, page_table,
                                  scale_cache=sc2)

                return attend_partner, q_in, k_in

            return _attend_row_split_balanced(
                attend, q, kv, keys, local, limit, page_table,
                steal=steal, end=end, max_tiles=max_tiles,
                partner_start=partner_start,
                partner_wait=partner_wait, rank=rank, replicas=replicas,
                block_q=block_q, axis=axis, dcp=dcp, interpret=interpret,
                asynchronous=asynchronous,
            )

        if not merge:
            # One call over every row: the kernel chunks the rows and writes
            # its output in place; rows outside the round see no page and
            # come back zero (cheap empty tiles).
            return call(q, keys, local, limit)

        def window_body(i, acc):
            start = jnp.minimum(lo + i * window, num_tokens - window)

            def sl(x):
                return jax.lax.dynamic_slice_in_dim(x, start, window, 0)

            out = call(sl(q), sl(keys), sl(local), sl(limit))
            old = jax.lax.dynamic_slice_in_dim(acc, start, window, 1)
            out = jnp.where(sl(mine)[None, :, None], out, old)  # earlier rounds
            return jax.lax.dynamic_update_slice_in_dim(acc, out, start, 1)

        return jax.lax.fori_loop(0, -(-(hi - lo) // window), window_body, acc)

    def more_rounds_from(first, acc, selected):
        # Further rounds sit under a cond: a while loop that reads the cache
        # makes XLA copy the whole cache ahead of it (0.45 ms per layer at
        # c64).
        def more_rounds(state):
            def body(s):
                plan = plan_round(s[0])
                return plan[4], attend_round(plan, s[1], merge=True, selected=selected)

            return jax.lax.while_loop(lambda s: s[0] < max_num_seqs, body, state)

        _, acc = jax.lax.cond(
            first < max_num_seqs, more_rounds, lambda s: s, (first, acc)
        )
        return acc

    first = jnp.min(jnp.where(hp_req, seq_idx, max_num_seqs))
    rungs = [] if exchange_async() else hp_round_ladder(
        block_size, dcp, replicas, part * replicas
    )
    if rungs:
        # TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER: a step whose prefill requests
        # all fit the smallest rung that holds them attends in one round of
        # that buffer (a `lax.switch` in the same program). The full buffer
        # would also take them in one round, with the same pages in the same
        # order: a key's slot ``(j * dcp + src) * part + i`` orders like ``(j,
        # src, i)`` whatever ``part``, so the kernel's sorts, unions and
        # page cache see the same sequence and the output is the same bits.
        # The scales' gather, all-to-all, all-gather and padding (static, the
        # whole buffer) shrink with it.
        extra = None if between is None else between(q, candidates)
        selected = _own_selection(
            candidates,
            dcp=dcp,
            replicas=replicas,
            topk=topk,
            axis=axis,
            interpret=interpret,
        )
        need = jnp.sum(blocks)
        branch = sum((need > r).astype(jnp.int32) for r in rungs)

        def rung(r):
            def run(selected):
                plan = plan_round(first, capacity=r)
                return attend_round(
                    plan, None, merge=False, selected=selected, part=r // replicas
                )

            return run

        def full(selected):
            plan = plan_round(first)
            acc = attend_round(plan, None, merge=False, selected=selected)
            return more_rounds_from(plan[4], acc, selected)

        acc = jax.lax.switch(branch, [rung(r) for r in rungs] + [full], selected)
        return acc if between is None else (acc, extra)

    plan = plan_round(first)
    future, extra = None, None
    if between is not None and exchange_async():
        future = exchange_head_pages_start(
            kv_cache,
            round_pages(plan),
            plan[3].reshape(1),
            jnp.reshape(rank, (1,)),
            axis=axis,
            dcp=dcp,
            replicas=replicas,
            head_width=head_width,
            direct=exchange_direct(replicas),
            interpret=interpret,
        )
        # The overlapped work starts after the copies.
        future, q, candidates = jax.lax.optimization_barrier((future, q, candidates))
        extra = between(q, candidates)
    elif between is not None:
        extra = between(q, candidates)
    selected = _own_selection(
        candidates,
        dcp=dcp,
        replicas=replicas,
        topk=topk,
        axis=axis,
        interpret=interpret,
    )
    acc = attend_round(
        plan,
        None,
        merge=False,
        selected=selected,
        future=future,
        deps=() if future is None else (extra,),
    )
    acc = more_rounds_from(plan[4], acc, selected)
    return acc if between is None else (acc, extra)


def attend_and_merge(
    rank: jax.Array,
    q: jax.Array,
    kv_cache: jax.Array,
    scale_cache: jax.Array,
    candidates: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    axis: str,
    dcp: int,
    kv_head_replicas: int,
    sm_scale: float,
    topk: int,
    decode_rows: int,
    rs_out_precision: bool,
    head_parallel_min_tokens: int = HEAD_PARALLEL_MIN_TOKENS,
    interpret: bool = False,
    q_gathered: bool = False,
    q_dec: jax.Array | None = None,
) -> jax.Array:
    """This rank's (``rank``, ``int32[1]``) attention output ``[T, Hl, D]``
    (unrotated) for its rotated query heads ``q [T, Hl, D]`` and its packed
    ``candidates [T, G * 2 * topk]`` (`msa_dcp.msa_write_and_candidates_dcp`);
    4-bit caches. `q_gathered`: ``q`` is the group's unrotated query heads
    ``[T, dcp * Hl, D]`` (`EARLY_Q_GATHER_ENV`); only for steps that
    `uses_dcp_attention`. `q_dec`: on a head-parallel step, the group's
    unrotated query heads of the compacted decode rows ``[max_num_seqs *
    decode_rows, dcp * Hl, D]`` (`decode_row_plan`), gathered by the caller.
    Runs inside a ``shard_map`` over `axis` (the DCP group)."""
    num_tokens = q.shape[0]
    # The rank is an input: torch_tpu's SPMD partitioner refuses the
    # partition-id that `lax.axis_index` lowers to.
    rank = msa_dcp.rank_array(rank)[0]
    common = dict(
        axis=axis, rank=rank, dcp=dcp, sm_scale=sm_scale, topk=topk, interpret=interpret
    )
    max_num_seqs = seq_lens.shape[0]
    rows = max_num_seqs * decode_rows
    small = uses_dcp_attention(
        num_tokens, max_num_seqs, decode_rows, head_parallel_min_tokens
    )
    if q_gathered and not small:
        raise ValueError(
            f"q_gathered needs a DCP-attention step, got {num_tokens} tokens"
        )
    if small:
        return _dcp_attend(
            q,
            kv_cache,
            scale_cache,
            candidates,
            seq_lens,
            block_tables,
            cu_q_lens,
            rs_out_precision=rs_out_precision,
            q_gathered=q_gathered,
            **common,
        )
    # The decode requests' rows, compacted: their own cu_q_lens.
    _, hp_req, cu_d, src_row = decode_row_plan(
        cu_q_lens, distribution, max_num_seqs, decode_rows
    )
    p = jnp.arange(rows, dtype=jnp.int32)

    def decode_rows_attend(q, candidates, q_dec=q_dec):
        return _dcp_attend(
            q[src_row] if q_dec is None else q_dec,
            kv_cache,
            scale_cache,
            candidates[src_row],
            seq_lens,
            block_tables,
            cu_d,
            rs_out_precision=rs_out_precision,
            q_gathered=q_dec is not None,
            merge_a2a=hp_decode_merge_a2a(),
            **common,
        )

    tokens = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    hp_args = (q, kv_cache, scale_cache, candidates, seq_lens, block_tables)
    hp_args += (cu_q_lens, hp_req, tokens)
    if exchange_async():
        # The page exchange overlaps the decode rows' attention.
        hp_out, dcp_out = _head_parallel_attend(
            *hp_args,
            kv_head_replicas=kv_head_replicas,
            between=decode_rows_attend,
            **common,
        )
    else:
        dcp_out = decode_rows_attend(q, candidates)
        hp_out = _head_parallel_attend(
            *hp_args, kv_head_replicas=kv_head_replicas, **common
        )
    hp_out = msa_fp4.unrotate_output(hp_out.transpose(1, 0, 2))
    num_decode = cu_d[-1]

    def decode_prefix(out):
        # The runner orders decodes ahead of prefills: the decode rows are
        # then the step's first rows, one in-place update of `rows` rows.
        head = jnp.where((p < num_decode)[:, None, None], dcp_out, out[:rows])
        return jax.lax.dynamic_update_slice_in_dim(out, head, 0, 0)

    def scatter(out):
        # The decode rows go back to their places (the others are dropped):
        # a scatter of `rows` rows.
        dest = jnp.where(p < num_decode, src_row, num_tokens)
        return out.at[dest].set(dcp_out, mode="drop")

    prefix = jnp.all((p >= num_decode) | (src_row == p))
    return jax.lax.cond(prefix, decode_prefix, scatter, hp_out)

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
"""Stateless Pallas TPU attention math, KV unpacking, and overflow fix-up."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.sparse import dcp_exchange, native_sc_gather
from vllm_torchtpu.kernels.mla.sparse.config import (
    Q_ROPE_HEAD_DIM,
    AttentionKernelConfig,
    BodyExecutionStrategy,
    BodyOptimizationConfig,
    cdiv,
)
from vllm_torchtpu.kernels.mla.sparse.kernel_context import KernelGeometry
from vllm_torchtpu.kernels.mla.sparse.ref_registry import AttentionRefs
from vllm_torchtpu.kernels.mla.sparse.testing_faults import NO_FAULT_HOOKS, FaultHooks

_DEFAULT_BODY = BodyOptimizationConfig()
_FIXUP_NOPE_SEM = 0
_FIXUP_ROPE_SEM = 1


def rope_q_block(
    x: jax.Array,
    tab: jax.Array,
    fault: FaultHooks | str | None = None,
    *,
    hooks: FaultHooks = NO_FAULT_HOOKS,
) -> jax.Array:
    """Applies NeoX RoPE to the q_pe lanes of one raw head block as ``[q_pe | 0]``.

    ``x`` is ``[heads, 128]`` covering lanes ``128:256`` of the raw query head
    (``q_nope[128:192]`` in lanes ``0:64``, ``q_pe`` in lanes ``64:128``), and
    ``tab`` is ``f32[2, 128]`` holding ``(C, S)`` with ``C[64:128] = [cos, cos]``
    and ``S[64:128] = [-sin, sin]``. Returns ``[heads, 128]`` in ``x``'s dtype
    with roped ``q_pe`` in lanes ``0:64`` and zeros in ``64:128``.
    """
    if fault is not None and hooks != NO_FAULT_HOOKS:
        raise ValueError("cannot pass both fault and hooks to rope_q_block")
    if isinstance(fault, FaultHooks):
        hooks = fault
    elif fault is not None:
        hooks = FaultHooks.from_names(prologue_fault=fault)
    xf = x.astype(jnp.float32)
    lane = lax.broadcasted_iota(jnp.int32, xf.shape, 1)
    # rot(x)[64:96] = x2 = x[96:128] (roll by 96 == -32);
    # rot(x)[96:128] = x1 = x[64:96] (roll by 32).
    lo, hi = hooks.q_rope_roll_shifts()
    rot = jnp.where(lane < 96, pltpu.roll(xf, lo, 1), pltpu.roll(xf, hi, 1))
    c = tab[0:1, :]
    s = tab[1:2, :]
    r = hooks.apply_q_rope_math(xf, rot, c, s, x.dtype)
    r = pltpu.roll(r, 64, 1)
    return jnp.where(lane < 64, r, 0.0).astype(x.dtype)


def load_bq(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bq_sem_idx: Any,
    batch_idx: Any,
) -> jax.Array:
    """Loads one query tile from VMEM, applying split-Q concat and Q-RoPE."""
    q = refs.bq_x2.at[bq_sem_idx, batch_idx][...]
    if cfg.q_rope:
        assert refs.bq_pe_x2 is not None and refs.bq_tab_x2 is not None
        # Rebuild caller's rope + concat + zero pad in VMEM: lanes 128:256 of
        # the raw head form one aligned vreg column.
        if cfg.body.narrow_q_rope_dma:
            raw = refs.bq_pe_x2.at[bq_sem_idx, batch_idx][...]
        else:
            raw = refs.bq_pe_x2.at[bq_sem_idx, batch_idx][:, 128:Q_ROPE_HEAD_DIM]
        tab = refs.bq_tab_x2.at[bq_sem_idx, batch_idx][...]
        q = jnp.concatenate([q, rope_q_block(raw, tab, hooks=hooks)], -1)
    elif cfg.split_q:
        assert refs.bq_pe_x2 is not None
        q = jnp.concatenate([q, refs.bq_pe_x2.at[bq_sem_idx, batch_idx][...]], -1)
        if q.shape[-1] < geom.head_dim:
            q = jnp.pad(q, ((0, 0), (0, geom.head_dim - q.shape[-1])))
    return q


def unpack_nope_transposed(
    nope_ref: Any,
    rows: int,
    kv_len: Any,
    *,
    zero_words_only: bool,
) -> jax.Array:
    """Unpacks packed ``uint32`` NOPE rows into ``float8_e4m3fn`` via 4x4 byte swap."""
    # Row r holds key r's 512 bytes as 128 words; the fp8 tile wants each word
    # to hold one byte column of four consecutive keys. xs[j] = keys 4s + j.
    quarter = rows // 4
    xs = [nope_ref[pl.ds(j, quarter, stride=4), :] for j in range(4)]
    if zero_words_only:
        key4 = lax.broadcasted_iota(jnp.int32, xs[0].shape, 0) * 4
        xs = [jnp.where(key4 < kv_len - j, x, jnp.uint32(0)) for j, x in enumerate(xs)]
    lo16, hi16 = jnp.uint32(0x0000FFFF), jnp.uint32(0xFFFF0000)
    ev8, od8 = jnp.uint32(0x00FF00FF), jnp.uint32(0xFF00FF00)
    a0 = (xs[0] & lo16) | (xs[2] << 16)
    a2 = (xs[0] >> 16) | (xs[2] & hi16)
    a1 = (xs[1] & lo16) | (xs[3] << 16)
    a3 = (xs[1] >> 16) | (xs[3] & hi16)
    cols = [
        (a0 & ev8) | ((a1 & ev8) << 8),  # byte 0 of keys 4s..4s+3
        ((a0 >> 8) & ev8) | (a1 & od8),  # byte 1
        (a2 & ev8) | ((a3 & ev8) << 8),  # byte 2
        ((a2 >> 8) & ev8) | (a3 & od8),  # byte 3
    ]
    return jnp.concatenate([pltpu.bitcast(c, jnp.float8_e4m3fn) for c in cols], axis=-1)


def unpack_bkv(
    bkv_nope_ref: Any,
    bkv_rope_ref: Any,
    get_kv_len: Callable[[], Any],
    *,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    slice_sz: int | None = None,
) -> jax.Array:
    """Unpacks NOPE/ROPE VMEM refs into a zero-masked ``[slice_sz, head_dim]`` tile."""
    rows = geom.bkv_sz if slice_sz is None else slice_sz
    if rows == geom.bkv_sz:
        bkv_nope = bkv_nope_ref[...]
    elif cfg.sparsecore_layout:
        bkv_nope = bkv_nope_ref.at[pl.ds(0, rows), :][...]
    else:
        bkv_nope = bkv_nope_ref.at[pl.ds(0, rows * kv_cache_utils.WORD_BYTES), :][...]
    if cfg.sparsecore_layout:
        # Native SC returns one uint32[128] row per token; expand packed byte
        # lanes before folding them into the logical 512-byte NOPE vector.
        bkv_nope = pltpu.bitcast(bkv_nope, jnp.uint8)
    bkv_nope = bkv_nope.reshape(rows, geom.nope_dim)
    bkv_nope = pltpu.bitcast(bkv_nope, jnp.float8_e4m3fn)

    if rows == geom.bkv_sz:
        bkv_rope = bkv_rope_ref[...]
    elif cfg.sparsecore_layout:
        bkv_rope = bkv_rope_ref.at[pl.ds(0, rows // native_sc_gather.ROPE_PACKING), :][
            ...
        ]
    else:
        bkv_rope = bkv_rope_ref.at[pl.ds(0, rows), :][...]
    if cfg.sparsecore_layout:
        # Native gather delta-swap groups four tokens per uint32[128] row.
        bkv_rope = pltpu.bitcast(bkv_rope, jnp.uint8)
        bkv_rope = bkv_rope.reshape(rows, 128)
    bkv_rope = pltpu.bitcast(bkv_rope, jnp.float8_e4m3fn)
    bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)

    if bkv.shape[-1] < geom.head_dim:
        bkv = jnp.pad(bkv, ((0, 0), (0, geom.head_dim - bkv.shape[-1])))

    # Zero keys past kv_len: scattered "-1" tail rows can decode to fp8 NaN,
    # and 0 * NaN = NaN in the PV einsum even after score masking.
    kv_len = get_kv_len()
    k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
    return jnp.where(k_span < kv_len, bkv, 0)


def load_bkv_levers(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    bkv_sem_idx: Any,
    batch_idx: Any,
    rows: int,
    body: BodyOptimizationConfig,
) -> jax.Array:
    """Unpacks a SparseCore KV tile using ``BodyOptimizationConfig`` levers."""
    del cfg
    nope_ref = refs.bkv_nope_x2.at[bkv_sem_idx, batch_idx]
    rope_ref = refs.bkv_rope_x2.at[bkv_sem_idx, batch_idx]
    rope_ref = rope_ref.at[pl.ds(0, rows * rope_ref.shape[0] // geom.bkv_sz)]
    kv_len = refs.kv_lens[geom.batch_start_seq_idx + batch_idx]
    zu = body.zero_nope_words_only
    if body.transpose_nope_unpack:
        bkv_nope = unpack_nope_transposed(nope_ref, rows, kv_len, zero_words_only=zu)
    else:
        words = nope_ref[pl.ds(0, rows)]
        if zu:
            key = lax.broadcasted_iota(jnp.int32, words.shape, 0)
            words = jnp.where(key < kv_len, words, jnp.uint32(0))
        bkv_nope = pltpu.bitcast(words, jnp.uint8).reshape(rows, geom.nope_dim)
        bkv_nope = pltpu.bitcast(bkv_nope, jnp.float8_e4m3fn)
    bkv_rope = pltpu.bitcast(rope_ref[...], jnp.uint8).reshape(rows, 128)
    bkv_rope = pltpu.bitcast(bkv_rope, jnp.float8_e4m3fn)
    bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)
    if bkv.shape[-1] < geom.head_dim:
        bkv = jnp.pad(bkv, ((0, 0), (0, geom.head_dim - bkv.shape[-1])))
    if not zu:
        k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
        bkv = jnp.where(k_span < kv_len, bkv, 0)
    return bkv


def load_bkv(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    bkv_sem_idx: Any,
    batch_idx: Any,
    rows: int | None = None,
    body: BodyOptimizationConfig = _DEFAULT_BODY,
) -> jax.Array:
    """Loads and unpacks one token's KV tile (``rows=None`` loads ``bkv_sz``)."""
    slice_sz = geom.bkv_sz if rows is None else rows
    if body.use_lever_loader:
        return load_bkv_levers(refs, cfg, geom, bkv_sem_idx, batch_idx, slice_sz, body)
    nope_ref = refs.bkv_nope_x2.at[bkv_sem_idx, batch_idx]
    rope_ref = refs.bkv_rope_x2.at[bkv_sem_idx, batch_idx]
    return unpack_bkv(
        nope_ref,
        rope_ref,
        lambda: refs.kv_lens[geom.batch_start_seq_idx + batch_idx],
        cfg=cfg,
        geom=geom,
        slice_sz=slice_sz,
    )


def normalize_pv_output(
    pv: jax.Array,
    l_sum: jax.Array,
    *,
    k_scale: float,
    q_dtype: Any,
) -> jax.Array:
    """Applies ``k_scale`` and normalizes a PV accumulator by ``l_sum``."""
    # Shared compressed latent uses k_scale ~= v_scale for the PV product.
    if k_scale != 1.0:
        pv *= k_scale
    if q_dtype == jnp.float32:
        return lax.div(pv, l_sum)
    return (pv * pl.reciprocal(l_sum, approx=True)).astype(q_dtype)


def attention_step1_qk_softmax(
    q: jax.Array,
    kv: jax.Array,
    kv_len: Any,
    *,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes QK scores, tail masking, and unnormalized softmax weights."""
    assert len(q.shape) == 2
    assert len(kv.shape) == 2
    assert q.shape[0] % geom.num_q_heads == 0
    assert q.shape[1] == geom.head_dim
    assert kv.shape[1] == geom.head_dim

    # Single-block softmax: each query attends one gathered kv block, so no
    # FlashAttention running-max/rescale state is needed here.
    s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
    s *= cfg.sm_scale
    if cfg.k_scale != 1.0:
        s *= cfg.k_scale

    # Mask scores past kv_len so zeroed "-1" keys do not inflate the denominator.
    kv_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
    s = jnp.where(kv_span < kv_len, s, jnp.finfo(jnp.float32).min)

    m = jnp.max(s, axis=1, keepdims=True)
    p = jnp.exp(s - m)
    l_sum = jnp.sum(p, axis=1, keepdims=True)
    return p, l_sum, m


def attention_step2_pv(
    p: jax.Array,
    kv: jax.Array,
    l_sum: jax.Array,
    *,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    body: BodyOptimizationConfig = _DEFAULT_BODY,
) -> jax.Array:
    """Computes the normalized PV output from softmax weights and KV tile."""
    if body.pv_nope_only and cfg.split_q:
        kv = kv[:, : geom.o_dim]
    if body.bf16_pv_weights:
        p = p.astype(geom.q_dtype)
    pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
    return normalize_pv_output(pv, l_sum, k_scale=cfg.k_scale, q_dtype=geom.q_dtype)


def run_token_pipelined(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bi_sem_idx: Any,
    bo_sem_idx: Any,
    rows: int | None,
    body: BodyOptimizationConfig,
) -> None:
    """Runs the baseline token-pipelined QK/softmax/PV loop over one batch."""
    prev_p: jax.Array | None = None
    prev_bkv: jax.Array | None = None
    prev_l_sum: jax.Array | None = None
    prev_m: jax.Array | None = None
    prev_out: jax.Array | None = None

    for batch_idx in range(cfg.batch_size):
        kv_len = refs.kv_lens[geom.batch_start_seq_idx + batch_idx]
        bkv = load_bkv(refs, cfg, geom, bi_sem_idx, batch_idx, rows, body)
        bq = load_bq(refs, cfg, geom, hooks, bi_sem_idx, batch_idx)

        if prev_out is not None and not body.drop_runahead_dep:
            # Artificial dependency limiting LLO QK runahead so MXU and VPU
            # interleave; jnp.where keeps XLA from folding it away.
            dep = prev_out
            if dep.shape != bq.shape:  # pv_nope_only: NOPE-wide output
                dep = dep[:, :1]
            bq = jnp.where(dep == jnp.inf, dep, bq)

        p, l_sum, m = attention_step1_qk_softmax(bq, bkv, kv_len, cfg=cfg, geom=geom)

        if prev_p is not None:
            assert prev_bkv is not None
            assert prev_l_sum is not None
            assert prev_m is not None
            out = attention_step2_pv(
                prev_p, prev_bkv, prev_l_sum, cfg=cfg, geom=geom, body=body
            )
            dcp_exchange.store_out(
                refs, cfg, geom, hooks, bo_sem_idx, batch_idx - 1, out
            )
            dcp_exchange.write_blse(
                refs, cfg, geom, bo_sem_idx, batch_idx - 1, prev_m, prev_l_sum
            )
            prev_out = out

        prev_p = p
        prev_bkv = bkv
        prev_l_sum = l_sum
        prev_m = m

    assert prev_p is not None and prev_bkv is not None
    assert prev_l_sum is not None and prev_m is not None
    out = attention_step2_pv(
        prev_p, prev_bkv, prev_l_sum, cfg=cfg, geom=geom, body=body
    )
    dcp_exchange.store_out(refs, cfg, geom, hooks, bo_sem_idx, cfg.batch_size - 1, out)
    dcp_exchange.write_blse(
        refs, cfg, geom, bo_sem_idx, cfg.batch_size - 1, prev_m, prev_l_sum
    )


def _finish_pair(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bo_sem_idx: Any,
    k: int,
    par: int,
    p: jax.Array,
    l_sum: jax.Array,
    m: jax.Array,
    body: BodyOptimizationConfig,
) -> None:
    """Runs PV, normalization, and stores for token pair ``k`` in slot ``par``."""
    assert refs.kv_stage is not None
    h = geom.num_q_heads
    for j in range(2):
        kv = refs.kv_stage.at[par, j, pl.ds(0, p.shape[1])][...]
        p_j = p[j * h : (j + 1) * h]
        l_j = l_sum[j * h : (j + 1) * h]
        m_j = m[j * h : (j + 1) * h]
        out = attention_step2_pv(p_j, kv, l_j, cfg=cfg, geom=geom, body=body)
        dcp_exchange.store_out(refs, cfg, geom, hooks, bo_sem_idx, 2 * k + j, out)
        dcp_exchange.write_blse(refs, cfg, geom, bo_sem_idx, 2 * k + j, m_j, l_j)


def run_paired_softmax(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bi_sem_idx: Any,
    bo_sem_idx: Any,
    rows: int | None,
    body: BodyOptimizationConfig,
) -> None:
    """Runs the PAIR strategy: stacked 2-token softmax overlapped with prior PV."""
    assert cfg.batch_size % 2 == 0
    assert refs.kv_stage is not None
    h = geom.num_q_heads
    prev: tuple[int, jax.Array, jax.Array, jax.Array] | None = None
    for k in range(cfg.batch_size // 2):
        par = k % 2
        t0, t1 = 2 * k, 2 * k + 1
        kv_len0 = refs.kv_lens[geom.batch_start_seq_idx + t0]
        kv_len1 = refs.kv_lens[geom.batch_start_seq_idx + t1]
        kv0 = load_bkv(refs, cfg, geom, bi_sem_idx, t0, rows, body)
        kv1 = load_bkv(refs, cfg, geom, bi_sem_idx, t1, rows, body)
        w = kv0.shape[0]
        refs.kv_stage.at[par, 0, pl.ds(0, w)][...] = kv0
        refs.kv_stage.at[par, 1, pl.ds(0, w)][...] = kv1
        q0 = load_bq(refs, cfg, geom, hooks, bi_sem_idx, t0)
        q1 = load_bq(refs, cfg, geom, hooks, bi_sem_idx, t1)
        s = jnp.concatenate(
            [
                jnp.einsum("nd,md->nm", q0, kv0, preferred_element_type=jnp.float32),
                jnp.einsum("nd,md->nm", q1, kv1, preferred_element_type=jnp.float32),
            ],
            axis=0,
        )
        s *= cfg.sm_scale
        if cfg.k_scale != 1.0:
            s *= cfg.k_scale
        pair_row = lax.broadcasted_iota(jnp.int32, s.shape, 0)
        pair_lim = jnp.where(pair_row < h, kv_len0, kv_len1)
        pair_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = jnp.where(pair_span < pair_lim, s, jnp.finfo(jnp.float32).min)
        m = jnp.max(s, axis=1, keepdims=True)
        p = jnp.exp(s - m)
        l_sum = jnp.sum(p, axis=1, keepdims=True)
        if prev is not None:
            _finish_pair(refs, cfg, geom, hooks, bo_sem_idx, k - 1, *prev, body)
        prev = (par, p, l_sum, m)
    assert prev is not None
    _finish_pair(
        refs, cfg, geom, hooks, bo_sem_idx, cfg.batch_size // 2 - 1, *prev, body
    )


def run_quad_softmax(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bi_sem_idx: Any,
    bo_sem_idx: Any,
    rows: int | None,
    body: BodyOptimizationConfig,
) -> None:
    """Runs the QUAD strategy: full-batch stacked softmax followed by staged PVs."""
    assert refs.kv_stage is not None
    h = geom.num_q_heads
    scores: list[jax.Array] = []
    kv_lens: list[Any] = []
    w: int | None = None
    for t in range(cfg.batch_size):
        kv_lens.append(refs.kv_lens[geom.batch_start_seq_idx + t])
        kv = load_bkv(refs, cfg, geom, bi_sem_idx, t, rows, body)
        w = kv.shape[0]
        refs.kv_stage.at[0, t, pl.ds(0, w)][...] = kv
        q = load_bq(refs, cfg, geom, hooks, bi_sem_idx, t)
        scores.append(
            jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        )
    assert w is not None
    s = jnp.concatenate(scores, axis=0)
    s *= cfg.sm_scale
    if cfg.k_scale != 1.0:
        s *= cfg.k_scale
    row = lax.broadcasted_iota(jnp.int32, s.shape, 0)
    lim = jnp.full(s.shape, kv_lens[cfg.batch_size - 1], jnp.int32)
    for t in range(cfg.batch_size - 2, -1, -1):
        lim = jnp.where(row < (t + 1) * h, kv_lens[t], lim)
    span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
    s = jnp.where(span < lim, s, jnp.finfo(jnp.float32).min)
    m = jnp.max(s, axis=1, keepdims=True)
    p = jnp.exp(s - m)
    l_sum = jnp.sum(p, axis=1, keepdims=True)
    for t in range(cfg.batch_size):
        kv = refs.kv_stage.at[0, t, pl.ds(0, w)][...]
        out = attention_step2_pv(
            p[t * h : (t + 1) * h],
            kv,
            l_sum[t * h : (t + 1) * h],
            cfg=cfg,
            geom=geom,
            body=body,
        )
        dcp_exchange.store_out(refs, cfg, geom, hooks, bo_sem_idx, t, out)
        dcp_exchange.write_blse(
            refs,
            cfg,
            geom,
            bo_sem_idx,
            t,
            m[t * h : (t + 1) * h],
            l_sum[t * h : (t + 1) * h],
        )


def _xqk_row_to_col(row: jax.Array, num_q_heads: int) -> jax.Array:
    """Transposes ``[1, heads]`` to ``[heads, 1]`` exactly via masked max."""
    h = num_q_heads
    eye = lax.broadcasted_iota(jnp.int32, (h, h), 0) == (
        lax.broadcasted_iota(jnp.int32, (h, h), 1)
    )
    return jnp.max(
        jnp.where(eye, jnp.broadcast_to(row, (h, h)), -jnp.inf),
        axis=1,
        keepdims=True,
    )


def _xqk_step1(
    q: jax.Array,
    kv: jax.Array,
    kv_len: Any,
    *,
    cfg: AttentionKernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes transposed QK scores ``[w, heads]`` and column-wise softmax."""
    st = jnp.einsum("md,nd->mn", kv, q, preferred_element_type=jnp.float32)
    st *= cfg.sm_scale
    if cfg.k_scale != 1.0:
        st *= cfg.k_scale
    xqk_key = lax.broadcasted_iota(jnp.int32, st.shape, 0)
    st = jnp.where(xqk_key < kv_len, st, jnp.finfo(jnp.float32).min)
    m = jnp.max(st, axis=0, keepdims=True)  # [1, heads]
    p = jnp.exp(st - m)
    l_sum = jnp.sum(p, axis=0, keepdims=True)  # [1, heads]
    return p, l_sum, m


def _xqk_step2(
    p: jax.Array,
    kv: jax.Array,
    l_sum: jax.Array,
    *,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    body: BodyOptimizationConfig,
) -> jax.Array:
    """Computes normalized PV output from transposed softmax weights."""
    if body.pv_nope_only and cfg.split_q:
        kv = kv[:, : geom.o_dim]
    pv = jnp.einsum("mn,md->nd", p, kv, preferred_element_type=jnp.float32)
    l_col = _xqk_row_to_col(l_sum, geom.num_q_heads)
    return normalize_pv_output(pv, l_col, k_scale=cfg.k_scale, q_dtype=geom.q_dtype)


def _xqk_write_blse(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    bo_sem_idx: Any,
    batch_idx: int,
    m: jax.Array,
    l_sum: jax.Array,
) -> None:
    """Stores transposed-QK unshifted LSE row into ``refs.blse_x2``."""
    if not cfg.return_lse:
        return
    lse_row = m + jnp.log(l_sum)  # [1, heads]
    if cfg.is_dcp_mode or cfg.lse_mask_empty:
        kv_len = refs.kv_lens[geom.batch_start_seq_idx + batch_idx]
        lse_row = jnp.where(kv_len > 0, lse_row, -jnp.inf)
    if cfg.lse_lane_dense and not cfg.send_enabled:
        assert refs.blse_x2 is not None
        refs.blse_x2.at[bo_sem_idx][pl.ds(batch_idx, 1), :] = lse_row
    else:
        dcp_exchange.store_blse(
            refs,
            cfg,
            geom,
            bo_sem_idx,
            batch_idx,
            _xqk_row_to_col(lse_row, geom.num_q_heads),
        )


def run_transposed_qk(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bi_sem_idx: Any,
    bo_sem_idx: Any,
    rows: int | None,
    body: BodyOptimizationConfig,
) -> None:
    """Runs the TRANSPOSED_QK strategy with token-pipelined softmax/PV overlap."""
    prev: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None
    for batch_idx in range(cfg.batch_size):
        kv_len = refs.kv_lens[geom.batch_start_seq_idx + batch_idx]
        bkv = load_bkv(refs, cfg, geom, bi_sem_idx, batch_idx, rows, body)
        bq = load_bq(refs, cfg, geom, hooks, bi_sem_idx, batch_idx)
        p, l_sum, m = _xqk_step1(bq, bkv, kv_len, cfg=cfg)
        if prev is not None:
            out = _xqk_step2(prev[0], prev[1], prev[2], cfg=cfg, geom=geom, body=body)
            dcp_exchange.store_out(
                refs, cfg, geom, hooks, bo_sem_idx, batch_idx - 1, out
            )
            _xqk_write_blse(
                refs, cfg, geom, bo_sem_idx, batch_idx - 1, prev[3], prev[2]
            )
        prev = (p, bkv, l_sum, m)
    assert prev is not None
    out = _xqk_step2(prev[0], prev[1], prev[2], cfg=cfg, geom=geom, body=body)
    dcp_exchange.store_out(refs, cfg, geom, hooks, bo_sem_idx, cfg.batch_size - 1, out)
    _xqk_write_blse(refs, cfg, geom, bo_sem_idx, cfg.batch_size - 1, prev[3], prev[2])


def dispatch_attention_body(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bi_sem_idx: Any,
    bo_sem_idx: Any,
    rows: int | None = None,
    body: BodyOptimizationConfig = _DEFAULT_BODY,
) -> None:
    """Dispatches the attention body to the configured execution strategy."""
    if body.strategy == BodyExecutionStrategy.PAIR:
        run_paired_softmax(refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx, rows, body)
    elif body.strategy == BodyExecutionStrategy.QUAD:
        run_quad_softmax(refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx, rows, body)
    elif body.strategy == BodyExecutionStrategy.TRANSPOSED_QK:
        run_transposed_qk(refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx, rows, body)
    else:
        run_token_pipelined(refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx, rows, body)


def _fixup_window_copies(
    refs: AttentionRefs,
    chunk_rows: int,
    local_idx: Any,
    j: Any,
    slot: Any,
) -> tuple[Any, Any]:
    """Creates async DMA descriptors for capped-overflow window ``j``."""
    fixup = refs.fixup
    assert fixup is not None
    row = j * chunk_rows + local_idx
    return (
        pltpu.make_async_copy(
            refs.cache_kv_nope.at[row],
            fixup.nope_x2.at[slot],
            fixup.sems.at[_FIXUP_NOPE_SEM, slot],
        ),
        pltpu.make_async_copy(
            refs.cache_kv_rope.at[row],
            fixup.rope_x2.at[slot],
            fixup.sems.at[_FIXUP_ROPE_SEM, slot],
        ),
    )


def _fixup_accumulate(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    q: jax.Array,
    kv: jax.Array,
    rem: Any,
    f32_min: Any,
) -> None:
    """Accumulates one capped-overflow KV window using online softmax."""
    fixup = refs.fixup
    assert fixup is not None
    s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
    s *= cfg.sm_scale
    if cfg.k_scale != 1.0:
        s *= cfg.k_scale
    kv_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
    s = jnp.where(kv_span < rem, s, f32_min)
    m_prev = fixup.m[...]
    m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
    p = jnp.exp(s - m_new)
    alpha = jnp.exp(m_prev - m_new)
    fixup.l_sum[...] = alpha * fixup.l_sum[...] + jnp.sum(p, axis=1, keepdims=True)
    fixup.acc[...] = alpha * fixup.acc[...] + jnp.einsum(
        "nm,md->nd", p, kv, preferred_element_type=jnp.float32
    )
    fixup.m[...] = m_new


def fixup_capped_overflow(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bi_sem_idx: Any,
    bo_sem_idx: Any,
) -> None:
    """Recomputes tokens with more than ``bkv_sz`` live rows (capped mode).

    The pipelined body attended only window 0. For tokens with ``kv_len >
    bkv_sz``, this pass replays windows ``0 .. cdiv(kv_len, bkv_sz) - 1`` with
    online softmax and overwrites ``bo`` and ``blse`` before they are sent.
    """
    fixup = refs.fixup
    assert fixup is not None and cfg.capped_windows is not None
    f32_min = jnp.finfo(jnp.float32).min
    chunk_rows = refs.cache_kv_nope.shape[0] // cfg.capped_windows

    def fix_token(batch_idx: Any, carry: Any) -> Any:
        seq_idx = geom.batch_start_seq_idx + batch_idx
        kv_len = refs.kv_lens[seq_idx]

        @pl.when(kv_len > geom.bkv_sz)
        def _fix() -> None:
            local_idx = seq_idx - geom.start_seq_idx
            num_windows = cdiv(kv_len, geom.bkv_sz)
            for cp in _fixup_window_copies(refs, chunk_rows, local_idx, 1, 1):
                cp.start()
            q = load_bq(refs, cfg, geom, hooks, bi_sem_idx, batch_idx)
            fixup.m[...] = jnp.full(fixup.m.shape, f32_min, jnp.float32)
            fixup.l_sum[...] = jnp.zeros(fixup.l_sum.shape, jnp.float32)
            fixup.acc[...] = jnp.zeros(fixup.acc.shape, jnp.float32)
            _fixup_accumulate(
                refs,
                cfg,
                q,
                unpack_bkv(
                    refs.bkv_nope_x2.at[bi_sem_idx, batch_idx],
                    refs.bkv_rope_x2.at[bi_sem_idx, batch_idx],
                    lambda: kv_len,
                    cfg=cfg,
                    geom=geom,
                    slice_sz=geom.bkv_sz,
                ),
                kv_len,
                f32_min,
            )

            def window(j: Any, window_carry: Any) -> Any:
                assert fixup is not None
                slot = lax.rem(j, 2)

                @pl.when(j + 1 < num_windows)
                def _prefetch_next() -> None:
                    for cp in _fixup_window_copies(
                        refs, chunk_rows, local_idx, j + 1, 1 - slot
                    ):
                        cp.start()

                for cp in _fixup_window_copies(refs, chunk_rows, local_idx, j, slot):
                    cp.wait()
                rem = kv_len - j * geom.bkv_sz
                kv = unpack_bkv(
                    fixup.nope_x2.at[slot],
                    fixup.rope_x2.at[slot],
                    lambda: rem,
                    cfg=cfg,
                    geom=geom,
                    slice_sz=geom.bkv_sz,
                )
                _fixup_accumulate(refs, cfg, q, kv, rem, f32_min)
                return window_carry

            lax.fori_loop(1, num_windows, window, 0)
            l_sum = fixup.l_sum[...]
            out = normalize_pv_output(
                fixup.acc[...], l_sum, k_scale=cfg.k_scale, q_dtype=geom.q_dtype
            )
            dcp_exchange.store_out(refs, cfg, geom, hooks, bo_sem_idx, batch_idx, out)
            if cfg.return_lse:
                lse_col = fixup.m[...] + jnp.log(l_sum)
                dcp_exchange.store_blse(refs, cfg, geom, bo_sem_idx, batch_idx, lse_col)

        return carry

    lax.fori_loop(0, cfg.batch_size, fix_token, 0)


def maybe_fixup_capped_overflow(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    geom: KernelGeometry,
    hooks: FaultHooks,
    bi_sem_idx: Any,
    bo_sem_idx: Any,
) -> None:
    """Runs capped-overflow fix-up if configured and required by this batch."""
    if cfg.capped_windows is None:
        return
    if cfg.body.skip_clean_overflow_fixup:
        # Skip overflow fix-up when no token in this step exceeds bkv_sz.
        fix_base = geom.batch_start_seq_idx
        fix_need = refs.kv_lens[fix_base]
        for b in range(1, cfg.batch_size):
            fix_need = jnp.maximum(fix_need, refs.kv_lens[fix_base + b])
        need_fix = fix_need > geom.bkv_sz
        pl.when(need_fix)(
            functools.partial(
                fixup_capped_overflow, refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx
            )
        )
    else:
        fixup_capped_overflow(refs, cfg, geom, hooks, bi_sem_idx, bo_sem_idx)

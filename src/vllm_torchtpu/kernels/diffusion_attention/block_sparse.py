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
"""Training-free on-the-fly block-sparse Pallas FlashAttention for diffusion DiTs.

Routes query blocks against key blocks using valid-token centroids, combining:
  - mandatory local band ``|qi - kj| <= 1`` (with self-block ``kj == qi`` placed
    at slot 0 so the in-kernel slot-0 reference ``exp2`` softmax anchors on the
    self-block row max without an online max-rescale loop on slots ``>= 1``),
  - mandatory prefix sinks ``kj < n_prefix`` (text/conditioning/audio prefix),
  - mandatory strided global sinks ``kj % sink_stride == 0``, and
  - top-scoring adaptive key blocks up to ``k_cap = ceil(max_density * n_b)``.

When ``prefix_tokens > 0`` and routing is sparse, prefix query blocks run exact
dense attention over all valid key blocks while video query blocks attend over
their compacted ``k_cap`` active blocks.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.diffusion_attention.block_sizes import detect_spec

LANES = 128
SUBLANES = 8
BF16 = jnp.bfloat16
F32 = jnp.float32
MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)
LOG2_E = 1.4426950408889634

# Slot-0 reference exp2 softmax constants:
# +64 log2 shift is an exact power-of-2 exponent bias in f32 that cancels in
# acc / l while giving 144 log2 units (~99.8 nats) of headroom above m0 before
# M0_CLAMP and 62 log2 units (~43.0 nats) below m0 before f32 min normal 2^-126.
M0_SHIFT = 64.0
M0_CLAMP = 80.0
M0_DENOM_FLOOR = 1.1754944e-38


def mandatory_k_cap(n_b: int, n_prefix: int, sink_stride: int) -> int:
    """Static upper bound on mandatory K blocks per Q block."""
    best = 0
    for qi in range(n_b):
        cnt = sum(
            1
            for kj in range(n_b)
            if abs(qi - kj) <= 1
            or kj < n_prefix
            or (sink_stride > 0 and kj % sink_stride == 0)
        )
        best = max(best, cnt)
    return best


def sol_k_cap(
    n_b: int,
    n_prefix: int,
    sink_stride: int,
    max_density: float,
    dense: bool,
) -> int:
    """Active K-block slots per Q block, never truncating mandatory blocks."""
    if dense:
        return n_b
    return min(
        n_b,
        max(
            math.ceil(max_density * n_b),
            mandatory_k_cap(n_b, n_prefix, sink_stride),
        ),
    )


def _route_core(
    q: jax.Array,
    k: jax.Array,
    seg: jax.Array,
    *,
    scale: float,
    block: int,
    tau: float,
    thresh_type: str,
    prefix_tokens: int,
    max_density: float,
    sink_stride: int,
    seg_k: jax.Array | None = None,
) -> dict[str, Any]:
    h, s_q, d = q.shape
    s_k = k.shape[1]
    n_q = s_q // block
    n_b = s_k // block
    n_ref = n_b - n_q
    if seg_k is None:
        seg_k = seg

    valid_q_tok = (seg == 0).reshape(n_q, block).astype(F32)
    q_counts = jnp.sum(valid_q_tok, axis=1)
    valid_q_blk = q_counts > 0.0
    q_denom = jnp.maximum(q_counts, 1.0)[None, :, None]
    q_tm = valid_q_tok[None, :, :, None]

    valid_tok = (seg_k == 0).reshape(n_b, block).astype(F32)
    block_counts = jnp.sum(valid_tok, axis=1)
    valid_blk = block_counts > 0.0
    n_valid = jnp.maximum(jnp.sum(valid_blk.astype(F32)), 1.0)
    denom = jnp.maximum(block_counts, 1.0)[None, :, None]
    tm = valid_tok[None, :, :, None]

    q_bar = (
        jnp.sum(
            q.reshape(h, n_q, block, d) * q_tm.astype(q.dtype),
            axis=2,
            dtype=F32,
        )
        / q_denom
    )
    k_bar = (
        jnp.sum(
            k.reshape(h, n_b, block, d) * tm.astype(k.dtype),
            axis=2,
            dtype=F32,
        )
        / denom
    )
    scores = jnp.einsum(
        "hqd,hkd->hqk", q_bar, k_bar, preferred_element_type=F32
    ) * float(scale)

    vm = valid_blk[None, None, :].astype(F32)
    mu = jnp.sum(scores * vm, axis=-1) / n_valid
    var = jnp.maximum(
        jnp.sum(((scores - mu[:, :, None]) ** 2) * vm, axis=-1) / n_valid,
        0.0,
    )
    std = jnp.sqrt(var + 1e-6)
    thresh = mu + tau * std
    if thresh_type == "diag":
        diag = jnp.diagonal(scores[:, :, n_ref:], axis1=1, axis2=2)
        thresh = jnp.minimum(thresh, diag)

    n_prefix = n_ref + (
        (prefix_tokens + block - 1) // block if prefix_tokens > 0 else 0
    )
    idx_q = jnp.arange(n_q, dtype=jnp.int32) + n_ref
    idx = jnp.arange(n_b, dtype=jnp.int32)
    qi_grid, kj_grid = idx_q[:, None], idx[None, :]
    is_band = jnp.abs(qi_grid - kj_grid) <= 1
    is_prefix_sink = kj_grid < n_prefix
    is_strided_sink = (
        (kj_grid % sink_stride == 0) if sink_stride > 0 else jnp.zeros_like(is_band)
    )
    both_valid = valid_q_blk[:, None] & valid_blk[None, :]
    is_pad_diag = (~valid_q_blk[:, None]) & (qi_grid == kj_grid)
    mandatory = (
        (is_band | is_prefix_sink | is_strided_sink) & both_valid
    ) | is_pad_diag
    routed = (scores >= thresh[:, :, None]) & both_valid

    is_dense = (tau <= -100.0) or (max_density >= 1.0)
    k_cap = sol_k_cap(n_b, n_prefix, sink_stride, max_density, is_dense)
    is_self_k = (qi_grid == kj_grid)[None, :, :]
    if is_dense:
        exact_mask = jnp.broadcast_to(
            both_valid[None, :, :] | is_pad_diag[None, :, :], (h, n_q, n_b)
        )
        active_k_count = jnp.sum(exact_mask.astype(jnp.int32), axis=-1)
        k_iota = jnp.broadcast_to(idx[None, None, :], exact_mask.shape)
        # Keep self-block (kj == qi) at slot 0 and sort remaining blocks ascending.
        sort_key = jnp.where(
            exact_mask,
            jnp.where(is_self_k, k_iota, n_b + k_iota),
            2 * n_b + k_iota,
        )
        compact = jnp.sort(sort_key, axis=-1) % n_b
    else:
        s_tie = jnp.tanh(scores * 0.1) * 10.0 - kj_grid.astype(F32)[None, :, :] * 1e-3
        rank_score = jnp.where(
            mandatory[None, :, :],
            1e6 + s_tie,
            jnp.where(
                routed,
                1e3 + s_tie,
                -1e6 - kj_grid.astype(F32)[None, :, :] * 1e-3,
            ),
        )
        top_vals, top_idx = jax.lax.top_k(rank_score, k_cap)
        slot_valid = top_vals > 0.0
        active_k_count = jnp.sum(slot_valid.astype(jnp.int32), axis=-1)
        top_is_self = top_idx == idx_q[None, :, None]
        sort_key = jnp.where(
            slot_valid,
            jnp.where(top_is_self, top_idx, n_b + top_idx),
            2 * n_b + top_idx,
        )
        compact = jnp.sort(sort_key, axis=-1) % n_b
        kth = top_vals[:, :, -1:]
        exact_mask = (rank_score >= kth) & (rank_score > 0.0)

    # Repeat the last valid block in inactive slots so Pallas elides duplicate K/V DMAs.
    last = jnp.take_along_axis(
        compact, jnp.maximum(active_k_count - 1, 0)[..., None], axis=-1
    )
    slot = jnp.arange(compact.shape[-1], dtype=jnp.int32)[None, None, :]
    active_k_indices = jnp.where(slot < active_k_count[..., None], compact, last)
    return {
        "active_k_indices": active_k_indices,
        "active_k_count": active_k_count,
        "exact_mask": exact_mask,
        "mandatory": jnp.broadcast_to(mandatory[None], exact_mask.shape),
        "k_cap": k_cap,
        "n_b": n_b,
        "n_q": n_q,
        "is_dense": is_dense,
    }


def sol_routing_stats(
    q: jax.Array,
    k: jax.Array,
    seg: jax.Array,
    *,
    scale: float | None = None,
    block: int = 1024,
    tau: float = 1.0,
    thresh_type: str = "diag",
    prefix_tokens: int = 0,
    max_density: float = 0.25,
    sink_stride: int = 8,
) -> dict[str, float | int]:
    """Compute realized active-block density and slot count on ``[H, S, D]``."""
    _, s0, d = q.shape
    eff_scale = float(d**-0.5) if scale is None else float(scale)
    b = min(block, -(-s0 // LANES) * LANES)
    s = -(-s0 // b) * b
    seg_i32 = seg.astype(jnp.int32)
    if s != s0:
        q = jnp.pad(q, ((0, 0), (0, s - s0), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, s - s0), (0, 0)))
        seg_i32 = jnp.concatenate(
            [seg_i32, jnp.broadcast_to(seg_i32[-1] + 1, (s - s0,))]
        )
    r = _route_core(
        q,
        k,
        seg_i32,
        scale=eff_scale,
        block=b,
        tau=tau,
        thresh_type=thresh_type,
        prefix_tokens=prefix_tokens,
        max_density=max_density,
        sink_stride=sink_stride,
    )
    em = r["exact_mask"]
    mand = em & r["mandatory"]
    return {
        "density": float(jnp.mean(em.astype(F32))),
        "mandatory_fraction": float(jnp.mean(mand.astype(F32))),
        "adaptive_fraction": float(jnp.mean((em & ~r["mandatory"]).astype(F32))),
        "k_cap": int(r["k_cap"]),
        "n_b": int(r["n_b"]),
        "block": int(b),
    }


@functools.cache
def _sparse_vmem_limit_bytes() -> int:
    try:
        return detect_spec().vmem_bytes
    except Exception:
        return 48 * 1024 * 1024


def _sol_attn_kernel_m0(
    idx_pf: Any,
    cnt_pf: Any,
    qfull_pf: Any,
    kfull_pf: Any,
    q_ref: Any,
    k_ref: Any,
    v_ref: Any,
    segq_ref: Any,
    segk_ref: Any,
    o_ref: Any,
    m_sc: Any,
    l_sc: Any,
    acc_sc: Any,
    *,
    n_b: int,
    k_cap: int,
    score_scale: float,
) -> None:
    hh, qi, slot = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    def _step0(masked: bool) -> None:
        v = v_ref[...]
        # Form (b_k, b_q) so max/sum reduce along axis=0 (vertical VREG reduce
        # with zero cross-lane shuffles), matching _flash_attention_kernel.
        qk = (
            jax.lax.dot_general(
                k_ref[...],
                q_ref[...],
                (((1,), (1,)), ((), ())),
                preferred_element_type=F32,
            )
            * score_scale
        )
        if masked:
            qk = jnp.where(segk_ref[:, :1] == segq_ref[:1, :], qk, MASK_VALUE)
        m0 = qk.max(axis=0, keepdims=True)
        mb = jnp.where(m0 > MASK_VALUE * 0.5, m0 + M0_SHIFT, 0.0)
        m_sc[...] = jnp.broadcast_to(mb, m_sc.shape)
        p = jax.lax.exp2(qk - mb)
        l_sc[...] = jnp.broadcast_to(p.sum(axis=0, keepdims=True), l_sc.shape)
        acc_sc[...] = jax.lax.dot_general(
            v,
            p.astype(v.dtype),
            (((0,), (0,)), ((), ())),
            preferred_element_type=F32,
        )

    def _step_next(masked: bool) -> None:
        v = v_ref[...]
        qk = (
            jax.lax.dot_general(
                k_ref[...],
                q_ref[...],
                (((1,), (1,)), ((), ())),
                preferred_element_type=F32,
            )
            * score_scale
        )
        if masked:
            qk = jnp.where(segk_ref[:, :1] == segq_ref[:1, :], qk, MASK_VALUE)
        p = jax.lax.exp2(jnp.minimum(qk - m_sc[:1, :], M0_CLAMP))
        l_sc[...] = l_sc[...] + p.sum(axis=0, keepdims=True)
        acc_sc[...] = acc_sc[...] + jax.lax.dot_general(
            v,
            p.astype(v.dtype),
            (((0,), (0,)), ((), ())),
            preferred_element_type=F32,
        )

    is_first = slot == 0
    active_next = (slot > 0) & (slot < cnt_pf[hh * n_b + qi])
    kj = idx_pf[(hh * n_b + qi) * k_cap + slot]
    full = (qfull_pf[qi] > 0) & (kfull_pf[kj] > 0)

    @pl.when(is_first & full)
    def _first_full() -> None:
        _step0(False)

    @pl.when(is_first & jnp.logical_not(full))
    def _first_masked() -> None:
        _step0(True)

    @pl.when(active_next & full)
    def _next_full() -> None:
        _step_next(False)

    @pl.when(active_next & jnp.logical_not(full))
    def _next_masked() -> None:
        _step_next(True)

    @pl.when(slot == pl.num_programs(2) - 1)
    def _store() -> None:
        o_ref[...] = (acc_sc[...] / jnp.maximum(l_sc[:1, :], M0_DENOM_FLOOR)).astype(
            o_ref.dtype
        )


def _build_prefix_m0_routing(
    n_pre_q: int,
    n_kb: int,
    n_ref: int,
    n_valid_kb: jax.Array,
    h: int,
) -> tuple[jax.Array, jax.Array]:
    """Build dense prefix routing with self-block ``kj = qi + n_ref`` at slot 0."""
    pre_q = (jnp.arange(n_pre_q, dtype=jnp.int32) + n_ref)[:, None]
    pre_k = jnp.arange(n_kb, dtype=jnp.int32)[None, :]
    is_valid = pre_k < n_valid_kb
    sort_key = jnp.where(
        is_valid,
        jnp.where(pre_k == pre_q, pre_k, n_kb + pre_k),
        2 * n_kb + pre_k,
    )
    compact = jnp.sort(sort_key, axis=-1) % n_kb
    last = jnp.take_along_axis(
        compact, jnp.maximum(n_valid_kb - 1, 0)[None, None], axis=-1
    )
    idx_1 = jnp.where(is_valid, compact, last)
    return (
        jnp.broadcast_to(idx_1[None, :, :], (h, n_pre_q, n_kb)),
        jnp.broadcast_to(n_valid_kb, (h, n_pre_q)),
    )


def _pad_scalar_pf(a: jax.Array) -> jax.Array:
    """Pad 1-D scalar prefetch array to a multiple of 128 int32s (512B SMEM)."""
    flat = a.reshape(-1).astype(jnp.int32)
    rem = (-flat.shape[0]) % LANES
    return jnp.pad(flat, (0, rem)) if rem else flat


def _run_m0_pallas(
    idx_pf: jax.Array,
    cnt_pf: jax.Array,
    qfull: jax.Array,
    kfull: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    seg_q_row: jax.Array,
    seg_k_col: jax.Array,
    *,
    b: int,
    score_scale: float,
    out_dtype: Any,
    interpret: bool,
    name: str,
) -> jax.Array:
    """Returns normalized output in ``[H, D, S_q]`` layout."""
    h, s_q, d = q.shape
    n_q_sl = s_q // b
    k_cap_sl = idx_pf.shape[-1]
    mspecs = [
        pl.BlockSpec(
            (None, b, d), lambda hh, qi, slot, i_r, c_r, qf_r, kf_r: (hh, qi, 0)
        ),
        pl.BlockSpec(
            (None, b, d),
            lambda hh, qi, slot, i_r, c_r, qf_r, kf_r: (
                hh,
                i_r[(hh * n_q_sl + qi) * k_cap_sl + slot],
                0,
            ),
        ),
        pl.BlockSpec(
            (None, b, d),
            lambda hh, qi, slot, i_r, c_r, qf_r, kf_r: (
                hh,
                i_r[(hh * n_q_sl + qi) * k_cap_sl + slot],
                0,
            ),
        ),
        pl.BlockSpec((SUBLANES, b), lambda hh, qi, slot, i_r, c_r, qf_r, kf_r: (0, qi)),
        pl.BlockSpec(
            (b, LANES),
            lambda hh, qi, slot, i_r, c_r, qf_r, kf_r: (
                i_r[(hh * n_q_sl + qi) * k_cap_sl + slot],
                0,
            ),
        ),
    ]
    spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(h, n_q_sl, k_cap_sl),
        in_specs=mspecs,
        out_specs=pl.BlockSpec(
            (None, d, b), lambda hh, qi, slot, i_r, c_r, qf_r, kf_r: (hh, 0, qi)
        ),
        scratch_shapes=[
            pltpu.VMEM((SUBLANES, b), F32),
            pltpu.VMEM((SUBLANES, b), F32),
            pltpu.VMEM((d, b), F32),
        ],
    )
    flops = 4 * h * s_q * (k_cap_sl * b) * d
    return pl.pallas_call(
        functools.partial(
            _sol_attn_kernel_m0,
            n_b=n_q_sl,
            k_cap=k_cap_sl,
            score_scale=score_scale,
        ),
        grid_spec=spec,
        out_shape=jax.ShapeDtypeStruct((h, d, s_q), out_dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
            vmem_limit_bytes=_sparse_vmem_limit_bytes(),
        ),
        cost_estimate=pl.CostEstimate(
            flops=flops,
            transcendentals=h * s_q * (k_cap_sl * b),
            bytes_accessed=4 * h * s_q * d * 2,
        ),
        interpret=interpret,
        name=name,
    )(
        _pad_scalar_pf(idx_pf),
        _pad_scalar_pf(cnt_pf),
        _pad_scalar_pf(qfull),
        _pad_scalar_pf(kfull),
        q,
        k,
        v,
        seg_q_row,
        seg_k_col,
    )


def sol_block_sparse_attention_hsd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    seg: jax.Array,
    *,
    scale: float | None = None,
    block: int = 1024,
    tau: float = 1.0,
    thresh_type: str = "diag",
    prefix_tokens: int = 0,
    max_density: float = 0.25,
    sink_stride: int = 8,
    out_dtype: Any = BF16,
    interpret: bool = False,
) -> jax.Array:
    """Run slot-0 reference block-sparse FlashAttention on ``[H, S, D]``."""
    h, s0, d = q.shape
    sk0 = k.shape[1]
    if d % LANES != 0:
        raise ValueError(f"head_dim must be a multiple of {LANES}, got {d}")
    eff_scale = float(d**-0.5) if scale is None else float(scale)
    score_scale = eff_scale * LOG2_E

    s128 = -(-s0 // LANES) * LANES
    b = min(block, s128)
    s = -(-s0 // b) * b
    n_q = s // b
    sk = -(-sk0 // b) * b
    n_kb = sk // b
    n_ref = n_kb - n_q

    seg_i32 = seg.astype(jnp.int32)
    if s != s0:
        q = jnp.pad(q, ((0, 0), (0, s - s0), (0, 0)))
        seg_i32 = jnp.concatenate(
            [seg_i32, jnp.broadcast_to(seg_i32[-1] + 1, (s - s0,))]
        )
    seg_kv = seg_i32
    if sk != sk0:
        k = jnp.pad(k, ((0, 0), (0, sk - sk0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, sk - sk0), (0, 0)))
        if sk != s:
            seg_kv = jnp.concatenate(
                [
                    seg_i32[:sk0],
                    jnp.broadcast_to(seg_i32[sk0 - 1] + 1, (sk - sk0,)),
                ]
            )

    r = _route_core(
        q,
        k,
        seg_i32,
        scale=eff_scale,
        block=b,
        tau=tau,
        thresh_type=thresh_type,
        prefix_tokens=prefix_tokens,
        max_density=max_density,
        sink_stride=sink_stride,
        seg_k=seg_kv,
    )
    idx_pf = r["active_k_indices"]
    cnt_pf = r["active_k_count"]
    seg_q_row = jnp.broadcast_to(seg_i32[None, :], (SUBLANES, s))
    seg_k_col = jnp.broadcast_to(seg_kv[:, None], (sk, LANES))
    qfull = jnp.all(seg_i32.reshape(n_q, b) == 0, axis=1).astype(jnp.int32)
    kfull = jnp.all(seg_kv.reshape(n_kb, b) == 0, axis=1).astype(jnp.int32)

    is_sparse_with_prefix = prefix_tokens > 0 and not r["is_dense"]
    n_pre_q = min(n_q, (prefix_tokens + b - 1) // b) if is_sparse_with_prefix else 0
    if 0 < n_pre_q < n_q:
        valid_kb = jnp.any(seg_kv.reshape(n_kb, b) == 0, axis=1)
        n_valid_kb = jnp.maximum(jnp.sum(valid_kb.astype(jnp.int32)), 1)
        idx_pre, cnt_pre = _build_prefix_m0_routing(n_pre_q, n_kb, n_ref, n_valid_kb, h)
        p_rows = n_pre_q * b
        out_pre = _run_m0_pallas(
            idx_pre,
            cnt_pre,
            qfull[:n_pre_q],
            kfull,
            q[:, :p_rows],
            k,
            v,
            seg_q_row[:, :p_rows],
            seg_k_col,
            b=b,
            score_scale=score_scale,
            out_dtype=out_dtype,
            interpret=interpret,
            name="diffusion_block_sparse_attention_m0_prefix",
        )
        out_vid = _run_m0_pallas(
            idx_pf[:, n_pre_q:, :],
            cnt_pf[:, n_pre_q:],
            qfull[n_pre_q:],
            kfull,
            q[:, p_rows:],
            k,
            v,
            seg_q_row[:, p_rows:],
            seg_k_col,
            b=b,
            score_scale=score_scale,
            out_dtype=out_dtype,
            interpret=interpret,
            name="diffusion_block_sparse_attention_m0",
        )
        out_hds = jnp.concatenate([out_pre, out_vid], axis=2)
    elif n_pre_q == n_q and is_sparse_with_prefix:
        valid_kb = jnp.any(seg_kv.reshape(n_kb, b) == 0, axis=1)
        n_valid_kb = jnp.maximum(jnp.sum(valid_kb.astype(jnp.int32)), 1)
        idx_pre, cnt_pre = _build_prefix_m0_routing(n_q, n_kb, n_ref, n_valid_kb, h)
        out_hds = _run_m0_pallas(
            idx_pre,
            cnt_pre,
            qfull,
            kfull,
            q,
            k,
            v,
            seg_q_row,
            seg_k_col,
            b=b,
            score_scale=score_scale,
            out_dtype=out_dtype,
            interpret=interpret,
            name="diffusion_block_sparse_attention_m0_prefix",
        )
    else:
        out_hds = _run_m0_pallas(
            idx_pf,
            cnt_pf,
            qfull,
            kfull,
            q,
            k,
            v,
            seg_q_row,
            seg_k_col,
            b=b,
            score_scale=score_scale,
            out_dtype=out_dtype,
            interpret=interpret,
            name="diffusion_block_sparse_attention_m0",
        )
    return jnp.swapaxes(out_hds[:, :, :s0], 1, 2)


def block_sparse_attention_bshd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    scale: float,
    valid_len: int,
    prefix_tokens: int = 0,
    block: int = 1024,
    max_density: float = 0.25,
    tau: float = 1.0,
    sink_stride: int = 8,
    interpret: bool = False,
) -> jax.Array:
    """``[B, S, H, D]`` wrapper around :func:`sol_block_sparse_attention_hsd`."""
    bsz, seq_len, num_heads, _ = q.shape
    if k.shape[2] != num_heads:
        if num_heads % k.shape[2] != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({k.shape[2]})"
            )
        rep = num_heads // k.shape[2]
        k = jnp.repeat(k, rep, axis=2)
        v = jnp.repeat(v, rep, axis=2)

    eff_valid = max(1, min(int(valid_len), seq_len))
    seg = (jnp.arange(seq_len, dtype=jnp.int32) >= eff_valid).astype(jnp.int32)

    def _run_one(q1: jax.Array, k1: jax.Array, v1: jax.Array) -> jax.Array:
        out_hsd = sol_block_sparse_attention_hsd(
            jnp.swapaxes(q1, 0, 1),
            jnp.swapaxes(k1, 0, 1),
            jnp.swapaxes(v1, 0, 1),
            seg,
            scale=scale,
            block=block,
            tau=tau,
            prefix_tokens=prefix_tokens,
            max_density=max_density,
            sink_stride=sink_stride,
            out_dtype=q.dtype,
            interpret=interpret,
        )
        return jnp.swapaxes(out_hsd, 0, 1)

    if bsz == 1:
        return _run_one(q[0], k[0], v[0])[None, ...]
    return jax.vmap(_run_one, in_axes=(0, 0, 0))(q, k, v)

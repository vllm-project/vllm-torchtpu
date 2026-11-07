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
"""Self-contained float64 reference oracle for the Qwen3.8 megakernel."""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vllm_torchtpu.models.qwen38_2_4t.megakernel.shapes import Qwen38Config

_SOFTPLUS_CUTOFF: float = 30.0
_RANK_POOL_NDIM: int = 6
_POOL_7D_NDIM: int = 7
_PAGES_PER_MAMBA_SLOT: int = 5  # 4 recurrent-state pages + 1 conv1d page


class SyntheticProblem(TypedDict):
    """Typed synthetic problem dictionary for Qwen3.8 parity checks."""

    x: jax.Array
    pos: jax.Array
    page_table: jax.Array
    pool: jax.Array
    conv: jax.Array
    rec: jax.Array
    mamba_slots: np.ndarray
    weights: dict[str, jax.Array]
    forced_top_ids: jax.Array


class ReferenceOutputs(TypedDict):
    """Typed output dictionary from the FP64 reference simulation."""

    x: np.ndarray
    pool: jax.Array
    conv: np.ndarray
    rec: np.ndarray


def _bf16_f64(x: np.ndarray) -> np.ndarray:
    """Cast array through bfloat16 rounding back to float64."""
    x32 = np.asarray(x, dtype=np.float32)
    u32 = x32.view(np.uint32)
    lsb = (u32 >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    u_rounded = (u32 + rounding_bias) & np.uint32(0xFFFF0000)
    u_out = np.where(np.isnan(x32), np.uint32(0x7FC00000), u_rounded)
    return u_out.view(np.float32).astype(np.float64)


def _fp8_round_f64(x: np.ndarray) -> np.ndarray:
    """Cast array through float8_e4m3fn rounding back to float64."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return t.to(torch.float8_e4m3fn).to(torch.float64).numpy()


def _sig_f64(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable float64 sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def _silu_f64(x: np.ndarray) -> np.ndarray:
    """Compute float64 SiLU activation."""
    return x * _sig_f64(x)


def _softplus_f64(x: np.ndarray) -> np.ndarray:
    """Compute float64 softplus with cutoff."""
    return np.where(
        x > _SOFTPLUS_CUTOFF, x, np.log1p(np.exp(np.clip(x, -80.0, _SOFTPLUS_CUTOFF)))
    )


def _rmsnorm_f64(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute float64 RMSNorm with bfloat16 intermediate rounding."""
    xf = np.asarray(x, dtype=np.float64)
    wf = np.asarray(w, dtype=np.float64)
    inv = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return _bf16_f64(xf * inv) * _bf16_f64(wf)


def _rope_f64(
    x: np.ndarray,
    pos: np.ndarray,
    rope_dim: int,
    theta: float = 10_000_000.0,
) -> np.ndarray:
    """Apply float64 partial rotary position embedding."""
    out = np.array(x, dtype=np.float64, copy=True)
    half = rope_dim // 2
    idx = np.arange(half, dtype=np.float64)
    inv_freq = 1.0 / (theta ** (2.0 * idx / float(rope_dim)))
    ang = pos[:, None, None].astype(np.float64) * inv_freq[None, None, :]
    cos = np.cos(ang)
    sin = np.sin(ang)
    x0 = out[:, :, :half].copy()
    x1 = out[:, :, half:rope_dim].copy()
    out[:, :, :half] = x0 * cos - x1 * sin
    out[:, :, half:rope_dim] = x1 * cos + x0 * sin
    return out


def _encode_fp4_from_values(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize floating point array into NVFP4 nibbles and scale."""
    midpoints = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
    shape = vals.shape
    k_dim = shape[-2]
    n_dim = shape[-1]
    if k_dim % 16 != 0 or n_dim % 2 != 0:
        msg = f"Expected k_dim={k_dim} % 16 == 0 and n_dim={n_dim} % 2 == 0"
        raise ValueError(msg)
    vg = vals.reshape(*shape[:-2], k_dim // 16, 16, n_dim)
    amax = np.max(np.abs(vg), axis=-2)
    sc = np.maximum(amax / 6.0, 1e-4).astype(np.float32)
    sc_q = (
        torch.from_numpy(np.ascontiguousarray(sc, dtype=np.float32))
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        .numpy()
    )
    sc_safe = np.where(sc_q == 0.0, 1e-4, sc_q)
    normed = vg / sc_safe[..., None, :]
    sign = (normed < 0).astype(np.uint8)
    mag = np.abs(normed)
    codes = np.searchsorted(midpoints, mag, side="left").astype(np.uint8)
    nib = ((sign << 3) | codes).reshape(*shape[:-2], k_dim, n_dim)
    lo = nib[..., 0::2] & 0xF
    hi = (nib[..., 1::2] & 0xF) << 4
    packed = (lo | hi).astype(np.uint8)
    return packed, sc_q


def _decode_fp4_f64(packed_u8: np.ndarray, scale_f32: np.ndarray) -> np.ndarray:
    """Dequantize packed NVFP4 bytes and scale into float64."""
    lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
    lo = packed_u8 & 0x0F
    hi = (packed_u8 >> 4) & 0x0F
    nib = np.empty((*packed_u8.shape[:-1], packed_u8.shape[-1] * 2), dtype=np.uint8)
    nib[..., 0::2] = lo
    nib[..., 1::2] = hi
    sign = np.where((nib & 0x08) != 0, -1.0, 1.0)
    mag = lut[nib & 0x07]
    raw = sign * mag
    sc = np.repeat(np.asarray(scale_f32, dtype=np.float64), 16, axis=-2)
    return _bf16_f64(raw * sc)


def max_fp8_ulp_diff(
    got_f8: jax.Array | np.ndarray,
    ref_f8_or_f64: jax.Array | np.ndarray,
) -> int:
    """Compute maximum unit in the last place difference between FP8 tensors."""
    got_vals = np.asarray(jax.device_get(got_f8).astype(jnp.float32), dtype=np.float64)
    ref_arr = jax.device_get(ref_f8_or_f64)
    ref_quant = np.asarray(
        jnp.clip(jnp.asarray(ref_arr, dtype=jnp.float32), -448.0, 448.0)
        .astype(jnp.float8_e4m3fn)
        .astype(jnp.float32),
        dtype=np.float64,
    )
    if np.array_equal(got_vals, ref_quant):
        return 0
    rms_floor = max(float(np.sqrt(np.mean(ref_quant**2))), 2.0**-6)
    scale_mag = np.maximum(np.abs(ref_quant), rms_floor)
    ulp_step = np.ldexp(1.0, np.floor(np.log2(scale_mag)).astype(np.int32) - 3)
    return int(np.max(np.rint(np.abs(got_vals - ref_quant) / ulp_step)))


def layer_bf16_error_bound(
    rms_out: float, n_subblocks: int = 2, n_layers: int = 4
) -> float:
    """Compute analytical bfloat16 roundoff error bound for layer."""
    eps_bf16 = 2.0**-8
    return max(
        4.0 * math.sqrt(float(n_subblocks * n_layers)) * eps_bf16 * rms_out, 1e-4
    )


def _make_synthetic_weights(
    rng: np.random.Generator,
    cfg: Qwen38Config,
) -> dict[str, jax.Array]:
    """Sample random synthetic model weights for test scenarios."""
    r_dim = cfg.R
    l_dim = cfg.L
    h_dim = cfg.H
    ngdn = cfg.NGDN
    natt = cfg.NATT

    def rbf16(shape: tuple[int, ...], scale: float = 0.08) -> jnp.ndarray:
        """Sample random normal bfloat16 test array."""
        arr = rng.normal(0.0, scale, size=shape).astype(np.float32)
        return jnp.asarray(arr, dtype=jnp.bfloat16)

    def rfp4(
        shape: tuple[int, ...], scale: float = 0.08
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample random normal NVFP4 packed and scaled weight pair."""
        vals = rng.normal(0.0, scale, size=shape).astype(np.float32)
        pk, sc = _encode_fp4_from_values(vals)
        return (
            jnp.asarray(pk, dtype=jnp.uint8),
            jnp.asarray(sc, dtype=jnp.float8_e4m3fn).astype(jnp.bfloat16),
        )

    wqkv_q, wqkv_s = rfp4((r_dim, natt, h_dim, cfg.ATT_QKV_R), 0.06)
    wo_q, wo_s = rfp4((r_dim, natt, cfg.Q_R * cfg.D, h_dim), 0.06)
    win_q, win_s = rfp4((r_dim, ngdn, h_dim, cfg.QKV + cfg.Z_R), 0.06)
    wout_q, wout_s = rfp4((r_dim, ngdn, cfg.Z_R, h_dim), 0.06)
    wse13_q, wse13_s = rfp4((r_dim, l_dim, h_dim, 2 * cfg.SE_R), 0.06)
    wse2_q, wse2_s = rfp4((r_dim, l_dim, cfg.SE_R, h_dim), 0.06)
    we13_q, we13_s = rfp4((r_dim, l_dim, cfg.E, h_dim, 2 * cfg.I_R), 0.06)
    we2_q, we2_s = rfp4((r_dim, l_dim, cfg.E, cfg.I_R, h_dim), 0.06)

    in_norm_1 = jnp.asarray(
        1.0 + rng.normal(0.0, 0.05, size=(l_dim, h_dim)), dtype=jnp.bfloat16
    )
    post_norm_1 = jnp.asarray(
        1.0 + rng.normal(0.0, 0.05, size=(l_dim, h_dim)), dtype=jnp.bfloat16
    )
    final_norm_1 = jnp.asarray(
        1.0 + rng.normal(0.0, 0.05, size=(h_dim,)), dtype=jnp.bfloat16
    )
    q_norm_1 = jnp.asarray(
        1.0 + rng.normal(0.0, 0.05, size=(natt, cfg.D)), dtype=jnp.bfloat16
    )
    k_norm_1 = jnp.asarray(
        1.0 + rng.normal(0.0, 0.05, size=(natt, cfg.D)), dtype=jnp.bfloat16
    )
    gdn_norm_1 = jnp.asarray(
        1.0 + rng.normal(0.0, 0.05, size=(ngdn, cfg.VD)), dtype=jnp.bfloat16
    )

    w_gate_1 = rbf16((l_dim, h_dim, cfg.E), 0.08)
    w_se_gate_1 = rbf16((l_dim, h_dim, 1), 0.08)

    return {
        "in_norm": jnp.broadcast_to(in_norm_1[None, ...], (r_dim, l_dim, h_dim)).copy(),
        "post_norm": jnp.broadcast_to(
            post_norm_1[None, ...], (r_dim, l_dim, h_dim)
        ).copy(),
        "final_norm": jnp.broadcast_to(final_norm_1[None, ...], (r_dim, h_dim)).copy(),
        "wqkv": wqkv_q,
        "wqkv_s": wqkv_s,
        "wo": wo_q,
        "wo_s": wo_s,
        "q_norm": jnp.broadcast_to(q_norm_1[None, ...], (r_dim, natt, cfg.D)).copy(),
        "k_norm": jnp.broadcast_to(k_norm_1[None, ...], (r_dim, natt, cfg.D)).copy(),
        "w_inproj": win_q,
        "w_inproj_s": win_s,
        "w_ba": rbf16((r_dim, ngdn, h_dim, 2 * cfg.VH_R), 0.05),
        "w_conv": rbf16((r_dim, ngdn, cfg.conv_kernel, cfg.QKV), 0.08),
        "a_log": jnp.asarray(
            rng.uniform(-2.0, -0.5, size=(r_dim, ngdn, cfg.VH_R)), dtype=jnp.float32
        ),
        "dt_bias": jnp.asarray(
            rng.uniform(-0.2, 0.2, size=(r_dim, ngdn, cfg.VH_R)), dtype=jnp.float32
        ),
        "gdn_norm": jnp.broadcast_to(
            gdn_norm_1[None, ...], (r_dim, ngdn, cfg.VD)
        ).copy(),
        "w_out": wout_q,
        "w_out_s": wout_s,
        "w_gate": jnp.broadcast_to(
            w_gate_1[None, ...], (r_dim, l_dim, h_dim, cfg.E)
        ).copy(),
        "w_se13": wse13_q,
        "w_se13_s": wse13_s,
        "w_se2": wse2_q,
        "w_se2_s": wse2_s,
        "w_se_gate": jnp.broadcast_to(
            w_se_gate_1[None, ...], (r_dim, l_dim, h_dim, 1)
        ).copy(),
        "w_e13": we13_q,
        "w_e13_s": we13_s,
        "w_e2": we2_q,
        "w_e2_s": we2_s,
    }


def _build_page_table(r_dim: int, natt: int, b_dim: int, max_pages: int) -> np.ndarray:
    """Build mock sequential page table for synthetic problems."""
    pt_np = np.zeros((r_dim, natt, b_dim, max_pages), dtype=np.int32)
    for r in range(r_dim):
        for a in range(natt):
            for b in range(b_dim):
                for lp in range(max_pages):
                    pt_np[r, a, b, lp] = (a * b_dim + b) * max_pages + lp
    return pt_np


def _build_forced_ids(
    cfg: Qwen38Config, forced_top_ids: jax.Array | None
) -> jnp.ndarray:
    """Build or broadcast forced expert top-k indices."""
    r_dim = cfg.R
    b_dim = cfg.B
    l_dim = cfg.L
    if forced_top_ids is None:
        forced_np = np.zeros((l_dim, b_dim, cfg.K), dtype=np.int32)
        for l_idx in range(l_dim):
            for b_idx in range(b_dim):
                forced_np[l_idx, b_idx] = (
                    np.arange(cfg.K, dtype=np.int32) + l_idx * 2 + b_idx * 3
                ) % cfg.E
    else:
        forced_np = np.asarray(forced_top_ids, dtype=np.int32)
    return jnp.broadcast_to(
        jnp.asarray(forced_np)[None, ...], (r_dim, l_dim, b_dim, cfg.K)
    ).copy()


def make_synthetic_problem(
    cfg: Qwen38Config,
    *,
    seed: int = 42,
    pos: tuple[int, ...] = (0, 130, 520, 1500),
    forced_top_ids: jax.Array | None = None,
) -> SyntheticProblem:
    """Construct synthetic problem dictionaries for megakernel parity tests."""
    rng = np.random.default_rng(seed)
    r_dim = cfg.R
    b_dim = cfg.B
    h_dim = cfg.H
    ngdn = cfg.NGDN
    natt = cfg.NATT

    def rbf16(shape: tuple[int, ...], scale: float = 0.08) -> jnp.ndarray:
        """Sample random normal bfloat16 test array."""
        arr = rng.normal(0.0, scale, size=shape).astype(np.float32)
        return jnp.asarray(arr, dtype=jnp.bfloat16)

    def rf32(shape: tuple[int, ...], scale: float = 0.08) -> jnp.ndarray:
        """Sample random normal float32 test array."""
        arr = rng.normal(0.0, scale, size=shape).astype(np.float32)
        return jnp.asarray(arr, dtype=jnp.float32)

    x0 = rbf16((b_dim, h_dim), 0.25)
    x_in = jnp.broadcast_to(x0[None, ...], (r_dim, b_dim, h_dim)).copy()
    pos_1d = np.asarray(pos, dtype=np.int32)
    if pos_1d.shape != (b_dim,):
        msg = f"Expected pos shape {(b_dim,)}, got {pos_1d.shape}"
        raise ValueError(msg)
    pos_in = jnp.broadcast_to(jnp.asarray(pos_1d)[None, :], (r_dim, b_dim)).copy()

    pt_np = _build_page_table(r_dim, natt, b_dim, cfg.max_pages_per_seq)

    pool_raw = rng.normal(
        0.0, 0.15, size=(r_dim, cfg.num_pages, 2, cfg.K_R, cfg.D, cfg.page_size)
    ).astype(np.float32)
    pool_fp8 = jnp.asarray(pool_raw, dtype=jnp.float8_e4m3fn)

    conv_init = rbf16((r_dim, ngdn, b_dim, cfg.conv_kernel - 1, cfg.QKV), 0.1)
    rec_init = rf32((r_dim, ngdn, b_dim, cfg.VH_R, cfg.KD, cfg.VD), 0.05)

    mamba_slots = np.zeros((ngdn, b_dim), dtype=np.int32)
    g_per_reg = max(1, cfg.att_every - 1)
    for g in range(ngdn):
        for b in range(b_dim):
            mamba_slots[g, b] = 4 + (g % g_per_reg) * b_dim + b

    weights = _make_synthetic_weights(rng, cfg)
    forced_jnp = _build_forced_ids(cfg, forced_top_ids)

    return {
        "x": x_in,
        "pos": pos_in,
        "page_table": jnp.asarray(pt_np, dtype=jnp.int32),
        "pool": pool_fp8,
        "conv": conv_init,
        "rec": rec_init,
        "mamba_slots": mamba_slots,
        "weights": weights,
        "forced_top_ids": forced_jnp,
    }


@dataclasses.dataclass
class AttnRefState:
    """State and cache buffers for attention reference simulation."""

    pos: np.ndarray
    pt: np.ndarray
    pool: np.ndarray


@dataclasses.dataclass
class GdnRefState:
    """State buffers for GDN reference simulation."""

    conv: np.ndarray
    rec: np.ndarray


def _project_and_write_kv(
    xn: np.ndarray,
    state: AttnRefState,
    w: dict[str, np.ndarray],
    att_idx: int,
    cfg: Qwen38Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Project inputs and update KV cache pool for one attention layer."""
    r_dim, dcp, b_dim = cfg.R, cfg.DCP, cfg.B
    d_dim, q_r, k_r = cfg.D, cfg.Q_R, cfg.K_R

    q_rot_all = np.zeros((r_dim, b_dim, q_r, d_dim), dtype=np.float64)
    q_gate_all = np.zeros((r_dim, b_dim, q_r, d_dim), dtype=np.float64)

    for r in range(r_dim):
        w_qkv = _decode_fp4_f64(w["wqkv"][r, att_idx], w["wqkv_s"][r, att_idx])
        qkv = xn @ w_qkv
        qg = qkv[:, : 2 * q_r * d_dim].reshape(b_dim, q_r, 2 * d_dim)
        q = qg[:, :, :d_dim]
        q_gate_all[r] = qg[:, :, d_dim:]
        k_new = qkv[:, 2 * q_r * d_dim : (2 * q_r + k_r) * d_dim].reshape(
            b_dim, k_r, d_dim
        )
        v_new = qkv[:, (2 * q_r + k_r) * d_dim :].reshape(b_dim, k_r, d_dim)

        qn = _rmsnorm_f64(q, w["q_norm"][r, att_idx])
        kn = _rmsnorm_f64(k_new, w["k_norm"][r, att_idx])

        q_rot_all[r] = _rope_f64(qn, state.pos, cfg.rope_dim)
        k_rot = _rope_f64(kn, state.pos, cfg.rope_dim)

        k_fp8 = _fp8_round_f64(k_rot)
        v_fp8 = _fp8_round_f64(v_new)

        dcp_rank = r % dcp
        for b in range(b_dim):
            pb = int(state.pos[b])
            gp = pb // cfg.page_size
            if gp % dcp == dcp_rank:
                lp = gp // dcp
                off = pb % cfg.page_size
                p_idx = int(state.pt[r, att_idx, b, lp])
                state.pool[r, att_idx, p_idx, 0, :, :, off] = k_fp8[b]
                state.pool[r, att_idx, p_idx, 1, :, :, off] = v_fp8[b]

    return q_rot_all, q_gate_all


def _compute_dcp_group_attention(
    q_grp: np.ndarray,
    r_base: int,
    state: AttnRefState,
    att_idx: int,
    cfg: Qwen38Config,
) -> np.ndarray:
    """Compute DCP multi-rank attention scores and context for one group."""
    dcp, b_dim, d_dim = cfg.DCP, cfg.B, cfg.D
    q_r, gqa = cfg.Q_R, cfg.gqa_ratio
    sm_scale = 1.0 / math.sqrt(float(d_dim))

    m_all = np.full((dcp, b_dim, dcp * q_r), -1e30, dtype=np.float64)
    l_all = np.zeros((dcp, b_dim, dcp * q_r), dtype=np.float64)
    acc_all = np.zeros((dcp, b_dim, dcp * q_r, d_dim), dtype=np.float64)

    for d_idx in range(dcp):
        r = r_base + d_idx
        for b in range(b_dim):
            pb = int(state.pos[b])
            m_b = np.full((dcp * q_r,), -1e30, dtype=np.float64)
            l_b = np.zeros((dcp * q_r,), dtype=np.float64)
            acc_b = np.zeros((dcp * q_r, d_dim), dtype=np.float64)

            for lp in range(cfg.max_pages_per_seq):
                p_idx = int(state.pt[r, att_idx, b, lp])
                k_pg = state.pool[r, att_idx, p_idx, 0]
                v_pg = state.pool[r, att_idx, p_idx, 1]

                k_rep = np.repeat(k_pg, dcp * gqa, axis=0)
                v_rep = np.repeat(v_pg, dcp * gqa, axis=0)

                scores = np.einsum("hd,hdp->hp", q_grp[b], k_rep) * sm_scale
                tok_pos = (
                    (lp * dcp + d_idx) * cfg.page_size
                    + np.arange(cfg.page_size, dtype=np.int32)
                )[None, :]
                valid = (tok_pos <= pb) & (tok_pos < cfg.max_seq_len)
                scores = np.where(valid, scores, -1e30)

                m_blk = np.max(scores, axis=-1)
                m_new = np.maximum(m_b, m_blk)
                alpha = np.exp(m_b - m_new)
                p = np.exp(scores - m_new[:, None])
                l_b = l_b * alpha + np.sum(p, axis=-1)
                acc_b = acc_b * alpha[:, None] + np.einsum("hp,hdp->hd", p, v_rep)
                m_b = m_new

            m_all[d_idx, b] = m_b
            l_all[d_idx, b] = l_b
            acc_all[d_idx, b] = acc_b

    m_glob = np.max(m_all, axis=0)
    num_glob = np.zeros((b_dim, dcp * q_r, d_dim), dtype=np.float64)
    den_glob = np.zeros((b_dim, dcp * q_r), dtype=np.float64)
    for d_idx in range(dcp):
        w_d = np.exp(m_all[d_idx] - m_glob)
        den_glob += l_all[d_idx] * w_d
        num_glob += acc_all[d_idx] * w_d[:, :, None]

    return num_glob / np.maximum(den_glob[:, :, None], 1e-30)


def _ref_attention_step(
    xn: np.ndarray,
    state: AttnRefState,
    w: dict[str, np.ndarray],
    att_idx: int,
    cfg: Qwen38Config,
) -> np.ndarray:
    """Execute float64 attention reference pass for one attention layer."""
    r_dim, dcp, b_dim = cfg.R, cfg.DCP, cfg.B
    h_dim, d_dim, q_r = cfg.H, cfg.D, cfg.Q_R

    q_rot_all, q_gate_all = _project_and_write_kv(xn, state, w, att_idx, cfg)
    part_attn = np.zeros((r_dim, b_dim, h_dim), dtype=np.float64)

    for grp in range(r_dim // dcp):
        r_base = grp * dcp
        q_grp = np.zeros((b_dim, dcp * q_r, d_dim), dtype=np.float64)
        for d_idx in range(dcp):
            q_grp[:, d_idx * q_r : (d_idx + 1) * q_r, :] = _bf16_f64(
                q_rot_all[r_base + d_idx]
            )

        attn_grp = _compute_dcp_group_attention(q_grp, r_base, state, att_idx, cfg)
        for d_idx in range(dcp):
            r = r_base + d_idx
            o_slice = _bf16_f64(attn_grp[:, d_idx * q_r : (d_idx + 1) * q_r, :])
            gated = (o_slice * _sig_f64(q_gate_all[r])).reshape(b_dim, q_r * d_dim)
            w_o = _decode_fp4_f64(w["wo"][r, att_idx], w["wo_s"][r, att_idx])
            part_attn[r] = _bf16_f64(gated) @ w_o

    return np.sum(_bf16_f64(part_attn), axis=0)


def _ref_gdn_step(
    xn: np.ndarray,
    state: GdnRefState,
    w: dict[str, np.ndarray],
    g_idx: int,
    cfg: Qwen38Config,
) -> np.ndarray:
    """Execute float64 GDN reference pass for one recurrent layer."""
    r_dim = cfg.R
    b_dim = cfg.B
    h_dim = cfg.H
    part_gdn = np.zeros((r_dim, b_dim, h_dim), dtype=np.float64)

    for r in range(r_dim):
        w_in = _decode_fp4_f64(w["w_inproj"][r, g_idx], w["w_inproj_s"][r, g_idx])
        qz = xn @ w_in
        qkv = qz[:, : cfg.QKV]
        z = qz[:, cfg.QKV :].reshape(b_dim, cfg.VH_R, cfg.VD)
        ba = xn @ _bf16_f64(w["w_ba"][r, g_idx])
        b_raw = ba[:, : cfg.VH_R]
        a_raw = ba[:, cfg.VH_R :]

        c_hist = state.conv[r, g_idx]
        c_win = np.concatenate([c_hist, _bf16_f64(qkv)[:, None, :]], axis=1)
        state.conv[r, g_idx] = _bf16_f64(c_win[:, 1:, :])

        wc = _bf16_f64(w["w_conv"][r, g_idx])
        conv_out = _silu_f64(np.sum(c_win * wc[None, :, :], axis=1))

        q_c = conv_out[:, : cfg.KH_R * cfg.KD].reshape(b_dim, cfg.KH_R, cfg.KD)
        k_c = conv_out[:, cfg.KH_R * cfg.KD : 2 * cfg.KH_R * cfg.KD].reshape(
            b_dim, cfg.KH_R, cfg.KD
        )
        v_c = conv_out[:, 2 * cfg.KH_R * cfg.KD :].reshape(b_dim, cfg.VH_R, cfg.VD)

        q_inv = 1.0 / np.sqrt(np.sum(q_c * q_c, axis=-1, keepdims=True) + 1e-6)
        k_inv = 1.0 / np.sqrt(np.sum(k_c * k_c, axis=-1, keepdims=True) + 1e-6)
        q_normed = (q_c * q_inv) * (1.0 / math.sqrt(float(cfg.KD)))
        k_normed = k_c * k_inv

        q_rep = np.repeat(q_normed, cfg.gdn_v_per_k, axis=1)
        k_rep = np.repeat(k_normed, cfg.gdn_v_per_k, axis=1)

        beta = _sig_f64(b_raw)
        dt = _softplus_f64(a_raw + w["dt_bias"][r, g_idx][None, :])
        g_decay = np.exp(-np.exp(w["a_log"][r, g_idx][None, :]) * dt)

        s_acc = state.rec[r, g_idx] * g_decay[:, :, None, None]
        kv_mem = np.einsum("bhkv,bhk->bhv", s_acc, k_rep)
        delta = (v_c - kv_mem) * beta[:, :, None]
        s_acc = s_acc + np.einsum("bhk,bhv->bhkv", k_rep, delta)
        state.rec[r, g_idx] = s_acc

        y = np.einsum("bhkv,bhk->bhv", s_acc, q_rep)
        yn = _rmsnorm_f64(y, w["gdn_norm"][r, g_idx])
        yg = (yn * _silu_f64(z)).reshape(b_dim, cfg.Z_R)

        w_out = _decode_fp4_f64(w["w_out"][r, g_idx], w["w_out_s"][r, g_idx])
        part_gdn[r] = _bf16_f64(yg) @ w_out

    return np.sum(_bf16_f64(part_gdn), axis=0)


def _ref_moe_step(
    x: np.ndarray,
    w: dict[str, np.ndarray],
    l_idx: int,
    forced: np.ndarray | None,
    cfg: Qwen38Config,
) -> np.ndarray:
    """Execute float64 MoE reference pass for one layer."""
    r_dim = cfg.R
    b_dim = cfg.B
    h_dim = cfg.H

    xm = _rmsnorm_f64(x, w["post_norm"][0, l_idx])
    router_logits = xm @ _bf16_f64(w["w_gate"][0, l_idx])
    router_probs = np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True))
    router_probs = router_probs / np.sum(router_probs, axis=-1, keepdims=True)

    if forced is not None:
        top_ids = forced[l_idx]
        top_p = np.take_along_axis(router_probs, top_ids, axis=-1)
    else:
        top_ids = np.argsort(-router_probs, axis=-1)[:, : cfg.K]
        top_p = np.take_along_axis(router_probs, top_ids, axis=-1)

    top_w = top_p / np.maximum(np.sum(top_p, axis=-1, keepdims=True), 1e-12)
    se_gate = _sig_f64(xm @ _bf16_f64(w["w_se_gate"][0, l_idx]))

    part_moe = np.zeros((r_dim, b_dim, h_dim), dtype=np.float64)
    for r in range(r_dim):
        w_se13 = _decode_fp4_f64(w["w_se13"][r, l_idx], w["w_se13_s"][r, l_idx])
        se_gu = xm @ w_se13
        se_act = _silu_f64(se_gu[:, : cfg.SE_R]) * se_gu[:, cfg.SE_R :]
        w_se2 = _decode_fp4_f64(w["w_se2"][r, l_idx], w["w_se2_s"][r, l_idx])
        rank_acc = (se_act @ w_se2) * se_gate

        for b in range(b_dim):
            for k_slot in range(cfg.K):
                e_id = int(top_ids[b, k_slot])
                wt = top_w[b, k_slot]
                w_e13 = _decode_fp4_f64(
                    w["w_e13"][r, l_idx, e_id], w["w_e13_s"][r, l_idx, e_id]
                )
                gu = xm[b : b + 1] @ w_e13
                act = _silu_f64(gu[:, : cfg.I_R]) * gu[:, cfg.I_R :]
                w_e2 = _decode_fp4_f64(
                    w["w_e2"][r, l_idx, e_id], w["w_e2_s"][r, l_idx, e_id]
                )
                out_e = (act @ w_e2)[0]
                rank_acc[b] += wt * out_e

        part_moe[r] = rank_acc

    moe_total = np.sum(_bf16_f64(part_moe), axis=0)
    return _bf16_f64(x + moe_total)


def run_fp64_reference(
    problem: SyntheticProblem | Mapping[str, Any],
    cfg: Qwen38Config,
    *,
    use_forced_routing: bool = True,
) -> ReferenceOutputs:
    """Execute full FP64 ground-truth simulation of Qwen3.8-2.4T step."""
    x = _bf16_f64(np.asarray(jax.device_get(problem["x"]))[0])
    pos = np.asarray(jax.device_get(problem["pos"])[0], dtype=np.int32)
    pt = np.asarray(jax.device_get(problem["page_table"]), dtype=np.int32)
    pool_jnp = jax.device_get(problem["pool"])
    pool_raw = np.asarray(pool_jnp.astype(jnp.float32), dtype=np.float64).copy()
    was_6d_pool = pool_raw.ndim == _RANK_POOL_NDIM
    if was_6d_pool:
        pool = np.repeat(pool_raw[:, None, ...], cfg.NATT, axis=1).copy()
    else:
        pool = pool_raw
    conv = _bf16_f64(np.asarray(jax.device_get(problem["conv"])))
    rec = np.asarray(jax.device_get(problem["rec"]), dtype=np.float64).copy()
    weights_map = cast("Mapping[str, jax.Array]", problem["weights"])
    w = {k: np.asarray(jax.device_get(v)) for k, v in weights_map.items()}
    forced = np.asarray(jax.device_get(problem["forced_top_ids"])[0], dtype=np.int32)

    attn_state = AttnRefState(pos=pos, pt=pt, pool=pool)
    gdn_state = GdnRefState(conv=conv, rec=rec)

    for l_idx in range(cfg.L):
        xn = _rmsnorm_f64(x, w["in_norm"][0, l_idx])
        is_att = (l_idx + 1) % cfg.att_every == 0

        if is_att:
            att_idx = l_idx // cfg.att_every
            attn_total = _ref_attention_step(xn, attn_state, w, att_idx, cfg)
            x = _bf16_f64(x + attn_total)
        else:
            g_idx = l_idx - (l_idx // cfg.att_every)
            gdn_total = _ref_gdn_step(xn, gdn_state, w, g_idx, cfg)
            x = _bf16_f64(x + gdn_total)

        forced_arg = forced if use_forced_routing else None
        x = _ref_moe_step(x, w, l_idx, forced_arg, cfg)

    x_final = _rmsnorm_f64(x, w["final_norm"][0])
    pool_out = (
        attn_state.pool[:, 0] if (was_6d_pool and cfg.NATT == 1) else attn_state.pool
    )
    pool_out_fp8 = jnp.asarray(pool_out, dtype=jnp.float32).astype(jnp.float8_e4m3fn)
    return {
        "x": x_final,
        "pool": pool_out_fp8,
        "conv": gdn_state.conv,
        "rec": gdn_state.rec,
    }


def _pack_rows_fp4(
    w_kn_u8: np.ndarray, s_gk_n_bf16: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert `(K, N//2)` row-packed nibbles into `(N, K//2)` output-first nibbles."""
    k_dim = w_kn_u8.shape[0]
    n_half = w_kn_u8.shape[1]
    n_dim = n_half * 2
    lo = w_kn_u8 & 0x0F
    hi = (w_kn_u8 >> 4) & 0x0F
    nib_kn = np.empty((k_dim, n_dim), dtype=np.uint8)
    nib_kn[:, 0::2] = lo
    nib_kn[:, 1::2] = hi
    nib_nk = np.ascontiguousarray(nib_kn.T)
    pk_nk = (nib_nk[:, 0::2] & 0x0F) | ((nib_nk[:, 1::2] & 0x0F) << 4)
    sc_nk = np.ascontiguousarray(np.asarray(s_gk_n_bf16, dtype=np.float32).T)
    return (
        torch.from_numpy(pk_nk).to(torch.uint8),
        torch.from_numpy(sc_nk).to(torch.bfloat16),
    )


def _build_mock_layer(
    l_idx: int,
    r: int,
    cfg: Qwen38Config,
    w: dict[str, np.ndarray],
    region_pools_t: list[torch.Tensor],
) -> SimpleNamespace:
    """Build one mock layer namespace with simulated linear and attention blocks."""
    is_att = (l_idx + 1) % cfg.att_every == 0
    reg_pool_t = region_pools_t[l_idx // cfg.att_every]
    in_ln = SimpleNamespace(
        weight=torch.from_numpy(
            np.asarray(w["in_norm"][r, l_idx], dtype=np.float32)
        ).to(torch.bfloat16)
    )
    post_ln = SimpleNamespace(
        weight=torch.from_numpy(
            np.asarray(w["post_norm"][r, l_idx], dtype=np.float32)
        ).to(torch.bfloat16)
    )

    w_gate_t = torch.from_numpy(
        np.asarray(w["w_gate"][r, l_idx], dtype=np.float32).T.copy()
    ).to(torch.bfloat16)
    w_se_gate_t = torch.from_numpy(
        np.asarray(w["w_se_gate"][r, l_idx], dtype=np.float32).T.copy()
    ).to(torch.bfloat16)
    se13_w, se13_s = _pack_rows_fp4(w["w_se13"][r, l_idx], w["w_se13_s"][r, l_idx])
    se2_w, se2_s = _pack_rows_fp4(w["w_se2"][r, l_idx], w["w_se2_s"][r, l_idx])

    e13_w_list = []
    e13_s_list = []
    e2_w_list = []
    e2_s_list = []
    for e_id in range(cfg.E):
        ew13, es13 = _pack_rows_fp4(
            w["w_e13"][r, l_idx, e_id], w["w_e13_s"][r, l_idx, e_id]
        )
        ew2, es2 = _pack_rows_fp4(
            w["w_e2"][r, l_idx, e_id], w["w_e2_s"][r, l_idx, e_id]
        )
        e13_w_list.append(ew13)
        e13_s_list.append(es13)
        e2_w_list.append(ew2)
        e2_s_list.append(es2)

    mlp = SimpleNamespace(
        gate=SimpleNamespace(weight=w_gate_t),
        shared_expert_gate=SimpleNamespace(weight=w_se_gate_t),
        shared_expert=SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=se13_w, weight_scale=se13_s),
            down_proj=SimpleNamespace(weight=se2_w, weight_scale=se2_s),
        ),
        experts=SimpleNamespace(
            w13_weight=torch.stack(e13_w_list, dim=0),
            w13_weight_scale=torch.stack(e13_s_list, dim=0),
            w2_weight=torch.stack(e2_w_list, dim=0),
            w2_weight_scale=torch.stack(e2_s_list, dim=0),
        ),
    )

    if is_att:
        att_idx = l_idx // cfg.att_every
        qkv_w, qkv_s = _pack_rows_fp4(w["wqkv"][r, att_idx], w["wqkv_s"][r, att_idx])
        wo_w, wo_s = _pack_rows_fp4(w["wo"][r, att_idx], w["wo_s"][r, att_idx])
        q_norm_t = torch.from_numpy(
            np.asarray(w["q_norm"][r, att_idx], dtype=np.float32)
        ).to(torch.bfloat16)
        k_norm_t = torch.from_numpy(
            np.asarray(w["k_norm"][r, att_idx], dtype=np.float32)
        ).to(torch.bfloat16)
        self_attn = SimpleNamespace(
            prefix=f"model.layers.{l_idx}.self_attn.attn",
            qkv_proj=SimpleNamespace(weight=qkv_w, weight_scale=qkv_s),
            o_proj=SimpleNamespace(weight=wo_w, weight_scale=wo_s),
            q_norm=SimpleNamespace(weight=q_norm_t),
            k_norm=SimpleNamespace(weight=k_norm_t),
            attn=SimpleNamespace(
                layer_name=f"model.layers.{l_idx}.self_attn.attn",
                kv_cache=reg_pool_t,
            ),
        )
        return SimpleNamespace(
            input_layernorm=in_ln,
            post_attention_layernorm=post_ln,
            self_attn=self_attn,
            mlp=mlp,
        )

    g_idx = l_idx - (l_idx // cfg.att_every)
    inproj_w, inproj_s = _pack_rows_fp4(
        w["w_inproj"][r, g_idx], w["w_inproj_s"][r, g_idx]
    )
    out_w, out_s = _pack_rows_fp4(w["w_out"][r, g_idx], w["w_out_s"][r, g_idx])
    ba_t = torch.from_numpy(
        np.asarray(w["w_ba"][r, g_idx], dtype=np.float32).T.copy()
    ).to(torch.bfloat16)
    conv_t = torch.from_numpy(
        np.asarray(w["w_conv"][r, g_idx], dtype=np.float32).T.copy()
    ).to(torch.bfloat16)
    alog_t = torch.from_numpy(
        np.asarray(w["a_log"][r, g_idx], dtype=np.float32).copy()
    ).to(torch.float32)
    dt_t = torch.from_numpy(
        np.asarray(w["dt_bias"][r, g_idx], dtype=np.float32).copy()
    ).to(torch.float32)
    gnorm_t = torch.from_numpy(
        np.asarray(w["gdn_norm"][r, g_idx], dtype=np.float32).copy()
    ).to(torch.bfloat16)
    linear_attn = SimpleNamespace(
        prefix=f"model.layers.{l_idx}.linear_attn",
        kv_cache=reg_pool_t,
        in_proj_qkvz=SimpleNamespace(weight=inproj_w, weight_scale=inproj_s),
        in_proj_ba=SimpleNamespace(weight=ba_t),
        conv1d=SimpleNamespace(weight=conv_t),
        A_log=alog_t,
        dt_bias=dt_t,
        norm=SimpleNamespace(weight=gnorm_t),
        out_proj=SimpleNamespace(weight=out_w, weight_scale=out_s),
    )
    return SimpleNamespace(
        input_layernorm=in_ln,
        post_attention_layernorm=post_ln,
        linear_attn=linear_attn,
        mlp=mlp,
    )


def build_mock_vllm_qwen38_rank_models(
    problem: SyntheticProblem | Mapping[str, Any], cfg: Qwen38Config
) -> SimpleNamespace:
    """Build a 32-rank mock vLLM Qwen3.8 model tree with aliased KV/GDN pools."""
    weights_map = cast("Mapping[str, jax.Array]", problem["weights"])
    w = {k: np.asarray(jax.device_get(v)) for k, v in weights_map.items()}
    pool_np = np.asarray(jax.device_get(problem["pool"]).view(jnp.uint8))
    conv_np = np.asarray(
        jax.device_get(problem["conv"]).astype(jnp.float32), dtype=np.float32
    )
    rec_np = np.asarray(jax.device_get(problem["rec"]), dtype=np.float32)
    mamba_slots = problem["mamba_slots"]

    rank_backbones: list[SimpleNamespace] = []
    pages_per_slot = _PAGES_PER_MAMBA_SLOT
    conv_bytes = (cfg.conv_kernel - 1) * cfg.QKV * 2
    g_per_reg = max(1, cfg.att_every - 1)

    for r in range(cfg.R):
        region_pools_t = [
            torch.from_numpy(
                (
                    pool_np[r, reg] if pool_np.ndim == _POOL_7D_NDIM else pool_np[r]
                ).copy()
            ).view(torch.float8_e4m3fn)
            for reg in range(cfg.NATT)
        ]
        for g_idx in range(cfg.NGDN):
            r_idx = min(g_idx // g_per_reg, cfg.NATT - 1)
            local_g = g_idx % g_per_reg
            pool_t = region_pools_t[r_idx]
            for b_idx in range(cfg.B):
                slot_raw = int(mamba_slots[g_idx, b_idx])
                if slot_raw < 0:
                    continue
                slot_g0 = int(mamba_slots[(g_idx // g_per_reg) * g_per_reg, b_idx])
                slot = (
                    slot_raw + local_g * cfg.B
                    if (local_g > 0 and slot_raw == slot_g0)
                    else slot_raw
                )
                p_base = slot * pages_per_slot
                rec_bytes = (
                    torch.from_numpy(rec_np[r, g_idx, b_idx].copy())
                    .to(torch.float32)
                    .view(torch.uint8)
                    .reshape(-1)
                )
                pool_t[p_base : p_base + 4].view(torch.uint8).reshape(-1)[
                    : rec_bytes.numel()
                ].copy_(rec_bytes)
                conv_bf16 = (
                    torch.from_numpy(conv_np[r, g_idx, b_idx].copy())
                    .to(torch.bfloat16)
                    .view(torch.uint8)
                    .reshape(-1)
                )
                pool_t[p_base + 4].view(torch.uint8).reshape(-1)[:conv_bytes].copy_(
                    conv_bf16
                )

        rank_layers = [
            _build_mock_layer(l_idx, r, cfg, w, region_pools_t)
            for l_idx in range(cfg.L)
        ]

        norm_t = torch.from_numpy(
            np.asarray(w["final_norm"][r], dtype=np.float32).copy()
        ).to(torch.bfloat16)
        x_embed_t = torch.from_numpy(
            np.asarray(
                jax.device_get(problem["x"])[r].astype(jnp.float32),
                dtype=np.float32,
            ).copy()
        ).to(torch.bfloat16)
        rank_backbones.append(
            SimpleNamespace(
                layers=rank_layers,
                norm=SimpleNamespace(weight=norm_t),
                embed_tokens=lambda ids, _x=x_embed_t: _x[ids.long()],
            )
        )

    return SimpleNamespace(
        model=rank_backbones[0],
        rank_backbones=rank_backbones,
        _rank_backbones=rank_backbones,
    )


# Public aliases for unit-test parity assertions.
bf16_f64 = _bf16_f64
rmsnorm_f64 = _rmsnorm_f64
rope_f64 = _rope_f64
decode_fp4_f64 = _decode_fp4_f64
compute_dcp_group_attention = _compute_dcp_group_attention
sig_f64 = _sig_f64
ref_gdn_step = _ref_gdn_step
silu_f64 = _silu_f64
pack_rows_fp4 = _pack_rows_fp4

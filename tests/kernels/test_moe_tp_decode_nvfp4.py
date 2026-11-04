# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU correctness and admission tests for moe_tp_decode_nvfp4 (W4A8).

Covers:
  - Trace-time admission & refusal rules (TP8 w13 b512 / w2 b256, EP2xTP4 w13
    b512 / w2 b512 and b256, small shape, invalid block sizes, dtypes, ranks).
  - Direct kernel & adapter ``_op_body`` correctness vs an ``f32`` dequantized
    reference and a plain-JAX W4A8 oracle across ``n = 1, 2, 4, 8, 16, 32, 64``
    (including ``< 16`` pad-to-16-and-slice, ``n_valid = 0``, partial padding,
    NaN-poisoned padded rows, and ``n > 64`` ``_tp_chain`` fallback).
  - Uniform, concentrated (``count = topk``, non-multiple-of-4 tail ``count =
    5``, partial single-group ``count = 3``, and EP remote-only ``count = 0``),
    and tied-score routing, plus ``fast=True`` vs ``fast=False``,
    ``hbm_operands=True``, and ``split_meta=False`` bitwise agreement.
"""

import functools
import importlib
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental import moe_tp_decode_nvfp4 as kernel
from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as fp8_kernel_impl
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_block

FP4 = jnp.float4_e2m1fn
FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32
FP4_MAX = 6.0
FP8_MAX = float(jnp.finfo(FP8).max)
Q_BLOCK = 512
N_RANKS = 8
ROUTED_SCALE = 2.5

SMALL_TP8 = dict(h=2048, e=16, i=256, fs=128, topk=4, b13=512, b2=256, ep=1)
SMALL_EP2TP4 = dict(h=2048, e=16, i=512, fs=128, topk=4, b13=512, b2=512, ep=2)
GLM52_TP8 = dict(h=6144, e=256, i=256, fs=256, topk=8, b13=512, b2=256, ep=1)
GLM52_EP2TP4 = dict(h=6144, e=256, i=512, fs=256, topk=8, b13=512, b2=512, ep=2)
GLM52_EP2TP4_B256 = dict(h=6144, e=256, i=512, fs=256, topk=8, b13=512, b2=256, ep=2)

# Kernel vs pure f32 dequantized reference (unquantized activations): W4A8
# activation quantization + bf16 combine/all-reduce roundings.
REF_TOL = dict(rel_l2=0.08, cos=0.998)
# |rel_l2(kernel, f32) - rel_l2(oracle, f32)| <= 0.005, and
# rel_l2(kernel, oracle) <= rel_l2(oracle_a, oracle_b) (measured spread between
# two valid W4A8 oracle orderings).
ORACLE_REF_REL_L2_DIFF_MAX = 0.005


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _require_tpu_mesh() -> Mesh:
    if jax.default_backend() != "tpu" or len(jax.devices("tpu")) < N_RANKS:
        pytest.skip(f"needs a TPU host with >= {N_RANKS} cores")
    return Mesh(np.asarray(jax.devices("tpu")[:N_RANKS]), ("d",))


def _shapes(
    m: int,
    cfg: dict,
    *,
    w_dtype=FP4,
    sw_dtype=FP8,
    b13: int | None = None,
    b2: int | None = None,
    local_experts: int | None = None,
    score_experts: int | None = None,
):
    sds = jax.ShapeDtypeStruct
    h, i, fs = cfg["h"], cfg["i"], cfg["fs"]
    e_global = score_experts if score_experts is not None else cfg["e"]
    el = local_experts if local_experts is not None else cfg["e"] // cfg["ep"]
    qb13 = b13 if b13 is not None else cfg["b13"]
    qb2 = b2 if b2 is not None else cfg["b2"]
    return (
        sds((m, h), BF16),
        sds((m, e_global), F32),
        sds((e_global,), F32),
        sds((el, h, 2 * i), w_dtype),
        sds((el, i, h), w_dtype),
        sds((el, h // qb13, 1, 2 * i), F32),
        sds((el, i // qb2, 1, h), F32),
        sds((h, 2 * fs), sw_dtype),
        sds((2 * fs,), F32),
        sds((fs, h), sw_dtype),
        sds((h,), F32),
    )


def _quantize_fp4_kmajor(w: jax.Array, block: int) -> tuple[jax.Array, jax.Array]:
    """Quantize ``[E, K, N]`` f32 -> ``(fp4 [E, K, N], f32 [E, K//block, 1, N])``."""
    el, k, n = w.shape
    wb = w.reshape(el, k // block, block, n)
    scale = jnp.maximum(jnp.max(jnp.abs(wb), axis=2), 1e-12) / FP4_MAX
    q = (wb / scale[:, :, None, :]).astype(FP4).reshape(el, k, n)
    return q, scale[:, :, None, :].astype(F32)


def _dequant_fp4_single(w_e: jax.Array, s_e: jax.Array) -> jax.Array:
    """Dequantize one expert ``w_e [K, N]`` fp4 with ``s_e [B, 1, N]`` f32."""
    k, n = w_e.shape
    b = s_e.shape[0]
    return (w_e.astype(F32).reshape(b, k // b, n) * s_e[:, 0:1, :]).reshape(k, n)


def _quantize_fp8_per_channel(w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Quantize ``[K, N]`` f32 -> ``(fp8 [K, N], f32 [N])``."""
    scale = jnp.maximum(jnp.max(jnp.abs(w), axis=0), 1e-12) / FP8_MAX
    return (w / scale[None, :]).astype(FP8), scale.astype(F32)


def _experts_start_per_rank(cfg: dict) -> np.ndarray:
    ep = cfg["ep"]
    if ep == 1:
        return np.zeros((N_RANKS, 1), dtype=np.int32)
    tp = N_RANKS // ep
    el = cfg["e"] // ep
    return np.asarray([(r // tp) * el for r in range(N_RANKS)], dtype=np.int32).reshape(
        N_RANKS, 1
    )


def _build_sharded_weights(mesh: Mesh, cfg: dict, seed: int = 0):
    el = cfg["e"] // cfg["ep"]
    h, i, fs = cfg["h"], cfg["i"], cfg["fs"]
    b13, b2 = cfg["b13"], cfg["b2"]

    def gen_rank(rank_arr):
        r = rank_arr.reshape(())
        k = jax.random.fold_in(jax.random.key(seed), r)
        k1, k2, k3, k4 = jax.random.split(k, 4)
        w13_f = jax.random.normal(k1, (el, h, 2 * i), F32) / jnp.sqrt(float(h))
        w2_f = jax.random.normal(k2, (el, i, h), F32) / jnp.sqrt(float(i * N_RANKS))
        w13, s13 = _quantize_fp4_kmajor(w13_f, b13)
        w2, s2 = _quantize_fp4_kmajor(w2_f, b2)
        sw13_f = jax.random.normal(k3, (h, 2 * fs), F32) / jnp.sqrt(float(h))
        sw2_f = jax.random.normal(k4, (fs, h), F32) / jnp.sqrt(float(fs * N_RANKS))
        sw13, ss13 = _quantize_fp8_per_channel(sw13_f)
        sw2, ss2 = _quantize_fp8_per_channel(sw2_f)
        return w13, w2, s13, s2, sw13, ss13, sw2, ss2

    ranks = jax.device_put(
        jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1),
        NamedSharding(mesh, P("d")),
    )
    return jax.jit(
        jax.shard_map(
            gen_rank,
            mesh=mesh,
            in_specs=(P("d"),),
            out_specs=(P("d"),) * 8,
            check_vma=False,
        )
    )(ranks)


def _make_routing_inputs(
    rows: int,
    cfg: dict,
    *,
    mode: str = "uniform",
    seed: int = 10,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    h, e, topk = cfg["h"], cfg["e"], cfg["topk"]
    x = rng.standard_normal((rows, h)).astype(np.float32)
    x_bf16 = jnp.asarray(x, dtype=BF16)
    bias = rng.uniform(-0.05, 0.05, size=(e,)).astype(np.float32)

    if mode == "uniform":
        gate = rng.standard_normal((e, h)).astype(np.float32) / np.sqrt(h)
        logits = x @ gate.T
        scores = 1.0 / (1.0 + np.exp(-logits))
    elif mode == "concentrated_topk":
        scores = rng.uniform(0.01, 0.1, size=(rows, e)).astype(np.float32)
        hot = np.arange(topk)
        scores[:, hot] = np.linspace(0.95, 0.80, topk, dtype=np.float32)
    elif mode == "concentrated_tail":
        # Exactly 5 distinct active experts across the batch (5 % 4 == 1).
        scores = np.full((rows, e), 0.01, dtype=np.float32)
        bias[:] = 0.0
        pool = np.array([1, 3, 6, 9, 12], dtype=np.int32) % e
        for r in range(rows):
            chosen = np.roll(pool, r % len(pool))[: min(topk, len(pool))]
            scores[r, chosen] = np.linspace(0.95, 0.75, len(chosen), dtype=np.float32)
    elif mode == "concentrated_sub_group":
        # At most 3 active experts when topk <= 3, or concentrated in 3 heavy +
        # fixed tail; exercises single/few groups.
        scores = np.full((rows, e), 0.01, dtype=np.float32)
        bias[:] = 0.0
        hot = np.arange(min(e, max(topk, 3)))
        scores[:, hot[:topk]] = np.linspace(0.92, 0.70, topk, dtype=np.float32)
    elif mode == "concentrated_ep0":
        # Under EP2xTP4 (el=128), all picks land in [0, el), so shard 1 sees
        # count == 0 active local experts.
        el = cfg["e"] // cfg["ep"]
        scores = rng.uniform(0.01, 0.05, size=(rows, e)).astype(np.float32)
        bias[:] = 0.0
        hot = rng.choice(el, size=(rows, topk), replace=True)
        for r in range(rows):
            cols = np.unique(hot[r])
            while len(cols) < topk:
                cols = np.unique(np.append(cols, rng.integers(0, el)))
            scores[r, cols[:topk]] = np.linspace(0.95, 0.75, topk, dtype=np.float32)
    elif mode == "ties":
        bias[:] = 0.0
        scores = (rng.integers(0, 8, size=(rows, e)) / 8.0).astype(np.float32)
    else:
        raise ValueError(f"unknown routing mode {mode!r}")

    return x_bf16, jnp.asarray(scores, dtype=F32), jnp.asarray(bias, dtype=F32)


def _fp4_to_fp8_exact(w4: jax.Array) -> jax.Array:
    """Convert ``float4_e2m1fn`` to ``float8_e4m3fn`` with an HLO barrier so XLA
    cannot fold the cast into a mixed-dtype ``f8e4m3fn x f4e2m1fn`` dot."""
    u8 = lax.bitcast_convert_type(w4.astype(FP8), jnp.uint8)
    return lax.bitcast_convert_type(lax.optimization_barrier(u8), FP8)


@functools.lru_cache(maxsize=8)
def _compiled_oracles(mesh: Mesh, h: int, e: int, i: int, fs: int, topk: int, ep: int):
    specs = (P("d"), P("d"), P(), P(), P("d")) + (P("d"),) * 8

    def ref_body(x, scores, bias, nv, e0, w13, w2, s13, s2, sw13, ss13, sw2, ss2):
        hp = lax.Precision.HIGHEST
        rows = x.shape[0]
        el = w2.shape[0]
        n_valid = nv[0]
        e_start = e0.reshape(())
        valid_row = (lax.broadcasted_iota(jnp.int32, (rows, 1), 0) < n_valid).astype(
            F32
        )
        x_clean = jnp.where(valid_row != 0, x.astype(F32), 0.0)
        wtab = fp8_kernel_impl._route(scores, bias, n_valid, topk=topk)
        expert_cols = lax.broadcasted_iota(jnp.int32, wtab.shape, 1)

        sw13_f = sw13.astype(F32) * ss13[None, :]
        sw2_f = sw2.astype(F32) * ss2[None, :]
        sgu = jnp.dot(x_clean, sw13_f, precision=hp)
        sact = jax.nn.silu(sgu[:, :fs]) * sgu[:, fs:]
        shared = jnp.dot(sact, sw2_f, precision=hp)

        def expert_step(idx, acc):
            coeff = jnp.sum(
                jnp.where(expert_cols == e_start + idx, wtab, 0.0),
                axis=1,
                keepdims=True,
            )
            w13_f = _dequant_fp4_single(w13[idx], s13[idx])
            w2_f = _dequant_fp4_single(w2[idx], s2[idx])
            gu = jnp.dot(x_clean, w13_f, precision=hp)
            act = jax.nn.silu(gu[:, :i]) * gu[:, i:]
            y = jnp.dot(act * coeff, w2_f, precision=hp)
            return acc + y

        routed = lax.fori_loop(0, el, expert_step, jnp.zeros((rows, h), F32))
        total = lax.psum(shared + ROUTED_SCALE * routed, "d")
        return jnp.where(valid_row != 0, total, 0.0)

    def exp_body(x, scores, bias, nv, e0, w13, w2, s13, s2, sw13, ss13, sw2, ss2):
        rows = x.shape[0]
        el = w2.shape[0]
        b13 = s13.shape[1]
        b2 = s2.shape[1]
        bk2 = i // b2
        n_valid = nv[0]
        e_start = e0.reshape(())
        wtab = fp8_kernel_impl._route(scores, bias, n_valid, topk=topk)
        expert_cols = lax.broadcasted_iota(jnp.int32, wtab.shape, 1)

        # Shared expert (exact match to _shared_expert: FP8 x FP8 -> F32 dot).
        sxq, sxs = quantize_block(x, axis=1, target_dtype=FP8)
        sgu = jnp.dot(sxq, sw13, preferred_element_type=F32) * sxs * ss13[None, :]
        sgu = sgu.astype(BF16).astype(F32)
        sact = jax.nn.silu(sgu[:, :fs]).astype(BF16).astype(F32) * sgu[:, fs:]
        saq, sas = quantize_block(sact.astype(BF16), axis=1, target_dtype=FP8)
        shared = jnp.dot(saq, sw2, preferred_element_type=F32) * sas * ss2[None, :]

        # Routed W4A8 input quantization per 512-column block (_quantize_blocks_f32).
        xb = x.astype(F32).reshape(rows, b13, Q_BLOCK)
        xs = jnp.max(jnp.abs(xb), axis=2) / FP8_MAX
        inv_xs = jnp.where(xs == 0, 0.0, 1.0 / xs)
        xq = (xb * inv_xs[:, :, None]).astype(FP8)

        def expert_step(idx, acc):
            coeff = jnp.sum(
                jnp.where(expert_cols == e_start + idx, wtab, 0.0),
                axis=1,
                keepdims=True,
            )
            w1_fp8 = _fp4_to_fp8_exact(w13[idx]).reshape(b13, Q_BLOCK, 2 * i)
            s1_e = s13[idx, :, 0, :]
            gu = jnp.zeros((rows, 2 * i), F32)
            for b in range(b13):
                part = jnp.matmul(xq[:, b, :], w1_fp8[b], preferred_element_type=F32)
                gu = gu + part * xs[:, b : b + 1] * s1_e[b : b + 1, :]
            act = jax.nn.silu(gu[:, :i]) * gu[:, i:]
            mid = jnp.where(coeff != 0, act * coeff, 0.0)
            ms = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / FP8_MAX
            mq = (mid * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(FP8)
            w2_fp8 = _fp4_to_fp8_exact(w2[idx]).reshape(b2, bk2, h)
            s2_e = s2[idx, :, 0, :]
            y = jnp.zeros((rows, h), F32)
            for c in range(b2):
                part2 = jnp.matmul(
                    mq[:, c * bk2 : (c + 1) * bk2],
                    w2_fp8[c],
                    preferred_element_type=F32,
                )
                y = y + part2 * s2_e[c : c + 1, :]
            return acc + y * ms

        routed = lax.fori_loop(0, el, expert_step, jnp.zeros((rows, h), F32))
        partial = fp8_kernel_impl._combine(routed, shared, ROUTED_SCALE)
        gathered = lax.all_gather(partial, "d", axis=0, tiled=False)
        acc = jnp.zeros((rows, h), F32)
        for r in range(N_RANKS):
            acc = acc + gathered[r].astype(F32)
        valid_row = lax.broadcasted_iota(jnp.int32, (rows, h), 0) < n_valid
        return jnp.where(valid_row, acc.astype(BF16), jnp.zeros((rows, h), BF16))

    def exp_alt_body(x, scores, bias, nv, e0, w13, w2, s13, s2, sw13, ss13, sw2, ss2):
        """Second valid W4A8 oracle ordering (#1763 _gmm_v2 bf16 scale/K-block
        accumulation + SwiGLU bf16 rounding + r1 direct-division mid quant and
        post-w2 coeff application)."""
        rows = x.shape[0]
        el = w2.shape[0]
        b13 = s13.shape[1]
        b2 = s2.shape[1]
        bk2 = i // b2
        n_valid = nv[0]
        e_start = e0.reshape(())
        wtab = fp8_kernel_impl._route(scores, bias, n_valid, topk=topk)
        expert_cols = lax.broadcasted_iota(jnp.int32, wtab.shape, 1)

        sxs_b = jnp.max(jnp.abs(x), axis=1, keepdims=True) / jnp.asarray(FP8_MAX, BF16)
        sxs_inv = jnp.where(
            sxs_b == 0, jnp.asarray(0.0, BF16), jnp.asarray(1.0, BF16) / sxs_b
        )
        sxq = (x * sxs_inv).astype(FP8)
        sgu = (
            jnp.dot(sxq, sw13, preferred_element_type=F32).astype(BF16)
            * sxs_b
            * ss13[None, :].astype(BF16)
        )
        sact = (jax.nn.silu(sgu[:, :fs]) * sgu[:, fs:]).astype(BF16)
        sas_b = jnp.max(jnp.abs(sact), axis=1, keepdims=True) / jnp.asarray(
            FP8_MAX, BF16
        )
        sas_inv = jnp.where(
            sas_b == 0, jnp.asarray(0.0, BF16), jnp.asarray(1.0, BF16) / sas_b
        )
        saq = (sact * sas_inv).astype(FP8)
        shared = (
            jnp.dot(saq, sw2, preferred_element_type=F32).astype(BF16)
            * sas_b
            * ss2[None, :].astype(BF16)
        ).astype(F32)

        xb = x.astype(BF16).reshape(rows, b13, Q_BLOCK)
        xs = jnp.max(jnp.abs(xb), axis=2) / jnp.asarray(FP8_MAX, BF16)
        xq = jnp.where(
            xs[:, :, None] == 0, jnp.zeros_like(xb), xb / xs[:, :, None]
        ).astype(FP8)

        def expert_step(idx, acc):
            coeff = jnp.sum(
                jnp.where(expert_cols == e_start + idx, wtab, 0.0),
                axis=1,
                keepdims=True,
            )
            w1_fp8 = _fp4_to_fp8_exact(w13[idx]).reshape(b13, Q_BLOCK, 2 * i)
            s1_e = s13[idx, :, 0, :].astype(BF16)
            gu = jnp.zeros((rows, 2 * i), BF16)
            for b in range(b13):
                part = jnp.matmul(
                    xq[:, b, :], w1_fp8[b], preferred_element_type=F32
                ).astype(BF16)
                gu = gu + part * xs[:, b : b + 1] * s1_e[b : b + 1, :]
            act = (jax.nn.silu(gu[:, :i]).astype(BF16) * gu[:, i:]).astype(F32)
            ms = jnp.max(jnp.abs(act), axis=1, keepdims=True) / FP8_MAX
            mq = jnp.where(ms == 0, 0.0, act / ms).astype(FP8)
            w2_fp8 = _fp4_to_fp8_exact(w2[idx]).reshape(b2, bk2, h)
            s2_e = s2[idx, :, 0, :]
            y = jnp.zeros((rows, h), F32)
            for c in range(b2):
                part2 = jnp.matmul(
                    mq[:, c * bk2 : (c + 1) * bk2],
                    w2_fp8[c],
                    preferred_element_type=F32,
                )
                y = y + (part2 * s2_e[c : c + 1, :]).astype(BF16).astype(F32)
            return acc + y * (ms * coeff)

        routed = lax.fori_loop(0, el, expert_step, jnp.zeros((rows, h), F32))
        partial = fp8_kernel_impl._combine(routed, shared, ROUTED_SCALE)
        gathered = lax.all_gather(partial, "d", axis=0, tiled=False)
        acc = jnp.zeros((rows, h), F32)
        for r in range(N_RANKS):
            acc = acc + gathered[r].astype(F32)
        valid_row = lax.broadcasted_iota(jnp.int32, (rows, h), 0) < n_valid
        return jnp.where(valid_row, acc.astype(BF16), jnp.zeros((rows, h), BF16))

    ref_fn = jax.jit(
        jax.shard_map(
            ref_body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
        )
    )
    exp_fn = jax.jit(
        jax.shard_map(
            exp_body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
        )
    )
    exp_alt_fn = jax.jit(
        jax.shard_map(
            exp_alt_body,
            mesh=mesh,
            in_specs=specs,
            out_specs=P("d"),
            check_vma=False,
        )
    )
    return ref_fn, exp_fn, exp_alt_fn


def _metrics(
    got_all: np.ndarray,
    ref: np.ndarray,
    exp: np.ndarray,
    exp_alt: np.ndarray,
    n_valid: int,
):
    rows = got_all.shape[0] // N_RANKS
    per_rank = got_all.reshape(N_RANKS, rows, -1)
    bits = per_rank.view(np.uint16)
    ranks_identical = bool(np.all(bits == bits[0:1]))
    pad_bits = bits[:, n_valid:, :]
    padding_positive_zero = bool(np.all(pad_bits == 0))
    got = per_rank[0].astype(np.float32)
    nan_count = int(np.isnan(got).sum())
    if n_valid == 0:
        return dict(
            ranks_identical=ranks_identical,
            padding_positive_zero=padding_positive_zero,
            nan=nan_count,
            rel_l2=0.0,
            cos=1.0,
            expected_rel_l2=0.0,
            oracle_vs_ref_rel_l2=0.0,
            rel_l2_diff_vs_oracle=0.0,
            oracle_spread_rel_l2=0.0,
        )
    g_v = got[:n_valid]
    r_v = ref[:n_valid].astype(np.float32)
    e_v = exp[:n_valid].astype(np.float32)
    ea_v = exp_alt[:n_valid].astype(np.float32)
    rel_l2 = float(np.linalg.norm(g_v - r_v) / max(np.linalg.norm(r_v), 1e-12))
    ora_rel_l2 = float(np.linalg.norm(e_v - r_v) / max(np.linalg.norm(r_v), 1e-12))
    denom = float(np.linalg.norm(g_v) * np.linalg.norm(r_v))
    cos = float(np.sum(g_v * r_v) / max(denom, 1e-12))
    exp_rel_l2 = float(np.linalg.norm(g_v - e_v) / max(np.linalg.norm(e_v), 1e-12))
    spread_rel_l2 = float(np.linalg.norm(e_v - ea_v) / max(np.linalg.norm(ea_v), 1e-12))
    return dict(
        ranks_identical=ranks_identical,
        padding_positive_zero=padding_positive_zero,
        nan=nan_count,
        rel_l2=rel_l2,
        cos=cos,
        expected_rel_l2=exp_rel_l2,
        oracle_vs_ref_rel_l2=ora_rel_l2,
        rel_l2_diff_vs_oracle=abs(rel_l2 - ora_rel_l2),
        oracle_spread_rel_l2=spread_rel_l2,
    )


# ---------------------------------------------------------------------------
# Admission & metadata unit tests (run via monkeypatched TPU 7x info)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg", [SMALL_TP8, SMALL_EP2TP4, GLM52_TP8, GLM52_EP2TP4, GLM52_EP2TP4_B256]
)
@pytest.mark.parametrize(
    "rows, admitted",
    [(16, True), (32, True), (64, True), (8, False), (48, False), (128, False)],
)
def test_admits_decode_buckets(monkeypatch, cfg, rows, admitted):
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    shapes = _shapes(rows, cfg)
    assert (
        kernel.can_use_moe_tp_decode_nvfp4(*shapes, topk=cfg["topk"], n_ranks=N_RANKS)
        is admitted
    )


def test_group_size_and_kernel_name():
    assert kernel.group_size(256, 256) == 4
    assert kernel.group_size(512, 128) == 2
    assert kernel.group_size(256, 2) == 2
    tp8_name = kernel.kernel_name(256, 16, 6144, 256, 256, 8, 256, fast=True)
    ep_name = kernel.kernel_name(128, 16, 6144, 512, 256, 8, 256, fast=True)
    assert "-g_4-" in tp8_name and "-ne_" not in tp8_name and tp8_name.endswith("-fast")
    assert "-g_2-" in ep_name and "-ne_256" in ep_name


@pytest.mark.parametrize(
    "info_fn, shapes, n_ranks",
    [
        pytest.param(
            lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1),
            _shapes(16, GLM52_TP8),
            N_RANKS,
            id="no-fp8-mxu",
        ),
        pytest.param(
            _tpu7x,
            _shapes(16, GLM52_TP8, w_dtype=FP8),
            N_RANKS,
            id="fp8-routed-weights",
        ),
        pytest.param(
            _tpu7x,
            _shapes(16, GLM52_TP8, sw_dtype=FP4),
            N_RANKS,
            id="fp4-shared-weights",
        ),
        pytest.param(
            _tpu7x,
            _shapes(16, GLM52_TP8, b13=16, b2=16),
            N_RANKS,
            id="group16-unrequantized-scales",
        ),
        pytest.param(
            _tpu7x,
            _shapes(16, GLM52_TP8, b2=64),
            N_RANKS,
            id="w2-block-below-128",
        ),
        pytest.param(
            _tpu7x,
            _shapes(16, GLM52_EP2TP4, local_experts=48),
            N_RANKS,
            id="local-experts-do-not-tile-global",
        ),
        pytest.param(_tpu7x, _shapes(16, GLM52_TP8), 1, id="single-rank"),
        pytest.param(_tpu7x, _shapes(16, GLM52_TP8), 32, id="chunk-not-lane-aligned"),
    ],
)
def test_refuses_invalid_inputs(monkeypatch, info_fn, shapes, n_ranks):
    monkeypatch.setattr(pltpu, "get_tpu_info", info_fn)
    assert not kernel.can_use_moe_tp_decode_nvfp4(*shapes, topk=8, n_ranks=n_ranks)


def test_refuses_misaligned_w2_n_chunk(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    shapes = _shapes(16, GLM52_TP8)
    assert not kernel.can_use_moe_tp_decode_nvfp4(
        *shapes, topk=8, n_ranks=N_RANKS, w2_n_chunk=100
    )
    assert kernel.can_use_moe_tp_decode_nvfp4(
        *shapes, topk=8, n_ranks=N_RANKS, w2_n_chunk=6144
    )
    assert "-nc_6144" in kernel.kernel_name(
        256, 16, 6144, 256, 256, N_RANKS, w2_n_chunk=6144
    )


@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(
            dict(h=6144, e=8, el=8, i=256, fs=256, topk=4, b13=512, b2=256, ep=1),
            id="tp8_b512_b256",
        ),
        pytest.param(
            dict(h=6144, e=16, el=8, i=512, fs=256, topk=4, b13=512, b2=512, ep=2),
            id="ep2tp4_b512_b512",
        ),
    ],
)
def test_cpu_served_per_expert_shapes_stage_audit_and_oracle_spread(cfg: dict):
    """CPU rounding-points audit at served per-expert shapes (H=6144, I=256/512)."""
    rows, n_valid = 16, 12
    h, e, el, inter, fs, topk = (
        cfg["h"],
        cfg["e"],
        cfg["el"],
        cfg["i"],
        cfg["fs"],
        cfg["topk"],
    )
    b13 = h // cfg["b13"]
    b2 = inter // cfg["b2"]
    bk2 = cfg["b2"]
    g = kernel.group_size(inter, el)
    rng = np.random.default_rng(31415 + inter)

    w13_f32 = jnp.asarray(
        rng.standard_normal((el, h, 2 * inter), dtype=np.float32) / np.sqrt(float(h))
    )
    w13_q, s13 = _quantize_fp4_kmajor(w13_f32, cfg["b13"])
    w2_f32 = jnp.asarray(
        rng.standard_normal((el, inter, h), dtype=np.float32)
        / np.sqrt(float(inter * N_RANKS))
    )
    w2_q, s2 = _quantize_fp4_kmajor(w2_f32, cfg["b2"])
    sw13_f32 = jnp.asarray(
        rng.standard_normal((h, 2 * fs), dtype=np.float32) / np.sqrt(float(h))
    )
    sw13_q, ss13 = _quantize_fp8_per_channel(sw13_f32)
    sw2_f32 = jnp.asarray(
        rng.standard_normal((fs, h), dtype=np.float32) / np.sqrt(float(fs * N_RANKS))
    )
    sw2_q, ss2 = _quantize_fp8_per_channel(sw2_f32)

    x, scores, bias = _make_routing_inputs(rows, cfg, mode="uniform", seed=2718 + inter)
    valid = (jnp.arange(rows, dtype=jnp.int32) < n_valid)[:, None]
    x_m = jnp.where(valid, x, jnp.zeros_like(x))

    # Stage 0: _route_fast vs _route
    wtab_slow = jnp.where(
        valid, fp8_kernel_impl._route(scores, bias, n_valid, topk=topk), 0.0
    )
    wtab_fast, flag_fast = fp8_kernel_impl._route_fast(scores, bias, n_valid, topk=topk)
    wtab_fast = jnp.where(valid, wtab_fast, 0.0)
    assert float(flag_fast.reshape(())) == 0.0
    assert np.array_equal(np.asarray(wtab_fast), np.asarray(wtab_slow))
    expert_cols = lax.broadcasted_iota(jnp.int32, (rows, e), 1)

    # 1. Dequantized F32 reference
    x_f = x_m.astype(F32)
    w13_deq = (w13_q.astype(F32).reshape(el, b13, cfg["b13"], 2 * inter) * s13).reshape(
        el, h, 2 * inter
    )
    w2_deq = (w2_q.astype(F32).reshape(el, b2, bk2, h) * s2).reshape(el, inter, h)
    sg_r, su_r = jnp.split(
        jnp.dot(
            x_f,
            sw13_q.astype(F32) * ss13[None, :],
            precision=lax.Precision.HIGHEST,
        ),
        2,
        axis=-1,
    )
    sh_r = jnp.dot(
        jax.nn.silu(sg_r) * su_r,
        sw2_q.astype(F32) * ss2[None, :],
        precision=lax.Precision.HIGHEST,
    )
    routed_r = jnp.zeros((rows, h), F32)
    for idx in range(el):
        c = jnp.sum(
            jnp.where(expert_cols == idx, wtab_slow, 0.0), axis=1, keepdims=True
        )
        gu_r = jnp.dot(x_f, w13_deq[idx], precision=lax.Precision.HIGHEST)
        g_r, u_r = jnp.split(gu_r, 2, axis=-1)
        routed_r = routed_r + jnp.dot(
            (c * jax.nn.silu(g_r) * u_r), w2_deq[idx], precision=lax.Precision.HIGHEST
        )
    ref_out = jnp.where(
        valid,
        (
            sh_r.astype(BF16).astype(F32)
            + routed_r.astype(BF16).astype(F32) * F32(ROUTED_SCALE)
        ).astype(BF16),
        jnp.zeros((rows, h), BF16),
    )

    # 2a. Oracle A (exp_body)
    sxq_a, sxs_a = quantize_block(x_m, axis=1, target_dtype=FP8)
    sgu_a = jnp.dot(sxq_a, sw13_q, preferred_element_type=F32) * sxs_a * ss13[None, :]
    sgu_a = sgu_a.astype(BF16).astype(F32)
    sact_a = jax.nn.silu(sgu_a[:, :fs]).astype(BF16).astype(F32) * sgu_a[:, fs:]
    saq_a, sas_a = quantize_block(sact_a.astype(BF16), axis=1, target_dtype=FP8)
    shared_a = jnp.dot(saq_a, sw2_q, preferred_element_type=F32) * sas_a * ss2[None, :]

    xb_a = x_m.astype(F32).reshape(rows, b13, Q_BLOCK)
    xs_a = jnp.max(jnp.abs(xb_a), axis=2) / FP8_MAX
    inv_xs_a = jnp.where(xs_a == 0, 0.0, 1.0 / xs_a)
    xq_a = (xb_a * inv_xs_a[:, :, None]).astype(FP8)

    xq_buf = np.zeros((rows, h), dtype=FP8)
    xs_buf = np.zeros((b13, rows, 1), dtype=np.float32)
    kernel.kernel._quantize_blocks_f32(x_m, xq_buf, xs_buf)
    assert np.array_equal(
        xq_buf.reshape(rows, b13, Q_BLOCK).view(np.uint8),
        np.asarray(xq_a).view(np.uint8),
    )
    assert np.array_equal(xs_buf[:, :, 0].T, np.asarray(xs_a))

    routed_a = jnp.zeros((rows, h), F32)
    for idx in range(el):
        c = jnp.sum(
            jnp.where(expert_cols == idx, wtab_slow, 0.0), axis=1, keepdims=True
        )
        w1_fp8 = _fp4_to_fp8_exact(w13_q[idx]).reshape(b13, Q_BLOCK, 2 * inter)
        s1_e = s13[idx, :, 0, :]
        gu = jnp.zeros((rows, 2 * inter), F32)
        for b in range(b13):
            part = jnp.matmul(xq_a[:, b, :], w1_fp8[b], preferred_element_type=F32)
            gu = gu + (part * xs_a[:, b : b + 1]) * s1_e[b : b + 1, :]
        act = jax.nn.silu(gu[:, :inter]) * gu[:, inter:]
        act = jnp.where(c != 0, act * c, 0.0)
        ms = jnp.max(jnp.abs(act), axis=1, keepdims=True) / FP8_MAX
        ms_inv = jnp.where(ms == 0, 0.0, 1.0 / ms)
        mq = (act * ms_inv).astype(FP8)
        w2_fp8 = _fp4_to_fp8_exact(w2_q[idx]).reshape(b2, bk2, h)
        s2_e = s2[idx, :, 0, :]
        for cb in range(b2):
            part2 = jnp.matmul(
                mq[:, cb * bk2 : (cb + 1) * bk2],
                w2_fp8[cb],
                preferred_element_type=F32,
            )
            routed_a = routed_a + (part2 * ms) * s2_e[cb : cb + 1, :]
    exp_out = jnp.where(
        valid,
        fp8_kernel_impl._combine(routed_a, shared_a, ROUTED_SCALE),
        jnp.zeros((rows, h), BF16),
    )

    # 2b. Kernel grouped (g=4/2) stage-by-stage mirror with w2_n_chunk=2048, 6144
    for nc in (2048, 6144):
        owner_k = jnp.zeros((rows, h), F32)
        for eg in range(0, el, g):
            acc1_g = jnp.zeros((g, rows, 2 * inter), F32)
            for b in range(b13):
                parts = []
                for gi in range(g):
                    e_id = eg + gi
                    w1_b = _fp4_to_fp8_exact(
                        w13_q[e_id, b * Q_BLOCK : (b + 1) * Q_BLOCK, :]
                    )
                    p = jnp.matmul(xq_a[:, b, :], w1_b, preferred_element_type=F32)
                    parts.append((p * xs_a[:, b : b + 1]) * s13[e_id, b, :, :])
                acc1_g = acc1_g + jnp.stack(parts, axis=0)
            for gi in range(g):
                e_id = eg + gi
                c = jnp.sum(
                    jnp.where(expert_cols == e_id, wtab_fast, 0.0),
                    axis=1,
                    keepdims=True,
                )
                gu_k = acc1_g[gi]
                act_k = jax.nn.silu(gu_k[:, :inter]) * gu_k[:, inter:]
                act_k = jnp.where(c != 0, act_k * c, 0.0)
                ms_k = jnp.max(jnp.abs(act_k), axis=1, keepdims=True) / FP8_MAX
                ms_inv_k = jnp.where(ms_k == 0, 0.0, 1.0 / ms_k)
                mq_k = (act_k * ms_inv_k).astype(FP8)
                for cb in range(b2):
                    mq_sub = mq_k[:, cb * bk2 : (cb + 1) * bk2]
                    for ni in range(h // nc):
                        n0, n1 = ni * nc, (ni + 1) * nc
                        w2_sub = _fp4_to_fp8_exact(
                            w2_q[e_id, cb * bk2 : (cb + 1) * bk2, n0:n1]
                        )
                        p2 = jnp.matmul(mq_sub, w2_sub, preferred_element_type=F32)
                        owner_k = owner_k.at[:, n0:n1].add(
                            (p2 * ms_k) * s2[e_id, cb, :, n0:n1]
                        )
        got_nc = jnp.where(
            valid,
            fp8_kernel_impl._combine(owner_k, shared_a, ROUTED_SCALE),
            jnp.zeros((rows, h), BF16),
        )
        assert np.array_equal(
            np.asarray(got_nc).view(np.uint16), np.asarray(exp_out).view(np.uint16)
        )

    # 2c. Oracle B (exp_alt_body: #1763 _gmm_v2 bf16 scale/K-block + r1 mid/coeff)
    sxs_b = jnp.max(jnp.abs(x_m), axis=1, keepdims=True) / jnp.asarray(FP8_MAX, BF16)
    sxs_inv_b = jnp.where(
        sxs_b == 0, jnp.asarray(0.0, BF16), jnp.asarray(1.0, BF16) / sxs_b
    )
    sxq_b = (x_m * sxs_inv_b).astype(FP8)
    sgu_b = (
        jnp.dot(sxq_b, sw13_q, preferred_element_type=F32).astype(BF16)
        * sxs_b
        * ss13[None, :].astype(BF16)
    )
    sact_b = (jax.nn.silu(sgu_b[:, :fs]) * sgu_b[:, fs:]).astype(BF16)
    sas_b = jnp.max(jnp.abs(sact_b), axis=1, keepdims=True) / jnp.asarray(FP8_MAX, BF16)
    sas_inv_b = jnp.where(
        sas_b == 0, jnp.asarray(0.0, BF16), jnp.asarray(1.0, BF16) / sas_b
    )
    saq_b = (sact_b * sas_inv_b).astype(FP8)
    shared_b = (
        jnp.dot(saq_b, sw2_q, preferred_element_type=F32).astype(BF16)
        * sas_b
        * ss2[None, :].astype(BF16)
    ).astype(F32)
    xb_b = x_m.astype(BF16).reshape(rows, b13, Q_BLOCK)
    xs_b = jnp.max(jnp.abs(xb_b), axis=2) / jnp.asarray(FP8_MAX, BF16)
    xq_b = jnp.where(
        xs_b[:, :, None] == 0, jnp.zeros_like(xb_b), xb_b / xs_b[:, :, None]
    ).astype(FP8)
    routed_b = jnp.zeros((rows, h), F32)
    for idx in range(el):
        c = jnp.sum(
            jnp.where(expert_cols == idx, wtab_slow, 0.0), axis=1, keepdims=True
        )
        w1_fp8 = _fp4_to_fp8_exact(w13_q[idx]).reshape(b13, Q_BLOCK, 2 * inter)
        s1_e = s13[idx, :, 0, :].astype(BF16)
        gu_b = jnp.zeros((rows, 2 * inter), BF16)
        for b in range(b13):
            part = jnp.matmul(
                xq_b[:, b, :], w1_fp8[b], preferred_element_type=F32
            ).astype(BF16)
            gu_b = gu_b + part * xs_b[:, b : b + 1] * s1_e[b : b + 1, :]
        act_b = (jax.nn.silu(gu_b[:, :inter]).astype(BF16) * gu_b[:, inter:]).astype(
            F32
        )
        ms_b = jnp.max(jnp.abs(act_b), axis=1, keepdims=True) / FP8_MAX
        mq_b = jnp.where(ms_b == 0, 0.0, act_b / ms_b).astype(FP8)
        w2_fp8 = _fp4_to_fp8_exact(w2_q[idx]).reshape(b2, bk2, h)
        s2_e = s2[idx, :, 0, :]
        y_b = jnp.zeros((rows, h), F32)
        for cb in range(b2):
            part2 = jnp.matmul(
                mq_b[:, cb * bk2 : (cb + 1) * bk2],
                w2_fp8[cb],
                preferred_element_type=F32,
            )
            y_b = y_b + (part2 * s2_e[cb : cb + 1, :]).astype(BF16).astype(F32)
        routed_b = routed_b + y_b * (ms_b * c)
    exp_alt_out = jnp.where(
        valid,
        fp8_kernel_impl._combine(routed_b, shared_b, ROUTED_SCALE),
        jnp.zeros((rows, h), BF16),
    )

    got_all = np.tile(np.asarray(exp_out), (N_RANKS, 1))
    m = _metrics(
        got_all,
        np.asarray(ref_out),
        np.asarray(exp_out),
        np.asarray(exp_alt_out),
        n_valid,
    )
    assert m["ranks_identical"], m
    assert m["padding_positive_zero"], m
    assert m["nan"] == 0, m
    assert m["rel_l2"] <= REF_TOL["rel_l2"], m
    assert m["cos"] >= REF_TOL["cos"], m
    assert m["rel_l2_diff_vs_oracle"] <= ORACLE_REF_REL_L2_DIFF_MAX, m
    assert m["expected_rel_l2"] == 0.0, m
    assert m["oracle_spread_rel_l2"] > 0.01, m
    assert m["expected_rel_l2"] <= m["oracle_spread_rel_l2"], m


# ---------------------------------------------------------------------------
# Multi-core TPU device correctness tests (kernel & adapter _op_body)
# ---------------------------------------------------------------------------


def _run_kernel_case(
    mesh: Mesh,
    cfg: dict,
    sharded_weights,
    x: jax.Array,
    scores: jax.Array,
    bias: jax.Array,
    n_valid: int,
    *,
    fast: bool = True,
    hbm_operands: bool = False,
    split_meta: bool = True,
    w2_n_chunk: int = 2048,
) -> np.ndarray:
    specs = (P("d"), P("d"), P(), P(), P("d"), P("d")) + (P("d"),) * 8
    ranks = jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1)
    e0s = jnp.asarray(_experts_start_per_rank(cfg))

    def body(x_s, s_s, b, nv, r, e0, *ws):
        assert kernel.can_use_moe_tp_decode_nvfp4(
            x_s,
            s_s,
            b,
            *ws,
            topk=cfg["topk"],
            n_ranks=N_RANKS,
            w2_n_chunk=w2_n_chunk,
        )
        return kernel.moe_tp_decode_nvfp4(
            x_s,
            s_s,
            b,
            nv[0],
            r.reshape(()),
            *ws,
            e0.reshape(()),
            routed_scale=ROUTED_SCALE,
            topk=cfg["topk"],
            n_ranks=N_RANKS,
            fast=fast,
            hbm_operands=hbm_operands,
            split_meta=split_meta,
            w2_n_chunk=w2_n_chunk,
        )

    fn = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
        )
    )
    args = [
        jax.device_put(a, NamedSharding(mesh, sp))
        for a, sp in zip(
            (
                jnp.tile(x, (N_RANKS, 1)),
                jnp.tile(scores, (N_RANKS, 1)),
                bias,
                jnp.asarray([n_valid], dtype=jnp.int32),
                ranks,
                e0s,
            ),
            specs[:6],
        )
    ] + list(sharded_weights)
    return np.asarray(fn(*args))


def _eval_oracles(mesh: Mesh, cfg: dict, sharded_weights, x, scores, bias, n_valid):
    ref_fn, exp_fn, exp_alt_fn = _compiled_oracles(
        mesh, cfg["h"], cfg["e"], cfg["i"], cfg["fs"], cfg["topk"], cfg["ep"]
    )
    specs = (P("d"), P("d"), P(), P(), P("d"))
    e0s = jnp.asarray(_experts_start_per_rank(cfg))
    prefix = [
        jax.device_put(a, NamedSharding(mesh, sp))
        for a, sp in zip(
            (
                jnp.tile(x, (N_RANKS, 1)),
                jnp.tile(scores, (N_RANKS, 1)),
                bias,
                jnp.asarray([n_valid], dtype=jnp.int32),
                e0s,
            ),
            specs,
        )
    ]
    args = prefix + list(sharded_weights)
    rows = x.shape[0]
    ref = np.asarray(ref_fn(*args))[:rows]
    exp = np.asarray(exp_fn(*args))[:rows]
    exp_alt = np.asarray(exp_alt_fn(*args))[:rows]
    return ref, exp, exp_alt


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "rows, n_valid",
    [
        (16, 0),
        (16, 1),
        (16, 4),
        (16, 8),
        (16, 16),
        (32, 1),
        (32, 12),
        (32, 32),
        (64, 8),
        (64, 33),
        (64, 64),
    ],
)
def test_small_shape_kernel_padding_and_buckets(rows: int, n_valid: int):
    mesh = _require_tpu_mesh()
    ws = _build_sharded_weights(mesh, SMALL_TP8, seed=1)
    x, scores, bias = _make_routing_inputs(
        rows, SMALL_TP8, mode="uniform", seed=100 + rows + n_valid
    )
    if n_valid < rows:
        # Poison padded rows with NaNs to verify NaN-safe masking.
        x = x.at[n_valid:].set(jnp.nan)
        scores = scores.at[n_valid:].set(jnp.nan)
    got = _run_kernel_case(mesh, SMALL_TP8, ws, x, scores, bias, n_valid, fast=True)
    ref, exp, exp_alt = _eval_oracles(mesh, SMALL_TP8, ws, x, scores, bias, n_valid)
    m = _metrics(got, ref, exp, exp_alt, n_valid)
    assert m["ranks_identical"], m
    assert m["padding_positive_zero"], m
    assert m["nan"] == 0, m
    if n_valid > 0:
        assert m["rel_l2"] <= REF_TOL["rel_l2"], m
        assert m["cos"] >= REF_TOL["cos"], m
        assert m["rel_l2_diff_vs_oracle"] <= ORACLE_REF_REL_L2_DIFF_MAX, m
        assert m["expected_rel_l2"] <= m["oracle_spread_rel_l2"], m


@pytest.mark.tpu_test
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
def test_op_body_across_decode_token_counts(n: int):
    mesh = _require_tpu_mesh()
    from vllm_torchtpu.layers.adapter import moe_tp_decode_nvfp4 as adapter

    cfg = SMALL_TP8
    ws = _build_sharded_weights(mesh, cfg, seed=2)
    x, scores, bias = _make_routing_inputs(n, cfg, mode="uniform", seed=200 + n)
    pad_mask = jnp.zeros((64,), dtype=jnp.bool_)
    statics = adapter._OpStatics(
        topk=cfg["topk"],
        routed_scale=ROUTED_SCALE,
        activation="silu",
        use_sparse_core=False,
        onehot_threshold=0,
        use_ep=False,
        router_bias=False,
        fast=True,
    )
    specs = (P("d"), P("d"), P(), P()) + (P("d"),) * 10
    body = adapter._op_body(statics, N_RANKS, "d")
    fn = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
        )
    )
    ranks = jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1)
    e0s = jnp.zeros((N_RANKS, 1), dtype=jnp.int32)
    args = (
        [
            jax.device_put(a, NamedSharding(mesh, sp))
            for a, sp in zip(
                (
                    jnp.tile(x, (N_RANKS, 1)),
                    jnp.tile(scores, (N_RANKS, 1)),
                    bias,
                    pad_mask,
                ),
                specs[:4],
            )
        ]
        + list(ws)
        + [
            jax.device_put(ranks, NamedSharding(mesh, P("d"))),
            jax.device_put(e0s, NamedSharding(mesh, P("d"))),
        ]
    )
    got = np.asarray(fn(*args))
    # Pad inputs to 16 rows for the oracle when n < 16 and slice [:n].
    m_oracle = max(16, n)
    xp = jnp.pad(x, ((0, m_oracle - n), (0, 0)))
    sp = jnp.pad(scores, ((0, m_oracle - n), (0, 0)))
    ref, exp, exp_alt = _eval_oracles(mesh, cfg, ws, xp, sp, bias, n)
    m = _metrics(got, ref[:n], exp[:n], exp_alt[:n], n)
    assert m["ranks_identical"], m
    assert m["nan"] == 0, m
    assert m["rel_l2"] <= REF_TOL["rel_l2"], m
    assert m["cos"] >= REF_TOL["cos"], m
    assert m["rel_l2_diff_vs_oracle"] <= ORACLE_REF_REL_L2_DIFF_MAX, m
    assert m["expected_rel_l2"] <= m["oracle_spread_rel_l2"], m


@pytest.mark.tpu_test
@pytest.mark.parametrize("cfg", [SMALL_TP8, SMALL_EP2TP4])
def test_op_body_above_64_falls_back_to_tp_chain(cfg: dict):
    mesh = _require_tpu_mesh()
    from vllm_torchtpu.layers.adapter import moe_tp_decode_nvfp4 as adapter

    n = 128
    ws = _build_sharded_weights(mesh, cfg, seed=3)
    x, scores, bias = _make_routing_inputs(n, cfg, mode="uniform", seed=303)
    pad_mask = jnp.zeros((n,), dtype=jnp.bool_).at[100:].set(True)
    statics = adapter._OpStatics(
        topk=cfg["topk"],
        routed_scale=ROUTED_SCALE,
        activation="silu",
        use_sparse_core=False,
        onehot_threshold=0,
        use_ep=cfg["ep"] > 1,
        router_bias=False,
        fast=True,
    )
    specs = (P("d"), P("d"), P(), P()) + (P("d"),) * 10
    fn = jax.jit(
        jax.shard_map(
            adapter._op_body(statics, N_RANKS, "d"),
            mesh=mesh,
            in_specs=specs,
            out_specs=P("d"),
            check_vma=False,
        )
    )
    ranks = jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1)
    e0s = jnp.asarray(_experts_start_per_rank(cfg))
    args = (
        [
            jax.device_put(a, NamedSharding(mesh, sp))
            for a, sp in zip(
                (
                    jnp.tile(x, (N_RANKS, 1)),
                    jnp.tile(scores, (N_RANKS, 1)),
                    bias,
                    pad_mask,
                ),
                specs[:4],
            )
        ]
        + list(ws)
        + [
            jax.device_put(ranks, NamedSharding(mesh, P("d"))),
            jax.device_put(e0s, NamedSharding(mesh, P("d"))),
        ]
    )
    got = np.asarray(fn(*args))
    ref, _, _ = _eval_oracles(mesh, cfg, ws, x, scores, bias, 100)
    per_rank = got.reshape(N_RANKS, n, -1)
    assert np.array_equal(
        per_rank.view(np.uint16),
        np.broadcast_to(per_rank[0:1].view(np.uint16), per_rank.shape),
    )
    g_v = per_rank[0, :100].astype(np.float32)
    r_v = ref[:100].astype(np.float32)
    rel_l2 = float(np.linalg.norm(g_v - r_v) / np.linalg.norm(r_v))
    assert rel_l2 <= REF_TOL["rel_l2"]


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "cfg_name, cfg",
    [
        ("glm52_tp8", GLM52_TP8),
        ("glm52_ep2tp4_b512", GLM52_EP2TP4),
        ("glm52_ep2tp4_b256", GLM52_EP2TP4_B256),
    ],
)
@pytest.mark.parametrize(
    "rows, n_valid, mode",
    [
        (16, 4, "uniform"),
        (16, 16, "concentrated_topk"),
        (16, 16, "concentrated_tail"),
        (16, 16, "concentrated_sub_group"),
        (32, 16, "uniform"),
        (64, 64, "uniform"),
    ],
)
def test_glm52_shapes_and_concentrated_routing(
    cfg_name: str, cfg: dict, rows: int, n_valid: int, mode: str
):
    del cfg_name
    mesh = _require_tpu_mesh()
    ws = _build_sharded_weights(mesh, cfg, seed=7)
    x, scores, bias = _make_routing_inputs(rows, cfg, mode=mode, seed=400 + rows)
    got = _run_kernel_case(mesh, cfg, ws, x, scores, bias, n_valid, fast=True)
    ref, exp, exp_alt = _eval_oracles(mesh, cfg, ws, x, scores, bias, n_valid)
    m = _metrics(got, ref, exp, exp_alt, n_valid)
    assert m["ranks_identical"], m
    assert m["padding_positive_zero"], m
    assert m["nan"] == 0, m
    assert m["rel_l2"] <= REF_TOL["rel_l2"], m
    assert m["cos"] >= REF_TOL["cos"], m
    assert m["rel_l2_diff_vs_oracle"] <= ORACLE_REF_REL_L2_DIFF_MAX, m
    assert m["expected_rel_l2"] <= m["oracle_spread_rel_l2"], m


@pytest.mark.tpu_test
def test_ep2tp4_zero_active_experts_on_one_ep_group():
    mesh = _require_tpu_mesh()
    ws = _build_sharded_weights(mesh, GLM52_EP2TP4, seed=9)
    x, scores, bias = _make_routing_inputs(
        16, GLM52_EP2TP4, mode="concentrated_ep0", seed=505
    )
    got = _run_kernel_case(mesh, GLM52_EP2TP4, ws, x, scores, bias, 16, fast=True)
    ref, exp, exp_alt = _eval_oracles(mesh, GLM52_EP2TP4, ws, x, scores, bias, 16)
    m = _metrics(got, ref, exp, exp_alt, 16)
    assert m["ranks_identical"], m
    assert m["nan"] == 0, m
    assert m["rel_l2"] <= REF_TOL["rel_l2"], m
    assert m["cos"] >= REF_TOL["cos"], m
    assert m["rel_l2_diff_vs_oracle"] <= ORACLE_REF_REL_L2_DIFF_MAX, m
    assert m["expected_rel_l2"] <= m["oracle_spread_rel_l2"], m


@pytest.mark.tpu_test
@pytest.mark.parametrize("mode", ["uniform", "concentrated_tail", "ties"])
def test_fast_hbm_operands_and_split_meta_are_bitwise(mode: str):
    mesh = _require_tpu_mesh()
    ws = _build_sharded_weights(mesh, SMALL_TP8, seed=11)
    x, scores, bias = _make_routing_inputs(16, SMALL_TP8, mode=mode, seed=606)
    ref_slow = _run_kernel_case(mesh, SMALL_TP8, ws, x, scores, bias, 12, fast=False)
    got_fast = _run_kernel_case(mesh, SMALL_TP8, ws, x, scores, bias, 12, fast=True)
    got_hbm = _run_kernel_case(
        mesh, SMALL_TP8, ws, x, scores, bias, 12, fast=True, hbm_operands=True
    )
    got_nosm = _run_kernel_case(
        mesh, SMALL_TP8, ws, x, scores, bias, 12, fast=True, split_meta=False
    )
    got_nc512 = _run_kernel_case(
        mesh, SMALL_TP8, ws, x, scores, bias, 12, fast=True, w2_n_chunk=512
    )
    got_nc6144 = _run_kernel_case(
        mesh, SMALL_TP8, ws, x, scores, bias, 12, fast=True, w2_n_chunk=6144
    )
    b_slow = ref_slow.view(np.uint16)
    assert np.array_equal(got_fast.view(np.uint16), b_slow)
    assert np.array_equal(got_hbm.view(np.uint16), b_slow)
    assert np.array_equal(got_nosm.view(np.uint16), b_slow)
    assert np.array_equal(got_nc512.view(np.uint16), b_slow)
    assert np.array_equal(got_nc6144.view(np.uint16), b_slow)


# ---------------------------------------------------------------------------
# CPU runtime wiring tests (env flag, per-matrix requant, hooks, runner flag)
# ---------------------------------------------------------------------------


def _has_real_module(name: str) -> bool:
    existing = sys.modules.get(name)
    if existing is not None:
        return getattr(existing, "__spec__", None) is not None
    return importlib.util.find_spec(name) is not None


def _ensure_layers_core_pkg(monkeypatch):
    """Import ``fused_moe_gmm`` without triggering ``layers/__init__.py`` on
    pure-JAX hosts where ``torch`` / ``torch_tpu`` is not installed."""
    if _has_real_module("torch") and _has_real_module("torch_tpu"):
        return importlib.import_module("vllm_torchtpu.layers.core.fused_moe_gmm")
    src_root = pathlib.Path(envs.__file__).resolve().parent
    for pkg_name, rel in (
        ("vllm_torchtpu.layers", "layers"),
        ("vllm_torchtpu.layers.core", "layers/core"),
    ):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(src_root / rel)]
            monkeypatch.setitem(sys.modules, pkg_name, pkg)
    mod_name = "vllm_torchtpu.layers.core.fused_moe_gmm"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            mod_name, src_root / "layers/core/fused_moe_gmm.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, mod_name, mod)
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


def _load_runtime_modules(monkeypatch):
    """Load real ``nvfp4``, ``moe_runner``, and ``moe_tp_decode_nvfp4`` modules.

    Uses real ``torch`` / ``vllm`` / ``torch_tpu`` whenever installed; falls
    back to minimal scoped stubs via ``monkeypatch`` on pure-JAX hosts so the
    actual adapter code still executes on CPU.
    """
    has_full = (
        _has_real_module("torch")
        and _has_real_module("vllm")
        and _has_real_module("torch_tpu")
    )
    if has_full:
        import torch as real_torch

        return SimpleNamespace(
            torch=real_torch,
            nvfp4=importlib.import_module(
                "vllm_torchtpu.layers.adapter.quantization.nvfp4"
            ),
            moe_runner=importlib.import_module(
                "vllm_torchtpu.layers.adapter.custom_ops.moe_runner"
            ),
            adapter=importlib.import_module(
                "vllm_torchtpu.layers.adapter.moe_tp_decode_nvfp4"
            ),
        )

    class _FakeTensor:
        def __init__(self, arr: np.ndarray, dtype: str = "float32"):
            self._arr = np.asarray(arr)
            self.dtype = dtype
            self.device = SimpleNamespace(type="cpu")

        @property
        def data(self):
            return self

        @property
        def shape(self):
            return self._arr.shape

        def to(self, dtype):
            np_dt = np.float32 if dtype == "float32" else self._arr.dtype
            return _FakeTensor(self._arr.astype(np_dt), dtype=str(dtype))

        def view(self, *shape):
            return _FakeTensor(self._arr.reshape(*shape), dtype=self.dtype)

        def numel(self) -> int:
            return int(self._arr.size)

        def transpose(self, d0: int, d1: int):
            return _FakeTensor(np.swapaxes(self._arr, d0, d1), dtype=self.dtype)

        def contiguous(self):
            return _FakeTensor(np.ascontiguousarray(self._arr), dtype=self.dtype)

        def unsqueeze(self, dim: int):
            return _FakeTensor(np.expand_dims(self._arr, dim), dtype=self.dtype)

        def squeeze(self, dim: int):
            return _FakeTensor(np.squeeze(self._arr, axis=dim), dtype=self.dtype)

        def clone(self):
            return _FakeTensor(self._arr.copy(), dtype=self.dtype)

        def detach(self):
            return self

        def __getitem__(self, idx):
            return _FakeTensor(self._arr[idx], dtype=self.dtype)

        def __mul__(self, other):
            rhs = other._arr if isinstance(other, _FakeTensor) else other
            return _FakeTensor(self._arr * rhs, dtype=self.dtype)

    class _FakeParameter(_FakeTensor):
        def __init__(self, data: _FakeTensor, requires_grad: bool = False):
            super().__init__(data._arr, dtype=data.dtype)
            self.requires_grad = requires_grad

    class _FakeModule:
        def __init__(self):
            self._buffers: dict[str, object] = {}

        def register_buffer(self, name: str, tensor: object, persistent: bool = True):
            del persistent
            self._buffers[name] = tensor
            setattr(self, name, tensor)

        def modules(self):
            return [self]

    def _put_mod(name: str, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    fake_nn = _put_mod("torch.nn", Module=_FakeModule, Parameter=_FakeParameter)
    fake_lib = _put_mod("torch.library", CustomOpDef=object)
    fake_torch = _put_mod(
        "torch",
        Tensor=_FakeTensor,
        nn=fake_nn,
        library=fake_lib,
        float32="float32",
        bfloat16="bfloat16",
        uint8="uint8",
        float8_e4m3fn="float8_e4m3fn",
        float4_e2m1fn_x2="float4_e2m1fn_x2",
        cat=lambda ts, dim=0: _FakeTensor(
            np.concatenate([t._arr for t in ts], axis=dim), dtype=ts[0].dtype
        ),
        zeros=lambda *s, dtype="float32": _FakeTensor(
            np.zeros(s, dtype=np.uint8 if dtype == "uint8" else np.float32),
            dtype=dtype,
        ),
        ones=lambda *s, dtype="float32": _FakeTensor(
            np.ones(s, dtype=np.float32), dtype=dtype
        ),
        empty_like=lambda t: _FakeTensor(np.empty_like(t._arr), dtype=t.dtype),
    )

    class _RoutedExperts(_FakeModule):
        def _ensure_moe_quant_config_init(self):
            pass

    class _FusedMoEMethodBase:
        def __init__(self, moe):
            self.moe = moe

    class _TpuMoEActivationMixin:
        _tpu_activation_str: str | None = None

        def _resolve_tpu_activation(self, layer) -> str:
            return getattr(layer, "activation", "silu")

        def _set_tpu_activation(self, layer) -> str:
            act = self._resolve_tpu_activation(layer)
            self._tpu_activation_str = act
            return act

    class _MoERunnerBase(_FakeModule):
        @classmethod
        def register_oot(cls, subcls):
            return subcls

        @property
        def _fused_output_is_reduced(self) -> bool:
            return False

    vllm_runner_pkg = _put_mod(
        "vllm.model_executor.layers.fused_moe.runner",
        moe_runner=SimpleNamespace(
            _moe_forward=lambda *a, **k: None,
            _moe_forward_shared=lambda *a, **k: None,
        ),
    )
    _put_mod(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner",
        MoERunner=_MoERunnerBase,
    )
    vllm_runner_pkg.moe_runner.MoERunner = _MoERunnerBase
    _put_mod("vllm", __path__=[])
    _put_mod(
        "vllm.logger",
        init_logger=lambda _name: SimpleNamespace(
            info=lambda *_a, **_k: None,
            info_once=lambda *_a, **_k: None,
            warning=lambda *_a, **_k: None,
            debug=lambda *_a, **_k: None,
        ),
    )
    _put_mod("vllm.model_executor", __path__=[])
    _put_mod("vllm.model_executor.layers", __path__=[])
    _put_mod(
        "vllm.model_executor.layers.attention", Attention=type("Attention", (), {})
    )
    _put_mod("vllm.model_executor.layers.linear", LinearBase=type("LinearBase", (), {}))
    _put_mod(
        "vllm.model_executor.layers.fused_moe",
        FusedMoEMethodBase=_FusedMoEMethodBase,
        RoutedExperts=_RoutedExperts,
        FusedMoeWeightScaleSupported=SimpleNamespace(
            BLOCK=SimpleNamespace(value="block"),
            TENSOR=SimpleNamespace(value="tensor"),
        ),
    )
    _put_mod("vllm.model_executor.layers.fused_moe.router")
    _put_mod(
        "vllm.model_executor.layers.fused_moe.router.zero_expert_router",
        ZeroExpertRouter=type("ZeroExpertRouter", (), {}),
    )
    _put_mod(
        "vllm.model_executor.layers.quantization",
        QuantizationMethods=str,
        register_quantization_config=lambda _name: lambda cls: cls,
    )
    _put_mod(
        "vllm.model_executor.layers.quantization.base_config",
        QuantizeMethodBase=type("QuantizeMethodBase", (), {}),
    )
    _put_mod(
        "vllm.model_executor.layers.quantization.modelopt",
        ModelOptNvFp4Config=type("ModelOptNvFp4Config", (), {}),
        ModelOptNvFp4FusedMoE=type("ModelOptNvFp4FusedMoE", (), {}),
        ModelOptNvFp4LinearMethod=type("ModelOptNvFp4LinearMethod", (), {}),
    )
    _put_mod(
        "vllm.model_executor.utils",
        replace_parameter=lambda layer, name, param: setattr(layer, name, param),
        set_weight_attrs=lambda *_a, **_k: None,
    )
    _put_mod("torch_tpu")
    _put_mod(
        "torch_tpu._internal",
        pallas=SimpleNamespace(
            jax_op=lambda *a, **k: SimpleNamespace(register_fake=lambda _f: None),
            pallas=SimpleNamespace(
                TORCH_TO_JAX_DTYPE_MAP={
                    "float32": jnp.float32,
                    "bfloat16": jnp.bfloat16,
                    "uint8": jnp.uint8,
                    "float8_e4m3fn": jnp.float8_e4m3fn,
                    "float4_e2m1fn_x2": jnp.float4_e2m1fn,
                }
            ),
        ),
    )

    src_root = pathlib.Path(envs.__file__).resolve().parent
    for pkg_name, rel in (
        ("vllm_torchtpu.layers", "layers"),
        ("vllm_torchtpu.layers.core", "layers/core"),
        ("vllm_torchtpu.layers.adapter", "layers/adapter"),
        ("vllm_torchtpu.layers.adapter.quantization", "layers/adapter/quantization"),
        ("vllm_torchtpu.layers.adapter.custom_ops", "layers/adapter/custom_ops"),
        (
            "vllm_torchtpu.kernels.experimental.adaptive_fused_moe",
            "kernels/experimental/adaptive_fused_moe",
        ),
    ):
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(src_root / rel)]
        monkeypatch.setitem(sys.modules, pkg_name, pkg)

    _put_mod(
        "vllm_torchtpu.utils",
        align_to=lambda x, a: ((x + a - 1) // a) * a,
        fresh_buffer=lambda t: t.clone(),
        synchronize_tensors=lambda _ts: None,
    )
    _put_mod(
        "vllm_torchtpu.kernels.experimental.adaptive_fused_moe.vllm_adapter",
        adaptive_fused_moe_supported=lambda _qm: False,
        prebuild_adaptive_nvfp4=lambda *_a, **_k: (None, None),
        fused_moe_ep_unsupported_reason=lambda *_a, **_k: None,
        run_adaptive_fused_moe=lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        sys.modules["vllm_torchtpu.kernels.experimental.adaptive_fused_moe"],
        "vllm_adapter",
        sys.modules[
            "vllm_torchtpu.kernels.experimental.adaptive_fused_moe.vllm_adapter"
        ],
        raising=False,
    )

    def _get_experts_start(layer) -> int | None:
        if not layer.moe_config.moe_parallel_config.use_ep:
            return None
        base = layer.global_num_experts // layer.moe_config.ep_size
        rem = layer.global_num_experts % layer.moe_config.ep_size
        return layer.moe_config.ep_rank * base + min(layer.moe_config.ep_rank, rem)

    def _register_experts_start_buffer(layer, *, device=None):
        del device
        e0 = _get_experts_start(layer)
        buf = (
            fake_torch.Tensor(np.asarray(e0, dtype=np.int32), dtype="int32")
            if e0 is not None
            else None
        )
        layer.register_buffer("_experts_start", buf, persistent=False)

    moe_routing_stub = _put_mod(
        "vllm_torchtpu.layers.adapter.moe_routing",
        register_experts_start_buffer=_register_experts_start_buffer,
        validate_linear_ep_placement=lambda *_a, **_k: None,
        route=lambda *_a, **_k: ("w", "ids"),
        _apply_scoring_fn=lambda _fn, logits: logits,
        _SIMULATION_STRATEGY=None,
        _ROUTER_TOPK="rowmax",
        _ROUTER_TOPK_BIAS=False,
        get_experts_start=_get_experts_start,
    )
    token_padding_stub = _put_mod(
        "vllm_torchtpu.layers.adapter.token_padding",
        _padding_state=None,
    )
    fp8_decode_stub = _put_mod(
        "vllm_torchtpu.layers.adapter.moe_tp_decode",
        _SELFTEST_DONE=False,
        moe_tp_decode_supported=lambda qm: (
            getattr(qm, "_moe_tp_decode_op", None) is not None
        ),
    )
    monkeypatch.setattr(
        sys.modules["vllm_torchtpu.layers.adapter"],
        "moe_routing",
        moe_routing_stub,
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["vllm_torchtpu.layers.adapter"],
        "token_padding",
        token_padding_stub,
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["vllm_torchtpu.layers.adapter"],
        "moe_tp_decode",
        fp8_decode_stub,
        raising=False,
    )
    _put_mod(
        "vllm_torchtpu.layers.adapter.fused_moe",
        TpuMoEActivationMixin=_TpuMoEActivationMixin,
        fused_moe_gmm=lambda **_k: "gmm_out",
        load_kmajor_fp4=lambda t: t,
        prebuild_fused_moe_kernel=lambda **_k: None,
        requant_load_kmajor_fp4=lambda t, s, b: (t, s),
    )
    _put_mod(
        "vllm_torchtpu.layers.adapter.fused_moe_ep",
        FUSED_MOE_EP_OP_ATTR="_tpu_fused_moe_ep_op",
        fused_moe_ep=lambda *_a, **_k: "ep_out",
        fused_moe_ep_supported=lambda qm: (
            getattr(qm, "_tpu_fused_moe_ep_op", None) is not None
        ),
        fused_moe_ep_unsupported_reason=lambda *_a, **_k: None,
        prebuild_fused_moe_ep=lambda *_a, **_k: None,
        register_score_bias_buffer=lambda *_a, **_k: None,
        score_bias_operand=lambda _layer: None,
    )
    _put_mod(
        "vllm_torchtpu.layers.adapter.linear_common",
        quantized_matmul_fp4=lambda *_a, **_k: None,
    )
    _put_mod(
        "vllm_torchtpu.layers.adapter.pipelined_fused_moe",
        enable_pipelined_collective_and_compute=lambda: False,
        pipelined_fused_moe_gmm=lambda **_k: "pipelined_out",
    )
    _put_mod(
        "vllm_torchtpu.layers.adapter.quantization.configs",
        VllmQuantConfig=type("VllmQuantConfig", (), {}),
        VllmQuantLinearConfig=type("VllmQuantLinearConfig", (), {}),
    )
    _put_mod(
        "vllm_torchtpu.layers.adapter.quantization.fp8",
        resolve_online_fp8=lambda _c, _l, _p, m: m,
    )
    _put_mod(
        "vllm_torchtpu.layers.adapter.quantization.online_fp8",
        map_online_fp8=lambda *_a, **_k: None,
    )
    _put_mod(
        "vllm_torchtpu.layers.core.quant_methods",
        NVFP4="modelopt_fp4",
        get_tpu_quant_method=lambda name: name,
    )
    _put_mod(
        "vllm_torchtpu.layers.core.quantization",
        dequantize_tensor=lambda *_a, **_k: None,
        pack_fp4_indices=lambda *_a, **_k: None,
        quantize_tensor_to_fp4=lambda *_a, **_k: None,
        unpack_uint8_to_fp4=lambda *_a, **_k: None,
    )
    _ensure_layers_core_pkg(monkeypatch)

    loaded = {}
    for key, mod_name, rel_file in (
        (
            "adapter",
            "vllm_torchtpu.layers.adapter.moe_tp_decode_nvfp4",
            "layers/adapter/moe_tp_decode_nvfp4.py",
        ),
        (
            "moe_runner",
            "vllm_torchtpu.layers.adapter.custom_ops.moe_runner",
            "layers/adapter/custom_ops/moe_runner.py",
        ),
        (
            "nvfp4",
            "vllm_torchtpu.layers.adapter.quantization.nvfp4",
            "layers/adapter/quantization/nvfp4.py",
        ),
    ):
        spec = importlib.util.spec_from_file_location(mod_name, src_root / rel_file)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, mod_name, mod)
        spec.loader.exec_module(mod)
        loaded[key] = mod
    monkeypatch.setattr(
        sys.modules["vllm_torchtpu.layers.adapter"],
        "moe_tp_decode_nvfp4",
        loaded["adapter"],
        raising=False,
    )
    return SimpleNamespace(torch=fake_torch, **loaded)


def test_env_flag_default_and_parsing(monkeypatch):
    envs_mod: Any = envs
    monkeypatch.delenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", raising=False)
    assert envs_mod.USE_MOE_TP_DECODE_NVFP4_KERNEL is False
    assert envs.is_set("USE_MOE_TP_DECODE_NVFP4_KERNEL") is False

    for raw in ("1", "true", "True", "yes", "on"):
        monkeypatch.setenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", raw)
        assert envs_mod.USE_MOE_TP_DECODE_NVFP4_KERNEL is True
        assert envs.is_set("USE_MOE_TP_DECODE_NVFP4_KERNEL") is True

    for raw in ("0", "false", "False", "no", "off"):
        monkeypatch.setenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", raw)
        assert envs_mod.USE_MOE_TP_DECODE_NVFP4_KERNEL is False
        assert envs.is_set("USE_MOE_TP_DECODE_NVFP4_KERNEL") is True

    monkeypatch.setenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", "")
    assert envs_mod.USE_MOE_TP_DECODE_NVFP4_KERNEL is False
    assert envs.is_set("USE_MOE_TP_DECODE_NVFP4_KERNEL") is False

    monkeypatch.setenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", "invalid")
    with pytest.raises(ValueError, match="USE_MOE_TP_DECODE_NVFP4_KERNEL"):
        _ = envs_mod.USE_MOE_TP_DECODE_NVFP4_KERNEL


@pytest.mark.parametrize(
    "two_i, inter, b13, b2",
    [
        pytest.param(512, 256, 512, 256, id="tp8"),
        pytest.param(1024, 512, 512, 512, id="ep2_tp4"),
    ],
)
def test_per_matrix_jax_requant_unpack_kmajor_shapes(
    monkeypatch, two_i: int, inter: int, b13: int, b2: int
):
    gmm = _ensure_layers_core_pkg(monkeypatch)
    rng = np.random.default_rng(42)
    e, h = 2, 6144
    w13_u8 = jnp.asarray(rng.integers(0, 256, (e, two_i, h // 2), dtype=np.uint8))
    s13_f = jnp.asarray(rng.uniform(0.05, 0.5, (e, two_i, h // 16)).astype(np.float32))
    w2_u8 = jnp.asarray(rng.integers(0, 256, (e, h, inter // 2), dtype=np.uint8))
    s2_f = jnp.asarray(rng.uniform(0.05, 0.5, (e, h, inter // 16)).astype(np.float32))

    w13_q, s13_q = gmm.requant_unpack_kmajor(w13_u8, s13_f, b13)
    w2_q, s2_q = gmm.requant_unpack_kmajor(w2_u8, s2_f, b2)

    assert w13_q.shape == (e, h, two_i) and w13_q.dtype == FP4
    assert s13_q.shape == (e, h // b13, 1, two_i) and s13_q.dtype == F32
    assert w2_q.shape == (e, inter, h) and w2_q.dtype == FP4
    assert s2_q.shape == (e, inter // b2, 1, h) and s2_q.dtype == F32


def _make_nvfp4_moe_layer(
    rt,
    *,
    e: int,
    h: int,
    inter: int,
    use_ep: bool = False,
    ep_size: int = 1,
    ep_rank: int = 0,
    tp_size: int = N_RANKS,
):
    t = rt.torch
    two_i = 2 * inter

    class _Layer(rt.nvfp4.RoutedExperts):
        pass

    layer = _Layer()
    layer.activation = "silu"
    layer.expert_placement_strategy = "linear"
    layer.global_num_experts = e * (ep_size if use_ep else 1)
    layer.moe_config = SimpleNamespace(
        experts_per_token=8,
        ep_size=ep_size if use_ep else 1,
        ep_rank=ep_rank if use_ep else 0,
        tp_size=tp_size,
        moe_parallel_config=SimpleNamespace(
            use_ep=use_ep,
            ep_size=ep_size if use_ep else 1,
            ep_rank=ep_rank if use_ep else 0,
            tp_size=tp_size,
        ),
    )
    layer.w13_weight = t.nn.Parameter(
        t.zeros(e, two_i, h // 2, dtype=t.uint8), requires_grad=False
    )
    layer.w2_weight = t.nn.Parameter(
        t.zeros(e, h, inter // 2, dtype=t.uint8), requires_grad=False
    )
    layer.w13_weight_scale = t.nn.Parameter(
        t.ones(e, two_i, h // 16, dtype=t.float32), requires_grad=False
    )
    layer.w2_weight_scale = t.nn.Parameter(
        t.ones(e, h, inter // 16, dtype=t.float32), requires_grad=False
    )
    layer.w13_weight_scale_2 = t.nn.Parameter(
        t.ones(e, 2, dtype=t.float32), requires_grad=False
    )
    layer.w2_weight_scale_2 = t.nn.Parameter(
        t.ones(e, dtype=t.float32), requires_grad=False
    )
    method = rt.nvfp4.VllmNvfp4MoEMethod(
        SimpleNamespace(group_size=16),
        SimpleNamespace(has_bias=False, is_act_and_mul=True),
    )
    return layer, method


def test_nvfp4_prepare_moe_weights_flag_off_vs_on(monkeypatch):
    rt = _load_runtime_modules(monkeypatch)
    t = rt.torch
    nvfp4 = rt.nvfp4
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "0")
    monkeypatch.setenv("MOE_FUSED_EP_ENABLE_W4A8", "0")
    monkeypatch.setattr(nvfp4, "prebuild_fused_moe_kernel", lambda **k: None)

    jax_calls: list[tuple[tuple[int, ...], int]] = []
    torch_calls: list[int] = []

    def fake_jax_requant(w_u8, _scale_f, block):
        jax_calls.append((tuple(w_u8.shape), block))
        e_dim, n_dim, k_half = w_u8.shape
        k_dim = k_half * 2
        return (
            t.zeros(e_dim, k_dim, n_dim // 2, dtype=t.uint8),
            t.zeros(e_dim, k_dim // block, 1, n_dim, dtype=t.float32),
        )

    def fake_torch_requant(w13, _s13, _w2, _s2, block):
        torch_calls.append(block)
        e_dim = w13.shape[0]
        return (
            t.zeros(e_dim, 1024, 3072, dtype=t.uint8),
            t.zeros(e_dim, 12, 1, 1024, dtype=t.float32),
            t.zeros(e_dim, 6144, 256, dtype=t.uint8),
            t.zeros(e_dim, 1, 1, 6144, dtype=t.float32),
        )

    monkeypatch.setattr(nvfp4, "requant_load_kmajor_fp4", fake_jax_requant)
    monkeypatch.setattr(nvfp4, "_requant_moe_w4a8", fake_torch_requant)
    monkeypatch.setattr(
        nvfp4,
        "load_kmajor_fp4",
        lambda w: t.zeros(w.shape[0], w.shape[2] * 2, w.shape[1] // 2, dtype=t.uint8),
    )

    # 1. Flag OFF (default): byte-identical loader path for TP8, EP2xTP4, and
    #    unrequantized block-16.
    monkeypatch.delenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", raising=False)
    layer_off_tp8, method_off_tp8 = _make_nvfp4_moe_layer(rt, e=2, h=6144, inter=256)
    method_off_tp8.process_weights_after_loading(layer_off_tp8)
    assert jax_calls == []
    assert torch_calls == [512]
    assert tuple(layer_off_tp8.w13_weight.shape) == (2, 6144, 512)
    assert tuple(layer_off_tp8.w2_weight.shape) == (2, 512, 3072)

    jax_calls.clear()
    torch_calls.clear()
    layer_off_ep2, method_off_ep2 = _make_nvfp4_moe_layer(
        rt, e=2, h=6144, inter=512, use_ep=True, ep_size=2, ep_rank=1, tp_size=4
    )
    method_off_ep2.process_weights_after_loading(layer_off_ep2)
    assert torch_calls == []
    assert jax_calls == [((2, 1024, 3072), 512), ((2, 6144, 256), 512)]
    assert tuple(layer_off_ep2.w13_weight.shape) == (2, 6144, 512)
    assert tuple(layer_off_ep2.w2_weight.shape) == (2, 512, 3072)

    jax_calls.clear()
    torch_calls.clear()
    monkeypatch.delenv("MOE_REQUANTIZE_BLOCK_SIZE", raising=False)
    layer_off_b16, method_off_b16 = _make_nvfp4_moe_layer(rt, e=2, h=6144, inter=256)
    method_off_b16.process_weights_after_loading(layer_off_b16)
    assert jax_calls == [] and torch_calls == []
    assert tuple(layer_off_b16.w13_weight.shape) == (2, 6144, 256)
    assert tuple(layer_off_b16.w13_weight_scale.shape) == (2, 384, 1, 512)
    assert tuple(layer_off_b16.w2_weight.shape) == (2, 256, 3072)
    assert tuple(layer_off_b16.w2_weight_scale.shape) == (2, 16, 1, 6144)

    # 2. Flag ON + op is None at TP8 (H=6144, I_rank=256): clamps per matrix
    #    (b13=512, b2=256), takes requant_load_kmajor_fp4, and stays unpadded.
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.setenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", "1")
    layer_on_tp8, method_on_tp8 = _make_nvfp4_moe_layer(rt, e=2, h=6144, inter=256)
    _, mode_tp8 = nvfp4._prepare_moe_weights(
        layer_on_tp8,
        layer_on_tp8.w13_weight_scale.data,
        layer_on_tp8.w2_weight_scale.data,
        512,
        clamp_per_matrix=True,
    )
    assert mode_tp8 == "FP4 (jax requant block-512/256)"
    jax_calls.clear()
    method_on_tp8.process_weights_after_loading(layer_on_tp8)
    assert torch_calls == []
    assert jax_calls == [((2, 512, 3072), 512), ((2, 6144, 128), 256)]
    assert tuple(layer_on_tp8.w13_weight.shape) == (2, 6144, 256)
    assert tuple(layer_on_tp8.w13_weight_scale.shape) == (2, 12, 1, 512)
    assert tuple(layer_on_tp8.w2_weight.shape) == (2, 256, 3072)
    assert tuple(layer_on_tp8.w2_weight_scale.shape) == (2, 1, 1, 6144)
    assert layer_on_tp8._experts_start is None

    # 3. Flag ON + op is None at EP2xTP4 (H=6144, I_rank=512, use_ep=True,
    #    ep_size=2, tp_size=4): b13=512, b2=512, unpadded shapes + _experts_start.
    jax_calls.clear()
    layer_ep2, method_ep2 = _make_nvfp4_moe_layer(
        rt, e=2, h=6144, inter=512, use_ep=True, ep_size=2, ep_rank=1, tp_size=4
    )
    _, mode_ep2 = nvfp4._prepare_moe_weights(
        layer_ep2,
        layer_ep2.w13_weight_scale.data,
        layer_ep2.w2_weight_scale.data,
        512,
        clamp_per_matrix=True,
    )
    assert mode_ep2 == "FP4 (jax requant block-512)"
    jax_calls.clear()
    method_ep2.process_weights_after_loading(layer_ep2)
    assert torch_calls == []
    assert jax_calls == [((2, 1024, 3072), 512), ((2, 6144, 256), 512)]
    assert tuple(layer_ep2.w13_weight.shape) == (2, 6144, 512)
    assert tuple(layer_ep2.w13_weight_scale.shape) == (2, 12, 1, 1024)
    assert tuple(layer_ep2.w2_weight.shape) == (2, 512, 3072)
    assert tuple(layer_ep2.w2_weight_scale.shape) == (2, 1, 1, 6144)
    assert int(layer_ep2._experts_start._arr) == 2

    # 4. Flag ON when MOE_REQUANTIZE_BLOCK_SIZE is unset: defaults block to 512
    #    so both TP8 (512/256) and EP2xTP4 (512/512) still prepare unpadded
    #    block-quantized weights ready for arming.
    jax_calls.clear()
    monkeypatch.delenv("MOE_REQUANTIZE_BLOCK_SIZE", raising=False)
    layer_auto_tp8, method_auto_tp8 = _make_nvfp4_moe_layer(rt, e=2, h=6144, inter=256)
    method_auto_tp8.process_weights_after_loading(layer_auto_tp8)
    assert jax_calls == [((2, 512, 3072), 512), ((2, 6144, 128), 256)]
    jax_calls.clear()
    layer_auto_ep2, method_auto_ep2 = _make_nvfp4_moe_layer(
        rt, e=2, h=6144, inter=512, use_ep=True, ep_size=2, ep_rank=0, tp_size=4
    )
    method_auto_ep2.process_weights_after_loading(layer_auto_ep2)
    assert jax_calls == [((2, 1024, 3072), 512), ((2, 6144, 256), 512)]

    # 5. Flag ON + admitted fused-EP W4A8 op (op is not None): does NOT clamp
    #    per matrix so fused-EP retains its uniform rhs_qb=512 layout.
    jax_calls.clear()
    torch_calls.clear()
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "512")
    ep_mod = importlib.import_module("vllm_torchtpu.layers.adapter.fused_moe_ep")
    monkeypatch.setattr(nvfp4, "_prebuild_w4a8", lambda *_a: ("admitted_ep_op", 512))
    monkeypatch.setattr(
        ep_mod, "fused_moe_ep_unsupported_reason", lambda *_a, **_k: None
    )
    monkeypatch.setattr(ep_mod, "register_score_bias_buffer", lambda *_a, **_k: None)
    layer_ep, method_ep = _make_nvfp4_moe_layer(rt, e=2, h=6144, inter=256)
    method_ep.process_weights_after_loading(layer_ep)
    assert jax_calls == []
    assert torch_calls == [512]


def test_nvfp4_apply_monolithic_dispatch_flag_off_vs_on(monkeypatch):
    rt = _load_runtime_modules(monkeypatch)
    t = rt.torch
    nvfp4 = rt.nvfp4
    adapter = rt.adapter

    layer, method = _make_nvfp4_moe_layer(rt, e=2, h=6144, inter=256)
    layer._experts_start = 0
    method._tpu_activation_str = "silu"
    x = t.zeros(4, 6144, dtype=t.bfloat16)
    logits = t.zeros(4, 256, dtype=t.float32)

    route_calls = 0
    gmm_calls = 0
    fwd_calls: list[tuple] = []

    def fake_route(*_a, **_k):
        nonlocal route_calls
        route_calls += 1
        return ("tw", "ti")

    def fake_gmm(**_k):
        nonlocal gmm_calls
        gmm_calls += 1
        return "gmm_result"

    def fake_nvfp4_fwd(qm, lyr, hs, rl):
        fwd_calls.append((qm, lyr, hs, rl))
        return "tp_decode_nvfp4_result"

    monkeypatch.setattr(nvfp4.moe_routing, "route", fake_route)
    monkeypatch.setattr(nvfp4, "fused_moe_gmm", fake_gmm)
    monkeypatch.setattr(adapter, "moe_tp_decode_nvfp4_forward", fake_nvfp4_fwd)

    # Flag OFF: never dispatches to moe_tp_decode_nvfp4_forward even if op attr exists.
    monkeypatch.delenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", raising=False)
    method._moe_tp_decode_nvfp4_op = object()
    assert method.apply_monolithic(layer, x, logits) == "gmm_result"
    assert route_calls == 1 and gmm_calls == 1 and fwd_calls == []

    # Flag ON + unarmed: falls through to GMM.
    monkeypatch.setenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", "1")
    del method._moe_tp_decode_nvfp4_op
    assert method.apply_monolithic(layer, x, logits) == "gmm_result"
    assert route_calls == 2 and gmm_calls == 2 and fwd_calls == []

    # Flag ON + armed: dispatches to moe_tp_decode_nvfp4_forward.
    method._moe_tp_decode_nvfp4_op = object()
    assert method.apply_monolithic(layer, x, logits) == "tp_decode_nvfp4_result"
    assert route_calls == 2 and gmm_calls == 2
    assert fwd_calls == [(method, layer, x, logits)]


def test_moe_runner_fused_output_is_reduced_and_should_arm(monkeypatch):
    rt = _load_runtime_modules(monkeypatch)
    adapter = rt.adapter
    VllmMoERunner = rt.moe_runner.VllmMoERunner

    monkeypatch.delenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", raising=False)
    assert adapter.should_arm() is False
    monkeypatch.setenv("USE_MOE_TP_DECODE_NVFP4_KERNEL", "1")
    assert adapter.should_arm() is True

    qm = SimpleNamespace()
    assert adapter.moe_tp_decode_nvfp4_supported(qm) is False

    runner: Any = VllmMoERunner.__new__(VllmMoERunner)
    runner._quant_method = qm
    runner.moe_config = SimpleNamespace(skip_final_all_reduce=False)
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "0")
    assert runner._fused_output_is_reduced is False

    qm._moe_tp_decode_nvfp4_op = object()
    assert adapter.moe_tp_decode_nvfp4_supported(qm) is True
    assert runner._fused_output_is_reduced is True

    runner.moe_config.skip_final_all_reduce = True
    assert runner._fused_output_is_reduced is False


def test_arm_target_nvfp4_and_refuse_bf16_drafter(monkeypatch):
    rt = _load_runtime_modules(monkeypatch)
    t = rt.torch
    adapter = rt.adapter
    VllmMoERunner = rt.moe_runner.VllmMoERunner
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    adapter._OPS.clear()
    monkeypatch.setattr(adapter, "_SELFTEST_DONE", False)

    fake_mesh = SimpleNamespace(size=N_RANKS, axis_names=("ep",))
    topo = SimpleNamespace(mesh=fake_mesh, mesh_index=0)
    monkeypatch.setattr(adapter.fp8_decode, "_topology", lambda: topo, raising=False)
    monkeypatch.setattr(
        adapter.fp8_decode, "_check_ranks_agree", lambda _d: None, raising=False
    )
    monkeypatch.setattr(adapter.fp8_decode, "_selftest", lambda *_a: [], raising=False)
    monkeypatch.setattr(adapter.fp8_decode, "_gather", lambda d: [d], raising=False)
    monkeypatch.setattr(
        adapter.fp8_decode,
        "_rank_tensor",
        lambda _dev, r: t.zeros(1, dtype=t.float32),
        raising=False,
    )
    monkeypatch.setattr(
        adapter.fp8_decode,
        "_e0_tensor",
        lambda _dev, e0: t.zeros(1, dtype=t.float32),
        raising=False,
    )
    shared_tensors = (
        t.zeros(6144, 512, dtype=t.float8_e4m3fn),
        t.zeros(512, dtype=t.float32),
        t.zeros(256, 6144, dtype=t.float8_e4m3fn),
        t.zeros(6144, dtype=t.float32),
    )
    monkeypatch.setattr(
        adapter.fp8_decode, "_shared_slice", lambda _r: shared_tensors, raising=False
    )
    built_ops: list[bool] = []
    monkeypatch.setattr(
        adapter,
        "_get_op",
        lambda _mesh, statics: built_ops.append(statics.hbm_operands) or "compiled_op",
    )

    def _build_runner(layer_name: str, quant_method):
        layer = rt.nvfp4.RoutedExperts()
        layer.quant_method = quant_method
        layer.w13_weight = t.zeros(256, 6144, 256, dtype=t.float4_e2m1fn_x2)
        layer.w2_weight = t.zeros(256, 256, 3072, dtype=t.float4_e2m1fn_x2)
        layer.w13_weight_scale = t.ones(256, 12, 1, 512, dtype=t.float32)
        layer.w2_weight_scale = t.ones(256, 1, 1, 6144, dtype=t.float32)
        layer.e_score_correction_bias = t.zeros(256, dtype=t.float32)
        layer.global_num_experts = 256
        layer.use_grouped_topk = False
        layer.num_expert_group = 1
        layer.topk_group = 1
        layer.scoring_func = "sigmoid"
        layer.renormalize = True
        layer.custom_routing_function = None
        layer.routed_scaling_factor = 1.0

        r: Any = VllmMoERunner.__new__(VllmMoERunner)
        t.nn.Module.__init__(r)
        r.layer_name = layer_name
        r.routed_experts = layer
        r._quant_method = quant_method
        r.moe_config = SimpleNamespace(
            experts_per_token=8,
            is_sequence_parallel=False,
            skip_final_all_reduce=False,
            moe_parallel_config=SimpleNamespace(
                use_ep=False,
                ep_size=1,
                tp_size=N_RANKS,
                dp_size=1,
                pcp_size=1,
                enable_eplb=False,
            ),
        )
        r.gate = object()
        r.shared_expert_gate = None
        r.routed_input_transform = None
        r.routed_output_transform = None
        r.router = object()
        r.routed_scaling_factor = 2.5
        r._shared_experts = "shared_mlp"
        r.shared_experts = "shared_mlp"
        return r

    nvfp4_qm = rt.nvfp4.VllmNvfp4MoEMethod(
        SimpleNamespace(group_size=16),
        SimpleNamespace(has_bias=False, is_act_and_mul=True),
    )
    nvfp4_qm._tpu_activation_str = "silu"
    target_runner = _build_runner("model.layers.3.mlp", nvfp4_qm)

    bf16_qm = SimpleNamespace(_tpu_activation_str="silu")
    draft_runner = _build_runner("model.layers.78.mlp", bf16_qm)

    class _Container(t.nn.Module):
        def __init__(self, r):
            super().__init__()
            self._r = r

        def modules(self):
            return [self, self._r]

    target_model = _Container(target_runner)
    draft_model = _Container(draft_runner)

    # 1. Without TPU_MOE_SKIP_PADDED_TOKENS=1, arming refuses cleanly.
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "0")
    monkeypatch.setattr(adapter.token_padding, "_padding_state", None)
    assert adapter.arm(target_model, draft_model) == 0
    assert not adapter.moe_tp_decode_nvfp4_supported(nvfp4_qm)

    # 2. With TPU_MOE_SKIP_PADDED_TOKENS=1 + padding state, target NVFP4 layer
    #    arms while BF16 drafter layer 78 stays untouched on its stock path.
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "0")
    monkeypatch.setenv("TPU_MOE_TP_DECODE_HBM_OPERANDS", "drafter")
    monkeypatch.setattr(
        adapter.token_padding,
        "_padding_state",
        SimpleNamespace(local_padding_mask=t.zeros(64, dtype=t.uint8)),
    )
    assert adapter.arm(target_model, draft_model) == 1
    assert built_ops == [False]
    assert adapter.moe_tp_decode_nvfp4_supported(nvfp4_qm) is True
    assert target_runner._shared_experts is None
    assert target_runner.routed_scaling_factor == 1.0
    assert target_runner._fused_output_is_reduced is True

    assert adapter.moe_tp_decode_nvfp4_supported(bf16_qm) is False
    assert draft_runner._shared_experts == "shared_mlp"
    assert draft_runner.routed_scaling_factor == 2.5
    assert draft_runner._fused_output_is_reduced is False


def test_arm_ep2_tp4_and_forward_dispatch(monkeypatch):
    """EP2xTP4 (TPU_MOE_EP_TP_SIZE=4) admission, arming, and forward."""
    rt = _load_runtime_modules(monkeypatch)
    t = rt.torch
    adapter = rt.adapter
    VllmMoERunner = rt.moe_runner.VllmMoERunner
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    adapter._OPS.clear()
    monkeypatch.setattr(adapter, "_SELFTEST_DONE", False)

    fake_mesh = SimpleNamespace(size=N_RANKS, axis_names=("ep",))
    topo = SimpleNamespace(mesh=fake_mesh, mesh_index=5)
    monkeypatch.setattr(adapter.fp8_decode, "_topology", lambda: topo, raising=False)
    monkeypatch.setattr(
        adapter.fp8_decode, "_check_ranks_agree", lambda _d: None, raising=False
    )
    monkeypatch.setattr(adapter.fp8_decode, "_selftest", lambda *_a: [], raising=False)
    monkeypatch.setattr(adapter.fp8_decode, "_gather", lambda d: [d], raising=False)
    monkeypatch.setattr(
        adapter.fp8_decode,
        "_rank_tensor",
        lambda _dev, r: t.Tensor(
            np.array([float(r)], dtype=np.float32), dtype=t.float32
        ),
        raising=False,
    )
    monkeypatch.setattr(
        adapter.fp8_decode,
        "_e0_tensor",
        lambda _dev, e0: t.Tensor(
            np.array([float(e0)], dtype=np.float32), dtype=t.float32
        ),
        raising=False,
    )
    shared_tensors = (
        t.zeros(6144, 512, dtype=t.float8_e4m3fn),
        t.zeros(512, dtype=t.float32),
        t.zeros(256, 6144, dtype=t.float8_e4m3fn),
        t.zeros(6144, dtype=t.float32),
    )
    monkeypatch.setattr(
        adapter.fp8_decode, "_shared_slice", lambda _r: shared_tensors, raising=False
    )

    captured_calls: list[tuple[Any, ...]] = []
    built_statics: list[Any] = []

    def _fake_op(*args):
        captured_calls.append(args)
        return args[0].clone()

    monkeypatch.setattr(
        adapter,
        "_get_op",
        lambda _mesh, statics: built_statics.append(statics) or _fake_op,
    )

    def _build_ep2_tp4_runner(
        ep_rank: int,
        *,
        ep_size: int = 2,
        tp_size: int = 4,
        local_experts: int = 128,
        global_experts: int = 256,
    ):
        qm = rt.nvfp4.VllmNvfp4MoEMethod(
            SimpleNamespace(group_size=16),
            SimpleNamespace(has_bias=False, is_act_and_mul=True),
        )
        qm._tpu_activation_str = "silu"

        layer = rt.nvfp4.RoutedExperts()
        layer.quant_method = qm
        # EP2xTP4: El=128, I_rank=512 (packed 256), b13=512 (12 blk), b2=512 (1 blk)
        layer.w13_weight = t.zeros(local_experts, 6144, 512, dtype=t.float4_e2m1fn_x2)
        layer.w2_weight = t.zeros(local_experts, 512, 3072, dtype=t.float4_e2m1fn_x2)
        layer.w13_weight_scale = t.ones(local_experts, 12, 1, 1024, dtype=t.float32)
        layer.w2_weight_scale = t.ones(local_experts, 1, 1, 6144, dtype=t.float32)
        layer.e_score_correction_bias = t.zeros(global_experts, dtype=t.float32)
        layer.global_num_experts = global_experts
        layer.use_grouped_topk = False
        layer.num_expert_group = 1
        layer.topk_group = 1
        layer.scoring_func = "sigmoid"
        layer.renormalize = True
        layer.custom_routing_function = None
        layer.routed_scaling_factor = 1.0
        layer.moe_parallel_config = SimpleNamespace(
            use_ep=True,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tp_size=tp_size,
            tp_rank=topo.mesh_index % tp_size,
            dp_size=1,
            pcp_size=1,
            enable_eplb=False,
        )

        r: Any = VllmMoERunner.__new__(VllmMoERunner)
        t.nn.Module.__init__(r)
        r.layer_name = "model.layers.3.mlp"
        r.routed_experts = layer
        r._quant_method = qm
        r.moe_config = SimpleNamespace(
            experts_per_token=8,
            is_sequence_parallel=False,
            skip_final_all_reduce=False,
            ep_size=ep_size,
            ep_rank=ep_rank,
            moe_parallel_config=layer.moe_parallel_config,
        )
        layer.moe_config = r.moe_config
        r.gate = object()
        r.shared_expert_gate = None
        r.routed_input_transform = None
        r.routed_output_transform = None
        r.router = object()
        r.routed_scaling_factor = 2.5
        r._shared_experts = "shared_mlp"
        r.shared_experts = "shared_mlp"
        return r, qm, layer

    # 1. Refusal when ep_size * tp_size != n_ranks or local_E * ep_size != global_E.
    bad_ep_tp_runner, _, _ = _build_ep2_tp4_runner(0, ep_size=2, tp_size=2)
    reason1 = adapter._admit(bad_ep_tp_runner, N_RANKS)
    assert (
        isinstance(reason1, str)
        and "needs EP x TP MoE over 8 ranks (ep 2, tp 2)" in reason1
    )

    bad_local_e_runner, _, _ = _build_ep2_tp4_runner(
        0, local_experts=128, global_experts=128
    )
    reason2 = adapter._admit(bad_local_e_runner, N_RANKS)
    assert isinstance(reason2, str) and "128 local experts x ep 2 != 128" in reason2

    # 2. Admission on ep_rank=0 (e0=0) and ep_rank=1 (e0=128) produces identical
    #    _OpStatics (use_ep=True) while experts_start differs per EP rank.
    r_ep0, _, _ = _build_ep2_tp4_runner(0)
    r_ep1, qm_ep1, layer_ep1 = _build_ep2_tp4_runner(1)
    adm0 = adapter._admit(r_ep0, N_RANKS)
    adm1 = adapter._admit(r_ep1, N_RANKS)
    assert not isinstance(adm0, str) and not isinstance(adm1, str)
    assert adm0.statics.use_ep is True and adm1.statics.use_ep is True
    assert adm0.experts_start == 0 and adm1.experts_start == 128
    assert adm0.statics == adm1.statics and adm0.rows == adm1.rows

    # 3. Arm EP2xTP4 runner on ep_rank=1 (mesh_index=5 -> e0=128) and verify forward.
    class _Container(t.nn.Module):
        def __init__(self, r):
            super().__init__()
            self._r = r

        def modules(self):
            return [self, self._r]

    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "0")
    monkeypatch.setenv("TPU_MOE_TP_DECODE_HBM_OPERANDS", "0")
    pad_mask = t.zeros(64, dtype=t.uint8)
    monkeypatch.setattr(
        adapter.token_padding,
        "_padding_state",
        SimpleNamespace(local_padding_mask=pad_mask),
    )

    assert adapter.arm(_Container(r_ep1)) == 1
    assert len(built_statics) == 1 and built_statics[0].use_ep is True
    assert adapter.moe_tp_decode_nvfp4_supported(qm_ep1) is True
    assert r_ep1._shared_experts is None
    assert r_ep1.routed_scaling_factor == 1.0
    assert r_ep1._fused_output_is_reduced is True
    assert float(layer_ep1._moe_tp_decode_nvfp4_rank._arr[0]) == 5.0
    assert float(layer_ep1._moe_tp_decode_nvfp4_e0._arr[0]) == 128.0

    x = t.zeros(16, 6144, dtype=t.bfloat16)
    logits = t.zeros(16, 256, dtype=t.bfloat16)
    out = adapter.moe_tp_decode_nvfp4_forward(qm_ep1, layer_ep1, x, logits)
    assert out.shape == (16, 6144)
    assert len(captured_calls) == 1
    call_args = captured_calls[0]
    assert len(call_args) == 14
    assert call_args[0] is x
    assert call_args[2] is layer_ep1._moe_tp_decode_nvfp4_bias
    assert call_args[3] is pad_mask
    assert call_args[12] is layer_ep1._moe_tp_decode_nvfp4_rank
    assert call_args[13] is layer_ep1._moe_tp_decode_nvfp4_e0

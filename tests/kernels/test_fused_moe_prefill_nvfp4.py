# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Unit, TPU-lowering, and hardware correctness tests for fused_moe_prefill_nvfp4.

Covers:
  - Static admission & refusal rules (EP2xTP4, TP8, EP8 VMEM refusal, pre-v7x
    refusal, block sizes, dtypes, w2_n_chunk alignment, input validation).
  - End-to-end 8-rank ``shard_map`` execution on CPU (small shapes < 5 MiB) and
    on 8-core TPU hardware across ``act_mode="w4a16"`` (default) and ``"w4a8"``:
      (1a) Partial EP empty tokens: all routes of a subset of tokens go to the
           remote EP group (0 local routes for those tokens).
      (1b) One EP group gets 100% of routes across the entire batch (0 local
           routes / ``routed_visits == 0`` on the other EP group).
      (2)  Shared-expert-once check: zero routed weights (``w1=w2=0`` and
           ``topk_weights=0``) produce the TP8-sharded ``xla_quantized_matmul``
           shared expert summed once across 8 ranks (never 2x under EP2xTP4),
           unaffected by ``routed_scale``.
      (3)  Non-tile-multiple ``T = 1000`` and ``T = 128``, NaN-poisoned padded
           rows, gated & ungated shared expert, ``w2_n_chunk`` invariance, and
           pre-quantized ``(x_q, x_scale)`` equivalence.
  - Real TPU Mosaic + StableHLO compiler lowering (``lowering_platforms=("tpu",)``)
    on CPU for ``SMALL_*`` and abstract ``GLM-5.2`` ``ShapeDtypeStruct``s (zero
    host RAM allocation for ``GLM-5.2`` weights).
"""

from __future__ import annotations

import os
from typing import Any

# Ensure 8 host CPU devices are available when pytest starts JAX on CPU.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental import fused_moe_prefill_nvfp4 as mod
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel_v2 as prefill_v2,
)
from vllm_torchtpu.kernels.experimental.fused_moe_prefill.fused_moe_prefill_kernel import (  # noqa: E501
    _LANES,
    _TILE_RUNGS,
    _psum_rows,
)
from vllm_torchtpu.kernels.experimental.fused_moe_prefill_nvfp4 import (
    kernel as nvfp4_kernel,
)
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    DN,
    _intermediate_rows,
    _tile_row_scales,
)
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_block,
    xla_quantized_matmul,
)

FP4 = jnp.float4_e2m1fn
FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32
FP4_MAX = 6.0
FP8_MAX = float(jnp.finfo(FP8).max)
N_RANKS = 8
ROUTED_SCALE = 2.5

# Small configs keep per-rank weights well under 1 MiB for fast CPU execution.
SMALL_EP2TP4 = dict(h=512, e=16, i=256, i_s=128, topk=4, qb1=512, bk2=256, ep=2)
SMALL_TP8 = dict(h=512, e=16, i=128, i_s=128, topk=4, qb1=512, bk2=128, ep=1)

# Full GLM-5.2 shapes (used as abstract ShapeDtypeStructs ONLY on CPU).
GLM52_EP2TP4 = dict(h=6144, e=256, i=512, i_s=256, topk=8, qb1=512, bk2=512, ep=2)
GLM52_EP2TP4_B256 = dict(h=6144, e=256, i=512, i_s=256, topk=8, qb1=512, bk2=256, ep=2)
GLM52_TP8 = dict(h=6144, e=256, i=256, i_s=256, topk=8, qb1=512, bk2=256, ep=1)
GLM52_EP8 = dict(h=6144, e=256, i=2048, i_s=256, topk=8, qb1=512, bk2=512, ep=8)


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _install_mock_tpu7x(monkeypatch: pytest.MonkeyPatch) -> None:
    from jax._src.pallas.mosaic import tpu_info as mosaic_tpu_info

    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    monkeypatch.setattr(mosaic_tpu_info, "get_tpu_info", _tpu7x)
    monkeypatch.setattr(mosaic_tpu_info, "is_tpu_device", lambda: True)
    try:
        from jax._src import tpu_info as jax_tpu_info

        monkeypatch.setattr(jax_tpu_info, "get_tpu_info", _tpu7x)
        if hasattr(jax_tpu_info, "is_tpu_device"):
            monkeypatch.setattr(jax_tpu_info, "is_tpu_device", lambda: True)
    except ImportError:
        pass


def _cpu_or_any_mesh() -> Mesh:
    devs = (
        jax.devices("cpu")
        if "cpu" in {d.platform for d in jax.devices()}
        else jax.devices()
    )
    if len(devs) < N_RANKS:
        pytest.skip(
            f"needs >= {N_RANKS} JAX devices "
            "(set --xla_force_host_platform_device_count=8)"
        )
    return Mesh(np.asarray(devs[:N_RANKS]), ("d",))


def _require_tpu_mesh() -> Mesh:
    if jax.default_backend() != "tpu" or len(jax.devices("tpu")) < N_RANKS:
        pytest.skip(f"needs a TPU host with >= {N_RANKS} cores")
    return Mesh(np.asarray(jax.devices("tpu")[:N_RANKS]), ("d",))


def _rel_l2(got: Any, want: Any) -> float:
    g = np.asarray(got, dtype=np.float64).reshape(-1)
    w = np.asarray(want, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(w)), 1e-30)
    return float(np.linalg.norm(g - w) / denom)


def _cos_sim(got: Any, want: Any) -> float:
    g = np.asarray(got, dtype=np.float64).reshape(-1)
    w = np.asarray(want, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(g) * np.linalg.norm(w))
    if denom <= 1e-30:
        return 1.0 if float(np.linalg.norm(g - w)) <= 1e-30 else 0.0
    return float(np.dot(g, w) / denom)


def _shapes(
    tokens: int,
    cfg: dict[str, int],
    *,
    w_dtype=FP4,
    sw_dtype=FP8,
    qb1: int | None = None,
    bk2: int | None = None,
    scale_ndim: int = 4,
    with_gate: bool = False,
):
    sds = jax.ShapeDtypeStruct
    h, e, i, i_s, topk, ep = (
        cfg["h"],
        cfg["e"],
        cfg["i"],
        cfg["i_s"],
        cfg["topk"],
        cfg["ep"],
    )
    el = e // ep
    q1 = qb1 if qb1 is not None else cfg["qb1"]
    q2 = bk2 if bk2 is not None else cfg["bk2"]
    b13 = h // q1
    b2 = i // q2
    s1_shape = (el, b13, 1, 2 * i) if scale_ndim == 4 else (el, b13, 2 * i)
    s2_shape = (el, b2, 1, h) if scale_ndim == 4 else (el, b2, h)
    gate = sds((h, 1), BF16) if with_gate else None
    return (
        sds((tokens, h), BF16),
        sds((el, h, 2 * i), w_dtype),
        sds((el, i, h), w_dtype),
        sds(s1_shape, F32),
        sds(s2_shape, F32),
        sds((h, 2 * i_s), sw_dtype),
        sds((2 * i_s,), F32),
        sds((i_s, h), sw_dtype),
        sds((h,), F32),
        sds((tokens, topk), BF16),
        sds((tokens, topk), jnp.int32),
        gate,
    )


def _quantize_fp4_kmajor(w: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    el, k, n = w.shape
    wb = w.reshape(el, k // block, block, n)
    scale = np.maximum(np.max(np.abs(wb), axis=2), 1e-12) / FP4_MAX
    q = jnp.asarray(wb / scale[:, :, None, :], dtype=FP4).reshape(el, k, n)
    return np.asarray(q), scale[:, :, None, :].astype(np.float32)


def _quantize_fp8_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.max(np.abs(w), axis=0), 1e-12) / FP8_MAX
    q = jnp.asarray(np.clip(w / scale[None, :], -FP8_MAX, FP8_MAX), dtype=FP8)
    return np.asarray(q), scale.astype(np.float32)


def _dequant_fp4_single(w_e: jax.Array, s_e: jax.Array) -> jax.Array:
    k, n = w_e.shape
    b = s_e.shape[0]
    return (w_e.astype(F32).reshape(b, k // b, n) * s_e[:, 0:1, :]).reshape(k, n)


def _fp4_to_exact(w4: jax.Array, target_dtype) -> jax.Array:
    if target_dtype == FP8:
        u8 = lax.bitcast_convert_type(w4.astype(FP8), jnp.uint8)
        return lax.bitcast_convert_type(lax.optimization_barrier(u8), FP8)
    u16 = lax.bitcast_convert_type(w4.astype(BF16), jnp.uint16)
    return lax.bitcast_convert_type(lax.optimization_barrier(u16), BF16)


def _expert_starts_for(cfg: dict[str, int]) -> np.ndarray:
    ep = cfg["ep"]
    if ep == 1:
        return np.zeros((N_RANKS, 1), dtype=np.int32)
    tp = N_RANKS // ep
    el = cfg["e"] // ep
    return np.asarray([(r // tp) * el for r in range(N_RANKS)], dtype=np.int32).reshape(
        N_RANKS, 1
    )


def _build_small_sharded_weights(
    mesh: Mesh,
    cfg: dict[str, int],
    *,
    seed: int = 0,
    zero_routed: bool = False,
):
    rng = np.random.default_rng(seed)
    h, e, i, i_s, qb1, bk2, ep = (
        cfg["h"],
        cfg["e"],
        cfg["i"],
        cfg["i_s"],
        cfg["qb1"],
        cfg["bk2"],
        cfg["ep"],
    )
    el = e // ep
    w1_ranks, w2_ranks, s1_ranks, s2_ranks = [], [], [], []
    sw1_ranks, ss1_ranks, sw2_ranks, ss2_ranks = [], [], [], []
    for _ in range(N_RANKS):
        if zero_routed:
            w1_f = np.zeros((el, h, 2 * i), dtype=np.float32)
            w2_f = np.zeros((el, i, h), dtype=np.float32)
        else:
            w1_f = (rng.standard_normal((el, h, 2 * i)) / np.sqrt(h)).astype(np.float32)
            w2_f = (rng.standard_normal((el, i, h)) / np.sqrt(i * N_RANKS)).astype(
                np.float32
            )
        q1, s1 = _quantize_fp4_kmajor(w1_f, qb1)
        q2, s2 = _quantize_fp4_kmajor(w2_f, bk2)
        if zero_routed:
            s1[:] = 0.0
            s2[:] = 0.0
        sw1_f = (rng.standard_normal((h, 2 * i_s)) / np.sqrt(h)).astype(np.float32)
        sw2_f = (rng.standard_normal((i_s, h)) / np.sqrt(i_s * N_RANKS)).astype(
            np.float32
        )
        sq1, ss1 = _quantize_fp8_per_channel(sw1_f)
        sq2, ss2 = _quantize_fp8_per_channel(sw2_f)
        w1_ranks.append(q1)
        w2_ranks.append(q2)
        s1_ranks.append(s1)
        s2_ranks.append(s2)
        sw1_ranks.append(sq1)
        ss1_ranks.append(ss1)
        sw2_ranks.append(sq2)
        ss2_ranks.append(ss2)

    shd = NamedSharding(mesh, P("d"))
    e0 = jax.device_put(jnp.asarray(_expert_starts_for(cfg)), shd)
    return (
        jax.device_put(jnp.asarray(np.concatenate(w1_ranks, axis=0), dtype=FP4), shd),
        jax.device_put(jnp.asarray(np.concatenate(w2_ranks, axis=0), dtype=FP4), shd),
        jax.device_put(jnp.asarray(np.concatenate(s1_ranks, axis=0), dtype=F32), shd),
        jax.device_put(jnp.asarray(np.concatenate(s2_ranks, axis=0), dtype=F32), shd),
        jax.device_put(jnp.asarray(np.concatenate(sw1_ranks, axis=0), dtype=FP8), shd),
        jax.device_put(jnp.asarray(np.concatenate(ss1_ranks, axis=0), dtype=F32), shd),
        jax.device_put(jnp.asarray(np.concatenate(sw2_ranks, axis=0), dtype=FP8), shd),
        jax.device_put(jnp.asarray(np.concatenate(ss2_ranks, axis=0), dtype=F32), shd),
        e0,
    )


def _make_inputs(
    tokens: int,
    cfg: dict[str, int],
    *,
    mode: str = "uniform",
    seed: int = 42,
    pad_tokens: int = 0,
    poison_pad_nan: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, e, topk, ep = cfg["h"], cfg["e"], cfg["topk"], cfg["ep"]
    el = e // ep
    x = (rng.standard_normal((tokens, h)) * 0.25).astype(np.float32)

    if mode == "uniform":
        ids = np.stack(
            [rng.choice(e, size=topk, replace=False) for _ in range(tokens)]
        ).astype(np.int32)
    elif mode == "partial_ep_empty_tokens":
        # Even tokens route 100% to EP group 1 [el, 2*el) -> 0 local routes on EP0;
        # odd tokens route 100% to EP group 0 [0, el) -> 0 local routes on EP1;
        # every 5th token spans both EP groups.
        ids = np.zeros((tokens, topk), dtype=np.int32)
        for t in range(tokens):
            if t % 5 == 0:
                ids[t] = rng.choice(e, size=topk, replace=False)
            elif t % 2 == 0:
                ids[t] = rng.choice(np.arange(el, e), size=topk, replace=False)
            else:
                ids[t] = rng.choice(np.arange(0, el), size=topk, replace=False)
    elif mode == "ep0_all_routes":
        # 100% of routes across the entire batch land in EP group 0 [0, el).
        ids = np.stack(
            [
                rng.choice(np.arange(0, el), size=topk, replace=False)
                for _ in range(tokens)
            ]
        ).astype(np.int32)
    elif mode == "ep1_all_routes":
        # 100% of routes across the entire batch land in EP group 1 [el, e).
        ids = np.stack(
            [
                rng.choice(np.arange(el, e), size=topk, replace=False)
                for _ in range(tokens)
            ]
        ).astype(np.int32)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    raw_w = rng.uniform(0.1, 1.0, size=(tokens, topk)).astype(np.float32)
    tw = raw_w / raw_w.sum(axis=1, keepdims=True)
    if mode == "zero_weights":
        tw[:] = 0.0

    if pad_tokens > 0:
        valid = tokens - pad_tokens
        tw[valid:] = 0.0
        if poison_pad_nan:
            x[valid:, :8] = np.nan
            x[valid:, 8:16] = np.inf
    return x, tw, ids


def _install_cpu_expert_emulator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emulate ``build_expert_kernel_nvfp4`` + ``combine`` Pallas calls on CPU.

    Executes the exact tile/rung/K-block/N-chunk math of ``compute_routed`` and
    ``compute_shared`` against the host-built ``visit_*``, ``row_token``, and
    ``row_scale`` tables so ``fused_moe_prefill_nvfp4`` runs end-to-end inside an
    8-rank ``jax.shard_map`` on CPU.
    """

    def cpu_build_expert_kernel_nvfp4(
        *,
        experts: int,
        h: int,
        i: int,
        i_s: int,
        b13: int,
        b2: int,
        tokens: int,
        routes: int,
        tile_rows: int,
        act_mode: mod.ActMode = "w4a16",
        nb: int = 4,
        rung_rows: int | None = None,
        w2_n_chunk: int = 2048,
        include_shared: bool = True,
    ):
        del nb
        has_act_scale = act_mode == "w4a8"
        act_dtype = FP8 if has_act_scale else BF16
        window = tile_rows + _LANES
        n_rungs = _TILE_RUNGS if rung_rows is None else tile_rows // rung_rows
        rungs = [tile_rows * (r + 1) // n_rungs for r in range(n_rungs)]
        lane_blocks = h // _LANES
        qb1 = h // b13
        bk2 = i // b2
        n_chunk = min(h, w2_n_chunk)
        slab_rows = routes + (tokens if include_shared else 1) + tile_rows
        scale_rows = (_cdiv_py(slab_rows, _LANES) + 1) if has_act_scale else 0
        token_rows = _cdiv_py(routes, _LANES) * _LANES + window

        def call(*call_args):
            visit_expert, visit_start, visit_rows, num_visits = call_args[:4]
            if has_act_scale:
                x_hbm, row_scale_hbm, row_token_hbm = call_args[4:7]
                rest = call_args[7:]
            else:
                x_hbm, row_token_hbm = call_args[4:6]
                row_scale_hbm = None
                rest = call_args[6:]
            w1_hbm, w2_hbm, w1_scale_hbm, w2_scale_hbm = rest[:4]
            if include_shared:
                (
                    shared_w1_hbm,
                    shared_w2_hbm,
                    shared_w1_scale_hbm,
                    shared_w2_scale_hbm,
                ) = rest[4:8]
            else:
                shared_w1_hbm = shared_w2_hbm = None
                shared_w1_scale_hbm = shared_w2_scale_hbm = None

            n_visit = num_visits[0]
            x_flat = x_hbm.reshape(tokens, h).astype(act_dtype)
            slab_init = jnp.zeros((slab_rows, lane_blocks, _LANES), BF16)

            def cond_fn(state):
                v, _, _ = state
                return v < n_visit

            def body_fn(state):
                v, t, slab = state
                start = visit_start[v] + t * tile_rows
                live = jnp.minimum(visit_rows[v] - t * tile_rows, tile_rows)
                is_sh = visit_expert[v] == experts

                # Rung selection matches the Pallas kernel's exact rung height.
                rung_h = jnp.int32(tile_rows)
                for r_idx in range(n_rungs - 1, -1, -1):
                    rung_h = jnp.where(live <= rungs[r_idx], rungs[r_idx], rung_h)

                # Gather `tile_rows` activation rows (shared: contiguous;
                # routed: indexed).
                row_iota = jnp.arange(tile_rows, dtype=jnp.int32)
                sh_idx = jnp.minimum(t * tile_rows + row_iota, tokens - 1)
                rt_idx = row_token_hbm[start + row_iota]
                tok_idx = jnp.where(is_sh, sh_idx, rt_idx) if include_shared else rt_idx
                act_full = x_flat[tok_idx]  # [tile_rows, h]
                valid_r = (row_iota < rung_h)[:, None]
                act = jnp.where(valid_r, act_full, 0)

                if has_act_scale:
                    assert row_scale_hbm is not None
                    scale_win = lax.dynamic_slice_in_dim(
                        row_scale_hbm, start // _LANES, tile_rows // _LANES + 1, axis=0
                    )
                    act_scale = _tile_row_scales(scale_win, start, tile_rows)
                else:
                    act_scale = None

                def run_shared():
                    assert (
                        shared_w1_hbm is not None
                        and shared_w2_hbm is not None
                        and shared_w1_scale_hbm is not None
                        and shared_w2_scale_hbm is not None
                    )
                    sw1 = shared_w1_hbm[0]
                    sw2 = shared_w2_hbm[0]
                    ss1 = shared_w1_scale_hbm[0]
                    ss2 = shared_w2_scale_hbm[0]
                    if has_act_scale:
                        gu = lax.dot_general(act, sw1, DN, preferred_element_type=F32)
                        mid_q, mid_s = _intermediate_rows(
                            gu,
                            act_scale,
                            i_s,
                            w1s=ss1,
                            act_fn="silu",
                            w1b=None,
                            requantize=True,
                        )
                        assert mid_s is not None
                    else:
                        sxq, sxs = quantize_block(act, axis=1, target_dtype=FP8)
                        sxs = sxs.astype(F32)
                        gu = lax.dot_general(sxq, sw1, DN, preferred_element_type=F32)
                        gu_bf16 = (gu * sxs * ss1).astype(BF16)
                        mid_bf16 = (
                            jax.nn.silu(gu_bf16[:, :i_s].astype(F32)).astype(BF16)
                            * gu_bf16[:, i_s:]
                        )
                        mid_q, mid_s = quantize_block(
                            mid_bf16, axis=1, target_dtype=FP8
                        )
                        mid_s = mid_s.astype(F32)
                    chunks = []
                    for n0 in range(0, h, n_chunk):
                        width = min(n_chunk, h - n0)
                        acc2 = lax.dot_general(
                            mid_q,
                            sw2[:, n0 : n0 + width],
                            DN,
                            preferred_element_type=F32,
                        )
                        out_c = (acc2 * mid_s * ss2[:, n0 : n0 + width]).astype(BF16)
                        chunks.append(out_c)
                    return jnp.concatenate(chunks, axis=1)

                def run_routed():
                    exp_id = jnp.clip(visit_expert[v], 0, experts - 1)
                    w1_e = _fp4_to_exact(w1_hbm[exp_id], act_dtype)
                    w2_e = _fp4_to_exact(w2_hbm[exp_id], act_dtype)
                    s1_e = w1_scale_hbm[exp_id]
                    s2_e = w2_scale_hbm[exp_id]
                    acc1 = None
                    for b in range(b13):
                        part = lax.dot_general(
                            act[:, b * qb1 : (b + 1) * qb1],
                            w1_e[b * qb1 : (b + 1) * qb1, :],
                            DN,
                            preferred_element_type=F32,
                        )
                        scaled = part * s1_e[b : b + 1, :]
                        acc1 = scaled if acc1 is None else acc1 + scaled
                    assert acc1 is not None
                    mid, mid_s = _intermediate_rows(
                        acc1,
                        act_scale,
                        i,
                        w1s=None,
                        act_fn="silu",
                        w1b=None,
                        requantize=has_act_scale,
                    )
                    chunks = []
                    for n0 in range(0, h, n_chunk):
                        width = min(n_chunk, h - n0)
                        acc2 = None
                        for c in range(b2):
                            part2 = lax.dot_general(
                                mid[:, c * bk2 : (c + 1) * bk2],
                                w2_e[c * bk2 : (c + 1) * bk2, n0 : n0 + width],
                                DN,
                                preferred_element_type=F32,
                            )
                            sp = part2 * s2_e[c : c + 1, n0 : n0 + width]
                            acc2 = sp if acc2 is None else acc2 + sp
                        assert acc2 is not None
                        out_c = (acc2 * mid_s if mid_s is not None else acc2).astype(
                            BF16
                        )
                        chunks.append(out_c)
                    return jnp.concatenate(chunks, axis=1)

                tile_out = (
                    lax.cond(is_sh, run_shared, run_routed)
                    if include_shared
                    else run_routed()
                ).reshape(tile_rows, lane_blocks, _LANES)
                cur = lax.dynamic_slice_in_dim(slab, start, tile_rows, axis=0)
                keep = (row_iota < live)[:, None, None]
                merged = jnp.where(keep, tile_out, cur)
                slab = lax.dynamic_update_slice_in_dim(slab, merged, start, axis=0)

                more = (t + 1) * tile_rows < visit_rows[v]
                vn = jnp.where(more, v, v + 1)
                tn = jnp.where(more, t + 1, 0)
                return vn, tn, slab

            _, _, final_slab = lax.while_loop(
                cond_fn, body_fn, (jnp.int32(0), jnp.int32(0), slab_init)
            )
            return final_slab

        return call, scale_rows, token_rows

    monkeypatch.setattr(
        nvfp4_kernel, "build_expert_kernel_nvfp4", cpu_build_expert_kernel_nvfp4
    )
    monkeypatch.setattr(prefill_v2, "combine_step_tokens", lambda _t: 0)


def _cdiv_py(a: int, b: int) -> int:
    return (a + b - 1) // b


def _run_oracles(
    mesh: Mesh,
    cfg: dict[str, int],
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    s1: jax.Array,
    s2: jax.Array,
    sw1: jax.Array,
    ss1: jax.Array,
    sw2: jax.Array,
    ss2: jax.Array,
    gate_w: jax.Array | None,
    tw: jax.Array,
    tids: jax.Array,
    e0: jax.Array,
    *,
    routed_scale: float = ROUTED_SCALE,
    act_mode: mod.ActMode = "w4a16",
) -> tuple[jax.Array, jax.Array]:
    """Return ``(same_mode_oracle, f32_dequant_ref)`` across the 8-rank mesh."""
    h, e, i, i_s, ep = cfg["h"], cfg["e"], cfg["i"], cfg["i_s"], cfg["ep"]
    el = e // ep
    b13 = s1.shape[1]
    b2 = s2.shape[1]
    qb1 = h // b13
    bk2 = i // b2

    def f32_body(x_, w1_, w2_, s1_, s2_, sw1_, ss1_, sw2_, ss2_, tw_, tids_, e0_, gw_):
        hp = lax.Precision.HIGHEST
        tokens = x_.shape[0]
        row_fin = jnp.all(jnp.isfinite(x_), axis=-1, keepdims=True)
        xc = jnp.where(row_fin, x_.astype(F32), 0.0)

        sw1_f = sw1_.astype(F32) * ss1_[None, :]
        sw2_f = sw2_.astype(F32) * ss2_[None, :]
        sgu = jnp.dot(xc, sw1_f, precision=hp)
        sact = jax.nn.silu(sgu[:, :i_s]) * sgu[:, i_s:]
        shared = jnp.dot(sact, sw2_f, precision=hp)
        if gw_ is not None:
            gate = jax.nn.sigmoid(
                jnp.dot(xc, gw_.reshape(h, 1).astype(F32), precision=hp)
            )
            shared = shared * gate

        e_start = e0_.reshape(()) if ep > 1 else jnp.int32(0)

        def exp_step(idx, acc):
            gid = e_start + idx
            hit = (tids_ == gid) & (tw_ != 0)
            coeff = jnp.sum(jnp.where(hit, tw_.astype(F32), 0.0), axis=1, keepdims=True)
            w1_f = _dequant_fp4_single(w1_[idx], s1_[idx])
            w2_f = _dequant_fp4_single(w2_[idx], s2_[idx])
            gu = jnp.dot(xc, w1_f, precision=hp)
            act = jax.nn.silu(gu[:, :i]) * gu[:, i:]
            return acc + jnp.dot(act * coeff, w2_f, precision=hp)

        routed = lax.fori_loop(0, el, exp_step, jnp.zeros((tokens, h), F32))
        return lax.psum(shared + routed_scale * routed, "d")

    def mode_oracle_body(
        x_, w1_, w2_, s1_, s2_, sw1_, ss1_, sw2_, ss2_, tw_, tids_, e0_, gw_
    ):
        tokens = x_.shape[0]
        row_fin = jnp.all(jnp.isfinite(x_), axis=-1, keepdims=True)
        xc_bf16 = jnp.where(row_fin, x_, jnp.zeros_like(x_))
        e_start = e0_.reshape(()) if ep > 1 else jnp.int32(0)

        if act_mode == "w4a16":
            gate_up = xla_quantized_matmul(xc_bf16, sw1_, ss1_)
            sact = (
                jax.nn.silu(gate_up[:, :i_s].astype(F32)).astype(BF16)
                * gate_up[:, i_s:]
            )
            shared_bf16 = xla_quantized_matmul(sact, sw2_, ss2_)
            act_in = xc_bf16
            x_s = None
            dot_dtype = BF16
        else:
            xq3, xs1 = prefill_v2._quantize_rows(x_, FP8)
            act_in = xq3.reshape(tokens, h)
            x_s = xs1[:, None]
            sgu = lax.dot_general(act_in, sw1_, DN, preferred_element_type=F32)
            smid_q, smid_s = _intermediate_rows(
                sgu,
                x_s,
                i_s,
                w1s=ss1_[None, :],
                act_fn="silu",
                w1b=None,
                requantize=True,
            )
            assert smid_s is not None
            sdn = lax.dot_general(smid_q, sw2_, DN, preferred_element_type=F32)
            shared_bf16 = (sdn * smid_s * ss2_[None, :]).astype(BF16)
            dot_dtype = FP8

        if gw_ is None:
            gate = jnp.ones((tokens, 1), F32)
        else:
            gate = jax.nn.sigmoid(
                jnp.dot(
                    xc_bf16, gw_.reshape(h, 1).astype(BF16), preferred_element_type=F32
                )
            )

        def exp_out(idx):
            w1_e = _fp4_to_exact(w1_[idx], dot_dtype)
            w2_e = _fp4_to_exact(w2_[idx], dot_dtype)
            s1_e = s1_[idx, :, 0, :]
            s2_e = s2_[idx, :, 0, :]
            acc1 = None
            for b in range(b13):
                part = lax.dot_general(
                    act_in[:, b * qb1 : (b + 1) * qb1],
                    w1_e[b * qb1 : (b + 1) * qb1, :],
                    DN,
                    preferred_element_type=F32,
                )
                sc = part * s1_e[b : b + 1, :]
                acc1 = sc if acc1 is None else acc1 + sc
            assert acc1 is not None
            mid, mid_s = _intermediate_rows(
                acc1,
                x_s,
                i,
                w1s=None,
                act_fn="silu",
                w1b=None,
                requantize=(act_mode == "w4a8"),
            )
            acc2 = None
            for c in range(b2):
                part2 = lax.dot_general(
                    mid[:, c * bk2 : (c + 1) * bk2],
                    w2_e[c * bk2 : (c + 1) * bk2, :],
                    DN,
                    preferred_element_type=F32,
                )
                sp = part2 * s2_e[c : c + 1, :]
                acc2 = sp if acc2 is None else acc2 + sp
            assert acc2 is not None
            return (acc2 * mid_s if mid_s is not None else acc2).astype(BF16)

        y_all = lax.map(exp_out, jnp.arange(el, dtype=jnp.int32))  # [el, T, H] bf16
        local_ids = tids_ - e_start
        valid = (tw_ != 0) & (local_ids >= 0) & (local_ids < el)
        safe_ids = jnp.where(valid, local_ids, 0)
        tok_iota = jnp.arange(tokens, dtype=jnp.int32)[:, None]
        routed_slots = y_all[safe_ids, tok_iota]  # [T, K, H] bf16
        all_slots = jnp.concatenate(
            [routed_slots, shared_bf16[:, None, :]], axis=1
        ).astype(F32)
        slot_coeff = jnp.concatenate(
            [jnp.where(valid, tw_.astype(F32), 0.0) * routed_scale, gate], axis=1
        )
        rank_bf16 = jnp.sum(all_slots * slot_coeff[..., None], axis=1).astype(BF16)
        return _psum_rows(rank_bf16, "d")

    gw_spec = P() if gate_w is not None else None
    in_specs = (
        P(),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P(),
        P(),
        P("d"),
        gw_spec,
    )
    args = (x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, tw, tids, e0, gate_w)
    ora = jax.jit(
        jax.shard_map(
            mode_oracle_body,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=P(),
            check_vma=False,
        )
    )(*args)
    ref = jax.jit(
        jax.shard_map(
            f32_body, mesh=mesh, in_specs=in_specs, out_specs=P(), check_vma=False
        )
    )(*args)
    return ora, ref


def _run_kernel(
    mesh: Mesh,
    cfg: dict[str, int],
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    s1: jax.Array,
    s2: jax.Array,
    sw1: jax.Array,
    ss1: jax.Array,
    sw2: jax.Array,
    ss2: jax.Array,
    gate_w: jax.Array | None,
    tw: jax.Array,
    tids: jax.Array,
    e0: jax.Array,
    *,
    routed_scale: float = ROUTED_SCALE,
    act_mode: mod.ActMode = "w4a16",
    nb: int = 4,
    w2_n_chunk: int = 512,
    return_per_rank: bool = False,
) -> jax.Array:
    ep = cfg["ep"]
    gw_spec = P() if gate_w is not None else None
    in_specs = (
        P(),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P(),
        P(),
        P("d"),
        gw_spec,
    )

    def body(x_, w1_, w2_, s1_, s2_, sw1_, ss1_, sw2_, ss2_, tw_, tids_, e0_, gw_):
        return mod.fused_moe_prefill_nvfp4(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            sw1_,
            ss1_,
            sw2_,
            ss2_,
            gw_,
            tw_,
            tids_,
            axis_name="d",
            expert_start=e0_.reshape(()) if ep > 1 else None,
            routed_scale=routed_scale,
            act_mode=act_mode,
            nb=nb,
            w2_n_chunk=w2_n_chunk,
        )

    out_spec = P("d") if return_per_rank else P()
    return jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=in_specs, out_specs=out_spec, check_vma=False
        )
    )(x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, tw, tids, e0, gate_w)


# ==============================================================================
# 1. Admission, Refusal & Input Validation Tests (CPU)
# ==============================================================================


@pytest.mark.parametrize(
    "cfg",
    [SMALL_EP2TP4, SMALL_TP8, GLM52_EP2TP4, GLM52_EP2TP4_B256, GLM52_TP8],
)
@pytest.mark.parametrize("act_mode", ["w4a16", "w4a8"])
@pytest.mark.parametrize("nb", [3, 4])
@pytest.mark.parametrize("tokens", [16, 128, 1000, 4096])
def test_admission_positive(
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict[str, int],
    act_mode: mod.ActMode,
    nb: int,
    tokens: int,
):
    _install_mock_tpu7x(monkeypatch)
    x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, tw, tids, gw = _shapes(
        tokens, cfg, scale_ndim=4, with_gate=True
    )
    assert mod.can_use_fused_moe_prefill_nvfp4(
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        tw,
        tids,
        shared_gate_weight=gw,
        act_mode=act_mode,
        nb=nb,
    )
    # 3D routed scale shape [E, B, N] is also admitted.
    _, _, _, s1_3d, s2_3d, *_ = _shapes(tokens, cfg, scale_ndim=3)
    assert mod.can_use_fused_moe_prefill_nvfp4(
        x, w1, w2, s1_3d, s2_3d, sw1, ss1, sw2, ss2, tw, tids, act_mode=act_mode, nb=nb
    )


def test_admission_refusal(monkeypatch: pytest.MonkeyPatch):
    _install_mock_tpu7x(monkeypatch)
    sds = jax.ShapeDtypeStruct

    # EP8 (I=2048) exceeds 64 MiB VMEM at both nb=3 and nb=4 -> cleanly refused.
    for act_mode in ("w4a16", "w4a8"):
        for nb in (3, 4):
            args = _shapes(512, GLM52_EP8)[:11]
            assert not mod.can_use_fused_moe_prefill_nvfp4(
                *args, act_mode=act_mode, nb=nb
            )

    # Sub-tile quantization block sizes (qb1=16, qb1=64, bk2=16, bk2=64) refused.
    for qb1, bk2 in ((16, 512), (64, 512), (512, 16), (512, 64)):
        args = _shapes(128, GLM52_EP2TP4, qb1=qb1, bk2=bk2)[:11]
        assert not mod.can_use_fused_moe_prefill_nvfp4(*args)

    # Misaligned w2_n_chunk refused.
    base = _shapes(128, GLM52_EP2TP4)[:11]
    assert not mod.can_use_fused_moe_prefill_nvfp4(*base, w2_n_chunk=0)
    assert not mod.can_use_fused_moe_prefill_nvfp4(*base, w2_n_chunk=128)
    assert not mod.can_use_fused_moe_prefill_nvfp4(*base, w2_n_chunk=384)
    assert mod.can_use_fused_moe_prefill_nvfp4(*base, w2_n_chunk=1536)

    # Bad dtypes refused.
    bad_w = _shapes(128, GLM52_EP2TP4, w_dtype=FP8)[:11]
    assert not mod.can_use_fused_moe_prefill_nvfp4(*bad_w)
    bad_sw = _shapes(128, GLM52_EP2TP4, sw_dtype=BF16)[:11]
    assert not mod.can_use_fused_moe_prefill_nvfp4(*bad_sw)
    bad_x = (sds((128, 6144), F32), *base[1:])
    assert not mod.can_use_fused_moe_prefill_nvfp4(*bad_x)

    # Pre-v7x chip without FP8xFP8 matmul refused.
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1),
    )
    assert not mod.can_use_fused_moe_prefill_nvfp4(*base)


def test_input_validation_errors():
    x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, tw, tids, _ = _shapes(16, SMALL_EP2TP4)

    def z(s: jax.ShapeDtypeStruct) -> jax.Array:
        return jnp.zeros(s.shape, s.dtype)

    with pytest.raises(ValueError, match="act_mode"):
        mod.fused_moe_prefill_nvfp4(
            z(x),
            z(w1),
            z(w2),
            z(s1),
            z(s2),
            z(sw1),
            z(ss1),
            z(sw2),
            z(ss2),
            None,
            z(tw),
            z(tids),
            axis_name="d",
            act_mode="invalid",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="x_q and x_scale"):
        mod.fused_moe_prefill_nvfp4(
            z(x),
            z(w1),
            z(w2),
            z(s1),
            z(s2),
            z(sw1),
            z(ss1),
            z(sw2),
            z(ss2),
            None,
            z(tw),
            z(tids),
            axis_name="d",
            act_mode="w4a16",
            x_q=jnp.zeros((16, 4, 128), FP8),
            x_scale=jnp.ones((16,), F32),
        )


# ==============================================================================
# 2. 8-Rank CPU End-to-End Pipeline & Mandatory Edge-Case Tests
# ==============================================================================


@pytest.mark.parametrize("act_mode", ["w4a16", "w4a8"])
def test_ep_some_tokens_zero_local_routes(
    monkeypatch: pytest.MonkeyPatch, act_mode: mod.ActMode
):
    """Mandatory case (1a): subset of tokens has ALL routes on the remote EP group."""
    _install_cpu_expert_emulator(monkeypatch)
    mesh = _cpu_or_any_mesh()
    rep = NamedSharding(mesh, P())
    w1, w2, s1, s2, sw1, ss1, sw2, ss2, e0 = _build_small_sharded_weights(
        mesh, SMALL_EP2TP4, seed=11
    )
    x_np, tw_np, ids_np = _make_inputs(
        128, SMALL_EP2TP4, mode="partial_ep_empty_tokens", seed=12
    )
    x = jax.device_put(jnp.asarray(x_np, dtype=BF16), rep)
    tw = jax.device_put(jnp.asarray(tw_np, dtype=BF16), rep)
    tids = jax.device_put(jnp.asarray(ids_np, dtype=jnp.int32), rep)

    per_rank = _run_kernel(
        mesh,
        SMALL_EP2TP4,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        act_mode=act_mode,
        return_per_rank=True,
    )
    ranks_np = np.asarray(per_rank).reshape(N_RANKS, 128, SMALL_EP2TP4["h"])
    assert np.isfinite(ranks_np).all()
    for r in range(1, N_RANKS):
        np.testing.assert_array_equal(ranks_np[0], ranks_np[r])

    ora, ref = _run_oracles(
        mesh,
        SMALL_EP2TP4,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        act_mode=act_mode,
    )
    got = ranks_np[0]
    assert _rel_l2(got, ora) <= 2e-3
    assert _rel_l2(got, ref) <= (0.05 if act_mode == "w4a16" else 0.08)
    assert _cos_sim(got, ref) >= 0.998


@pytest.mark.parametrize("route_mode", ["ep0_all_routes", "ep1_all_routes"])
@pytest.mark.parametrize("act_mode", ["w4a16", "w4a8"])
def test_ep_one_group_gets_all_routes(
    monkeypatch: pytest.MonkeyPatch, route_mode: str, act_mode: mod.ActMode
):
    """Mandatory case (1b): one EP group gets 100% of routes (0 on the other group)."""
    _install_cpu_expert_emulator(monkeypatch)
    mesh = _cpu_or_any_mesh()
    rep = NamedSharding(mesh, P())
    w1, w2, s1, s2, sw1, ss1, sw2, ss2, e0 = _build_small_sharded_weights(
        mesh, SMALL_EP2TP4, seed=21
    )
    x_np, tw_np, ids_np = _make_inputs(128, SMALL_EP2TP4, mode=route_mode, seed=22)
    x = jax.device_put(jnp.asarray(x_np, dtype=BF16), rep)
    tw = jax.device_put(jnp.asarray(tw_np, dtype=BF16), rep)
    tids = jax.device_put(jnp.asarray(ids_np, dtype=jnp.int32), rep)

    got = _run_kernel(
        mesh,
        SMALL_EP2TP4,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        act_mode=act_mode,
    )
    ora, ref = _run_oracles(
        mesh,
        SMALL_EP2TP4,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        act_mode=act_mode,
    )
    assert np.isfinite(np.asarray(got)).all()
    assert _rel_l2(got, ora) <= 2e-3
    assert _rel_l2(got, ref) <= (0.05 if act_mode == "w4a16" else 0.08)


def test_shared_expert_counted_once_and_matches_tp_chain(
    monkeypatch: pytest.MonkeyPatch,
):
    """Mandatory case (2): zero routed weights -> shared expert counted once (not 2x)
    under EP2xTP4, bit-identical in W4A16 to ``_tp_chain``'s ``xla_quantized_matmul``
    shared expert, and invariant to ``routed_scale``."""
    _install_cpu_expert_emulator(monkeypatch)
    mesh = _cpu_or_any_mesh()
    rep = NamedSharding(mesh, P())

    # Subcase A: w1 == 0, w2 == 0 with non-zero routing weights under EP2xTP4.
    w1_z, w2_z, s1_z, s2_z, sw1, ss1, sw2, ss2, e0 = _build_small_sharded_weights(
        mesh, SMALL_EP2TP4, seed=31, zero_routed=True
    )
    x_np, tw_np, ids_np = _make_inputs(128, SMALL_EP2TP4, mode="uniform", seed=32)
    x = jax.device_put(jnp.asarray(x_np, dtype=BF16), rep)
    tw = jax.device_put(jnp.asarray(tw_np, dtype=BF16), rep)
    tids = jax.device_put(jnp.asarray(ids_np, dtype=jnp.int32), rep)

    def tp_chain_shared_once(x_, sw1_, ss1_, sw2_, ss2_):
        gu = xla_quantized_matmul(x_, sw1_, ss1_)
        fs = sw2_.shape[0]
        act = jax.nn.silu(gu[:, :fs].astype(F32)).astype(BF16) * gu[:, fs:]
        sh = xla_quantized_matmul(act, sw2_, ss2_)
        return _psum_rows(sh, "d")

    expected_shared_once = jax.jit(
        jax.shard_map(
            tp_chain_shared_once,
            mesh=mesh,
            in_specs=(P(), P("d"), P("d"), P("d"), P("d")),
            out_specs=P(),
            check_vma=False,
        )
    )(x, sw1, ss1, sw2, ss2)

    got_zero_w = _run_kernel(
        mesh,
        SMALL_EP2TP4,
        x,
        w1_z,
        w2_z,
        s1_z,
        s2_z,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        routed_scale=2.5,
        act_mode="w4a16",
    )
    assert (
        float(np.linalg.norm(np.asarray(expected_shared_once, dtype=np.float32))) > 1.0
    )
    # Bit-exact match to _tp_chain's xla_quantized_matmul shared expert summed once!
    np.testing.assert_array_equal(
        np.asarray(got_zero_w), np.asarray(expected_shared_once)
    )
    # Explicitly verify it is NOT double-counted (2x) across EP2 groups.
    assert _rel_l2(got_zero_w, 2.0 * expected_shared_once.astype(F32)) > 0.45

    # Subcase B: non-zero w1/w2 with topk_weights == 0 and varying routed_scale.
    w1, w2, s1, s2, *_ = _build_small_sharded_weights(mesh, SMALL_EP2TP4, seed=31)
    tw_zero = jax.device_put(jnp.zeros_like(tw), rep)
    got_rs1 = _run_kernel(
        mesh,
        SMALL_EP2TP4,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw_zero,
        tids,
        e0,
        routed_scale=1.0,
        act_mode="w4a16",
    )
    got_rs7 = _run_kernel(
        mesh,
        SMALL_EP2TP4,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw_zero,
        tids,
        e0,
        routed_scale=7.0,
        act_mode="w4a16",
    )
    np.testing.assert_array_equal(np.asarray(got_rs1), np.asarray(expected_shared_once))
    np.testing.assert_array_equal(np.asarray(got_rs7), np.asarray(expected_shared_once))

    # Subcase C: routed_scale scales ONLY routed experts (never shared).
    out_s1 = _run_kernel(
        mesh,
        SMALL_EP2TP4,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        routed_scale=1.0,
        act_mode="w4a16",
    )
    out_s25 = _run_kernel(
        mesh,
        SMALL_EP2TP4,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        routed_scale=2.5,
        act_mode="w4a16",
    )
    sh_f = np.asarray(expected_shared_once, dtype=np.float32)
    routed_1 = np.asarray(out_s1, dtype=np.float32) - sh_f
    routed_25 = np.asarray(out_s25, dtype=np.float32) - sh_f
    assert _rel_l2(routed_25, 2.5 * routed_1) < 0.02


@pytest.mark.parametrize("cfg", [SMALL_EP2TP4, SMALL_TP8])
@pytest.mark.parametrize("tokens", [128, 1000])
@pytest.mark.parametrize("act_mode", ["w4a16", "w4a8"])
def test_token_counts_128_and_1000(
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict[str, int],
    tokens: int,
    act_mode: mod.ActMode,
):
    """Mandatory case (3): T = 128 and T = 1000 across EP2xTP4 & TP8."""
    _install_cpu_expert_emulator(monkeypatch)
    mesh = _cpu_or_any_mesh()
    rep = NamedSharding(mesh, P())
    w1, w2, s1, s2, sw1, ss1, sw2, ss2, e0 = _build_small_sharded_weights(
        mesh, cfg, seed=41
    )
    x_np, tw_np, ids_np = _make_inputs(tokens, cfg, mode="uniform", seed=42 + tokens)
    x = jax.device_put(jnp.asarray(x_np, dtype=BF16), rep)
    tw = jax.device_put(jnp.asarray(tw_np, dtype=BF16), rep)
    tids = jax.device_put(jnp.asarray(ids_np, dtype=jnp.int32), rep)

    got = _run_kernel(
        mesh,
        cfg,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        act_mode=act_mode,
    )
    ora, ref = _run_oracles(
        mesh,
        cfg,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        act_mode=act_mode,
    )
    assert got.shape == (tokens, cfg["h"])
    assert np.isfinite(np.asarray(got)).all()
    assert _rel_l2(got, ora) <= 2e-3
    assert _rel_l2(got, ref) <= (0.05 if act_mode == "w4a16" else 0.08)
    assert _cos_sim(got, ref) >= 0.998


@pytest.mark.parametrize("act_mode", ["w4a16", "w4a8"])
def test_nan_padded_rows_and_gated_shared_expert(
    monkeypatch: pytest.MonkeyPatch, act_mode: mod.ActMode
):
    """NaN-poisoned padded rows stay finite and gated shared expert matches oracle."""
    _install_cpu_expert_emulator(monkeypatch)
    mesh = _cpu_or_any_mesh()
    rep = NamedSharding(mesh, P())
    w1, w2, s1, s2, sw1, ss1, sw2, ss2, e0 = _build_small_sharded_weights(
        mesh, SMALL_EP2TP4, seed=51
    )
    tokens, pad = 100, 12
    valid = tokens - pad
    x_clean_np, tw_np, ids_np = _make_inputs(
        tokens,
        SMALL_EP2TP4,
        mode="uniform",
        seed=52,
        pad_tokens=pad,
        poison_pad_nan=False,
    )
    x_clean_np[valid:] = 0.0
    x_nan_np, _, _ = _make_inputs(
        tokens,
        SMALL_EP2TP4,
        mode="uniform",
        seed=52,
        pad_tokens=pad,
        poison_pad_nan=True,
    )

    rng = np.random.default_rng(53)
    gw_np = (rng.standard_normal((SMALL_EP2TP4["h"], 1)) * 0.1).astype(np.float32)

    x_clean = jax.device_put(jnp.asarray(x_clean_np, dtype=BF16), rep)
    x_nan = jax.device_put(jnp.asarray(x_nan_np, dtype=BF16), rep)
    tw = jax.device_put(jnp.asarray(tw_np, dtype=BF16), rep)
    tids = jax.device_put(jnp.asarray(ids_np, dtype=jnp.int32), rep)
    gw = jax.device_put(jnp.asarray(gw_np, dtype=BF16), rep)

    got_nan = _run_kernel(
        mesh,
        SMALL_EP2TP4,
        x_nan,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        gw,
        tw,
        tids,
        e0,
        act_mode=act_mode,
    )
    got_clean = _run_kernel(
        mesh,
        SMALL_EP2TP4,
        x_clean,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        gw,
        tw,
        tids,
        e0,
        act_mode=act_mode,
    )
    ora, ref = _run_oracles(
        mesh,
        SMALL_EP2TP4,
        x_nan,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        gw,
        tw,
        tids,
        e0,
        act_mode=act_mode,
    )
    got_np = np.asarray(got_nan)
    assert np.isfinite(got_np).all()
    np.testing.assert_array_equal(got_np[:valid], np.asarray(got_clean)[:valid])
    np.testing.assert_array_equal(got_np[valid:], np.zeros_like(got_np[valid:]))
    assert _rel_l2(got_nan, ora) <= 2e-3
    assert _rel_l2(got_nan, ref) <= (0.05 if act_mode == "w4a16" else 0.08)


def test_w2_n_chunk_and_prequantized_invariance(monkeypatch: pytest.MonkeyPatch):
    """Chunk width (256 vs 512) and pre-quantized (x_q, x_scale) are bit-identical."""
    _install_cpu_expert_emulator(monkeypatch)
    mesh = _cpu_or_any_mesh()
    rep = NamedSharding(mesh, P())
    w1, w2, s1, s2, sw1, ss1, sw2, ss2, e0 = _build_small_sharded_weights(
        mesh, SMALL_EP2TP4, seed=61
    )
    x_np, tw_np, ids_np = _make_inputs(64, SMALL_EP2TP4, mode="uniform", seed=62)
    x = jax.device_put(jnp.asarray(x_np, dtype=BF16), rep)
    tw = jax.device_put(jnp.asarray(tw_np, dtype=BF16), rep)
    tids = jax.device_put(jnp.asarray(ids_np, dtype=jnp.int32), rep)

    for act_mode in ("w4a16", "w4a8"):
        c256 = _run_kernel(
            mesh,
            SMALL_EP2TP4,
            x,
            w1,
            w2,
            s1,
            s2,
            sw1,
            ss1,
            sw2,
            ss2,
            None,
            tw,
            tids,
            e0,
            act_mode=act_mode,
            w2_n_chunk=256,
        )
        c512 = _run_kernel(
            mesh,
            SMALL_EP2TP4,
            x,
            w1,
            w2,
            s1,
            s2,
            sw1,
            ss1,
            sw2,
            ss2,
            None,
            tw,
            tids,
            e0,
            act_mode=act_mode,
            w2_n_chunk=512,
        )
        np.testing.assert_array_equal(np.asarray(c256), np.asarray(c512))

    # Pre-quantized (x_q, x_scale) in w4a8 matches internal _quantize_rows bit-for-bit.
    def body_preq(x_, w1_, w2_, s1_, s2_, sw1_, ss1_, sw2_, ss2_, tw_, tids_, e0_):
        xq, xs = prefill_v2._quantize_rows(x_, FP8)
        return mod.fused_moe_prefill_nvfp4_stage_a(
            None,
            w1_,
            w2_,
            s1_,
            s2_,
            sw1_,
            ss1_,
            sw2_,
            ss2_,
            None,
            tw_,
            tids_,
            axis_name="d",
            expert_start=e0_.reshape(()),
            routed_scale=ROUTED_SCALE,
            act_mode="w4a8",
            w2_n_chunk=512,
            x_q=xq,
            x_scale=xs,
        )

    got_preq = jax.jit(
        jax.shard_map(
            body_preq,
            mesh=mesh,
            in_specs=(
                P(),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P(),
                P(),
                P("d"),
            ),
            out_specs=P(),
            check_vma=False,
        )
    )(x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, tw, tids, e0)
    got_internal = _run_kernel(
        mesh,
        SMALL_EP2TP4,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        act_mode="w4a8",
        w2_n_chunk=512,
    )
    np.testing.assert_array_equal(np.asarray(got_preq), np.asarray(got_internal))


# ==============================================================================
# 3. Real TPU Mosaic / StableHLO Lowering Tests on CPU (Zero GLM-5.2 Allocation)
# ==============================================================================


@pytest.mark.parametrize("cfg", [GLM52_EP2TP4, GLM52_TP8])
@pytest.mark.parametrize("act_mode", ["w4a16", "w4a8"])
@pytest.mark.parametrize("tokens", [128, 1000, 4096])
def test_tpu_mosaic_lowering_glm52_abstract(
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict[str, int],
    act_mode: mod.ActMode,
    tokens: int,
):
    """Lower full GLM-5.2 shapes to TPU Mosaic MLIR using ShapeDtypeStructs only."""
    _install_mock_tpu7x(monkeypatch)
    mesh = _cpu_or_any_mesh()
    rep = NamedSharding(mesh, P())
    shd = NamedSharding(mesh, P("d"))
    sds = jax.ShapeDtypeStruct

    h, e, i, i_s, topk, qb1, bk2, ep = (
        cfg["h"],
        cfg["e"],
        cfg["i"],
        cfg["i_s"],
        cfg["topk"],
        cfg["qb1"],
        cfg["bk2"],
        cfg["ep"],
    )
    el = e // ep
    b13 = h // qb1
    b2 = i // bk2

    x = sds((tokens, h), BF16, sharding=rep)
    w1 = sds((N_RANKS * el, h, 2 * i), FP4, sharding=shd)
    w2 = sds((N_RANKS * el, i, h), FP4, sharding=shd)
    s1 = sds((N_RANKS * el, b13, 1, 2 * i), F32, sharding=shd)
    s2 = sds((N_RANKS * el, b2, 1, h), F32, sharding=shd)
    sw1 = sds((N_RANKS * h, 2 * i_s), FP8, sharding=shd)
    ss1 = sds((N_RANKS * 2 * i_s,), F32, sharding=shd)
    sw2 = sds((N_RANKS * i_s, h), FP8, sharding=shd)
    ss2 = sds((N_RANKS * h,), F32, sharding=shd)
    tw = sds((tokens, topk), BF16, sharding=rep)
    tids = sds((tokens, topk), jnp.int32, sharding=rep)
    e0 = sds((N_RANKS, 1), jnp.int32, sharding=shd)

    def body(x_, w1_, w2_, s1_, s2_, sw1_, ss1_, sw2_, ss2_, tw_, tids_, e0_):
        return mod.fused_moe_prefill_nvfp4_stage_a(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            sw1_,
            ss1_,
            sw2_,
            ss2_,
            None,
            tw_,
            tids_,
            axis_name="d",
            expert_start=e0_.reshape(()) if ep > 1 else None,
            routed_scale=ROUTED_SCALE,
            act_mode=act_mode,
        )

    fn = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(
                P(),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P(),
                P(),
                P("d"),
            ),
            out_specs=P(),
            check_vma=False,
        )
    )
    tr = fn.trace(x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, tw, tids, e0)
    hlo = tr.lower(lowering_platforms=("tpu",)).as_text()
    assert tr.out_info.shape == (tokens, h)
    assert tr.out_info.dtype == BF16
    assert f"fused_moe_prefill_nvfp4-t_{tokens}" in hlo
    if tokens == 4096:
        assert "_flat" in hlo
        if cfg["i"] > 256:
            assert f"-m_256-{act_mode}-nb4-nc2048" in hlo
            assert "-r32" not in hlo
        else:
            assert f"-m_256-{act_mode}-nb4-nc2048-r32" in hlo


@pytest.mark.parametrize("cfg", [SMALL_EP2TP4, SMALL_TP8])
@pytest.mark.parametrize("act_mode", ["w4a16", "w4a8"])
def test_cpu_routed_only_and_combine_chunk_tokens(
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict[str, int],
    act_mode: mod.ActMode,
):
    _install_cpu_expert_emulator(monkeypatch)
    mesh = _cpu_or_any_mesh()
    rep = NamedSharding(mesh, P())
    w1, w2, s1, s2, sw1, ss1, sw2, ss2, e0 = _build_small_sharded_weights(
        mesh, cfg, seed=131
    )
    tokens = 128
    route_mode = "ep0_all_routes" if cfg["ep"] > 1 else "uniform"
    x_np, tw_np, ids_np = _make_inputs(
        tokens, cfg, mode=route_mode, seed=132, pad_tokens=8, poison_pad_nan=True
    )
    x = jax.device_put(jnp.asarray(x_np, dtype=BF16), rep)
    tw = jax.device_put(jnp.asarray(tw_np, dtype=BF16), rep)
    tids = jax.device_put(jnp.asarray(ids_np, dtype=jnp.int32), rep)

    # Zero-shared weights run with include_shared=True must match include_shared=False.
    z_sw1 = jnp.zeros_like(sw1)
    z_sw2 = jnp.zeros_like(sw2)

    def run_opt(*, inc_sh: bool, cchunk: int):
        ep = cfg["ep"]
        in_sp = (
            (
                P(),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P(),
                P(),
                P("d"),
            )
            if inc_sh
            else (P(), P("d"), P("d"), P("d"), P("d"), P(), P(), P("d"))
        )

        def body(*bargs):
            if inc_sh:
                x_, w1_, w2_, s1_, s2_, sw1_, ss1_, sw2_, ss2_, tw_, tids_, e0_ = bargs
            else:
                x_, w1_, w2_, s1_, s2_, tw_, tids_, e0_ = bargs
                sw1_ = ss1_ = sw2_ = ss2_ = None
            return mod.fused_moe_prefill_nvfp4(
                x_,
                w1_,
                w2_,
                s1_,
                s2_,
                sw1_,
                ss1_,
                sw2_,
                ss2_,
                None,
                tw_,
                tids_,
                axis_name="d",
                expert_start=e0_.reshape(()) if ep > 1 else None,
                routed_scale=ROUTED_SCALE,
                act_mode=act_mode,
                nb=2,
                include_shared=inc_sh,
                combine_chunk_tokens=cchunk,
                w2_n_chunk=512,
            )

        fn = jax.jit(
            jax.shard_map(
                body, mesh=mesh, in_specs=in_sp, out_specs=P(), check_vma=False
            )
        )
        if inc_sh:
            return fn(x, w1, w2, s1, s2, z_sw1, ss1, z_sw2, ss2, tw, tids, e0)
        return fn(x, w1, w2, s1, s2, tw, tids, e0)

    got_routed = run_opt(inc_sh=False, cchunk=2048)
    got_chunk512 = run_opt(inc_sh=False, cchunk=512)
    got_chunk64 = run_opt(inc_sh=False, cchunk=64)
    got_zero_sh = run_opt(inc_sh=True, cchunk=2048)
    assert np.isfinite(np.asarray(got_routed)).all()
    assert _rel_l2(got_routed, got_zero_sh) <= 1e-6
    assert _rel_l2(got_routed, got_chunk512) <= 1e-6
    # Chunking `[128, H]` into two `[64, H]` BF16 reduce-scatter passes across
    # 8 ranks can shift boundary values by 1 BF16 ULP (~1.8e-3 rel_l2).
    assert _rel_l2(got_routed, got_chunk64) <= 3e-3


def test_glm52_4k_sweep_tpu_hlo_lowerings(monkeypatch: pytest.MonkeyPatch):
    _install_mock_tpu7x(monkeypatch)
    mesh = _cpu_or_any_mesh()
    shd = NamedSharding(mesh, P("d"))
    rep = NamedSharding(mesh, P())
    sds = jax.ShapeDtypeStruct
    tokens = 4096
    cfg = GLM52_EP2TP4
    h, e, i, i_s, ep, topk = (
        cfg["h"],
        cfg["e"],
        cfg["i"],
        cfg["i_s"],
        cfg["ep"],
        cfg["topk"],
    )
    el = e // ep

    x = sds((tokens, h), BF16, sharding=rep)
    w1 = sds((N_RANKS * el, h, 2 * i), FP4, sharding=shd)
    w2 = sds((N_RANKS * el, i, h), FP4, sharding=shd)
    s1 = sds((N_RANKS * el, 12, 1, 2 * i), F32, sharding=shd)
    s2 = sds((N_RANKS * el, 1, 1, h), F32, sharding=shd)
    sw1 = sds((N_RANKS * h, 2 * i_s), FP8, sharding=shd)
    ss1 = sds((N_RANKS * 2 * i_s,), F32, sharding=shd)
    sw2 = sds((N_RANKS * i_s, h), FP8, sharding=shd)
    ss2 = sds((N_RANKS * h,), F32, sharding=shd)
    tw = sds((tokens, topk), BF16, sharding=rep)
    tids = sds((tokens, topk), jnp.int32, sharding=rep)
    e0 = sds((N_RANKS, 1), jnp.int32, sharding=shd)

    for tile_min, nb, tall_r, inc_sh, cchunk in (
        (8192, 4, 32, True, 2048),
        (8192, 2, 32, True, 2048),
        (4096, 4, None, True, 2048),
        (8192, 4, 32, False, 4096),
    ):
        assert mod.can_use_fused_moe_prefill_nvfp4(
            sds((tokens, h), BF16),
            sds((el, h, 2 * i), FP4),
            sds((el, i, h), FP4),
            sds((el, 12, 1, 2 * i), F32),
            sds((el, 1, 1, h), F32),
            sds((h, 2 * i_s), FP8) if inc_sh else None,
            sds((2 * i_s,), F32) if inc_sh else None,
            sds((i_s, h), FP8) if inc_sh else None,
            sds((h,), F32) if inc_sh else None,
            sds((tokens, topk), BF16),
            sds((tokens, topk), jnp.int32),
            nb=nb,
            include_shared=inc_sh,
        )

        def body(
            x_,
            w1_,
            w2_,
            s1_,
            s2_,
            sw1_,
            ss1_,
            sw2_,
            ss2_,
            tw_,
            tids_,
            e0_,
            _tm=tile_min,
            _nb=nb,
            _tr=tall_r,
            _is=inc_sh,
            _cc=cchunk,
        ):
            return mod.fused_moe_prefill_nvfp4(
                x_,
                w1_,
                w2_,
                s1_,
                s2_,
                sw1_ if _is else None,
                ss1_ if _is else None,
                sw2_ if _is else None,
                ss2_ if _is else None,
                None,
                tw_,
                tids_,
                axis_name="d",
                expert_start=e0_.reshape(()),
                routed_scale=ROUTED_SCALE,
                act_mode="w4a8",
                nb=_nb,
                tile_min=_tm,
                tall_rung_rows=_tr,
                include_shared=_is,
                combine_chunk_tokens=_cc,
            )

        fn = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(
                    P(),
                    P("d"),
                    P("d"),
                    P("d"),
                    P("d"),
                    P("d"),
                    P("d"),
                    P("d"),
                    P("d"),
                    P(),
                    P(),
                    P("d"),
                ),
                out_specs=P(),
                check_vma=False,
            )
        )
        tr = fn.trace(x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, tw, tids, e0)
        hlo = tr.lower(lowering_platforms=("tpu",)).as_text()
        expect_m = 128 if tile_min == 8192 else 256
        assert f"-m_{expect_m}-" in hlo
        if not inc_sh:
            assert "-routed" in hlo


# ==============================================================================
# 4. 8-Core TPU Hardware Execution Tests (Skipped on CPU)
# ==============================================================================


@pytest.mark.parametrize("cfg", [SMALL_EP2TP4, SMALL_TP8])
@pytest.mark.parametrize("act_mode", ["w4a16", "w4a8"])
@pytest.mark.parametrize(
    "tokens,route_mode",
    [
        (128, "uniform"),
        (1000, "uniform"),
        (128, "partial_ep_empty_tokens"),
        (128, "ep0_all_routes"),
    ],
)
def test_tpu_hardware_correctness(
    cfg: dict[str, int],
    act_mode: mod.ActMode,
    tokens: int,
    route_mode: str,
):
    mesh = _require_tpu_mesh()
    if cfg["ep"] == 1 and route_mode != "uniform":
        pytest.skip("EP routing edge cases apply to EP2xTP4")
    rep = NamedSharding(mesh, P())
    w1, w2, s1, s2, sw1, ss1, sw2, ss2, e0 = _build_small_sharded_weights(
        mesh, cfg, seed=91
    )
    x_np, tw_np, ids_np = _make_inputs(
        tokens,
        cfg,
        mode=route_mode,
        seed=92 + tokens,
        pad_tokens=8,
        poison_pad_nan=True,
    )
    x = jax.device_put(jnp.asarray(x_np, dtype=BF16), rep)
    tw = jax.device_put(jnp.asarray(tw_np, dtype=BF16), rep)
    tids = jax.device_put(jnp.asarray(ids_np, dtype=jnp.int32), rep)

    got = _run_kernel(
        mesh,
        cfg,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        act_mode=act_mode,
    )
    ora, ref = _run_oracles(
        mesh,
        cfg,
        x,
        w1,
        w2,
        s1,
        s2,
        sw1,
        ss1,
        sw2,
        ss2,
        None,
        tw,
        tids,
        e0,
        act_mode=act_mode,
    )
    assert np.isfinite(np.asarray(got)).all()
    assert abs(_rel_l2(got, ref) - _rel_l2(ora, ref)) <= 0.005
    assert _rel_l2(got, ora) <= 0.025
    assert _rel_l2(got, ref) <= (0.05 if act_mode == "w4a16" else 0.08)
    assert _cos_sim(got, ref) >= 0.998


# ==============================================================================
# 5. Code-Size-Aware Rung Rule, Env Flag, & moe_tp_decode_nvfp4 Adapter Tests
# ==============================================================================


def test_tall_rung_rows_for_rule():
    """Verify code-size-aware rung step selection from (tile_rows, i)."""
    assert mod.STAGE_A_NVFP4["tall_rung_rows"] == "auto"
    assert mod.tall_rung_rows_for(64, 256) is None
    assert mod.tall_rung_rows_for(64, 512) is None
    assert mod.tall_rung_rows_for(128, 256) is None
    assert mod.tall_rung_rows_for(128, 512) is None
    # At tile_rows=256: fits in 8 MiB IMEM when tile_rows * i <= 65_536 (TP8 I=256)
    assert mod.tall_rung_rows_for(256, 128) == 32
    assert mod.tall_rung_rows_for(256, 256) == 32
    # Exceeds 8 MiB IMEM when tile_rows * i > 65_536 (EP2xTP4 I=512 -> 4 rungs)
    assert mod.tall_rung_rows_for(256, 384) is None
    assert mod.tall_rung_rows_for(256, 512) is None
    assert mod.tall_rung_rows_for(256, 2048) is None


def test_fused_moe_prefill_nvfp4_env_flag_parsing(monkeypatch: pytest.MonkeyPatch):
    import vllm_torchtpu.envs as envs

    envs_mod: Any = envs
    monkeypatch.delenv("FUSED_MOE_PREFILL_NVFP4_KERNEL", raising=False)
    assert envs_mod.FUSED_MOE_PREFILL_NVFP4_KERNEL is False
    assert envs.is_set("FUSED_MOE_PREFILL_NVFP4_KERNEL") is False

    for raw in ("1", "true", "True", "yes", "on"):
        monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_KERNEL", raw)
        assert envs_mod.FUSED_MOE_PREFILL_NVFP4_KERNEL is True
        assert envs.is_set("FUSED_MOE_PREFILL_NVFP4_KERNEL") is True

    for raw in ("0", "false", "False", "no", "off"):
        monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_KERNEL", raw)
        assert envs_mod.FUSED_MOE_PREFILL_NVFP4_KERNEL is False
        assert envs.is_set("FUSED_MOE_PREFILL_NVFP4_KERNEL") is True

    monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_KERNEL", "")
    assert envs_mod.FUSED_MOE_PREFILL_NVFP4_KERNEL is False
    assert envs.is_set("FUSED_MOE_PREFILL_NVFP4_KERNEL") is False

    monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_KERNEL", "invalid")
    with pytest.raises(ValueError, match="FUSED_MOE_PREFILL_NVFP4_KERNEL"):
        _ = envs_mod.FUSED_MOE_PREFILL_NVFP4_KERNEL

    monkeypatch.delenv("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS", raising=False)
    assert envs_mod.FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS == 2048
    assert envs.is_set("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS") is False

    monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS", "512")
    assert envs_mod.FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS == 512
    assert envs.is_set("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS") is True

    monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS", "")
    assert envs_mod.FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS == 2048
    assert envs.is_set("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS") is False

    monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS", "invalid")
    with pytest.raises(ValueError, match="FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS"):
        _ = envs_mod.FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS


def _load_nvfp4_adapter_for_test(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load ``layers/adapter/moe_tp_decode_nvfp4.py`` with lightweight stubs."""
    import importlib.util
    import pathlib
    import sys
    import types
    from types import SimpleNamespace

    import vllm_torchtpu.envs as envs

    def _put_mod(name: str, **attrs: Any) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    class _FakeTensor:
        def __init__(self, shape: tuple[int, ...], dtype: str = "float32") -> None:
            self.shape = tuple(shape)
            self.dtype = dtype
            self.device = "cpu"

    class _FakeModule:
        pass

    fake_torch = _put_mod(
        "torch",
        Tensor=_FakeTensor,
        device=object,
        float32="float32",
        bfloat16="bfloat16",
        uint8="uint8",
        int32="int32",
        float8_e4m3fn="float8_e4m3fn",
        float4_e2m1fn_x2="float4_e2m1fn_x2",
        empty_like=lambda x: x,
        nn=SimpleNamespace(Module=_FakeModule),
    )

    def _fake_jax_op(_name, fn, **_kwargs):
        return SimpleNamespace(fn=fn, register_fake=lambda _f: None)

    _put_mod("torch_tpu")
    _put_mod(
        "torch_tpu._internal",
        pallas=SimpleNamespace(
            jax_op=_fake_jax_op,
            pallas=SimpleNamespace(
                TORCH_TO_JAX_DTYPE_MAP={
                    "float32": jnp.float32,
                    "bfloat16": jnp.bfloat16,
                    "uint8": jnp.uint8,
                    "int32": jnp.int32,
                    "float8_e4m3fn": jnp.float8_e4m3fn,
                    "float4_e2m1fn_x2": jnp.float4_e2m1fn,
                }
            ),
        ),
    )

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
    _put_mod("vllm.model_executor.layers.fused_moe", __path__=[])
    _put_mod("vllm.model_executor.layers.fused_moe.router", __path__=[])
    _put_mod(
        "vllm.model_executor.layers.fused_moe.router.zero_expert_router",
        ZeroExpertRouter=type("ZeroExpertRouter", (), {}),
    )
    _put_mod("vllm.model_executor.layers.fused_moe.runner", __path__=[])
    _put_mod(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner",
        MoERunner=type("MoERunner", (_FakeModule,), {}),
    )

    def _route(x, scores, bias, is_pad, st):
        bias_f = bias.astype(F32)
        _, ids = lax.top_k(scores + bias_f[None, :], st.topk)
        w = jnp.take_along_axis(scores, ids, axis=-1)
        total = jnp.sum(w, axis=-1, keepdims=True)
        total = jnp.where(jnp.isnan(total), total, jnp.maximum(total, 1e-20))
        w = (w / total).astype(x.dtype)
        w = jnp.where(is_pad[:, None], jnp.zeros_like(w), w)
        return w, ids

    def _tp_chain_w4a16(
        x, scores, bias, is_pad, w13, w2, s13, s2, sw13, ss13, sw2, ss2, st, e0=None
    ):
        w, ids = _route(x, scores, bias, is_pad, st)
        row_fin = jnp.all(jnp.isfinite(x), axis=-1, keepdims=True)
        xc = jnp.where(row_fin, x, jnp.zeros_like(x))
        gu_s = xla_quantized_matmul(xc, sw13, ss13)
        fs = sw2.shape[0]
        sact = jax.nn.silu(gu_s[:, :fs].astype(F32)).astype(x.dtype) * gu_s[:, fs:]
        shared = xla_quantized_matmul(sact, sw2, ss2)

        el, h_c, two_i = w13.shape
        i_c = two_i // 2
        b13 = s13.shape[1]
        b2 = s2.shape[1]
        qb1 = h_c // b13
        bk2 = i_c // b2
        e_start = (
            jnp.int32(0) if (not st.use_ep or e0 is None) else e0.astype(jnp.int32)
        )
        local_ids = ids.astype(jnp.int32) - e_start
        valid = (w != 0) & (local_ids >= 0) & (local_ids < el)

        def exp_out(idx):
            w1_e = _fp4_to_exact(w13[idx], BF16)
            w2_e = _fp4_to_exact(w2[idx], BF16)
            s1_e = s13[idx, :, 0, :]
            s2_e = s2[idx, :, 0, :]
            acc1 = None
            for b in range(b13):
                p1 = lax.dot_general(
                    xc[:, b * qb1 : (b + 1) * qb1],
                    w1_e[b * qb1 : (b + 1) * qb1, :],
                    DN,
                    preferred_element_type=F32,
                )
                sc1 = p1 * s1_e[b : b + 1, :]
                acc1 = sc1 if acc1 is None else acc1 + sc1
            assert acc1 is not None
            mid, _ = _intermediate_rows(
                acc1, None, i_c, w1s=None, act_fn="silu", w1b=None, requantize=False
            )
            acc2 = None
            for c in range(b2):
                p2 = lax.dot_general(
                    mid[:, c * bk2 : (c + 1) * bk2],
                    w2_e[c * bk2 : (c + 1) * bk2, :],
                    DN,
                    preferred_element_type=F32,
                )
                sc2 = p2 * s2_e[c : c + 1, :]
                acc2 = sc2 if acc2 is None else acc2 + sc2
            assert acc2 is not None
            return acc2.astype(BF16)

        y_all = lax.map(exp_out, jnp.arange(el, dtype=jnp.int32))
        safe_ids = jnp.where(valid, local_ids, 0)
        tok_iota = jnp.arange(x.shape[0], dtype=jnp.int32)[:, None]
        picked = y_all[safe_ids, tok_iota]
        w_eff = jnp.where(valid, w, jnp.zeros_like(w))
        routed = jnp.sum(picked * w_eff[:, :, None], axis=1).astype(x.dtype)
        return shared + routed * st.routed_scale

    src_root = pathlib.Path(envs.__file__).resolve().parent
    for pkg_name, rel in (
        ("vllm_torchtpu.layers", "layers"),
        ("vllm_torchtpu.layers.core", "layers/core"),
        ("vllm_torchtpu.layers.adapter", "layers/adapter"),
        ("vllm_torchtpu.layers.adapter.quantization", "layers/adapter/quantization"),
    ):
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(src_root / rel)]
        monkeypatch.setitem(sys.modules, pkg_name, pkg)

    _put_mod(
        "vllm_torchtpu.layers.core.fused_moe_gmm",
        fused_moe_func=lambda *_a, **_k: None,
        resolve_onehot_permute_threshold=lambda: 2048,
    )
    _put_mod(
        "vllm.distributed",
        parallel_state=SimpleNamespace(
            get_tp_group=lambda: SimpleNamespace(
                ranks=tuple(range(N_RANKS)), world_size=1, cpu_group=None
            )
        ),
    )
    _put_mod("vllm_torchtpu.distributed", __path__=[])
    _put_mod(
        "vllm_torchtpu.distributed.ep_mesh",
        build_ep_mesh=lambda: None,
        ep_mesh_index=lambda: 0,
        get_ep_group=lambda: None,
    )
    _put_mod(
        "vllm_torchtpu.kernels.router_topk",
        select=lambda *_a, **_k: None,
        select_bias=lambda *_a, **_k: None,
    )
    moe_routing_stub = _put_mod(
        "vllm_torchtpu.layers.adapter.moe_routing",
        get_experts_start=lambda layer: int(getattr(layer, "_test_e0", 0)),
        _apply_scoring_fn=lambda _fn, logits: logits,
        _SIMULATION_STRATEGY=None,
        _ROUTER_TOPK="rowmax",
        _ROUTER_TOPK_BIAS=False,
    )
    token_padding_stub = _put_mod(
        "vllm_torchtpu.layers.adapter.token_padding",
        _padding_state=None,
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
    _put_mod(
        "vllm_torchtpu.layers.adapter.fused_moe_ep",
        fused_moe_ep_supported=lambda _qm: False,
    )
    _put_mod(
        "vllm_torchtpu.layers.adapter.pipelined_fused_moe",
        enable_pipelined_collective_and_compute=lambda: False,
    )
    nvfp4_qm_cls = type(
        "VllmNvfp4MoEMethod",
        (),
        {"_tpu_activation_str": "silu", "_use_adaptive_fused_moe": False},
    )
    _put_mod(
        "vllm_torchtpu.layers.adapter.quantization.nvfp4",
        VllmNvfp4MoEMethod=nvfp4_qm_cls,
    )

    fp8_spec = importlib.util.spec_from_file_location(
        "vllm_torchtpu.layers.adapter.moe_tp_decode",
        src_root / "layers/adapter/moe_tp_decode.py",
    )
    assert fp8_spec is not None and fp8_spec.loader is not None
    fp8_decode_mod = importlib.util.module_from_spec(fp8_spec)
    monkeypatch.setitem(
        sys.modules, "vllm_torchtpu.layers.adapter.moe_tp_decode", fp8_decode_mod
    )
    monkeypatch.setattr(
        sys.modules["vllm_torchtpu.layers.adapter"],
        "moe_tp_decode",
        fp8_decode_mod,
        raising=False,
    )
    fp8_spec.loader.exec_module(fp8_decode_mod)
    monkeypatch.setattr(fp8_decode_mod, "_route", _route)
    monkeypatch.setattr(fp8_decode_mod, "_tp_chain", _tp_chain_w4a16)
    monkeypatch.setattr(
        fp8_decode_mod, "_shared_slice", lambda runner: runner._test_shared
    )
    monkeypatch.setattr(fp8_decode_mod, "moe_tp_decode_supported", lambda _qm: False)

    spec = importlib.util.spec_from_file_location(
        "vllm_torchtpu.layers.adapter.moe_tp_decode_nvfp4",
        src_root / "layers/adapter/moe_tp_decode_nvfp4.py",
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(
        sys.modules, "vllm_torchtpu.layers.adapter.moe_tp_decode_nvfp4", adapter
    )
    spec.loader.exec_module(adapter)
    return SimpleNamespace(
        adapter=adapter,
        torch=fake_torch,
        VllmNvfp4MoEMethod=nvfp4_qm_cls,
        fp8_decode=fp8_decode_mod,
    )


def _make_stub_runner(rt: Any, cfg: dict[str, int]) -> Any:
    from types import SimpleNamespace

    t = rt.torch
    h, e, i, i_s, topk, qb1, bk2, ep = (
        cfg["h"],
        cfg["e"],
        cfg["i"],
        cfg["i_s"],
        cfg["topk"],
        cfg["qb1"],
        cfg["bk2"],
        cfg["ep"],
    )
    el = e // ep
    b13 = h // qb1
    b2 = i // bk2
    qm = rt.VllmNvfp4MoEMethod()
    layer = SimpleNamespace(
        quant_method=qm,
        # Packed x2 trailing dim matches torch.float4_e2m1fn_x2 layout in _sds()
        w13_weight=t.Tensor((el, h, i), dtype=t.float4_e2m1fn_x2),
        w2_weight=t.Tensor((el, i, h // 2), dtype=t.float4_e2m1fn_x2),
        w13_weight_scale=t.Tensor((el, b13, 1, 2 * i), dtype=t.float32),
        w2_weight_scale=t.Tensor((el, b2, 1, h), dtype=t.float32),
        e_score_correction_bias=t.Tensor((e,), dtype=t.float32),
        global_num_experts=e,
        expert_placement_strategy="linear",
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        renormalize=True,
        custom_routing_function=None,
        routed_scaling_factor=1.0,
        _test_e0=el if ep > 1 else 0,
    )
    shared = (
        t.Tensor((h, 2 * i_s), dtype=t.float8_e4m3fn),
        t.Tensor((2 * i_s,), dtype=t.float32),
        t.Tensor((i_s, h), dtype=t.float8_e4m3fn),
        t.Tensor((h,), dtype=t.float32),
    )
    return SimpleNamespace(
        layer_name="model.layers.3.mlp",
        routed_experts=layer,
        _quant_method=qm,
        _test_shared=shared,
        moe_config=SimpleNamespace(
            experts_per_token=topk,
            is_sequence_parallel=False,
            skip_final_all_reduce=False,
            moe_parallel_config=SimpleNamespace(
                use_ep=(ep > 1),
                ep_size=ep,
                tp_size=N_RANKS // ep,
                dp_size=1,
                pcp_size=1,
                enable_eplb=False,
            ),
        ),
        gate=object(),
        shared_expert_gate=None,
        routed_input_transform=None,
        routed_output_transform=None,
        router=object(),
        routed_scaling_factor=ROUTED_SCALE,
    )


ADAPTER_SMALL_EP2TP4 = dict(
    h=1024, e=16, i=256, i_s=256, topk=4, qb1=512, bk2=256, ep=2
)
ADAPTER_SMALL_TP8 = dict(h=1024, e=16, i=128, i_s=256, topk=4, qb1=512, bk2=128, ep=1)


@pytest.mark.parametrize(
    "cfg", [GLM52_EP2TP4, GLM52_TP8, ADAPTER_SMALL_EP2TP4, ADAPTER_SMALL_TP8]
)
def test_adapter_flag_off_unchanged_and_flag_on_admission(
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict[str, int],
):
    """Flag off leaves prefill_above=None; flag on defaults prefill_above to 2047."""
    _install_mock_tpu7x(monkeypatch)
    monkeypatch.setenv("FUSED_MOE_DECODE_KERNEL", "0")
    monkeypatch.setenv("TPU_MOE_FP8_ALL_GATHER", "0")
    monkeypatch.delenv("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS", raising=False)
    rt = _load_nvfp4_adapter_for_test(monkeypatch)
    adapter = rt.adapter
    runner = _make_stub_runner(rt, cfg)

    # 1. Flag OFF (default): prefill_above is None and prefill admission is not called.
    monkeypatch.delenv("FUSED_MOE_PREFILL_NVFP4_KERNEL", raising=False)
    probe_calls: list[int] = []
    orig_can = adapter.prefill_nvfp4.can_use_fused_moe_prefill_nvfp4

    def spy_can(*a: Any, **kw: Any) -> bool:
        probe_calls.append(1)
        return orig_can(*a, **kw)

    monkeypatch.setattr(
        adapter.prefill_nvfp4, "can_use_fused_moe_prefill_nvfp4", spy_can
    )
    adm_off = adapter._admit(runner, N_RANKS)
    assert not isinstance(adm_off, str), adm_off
    assert adm_off.rows == (16, 32, 64)
    assert adm_off.statics.prefill_above is None
    assert probe_calls == []

    # 2. Flag ON (default min_tokens=2048): prefill_above == 2047 (n >= 2048).
    monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_KERNEL", "1")
    adm_on = adapter._admit(runner, N_RANKS)
    assert not isinstance(adm_on, str), adm_on
    assert adm_on.rows == (16, 32, 64)
    assert adm_on.statics.prefill_above == 2047
    assert len(probe_calls) == 1

    # 3. Flag ON when prefill admission refuses: decode still arms cleanly,
    #    prefill_above falls back to None (_tp_chain + lax.psum).
    monkeypatch.setattr(
        adapter.prefill_nvfp4,
        "can_use_fused_moe_prefill_nvfp4",
        lambda *_a, **_k: False,
    )
    adm_refused = adapter._admit(runner, N_RANKS)
    assert not isinstance(adm_refused, str), adm_refused
    assert adm_refused.rows == (16, 32, 64)
    assert adm_refused.statics.prefill_above is None


@pytest.mark.parametrize(
    "cfg", [GLM52_EP2TP4, GLM52_TP8, ADAPTER_SMALL_EP2TP4, ADAPTER_SMALL_TP8]
)
def test_adapter_prefill_min_tokens_threshold_resolution(
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict[str, int],
):
    """FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS resolves prefill_above = max(64, min - 1)."""
    _install_mock_tpu7x(monkeypatch)
    monkeypatch.setenv("FUSED_MOE_DECODE_KERNEL", "0")
    monkeypatch.setenv("TPU_MOE_FP8_ALL_GATHER", "0")
    rt = _load_nvfp4_adapter_for_test(monkeypatch)
    adapter = rt.adapter
    runner = _make_stub_runner(rt, cfg)

    # Flag off ignores min_tokens override completely.
    monkeypatch.delenv("FUSED_MOE_PREFILL_NVFP4_KERNEL", raising=False)
    monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS", "512")
    adm_off = adapter._admit(runner, N_RANKS)
    assert not isinstance(adm_off, str), adm_off
    assert adm_off.statics.prefill_above is None

    monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_KERNEL", "1")

    # 1. Default (unset -> 2048) -> prefill_above == 2047 (dispatches n >= 2048).
    monkeypatch.delenv("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS", raising=False)
    adm_def = adapter._admit(runner, N_RANKS)
    assert not isinstance(adm_def, str), adm_def
    assert adm_def.statics.prefill_above == 2047

    # 2. Explicit overrides above 64 (e.g. 512, 1024, 4096) -> min_tokens - 1.
    for min_tok, want_above in ((512, 511), (1024, 1023), (4096, 4095)):
        monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS", str(min_tok))
        adm_ov = adapter._admit(runner, N_RANKS)
        assert not isinstance(adm_ov, str), adm_ov
        assert adm_ov.statics.prefill_above == want_above

    # 3. Values <= 64 (65 -> 64, 64 -> 64, 32 -> 64, 0 -> 64, -5 -> 64) clamp to
    #    ROW_BUCKETS[-1] == 64 so decode buckets (<= 64) never dispatch to prefill.
    for raw_low in ("65", "64", "32", "1", "0", "-5"):
        monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS", raw_low)
        adm_clamped = adapter._admit(runner, N_RANKS)
        assert not isinstance(adm_clamped, str), adm_clamped
        assert adm_clamped.statics.prefill_above == 64


@pytest.mark.parametrize("cfg", [ADAPTER_SMALL_EP2TP4, ADAPTER_SMALL_TP8])
def test_adapter_op_body_flag_off_vs_flag_on_dispatch_small_shapes(
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict[str, int],
):
    """Single _op_body dispatches decode at n<=64, _tp_chain when flag off, and
    fused_moe_prefill_nvfp4_stage_a (W4A16) at n>64 when flag on."""
    _install_mock_tpu7x(monkeypatch)
    _install_cpu_expert_emulator(monkeypatch)
    monkeypatch.setenv("FUSED_MOE_DECODE_KERNEL", "0")
    monkeypatch.setenv("TPU_MOE_FP8_ALL_GATHER", "0")
    rt = _load_nvfp4_adapter_for_test(monkeypatch)
    adapter = rt.adapter

    mesh = _cpu_or_any_mesh()
    rep = NamedSharding(mesh, P())
    shd = NamedSharding(mesh, P("d"))
    w1, w2, s1, s2, sw1, ss1, sw2, ss2, e0 = _build_small_sharded_weights(
        mesh, cfg, seed=301
    )
    ranks = jax.device_put(
        jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1), shd
    )

    decode_calls: list[int] = []
    prefill_calls: list[int] = []
    tp_chain_calls: list[int] = []

    def fake_decode_kernel(x_in, *_rest, **_kw):
        decode_calls.append(int(x_in.shape[0]))
        return jnp.full_like(x_in, 7.0)

    orig_prefill_stage_a = adapter.prefill_nvfp4.fused_moe_prefill_nvfp4_stage_a
    orig_tp_chain = rt.fp8_decode._tp_chain

    def spy_prefill(*a: Any, **kw: Any):
        prefill_calls.append(int(a[0].shape[0]))
        return orig_prefill_stage_a(*a, **kw)

    def spy_tp_chain(*a: Any, **kw: Any):
        tp_chain_calls.append(int(a[0].shape[0]))
        return orig_tp_chain(*a, **kw)

    monkeypatch.setattr(adapter.kernel, "moe_tp_decode_nvfp4", fake_decode_kernel)
    monkeypatch.setattr(
        adapter.prefill_nvfp4, "fused_moe_prefill_nvfp4_stage_a", spy_prefill
    )
    monkeypatch.setattr(rt.fp8_decode, "_tp_chain", spy_tp_chain)

    runner = _make_stub_runner(rt, cfg)
    monkeypatch.delenv("FUSED_MOE_PREFILL_NVFP4_KERNEL", raising=False)
    monkeypatch.delenv("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS", raising=False)
    adm_off = adapter._admit(runner, N_RANKS)
    assert not isinstance(adm_off, str)
    monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_KERNEL", "1")
    adm_on_default = adapter._admit(runner, N_RANKS)
    assert not isinstance(adm_on_default, str)
    assert adm_on_default.statics.prefill_above == 2047
    monkeypatch.setenv("FUSED_MOE_PREFILL_NVFP4_MIN_TOKENS", "128")
    adm_on = adapter._admit(runner, N_RANKS)
    assert not isinstance(adm_on, str)
    assert adm_on.statics.prefill_above == 127

    specs = (
        P(),
        P(),
        P(),
        P(),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
    )
    fn_off = jax.jit(
        jax.shard_map(
            adapter._op_body(adm_off.statics, N_RANKS, "d"),
            mesh=mesh,
            in_specs=specs,
            out_specs=P("d"),
            check_vma=False,
        )
    )
    fn_on_default = jax.jit(
        jax.shard_map(
            adapter._op_body(adm_on_default.statics, N_RANKS, "d"),
            mesh=mesh,
            in_specs=specs,
            out_specs=P("d"),
            check_vma=False,
        )
    )
    fn_on = jax.jit(
        jax.shard_map(
            adapter._op_body(adm_on.statics, N_RANKS, "d"),
            mesh=mesh,
            in_specs=specs,
            out_specs=P("d"),
            check_vma=False,
        )
    )

    rng = np.random.default_rng(302 + cfg["ep"])
    h, e = cfg["h"], cfg["e"]
    bias = jax.device_put(
        jnp.asarray(rng.uniform(-0.05, 0.05, size=(e,)).astype(np.float32)), rep
    )

    # 1. Decode buckets (n = 8 < 16 and n = 16 in ROW_BUCKETS) inside fn_on
    #    still hit moe_tp_decode_nvfp4 (never prefill, never _tp_chain).
    for n_dec in (8, 16):
        x_dec = jax.device_put(
            jnp.asarray(rng.standard_normal((n_dec, h)).astype(np.float32), BF16), rep
        )
        sc_dec = jax.device_put(
            jnp.asarray(rng.uniform(0.1, 0.9, size=(n_dec, e)).astype(np.float32)), rep
        )
        pm_dec = jax.device_put(jnp.zeros((64,), dtype=jnp.bool_), rep)
        out_dec = np.asarray(
            fn_on(
                x_dec,
                sc_dec,
                bias,
                pm_dec,
                w1,
                w2,
                s1,
                s2,
                sw1,
                ss1,
                sw2,
                ss2,
                ranks,
                e0,
            )
        ).reshape(N_RANKS, n_dec, h)
        assert np.all(out_dec == 7.0)
    assert decode_calls == [16, 16]
    assert prefill_calls == []
    assert tp_chain_calls == []

    # 2. Prefill bucket (n = 128 > 64, with 8 trailing padded rows):
    #    - fn_off traces _tp_chain + lax.psum ONLY (byte-identical to standalone).
    #    - fn_on traces _prefill_chain -> fused_moe_prefill_nvfp4_stage_a ONLY.
    n_pre = 128
    n_valid = 120
    x_np = (rng.standard_normal((n_pre, h)) * 0.2).astype(np.float32)
    sc_np = (1.0 / (1.0 + np.exp(-rng.standard_normal((n_pre, e))))).astype(np.float32)
    pm_np = np.zeros((n_pre,), dtype=bool)
    pm_np[n_valid:] = True
    x_pre = jax.device_put(jnp.asarray(x_np, BF16), rep)
    sc_pre = jax.device_put(jnp.asarray(sc_np, F32), rep)
    pm_pre = jax.device_put(jnp.asarray(pm_np), rep)

    got_off_all = np.asarray(
        fn_off(
            x_pre,
            sc_pre,
            bias,
            pm_pre,
            w1,
            w2,
            s1,
            s2,
            sw1,
            ss1,
            sw2,
            ss2,
            ranks,
            e0,
        )
    ).reshape(N_RANKS, n_pre, h)
    assert tp_chain_calls == [n_pre]
    assert prefill_calls == []

    # Standalone _tp_chain + lax.psum reference is bit-identical to fn_off.
    ref_fn = jax.jit(
        jax.shard_map(
            lambda x_, sc_, b_, pm_, w1_, w2_, s1_, s2_, sw1_, ss1_, sw2_, ss2_, e0_: (
                lax.psum(
                    orig_tp_chain(
                        x_,
                        sc_,
                        b_,
                        pm_[: x_.shape[0]],
                        w1_,
                        w2_,
                        s1_,
                        s2_,
                        sw1_,
                        ss1_,
                        sw2_,
                        ss2_,
                        adm_off.statics,
                        e0_.reshape(()),
                    ),
                    "d",
                )
            ),
            mesh=mesh,
            in_specs=(
                P(),
                P(),
                P(),
                P(),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
                P("d"),
            ),
            out_specs=P("d"),
            check_vma=False,
        )
    )
    want_off_all = np.asarray(
        ref_fn(
            x_pre,
            sc_pre,
            bias,
            pm_pre,
            w1,
            w2,
            s1,
            s2,
            sw1,
            ss1,
            sw2,
            ss2,
            e0,
        )
    ).reshape(N_RANKS, n_pre, h)
    np.testing.assert_array_equal(got_off_all, want_off_all)

    # With default min_tokens=2048 (prefill_above=2047), n=128 stays on _tp_chain.
    got_on_def_all = np.asarray(
        fn_on_default(
            x_pre,
            sc_pre,
            bias,
            pm_pre,
            w1,
            w2,
            s1,
            s2,
            sw1,
            ss1,
            sw2,
            ss2,
            ranks,
            e0,
        )
    ).reshape(N_RANKS, n_pre, h)
    np.testing.assert_array_equal(got_on_def_all, want_off_all)
    assert tp_chain_calls == [n_pre, n_pre]
    assert prefill_calls == []

    # With min_tokens=128 (prefill_above=127), n=128 dispatches to _prefill_chain.
    got_on_all = np.asarray(
        fn_on(
            x_pre,
            sc_pre,
            bias,
            pm_pre,
            w1,
            w2,
            s1,
            s2,
            sw1,
            ss1,
            sw2,
            ss2,
            ranks,
            e0,
        )
    ).reshape(N_RANKS, n_pre, h)
    assert prefill_calls == [n_pre]
    assert tp_chain_calls == [n_pre, n_pre]  # fn_on did NOT call _tp_chain

    assert np.isfinite(got_on_all).all()
    for r in range(1, N_RANKS):
        np.testing.assert_array_equal(got_on_all[0], got_on_all[r])
    assert _rel_l2(got_on_all[0, :n_valid], got_off_all[0, :n_valid]) <= 5e-3
    assert _cos_sim(got_on_all[0, :n_valid], got_off_all[0, :n_valid]) >= 0.9999

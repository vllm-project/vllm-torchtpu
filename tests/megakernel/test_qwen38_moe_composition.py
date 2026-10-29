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
"""Qwen3.8-2.4T-shaped (`KH=16 < R=32`, `IS=2048 < 128*R`) composition at 32 ranks."""

from __future__ import annotations

import ast
import copy
import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.megakernel as mk
from vllm_torchtpu.megakernel import (
    kernel,
    load_logical_weights,
    model_def,
    planner,
    profile,
    vmem,
)
from vllm_torchtpu.megakernel.comm.replication import shard_count
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.model_def import find_model, get_model, register_model
from vllm_torchtpu.megakernel.models import (
    qwen35,
    qwen35_profile,
    qwen38_moe,
    qwen38_moe_profile,
)
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES
from vllm_torchtpu.megakernel.ops import F32, I32

R = 32
TOL = 3e-2
CTX = 2048
POS0 = 1800
_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_CHILD_TIMEOUT_S = 540

QWEN38_FULL_HF_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 8192,
        "num_hidden_layers": 92,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 23,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 128,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 2048,
        "shared_expert_intermediate_size": 2048,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "rope_parameters": {
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
        },
    },
}


def _tiny_qwen38_geometry() -> qwen38_moe.Qwen38MoeGeometry:
    """Tiny Qwen3.8-shaped geometry (`KH=16 < R=32`, `IS=2048 < 128*R`) at R=32."""
    return qwen38_moe.Qwen38MoeGeometry(
        L=4,
        att_every=4,
        H=1024,
        NH=32,
        NKV=4,
        HD=128,
        ROT=32,
        KH=16,
        VH=32,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=2048,
        moe_tp=False,
    )


def _tiny_qwen38_r8_geometry() -> qwen38_moe.Qwen38MoeGeometry:
    """8-rank geometry with identical replication factors (`kh_rep=2`, `sh_rep=2`)."""
    return qwen38_moe.Qwen38MoeGeometry(
        L=4,
        att_every=4,
        H=1024,
        NH=8,
        NKV=2,
        HD=128,
        ROT=32,
        KH=4,
        VH=8,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=512,
        moe_tp=False,
    )


def _load_logical(hf, g, ex):
    """Shard unsharded HF weights with the generic declaration-driven loader."""
    return load_logical_weights(g._model_def, ex, hf, g=g)


def _rms(x) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.square(x.astype(F32)))))


def _report(name: str, got, exp) -> None:
    err = jnp.abs(got.astype(F32) - exp.astype(F32))
    print(
        f"{name}: max_err={float(err.max()):.3e} rms_err={_rms(err):.3e} "
        f"rms_ref={_rms(exp):.3e}"
    )


def _check(name: str, got, exp, controls) -> None:
    _report(name, got, exp)
    assert jnp.allclose(got, exp, atol=TOL, rtol=TOL), name
    for label, bad in controls:
        _report(f"{name} vs {label}", bad, exp)
        assert not jnp.allclose(bad, exp, atol=TOL, rtol=TOL), f"{name}: {label}"


U_BF16 = 2.0**-8


def _one_rounding(x, seed: int):
    """Move every bf16 element of `x` by one rounding step with fixed random signs."""
    sgn = jnp.where(
        jax.random.bernoulli(jax.random.PRNGKey(seed), 0.5, x.shape), 1.0, -1.0
    )
    return (x.astype(F32) * (1.0 + U_BF16 * sgn)).astype(x.dtype)


def _check_kv(name: str, got, ref, ref_p, pos, group: int, controls) -> None:
    """KV cache after one T=1 step: exact off the written rows, noise-bounded on them.

    `ref_p` is the reference run on `_one_rounding(hid)`. The input is only known to one
    bf16 rounding, so `rmse(ref_p - ref)` on the written rows is this test's resolution.
    """
    got, ref, ref_p = (np.asarray(a.astype(F32), np.float64) for a in (got, ref, ref_p))
    m = np.zeros(got.shape, bool)  # [R, L, NS, ctx, hd]
    for s, p in enumerate(np.asarray(pos)):
        m[:, :, s, int(p), :] = True
    assert np.array_equal(got[~m], ref[~m]), f"{name}: unwritten rows changed"
    for r in range(got.shape[0]):
        assert np.array_equal(got[r], got[r - r % group]), (
            f"{name}: replica {r} differs"
        )
    floor = float(np.sqrt(np.mean((ref_p - ref)[m] ** 2)))
    err = float(np.sqrt(np.mean((got - ref)[m] ** 2)))
    print(f"{name}: written rmse={err:.3e} noise_floor={floor:.3e}")
    assert err <= floor, f"{name}: rmse {err:.3e} > noise floor {floor:.3e}"
    for label, bad in controls:
        bad = np.asarray(bad.astype(F32), np.float64)
        assert float(np.sqrt(np.mean((bad - ref)[m] ** 2))) > floor, f"{name}: {label}"


def _swap(x, a: int, b: int):
    return x.at[a].set(x[b]).at[b].set(x[a])


def test_qwen38_full_geometry_and_shared_expert_replication_at_r32():
    """Verify full Qwen3.8 geometry (`L=92, H=8192, KH=16, IS=2048`) plans at `R=32`."""
    assert find_model(QWEN38_FULL_HF_CONFIG) is qwen38_moe.Qwen38Moe
    assert find_model(qwen38_moe_profile.DEFAULT_HF_CONFIG) is qwen38_moe.Qwen38Moe
    assert find_model(qwen35_profile.DEFAULT_HF_CONFIG) is qwen35.Qwen35
    assert get_model("qwen3_8_moe") is qwen38_moe.Qwen38Moe
    assert qwen38_moe.Qwen38Moe.default_profile().name == "qwen38-baseline"
    assert qwen38_moe.Qwen38Moe.get_profile("tp32-dcp8").name == "qwen38-tp32-dcp8"

    g = qwen38_moe.Qwen38Moe.geometry(QWEN38_FULL_HF_CONFIG, moe_tp=False)
    ex = ExecSpec(B=4, T=1, R=R, ctx=CTX)
    g.check(ex)
    assert g.kh_rep(R) == 2
    assert g.kh(R) == 1
    assert g.vh(R) == 4
    assert g.sh_rep(R) == 2
    assert g.ISR(R) == 128
    assert g.ER(R) == 16

    ms = g.to_model_spec(R=R)
    assert ms.name == "qwen3_8_moe"
    assert ms.layer_type.name == "qwen38_moe_layer"
    assert qwen38_moe.Qwen38Moe.supported_tp_sizes == (32,)
    p = planner.plan(ms, ex)
    assert p.module_params["gdn"].kh == 1
    assert p.module_params["gdn"].vh == 4
    assert p.module_params["moe"].ISR == 128
    assert p.module_params["moe"].sh_rep == 2
    assert ms.cache_backend is not None
    play = ms.cache_backend.layout
    assert play.kh_per_rank(R) == 1
    assert play.vh_per_rank(R) == 4
    assert play.qkv_per_rank(R) == 768
    assert play.rec_pages_per_slot == 4
    play.validate(ex)

    g_tiny = _tiny_qwen38_geometry()
    ex_tiny = ExecSpec(B=2, T=1, R=R, ctx=CTX)

    # Also verify 32-rank weight splitting, packing, and reference step on tiny Qwen3.8.
    k_in, k_w = jax.random.split(jax.random.PRNGKey(3800))
    _, hid_l, _, st_l = mk.random_logical_inputs(k_in, g_tiny, ex_tiny)
    w_l = _load_logical(mk.random_hf_weights(k_w, g_tiny), g_tiny, ex_tiny)
    pos = POS0 + jnp.arange(ex_tiny.NS, dtype=I32)
    qk_cols = 2 * g_tiny.kh(R) * g_tiny.KD
    assert jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][1, :, :, :qk_cols]
    )
    assert not jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][2, :, :, :qk_cols]
    )
    assert jnp.array_equal(w_l["sh13"][0], w_l["sh13"][1])
    assert not jnp.array_equal(w_l["sh13"][0], w_l["sh13"][2])
    ref_hid, ref_st = qwen35.reference_step(g_tiny, ex_tiny, pos, hid_l, w_l, st_l)
    _, _, w_mk, _ = mk.pack_inputs(pos, hid_l, w_l, st_l, g_tiny, ex_tiny, mesh=None)
    assert jnp.array_equal(w_mk["sh13"][0], w_mk["sh13"][1])
    assert not jnp.array_equal(w_mk["sh13"][0], w_mk["sh13"][2])
    conv_swap_01_qk = (
        ref_st["conv"]
        .at[:, :, :, :qk_cols]
        .set(_swap(ref_st["conv"][:, :, :, :qk_cols], 0, 1))
    )
    assert jnp.array_equal(
        conv_swap_01_qk[:, :, : 3 * ex_tiny.B, :qk_cols],
        ref_st["conv"][:, :, : 3 * ex_tiny.B, :qk_cols],
    )
    assert not jnp.allclose(
        _swap(ref_st["conv"], 0, 2), ref_st["conv"], atol=TOL, rtol=TOL
    )
    assert not jnp.allclose(hid_l, ref_hid, atol=TOL, rtol=TOL)


def test_qwen38_sub_rank_replication_step_in_process():
    """Run `make_step` on 8 CPU devices with `kh_rep=2` and `sh_rep=2`."""
    r8 = 8
    devs = jax.devices("cpu")
    assert len(devs) >= r8, f"Requires {r8} CPU devices, found {len(devs)}"
    g = _tiny_qwen38_r8_geometry()
    assert g.kh_rep(r8) == 2 and g.sh_rep(r8) == 2
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, hier=3, mb8=2, inplace=False
    ).with_modules({"gqa": {"split": True}})
    ex = ExecSpec(B=2, T=1, R=r8, ctx=1024, tiles=tiles, interpret=True)

    k_in, k_w = jax.random.split(jax.random.PRNGKey(3808))
    _, hid_l, _, st_l = mk.random_logical_inputs(k_in, g, ex)
    w_l = _load_logical(mk.random_hf_weights(k_w, g), g, ex)
    pos = 900 + jnp.arange(ex.NS, dtype=I32)
    ref_hid, ref_st = qwen38_moe.reference_step(g, ex, pos, hid_l, w_l, st_l)
    _, ref_st_p = qwen38_moe.reference_step(
        g, ex, pos, _one_rounding(hid_l, 3808), w_l, st_l
    )

    mesh = Mesh(devs[:r8], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    step, _ = kernel.make_step(mesh, g.to_model_spec(R=ex.R), ex, donate_states=True)
    st_in = {
        k: jax.device_get(kernel.unfold(st_mk[k], r8))
        for k in ("conv", "rec", "kc", "vc")
    }
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    hid_r = kernel.unfold(out_hid, r8)[:, : ex.B]
    for r in range(1, r8):
        assert jnp.array_equal(hid_r[r], hid_r[0]), f"hidden differs on rank {r}"
    _check("hidden", hid_r[0], ref_hid, [("input hidden", hid_l)])

    qk_cols = 2 * g.kh(r8) * g.KD
    conv_got = kernel.unfold(out_st["conv"], r8)
    conv_swap_01_qk = conv_got.at[:, :, :, :qk_cols].set(
        _swap(conv_got[:, :, :, :qk_cols], 0, 1)
    )
    assert jnp.array_equal(
        conv_swap_01_qk[:, :, : 3 * ex.B, :qk_cols],
        conv_got[:, :, : 3 * ex.B, :qk_cols],
    )
    _check(
        "conv",
        conv_got,
        ref_st["conv"],
        [
            ("ranks 0 and 2 swapped", _swap(conv_got, 0, 2)),
            ("input state", st_in["conv"]),
        ],
    )
    got_rec = kernel.unfold(out_st["rec"], r8)
    _check(
        "rec",
        got_rec,
        ref_st["rec"],
        [
            ("ranks 0 and 1 swapped", _swap(got_rec, 0, 1)),
            ("input state", st_in["rec"]),
        ],
    )
    for k in ("kc", "vc"):
        got = kernel.unfold(out_st[k], r8)
        _check_kv(
            k,
            got,
            ref_st[k],
            ref_st_p[k],
            pos,
            g.G(r8),
            [
                ("ranks 0 and 4 swapped", _swap(got, 0, 4)),
                ("input state", st_in[k]),
            ],
        )


def _step_qwen38_32_ranks_in_process() -> None:
    devs = jax.devices("cpu")
    assert len(devs) >= R, f"Requires {R} CPU devices, found {len(devs)}"
    g = _tiny_qwen38_geometry()
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, hier=3, mb8=2, inplace=False
    ).with_modules({"gqa": {"split": True}})
    ex = ExecSpec(B=2, T=1, R=R, ctx=256, tiles=tiles, interpret=True)

    k_in, k_w = jax.random.split(jax.random.PRNGKey(3801))
    _, hid_l, _, st_l = mk.random_logical_inputs(k_in, g, ex)
    w_l = _load_logical(mk.random_hf_weights(k_w, g), g, ex)
    pos = 100 + jnp.arange(ex.NS, dtype=I32)

    # Ranks 0 and 1 share key/query head 0 and shared-expert slice 0; rank 2 does not.
    qk_cols = 2 * g.kh(R) * g.KD
    assert jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][1, :, :, :qk_cols]
    )
    assert not jnp.array_equal(
        w_l["lin_in"][0, :, :, :qk_cols], w_l["lin_in"][2, :, :, :qk_cols]
    )
    assert jnp.array_equal(w_l["sh13"][0], w_l["sh13"][1])
    assert not jnp.array_equal(w_l["sh13"][0], w_l["sh13"][2])

    ref_hid, ref_st = qwen38_moe.reference_step(g, ex, pos, hid_l, w_l, st_l)
    _, ref_st_p = qwen38_moe.reference_step(
        g, ex, pos, _one_rounding(hid_l, 3801), w_l, st_l
    )

    mesh = Mesh(devs[:R], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    step, _ = kernel.make_step(mesh, g.to_model_spec(R=ex.R), ex, donate_states=True)
    st_in = {
        k: jax.device_get(kernel.unfold(st_mk[k], R))
        for k in ("conv", "rec", "kc", "vc")
    }
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    hid_r = kernel.unfold(out_hid, R)[:, : ex.B]
    for r in range(1, R):
        assert jnp.array_equal(hid_r[r], hid_r[0]), f"hidden differs on rank {r}"
    _check("hidden", hid_r[0], ref_hid, [("input hidden", hid_l)])

    conv_got = kernel.unfold(out_st["conv"], R)
    # Positive control: swapping Q/K conv state between replica ranks 0 and 1 is exact.
    conv_swap_01_qk = conv_got.at[:, :, :, :qk_cols].set(
        _swap(conv_got[:, :, :, :qk_cols], 0, 1)
    )
    assert jnp.array_equal(
        conv_swap_01_qk[:, :, : 3 * ex.B, :qk_cols],
        conv_got[:, :, : 3 * ex.B, :qk_cols],
    )
    # Negative control: swapping Q/K conv state between ranks 0 and 2 fails.
    conv_swap_02 = _swap(conv_got, 0, 2)
    _check(
        "conv",
        conv_got,
        ref_st["conv"],
        [("ranks 0 and 2 swapped", conv_swap_02), ("input state", st_in["conv"])],
    )

    got_rec = kernel.unfold(out_st["rec"], R)
    _check(
        "rec",
        got_rec,
        ref_st["rec"],
        [
            ("ranks 0 and 1 swapped", _swap(got_rec, 0, 1)),
            ("input state", st_in["rec"]),
        ],
    )
    for k in ("kc", "vc"):
        got = kernel.unfold(out_st[k], R)
        _check_kv(
            k,
            got,
            ref_st[k],
            ref_st_p[k],
            pos,
            g.G(R),
            [
                ("ranks 0 and 8 swapped", _swap(got, 0, 8)),
                ("input state", st_in[k]),
            ],
        )
    print("32-rank step completed successfully with L=4, att_every=4, CTX=256")


@pytest.mark.slow
def test_qwen38_decode_with_32_ranks():
    flags = [
        f
        for f in os.environ.get("XLA_FLAGS", "").split()
        if not f.startswith(_DEVICE_COUNT_FLAG)
    ]
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        MALLOC_ARENA_MAX="2",
        XLA_FLAGS=" ".join([*flags, f"{_DEVICE_COUNT_FLAG}={R}"]),
    )
    t0 = time.perf_counter()
    out = subprocess.run(
        [sys.executable or "python3", __file__, "qwen38_r32"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )
    child_time = time.perf_counter() - t0
    print(f"test_qwen38_decode_with_32_ranks child wall time: {child_time:.2f}s")
    print(out.stdout)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]


def _unsharded_qwen38_hybrid_oracle_f64(
    hf: dict[str, jax.Array],
    hid_in: jax.Array,
    pos: jax.Array,
    kc_unsharded: np.ndarray,
    vc_unsharded: np.ndarray,
    conv_unsharded: np.ndarray,
    rec_unsharded: np.ndarray,
    g: Any,
    b_cnt: int,
) -> np.ndarray:
    """Standalone unsharded (`R=1`) `numpy.float64` hybrid `GDN + GQA + MoE` oracle."""
    l_cnt, att_every = int(g.L), int(g.att_every)
    nh, nkv, hd, rot = int(g.NH), int(g.NKV), int(g.HD), int(g.ROT)
    kh, vh, kd, vd = int(g.KH), int(g.VH), int(g.KD), int(g.VD)
    topk, inter_d, is_dim = int(g.TOPK), int(g.I), int(g.IS)
    eps = float(g.eps)
    l2_eps = float(getattr(g, "gdn_l2_eps", 1e-6))
    theta = float(g.theta)
    kq_dim, v_dim = kh * kd, vh * vd
    qkv_full = 2 * kq_dim + v_dim
    expand_gdn = vh // kh
    q_per_kv = nh // nkv

    def _zc_rmsnorm(x_arr: np.ndarray, w_arr: np.ndarray) -> np.ndarray:
        rms_inv = 1.0 / np.sqrt(np.mean(x_arr * x_arr, axis=-1, keepdims=True) + eps)
        return x_arr * rms_inv * (1.0 + w_arr)

    def _silu(x_arr: np.ndarray) -> np.ndarray:
        return x_arr / (1.0 + np.exp(-x_arr))

    def _sigmoid(x_arr: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x_arr))

    half_rot = rot // 2

    def _rope(vec: np.ndarray, cos_val: np.ndarray, sin_val: np.ndarray) -> np.ndarray:
        r0 = vec[:, :half_rot]
        r1 = vec[:, half_rot:rot]
        rot_out = np.concatenate(
            [r0 * cos_val - r1 * sin_val, r1 * cos_val + r0 * sin_val], axis=-1
        )
        return np.concatenate([rot_out, vec[:, rot:]], axis=-1)

    inv_freq = 1.0 / (
        theta ** (np.arange(half_rot, dtype=np.float64)[None, :] * 2.0 / rot)
    )

    x_cur = np.asarray(hid_in[:b_cnt].astype(jnp.bfloat16), dtype=np.float32).astype(
        np.float64
    )
    pos_np = np.asarray(pos[:b_cnt], dtype=np.int64)

    gdn_idx = 0
    att_idx = 0
    for lyr in range(l_cnt):
        w_in_n = np.asarray(hf["in_norm"][lyr], dtype=np.float64)
        xb = _zc_rmsnorm(x_cur, w_in_n[None, :])

        if (lyr + 1) % att_every != 0:
            # Unsharded GDN linear_attention layer
            q_w = (
                np.asarray(hf["lin_q"][gdn_idx].astype(F32), dtype=np.float64)[:kq_dim]
                * np.asarray(hf["lin_q_s"][gdn_idx], dtype=np.float64)[:kq_dim, None]
            )
            k_w = (
                np.asarray(hf["lin_k"][gdn_idx].astype(F32), dtype=np.float64)[
                    kq_dim : 2 * kq_dim
                ]
                * np.asarray(hf["lin_k_s"][gdn_idx], dtype=np.float64)[
                    kq_dim : 2 * kq_dim, None
                ]
            )
            v_w = (
                np.asarray(hf["lin_v"][gdn_idx].astype(F32), dtype=np.float64)[
                    2 * kq_dim : qkv_full
                ]
                * np.asarray(hf["lin_v_s"][gdn_idx], dtype=np.float64)[
                    2 * kq_dim : qkv_full, None
                ]
            )
            lq, lk, lv = q_w.T, k_w.T, v_w.T
            lz = (
                np.asarray(hf["lin_z"][gdn_idx].astype(F32), dtype=np.float64)
                * np.asarray(hf["lin_z_s"][gdn_idx], dtype=np.float64)[:, None]
            ).T
            lb = (
                np.asarray(hf["lin_b"][gdn_idx].astype(F32), dtype=np.float64)
                * np.asarray(hf["lin_b_s"][gdn_idx], dtype=np.float64)[:, None]
            ).T
            la = (
                np.asarray(hf["lin_a"][gdn_idx].astype(F32), dtype=np.float64)
                * np.asarray(hf["lin_a_s"][gdn_idx], dtype=np.float64)[:, None]
            ).T

            qkv_proj = np.concatenate([xb @ lq, xb @ lk, xb @ lv], axis=-1)
            z_proj = xb @ lz
            b_proj = xb @ lb
            a_proj = xb @ la

            cw = np.asarray(hf["conv"][gdn_idx], dtype=np.float64)
            if cw.ndim == 3:
                cw = cw[:, 0, :].T
            c_hist = conv_unsharded[gdn_idx, :b_cnt]
            conv_out = (
                c_hist[:, 0, :] * cw[0][None, :]
                + c_hist[:, 1, :] * cw[1][None, :]
                + c_hist[:, 2, :] * cw[2][None, :]
                + qkv_proj * cw[3][None, :]
            )
            u_act = _silu(conv_out)
            alog = np.asarray(hf["alog"][gdn_idx], dtype=np.float64)
            dtb = np.asarray(hf["dtb"][gdn_idx], dtype=np.float64)
            gnw = np.asarray(
                hf["gnorm"][gdn_idx].astype(jnp.bfloat16).astype(F32), dtype=np.float64
            )

            mix_rows = []
            for b_i in range(b_cnt):
                st_b = rec_unsharded[gdn_idx, b_i].copy()
                q_b = u_act[b_i, :kq_dim].reshape(kh, kd)
                k_b = u_act[b_i, kq_dim : 2 * kq_dim].reshape(kh, kd)
                v_b = u_act[b_i, 2 * kq_dim : qkv_full].reshape(vh, vd)
                q_b = (
                    q_b
                    * (kd**-0.5)
                    / np.sqrt(np.sum(q_b * q_b, axis=-1, keepdims=True) + l2_eps)
                )
                k_b = k_b / np.sqrt(np.sum(k_b * k_b, axis=-1, keepdims=True) + l2_eps)
                qc = np.repeat(q_b, expand_gdn, axis=0)[:, :, None]
                kc = np.repeat(k_b, expand_gdn, axis=0)[:, :, None]
                beta = _sigmoid(b_proj[b_i])
                decay = -np.exp(alog) * np.logaddexp(0.0, a_proj[b_i] + dtb)
                st_b = st_b * np.exp(decay)[:, None, None]
                pred = np.sum(st_b * kc, axis=1)
                delta = (v_b - pred) * beta[:, None]
                st_b = st_b + kc * delta[:, None, :]
                out_h = np.sum(st_b * qc, axis=1)
                rms_h = 1.0 / np.sqrt(
                    np.mean(out_h * out_h, axis=-1, keepdims=True) + eps
                )
                val_h = out_h * rms_h * gnw[None, :]
                zz = z_proj[b_i].reshape(vh, vd)
                mix_rows.append((val_h * _silu(zz)).reshape(v_dim))
            mix_mat = np.stack(mix_rows, axis=0)
            l_out = (
                np.asarray(hf["lin_out"][gdn_idx].astype(F32), dtype=np.float64)
                * np.asarray(hf["lin_out_s"][gdn_idx], dtype=np.float64)[:, None]
            ).T
            mixer_delta = mix_mat @ l_out
            gdn_idx += 1
        else:
            # Unsharded GQA full_attention layer
            qp = (
                np.asarray(hf["q_proj"][att_idx].astype(F32), dtype=np.float64)
                * np.asarray(hf["q_proj_s"][att_idx], dtype=np.float64)[:, None]
            ).T
            kp = np.asarray(hf["k_proj"][att_idx].astype(F32), dtype=np.float64)
            ks = np.asarray(hf["k_proj_s"][att_idx], dtype=np.float64)
            vp = np.asarray(hf["v_proj"][att_idx].astype(F32), dtype=np.float64)
            vs = np.asarray(hf["v_proj_s"][att_idx], dtype=np.float64)
            qn_w = np.asarray(hf["qn"][att_idx], dtype=np.float64)
            kn_w = np.asarray(hf["kn"][att_idx], dtype=np.float64)

            qg_all = (xb @ qp).reshape(b_cnt, nh, 2 * hd)
            q_raw = qg_all[:, :, :hd]
            gate_att = qg_all[:, :, hd:].reshape(b_cnt, nh * hd)
            kp_3d = kp.reshape(nkv, hd, int(g.H)).transpose(0, 2, 1)
            vp_3d = vp.reshape(nkv, hd, int(g.H)).transpose(0, 2, 1)
            ks_2d = ks.reshape(nkv, hd)
            vs_2d = vs.reshape(nkv, hd)
            k_raw = np.einsum("bh,khd->bkd", xb, kp_3d) * ks_2d[None, :, :]
            v_raw = np.einsum("bh,khd->bkd", xb, vp_3d) * vs_2d[None, :, :]

            att_rows = []
            for b_i in range(b_cnt):
                ang = float(pos_np[b_i]) * inv_freq
                cos_v, sin_v = np.cos(ang), np.sin(ang)
                qb = _zc_rmsnorm(q_raw[b_i], qn_w)
                kb = _zc_rmsnorm(k_raw[b_i], kn_w)
                qb = _rope(qb, cos_v, sin_v) * (hd**-0.5)
                kb = _rope(kb, cos_v, sin_v)
                vb = v_raw[b_i]
                plen = int(pos_np[b_i])
                head_outs = []
                for h_i in range(nh):
                    kv_i = h_i // q_per_kv
                    kc_hist = kc_unsharded[att_idx, b_i, :plen, kv_i, :]
                    vc_hist = vc_unsharded[att_idx, b_i, :plen, kv_i, :]
                    k_full = np.concatenate([kc_hist, kb[kv_i : kv_i + 1]], axis=0)
                    v_full = np.concatenate([vc_hist, vb[kv_i : kv_i + 1]], axis=0)
                    scores = qb[h_i] @ k_full.T
                    scores = scores - np.max(scores)
                    probs = np.exp(scores) / np.sum(np.exp(scores))
                    head_outs.append(probs @ v_full)
                oh = np.concatenate(head_outs, axis=0)
                att_rows.append(oh * _sigmoid(gate_att[b_i]))
            att_mix = np.stack(att_rows, axis=0)
            a_out = (
                np.asarray(hf["att_out"][att_idx].astype(F32), dtype=np.float64)
                * np.asarray(hf["att_out_s"][att_idx], dtype=np.float64)[:, None]
            ).T
            mixer_delta = att_mix @ a_out
            att_idx += 1

        x_cur = x_cur + mixer_delta

        # Unsharded MoE layer (1 logical shared expert + TOPK routed experts)
        w_post_n = np.asarray(hf["post_norm"][lyr], dtype=np.float64)
        xb_moe = _zc_rmsnorm(x_cur, w_post_n[None, :])
        w_router = np.asarray(hf["router"][lyr].astype(F32), dtype=np.float64).T
        logits = xb_moe @ w_router
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

        sh_gate = np.asarray(hf["sh_gate"][lyr], dtype=np.float64).squeeze()
        sg = _sigmoid(np.sum(xb_moe * sh_gate[None, :], axis=-1, keepdims=True))
        if "sh13" in hf:
            sh13 = (
                np.asarray(hf["sh13"][lyr].astype(F32), dtype=np.float64)
                * np.asarray(hf["sh13_s"][lyr], dtype=np.float64)[None, :]
            )
        else:
            sh_g = (
                np.asarray(hf["sh_gate_proj"][lyr].astype(F32), dtype=np.float64)
                * np.asarray(hf["sh_gate_proj_s"][lyr], dtype=np.float64)[:, None]
            )
            sh_u = (
                np.asarray(hf["sh_up_proj"][lyr].astype(F32), dtype=np.float64)
                * np.asarray(hf["sh_up_proj_s"][lyr], dtype=np.float64)[:, None]
            )
            sh13 = np.concatenate([sh_g, sh_u], axis=0).T
        g13 = xb_moe @ sh13
        mid_sh = _silu(g13[:, :is_dim]) * g13[:, is_dim:]
        if "sh2" in hf:
            sh2_w = np.asarray(hf["sh2"][lyr].astype(F32), dtype=np.float64)
            sh2_s = np.asarray(hf["sh2_s"][lyr], dtype=np.float64)
            sh2 = (
                (sh2_w * sh2_s[:, None]).T
                if sh2_w.shape[0] != is_dim
                else (sh2_w * sh2_s[None, :])
            )
        else:
            sh2 = np.asarray(hf["sh_down_proj"][lyr].astype(F32), dtype=np.float64)
        moe_delta = (mid_sh @ sh2) * sg

        if "e13" in hf:
            e13_l = np.asarray(hf["e13"][lyr].astype(F32), dtype=np.float64)
            e13_s_l = np.asarray(hf["e13_s"][lyr], dtype=np.float64)
            e2_l = np.asarray(hf["e2"][lyr].astype(F32), dtype=np.float64)
            e2_s_l = np.asarray(hf["e2_s"][lyr], dtype=np.float64)
        else:
            e_g = (
                np.asarray(hf["e_gate_proj"][lyr].astype(F32), dtype=np.float64)
                * np.asarray(hf["e_gate_proj_s"][lyr], dtype=np.float64)[:, :, None]
            )
            e_u = (
                np.asarray(hf["e_up_proj"][lyr].astype(F32), dtype=np.float64)
                * np.asarray(hf["e_up_proj_s"][lyr], dtype=np.float64)[:, :, None]
            )
            e13_l = np.transpose(np.concatenate([e_g, e_u], axis=1), (0, 2, 1))
            e13_s_l = np.ones((e13_l.shape[0], e13_l.shape[2]), dtype=np.float64)
            e2_w = np.asarray(hf["e2"][lyr].astype(F32), dtype=np.float64)
            e2_s = np.asarray(hf["e2_s"][lyr], dtype=np.float64)
            e2_l = np.transpose(e2_w * e2_s[:, :, None], (0, 2, 1))
            e2_s_l = np.ones((e2_l.shape[0], e2_l.shape[2]), dtype=np.float64)

        for b_i in range(b_cnt):
            top_idx = np.argsort(-probs[b_i], kind="stable")[:topk]
            top_vals = probs[b_i, top_idx]
            top_w = top_vals / np.sum(top_vals)
            for k_i in range(topk):
                e_id = int(top_idx[k_i])
                h_e = (xb_moe[b_i : b_i + 1] @ e13_l[e_id]) * e13_s_l[e_id][None, :]
                mid_e = _silu(h_e[:, :inter_d]) * h_e[:, inter_d:]
                y_e = (mid_e @ e2_l[e_id]) * e2_s_l[e_id][None, :]
                moe_delta[b_i : b_i + 1] += top_w[k_i] * y_e

        x_cur = x_cur + moe_delta

    w_fn = np.asarray(hf["final_norm"], dtype=np.float64)
    return _zc_rmsnorm(x_cur, w_fn[None, :])


def _check_qwen38_r32_vs_unsharded_f64_in_process() -> None:
    devs = jax.devices("cpu")
    assert len(devs) >= R, f"Requires {R} CPU devices, found {len(devs)}"
    g = qwen38_moe.Qwen38MoeGeometry(
        L=4,
        att_every=4,
        H=1024,
        NH=32,
        NKV=4,
        HD=128,
        ROT=32,
        KH=16,
        VH=32,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=2048,
        moe_tp=False,
    )
    assert g.kh_rep(R) == 2 and g.sh_rep(R) == 2
    u_bf16 = 2.0**-8
    mesh = Mesh(devs[:R], ("tp",))

    b_cnt = 5
    tiles = dataclasses.replace(
        TINY_TILES,
        bank=256 << 10,
        vmem_mib=64,
        hier=4,
        hier_group=8,
        rsag=False,
        mb8=1,
        inplace=False,
    ).with_modules({"gqa": {"split": True}})
    ex = ExecSpec(B=b_cnt, T=1, R=R, ctx=256, tiles=tiles, interpret=True)
    step_fn, _ = kernel.make_step(mesh, g.to_model_spec(R=R), ex, donate_states=False)

    for seed_idx in (0, 1):
        rng = np.random.default_rng(910 + seed_idx)
        hf = dict(mk.random_hf_weights(jax.random.PRNGKey(910 + seed_idx), g))
        for sk in ("sh_gate_proj_s", "sh_up_proj_s", "sh13_s", "sh2_s"):
            if sk in hf:
                hf[sk] = hf[sk] * 2.0
        w_l = load_logical_weights(qwen38_moe.Qwen38Moe, ex, hf, g=g)

        # Positions spanning distinct DCP KV-group modulo residues (0..7)
        pos_vals = np.array(
            [16 + i * 9 + (i % 8) for i in range(ex.NS)], dtype=np.int32
        )
        pos = jnp.asarray(pos_vals, dtype=I32)
        hid_l = jnp.asarray(
            rng.standard_normal((ex.NS, g.H), dtype=np.float32) * 0.5,
            dtype=jnp.bfloat16,
        )

        # Build unsharded caches and slice consistently to R=32 ranks
        kc_un = (
            rng.standard_normal((g.NATT, ex.NS, ex.ctx, g.NKV, g.HD), dtype=np.float32)
            * 0.15
        )
        vc_un = (
            rng.standard_normal((g.NATT, ex.NS, ex.ctx, g.NKV, g.HD), dtype=np.float32)
            * 0.15
        )
        qkv_full = 2 * g.KH * g.KD + g.VH * g.VD
        conv_un = (
            rng.standard_normal((g.NGDN, ex.NS, 3, qkv_full), dtype=np.float32) * 0.15
        )
        rec_un = (
            rng.standard_normal((g.NGDN, ex.NS, g.VH, g.KD, g.VD), dtype=np.float32)
            * 0.08
        )

        kc_j = jnp.asarray(kc_un, dtype=jnp.bfloat16)
        vc_j = jnp.asarray(vc_un, dtype=jnp.bfloat16)
        conv_j = jnp.asarray(conv_un, dtype=jnp.bfloat16).astype(F32)
        rec_j = jnp.asarray(rec_un, dtype=jnp.bfloat16)

        kc_un_bf = np.asarray(kc_j.astype(F32), dtype=np.float64)
        vc_un_bf = np.asarray(vc_j.astype(F32), dtype=np.float64)
        conv_un_bf = np.asarray(conv_j, dtype=np.float64)
        rec_un_bf = np.asarray(rec_j.astype(F32), dtype=np.float64)

        kc_r32 = jnp.stack([kc_j[:, :, :, r // 8, :] for r in range(R)], axis=0)
        vc_r32 = jnp.stack([vc_j[:, :, :, r // 8, :] for r in range(R)], axis=0)
        kq_full = g.KH * g.KD
        kh_w = g.kh(R) * g.KD
        vh_w = g.vh(R) * g.VD
        conv_ranks = []
        for r in range(R):
            g_pair = r // 2
            q_sl = conv_j[:, :, :, g_pair * kh_w : (g_pair + 1) * kh_w]
            k_sl = conv_j[
                :, :, :, kq_full + g_pair * kh_w : kq_full + (g_pair + 1) * kh_w
            ]
            v_sl = conv_j[
                :, :, :, 2 * kq_full + r * vh_w : 2 * kq_full + (r + 1) * vh_w
            ]
            c_cat = jnp.concatenate([q_sl, k_sl, v_sl], axis=-1)
            conv_ranks.append(c_cat)
        conv_r32 = jnp.stack(conv_ranks, axis=0)
        vh_r = g.vh(R)
        rec_r32 = jnp.stack(
            [rec_j[:, :, r * vh_r : (r + 1) * vh_r] for r in range(R)],
            axis=0,
        )
        st_l = {"kc": kc_r32, "vc": vc_r32, "conv": conv_r32, "rec": rec_r32}

        oracle_f64 = _unsharded_qwen38_hybrid_oracle_f64(
            hf, hid_l, pos, kc_un_bf, vc_un_bf, conv_un_bf, rec_un_bf, g, b_cnt
        )

        pos_mk, hid_mk, w_mk, st_mk = mk.pack_inputs(
            pos, hid_l, w_l, st_l, g, ex, mesh=mesh
        )
        out_hid, _ = step_fn(pos_mk, hid_mk, w_mk, st_mk)
        got_r32 = np.asarray(kernel.unfold(out_hid, R)[:, :b_cnt], dtype=np.float64)
        for r in range(1, R):
            assert np.array_equal(got_r32[r], got_r32[0])

        rms_ref = max(1e-3, float(np.sqrt(np.mean(oracle_f64 * oracle_f64))))
        # Quadrature over L=4 layers (sqrt(4) * 8 BF16 rounds/layer = 16 * u_bf16)
        max_abs_bound = 16.0 * u_bf16 * rms_ref
        max_abs = float(np.max(np.abs(got_r32[0] - oracle_f64)))
        assert max_abs <= max_abs_bound, (
            f"B={b_cnt}: max_abs={max_abs:.4e} > bound={max_abs_bound:.4e}"
        )

        # Negative control 1: uncompensated sh_rep (1.0 vs 0.5) > 3x bound
        w_bad_sh = dict(w_l, sh2_s=w_l["sh2_s"] * 2.0)
        _, _, w_mk_sh, _ = mk.pack_inputs(pos, hid_l, w_bad_sh, st_l, g, ex, mesh=mesh)
        out_bad_sh, _ = step_fn(pos_mk, hid_mk, w_mk_sh, st_mk)
        bad_sh_f64 = np.asarray(
            kernel.unfold(out_bad_sh, R)[0, :b_cnt], dtype=np.float64
        )
        assert float(np.max(np.abs(bad_sh_f64 - oracle_f64))) > 3.0 * max_abs_bound

        # Negative control 2: wrong interleaved key-head map (r % 16 vs r // 2)
        perm_interleaved = jnp.asarray(
            [(r % 16) * 2 for r in range(R)], dtype=jnp.int32
        )
        qk_cols = 2 * kh_w
        bad_lin_in = (
            w_l["lin_in"]
            .at[:, :, :, :qk_cols]
            .set(w_l["lin_in"][perm_interleaved, :, :, :qk_cols])
        )
        bad_lin_s = (
            w_l["lin_in_s"]
            .at[:, :, :qk_cols]
            .set(w_l["lin_in_s"][perm_interleaved, :, :qk_cols])
        )
        w_bad_kh = dict(w_l, lin_in=bad_lin_in, lin_in_s=bad_lin_s)
        _, _, w_mk_kh, _ = mk.pack_inputs(pos, hid_l, w_bad_kh, st_l, g, ex, mesh=mesh)
        out_bad_kh, _ = step_fn(pos_mk, hid_mk, w_mk_kh, st_mk)
        bad_kh_f64 = np.asarray(
            kernel.unfold(out_bad_kh, R)[0, :b_cnt], dtype=np.float64
        )
        assert float(np.max(np.abs(bad_kh_f64 - oracle_f64))) > 3.0 * max_abs_bound


def test_qwen38_decoupled_from_qwen35_and_shrunk_find_model():
    """Verify Qwen38Moe decoupling, shrunk find_model, and R=32 oracle."""
    assert issubclass(qwen38_moe.Qwen38Moe, model_def.ModelDef)
    assert not issubclass(qwen38_moe.Qwen38Moe, qwen35.Qwen35)
    assert issubclass(qwen38_moe.Qwen38MoeGeometry, mk.ModelGeometry)
    assert not issubclass(qwen38_moe.Qwen38MoeGeometry, qwen35.Qwen35Geometry)

    q38_src = Path(qwen38_moe.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(q38_src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "qwen35" not in node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "qwen35" not in alias.name

    shrunk_38 = qwen38_moe.Qwen38Moe.shrink_hf_config(periods=1)
    shrunk_35 = qwen35.Qwen35.shrink_hf_config(periods=1)
    assert find_model(shrunk_38) is qwen38_moe.Qwen38Moe
    assert find_model(shrunk_35) is qwen35.Qwen35
    shrunk_38_full = qwen38_moe.Qwen38Moe.shrink_hf_config(
        QWEN38_FULL_HF_CONFIG, periods=1
    )
    assert find_model(shrunk_38_full) is qwen38_moe.Qwen38Moe
    tc_38 = QWEN38_FULL_HF_CONFIG.get("text_config")
    manual_38 = (
        {**QWEN38_FULL_HF_CONFIG, "text_config": {**tc_38, "num_hidden_layers": 4}}
        if isinstance(tc_38, dict)
        else {**QWEN38_FULL_HF_CONFIG, "num_hidden_layers": 4}
    )
    assert find_model(manual_38) is qwen38_moe.Qwen38Moe

    flags = [
        f
        for f in os.environ.get("XLA_FLAGS", "").split()
        if not f.startswith(_DEVICE_COUNT_FLAG)
    ]
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        MALLOC_ARENA_MAX="2",
        XLA_FLAGS=" ".join([*flags, f"{_DEVICE_COUNT_FLAG}={R}"]),
    )
    out = subprocess.run(
        [sys.executable or "python3", __file__, "qwen38_r32_oracle"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "qwen38_r32":
        _step_qwen38_32_ranks_in_process()
    elif len(sys.argv) > 1 and sys.argv[1] == "qwen38_r32_oracle":
        _check_qwen38_r32_vs_unsharded_f64_in_process()


def test_replicated_reference_matches_unreplicated_reference():
    """kh_rep = sh_rep = 2 at R=8 keeps the R=4 (unreplicated) layer output."""
    g = _tiny_qwen38_r8_geometry()
    assert (g.kh_rep(8), g.sh_rep(8)) == (2, 2)
    assert (g.kh_rep(4), g.sh_rep(4)) == (1, 1)
    hf = mk.random_hf_weights(jax.random.PRNGKey(3808), g)
    tiles = dataclasses.replace(
        TINY_TILES, bank=256 << 10, hier=0, mb8=2, inplace=False
    )
    delta, w_by_r, args_by_r = {}, {}, {}
    for r in (4, 8):
        ex = ExecSpec(B=2, T=1, R=r, ctx=1024, tiles=tiles, interpret=True)
        _, hid, _, st = mk.random_logical_inputs(jax.random.PRNGKey(7), g, ex)
        st0 = jax.tree.map(jnp.zeros_like, st)
        w = _load_logical(hf, g, ex)
        pos = 900 + jnp.arange(ex.NS, dtype=I32)
        out, _ = qwen35.reference_step(g, ex, pos, hid, w, st0)
        delta[r] = out.astype(F32) - hid.astype(F32)
        w_by_r[r], args_by_r[r] = w, (g, ex, pos, hid)
    # Two arms x 2 all-reduces per layer x L layers of bf16 partial rounding.
    n_ar = 2 * g.L
    tol = 2 * n_ar * 2.0**-8 * float(jnp.max(jnp.abs(delta[4])))
    _report("replicated R=8 vs unreplicated R=4", delta[8], delta[4])
    assert float(jnp.max(jnp.abs(delta[8] - delta[4]))) <= tol
    # Negative control: drop the 1/sh_rep compensation (shared expert counted twice).
    g8, ex8, pos8, hid8 = args_by_r[8]
    w_bad = dict(w_by_r[8], sh2_s=w_by_r[8]["sh2_s"] * 2.0)
    st8 = jax.tree.map(
        jnp.zeros_like, mk.random_logical_inputs(jax.random.PRNGKey(7), g8, ex8)[3]
    )
    out_bad, _ = qwen35.reference_step(g8, ex8, pos8, hid8, w_bad, st8)
    d_bad = out_bad.astype(F32) - hid8.astype(F32)
    assert float(jnp.max(jnp.abs(d_bad - delta[4]))) > tol


def test_qwen38_default_spec_and_tp32_dcp8_profile_buckets():
    """Default `Qwen38Moe` spec and `tp32-dcp8` buckets plan within VMEM at `R=32`."""
    assert qwen38_moe.Qwen38Moe.validate_hf_config(QWEN38_FULL_HF_CONFIG).eligible
    assert qwen38_moe.Qwen38Moe.validate_hf_config(
        qwen38_moe_profile.DEFAULT_HF_CONFIG
    ).eligible

    g_def = qwen38_moe.Qwen38Moe.derive_geometry()
    assert not g_def.moe_tp
    assert g_def._model_def is qwen38_moe.Qwen38Moe
    assert qwen38_moe.Qwen38MoeGeometry()._model_def is qwen38_moe.Qwen38Moe

    ms = qwen38_moe.qwen38_moe_spec()
    assert ms.config == g_def
    prof = qwen38_moe.Qwen38Moe.get_profile("tp32-dcp8")
    assert prof.buckets == (4, 8, 16, 32, 64)
    cache_cfg = mk.CacheExec(paged_kv=True, page_size=128, num_pages=512)

    for b in prof.buckets:
        tiles_h3 = qwen38_moe_profile.tp32_dcp8_tiles(b, R=R)
        ex_h3 = ExecSpec(B=b, T=1, R=R, ctx=CTX, tiles=tiles_h3, cache=cache_cfg)
        g_def.check(ex_h3)
        cp_h3, _ = vmem.build_scratch(ms, ex_h3)
        assert (4 if b <= 32 else 0) == cp_h3.comm.HIER
        assert cp_h3.plan.module_params["gdn"].kh == 1
        assert cp_h3.plan.module_params["moe"].sh_rep == 2
        assert cp_h3.vmem_bytes <= tiles_h3.vmem_mib * 1024 * 1024

        tiles_h4 = dataclasses.replace(tiles_h3, hier=4, hier_group=8, rsag=False)
        ex_h4 = ExecSpec(B=b, T=1, R=R, ctx=CTX, tiles=tiles_h4, cache=cache_cfg)
        cp_h4, _ = vmem.build_scratch(ms, ex_h4)
        assert cp_h4.comm.HIER == 4 and cp_h4.comm.N_HOSTS == 4
        assert cp_h4.vmem_bytes <= tiles_h4.vmem_mib * 1024 * 1024


def test_qwen38_pooled_step_with_two_level_allreduce():
    """Run paged `PallasWriter` step with `hier=4` and `kh_rep = sh_rep = 2`
    on 8 ranks.
    """
    r8 = 8
    devs = jax.devices("cpu")
    assert len(devs) >= r8, f"Requires {r8} CPU devices, found {len(devs)}"
    g = qwen38_moe.Qwen38MoeGeometry(
        L=4,
        att_every=4,
        H=1024,
        NH=8,
        NKV=2,
        HD=256,
        ROT=64,
        KH=4,
        VH=8,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=512,
        moe_tp=False,
    )
    assert g._model_def is qwen38_moe.Qwen38Moe
    assert (g.kh_rep(r8), g.sh_rep(r8)) == (2, 2)
    tiles = dataclasses.replace(
        TINY_TILES,
        bank=256 << 10,
        hier=4,
        hier_group=4,
        rsag=False,
        mb8=2,
        inplace=False,
    ).with_modules({"gqa": {"split": True}})
    ex = ExecSpec(
        B=2,
        T=1,
        R=r8,
        ctx=512,
        tiles=tiles,
        cache=mk.CacheExec(paged_kv=True, page_size=128, num_pages=96),
        interpret=True,
    )
    g.check(ex)
    mesh = Mesh(np.array(devs[:r8]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))

    k_in, k_w = jax.random.split(jax.random.PRNGKey(3812))
    _, hid_l, _, st_l = mk.random_logical_inputs(k_in, g, ex)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    w_l = _load_logical(mk.random_hf_weights(k_w, g), g, ex)
    pos = 140 + jnp.arange(ex.NS, dtype=I32)

    ref_hid, ref_st = qwen38_moe.Qwen38Moe.reference_step(
        g, ex, pos, hid_l, w_l, st_l, paged_kv=True
    )
    pos_mk, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    raw_pools = tuple(
        jax.device_put(r, tp) for r in mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)
    )
    spec = g.to_model_spec(R=r8)
    pooled_step, _ = mk.PallasWriter().make_step(mesh, spec, ex, donate=False)
    out_hid, raw_out = pooled_step(pos_mk, hid_mk, w_mk, raw_pools, pt, ckpt, roff)

    hid_r = kernel.unfold(out_hid, r8)[:, : ex.B]
    for r in range(1, r8):
        assert jnp.array_equal(hid_r[r], hid_r[0]), f"hidden differs on rank {r}"
    _check("pooled hidden", hid_r[0], ref_hid, [("input hidden", hid_l)])

    st_after = mk.pool_to_states(g, ex, raw_out, pt, ckpt)
    _check(
        "pooled rec",
        st_after["rec"],
        ref_st["rec"],
        [("ranks 0 and 1 swapped", _swap(st_after["rec"], 0, 1))],
    )
    qk_cols = 2 * g.kh(r8) * g.KD
    assert spec.cache_backend is not None
    qkv_w = spec.cache_backend.layout.qkv_per_rank(r8)
    ref_conv = (
        ref_st["conv"][:, :, : 3 * ex.B, :]
        .reshape(r8, g.NGDN, 3, ex.B, qkv_w)
        .swapaxes(2, 3)
    )
    got_conv = st_after["conv"][:, :, :, 0]
    assert jnp.array_equal(
        got_conv[0, :, :, :, :qk_cols], got_conv[1, :, :, :, :qk_cols]
    )
    _check(
        "pooled conv",
        got_conv,
        ref_conv,
        [("ranks 0 and 2 swapped", _swap(got_conv, 0, 2))],
    )
    for key in ("kc", "vc"):
        for b in range(ex.B):
            p_idx = int(pos[b])
            got_kv = st_after[key][:, :, b, p_idx, :].astype(F32)
            exp_kv = ref_st[key][:, :, b, p_idx, :].astype(F32)
            assert jnp.allclose(got_kv, exp_kv, atol=1e-1, rtol=1e-1)
            assert not jnp.allclose(_swap(got_kv, 0, 4), exp_kv, atol=1e-1, rtol=1e-1)


def test_qwen38_rejects_invalid_ranks_and_sharding():
    """Reject unsupported rank counts, unaligned TP slices, and invalid
    `hier=4` groups.
    """
    g = qwen38_moe.Qwen38Moe.derive_geometry()
    ms = qwen38_moe.qwen38_moe_spec()
    ex32 = ExecSpec(B=4, T=1, R=R, ctx=CTX)

    # Routed-expert TP at R=32 shrinks I=2048 to IR=64 (< 128-column tile minimum).
    with pytest.raises(
        MegakernelConfigError, match="Dimension 64 must be a multiple of 128"
    ):
        qwen38_moe.FULL.check(ex32)

    # Non-power-of-two R=24 and R=128 > NH=64 fail geometry checks.
    with pytest.raises(MegakernelConfigError):
        g.check(ExecSpec(B=4, T=1, R=24, ctx=CTX))
    with pytest.raises(MegakernelConfigError, match="incompatible with R=128"):
        g.check(ExecSpec(B=4, T=1, R=128, ctx=CTX))

    # Indivisible key/value head counts or shared-expert widths at R=32 fail closed.
    with pytest.raises(MegakernelConfigError, match="Invalid GDN heads"):
        dataclasses.replace(g, KH=12, VH=96).check(ex32)
    with pytest.raises(MegakernelConfigError, match="Invalid GDN heads"):
        dataclasses.replace(g, VH=16).check(ex32)
    with pytest.raises(
        ValueError, match="Rank count R=32 must be divisible by shard count 12"
    ):
        shard_count(R, 12)
    with pytest.raises(
        MegakernelConfigError, match="Dimension 96 must be a multiple of 128"
    ):
        dataclasses.replace(g, IS=1536).check(ex32)
    with pytest.raises(MegakernelConfigError, match="Invalid IS=768"):
        dataclasses.replace(g, IS=768).check(ex32)

    # Invalid hier=4 group sizes fail comm_params validation.
    base_tiles = qwen38_moe_profile.tp32_dcp8_tiles(4, R=R)
    with pytest.raises(MegakernelConfigError, match="unsupported all-reduce mode"):
        vmem.build_scratch(
            ms,
            ExecSpec(
                B=4,
                T=1,
                R=R,
                ctx=CTX,
                tiles=dataclasses.replace(
                    base_tiles, hier=4, hier_group=64, rsag=False
                ),
            ),
        )
    with pytest.raises(MegakernelConfigError):
        vmem.build_scratch(
            ms,
            ExecSpec(
                B=4,
                T=1,
                R=R,
                ctx=CTX,
                tiles=dataclasses.replace(base_tiles, hier=4, hier_group=6, rsag=False),
            ),
        )


def test_qwen38_rejects_invalid_hf_and_cache_configs():
    """Reject invalid paged-KV settings, incompatible HF configs, and
    duplicate signatures.
    """
    g = qwen38_moe.Qwen38Moe.derive_geometry()
    ex32 = ExecSpec(B=4, T=1, R=R, ctx=CTX)

    with pytest.raises(MegakernelConfigError, match="Invalid paged_kv config"):
        g.check(
            ExecSpec(
                B=4,
                T=1,
                R=R,
                ctx=CTX,
                cache=mk.CacheExec(paged_kv=True, page_size=384, num_pages=64),
            )
        )
    with pytest.raises(MegakernelConfigError, match="Invalid paged_kv config"):
        g.check(
            ExecSpec(
                B=4,
                T=1,
                R=R,
                ctx=CTX,
                cache=mk.CacheExec(paged_kv=True, page_size=128, num_pages=0),
            )
        )
    with pytest.raises(MegakernelConfigError, match="hd=256"):
        qwen38_moe.Qwen38Moe.pool_layout(
            dataclasses.replace(g, HD=128, ROT=32), R=R
        ).validate(ex32)

    bad_rope = copy.deepcopy(QWEN38_FULL_HF_CONFIG)
    bad_rope["text_config"]["rope_parameters"]["rope_type"] = "yarn"
    with pytest.raises(MegakernelConfigError, match="unsupported rope_parameters"):
        qwen38_moe.Qwen38Moe.derive_geometry(bad_rope)
    elig_rope = qwen38_moe.Qwen38Moe.validate_hf_config(bad_rope)
    assert not elig_rope.eligible and elig_rope.reason == "invalid_geometry"

    elig_dense = qwen38_moe.Qwen38Moe.validate_hf_config(
        dict(qwen38_moe_profile.DEFAULT_HF_CONFIG, num_experts=0)
    )
    assert not elig_dense.eligible and elig_dense.reason == "non_moe_checkpoint"

    elig_q35 = qwen38_moe.Qwen38Moe.validate_hf_config(qwen35_profile.DEFAULT_HF_CONFIG)
    assert not elig_q35.eligible and elig_q35.reason == "geometry_mismatch"
    assert find_model(qwen35_profile.DEFAULT_HF_CONFIG) is qwen35.Qwen35

    snap = (
        dict(model_def._MODEL_REGISTRY),
        dict(model_def._ARCH_REGISTRY),
        dict(model_def._MODEL_TYPE_REGISTRY),
        dict(profile._REGISTRY),
        dict(profile._MODEL_DEFAULTS),
    )
    with pytest.raises(MegakernelConfigError, match="already registered"):

        @register_model(
            name="qwen3_8_moe_dup",
            hf_architectures=("Qwen3_5MoeForCausalLM",),
            hf_model_type=("qwen3_5_moe_text",),
        )
        class _Collision(qwen38_moe.Qwen38Moe):
            profiles = {}

    after = (
        model_def._MODEL_REGISTRY,
        model_def._ARCH_REGISTRY,
        model_def._MODEL_TYPE_REGISTRY,
        profile._REGISTRY,
        profile._MODEL_DEFAULTS,
    )
    assert after == snap

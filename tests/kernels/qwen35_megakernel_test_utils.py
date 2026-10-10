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
"""Shared fixtures and reference helpers for Qwen3.5 megakernel unit tests."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import torch

from vllm_torchtpu.kernels.qwen35_megakernel import (
    MKBuildContext,
    TINY,
    TINY_TILES,
    megakernel as K,
    vllm_adapter as VA,
)
from vllm_torchtpu.kernels.qwen35_megakernel.megakernel import (
    BF16,
    F32,
    I32,
    rnd,
    smalls_layout,
    state_shapes,
    weight_shapes,
)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

try:
    jax.config.update("jax_num_cpu_devices", 8)
except RuntimeError:
    pass

POOL_IO_ENV = (
    "VLLM_TPU_QWEN35_MK_FUSED_POOL_IO",
    "VLLM_TPU_QWEN35_MK_FUSED_POOL_IO_MAX_B",
    "VLLM_TPU_QWEN35_MK_FUSED_POOL_IO_MIN_B_HI",
)


def setup_8_cpu_devices():
    """Return all available CPU devices (1 or 8)."""
    return jax.devices("cpu")


def bare_mk(monkeypatch, **attrs) -> VA.Qwen35Megakernel:
    """Create a bare `Qwen35Megakernel` instance without `__init__` (no runner / model / TPU)."""
    for name in POOL_IO_ENV:
        monkeypatch.delenv(name, raising=False)
    mk = object.__new__(VA.Qwen35Megakernel)
    mk.fused_pool_io = True
    mk.fused_pool_io_max_b = VA.parse_mk_fused_pool_io_max_b()
    mk.fused_pool_io_min_b_hi = VA.parse_mk_fused_pool_io_min_b_hi()
    mk._bucket_steps, mk._bucket_ops, mk._bucket_out_rows = {}, {}, {}
    mk._bucket_fused, mk._prep_steps = {}, {}
    for k, v in attrs.items():
        setattr(mk, k, v)
    return mk


def make_test_ctx_and_inputs(seed: int = 42):
    """Construct a fast single-rank TINY config, MKBuildContext, and synthetic weights/states."""
    c = dataclasses.replace(
        TINY,
        R=1,
        B=4,
        T=1,
        L=4,
        att_every=4,
        NH=2,
        NKV=1,
        KH=2,
        VH=4,
        E=128,
        TOPK=2,
        I=128,
        IS=128,
        stack_e=True,
        paged_kv=False,
        fused_pool_io=False,
    )
    tiles = dataclasses.replace(
        TINY_TILES,
        bank=256 << 10,
        nbd=20,
        nbe=8,
        lazy=False,
        inplace=False,
        pack=0,
        pair_kv=False,
        pipe=False,
    )
    ctx = MKBuildContext(c, tiles, interpret=True)
    w_sh = weight_shapes(c, tiles)
    st_sh = state_shapes(c)
    lay, _ = smalls_layout(c)

    rng = jax.random.PRNGKey(seed)
    w_vals = {}
    for idx, (k, (sh, dt)) in enumerate(w_sh.items()):
        k_rng = jax.random.fold_in(rng, idx)
        val = jax.random.normal(k_rng, sh, F32) * 0.05
        if k in ("es", "smalls", "final_norm"):
            val = jnp.abs(val) + 0.5
            if k == "smalls":
                ow_a, _ = lay["alog"]
                val = val.at[:, :, ow_a : ow_a + 128].set(-1.0)
        w_vals[k] = val.astype(dt)

    st_vals = {}
    for idx, (k, (sh, dt)) in enumerate(st_sh.items()):
        k_rng = jax.random.fold_in(rng, 100 + idx)
        st_vals[k] = (jax.random.normal(k_rng, sh, F32) * 0.1).astype(dt)

    pos = jnp.array([12, 25, 7, 40], dtype=I32)
    hid = (
        jax.random.normal(jax.random.fold_in(rng, 200), (c.M, c.H), F32) * 0.2
    ).astype(BF16)
    hid = hid.at[c.B :].set(0)
    return c, tiles, ctx, w_sh, st_sh, lay, w_vals, st_vals, pos, hid


def rms_ref(x, w, eps: float = 1e-6):
    v = x * lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return rnd(v * (1.0 + w))


def ref_dmatmul(x_bf16, w_tiled, mb: int = 16):
    """Exact reference for DenseWeightRing.dmatmul in interpret mode (including MB row padding)."""
    b_rows = x_bf16.shape[0]
    if b_rows < mb:
        x_pad = jnp.pad(x_bf16, ((0, mb - b_rows), (0, 0)))
    else:
        x_pad = x_bf16
    nt, tk, n_cols = w_tiled.shape
    dcw = 512
    chunks = []
    for c0 in range(0, n_cols, dcw):
        cw = min(dcw, n_cols - c0)
        acc = jnp.zeros((x_pad.shape[0], cw), F32)
        for t in range(nt):
            wt = w_tiled[t, :, c0 : c0 + cw].astype(x_pad.dtype)
            acc = acc + lax.dot_general(
                x_pad[:, t * tk : (t + 1) * tk],
                wt,
                (((1,), (0,)), ((), ())),
                preferred_element_type=F32,
            )
        chunks.append(acc[:b_rows])
    return jnp.concatenate(chunks, axis=-1)


def make_fp8_linear(in_dim: int, out_dim: int, flipped: bool = True) -> SimpleNamespace:
    if flipped:
        w = torch.randint(0, 120, (in_dim, out_dim), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    else:
        w = torch.randint(0, 120, (out_dim, in_dim), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    s = torch.rand((out_dim, 1), dtype=torch.float32) + 0.1
    ns = SimpleNamespace(
        weight=torch.nn.Parameter(w, requires_grad=False),
        weight_scale=torch.nn.Parameter(s, requires_grad=False),
    )
    setattr(ns, WEIGHT_FLIPPED_ATTR, flipped)
    return ns


def make_tiny_backbone(c: K.MKConfig, e_slots: int) -> SimpleNamespace:
    layers = []
    for l in range(c.L):
        is_att = (l % c.att_every) == (c.att_every - 1)
        ltype = "full_attention" if is_att else "linear_attention"

        if is_att:
            self_attn = SimpleNamespace(
                q_proj=make_fp8_linear(c.H, c.nh * 2 * c.HD),
                k_proj=make_fp8_linear(c.H, c.HD),
                v_proj=make_fp8_linear(c.H, c.HD),
                o_proj=make_fp8_linear(c.nh * c.HD, c.H),
                q_norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn(c.HD, dtype=torch.bfloat16), requires_grad=False
                    )
                ),
                k_norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn(c.HD, dtype=torch.bfloat16), requires_grad=False
                    )
                ),
                attn=SimpleNamespace(
                    layer_name=f"model.layers.{l}.self_attn.attn", kv_cache=[]
                ),
            )
            linear_attn = None
        else:
            in_qkvz = make_fp8_linear(c.H, c.QKV + c.Z)
            in_ba = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((2 * c.vh, c.H), dtype=torch.bfloat16) * 0.05,
                    requires_grad=False,
                )
            )
            linear_attn = SimpleNamespace(
                prefix=f"model.layers.{l}.linear_attn",
                in_proj_qkvz=in_qkvz,
                in_proj_ba=in_ba,
                out_proj=make_fp8_linear(c.Z, c.H),
                conv1d=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn((c.QKV, 1, 4), dtype=torch.bfloat16),
                        requires_grad=False,
                    )
                ),
                A_log=torch.nn.Parameter(
                    torch.randn(c.vh, dtype=torch.float32), requires_grad=False
                ),
                dt_bias=torch.nn.Parameter(
                    torch.randn(c.vh, dtype=torch.bfloat16), requires_grad=False
                ),
                norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn(c.VD, dtype=torch.float32), requires_grad=False
                    )
                ),
                kv_cache=[],
            )
            self_attn = None

        w13 = torch.randint(
            0, 120, (e_slots, c.H, 2 * c.IR), dtype=torch.uint8
        ).view(torch.float8_e4m3fn)
        w2 = torch.randint(0, 120, (e_slots, c.IR, c.H), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        w13_s = torch.rand((e_slots, 1, 2 * c.IR), dtype=torch.float32) + 0.1
        w2_s = torch.rand((e_slots, 1, c.H), dtype=torch.float32) + 0.1
        experts = SimpleNamespace(
            w13_weight=torch.nn.Parameter(w13, requires_grad=False),
            w2_weight=torch.nn.Parameter(w2, requires_grad=False),
            w13_weight_scale=torch.nn.Parameter(w13_s, requires_grad=False),
            w2_weight_scale=torch.nn.Parameter(w2_s, requires_grad=False),
        )

        fused_gu_cols = c.E + 2 * c.IS
        fused_w = torch.randint(
            0, 120, (c.H, fused_gu_cols), dtype=torch.uint8
        ).view(torch.float8_e4m3fn)
        fused_s = torch.rand((1, fused_gu_cols), dtype=torch.float32) + 0.1
        shared_expert = SimpleNamespace(
            down_proj=make_fp8_linear(c.IS, c.H),
        )
        shared_expert_gate = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.randn((1, c.H), dtype=torch.bfloat16), requires_grad=False
            )
        )
        mlp = SimpleNamespace(
            experts=experts,
            _fused_gate_gu_fp8_weight=fused_w,
            _fused_gate_gu_fp8_scale=fused_s,
            shared_expert=shared_expert,
            shared_expert_gate=shared_expert_gate,
        )

        layer = SimpleNamespace(
            layer_type=ltype,
            self_attn=self_attn,
            linear_attn=linear_attn,
            mlp=mlp,
            input_layernorm=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn(c.H, dtype=torch.bfloat16), requires_grad=False
                )
            ),
            post_attention_layernorm=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn(c.H, dtype=torch.bfloat16), requires_grad=False
                )
            ),
        )
        layers.append(layer)

    return SimpleNamespace(
        layers=layers,
        norm=SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.randn(c.H, dtype=torch.bfloat16), requires_grad=False
            )
        ),
        embed_tokens=lambda ids: torch.zeros(
            (ids.shape[0], c.H), dtype=torch.bfloat16, device=ids.device
        ),
    )


def jnp_leaves(packed):
    return [
        jnp.asarray(packed[k].view(torch.uint8).numpy()).view(
            K.FP8
            if packed[k].dtype == torch.float8_e4m3fn
            else (jnp.bfloat16 if packed[k].dtype == torch.bfloat16 else jnp.float32)
        )
        for k in K.W_NAMES
    ]


def unpack_rec_checkpoint(
    raw_pages_u8: np.ndarray, vh: int, kd: int = 128, vd: int = 128
):
    """Unpack `(rec_pages, 2, 64, 4, 128) uint8` into `(vh, kd, vd) float32` via Pallas layout."""
    ssm_u32_rows = (vh * kd) // 2
    rec_pages = raw_pages_u8.shape[0]
    u32_val = (
        np.swapaxes(raw_pages_u8, -2, -1)
        .copy()
        .view(np.uint32)
        .reshape(-1, rec_pages * 128, 128)[:, :ssm_u32_rows, :]
    )
    bf16_val = np.swapaxes(
        u32_val[..., None].view(jnp.dtype(K.BF16)), -2, -1
    ).reshape(-1, vh, kd, vd)
    return bf16_val.astype(np.float32)


def unpack_conv_checkpoint(raw_page_u8: np.ndarray, qkv: int):
    """Unpack `(..., 2, 64, 4, 128) uint8` into `(..., 3, qkv) float32` via Pallas layout."""
    conv_rows = (3 * qkv) // 128
    bf16_256 = np.swapaxes(
        np.swapaxes(raw_page_u8, -2, -1).copy().view(jnp.dtype(K.BF16)), -2, -1
    ).reshape(-1, 256, 128)
    return bf16_256[:, :conv_rows, :].reshape(-1, 3, qkv).astype(np.float32)

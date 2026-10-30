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
"""A tiny Qwen3.5 geometry for CPU tests: the geometry and `ExecSpec`, a
synthetic vLLM-like PyTorch backbone, and its aliased KV+GDN raw pools."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh

from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.modules.gdn import GDNModule
from vllm_torchtpu.megakernel.pool.config import CacheExec
from vllm_torchtpu.megakernel_runtime.weight_collector import WEIGHT_FLIPPED_ATTR

from .. import model
from ..profiles.baseline import TINY_TILES


def cpu_mesh(r: int = 1) -> Mesh:
    return Mesh(np.array(jax.devices("cpu")[:r]), ("tp",))


def tiny_geom(L: int = 4, att_every: int = 4) -> model.Qwen35Geometry:
    return model.Qwen35Geometry(
        L=L,
        att_every=att_every,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=0,
    )


def tiny_exec(
    R: int = 1,
    B: int = 2,
    T: int = 1,
    ctx: int = 256,
    num_pages: int = 96,
) -> ExecSpec:
    tiles = replace(TINY_TILES, bank=512 << 10).with_modules(
        {"gqa": {"kv_blk": min(TINY_TILES.module("gqa").kv_blk, ctx)}}
    )
    return ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        stack_e=False,
        tiles=tiles,
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=num_pages, max_pages_per_seq=2),
    )


def allocate_aliased_pools(
    backbone: Any, g: model.Qwen35Geometry, num_pages: int
) -> list[torch.Tensor]:
    raw_pools_t: list[torch.Tensor] = []
    for r in range(g.NATT):
        pool_u8 = torch.zeros((num_pages, 2, 64, 4, 128), dtype=torch.uint8)
        pool = pool_u8.view(torch.float8_e4m3fn)
        att_layer = backbone.layers[g.att_every * r + 3]
        att_layer.self_attn.attn.kv_cache = [pool]
        for g_idx in range(3):
            la_layer = backbone.layers[g.att_every * r + g_idx]
            la_layer.linear_attn.kv_cache = [pool]
        raw_pools_t.append(pool)
    return raw_pools_t


def jax_to_torch(a: jax.Array) -> torch.Tensor:
    arr = np.asarray(a)
    if a.dtype == jnp.float8_e4m3fn:
        return torch.from_numpy(arr.view(np.uint8).copy()).view(torch.float8_e4m3fn)
    if a.dtype == jnp.bfloat16:
        return torch.from_numpy(arr.view(np.uint16).copy()).view(torch.bfloat16)
    return torch.from_numpy(arr.copy())


def make_fp8_linear(
    in_dim: int, out_dim: int, *, flipped: bool = True
) -> SimpleNamespace:
    if flipped:
        w = torch.randint(1, 120, (in_dim, out_dim), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    else:
        w = torch.randint(1, 120, (out_dim, in_dim), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    s = torch.rand((out_dim, 1), dtype=torch.float32) + 0.1
    ns = SimpleNamespace(
        weight=torch.nn.Parameter(w, requires_grad=False),
        weight_scale=torch.nn.Parameter(s, requires_grad=False),
    )
    setattr(ns, WEIGHT_FLIPPED_ATTR, flipped)
    return ns


def make_synthetic_backbone(
    g: model.Qwen35Geometry,
    ex: ExecSpec,
    *,
    fused_mlp: bool = True,
    fused_qkv: bool = False,
    e_cores: list[list[list[int]]] | None = None,
) -> SimpleNamespace:
    """Construct a synthetic Qwen3.5 backbone matching `g` and `ex`."""
    gdn_lay = GDNModule.lin_in_layout(g, ex)
    nh = g.nh(ex.R)
    nai = g.NAI(ex.R)
    q_dim = 2 * nh * g.HD
    kv_dim = (nai - q_dim) // 2
    ew = g.EW(ex.R)
    ir = g.IR(ex.R)
    layers = []

    for l_idx in range(g.L):
        is_att = (l_idx % g.att_every) == (g.att_every - 1)
        ltype = "full_attention" if is_att else "linear_attention"

        if is_att:
            if fused_qkv:
                self_attn = SimpleNamespace(
                    qkv_proj=make_fp8_linear(g.H, nai),
                    o_proj=make_fp8_linear(nh * g.HD, g.H),
                    q_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    k_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    attn=SimpleNamespace(
                        layer_name=f"model.layers.{l_idx}.self_attn.attn", kv_cache=[]
                    ),
                )
            else:
                self_attn = SimpleNamespace(
                    q_proj=make_fp8_linear(g.H, q_dim),
                    k_proj=make_fp8_linear(g.H, kv_dim, flipped=False),
                    v_proj=make_fp8_linear(g.H, kv_dim),
                    o_proj=make_fp8_linear(nh * g.HD, g.H),
                    q_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    k_norm=SimpleNamespace(
                        weight=torch.nn.Parameter(
                            torch.randn(g.HD, dtype=torch.bfloat16), requires_grad=False
                        )
                    ),
                    attn=SimpleNamespace(
                        layer_name=f"model.layers.{l_idx}.self_attn.attn", kv_cache=[]
                    ),
                )
            linear_attn = None
        else:
            in_qkvz = make_fp8_linear(g.H, gdn_lay.QKV + gdn_lay.Z)
            in_ba = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((2 * gdn_lay.vh, g.H), dtype=torch.bfloat16) * 0.05,
                    requires_grad=False,
                )
            )
            linear_attn = SimpleNamespace(
                prefix=f"model.layers.{l_idx}.linear_attn",
                in_proj_qkvz=in_qkvz,
                in_proj_ba=in_ba,
                out_proj=make_fp8_linear(gdn_lay.Z, g.H),
                conv1d=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn((gdn_lay.QKV, 1, 4), dtype=torch.bfloat16),
                        requires_grad=False,
                    )
                ),
                A_log=torch.nn.Parameter(
                    torch.randn(gdn_lay.vh, dtype=torch.float32), requires_grad=False
                ),
                dt_bias=torch.nn.Parameter(
                    torch.randn(gdn_lay.vh, dtype=torch.bfloat16), requires_grad=False
                ),
                norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        torch.randn(gdn_lay.VD, dtype=torch.float32),
                        requires_grad=False,
                    )
                ),
                kv_cache=[],
            )
            self_attn = None

        w13 = torch.randint(1, 120, (ew, g.H, 2 * ir), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        w2 = torch.randint(1, 120, (ew, ir, g.H), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        w13_s = torch.rand((ew, 1, 2 * ir), dtype=torch.float32) + 0.1
        w2_s = torch.rand((ew, 1, g.H), dtype=torch.float32) + 0.1
        experts = SimpleNamespace(
            w13_weight=torch.nn.Parameter(w13, requires_grad=False),
            w2_weight=torch.nn.Parameter(w2, requires_grad=False),
            w13_weight_scale=torch.nn.Parameter(w13_s, requires_grad=False),
            w2_weight_scale=torch.nn.Parameter(w2_s, requires_grad=False),
        )
        if e_cores is not None:
            experts._ecopy_cores = e_cores[l_idx]

        f_r = g.ISR(ex.R)
        if fused_mlp:
            fused_gu_cols = g.E + 2 * g.IS
            fused_w = torch.randint(
                1, 120, (g.H, fused_gu_cols), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            fused_s = torch.rand((1, fused_gu_cols), dtype=torch.float32) + 0.1
            se_down_w = torch.randint(1, 120, (g.IS, g.H), dtype=torch.uint8).view(
                torch.float8_e4m3fn
            )
            se_down_s = torch.rand((1, g.H), dtype=torch.float32) + 0.1
            se_gate_w = torch.randn((1, g.H), dtype=torch.bfloat16)
            mlp = SimpleNamespace(
                experts=experts,
                _fused_gate_gu_fp8_weight=fused_w,
                _fused_gate_gu_fp8_scale=fused_s,
                _fused_se_down_fp8_weight=se_down_w,
                _fused_se_down_fp8_scale=se_down_s,
                _fused_se_gate_weight=se_gate_w,
            )
        else:
            gate = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((g.E, g.H), dtype=torch.bfloat16), requires_grad=False
                )
            )
            shared_expert = SimpleNamespace(
                gate_up_proj=make_fp8_linear(g.H, 2 * f_r),
                down_proj=make_fp8_linear(f_r, g.H, flipped=False),
            )
            shared_expert_gate = SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((1, g.H), dtype=torch.bfloat16), requires_grad=False
                )
            )
            mlp = SimpleNamespace(
                experts=experts,
                gate=gate,
                shared_expert=shared_expert,
                shared_expert_gate=shared_expert_gate,
            )

        layer = SimpleNamespace(
            layer_type=ltype,
            input_layernorm=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn(g.H, dtype=torch.bfloat16), requires_grad=False
                )
            ),
            post_attention_layernorm=SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn(g.H, dtype=torch.bfloat16), requires_grad=False
                )
            ),
            linear_attn=linear_attn,
            self_attn=self_attn,
            mlp=mlp,
        )
        layers.append(layer)

    embed = torch.nn.Embedding(64, g.H, dtype=torch.bfloat16)
    norm = SimpleNamespace(
        weight=torch.nn.Parameter(
            torch.randn(g.H, dtype=torch.bfloat16), requires_grad=False
        )
    )
    return SimpleNamespace(embed_tokens=embed, layers=layers, norm=norm)

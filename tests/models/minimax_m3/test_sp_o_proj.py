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
"""sp_o_proj against the reduce-scatter of o_proj's per-rank partial sums it
replaces: prepare's rescaled weight on the plain path, and the all-to-all
path for every rank's row shard, bf16 and e4m3, gather and replicate."""

import types

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch import nn

from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.models.minimax_m3 import quant, sp_o_proj

TP, T, K, N = 8, 64, 128, 256


def _jax(t: torch.Tensor):
    if t.dtype == torch.float8_e4m3fn:
        return jnp.asarray(t.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _torch(a) -> torch.Tensor:
    if a.dtype == jnp.float8_e4m3fn:
        return torch.from_numpy(np.array(a.view(jnp.uint8))).view(torch.float8_e4m3fn)
    return torch.from_numpy(np.array(a.astype(jnp.float32)))


@pytest.fixture
def ranks():
    gen = torch.Generator().manual_seed(0)
    attn = [(torch.randn(T, K, generator=gen) * (1 + r)).bfloat16() for r in range(TP)]
    w8 = [torch.randn(K, N, generator=gen).to(torch.float8_e4m3fn) for _ in range(TP)]
    scale = [
        torch.exp2(torch.randint(-8, -2, (N,), generator=gen).float()).bfloat16()
        for _ in range(TP)
    ]
    ref = sum(a.float() @ (w.float() * s.float()) for a, w, s in zip(attn, w8, scale))
    return attn, w8, scale, ref


def _prepared(monkeypatch, mode, w8, scale):
    """Every rank's o_proj after prepare(), with the gloo gathers emulated."""
    monkeypatch.setenv("TPU_MINIMAX_M3_SP_A2A_OPROJ", mode)
    common = torch.stack(scale).float().amax(0)
    rescaled = [
        (w.float() * (s.float() / common)).to(torch.float8_e4m3fn)
        for w, s in zip(w8, scale)
    ]
    mods = []
    for r in range(TP):
        m = nn.Module()
        m.weight = nn.Parameter(w8[r].clone(), requires_grad=False)
        m.weight_scale = nn.Parameter(scale[r].clone(), requires_grad=False)
        setattr(m, WEIGHT_FLIPPED_ATTR, True)
        group = types.SimpleNamespace(
            world_size=TP, ranks=list(range(TP)), rank_in_group=r
        )
        monkeypatch.setattr(sp_o_proj, "get_tp_group", lambda g=group: g)
        gathers = iter(
            [
                torch.stack([s.float() for s in scale]),
                torch.stack([x.view(torch.int32) for x in rescaled]),
            ]
        )
        monkeypatch.setattr(sp_o_proj, "_all_gather_cpu", lambda t, g=gathers: next(g))
        sp_o_proj.prepare(m)
        _as_rank(monkeypatch, r)
        assert sp_o_proj.applies(m, 8192)
        mods.append(m)
    return mods, rescaled


def _as_rank(monkeypatch, r):
    group = types.SimpleNamespace(world_size=TP, ranks=list(range(TP)), rank_in_group=r)
    monkeypatch.setattr(sp_o_proj, "get_tp_group", lambda: group)
    rank = torch.tensor([[r]], dtype=torch.int32)
    monkeypatch.setattr(sp_o_proj.pallas_all_reduce, "armed_rank", lambda ranks: rank)


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.float() - want).norm() / want.norm())


def test_needs_the_armed_rank(monkeypatch, ranks):
    _, w8, scale, _ = ranks
    mods, _ = _prepared(monkeypatch, "gather", w8, scale)
    monkeypatch.setattr(sp_o_proj.pallas_all_reduce, "armed_rank", lambda ranks: None)
    assert not sp_o_proj.applies(mods[0], 8192)


def test_small_steps_reduce_scatter(monkeypatch, ranks):
    _, w8, scale, _ = ranks
    mods, _ = _prepared(monkeypatch, "gather", w8, scale)
    monkeypatch.setenv("TPU_MINIMAX_M3_SP_A2A_OPROJ_MIN_TOKENS", "1024")
    assert not sp_o_proj.applies(mods[0], 384)
    assert sp_o_proj.applies(mods[0], 1024)


@pytest.mark.parametrize("mode", ["gather", "replicate"])
def test_plain_path_reads_the_rescaled_block(monkeypatch, ranks, mode):
    attn, w8, scale, ref = ranks
    mods, _ = _prepared(monkeypatch, mode, w8, scale)
    method = quant.MiniMaxM3Mxfp8LinearMethod(keep_fp8=True)
    partials = 0
    for r, m in enumerate(mods):
        _as_rank(monkeypatch, r)
        partials = partials + method.apply(m, attn[r]).float()
    assert _rel(partials, ref) < 5e-3


@pytest.mark.parametrize("mode", ["gather", "replicate"])
@pytest.mark.parametrize("e4m3, tol", [(False, 5e-3), (True, 5e-2)])
def test_all_to_all_path_gives_each_rank_its_rows(monkeypatch, ranks, mode, e4m3, tol):
    attn, w8, scale, ref = ranks
    mods, _ = _prepared(monkeypatch, mode, w8, scale)
    monkeypatch.setenv("TPU_MINIMAX_M3_SP_A2A_E4M3", "1" if e4m3 else "0")
    monkeypatch.setattr(
        sp_o_proj,
        "tensor_model_parallel_all_gather",
        lambda w, dim: torch.cat([m.weight for m in mods], 0),
    )
    monkeypatch.setattr(
        sp_o_proj,
        "_quantize_op",
        lambda x: tuple(_torch(a) for a in sp_o_proj._quantize_rows_pow2(_jax(x))),
    )
    monkeypatch.setattr(
        sp_o_proj,
        "_o_proj_e4m3_op",
        lambda q, r, w, s: _torch(sp_o_proj._o_proj_e4m3(*map(_jax, (q, r, w, s)))),
    )
    rows = T // TP
    for r in range(TP):
        _as_rank(monkeypatch, r)
        sent = [sp_o_proj._quantize_op(a) if e4m3 else (a,) for a in attn]

        def all_to_all_rows(x, r=r, sent=sent):
            part = 1 if x.shape[-1] == 1 else 0
            return torch.cat([s[part][r * rows : (r + 1) * rows] for s in sent])

        monkeypatch.setattr(sp_o_proj, "all_to_all_rows", all_to_all_rows)
        got = sp_o_proj.sp_o_proj(mods[r], attn[r])
        assert got.shape == (rows, N)
        assert _rel(got, ref[r * rows : (r + 1) * rows]) < tol


def test_armed_rank_is_decided_for_the_whole_group(monkeypatch):
    """A mesh in device-id order (local ranks on devices 0,1,4,5,6,7,2,3):
    ranks whose own index matches must not take the path alone."""
    par = sp_o_proj.pallas_all_reduce
    seen = [{"mesh_index": m} for m in (0, 1, 4, 5, 6, 7, 2, 3)]
    for permuted, want in (
        (seen, False),
        (sorted(seen, key=lambda s: s["mesh_index"]), True),
    ):
        for r in range(TP):
            armed = par._Armed(
                ranks=tuple(range(TP)),
                op=None,
                rank=torch.tensor([[r]], dtype=torch.int32),
                rank_ordered=all(s["mesh_index"] == i for i, s in enumerate(permuted)),
            )
            monkeypatch.setattr(par, "_ARMED", armed)
            assert (par.armed_rank(tuple(range(TP))) is not None) == want


@pytest.mark.parametrize("pieces", [2, 4])
def test_all_to_all_pieces_give_each_rank_its_rows(monkeypatch, ranks, pieces):
    attn, w8, scale, ref = ranks
    mods, _ = _prepared(monkeypatch, "gather", w8, scale)
    monkeypatch.setenv("TPU_MINIMAX_M3_SP_A2A_E4M3", "1")
    monkeypatch.setattr(
        sp_o_proj,
        "tensor_model_parallel_all_gather",
        lambda w, dim: torch.cat([m.weight for m in mods], 0),
    )
    monkeypatch.setattr(
        sp_o_proj,
        "_quantize_op",
        lambda x: tuple(_torch(a) for a in sp_o_proj._quantize_rows_pow2(_jax(x))),
    )
    monkeypatch.setattr(
        sp_o_proj,
        "_o_proj_e4m3_op",
        lambda q, r, w, s: _torch(sp_o_proj._o_proj_e4m3(*map(_jax, (q, r, w, s)))),
    )
    rows = T // TP
    for r in range(TP):
        _as_rank(monkeypatch, r)
        sent = [sp_o_proj._quantize_op(a) for a in attn]
        calls = []

        # One all-to-all per part (rows, scales) for every piece: each source's
        # block r, all its pieces.
        def all_to_all_rows(x, r=r, sent=sent, calls=calls):
            part = 1 if x.shape[-1] == 1 else 0
            calls.append(part)
            return torch.cat([s[part].view(TP, -1, s[part].shape[-1])[r] for s in sent])

        monkeypatch.setattr(sp_o_proj, "all_to_all_rows", all_to_all_rows)
        got = sp_o_proj.sp_o_proj(mods[r], attn[r], pieces)
        assert sorted(calls) == [0, 1]
        assert got.shape == (rows, N)
        assert _rel(got, ref[r * rows : (r + 1) * rows]) < 5e-2

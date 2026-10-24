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
"""Qwen3.8 segment kernel vs the composed pure-JAX segment reference (CPU)."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import (
    pack_segment,
    random_segment_inputs,
)
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel.models.qwen38_reference import segment_reference
from vllm_torchtpu.megakernel.modules.hc import RESIDUAL_OUT

_TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
G = qwen38.TINY


def _run(
    T: int,
    key: int = 40,
    L: int = qwen38.SEGMENT_LAYERS,
    g: Any = G,
    head: bool = False,
):
    R, B = 2, 2 * T
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=_TILES)
    spec = qwen38.segment_spec(g, R, L=L, head=head)
    inp = random_segment_inputs(jax.random.PRNGKey(key), g, ex, L, head=head)
    ref, _ = segment_reference(
        g, ex, inp.x, inp.hc, inp.w_l, inp.st_l, inp.attn, inp.o_proj, L=L, head=head
    )
    w_mk, states, attn = pack_segment(spec, g, ex, inp, head=head)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, inp.pos, inp.x, w_mk, states
    )
    extra = (
        ()
        if head
        else (jax.device_put(kernel.fold(attn), NamedSharding(mesh, P("tp"))),)
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *extra)
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    jax.clear_caches()
    return res, np.asarray(ref, np.float32), np.asarray(inp.x, np.float32)


def _assert_within_budget(res: np.ndarray, ref: np.ndarray) -> None:
    """Eight BF16-rounded sublayers with BF16 all-reduce payloads: bound the error
    distribution (mean, 99.9th percentile, max) rather than every element."""
    scale = float(np.abs(ref).mean())
    for r in range(res.shape[0]):
        d = np.abs(res[r] - ref)
        assert d.mean() < 1e-2 * scale, (r, d.mean(), scale)
        assert np.quantile(d, 0.999) < 5e-2 * scale, (r, np.quantile(d, 0.999), scale)
        assert d.max() < 1e-1 * scale, (r, d.max(), scale)


@pytest.mark.parametrize("T,L", [(1, 4), (4, 4), (4, 1)])
def test_qwen38_segment_kernel_matches_reference(T: int, L: int):
    res, ref, x = _run(T, L=L)
    # Every sublayer must move the residual, or the comparison is vacuous.
    assert float(np.std(ref - x)) > 0.05
    _assert_within_budget(res, ref)


@pytest.mark.parametrize("L", [1, 2])
def test_qwen38_head_segment_matches_reference(L: int):
    """The all-GDN head segment (layers before the first QSA layer)."""
    res, ref, x = _run(4, L=L, head=True)
    assert float(np.std(ref - x)) > 0.05
    _assert_within_budget(res, ref)


def test_qwen38_segment_hidden_not_a_multiple_of_1024():
    """H = 1280 (like 2560) takes the 640-column MoE output chunk."""
    g = dataclasses.replace(G, H=1280)
    res, ref, x = _run(1, g=g, L=1)
    assert float(np.std(ref - x)) > 0.05
    _assert_within_budget(res, ref)


def test_qwen38_segment_budget_catches_a_dropped_layer(monkeypatch):
    """Positive control: the budget rejects a kernel that skips its last MoE."""
    from vllm_torchtpu.megakernel.modules.hc import HCSublayer

    orig = HCSublayer._combine

    def drop_last_moe(self, ctx, block):
        if self.name != "moe":
            return orig(self, ctx, block)
        from jax import lax

        keep = (
            ctx.stream.sp.module_ord(ctx._hc_layer, "moe") < qwen38.SEGMENT_LAYERS - 1
        )
        return lax.cond(keep, lambda: orig(self, ctx, block), lambda: None)

    orig_emit = HCSublayer.emit

    def emit(self, ctx, lyr, slot):
        ctx._hc_layer = lyr
        return orig_emit(self, ctx, lyr, slot)

    monkeypatch.setattr(HCSublayer, "_combine", drop_last_moe)
    monkeypatch.setattr(HCSublayer, "emit", emit)
    res, ref, _ = _run(1)
    with pytest.raises(AssertionError):
        _assert_within_budget(res, ref)


def test_qwen38_flat_segment_step_matches_make_step():
    """The positional bridge form returns the same residual as `make_step`."""
    from vllm_torchtpu.megakernel.models.qwen38_step import make_flat_segment_step

    T, L, R = 4, qwen38.SEGMENT_LAYERS, 2
    B = 2 * T
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=_TILES)
    spec = qwen38.segment_spec(G, R, L=L)
    inp = random_segment_inputs(jax.random.PRNGKey(7), G, ex, L)
    w_mk, states, attn = pack_segment(spec, G, ex, inp)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, inp.pos, inp.x, w_mk, states
    )
    attn_mk = jax.device_put(kernel.fold(attn), NamedSharding(mesh, P("tp")))
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, attn_mk)

    fn, in_specs, out_specs, n_w = make_flat_segment_step(mesh, spec, ex)
    flat = []
    for name in kernel.build(spec, ex)[1]["plan"].w_names:
        v = w_mk[name]
        flat.extend(v if isinstance(v, tuple) else (v,))
    assert (
        len(flat) == n_w
        and len(in_specs) == 3 + n_w + 2
        and fn.extra_names == ("qsa_attn",)
    )
    res = jax.jit(fn)(pos_mk, hid_mk, attn_mk, *flat, st_mk["conv"], st_mk["rec"])
    assert len(res) == len(out_specs) == 3
    np.testing.assert_array_equal(np.asarray(res[0]), np.asarray(outs[RESIDUAL_OUT]))
    np.testing.assert_array_equal(np.asarray(res[2]), np.asarray(outs["rec_ckpt"]))

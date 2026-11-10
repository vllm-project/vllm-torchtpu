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
"""Whole-step MK with redundant expert copies (Po2C), CPU interpret.

Split copies (`e13x` / `e2x` next to the model's own experts) must give the bytes
of the single-array copy layout; both stay inside the reference budget.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
from jax.sharding import Mesh

from vllm_torchtpu.megakernel import kernel, planner
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel.wholestep import inputs as wi
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

_TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
BUDGET, PAGES, L, R = 64, 8, 8, 2
KV_LEN = [0, 1, 9, 100, 255, 257, 130, 64]
COPIES = 8


def _run(g, key: int = 11, T: int = 4):
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        key, g, ex, L, budget=BUDGET, n_pages=PAGES, kv_len=KV_LEN,
        identical_experts=True,
    )
    ref, _ = wi.wholestep_reference(g, ex, d, L)
    spec = wholestep_spec(g, R, L=L, qsa=dict(budget=BUDGET))
    w_mk, states = wi.pack_wholestep(spec, g, ex, d, L)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos, hid, w_mk, st = shard_step_inputs(mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states)
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos, hid, w_mk, st, *wi.extras(d, mesh, R))
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, : ex.B], np.float32)
    return res, np.asarray(ref, np.float32), spec, ex


def test_split_copies_match_single_array_copies():
    g = dataclasses.replace(qwen38.TINY, e_slots=qwen38.TINY.E // R + COPIES)
    one, ref, spec1, ex = _run(g)
    split, _, spec2, _ = _run(dataclasses.replace(g, e_split=True))
    p1, p2 = planner.plan(spec1, ex), planner.plan(spec2, ex)
    assert "e13x" not in p1.weight_decls and "e13x" in p2.weight_decls
    assert p2.weight_decls["e13x"].lead_dims == (COPIES,)
    np.testing.assert_array_equal(split.view(np.uint32), one.view(np.uint32))
    scale = float(np.abs(ref).mean())
    for r in range(R):
        assert np.abs(split[r] - ref).mean() < 2e-2 * scale

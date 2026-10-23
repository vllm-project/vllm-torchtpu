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
"""Qwen3.8 segment kernel with the QSA sparse attention in-kernel (CPU interpret).

`segment_spec(..., qsa_attn=...)` replaces the external attention input with
`QSASparseAttnBlock`. The reference is the existing segment reference fed with
the per-rank attention of the pure-JAX oracle (`reference_rows`) over the same
pool, pages and lengths.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.blocks.qsa_attn.block import MICRO, reference_rows
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import pack_segment, random_segment_inputs
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel.models.qwen38_reference import segment_reference
from vllm_torchtpu.megakernel.ops import BF16, F32

_TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
G = qwen38.TINY
BUDGET, PAGES, OFF = 64, 8, 1


def test_segment_with_in_kernel_qsa_attention_matches_reference():
    T, L, R = 4, qwen38.SEGMENT_LAYERS, 2
    B = 2 * T
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=_TILES)
    nh = G.NH // R
    rng = np.random.default_rng(5)
    pool = jnp.asarray(rng.standard_normal((R, PAGES, 128, 1, 4, G.HD)), F32).astype(
        jnp.float8_e4m3fn
    )
    pages = jnp.asarray(
        rng.integers(0, PAGES * 128 // MICRO, size=(ex.M, BUDGET + 1)), jnp.int32
    )
    kv_len = jnp.asarray(([0, 1, 9, 100, 255, 257, 130, 64] + [0] * ex.M)[: ex.M], jnp.int32)
    q = jnp.asarray(rng.standard_normal((R, ex.M, nh, G.HD)), BF16)
    gate = jnp.asarray(rng.standard_normal((R, ex.M, nh, G.HD)), BF16)
    sm = G.HD**-0.5
    attn = jnp.stack(
        [
            reference_rows(q[r, :B], gate[r, :B], pool[r], pages[:B], kv_len[:B],
                           off=OFF, scale_k=1.0, scale_v=1.0, sm_scale=sm).reshape(B, -1)
            for r in range(R)
        ]
    ).astype(BF16)
    assert float(jnp.abs(attn.astype(F32)).mean()) > 1e-2

    inp = random_segment_inputs(jax.random.PRNGKey(9), G, ex, L)
    inp.attn = attn
    ref, _ = segment_reference(G, ex, inp.x, inp.hc, inp.w_l, inp.st_l, inp.attn, inp.o_proj, L=L)

    spec = qwen38.segment_spec(G, R, L=L, qsa_attn=dict(budget=BUDGET, kv_head_offset=OFF))
    w_mk, states, _ = pack_segment(qwen38.segment_spec(G, R, L=L), G, ex, inp)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, inp.pos, inp.x, w_mk, states)
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    extras = (
        jax.device_put(kernel.fold(pool), tp),
        jax.device_put(kernel.fold(q), tp),
        jax.device_put(kernel.fold(gate), tp),
        jax.device_put(pages, rep),
        jax.device_put(kv_len, rep),
    )
    assert [x.name for x in kernel.extra_inputs(spec, ex)] == [
        "qsa_pool", "qsa_q", "qsa_gate", "qsa_pages", "qsa_kvlen"
    ]
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *extras)
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    refn = np.asarray(ref, np.float32)
    scale = float(np.abs(refn).mean())
    for r in range(R):
        d = np.abs(res[r] - refn)
        assert d.mean() < 1e-2 * scale, (r, d.mean(), scale)
        assert d.max() < 1e-1 * scale, (r, d.max(), scale)

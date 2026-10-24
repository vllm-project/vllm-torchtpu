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
"""Sweep geometry fields of the Qwen3.8 tail segment in CPU interpret mode to find
which one makes kernel rows disagree with the reference (needs a large-RAM host)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
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


def run(name, R=8, B=8, T=4, L=1, key=8, same_experts=False, router_gain=30.0, **over):
    g = dataclasses.replace(qwen38.FULL, **over)
    tiles = Tiles(ebank=2 << 20, nbe=12, nbd=10, vmem_mib=8192)
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=tiles)
    spec = qwen38.segment_spec(g, R, L=L)
    inp = random_segment_inputs(
        jax.random.PRNGKey(key), g, ex, L, gain=0.25, router_gain=router_gain
    )
    if same_experts:
        # Every routed expert identical: the output no longer depends on which
        # experts a row selects, only on the (renormalized) weights summing to 1.
        w = dict(inp.w_l)
        for k in ("e13", "e2", "e13_s", "e2_s"):
            a = w[k]
            w[k] = jnp.broadcast_to(a[:1, :, :1], a.shape) if a.ndim >= 3 else a
        inp.w_l = w
    ref, _ = segment_reference(
        g, ex, inp.x, inp.hc, inp.w_l, inp.st_l, inp.attn, inp.o_proj, L=L
    )
    w, st, attn = pack_segment(spec, g, ex, inp)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos, hid, w, st = shard_step_inputs(mesh, ex, inp.pos, inp.x, w, st)
    a = jax.device_put(kernel.fold(attn), NamedSharding(mesh, P("tp")))
    step, _ = kernel.make_step(mesh, spec, ex)
    out = step(pos, hid, w, st, a)[1][RESIDUAL_OUT]
    res = np.asarray(out.reshape(R, ex.M, -1)[0, :B], np.float32)
    refn = np.asarray(ref, np.float32)
    d = np.abs(res - refn).max(1) / np.abs(refn).mean()
    print(f"{name}: rows>10%={int((d > 0.1).sum())}/{B}", np.round(d, 3), flush=True)


cases = [
    ("full_same_experts", dict(same_experts=True)),
    ("full_router1", dict(router_gain=1.0)),
    ("full_router1_same_experts", dict(router_gain=1.0, same_experts=True)),
    ("full_B32", dict(B=32)),
    ("full_B32_same_experts", dict(B=32, same_experts=True)),
    ("full", {}),
]
for name, over in cases:
    try:
        run(name, **over)
    except Exception as e:
        print(name, "ERR", type(e).__name__, str(e)[:200], flush=True)

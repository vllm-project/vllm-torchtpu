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
"""The kernel against the model reference on TPU v7x-8 (skipped elsewhere).

A tiny two-layer geometry (one KV head per rank, as the checkpoint at TP=8)
runs one paged decode step with RoPE from a per-row table at an M-RoPE
position 7 below the KV index, and the ungated attention, bf16 KV pages and
dense FFN through the whole model reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest  # pyrefly: ignore[missing-import]
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.core import CacheExec, ExecSpec, cache, pool
from vllm_torchtpu.megakernel.core.modules.gqa.decls import ROPE_TABLE_INPUT
from vllm_torchtpu.megakernel.core.ops import BF16, F32
from vllm_torchtpu.megakernel.core.weights.synthetic import (
    pack_inputs,
    random_logical_inputs,
)

from .. import model
from .tiny import tiny_geom, tiny_tiles

R = 8

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu" or jax.device_count() < R,
    reason="needs a TPU v7x-8",
)


def _rope_table(g: object, pos: jax.Array, mr: int) -> jax.Array:
    """`(2, MR, ROT // 2)` cos / sin of each row's position, bf16-rounded like
    vLLM's rotary cache."""
    half = g.ROT // 2
    inv = g.theta ** (-jnp.arange(half, dtype=F32) / half)
    ang = pos.astype(F32)[:, None] * inv
    cs = jnp.stack([jnp.cos(ang), jnp.sin(ang)]).astype(BF16).astype(F32)
    return jnp.pad(cs, ((0, 0), (0, mr - pos.shape[0]), (0, 0)))


def _check_close(got: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> None:
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol)


def test_kernel_matches_model_reference() -> None:
    mdef = model.register()
    g = tiny_geom(R=R)
    ex = ExecSpec(
        R=R,
        B=2,
        T=1,
        ctx=512,
        tiles=tiny_tiles(kv_blk=256),
        cache=CacheExec(paged_kv=True, num_pages=16, max_pages_per_seq=4),
    )
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos, hid, w_l, st_l = random_logical_inputs(
        jax.random.PRNGKey(0), g, ex, model_def=mdef
    )
    rope_cs = _rope_table(g, pos - 7, ex.MR)
    ref_out, ref_st = mdef.reference_step(
        g, ex, pos, hid, w_l, st_l, extra={ROPE_TABLE_INPUT: rope_cs}
    )

    pos_k, hid_k, w_k, st_k = pack_inputs(
        pos, hid, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True, model_def=mdef
    )
    step = cache.make_pooled_step(mesh, mdef.to_model_spec(g, R=R), ex)
    rope_k = jax.device_put(jnp.tile(rope_cs, (R, 1, 1)), NamedSharding(mesh, P("tp")))
    out, st_out = step(pos_k, hid_k, w_k, st_k, rope_k)

    out = np.asarray(out, np.float32).reshape(R, ex.M, g.H)[:, : ex.B]
    ref = np.asarray(ref_out, np.float32)
    for r in range(R):
        _check_close(out[r], ref, atol=4e-2, rtol=2e-2)
    cos = (out[0] * ref).sum() / np.linalg.norm(out[0]) / np.linalg.norm(ref)
    assert cos >= 0.9999, cos

    lay = mdef.pool_layout(g, R=R)
    pt, _, _, _ = pool.allocate_test_layout(lay, ex)
    kv = pool.pool_to_states(lay, ex, st_out[pool.POOL_STATE_NAME], pt)
    for k in ("kc", "vc"):
        got = np.asarray(kv[k], np.float32)
        _check_close(got, np.asarray(ref_st[k], np.float32), atol=3.5e-2, rtol=1.5e-2)

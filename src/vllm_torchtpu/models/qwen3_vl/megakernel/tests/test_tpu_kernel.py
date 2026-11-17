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
dense FFN through the whole model reference, at the page sizes vLLM picks:
128 tokens, and 512 when its SMEM page-table bound raises the block size. The
in-kernel KV write (`fused_pool_io`) matches the scatter program bitwise.
"""

from __future__ import annotations

import dataclasses

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


@pytest.mark.parametrize(
    ("page_size", "kv_blk"),
    [(128, 256), (512, 1024), (512, 256)],
    ids=["p128_blk2pages", "p512_blk2pages", "p512_blk_rounds_up_to_1page"],
)
def test_kernel_matches_model_reference(page_size: int, kv_blk: int) -> None:
    mdef = model.register()
    g = tiny_geom(R=R)
    ex = ExecSpec(
        R=R,
        B=2,
        T=1,
        ctx=4 * page_size,
        tiles=tiny_tiles(kv_blk=kv_blk),
        cache=CacheExec(
            paged_kv=True, page_size=page_size, num_pages=16, max_pages_per_seq=4
        ),
    )
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    _, hid, w_l, st_l = random_logical_inputs(
        jax.random.PRNGKey(0), g, ex, model_def=mdef
    )
    # The new tokens land past the first page: slot 7 of page 1, and the last
    # page's slot `page_size - 5`.
    pos = jnp.array([page_size + 7, 4 * page_size - 5], jnp.int32)
    rope_cs = _rope_table(g, pos - 7, ex.MR)
    ref_out, ref_st = mdef.reference_step(
        g, ex, pos, hid, w_l, st_l, extra={ROPE_TABLE_INPUT: rope_cs}
    )

    lay = dataclasses.replace(mdef.pool_layout(g, R=R), page_size=page_size)
    spec = mdef.to_model_spec(g, R=R).with_layout(lay)
    rope_k = jax.device_put(jnp.tile(rope_cs, (R, 1, 1)), NamedSharding(mesh, P("tp")))

    def run(fused: bool) -> tuple[jax.Array, dict]:
        pos_k, hid_k, w_k, st_k = pack_inputs(
            pos,
            hid,
            w_l,
            st_l,
            g,
            ex,
            mesh=mesh,
            build_default_pool=True,
            model_def=mdef,
        )
        step = cache.make_pooled_step(
            mesh, spec, ex, donate_pools=fused, fused_pool_io=fused
        )
        return step(pos_k, hid_k, w_k, st_k, rope_k)

    out, st_out = run(fused=False)
    # The in-kernel KV write (one 128-lane block RMW per layer) leaves the same
    # pools and hidden states as the scatter program.
    out_f, st_f = run(fused=True)
    np.testing.assert_array_equal(np.asarray(out_f), np.asarray(out))
    for a, b in zip(st_f[pool.POOL_STATE_NAME], st_out[pool.POOL_STATE_NAME]):
        np.testing.assert_array_equal(
            np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16)
        )

    out = np.asarray(out, np.float32).reshape(R, ex.M, g.H)[:, : ex.B]
    ref = np.asarray(ref_out, np.float32)
    for r in range(R):
        _check_close(out[r], ref, atol=4e-2, rtol=2e-2)
    cos = (out[0] * ref).sum() / np.linalg.norm(out[0]) / np.linalg.norm(ref)
    assert cos >= 0.9999, cos

    pt, _, _, _ = pool.allocate_test_layout(lay, ex)
    kv = pool.pool_to_states(lay, ex, st_out[pool.POOL_STATE_NAME], pt)
    for k in ("kc", "vc"):
        got = np.asarray(kv[k], np.float32)
        _check_close(got, np.asarray(ref_st[k], np.float32), atol=3.5e-2, rtol=1.5e-2)

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
"""`kernels/mla/rows_oproj.py`: the packed shard layout and the blocked-scale
o_proj, against the per-rank partials they replace (Pallas interpret mode on
CPU)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla import rows_oproj as ro
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)

GEOM = ro.PackedOProj(
    shards=4, k_shard=128, n_out=256, heads=2, kv_lora_rank=128, v_head_dim=64
)


def _shards(seed=0):
    rng = np.random.default_rng(seed)
    g = GEOM
    w_o = (rng.standard_normal((g.shards, g.k_shard, g.n_out)) * 64).astype(np.float32)
    w_o = jnp.asarray(w_o).astype(ro.FP8)
    w_uv = jnp.asarray(
        rng.standard_normal(
            (g.shards, g.heads, g.kv_lora_rank, g.v_head_dim), np.float32
        )
        * 0.1
    ).astype(jnp.bfloat16)
    w_scale = jnp.asarray(rng.uniform(0.5, 1.5, (g.shards, g.n_out)) / 64, jnp.float32)
    return w_o, w_uv, w_scale


def _packed(w_o, w_uv, w_scale):
    return jnp.stack(
        [ro.pack_shard(GEOM, w_o[s], w_uv[s], w_scale[s]) for s in range(GEOM.shards)]
    )


def test_pack_unpack_roundtrip():
    w_o, w_uv, w_scale = _shards()
    packed = _packed(w_o, w_uv, w_scale)
    assert packed.shape == (GEOM.shards, GEOM.rows, GEOM.n_out)
    assert packed.dtype == ro.FP8
    got_o, got_uv, got_scale = ro.unpack(GEOM, packed, packed)
    np.testing.assert_array_equal(
        np.asarray(got_o[:, : GEOM.k_shard]).view(np.uint8),
        np.asarray(w_o).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(got_uv), np.asarray(w_uv.reshape(-1, *w_uv.shape[2:]))
    )
    np.testing.assert_array_equal(np.asarray(got_scale), np.asarray(w_scale))


def test_pack_rejects_wrong_shapes():
    w_o, w_uv, w_scale = _shards()
    with pytest.raises(ValueError):
        ro.pack_shard(GEOM, w_o[0, :64], w_uv[0], w_scale[0])
    with pytest.raises(ValueError):
        ro.pack_shard(GEOM, w_o[0], w_uv[0].astype(jnp.float32), w_scale[0])
    with pytest.raises(ValueError):
        ro.PackedOProj(4, 100, 256, 2, 128, 50)


def test_quantize_rows_is_quantize_tensor():
    v = jax.random.normal(jax.random.key(5), (32, 256), jnp.float32) * 3
    v = v.astype(jnp.bfloat16).at[3].set(0)  # an all-zero row keeps scale 1
    got_q, got_s = ro._quantize_rows(v)
    want_q, want_s = quantize_tensor(v, ro.FP8)
    np.testing.assert_array_equal(
        np.asarray(got_q).view(np.uint8), np.asarray(want_q).view(np.uint8)
    )
    np.testing.assert_array_equal(np.asarray(got_s), np.asarray(want_s))


def _latent(rows, seed):
    heads = GEOM.shards * GEOM.heads
    return jax.random.normal(
        jax.random.key(seed), (rows, heads, GEOM.kv_lora_rank), jnp.float32
    ).astype(jnp.bfloat16)


@pytest.mark.parametrize(("rows", "block_rows"), [(16, 16), (48, 16), (48, 32)])
def test_rows_oproj_matches_reference(rows, block_rows):
    w_view, uv_all, ws = ro.unpack(GEOM, _packed(*_shards(1)), jnp.zeros(()))
    lat = _latent(rows, 2)
    got = ro.rows_oproj(
        lat,
        uv_all,
        w_view,
        ws,
        k_shard=GEOM.k_shard,
        block_rows=block_rows,
        chunks=2,
        interpret=True,
    )
    want = ro.rows_oproj_ref(lat, uv_all, w_view, ws, k_shard=GEOM.k_shard)
    np.testing.assert_array_equal(
        np.asarray(got, np.float32), np.asarray(want.astype(jnp.bfloat16), np.float32)
    )


def test_rows_oproj_matches_per_rank_partials():
    """Each shard's partial is what that rank computes today (W_UV of its
    heads, per-row fp8 quant of its slice, per-channel fp8 o_proj); only the
    sum differs: bf16 partials summed vs one f32 sum."""
    w_o, w_uv, w_scale = _shards(3)
    w_view, uv_all, ws = ro.unpack(GEOM, _packed(w_o, w_uv, w_scale), jnp.zeros(()))
    rows = 24
    lat = _latent(rows, 4)
    got = ro.rows_oproj(
        lat, uv_all, w_view, ws, k_shard=GEOM.k_shard, chunks=2, interpret=True
    )
    partials = []
    for s in range(GEOM.shards):
        h = slice(s * GEOM.heads, (s + 1) * GEOM.heads)
        # The per-rank W_UV bmm: f32 accumulation, bf16 out (as on the MXU).
        v = jnp.einsum(
            "rhl,hlv->rhv",
            lat[:, h].astype(jnp.float32),
            w_uv[s].astype(jnp.float32),
        ).astype(jnp.bfloat16)
        partials.append(
            xla_quantized_matmul(v.reshape(rows, -1), w_o[s], w_scale[s]).astype(
                jnp.float32
            )
        )
    exact = np.asarray(sum(partials), np.float32)
    got = np.asarray(got, np.float32)
    assert np.isfinite(got).all()
    # The bf16 partials and the bf16 result: a few bf16 ulps of the output.
    assert np.abs(got - exact).max() / np.abs(exact).max() < 2e-2


@pytest.mark.parametrize("ring", [None, (0, 1, 3, 2)])
def test_gather_shards_stacks_every_rank_in_mesh_order(ring):
    """Both gathers give every rank the stack of all ranks' packed shards in
    mesh order (CPU virtual devices; the ring is any cycle of the mesh)."""
    n = GEOM.shards
    if jax.device_count() < n:
        pytest.skip(f"requires {n} devices; CPU virtual devices are sufficient")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:n]), ("x",))
    w_o, w_uv, w_scale = _shards(6)
    after = jnp.zeros((n, 8), jnp.float32)
    spec = jax.sharding.PartitionSpec("x")

    def local(w_o, w_uv, w_scale, after):
        return ro.gather_shards(GEOM, w_o[0], w_uv[0], w_scale[0], after, "x", ring)[
            None
        ]

    got = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(spec,) * 4,
            out_specs=spec,
            check_vma=False,
        )
    )(w_o, w_uv, w_scale, after)
    want = np.asarray(_packed(w_o, w_uv, w_scale)).view(np.uint8)
    got = np.asarray(got).view(np.uint8)
    for r in range(n):
        np.testing.assert_array_equal(got[r], want, err_msg=f"rank {r}")

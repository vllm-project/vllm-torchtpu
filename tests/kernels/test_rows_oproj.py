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
"""`kernels/mla/rows_oproj.py`: the shard gathers and the blocked-scale
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

SHARDS, HEADS, KVL, VH, N_OUT = 4, 2, 128, 64, 256
K_SHARD = HEADS * VH


def _shards(seed=0):
    """Every rank's o_proj `[K_SHARD, N_OUT]` fp8, W_UV and scale, stacked."""
    rng = np.random.default_rng(seed)
    w_o = (rng.standard_normal((SHARDS, K_SHARD, N_OUT)) * 64).astype(np.float32)
    w_o = jnp.asarray(w_o).astype(ro.FP8)
    w_uv = jnp.asarray(
        rng.standard_normal((SHARDS, HEADS, KVL, VH), np.float32) * 0.1
    ).astype(jnp.bfloat16)
    w_scale = jnp.asarray(rng.uniform(0.5, 1.5, (SHARDS, N_OUT)) / 64, jnp.float32)
    return w_o, w_uv, w_scale


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
    return jax.random.normal(
        jax.random.key(seed), (rows, SHARDS * HEADS, KVL), jnp.float32
    ).astype(jnp.bfloat16)


@pytest.mark.parametrize(("rows", "block_rows"), [(16, 16), (48, 16), (48, 32)])
def test_rows_oproj_matches_reference(rows, block_rows):
    w_o, w_uv, w_scale = _shards(1)
    uv_all = w_uv.reshape(-1, KVL, VH)
    lat = _latent(rows, 2)
    got = ro.rows_oproj(
        lat, uv_all, w_o, w_scale, block_rows=block_rows, chunks=2, interpret=True
    )
    want = ro.rows_oproj_ref(lat, uv_all, w_o, w_scale)
    np.testing.assert_array_equal(
        np.asarray(got, np.float32), np.asarray(want.astype(jnp.bfloat16), np.float32)
    )


def test_ring_modes_need_a_ring():
    w_o, w_uv, w_scale = _shards()
    with pytest.raises(ValueError, match="ring order"):
        ro.gather_shards(w_o[0], w_uv[0], w_scale[0], w_o[0], "x", "hybrid")


def test_rows_oproj_rejects_mismatched_shards():
    w_o, w_uv, w_scale = _shards()
    lat = _latent(16, 0)
    with pytest.raises(ValueError):  # o_proj rows != heads x v_head_dim
        ro.rows_oproj(lat, w_uv.reshape(-1, KVL, VH), w_o[:, :64], w_scale)
    with pytest.raises(ValueError):  # o_proj not fp8
        ro.rows_oproj(lat, w_uv.reshape(-1, KVL, VH), w_o.astype(jnp.bfloat16), w_scale)


def test_rows_oproj_matches_per_rank_partials():
    """Each shard's partial is what that rank computes today (W_UV of its
    heads, per-row fp8 quant of its slice, per-channel fp8 o_proj); only the
    sum differs: bf16 partials summed vs one f32 sum."""
    w_o, w_uv, w_scale = _shards(3)
    rows = 24
    lat = _latent(rows, 4)
    got = ro.rows_oproj(
        lat, w_uv.reshape(-1, KVL, VH), w_o, w_scale, chunks=2, interpret=True
    )
    partials = []
    for s in range(SHARDS):
        h = slice(s * HEADS, (s + 1) * HEADS)
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


@pytest.mark.parametrize("mode", sorted(ro.GATHERS))
def test_gather_shards_stacks_every_rank_in_mesh_order(mode):
    """Every gather mode gives every rank every rank's shards, bit for bit,
    in mesh order (CPU virtual devices; the ring is any cycle of the mesh)."""
    ring = (0, 1, 3, 2)
    if jax.device_count() < SHARDS:
        pytest.skip(f"requires {SHARDS} devices; CPU virtual devices are sufficient")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:SHARDS]), ("x",))
    w_o, w_uv, w_scale = _shards(6)
    after = jnp.zeros((SHARDS, 8), jnp.float32)
    spec = jax.sharding.PartitionSpec("x")

    def local(w_o, w_uv, w_scale, after):
        got = ro.gather_shards(w_o[0], w_uv[0], w_scale[0], after, "x", mode, ring)
        return tuple(t[None] for t in got)

    got_o, got_uv, got_scale = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(spec,) * 4,
            out_specs=(spec,) * 3,
            check_vma=False,
        )
    )(w_o, w_uv, w_scale, after)
    # Indexed on the host: indexing a sharded fp8 array may turn -0 into +0.
    got_o, got_uv, got_scale = (np.asarray(t) for t in (got_o, got_uv, got_scale))
    for r in range(SHARDS):
        np.testing.assert_array_equal(
            got_o[r].view(np.uint8), np.asarray(w_o).view(np.uint8)
        )
        np.testing.assert_array_equal(got_uv[r], np.asarray(w_uv.reshape(-1, KVL, VH)))
        np.testing.assert_array_equal(got_scale[r], np.asarray(w_scale))

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
"""The chip-pair wo_a kernel against the XLA exchange + op it replaces, bitwise."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4.o_projection import (
    fused_reverse_rope_wo_a_projection,
    fused_reverse_rope_wo_a_projection_o_fetch,
)

from .rope_test import make_cos_sin_cache

pytestmark = pytest.mark.tpu_test

# DeepSeek-V4-Pro under intra-chip TP: 128 heads of 512, 8 heads per group,
# o_lora_rank 1024; each core holds 8 of the 16 wo_a groups.
NUM_HEADS = 128
HEAD_DIM = 512
LORA_RANK = 1024
ROTARY_DIM = 64
MAX_POSITION = 4096


def _pair_mesh():
    return Mesh(np.asarray(jax.devices()[:2]), ("x",))


def _per_chiplet(kernel, *stacked):
    """`kernel` on each chiplet of the pair; operand i of chiplet r is
    stacked[i][r]."""
    mesh = _pair_mesh()
    spec = P("x")
    args = [
        jax.device_put(jnp.concatenate(list(a)), NamedSharding(mesh, spec))
        for a in stacked
    ]
    out = jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(spec,) * len(args),
        out_specs=spec,
        check_vma=False,
    )(*args)
    return np.asarray(out).reshape(2, -1, *out.shape[1:])


def _neighbor_and_chiplet_index():
    """Per chiplet: the neighboring chiplet's mesh index, and its own index."""
    neighbor_mesh_index = (np.asarray([1], np.int32), np.asarray([0], np.int32))
    chiplet_index = (np.asarray([0], np.int32), np.asarray([1], np.int32))
    return neighbor_mesh_index, chiplet_index


class WoAOFetchTest(parameterized.TestCase):
    @parameterized.parameters(8, 16, 24, 32, 64, 96, 128, 144, 256)
    def test_wo_a_o_fetch_matches_exchange_then_fused_op(self, num_tokens):
        rng = np.random.default_rng(num_tokens)
        half = NUM_HEADS // 2
        groups = half // 8
        o = [
            jnp.asarray(
                rng.standard_normal((num_tokens, NUM_HEADS, HEAD_DIM)), jnp.bfloat16
            )
            for _ in range(2)
        ]
        wo_a = [
            jnp.asarray(
                rng.standard_normal((8 * HEAD_DIM, groups * LORA_RANK)),
                jnp.float8_e4m3fn,
            )
            for _ in range(2)
        ]
        scale = [
            jnp.asarray(rng.uniform(0.5, 1.5, groups * LORA_RANK), jnp.float32)
            for _ in range(2)
        ]
        positions = jnp.asarray(
            rng.integers(0, MAX_POSITION, 2 * num_tokens), jnp.int32
        )
        cache = jnp.asarray(make_cos_sin_cache(MAX_POSITION, ROTARY_DIM))
        got = _per_chiplet(
            functools.partial(
                fused_reverse_rope_wo_a_projection_o_fetch, head_dim=HEAD_DIM
            ),
            o,
            (positions, positions),
            (cache, cache),
            wo_a,
            scale,
            *_neighbor_and_chiplet_index(),
        )
        for r in range(2):
            heads_r = slice(r * half, (r + 1) * half)
            o_half = jnp.concatenate([o[0][:, heads_r], o[1][:, heads_r]])
            want = fused_reverse_rope_wo_a_projection(
                o_half, positions, cache, wo_a[r], scale[r], head_dim=HEAD_DIM
            )
            np.testing.assert_array_equal(got[r], np.asarray(want), err_msg=f"{r=}")

    def test_wo_a_o_fetch_rejects_unaligned_token_count(self):
        groups = NUM_HEADS // 2 // 8
        with self.assertRaisesRegex(AssertionError, "multiple of 8"):
            fused_reverse_rope_wo_a_projection_o_fetch(
                jnp.zeros((20, NUM_HEADS, HEAD_DIM), jnp.bfloat16),
                jnp.zeros((40,), jnp.int32),
                jnp.zeros((MAX_POSITION, ROTARY_DIM), jnp.float32),
                jnp.zeros((8 * HEAD_DIM, groups * LORA_RANK), jnp.float8_e4m3fn),
                jnp.ones((groups * LORA_RANK,), jnp.float32),
                jnp.asarray([1], jnp.int32),
                jnp.asarray([0], jnp.int32),
                head_dim=HEAD_DIM,
            )


if __name__ == "__main__":
    absltest.main()

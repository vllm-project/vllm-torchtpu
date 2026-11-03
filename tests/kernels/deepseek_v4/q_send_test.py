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
"""The chip-pair q send (q_send) and the in-place q-norm + RoPE
of both blocks (rope.qnorm_rope_pair), against the exchange they replace."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4 import q_send
from vllm_torchtpu.kernels.deepseek_v4.collective_ids import Q_SEND_COLLECTIVE_ID
from vllm_torchtpu.kernels.deepseek_v4.rope import qnorm_rope, qnorm_rope_pair

from .rope_test import make_cos_sin_cache

pytestmark = pytest.mark.tpu_test

# DeepSeek-V4-Pro under intra-chip TP: 128 heads of 512, half on each chiplet.
HALF = 64
HEAD_DIM = 512
ROTARY_DIM = 64
MAX_POSITION = 4096


class ChipPairQSendTest(parameterized.TestCase):
    @parameterized.product(num_tokens=[8, 64, 128, 144], chiplet_index=[0, 1])
    def test_qnorm_rope_pair_matches_qnorm_rope_of_concatenated_q(
        self, num_tokens, chiplet_index
    ):
        rng = np.random.default_rng(num_tokens + chiplet_index)
        # The current chiplet's tokens over all heads, in head order.
        q = jnp.asarray(
            rng.standard_normal((num_tokens, 2 * HALF, HEAD_DIM)), jnp.bfloat16
        )
        own = slice(chiplet_index * HALF, (chiplet_index + 1) * HALF)
        other = slice((1 - chiplet_index) * HALF, (2 - chiplet_index) * HALF)
        # [2T, half, D]: both chiplets' tokens with the current chiplet's heads;
        # its own tokens are block `chiplet_index`, the other block is unused.
        unused = jnp.asarray(
            rng.standard_normal((num_tokens, HALF, HEAD_DIM)), jnp.bfloat16
        )
        blocks = [unused, unused]
        blocks[chiplet_index] = q[:, own]
        current = jnp.concatenate(blocks)
        neighbor = q[:, other]
        positions = jnp.asarray(rng.integers(0, MAX_POSITION, num_tokens), jnp.int32)
        cache = jnp.asarray(make_cos_sin_cache(MAX_POSITION, ROTARY_DIM))
        got = qnorm_rope_pair(
            current, neighbor, positions, cache, chiplet_index=chiplet_index
        )
        want = qnorm_rope(q, positions, cache)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    @parameterized.parameters(8, 64, 128)
    def test_q_send_delivers_the_neighboring_chiplets_q_heads(self, num_tokens):
        mesh = Mesh(np.asarray(jax.devices()[:2]), ("x",))
        spec = P("x")
        rng = np.random.default_rng(num_tokens)
        # Per chiplet: [2, T, half, D], both chiplets' tokens, its own heads.
        qh = [
            jnp.asarray(
                rng.standard_normal((2, num_tokens, HALF, HEAD_DIM)), jnp.bfloat16
            )
            for _ in range(2)
        ]
        # Per chiplet c: the neighbor's mesh index (1 - c) and its own index c.
        neighbor_mesh_index = jnp.asarray([1, 0], jnp.int32)
        chiplet_index = jnp.asarray([0, 1], jnp.int32)

        def local(part, neighbor_mesh_index, chiplet_index):
            handle = q_send.start_q_send(
                part[0],
                neighbor_mesh_index,
                chiplet_index,
                collective_id=Q_SEND_COLLECTIVE_ID,
            )
            return handle[0][None], q_send.wait_q_send(handle)[None]

        def put(per_chiplet):
            return jax.device_put(jnp.stack(per_chiplet), NamedSharding(mesh, spec))

        source, received = jax.jit(
            jax.shard_map(
                local, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
            )
        )(put(qh), neighbor_mesh_index, chiplet_index)
        source, received = np.asarray(source), np.asarray(received)
        for c in range(2):
            # The source comes back unchanged, and chiplet c receives the block
            # of its own tokens from the neighboring chiplet's buffer.
            np.testing.assert_array_equal(source[c], np.asarray(qh[c]))
            np.testing.assert_array_equal(received[c], np.asarray(qh[1 - c][c]))


if __name__ == "__main__":
    absltest.main()

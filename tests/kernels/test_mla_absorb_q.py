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
"""`absorb_q` vs the head-major batched matmul it replaces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.absorb_q import absorb_q, absorb_q_head


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [4, 200, 256, 300, 4096])
def test_matches_head_major_bmm(tokens):
    """Decode sizes take XLA's einsum and prefill sizes the kernel (with a
    partial last block at 300); both equal the bmm the call site ran."""
    heads, nope, rank = 64, 192, 512
    k1, k2 = jax.random.split(jax.random.key(tokens))
    q_nope = jax.random.normal(k1, (tokens, heads, nope)).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (heads, nope, rank)) * 0.05).astype(jnp.bfloat16)
    bmm = jnp.matmul(jnp.swapaxes(q_nope, 0, 1), w, preferred_element_type=jnp.float32)
    want = jnp.swapaxes(bmm, 0, 1).astype(q_nope.dtype)
    np.testing.assert_array_equal(
        np.asarray(absorb_q(q_nope, w), np.float32), np.asarray(want, np.float32)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [256, 300, 4096])
def test_head_input_matches_slice(tokens):
    """`absorb_q_head` of the q projection's 2-D output equals `absorb_q` of
    its q_nope slice bit for bit, and returns each head's last 128 lanes."""
    heads, nope, rope, rank = 32, 192, 64, 512
    k1, k2 = jax.random.split(jax.random.key(tokens))
    q = jax.random.normal(k1, (tokens, heads, nope + rope)).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (heads, nope, rank)) * 0.05).astype(jnp.bfloat16)
    out, tail = absorb_q_head(q.reshape(tokens, -1), w)
    np.testing.assert_array_equal(
        np.asarray(out, np.float32),
        np.asarray(absorb_q(q[..., :nope], w), np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(tail, np.float32), np.asarray(q[..., 128:], np.float32)
    )

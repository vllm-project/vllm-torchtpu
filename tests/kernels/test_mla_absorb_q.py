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

from vllm_torchtpu.kernels.mla.absorb_q import absorb_q, absorb_q_full


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


def _full_q_case(tokens, heads, dtype):
    head_dim, nope, rank = 256, 192, 512
    k1, k2 = jax.random.split(jax.random.key(tokens + heads))
    q = jax.random.normal(k1, (tokens, heads * head_dim)).astype(dtype)
    w = (jax.random.normal(k2, (heads, nope, rank)) * 0.05).astype(dtype)
    q_nope = q.reshape(tokens, heads, head_dim)[..., :nope]
    return q, q_nope, w


@pytest.mark.tpu_test
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("tokens", [4, 256, 300, 4096])
def test_full_q_matches_sliced(tokens, heads):
    """TPU_MLA_ABSORB_FULL_Q: the kernel on the q_b output [T, H * 256] gives
    the same bits as absorb_q on its [T, H, :192] slice (the control path),
    including a partial last block (300) and the einsum below MIN_TOKENS (4).
    32 heads is the prefill twin's rank, 64 the served rank's."""
    q, q_nope, w = _full_q_case(tokens, heads, jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(absorb_q_full(q, w), np.float32),
        np.asarray(absorb_q(q_nope, w), np.float32),
    )


@pytest.mark.parametrize("tokens", [256, 300])
def test_full_q_interpret(tokens):
    """The kernel's lane slicing in Pallas interpret mode (no TPU; f32, since
    the CPU dot has no bf16 x bf16 -> f32), against the head-major bmm."""
    q, q_nope, w = _full_q_case(tokens, 8, jnp.float32)
    bmm = jnp.matmul(jnp.swapaxes(q_nope, 0, 1), w, preferred_element_type=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(absorb_q_full(q, w, interpret=True)),
        np.asarray(jnp.swapaxes(bmm, 0, 1)),
    )


def test_full_q_rejects_unaligned_head():
    q = jnp.zeros((256, 4 * 200), jnp.float32)
    w = jnp.zeros((4, 192, 512), jnp.float32)
    with pytest.raises(ValueError):
        absorb_q_full(q, w, interpret=True)

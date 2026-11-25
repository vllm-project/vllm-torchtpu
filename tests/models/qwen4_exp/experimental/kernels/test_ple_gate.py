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
"""VLLM_TPU_PLE_FUSED_GATE kernel matches the Torch-order reference (interpret)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.kernels.ple_gate import (
    normalized_gate_dot,
)


@pytest.mark.parametrize("tokens,streams,hidden", [(13, 4, 1024), (8, 2, 640)])
def test_fused_gate_dot_matches_reference(tokens, streams, hidden):
    rng = np.random.default_rng(tokens)
    shape = (tokens, streams * hidden)
    key = jnp.asarray(rng.normal(size=shape), jnp.bfloat16)
    query = jnp.asarray(rng.normal(size=shape), jnp.bfloat16)
    kw = jnp.asarray(rng.normal(size=shape[1:]) * 0.1, jnp.bfloat16)
    qw = jnp.asarray(rng.normal(size=shape[1:]) * 0.1, jnp.bfloat16)

    def inv(x):
        g = x.astype(jnp.float32).reshape(tokens, streams, hidden)
        return jax.lax.rsqrt((g * g).mean(-1) + 1e-6)

    def norm(x, w):
        g = x.astype(jnp.float32).reshape(tokens, streams, hidden) * inv(x)[..., None]
        w = (1 + w.astype(jnp.float32)).reshape(streams, hidden)
        return (g * w).astype(jnp.bfloat16).astype(jnp.float32)

    out = normalized_gate_dot(
        key, query, inv(key), inv(query), kw, qw, streams=streams, interpret=True
    )
    ref = (norm(key, kw) * norm(query, qw)).astype(jnp.bfloat16)
    ref = ref.astype(jnp.float32).sum(-1).astype(jnp.bfloat16)
    assert out.shape == (tokens, streams) and out.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(ref, np.float32), rtol=1e-2, atol=1e-2
    )

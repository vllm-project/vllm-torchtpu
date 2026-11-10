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
"""The fused q/k norm + rope + gate kernel against vLLM's eager path in jnp, at
a served per-core geometry (4 q heads, 1 kv head of 256, 64 rotary dims) and a
full-rope one."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.qk_norm_rope_gate import (
    qk_norm_rope_gate,
    reference,
)

GEOMETRIES = [(4, 1, 256, 64), (16, 1, 256, 64), (2, 2, 128, 128)]


def _operands(seed, tokens, heads, kv_heads, dim, rot):
    rng = np.random.default_rng(seed)
    q_gate = jnp.asarray(rng.standard_normal((tokens, heads * 2 * dim)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((tokens, kv_heads * dim)), jnp.bfloat16)
    q_w = jnp.asarray(1.0 + 0.1 * rng.standard_normal(dim), jnp.float32)
    k_w = jnp.asarray(1.0 + 0.1 * rng.standard_normal(dim), jnp.float32)
    angle = rng.random((tokens, rot // 2)) * 2 * np.pi
    cos_sin = jnp.asarray(
        np.concatenate([np.cos(angle), np.sin(angle)], axis=1), jnp.bfloat16
    )
    return q_gate, k, q_w, k_w, cos_sin


def _assert_parity(tokens, geometry, interpret, tile=512):
    heads, kv_heads, dim, rot = geometry
    operands = _operands(tokens * 3 + heads, tokens, heads, kv_heads, dim, rot)
    kw = dict(heads=heads, kv_heads=kv_heads, dim=dim, rot=rot, eps=1e-6)
    got = jax.block_until_ready(
        qk_norm_rope_gate(*operands, **kw, tile=tile, interpret=interpret)
    )
    want = reference(*operands, **kw)
    for a, b in zip(got, want):
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_allclose(
            np.asarray(a, np.float32), np.asarray(b, np.float32), atol=2e-2, rtol=2e-2
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens, tile", [(13, 512), (24, 8)])
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_parity_in_the_interpreter(tokens, tile, geometry):
    with jax.default_device(jax.devices("cpu")[0]):
        _assert_parity(tokens, geometry, interpret=True, tile=tile)


@pytest.mark.parametrize("tokens", [1, 4, 8, 13, 32, 64, 1024, 4096, 8192])
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_parity(tokens, geometry):
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    _assert_parity(tokens, geometry, interpret=False)

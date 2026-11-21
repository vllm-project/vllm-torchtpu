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
"""Parity of the fused q/gate split + Q/K RMSNorm + RoPE + gate kernel with
vLLM's eager path written in plain JAX. Off TPU the kernel runs in Pallas
interpret mode."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qk_norm_rope_gate import qk_norm_rope_gate, reference


def _on_tpu():
    # Called inside tests: initializing a JAX backend during collection is
    # refused by tests/conftest.py.
    return jax.default_backend() == "tpu"


def _operands(seed, rows, num_heads, num_kv_heads, head_dim, rotary_dim):
    rng = np.random.default_rng(seed)
    q_gate = jnp.asarray(
        rng.standard_normal((rows, num_heads * 2 * head_dim)), jnp.bfloat16
    )
    k = jnp.asarray(rng.standard_normal((rows, num_kv_heads * head_dim)), jnp.bfloat16)
    q_weight = jnp.asarray(1.0 + 0.1 * rng.standard_normal(head_dim), jnp.float32)
    k_weight = jnp.asarray(1.0 + 0.1 * rng.standard_normal(head_dim), jnp.float32)
    angles = rng.random((4096, rotary_dim // 2)) * 2 * np.pi
    cos_sin_cache = jnp.asarray(
        np.concatenate([np.cos(angles), np.sin(angles)], axis=1), jnp.bfloat16
    )
    positions = jnp.asarray(rng.integers(0, 4096, size=(rows,)), jnp.int32)
    return q_gate, k, q_weight, k_weight, cos_sin_cache, positions


def _assert_parity(rows, geometry):
    num_heads, num_kv_heads, head_dim, rotary_dim = geometry
    operands = _operands(
        rows * 3 + num_heads, rows, num_heads, num_kv_heads, head_dim, rotary_dim
    )
    kwargs = dict(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=1e-6,
    )
    got = qk_norm_rope_gate(*operands, interpret=not _on_tpu(), **kwargs)
    want = reference(*operands, **kwargs)
    for name, actual, expected in zip(("q", "k", "gate"), got, want):
        assert actual.shape == expected.shape, name
        assert actual.dtype == expected.dtype, name
        np.testing.assert_allclose(
            np.asarray(actual, np.float32),
            np.asarray(expected, np.float32),
            atol=2e-2,
            rtol=2e-2,
            err_msg=name,
        )


# (4, 1, 256, 64) is Qwen3.5-397B at TP8 (32 q heads and 2 kv heads over 8
# cores, kv replicated); (32, 2, 256, 64) is the same model with every head on
# one core; (2, 2, 128, 128) rotates the whole head.
@pytest.mark.parametrize("rows", [1, 4, 8, 13, 64])
@pytest.mark.parametrize(
    "geometry", [(4, 1, 256, 64), (2, 2, 128, 128), (32, 2, 256, 64)]
)
def test_parity(rows, geometry):
    _assert_parity(rows, geometry)


def test_parity_prompt_sized_rows():
    if not _on_tpu():
        pytest.skip("prompt-sized rows are slow in interpret mode")
    _assert_parity(2048, (32, 2, 256, 64))

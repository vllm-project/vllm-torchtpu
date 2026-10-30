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
"""``indexer_q_prep`` vs ``VllmTPUIndexer`` RoPE + ``quantize_tensor``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels import indexer_qprep
from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    _build_rope_matmul_tensors,
)


@pytest.mark.tpu_test
@pytest.mark.parametrize("n_head", [32, 64])
@pytest.mark.parametrize("is_neox_style", [False, True])
@pytest.mark.parametrize("tokens", [1, 8, 64, 128, 256, 512, 1024])
def test_indexer_q_prep_matches_reference(
    n_head: int, is_neox_style: bool, tokens: int
) -> None:
    head_dim, rope_dim = 128, 64
    rng = np.random.default_rng(1000 + tokens + int(is_neox_style) * 17 + n_head)
    q = jnp.asarray(
        rng.standard_normal((tokens, n_head * head_dim)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    p_t, r_t, b_t = _build_rope_matmul_tensors(
        rope_dim, head_dim, is_neox_style, torch.bfloat16, torch.device("cpu")
    )
    p_mat = jnp.asarray(p_t.to(torch.float32).numpy(), dtype=jnp.bfloat16)
    r_mat = jnp.asarray(r_t.to(torch.float32).numpy(), dtype=jnp.bfloat16)
    b_nope = (
        None
        if b_t is None
        else jnp.asarray(b_t.to(torch.float32).numpy(), dtype=jnp.bfloat16)
    )
    cos = jnp.asarray(
        rng.standard_normal((tokens, rope_dim // 2)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    sin = jnp.asarray(
        rng.standard_normal((tokens, rope_dim // 2)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    cos_full = jnp.dot(cos, r_mat, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )
    if b_nope is not None:
        cos_full = cos_full + b_nope
    sin_full = jnp.dot(sin, r_mat, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )

    want_q, want_s = indexer_qprep.reference(
        q, cos_full, sin_full, p_mat, n_head=n_head
    )
    got_q, got_s = indexer_qprep.indexer_q_prep(
        q, cos_full, sin_full, p_mat, n_head=n_head
    )
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(got_q, jnp.uint8)),
        np.asarray(jax.lax.bitcast_convert_type(want_q, jnp.uint8)),
    )
    np.testing.assert_array_equal(np.asarray(got_s), np.asarray(want_s))

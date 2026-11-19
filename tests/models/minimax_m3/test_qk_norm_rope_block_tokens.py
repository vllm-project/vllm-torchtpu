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
"""TPU_MINIMAX_M3_QK_NORM_ROPE_BLOCK_TOKENS: the fused qk norm + rope kernel
gives the same bits at any token block (interpret)."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import qk_norm_rope as K


@pytest.mark.parametrize("tokens,block", [(1024, 512), (1024, 1024), (1040, 512)])
def test_blocks_bit_equal(tokens, block):
    rng = np.random.default_rng(tokens + block)
    qkv = jnp.asarray(rng.standard_normal((tokens, 12 * 128), np.float32), jnp.bfloat16)
    cs = jnp.asarray(rng.uniform(-1, 1, (tokens, 64)).astype(np.float32), jnp.bfloat16)
    w = jnp.asarray(rng.uniform(0.5, 1.5, (11, 128)).astype(np.float32))
    kw = dict(num_heads=8, num_kv_heads=1, head_dim=128, eps=1e-6, q_head_major=True, interpret=True)
    ref = K.qk_norm_rope(qkv, cs, w, block_tokens=256, **kw)
    got = K.qk_norm_rope(qkv, cs, w, block_tokens=block, **kw)
    for a, b in zip(got, ref):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

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
"""CPU checks for the fused PCP GDN config and its triangular solve."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.gdn_pcp_fused.compute_gdn import (
    invert_triangular_matrix,
)
from vllm_torchtpu.kernels.experimental.gdn_pcp_fused.config import GDNConfig


@pytest.mark.parametrize(
    ("num_v_heads", "expected"),
    [(8, 16), (16, 8), (24, 4), (32, 4), (40, 2), (48, 2), (64, 2), (96, 2)],
)
def test_gdn_triangular_block_size_values(num_v_heads, expected):
    # 3 head groups (e.g. 24 value heads per rank) gave 16 // 3 = 5 and 5 head
    # groups gave 3; neither divides the chunk.
    cfg = SimpleNamespace(num_v_heads=num_v_heads, is_kda=False)
    assert GDNConfig.triangular_block_size.fget(cfg) == expected


@pytest.mark.parametrize("block_size", [2, 4, 8])
def test_invert_triangular_matrix_block_sizes_agree(block_size):
    # Every power-of-two block size that divides the chunk gives the same
    # inverse as the default 16-wide blocks.
    chunk = 64
    with jax.default_device(jax.devices("cpu")[0]):
        t = jax.random.normal(jax.random.key(0), (2, chunk, chunk)) * 0.1
        t = jnp.tril(t, -1) + jnp.eye(chunk)
        ref = invert_triangular_matrix(t, block_size=16)
        out = invert_triangular_matrix(t, block_size=block_size)
    assert out.shape == ref.shape == (2, chunk, chunk)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-6)

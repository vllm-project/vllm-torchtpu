# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The prefill stage's GDN state copies between manager blocks."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.core import gdn_attention
from vllm_torchtpu.megakernel.qwen38.stage import copy_state_blocks


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_a_copy_moves_every_page_of_the_manager_block(monkeypatch, layout):
    monkeypatch.setattr(gdn_attention.envs, "VLLM_KV_CACHE_LAYOUT", layout)
    page, block_tokens, blocks = 128, 640, 4  # five kernel pages per manager block
    pages = blocks * block_tokens // page
    shape = (pages, 2, 4, 4, page) if layout == "HND" else (pages, page, 2, 4)
    pool = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    copied = np.asarray(
        copy_state_blocks(pool, jnp.array([1]), jnp.array([3]), block_tokens)
    )
    split = block_tokens // page
    original = np.asarray(pool)
    np.testing.assert_array_equal(
        copied[3 * split : 4 * split], original[1 * split : 2 * split]
    )
    np.testing.assert_array_equal(copied[: 3 * split], original[: 3 * split])

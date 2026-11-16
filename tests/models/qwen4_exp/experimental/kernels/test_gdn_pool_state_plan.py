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
"""The model's GDN V3 state plan over the unified pool, on main's pool API."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.gdn_attention import (
    _build_v3_pool_state_plan,
)

# Pool (blocks, kernel block tokens, payload, lanes): 1 KiB token rows. Only
# its shape and dtype are read, so no array (and no JAX backend) is needed.
_POOL = jax.ShapeDtypeStruct((2, 16, 2, 128), jnp.float32)
_GDN = dict(conv_dim=512, n_v=2, d_k=128, d_v=128, kernel_size=4)


def test_fp32_conv_state_follows_the_ssm_region():
    plan = _build_v3_pool_state_plan(
        _POOL,
        **_GDN,
        pool_block_tokens=144,
        qk_pair_layout=False,
        recurrent_state_dtype=jnp.float32,
        conv_state_dtype=jnp.float32,
    )
    # fp32 SSM: 2 * 128 * 128 * 4 B = 128 token rows = 8 kernel blocks.
    assert plan.stride == 9
    assert (plan.recurrent.kb0, plan.recurrent.nblocks) == (0, 8)
    assert plan.recurrent.view_dtype == jnp.float32
    # fp32 conv: 3 * 512 * 4 B = 6 KiB -> 8 token rows right after the SSM.
    assert (plan.conv.kb0, plan.conv.row0, plan.conv.nrows) == (8, 0, 8)
    assert plan.conv.view_dtype == jnp.float32
    assert plan.conv.rows_used == 3 * 512 // 128
    assert not plan.whole_block_dma


def test_state_that_does_not_fit_the_manager_block_is_rejected():
    with pytest.raises(ValueError, match="does not fit"):
        _build_v3_pool_state_plan(
            _POOL,
            **_GDN,
            pool_block_tokens=128,
            qk_pair_layout=False,
            conv_state_dtype=jnp.float32,
        )

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kernels.gdn.v3.gdn_pool_state_utils import (
    read_states,
    recurrent_granules,
    write_states,
)

from vllm_torchtpu.kernels.gdn.v3 import config


# GDNConfig reads the TPU generation (pltpu.get_tpu_info) for its lane width.
@pytest.mark.tpu_test
@pytest.mark.parametrize("num_v_heads", [1, 4, 8])
@pytest.mark.parametrize("recurrent_dtype", [jnp.float32, jnp.bfloat16])
def test_conv_offset_matches_kernel_config(num_v_heads, recurrent_dtype):
    """The helper must place conv where the kernel reads it."""
    cfg = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.bfloat16.dtype,
            act_out=jnp.bfloat16.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.dtype(recurrent_dtype),
            conv_state=jnp.float32.dtype,
        ),
        batch_size=1,
        dim_size=(2 + num_v_heads) * 128,
        kernel_size=4,
        tile_size=64,
        num_kq_heads=1,
        num_v_heads=num_v_heads,
        kq_head_dim=128,
        v_head_dim=128,
        is_shared=True,
    )

    assert cfg.recurrent_hbm_offset == 0
    assert cfg.conv_hbm_offset == recurrent_granules(
        (num_v_heads, 128, 128), recurrent_dtype
    )


@pytest.mark.parametrize("pool_dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
@pytest.mark.parametrize("recurrent_dtype", [jnp.float32, jnp.bfloat16])
def test_write_read_roundtrip_leaves_other_states(pool_dtype, recurrent_dtype):
    split, block_tokens, lanes = 2, 256, 128
    packing = 4 // jnp.dtype(pool_dtype).itemsize
    num_states, num_v_heads, conv_shape = 4, 2, (3, 512)
    pool_shape = (num_states * split, block_tokens, 1, packing, lanes)
    rng = np.random.default_rng(0)
    pool_bytes = rng.integers(
        0,
        256,
        size=(*pool_shape[:-1], lanes * jnp.dtype(pool_dtype).itemsize),
        dtype=np.uint8,
    )
    pool = np.asarray(pool_bytes).view(jnp.dtype(pool_dtype)).reshape(pool_shape)
    idx = jnp.asarray([2, 1], dtype=jnp.int32)
    keys = jax.random.split(jax.random.key(0), 2)
    ssm = jax.random.normal(keys[0], (2, num_v_heads, 128, 128)).astype(recurrent_dtype)
    conv = jax.random.normal(keys[1], (2, *conv_shape), dtype=jnp.float32)

    new_pool = write_states(
        pool, ssm, conv, idx, split=split, recurrent_dtype=recurrent_dtype
    )
    got_ssm, got_conv = read_states(
        new_pool,
        idx,
        split=split,
        recurrent_shape=(num_v_heads, 128, 128),
        conv_shape=conv_shape,
        recurrent_dtype=recurrent_dtype,
    )

    np.testing.assert_array_equal(np.asarray(got_ssm), np.asarray(ssm))
    np.testing.assert_array_equal(np.asarray(got_conv), np.asarray(conv))
    # States 0 and 3 are not named by idx and must keep their bytes. Compare
    # on the host: a float path on TPU may rewrite NaN payloads.
    new_bytes = np.asarray(jax.device_get(new_pool)).view(np.uint8)
    old_bytes = pool.view(np.uint8)
    for state in (0, 3):
        rows = slice(state * split, (state + 1) * split)
        np.testing.assert_array_equal(new_bytes[rows], old_bytes[rows])

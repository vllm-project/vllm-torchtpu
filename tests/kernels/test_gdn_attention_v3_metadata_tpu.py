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

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.tpu_test


@pytest.mark.parametrize("capacity,active", [(1, 1), (3, 1), (3, 3), (5, 5)])
def test_pooled_gdn_odd_capacity_matches_aligned_capacity(capacity, active):
    from vllm_torchtpu.layers.core.gdn_attention import (
        run_jax_gdn_attention_pooled_local,
    )

    rng = np.random.default_rng(20261111)
    channels, n_v = 10240, 48

    def rand(shape, dtype=jnp.bfloat16):
        return jnp.asarray(rng.normal(0, 0.1, shape), dtype)

    qkv, beta, alpha = rand((32, channels)), rand((32, n_v)), rand((32, n_v))
    weight, bias = rand((channels, 1, 4)), rand((channels,))
    a_log, dt_bias = rand((n_v,), jnp.float32), rand((n_v,), jnp.float32)
    run = jax.jit(
        functools.partial(
            run_jax_gdn_attention_pooled_local,
            n_kq=16,
            n_v=n_v,
            d_k=128,
            d_v=128,
            kernel_size=4,
            pool_block_tokens=1664,
            recurrent_state_dtype=jnp.bfloat16,
            num_spec_tokens=1,
        ),
        donate_argnums=(3,),
    )
    page_count = 3 * capacity + 2
    poison = np.full((page_count, 1664, 1, 4, 256), 0x42, np.uint8)

    def step(rows):
        pool = jax.device_put(poison.copy().view(np.dtype(jnp.float8_e4m3fn)))
        qlens = np.zeros(rows, np.int32)
        qlens[:active] = 1
        ids = np.zeros(rows, np.int32)
        ids[:capacity] = np.arange(capacity, dtype=np.int32) * 3 + 1
        checkpoints = np.stack([ids, ids + 1], axis=1)
        return run(
            qkv,
            beta,
            alpha,
            pool,
            weight,
            bias,
            a_log,
            dt_bias,
            jnp.asarray(np.concatenate([np.zeros(1, np.int32), np.cumsum(qlens)])),
            jnp.asarray(ids),
            jnp.asarray([active, active, active], jnp.int32),
            jnp.asarray(qlens),
            read_offsets=jnp.zeros(rows, jnp.int32),
            ckpt_indices=jnp.asarray(checkpoints),
        )

    actual_state, actual = step(capacity)
    reference_state, expected = step(capacity + 1)
    # The control has an explicit final inactive request row. Both use K1 and
    # the production default tile, with independent poisoned physical pools.
    np.testing.assert_array_equal(
        np.asarray(actual)[:active], np.asarray(expected)[:active]
    )
    a = np.asarray(actual_state).view(np.uint8)
    b = np.asarray(reference_state).view(np.uint8)
    np.testing.assert_array_equal(a, b)
    written = np.arange(active) * 3 + 1
    untouched = np.setdiff1d(np.arange(page_count), written)
    np.testing.assert_array_equal(a[untouched], poison[untouched])
    # The first valid request must not be overwritten by inactive tile lanes.
    assert np.any(a[written[0]] != poison[written[0]])

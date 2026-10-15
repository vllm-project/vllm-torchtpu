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
"""Full-cache regression for HCA decode launch fusion and its fallback."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    compressor_v1,
    hca_decode_compressor,
)

pytestmark = pytest.mark.tpu_test


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(811)
    x = jnp.asarray(rng.normal(0, 0.2, (4, 1, 4096)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(0, 0.03, (4096, 1024)), jnp.bfloat16)
    ape = jnp.asarray(rng.normal(0, 0.1, (128, 512)), jnp.float32)
    norm = jnp.asarray(rng.uniform(0.8, 1.2, 512), jnp.bfloat16)
    phase = np.arange(1024)[:, None] * np.linspace(0.001, 0.01, 32)[None, :]
    cs = jnp.asarray(
        np.concatenate([np.cos(phase), np.sin(phase)], axis=1), jnp.float32
    )
    state = rng.normal(0, 0.1, (8, 256, 128)).astype(np.float32).view(np.int32)
    cache = jnp.zeros((4, 16, 4, 128), jnp.uint8)
    # Nontrivial physical mapping; a complete 128-token history spans four pages.
    st = jnp.asarray(np.tile(np.arange(32, dtype=np.int32)[::-1] % 8, 256))
    kt = jnp.asarray(np.tile(np.arange(8, dtype=np.int32)[::-1] % 4, 256))
    return x, w, ape, norm, cs, state, cache, st, kt


@pytest.mark.parametrize("layout", ["words", "slabs"])
@pytest.mark.parametrize(
    "mode,start",
    [
        ("decode", 0),
        ("decode", 120),
        ("prefill", 120),
        ("mixed", 120),
        ("empty", 120),
        ("leading_empty", 120),
    ],
)
def test_continuous_full_cache(inputs, layout, mode, start):
    x, w, ape, norm, cs, state, cache, st, kt = inputs
    if layout == "slabs":
        state = np.stack(
            [
                ((state.astype(np.uint32) >> (8 * i)) & 255).astype(np.uint8)
                for i in range(4)
            ],
            axis=-2,
        )
    state = jnp.asarray(state)
    cu = jnp.minimum(jnp.arange(257), 1).astype(jnp.int32)
    distribution = {
        "decode": [1, 1, 1],
        "prefill": [0, 0, 1],
        "mixed": [0, 1, 1],
        "empty": [0, 0, 0],
        "leading_empty": [2, 2, 2],
    }[mode]
    if mode == "empty":
        cu = jnp.zeros_like(cu)
    if mode == "leading_empty":
        cu = jnp.clip(jnp.arange(257) - 1, 0, 1).astype(jnp.int32)
    distribution = jnp.asarray(distribution, jnp.int32)

    def call(fn, hidden, pos, carry):
        main, history = carry
        main, _, history = fn(
            hidden,
            w,
            ape,
            norm,
            cs,
            pos,
            st,
            cu,
            kt,
            main,
            None,
            distribution,
            state_block_size=32,
            head_dim=512,
            compress_ratio=128,
            overlap=False,
            rms_eps=1e-6,
            quant_block=0,
            state_cache=history,
        )
        return main, history

    arms = [
        jax.jit(functools.partial(call, fn))
        for fn in (
            compressor_v1.compressor_forward_generic,
            compressor_v1.compressor_forward,
        )
    ]
    carry = [(cache, state), (cache, state)]
    for step in range(16):
        pos = jnp.array([start + step], jnp.int32)
        carry = [fn(x[step % 4], pos, c) for fn, c in zip(arms, carry)]
        for expected, actual in zip(*carry):
            np.testing.assert_array_equal(
                np.asarray(actual).view(np.uint8), np.asarray(expected).view(np.uint8)
            )


def test_static_envelope(inputs):
    x, w, _, _, _, state, cache, st, kt = inputs
    cu = jnp.minimum(jnp.arange(257), 1).astype(jnp.int32)
    state = jnp.asarray(state)
    kwargs = dict(head_dim=512, compress_ratio=128, overlap=False, state_block_size=32)
    assert hca_decode_compressor.is_supported(
        x[0], w, cache, state, cu, st, kt, **kwargs
    )
    assert not hca_decode_compressor.is_supported(
        x[0].astype(jnp.float32), w, cache, state, cu, st, kt, **kwargs
    )
    assert not hca_decode_compressor.is_supported(
        x[0], w, cache, state, cu, jnp.zeros(256 * 2081, jnp.int32), kt, **kwargs
    )


@pytest.mark.parametrize("requests", [1, 2, 256])
def test_short_metadata_rows_ignore_adjacent_requests(inputs, requests):
    """Aligned staging may fetch adjacent rows but must never consume them."""
    x, w, ape, norm, cs, state, cache, _, _ = inputs
    state = jnp.asarray(state)
    st = np.full((requests, 5), -7, np.int32)
    st[0] = [3, 0, 2, 1, 7]
    kt = np.full((requests, 1), -7, np.int32)
    kt[0] = 1
    st, kt = jnp.asarray(st.ravel()), jnp.asarray(kt.ravel())
    cu = jnp.minimum(jnp.arange(requests + 1), 1).astype(jnp.int32)
    distribution = jnp.ones(3, jnp.int32)

    def call(fn, hidden, pos, carry):
        main, _, history = fn(
            hidden,
            w,
            ape,
            norm,
            cs,
            pos,
            st,
            cu,
            kt,
            carry[0],
            None,
            distribution,
            state_block_size=32,
            head_dim=512,
            compress_ratio=128,
            overlap=False,
            rms_eps=1e-6,
            quant_block=0,
            state_cache=carry[1],
        )
        return main, history

    arms = [
        jax.jit(functools.partial(call, fn))
        for fn in (
            compressor_v1.compressor_forward_generic,
            compressor_v1.compressor_forward,
        )
    ]
    carry = [(cache, state), (cache, state)]
    for i, position in enumerate(range(126, 130)):
        pos = jnp.array([position], jnp.int32)
        carry = [fn(x[i], pos, c) for fn, c in zip(arms, carry)]
        for expected, actual in zip(*carry):
            np.testing.assert_array_equal(
                np.asarray(actual).view(np.uint8), np.asarray(expected).view(np.uint8)
            )

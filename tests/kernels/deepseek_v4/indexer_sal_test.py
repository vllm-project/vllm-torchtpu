# SPDX-License-Identifier: Apache-2.0
"""Persistent SAL cache must agree with the independent HAS reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compressor_v1 as comp

pytestmark = pytest.mark.tpu_test


def exact(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("request_capacity", [2, 256])
def test_sal_compressor_prefill_decode_and_mixed_state(request_capacity):
    rng = np.random.default_rng(121)
    w = jnp.asarray(rng.normal(0, 0.02, (4096, 512)), jnp.bfloat16)
    ape = jnp.asarray(rng.normal(0, 0.1, (4, 256)), jnp.float32)
    norm = jnp.asarray(rng.uniform(0.8, 1.2, 128), jnp.bfloat16)
    angles = (
        np.arange(96, dtype=np.float32)[:, None] * np.linspace(0.001, 0.01, 32)[None, :]
    )
    cs = jnp.asarray(np.concatenate((np.cos(angles), np.sin(angles)), 1), jnp.float32)
    st = jnp.asarray(
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]], jnp.int32
    ).ravel()
    st = jnp.pad(st.reshape(2, -1), ((0, request_capacity - 2), (0, 0))).ravel()
    kt = jnp.pad(jnp.array([1, 0], jnp.int32), (0, request_capacity - 2))
    # Distinct buffers and permuted physical pages; finite FP32 state sentinels.
    state = rng.normal(0, 0.1, (16, 16, 256)).astype(np.float32)
    state = jnp.asarray(
        state.view(np.uint8).reshape(16, 16, 256, 4).transpose(0, 1, 3, 2).copy()
    )
    oldcache = jnp.zeros((2, 64, 4, 256), jnp.uint8)
    newcache = jnp.zeros((2, 33, 4, 256), jnp.uint8)
    oldstate = newstate = state
    kw = dict(
        state_block_size=8,
        head_dim=128,
        compress_ratio=4,
        overlap=True,
        rms_eps=1e-6,
        quant_block=128,
    )

    def fn(f, x, pos, cu, dist, cache, state):
        return f(
            x,
            w,
            ape,
            norm,
            cs,
            pos,
            st,
            cu,
            kt,
            cache,
            None,
            dist,
            state_cache=state,
            **kw,
        )

    old = jax.jit(lambda *args: fn(comp.compressor_forward_generic, *args))
    new = jax.jit(lambda *args: fn(comp.compressor_forward, *args))
    steps = [
        (list(range(8)) + list(range(4)), [0, 8, 12], [0, 0, 2], "prefill"),
        ([8, 4], [0, 1, 2], [2, 2, 2], "decode"),
    ]
    steps += [([i, i - 4], [0, 1, 2], [2, 2, 2], "decode") for i in range(9, 16)]
    steps += [
        ([16, *range(12, 20)], [0, 1, 9], [1, 1, 2], "mixed"),
        ([17, 0, 0, 0], [0, 1, 1], [1, 1, 1], "padded"),
        ([0, 0, 0, 0], [0, 0, 0], [0, 0, 0], "empty"),
    ]
    steps += [([i], [0, 1, 1], [1, 1, 1], "single") for i in range(18, 24)]
    steps += [
        ([20], [0, 0, 1], [0, 0, 2], "leading_empty"),
        ([0], [0, 0, 0], [0, 0, 0], "single_empty"),
        ([24] + [0] * 63, [0, 1, 1], [1, 1, 1], "padded_64"),
        ([0] * 64, [0, 0, 0], [0, 0, 0], "empty_64"),
    ]
    for i, (pos, cu, dist, kind) in enumerate(steps):
        x = jnp.asarray(rng.normal(0, 0.2, (len(pos), 4096)), jnp.bfloat16)
        args = (
            x,
            jnp.array(pos, jnp.int32),
            jnp.pad(jnp.array(cu, jnp.int32), (0, request_capacity - 2), mode="edge"),
            jnp.array(dist, jnp.int32),
        )
        oldcache, _, oldstate = old(*args, oldcache, oldstate)
        newcache, _, newstate = new(*args, newcache, newstate)
        exact(oldstate, newstate)
        expected = np.asarray(oldcache).reshape(2, 256, 256)[..., :129]
        actual = np.asarray(newcache).reshape(2, 132, 256).transpose(0, 2, 1)[..., :129]
        np.testing.assert_array_equal(expected, actual)
    # Real compressor output consumed by both layouts, including empty/padded metadata.
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        KVLayout,
        streamindex_topk,
    )
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
        streamindex_topk_decode_dispatch,
    )

    q = jnp.asarray(rng.normal(size=(2, 64, 128)), jnp.float8_e4m3fn)
    weights = jnp.ones((2, 64), jnp.float32)
    lens = jnp.array([18, 20], jnp.int32)
    cu = jnp.pad(jnp.arange(3, dtype=jnp.int32), (0, request_capacity - 2), mode="edge")
    dist = jnp.array([2] * 3, jnp.int32)
    args = (q, weights, oldcache, lens, kt, cu, dist)
    a = streamindex_topk(
        *args,
        k=512,
        compression_ratio=4,
        num_kv_pages_per_block=(3, 2, 2),
        num_queries_per_block=(1, 128, 128),
    )
    b = streamindex_topk_decode_dispatch(
        q,
        weights,
        newcache,
        lens,
        kt,
        cu,
        dist,
        k=512,
        compression_ratio=4,
        num_kv_pages_per_block=(3, 2, 2),
        num_queries_per_block=(1, 128, 128),
        kv_layout=KVLayout.SEQ_ALONG_LANE,
    )

    exact(a, b)

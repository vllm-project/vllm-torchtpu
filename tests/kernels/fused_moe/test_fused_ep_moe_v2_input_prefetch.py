# SPDX-License-Identifier: Apache-2.0
"""CPU regression for the FFN hooks used by continuous input prefetch."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import host, kernel

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("logical_inter", [None, 768])
def test_input_hooks_preserve_group_reduction_and_fire_once_per_output_chunk(
    logical_inter,
):
    rng = np.random.default_rng(2160)
    rows, hidden, inter, block = 128, 512, 1024, 512
    x = jnp.asarray(rng.integers(-3, 4, (rows, hidden)), host.FP8)
    w1 = rng.integers(-3, 4, (hidden, 2 * inter)).astype(np.float32)
    w2 = rng.integers(-3, 4, (inter, hidden)).astype(np.float32)
    if logical_inter is not None:
        w1[:, logical_inter:inter] = 0
        w1[:, inter + logical_inter :] = 0
        w2[logical_inter:] = 0
    w1, w2 = jnp.asarray(w1, host.FP8), jnp.asarray(w2, host.FP8)
    s1 = jnp.full((1, 2 * inter), 0.01, jnp.float32)
    s2 = jnp.broadcast_to(jnp.array([[0.01], [0.03]]), (2, hidden))
    row_scale = jnp.linspace(0.01, 0.2, rows)[:, None]
    events = []

    def gate_reader(b, lo, hi):
        events.append(("gate", b, lo))
        return w1[b * block : (b + 1) * block, lo:hi]

    def down_reader(b, lo, hi, *, block_rows=block):
        events.append(("down", b, lo))
        return w2[b * block : b * block + block_rows, lo:hi]

    args = (x, row_scale, gate_reader, down_reader, s1, s2)
    options = dict(qb=block, logical_intermediate_size=logical_inter)
    expected = kernel.expert_ffn_blockscale(*args, **options)
    events.clear()
    actual = kernel.expert_ffn_blockscale(
        *args,
        **options,
        after_gate_up=lambda: events.append(("index_wait",)),
        after_down_dot=lambda chunk: events.append(("input_dma", chunk)),
    )
    for lhs, rhs in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))
    wait = events.index(("index_wait",))
    assert all(event[0] == "gate" for event in events[:wait])
    assert all(event[0] != "gate" for event in events[wait + 1 :])
    assert [event for event in events if event[0] == "input_dma"] == [
        ("input_dma", 0),
        ("input_dma", 1),
    ]
    for chunk in range(hidden // kernel.FP4_OUTPUT_CHUNK):
        first = events.index(("down", 0, chunk * kernel.FP4_OUTPUT_CHUNK))
        issue = events.index(("input_dma", chunk))
        second = events.index(("down", 1, chunk * kernel.FP4_OUTPUT_CHUNK))
        assert wait < first < issue < second

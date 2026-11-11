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
"""Real operator continuation; native Raiden transport is tested separately."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.tpu_test


def _relocate(owner, source, destination):
    before = np.asarray(owner).copy()
    moved = np.full(before.view(np.uint8).shape, 0x42, np.uint8)
    moved[destination] = before.view(np.uint8)[source]
    result = jax.device_put(moved.view(before.dtype))
    np.testing.assert_array_equal(np.asarray(result).view(np.uint8), moved)
    return result


def _assert_pages(actual, expected, destination, source):
    np.testing.assert_array_equal(
        np.asarray(actual)[destination].view(np.uint8),
        np.asarray(expected)[source].view(np.uint8),
    )


def _assert_written_checkpoints(actual, expected, destination, source, state_bytes):
    a = np.asarray(actual)[destination].view(np.uint8).reshape(len(destination), -1)
    b = np.asarray(expected)[source].view(np.uint8).reshape(len(source), -1)
    np.testing.assert_array_equal(a[:, :state_bytes], b[:, :state_bytes])
    # Only slot 0 was transferred. Padding in newly written scratch slots
    # belongs to D and must retain its poison, rather than P's zero padding.
    np.testing.assert_array_equal(a[1:, state_bytes:], 0x42)


@pytest.mark.parametrize("producer_k", [1, 3])
@pytest.mark.parametrize("prefix", [1, 3, 4, 5, 17, 129])
def test_pd_ple_prefill_commits_slot_zero_and_relocates(prefix, producer_k):
    from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv
    from vllm_torchtpu.kernels.qwen4_exp.pool import pooled_short_conv, read_region

    rng = np.random.default_rng(20261110)
    rows, channels, state_len, k = 8, 128, 9, 3
    values = jnp.asarray(rng.normal(0, 0.2, (prefix + 6, channels)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(0, 0.2, (channels, 4)), jnp.bfloat16)
    run = jax.jit(
        functools.partial(pooled_short_conv, dilation=3, state_len=state_len),
        donate_argnums=(2,),
    )
    source_pages, dest_pages = list(range(1, 5)), list(range(6, 10))
    source = jnp.zeros((12, 1664, 1, 4, 256), jnp.float8_e4m3fn)
    history = jnp.zeros((rows, state_len, channels), jnp.bfloat16)
    read = jax.jit(
        functools.partial(
            read_region, elements=state_len * channels, dtype=jnp.bfloat16
        )
    )

    def step(pool, start, count, pages, *, verify=False, offset=0):
        x = jnp.pad(values[start : start + count], ((0, 16 - count), (0, 0)))
        ids = jnp.array([pages[0]] + [0] * (rows - 1), jnp.int32)
        checkpoints = (
            jnp.zeros((rows, len(pages)), jnp.int32).at[0].set(jnp.array(pages))
        )
        offsets = jnp.zeros(12, jnp.int32).at[pages[0]].set(offset)
        starts = jnp.array([0] + [count] * rows, jnp.int32)
        initial = jnp.array([start > 0] + [False] * (rows - 1))
        dist = jnp.array([1, 1, 1] if verify or count == 1 else [0, 0, 1], jnp.int32)
        return run(x, weight, pool, starts, ids, initial, checkpoints, offsets, dist)

    for start in range(0, prefix, 16):
        count = min(16, prefix - start)
        source, actual = step(source, start, count, source_pages[: producer_k + 1])
        x = jnp.pad(values[start : start + count], ((0, 16 - count), (0, 0)))
        expected, history = short_conv(
            x,
            weight,
            history,
            jnp.array([0] + [count] * rows, jnp.int32),
            jnp.array([0] + [-1] * (rows - 1), jnp.int32),
            jnp.array([start > 0] + [False] * (rows - 1)),
            dilation=3,
        )
        np.testing.assert_array_equal(
            np.asarray(actual)[:count], np.asarray(expected)[:count]
        )
        state = read(source, jnp.array(source_pages, jnp.int32)).reshape(
            k + 1, state_len, channels
        )
        np.testing.assert_array_equal(np.asarray(state)[0], np.asarray(history)[0])
        np.testing.assert_array_equal(
            np.asarray(source)[source_pages[1:]].view(np.uint8), 0
        )
    destination = _relocate(source, source_pages[:1], dest_pages[:1])
    saved_dest = np.asarray(destination).copy()
    source, expected = step(source, prefix, 1, source_pages)
    destination, actual = step(destination, prefix, 1, dest_pages)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    _assert_pages(destination, source, dest_pages[:1], source_pages[:1])
    np.testing.assert_array_equal(
        np.asarray(destination)[dest_pages[1:]].view(np.uint8), 0x42
    )
    if prefix > 1:
        _, wrong = step(jax.device_put(saved_dest), prefix, 1, dest_pages, offset=3)
        assert not np.array_equal(np.asarray(wrong)[0], np.asarray(expected)[0])
    source, expected = step(source, prefix + 1, k + 1, source_pages, verify=True)
    destination, actual = step(destination, prefix + 1, k + 1, dest_pages, verify=True)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    # write_region pads this BF16 history to complete 1024-byte pool rows.
    state_bytes = ((state_len * channels * 2 + 1023) // 1024) * 1024
    _assert_written_checkpoints(
        destination, source, dest_pages, source_pages, state_bytes
    )


@pytest.mark.parametrize("producer_k", [1, 3])
@pytest.mark.parametrize("prefix", [1, 17])
def test_pd_gdn_prefill_commits_slot_zero_and_relocates(prefix, producer_k):
    from vllm_torchtpu.layers.core.gdn_attention import (
        run_jax_gdn_attention_pooled_local,
    )

    rng = np.random.default_rng(20261111)
    k, n_kq, n_v, width, channels = 3, 16, 48, 128, 10240

    def rand(shape, dtype=jnp.bfloat16):
        return jnp.asarray(rng.normal(0, 0.1, shape), dtype)

    qkv, beta, alpha = (
        rand((prefix + 6, channels)),
        rand((prefix + 6, n_v)),
        rand((prefix + 6, n_v)),
    )
    weight, bias = rand((channels, 1, 4)), rand((channels,))
    a_log, dt_bias = rand((n_v,), jnp.float32), rand((n_v,), jnp.float32)
    options = dict(
        n_kq=n_kq,
        n_v=n_v,
        d_k=width,
        d_v=width,
        kernel_size=4,
        pool_block_tokens=1664,
        recurrent_state_dtype=jnp.bfloat16,
    )
    runs = {
        spec: jax.jit(
            functools.partial(
                run_jax_gdn_attention_pooled_local, **options, num_spec_tokens=spec
            ),
            donate_argnums=(3,),
        )
        for spec in (0, 1, k)
    }
    source_pages, dest_pages = list(range(1, 5)), list(range(6, 10))
    source = jnp.zeros((12, 1664, 1, 4, 256), jnp.float8_e4m3fn)
    reference = jnp.zeros_like(source)

    def step(pool, start, count, pages, *, spec=k, verify=False, offset=0):
        def padded(x):
            return jnp.pad(x[start : start + count], ((0, 32 - count), (0, 0)))

        return runs[spec](
            padded(qkv),
            padded(beta),
            padded(alpha),
            pool,
            weight,
            bias,
            a_log,
            dt_bias,
            jnp.array([0, count], jnp.int32),
            jnp.array([pages[0]], jnp.int32),
            jnp.array([1, 1, 1] if verify or count == 1 else [0, 0, 1], jnp.int32),
            jnp.array([start + count], jnp.int32),
            read_offsets=jnp.array([offset], jnp.int32) if spec else None,
            ckpt_indices=jnp.array([pages], jnp.int32) if spec else None,
        )

    for start in range(0, prefix, 16):
        count = min(16, prefix - start)
        source, actual = step(
            source, start, count, source_pages[: producer_k + 1], spec=producer_k
        )
        reference, expected = step(
            reference, start, count, source_pages, spec=k if producer_k == 1 else 0
        )
        # K0/K1/K3 decode outputs can round differently, but committed state
        # must be identical. Both continuations below use the consumer's K3.
        if count > 1:
            np.testing.assert_array_equal(
                np.asarray(actual)[:count], np.asarray(expected)[:count]
            )
        _assert_pages(source, reference, [1], [1])
        np.testing.assert_array_equal(np.asarray(source)[2:5].view(np.uint8), 0)
    destination = _relocate(source, source_pages[:1], dest_pages[:1])
    saved_dest = np.asarray(destination).copy()
    if producer_k == 1:
        source = reference
    source, expected = step(source, prefix, 1, source_pages)
    destination, actual = step(destination, prefix, 1, dest_pages)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    _assert_pages(destination, source, dest_pages[:1], source_pages[:1])
    np.testing.assert_array_equal(
        np.asarray(destination)[dest_pages[1:]].view(np.uint8), 0x42
    )
    _, wrong = step(jax.device_put(saved_dest), prefix, 1, dest_pages, offset=3)
    assert not np.array_equal(np.asarray(wrong)[0], np.asarray(expected)[0])
    source, expected = step(source, prefix + 1, k + 1, source_pages, verify=True)
    destination, actual = step(destination, prefix + 1, k + 1, dest_pages, verify=True)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    # Unified GDN stores BF16 recurrent state, then FP32 convolution state.
    state_bytes = n_v * width * width * 2 + channels * (4 - 1) * 4
    _assert_written_checkpoints(
        destination, source, dest_pages, source_pages, state_bytes
    )


@pytest.mark.parametrize("prefix", [1, 3, 4, 5, 17, 127, 128, 129, 1665])
def test_pd_draft_qsa_ring_and_kv_continue_after_relocation(prefix):
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compact_pooled_qsa_step

    rng = np.random.default_rng(20261112)

    def rand(shape):
        return jnp.asarray(rng.normal(0, 0.2, shape), jnp.bfloat16)

    rows, tokens = 8, 16
    inputs = [rand((prefix + 6, width)) for width in (24 * 256 * 2, 512, 512, 5 * 128)]
    norms = [rand((width,)) for width in (256, 256, 128, 128)]
    angle = jnp.arange(4096)[:, None] * jnp.arange(32)[None, :] / 8192
    trig = jnp.pad(
        jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], 1), ((0, 0), (0, 64))
    ).astype(jnp.bfloat16)
    options = dict(
        compress_ratio=4,
        token_topk=2048,
        norm_eps=1e-6,
        rotary_dim=64,
        k_scale=0.375,
        v_scale=2.5,
        ring_capacity=8,
    )
    run = jax.jit(
        functools.partial(compact_pooled_qsa_step, **options), donate_argnums=(10, 11)
    )
    owners = (
        jnp.zeros((8, 1664, 1, 4, 256), jnp.float8_e4m3fn),
        jnp.zeros((8, 208, 1, 2, 128), jnp.bfloat16),
    )

    def step(cache, start, count, pages, ring, *, verify=False):
        selected = [
            jnp.pad(value[start : start + count], ((0, tokens - count), (0, 0)))
            for value in inputs
        ]
        table = jnp.zeros((rows, 2), jnp.int32).at[0].set(jnp.array(pages))
        return run(
            *selected,
            *norms,
            jnp.arange(tokens, dtype=jnp.int32) + start,
            trig,
            *cache,
            jnp.array([0] + [count] * rows, jnp.int32),
            jnp.array([start] + [0] * (rows - 1), jnp.int32),
            jnp.array([ring] + [0] * (rows - 1), jnp.int32),
            table,
            table,
            jnp.array([1, 1, 1] if verify or count == 1 else [0, 0, 1], jnp.int32),
        )

    for start in range(0, prefix, 16):
        result = step(owners, start, min(16, prefix - start), [1, 2], 6)
        owners = result[:2]
    destination = (
        _relocate(owners[0], [1, 2], [4, 5]),
        _relocate(owners[1], [1, 2, 6], [4, 5, 7]),
    )
    for start, count in ((prefix, 1), (prefix + 1, 4)):
        expected = step(owners, start, count, [1, 2], 6, verify=True)
        actual = step(destination, start, count, [4, 5], 7, verify=True)
        np.testing.assert_array_equal(
            np.asarray(actual[2])[:count], np.asarray(expected[2])[:count]
        )
        _assert_pages(actual[0], expected[0], [4, 5], [1, 2])
        _assert_pages(actual[1], expected[1], [4, 5, 7], [1, 2, 6])
        owners, destination = expected[:2], actual[:2]

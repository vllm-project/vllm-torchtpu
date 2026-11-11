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
"""Rejected suffixes must preserve indexer keys and selectable PLE histories."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv
from vllm_torchtpu.kernels.qwen4_exp.qsa import _indexer_updates


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("prefix", [1, 3, 4, 7, 127, 128, 129])
def test_qsa_ring_preserves_committed_open_group_after_every_rejection(k, prefix):
    ratio, capacity, width = 4, 8, 16
    rng = np.random.default_rng(31)
    values = jnp.asarray(rng.normal(size=(prefix + k + 1, width)), jnp.bfloat16)
    next_values = jnp.asarray(rng.normal(size=(ratio, width)), jnp.bfloat16)
    trig = jnp.concatenate([jnp.ones((256, 4)), jnp.zeros((256, 4))], axis=-1)
    empty_ring = jnp.zeros((1, capacity, width), jnp.bfloat16)
    empty_tags = jnp.full((1, capacity, 2), -1, jnp.int32)

    def update(x, context, ring, tags):
        return _indexer_updates(
            x,
            jnp.arange(context, context + len(x), dtype=jnp.int32),
            ring,
            tags,
            jnp.zeros(width, jnp.bfloat16),
            trig,
            jnp.array([0, len(x)], jnp.int32),
            jnp.array([context], jnp.int32),
            jnp.array([0], jnp.int32),
            ratio,
            1e-6,
            8,
            ring_capacity=capacity,
        )

    *_, ring, tags = update(values[:prefix], 0, empty_ring, empty_tags)
    *_, ring, tags = update(values[prefix:], prefix, ring, tags)
    for accepted_drafts in range(k + 1):
        committed = prefix + 1 + accepted_drafts
        open_start = committed // ratio * ratio
        for logical in range(open_start, committed):
            np.testing.assert_array_equal(ring[0, logical % capacity], values[logical])
            assert int(tags[0, logical % capacity, 0]) == logical
        actual = update(next_values, committed, ring, tags)
        reference = update(
            jnp.concatenate([values[:committed], next_values]),
            0,
            empty_ring,
            empty_tags,
        )
        mask = np.asarray(actual[1])
        np.testing.assert_array_equal(
            np.asarray(actual[0])[mask], np.asarray(reference[0])[-ratio:][mask]
        )


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_ple_checkpoint_selection_matches_each_accepted_prefix(k):
    rng = np.random.default_rng(81)
    length, channels, requests = 9, 16, 4
    # Verify, short prefill, long prefill and an empty padded row.
    query_lengths = [k + 1, 2, 11, 0]
    starts = np.cumsum([0] + query_lengths).astype(np.int32)
    x = jnp.asarray(rng.normal(size=(starts[-1] + 3, channels)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(requests, length, channels)), jnp.bfloat16)
    initial = jnp.array([True, False, True, False])
    weight = jnp.asarray(rng.normal(size=(channels, 4)), jnp.bfloat16)
    slots = jnp.array([0, 1, 2, -1], jnp.int32)
    output, states = short_conv(
        x,
        weight,
        history,
        jnp.asarray(starts),
        slots,
        initial,
        dilation=3,
        speculative_window_size=k + 1,
        request_distribution=jnp.array([1, 2, 3], jnp.int32),
    )
    expected_output, _ = short_conv(
        x, weight, history, jnp.asarray(starts), slots, initial, dilation=3
    )
    np.testing.assert_array_equal(output, expected_output)
    for accepted_drafts in range(k + 1):
        consumed = accepted_drafts + 1
        expected = np.concatenate([np.asarray(history[0]), np.asarray(x[:consumed])])[
            -length:
        ]
        np.testing.assert_array_equal(states[0, accepted_drafts], expected)
    for row in (1, 2):
        past = (
            np.asarray(history[row]) if initial[row] else np.zeros((length, channels))
        )
        expected = np.concatenate([past, np.asarray(x[starts[row] : starts[row + 1]])])[
            -length:
        ]
        np.testing.assert_array_equal(states[row, 0], expected)
    np.testing.assert_array_equal(states[3], 0)


@pytest.mark.tpu_test
@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("channels", [128, 10240])
def test_pooled_ple_all_rollback_offsets_match_next_token(k, channels):
    import jax

    from vllm_torchtpu.kernels.qwen4_exp.pool import (
        pooled_short_conv,
        read_region,
        write_region,
    )

    rng = np.random.default_rng(84)
    rows, state_len = 8, 9
    ids = jnp.array([1] + [0] * (rows - 1), jnp.int32)
    checkpoint_ids = jnp.zeros((rows, k + 1), jnp.int32).at[0].set(jnp.arange(1, k + 2))
    starts = jnp.array([0] + [k + 1] * rows, jnp.int32)
    initial = jnp.array([True] + [False] * (rows - 1))
    distribution = jnp.array([1, 1, 1], jnp.int32)
    read_offsets = jnp.zeros(k + 4, jnp.int32)
    history = jnp.asarray(rng.normal(size=(rows, state_len, channels)), jnp.bfloat16)
    x = jnp.asarray(rng.normal(size=(16, channels)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(size=(channels, 4)), jnp.bfloat16)
    pool = jnp.full((k + 4, 1664, 1, 4, 256), 0.5, jnp.float8_e4m3fn)
    pool = jax.jit(write_region, donate_argnums=(0,))(
        pool, history.reshape(rows, -1), ids
    )
    original = np.asarray(pool).view(np.uint8).copy()
    run = jax.jit(
        pooled_short_conv,
        donate_argnums=(2,),
        static_argnames=("dilation", "state_len"),
    )
    pool, actual = run(
        x,
        weight,
        pool,
        starts,
        ids,
        initial,
        checkpoint_ids,
        read_offsets,
        distribution,
        dilation=3,
        state_len=state_len,
    )
    expected, _ = short_conv(
        x,
        weight,
        history,
        starts,
        jnp.array([0] + [-1] * (rows - 1), jnp.int32),
        initial,
        dilation=3,
    )
    np.testing.assert_array_equal(
        np.asarray(actual)[: k + 1], np.asarray(expected)[: k + 1]
    )
    read = jax.jit(read_region, static_argnames=("elements", "dtype"))
    stored = read(
        pool, jnp.arange(1, k + 2), elements=state_len * channels, dtype=jnp.bfloat16
    ).reshape(k + 1, state_len, channels)
    verified = np.asarray(pool).copy()
    untouched = [0, k + 2, k + 3]
    np.testing.assert_array_equal(
        verified.view(np.uint8)[untouched], original[untouched]
    )
    next_x = jnp.asarray(rng.normal(size=(16, channels)), jnp.bfloat16)
    next_starts = jnp.array([0] + [1] * rows, jnp.int32)
    for accepted in range(k + 1):
        reference_history = np.concatenate(
            (np.asarray(history[0]), np.asarray(x[: accepted + 1]))
        )[-state_len:]
        np.testing.assert_array_equal(np.asarray(stored[accepted]), reference_history)
        # Rejection selects an arbitrary physical checkpoint, then only the
        # base checkpoint is written by the following one-token window.
        offsets = read_offsets.at[1].set(accepted)
        restored_pool = jax.device_put(verified.copy())
        restored_pool, next_actual = run(
            next_x,
            weight,
            restored_pool,
            next_starts,
            ids,
            initial,
            checkpoint_ids,
            offsets,
            distribution,
            dilation=3,
            state_len=state_len,
        )
        reference_states = history.at[0].set(jnp.asarray(reference_history))
        next_expected, _ = short_conv(
            next_x,
            weight,
            reference_states,
            next_starts,
            jnp.array([0] + [-1] * (rows - 1), jnp.int32),
            initial,
            dilation=3,
        )
        np.testing.assert_array_equal(
            np.asarray(next_actual)[0], np.asarray(next_expected)[0]
        )
        after = np.asarray(restored_pool).view(np.uint8)
        keep = [0] + list(range(2, k + 4))
        np.testing.assert_array_equal(after[keep], verified.view(np.uint8)[keep])


@pytest.mark.tpu_test
@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("prefix", [3, 7, 127, 128, 1663, 1664])
def test_pooled_qsa_rejection_matches_committed_prefix_next_output(k, prefix):
    import jax

    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compact_pooled_qsa_step
    from vllm_torchtpu.kernels.qwen4_exp.qsa_state import write_ring_and_tags

    rng = np.random.default_rng(731)

    def rand(shape):
        return jnp.asarray(rng.normal(0, 0.3, shape), jnp.bfloat16)

    rows, tokens = 8, 16
    pool = jnp.zeros((7, 1664, 1, 4, 256), jnp.float8_e4m3fn)
    index_pool = jnp.zeros((7, 208, 1, 2, 128), jnp.bfloat16)
    ring = rand((rows, 8, 128))
    tags = jnp.zeros((rows, 8, 2), jnp.int32)
    tags = tags.at[0, :, 0].set(prefix - 1 - (prefix - 1 - jnp.arange(8)) % 8)
    owners = jnp.array([5] + [0] * (rows - 1), jnp.int32)
    index_pool = jax.jit(write_ring_and_tags, donate_argnums=(0,))(
        index_pool, ring.reshape(rows, -1), tags.reshape(rows, -1), owners
    )
    original = (np.asarray(pool).copy(), np.asarray(index_pool).copy())
    main = jnp.zeros((rows, 2), jnp.int32).at[0].set(jnp.array([1, 2]))
    comp = main
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
        compact_pooled_qsa_step, donate_argnums=(10, 11), static_argnames=tuple(options)
    )
    verify = [rand((tokens, width)) for width in (24 * 256 * 2, 512, 512, 5 * 128)]
    next_inputs = [rand((tokens, width)) for width in (24 * 256 * 2, 512, 512, 5 * 128)]

    def step(owner, inputs, context, length, ring_ids=owners):
        starts = jnp.array([0] + [length] * rows, jnp.int32)
        contexts = jnp.array([context] + [0] * (rows - 1), jnp.int32)
        positions = jnp.arange(tokens, dtype=jnp.int32) + context
        return run(
            *inputs,
            *norms,
            positions,
            trig,
            *owner,
            starts,
            contexts,
            ring_ids,
            main,
            comp,
            jnp.ones((3,), jnp.int32),  # One live decode/verify request.
            **options,
        )

    def copy_owners(values):
        return tuple(jax.device_put(value.copy()) for value in values)

    speculative = step(copy_owners(original), verify, prefix, k + 1)
    verified = tuple(np.asarray(owner).copy() for owner in speculative[:2])
    for after, before in zip(verified, original):
        np.testing.assert_array_equal(
            after[[0, 6]].view(np.uint8), before[[0, 6]].view(np.uint8)
        )
    for accepted in range(k + 1):
        # The reference never executes the rejected suffix. Compare the next
        # output after rewind and overwrite, including compressed/page edges.
        committed = step(copy_owners(original), verify, prefix, accepted + 1)[:2]
        expected_main, expected_index, expected, expected_blocks = step(
            committed, next_inputs, prefix + accepted + 1, 1
        )
        actual_main, actual_index, actual, actual_blocks = step(
            copy_owners(verified), next_inputs, prefix + accepted + 1, 1
        )
        np.testing.assert_allclose(
            np.asarray(actual)[0].astype(np.float32),
            np.asarray(expected)[0].astype(np.float32),
            rtol=0.05,
            atol=0.03,
        )
        np.testing.assert_array_equal(
            np.asarray(actual_blocks)[0], np.asarray(expected_blocks)[0]
        )
        # Both stores use the same logical pages and visible committed prefix.
        end = prefix + accepted + 2
        for actual_owner, expected_owner, width, length in (
            (actual_main, expected_main, 1024, end),
            (actual_index, expected_index, 128, end // 4),
        ):
            actual_kv = np.asarray(actual_owner)[[1, 2]].reshape(-1, width)[:length]
            expected_kv = np.asarray(expected_owner)[[1, 2]].reshape(-1, width)[:length]
            np.testing.assert_array_equal(
                actual_kv.view(np.uint8), expected_kv.view(np.uint8)
            )
    idle = step(
        copy_owners(verified), next_inputs, prefix, 1, jnp.zeros(rows, jnp.int32)
    )
    for after, before in zip(idle[:2], verified):
        np.testing.assert_array_equal(
            np.asarray(after).view(np.uint8), before.view(np.uint8)
        )


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_qsa_many_verify_windows_match_mtp_off_token_reference(k):
    import jax

    rng = np.random.default_rng(193)
    raw = jnp.asarray(rng.normal(size=(256, 16)), jnp.bfloat16)
    angle = jnp.arange(256)[:, None] * jnp.arange(4)[None, :] / 1024
    trig = jnp.concatenate((jnp.cos(angle), jnp.sin(angle)), -1)

    def step(x, context, length, ring, tags, capacity):
        return _indexer_updates(
            x,
            jnp.arange(16, dtype=jnp.int32) + context,
            ring,
            tags,
            jnp.zeros(16, jnp.bfloat16),
            trig,
            jnp.array([0, length], jnp.int32),
            jnp.array([context], jnp.int32),
            jnp.array([0], jnp.int32),
            4,
            1e-6,
            8,
            ring_capacity=capacity,
        )

    run = jax.jit(step, static_argnames=("capacity",))
    rings = [jnp.zeros((1, 8, 16), jnp.bfloat16) for _ in range(2)]
    tags = [jnp.full((1, 8, 2), -1, jnp.int32) for _ in range(2)]
    context = 0
    # Initial prompt plus repeated accept/reject windows, traversing both
    # ring halves and many compression groups. The reference is K=0/ring=4.
    for length in (16, 5):
        for i, capacity in enumerate((4, 8)):
            *_, rings[i], tags[i] = run(
                raw[context : context + 16],
                context,
                length,
                rings[i],
                tags[i],
                capacity,
            )
        context += length
    for turn in range(24):
        accepted = turn % (k + 1)
        inputs = raw[context : context + 16]
        # Rejected suffix must differ from the committed sequence.
        inputs = inputs.at[accepted + 1 : k + 1].set(-inputs[accepted + 1 : k + 1])
        actual = run(inputs, context, k + 1, rings[1], tags[1], 8)
        rings[1], tags[1] = actual[-2:]
        for offset in range(accepted + 1):
            expected = run(
                raw[context + offset : context + offset + 16],
                context + offset,
                1,
                rings[0],
                tags[0],
                4,
            )
            rings[0], tags[0] = expected[-2:]
            if bool(expected[1][0]):
                np.testing.assert_array_equal(actual[0][offset], expected[0][0])
        context += accepted + 1


@pytest.mark.tpu_test
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_pooled_ple_batched_mixed_acceptance(k):
    import jax

    from vllm_torchtpu.kernels.qwen4_exp.pool import (
        pooled_short_conv,
        read_region,
        write_region,
    )

    rng = np.random.default_rng(337)
    requests, rows, channels, history_len = 4, 8, 10240, 9
    length = k + 1
    pages = (
        1 + jnp.arange(requests * length, dtype=jnp.int32).reshape(length, requests).T
    )[jnp.array([3, 0, 2, 1])]
    checkpoints = jnp.pad(pages, ((0, rows - requests), (0, 0)))
    ids = checkpoints[:, 0]
    query_lengths = [length, length, 2, 11] + [0] * (rows - requests)
    starts = jnp.asarray(np.cumsum([0] + query_lengths), jnp.int32)
    initial = jnp.array([True, True, False, True] + [False] * (rows - requests))
    slots = jnp.where(jnp.arange(rows) < requests, jnp.arange(rows), -1)
    distribution = jnp.array([2, 2, requests], jnp.int32)
    history = jnp.asarray(rng.normal(size=(rows, history_len, channels)), jnp.bfloat16)
    x = jnp.asarray(rng.normal(size=(32, channels)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(size=(channels, 4)), jnp.bfloat16)
    pool = jnp.full((requests * length + 3, 1664, 1, 4, 256), 0.5, jnp.float8_e4m3fn)
    pool = jax.jit(write_region, donate_argnums=(0,))(
        pool, history.reshape(rows, -1), ids
    )
    original = np.asarray(pool).view(np.uint8).copy()
    run = jax.jit(
        pooled_short_conv,
        donate_argnums=(2,),
        static_argnames=("dilation", "state_len"),
    )
    pool, actual = run(
        x,
        weight,
        pool,
        starts,
        ids,
        initial,
        checkpoints,
        jnp.zeros(pool.shape[0], jnp.int32),
        distribution,
        dilation=3,
        state_len=history_len,
    )
    expected, _ = short_conv(x, weight, history, starts, slots, initial, dilation=3)
    np.testing.assert_array_equal(actual, expected)
    read = jax.jit(read_region, static_argnames=("elements", "dtype"))
    stored = read(
        pool, pages.reshape(-1), elements=history_len * channels, dtype=jnp.bfloat16
    ).reshape(requests, length, history_len, channels)
    verified = np.asarray(pool).copy()
    for row in range(requests):
        for offset in range(length if row < 2 else 1):
            consumed = offset + 1 if row < 2 else query_lengths[row]
            past = (
                np.asarray(history[row])
                if initial[row]
                else np.zeros((history_len, channels))
            )
            reference = np.concatenate(
                (past, np.asarray(x[starts[row] : starts[row] + consumed]))
            )[-history_len:]
            np.testing.assert_array_equal(stored[row, offset], reference)
    unused = np.asarray(pages[2:, 1:]).reshape(-1)
    np.testing.assert_array_equal(verified.view(np.uint8)[unused], original[unused])
    next_x = jnp.asarray(rng.normal(size=(16, channels)), jnp.bfloat16)
    next_starts = jnp.array(
        list(range(requests + 1)) + [requests] * (rows - requests), jnp.int32
    )
    order = jnp.array([1, 3, 0, 2], jnp.int32)
    next_checkpoints = jnp.pad(pages[order], ((0, rows - requests), (0, 0)))
    next_ids = next_checkpoints[:, 0]
    for shift in range(length):
        accepted = jnp.array([shift, (shift + 1) % length, 0, 0], jnp.int32)
        offsets = jnp.zeros(pool.shape[0], jnp.int32).at[pages[:, 0]].set(accepted)
        reference = np.asarray(history).copy()
        for new_row, old_row in enumerate(np.asarray(order)):
            m = int(accepted[old_row])
            reference[new_row] = np.asarray(stored[old_row, m])
        restored, output = run(
            next_x,
            weight,
            jax.device_put(verified.copy()),
            next_starts,
            next_ids,
            jnp.arange(rows) < requests,
            next_checkpoints,
            offsets,
            jnp.full((3,), requests, jnp.int32),
            dilation=3,
            state_len=history_len,
        )
        expected, final_history = short_conv(
            next_x,
            weight,
            jnp.asarray(reference),
            next_starts,
            slots,
            jnp.arange(rows) < requests,
            dilation=3,
        )
        np.testing.assert_array_equal(output, expected)
        saved = read(
            restored,
            pages[order, 0],
            elements=history_len * channels,
            dtype=jnp.bfloat16,
        ).reshape(requests, history_len, channels)
        np.testing.assert_array_equal(saved, final_history[:requests])
        untouched = sorted(
            set(range(pool.shape[0])) - set(np.asarray(pages[:, 0]).tolist())
        )
        np.testing.assert_array_equal(
            np.asarray(restored).view(np.uint8)[untouched],
            verified.view(np.uint8)[untouched],
        )
    np.testing.assert_array_equal(
        verified.view(np.uint8)[[0, -2, -1]], original[[0, -2, -1]]
    )

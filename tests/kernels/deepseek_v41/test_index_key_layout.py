# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native halfplane keys preserve values, owner geometry, and score masks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout as layout
from vllm_torchtpu.kernels.deepseek_v41.index_selection import persistent_scores, scores
from vllm_torchtpu.kernels.deepseek_v41.index_word_scores import score_word_rows


def native_pool(keys):
    bits = layout.encode_records(keys)
    return (bits[::2] | (bits[1::2] << 16)).astype(jnp.int32)


def dense_scores(q, weights, keys):
    split = jnp.concatenate((keys[..., ::2], keys[..., 1::2]), axis=-1)
    logits = jnp.einsum(
        "...qhd,...kd->...qhk", q, split, preferred_element_type=jnp.float32
    )
    return (jnp.maximum(logits, 0) * weights[..., None]).sum(-2)


@pytest.mark.cpu_test
def test_all_bf16_patterns_gather_and_inverse():
    bits = jnp.arange(65536, dtype=jnp.uint16).reshape(-1, 128)
    keys = jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
    native = native_pool(keys)
    expected = jax.lax.bitcast_convert_type(
        bits.astype(jnp.uint32) << 16, jnp.float32
    ).astype(jnp.bfloat16)
    actual = layout.gather_records(
        (native[:128], native[128:]), jnp.arange(512, dtype=jnp.int32)
    )
    np.testing.assert_array_equal(
        layout.to_ordinary(actual).view(jnp.uint16), expected.view(jnp.uint16)
    )
    np.testing.assert_array_equal(
        layout.unpack_rows(native).view(jnp.uint16), actual.view(jnp.uint16)
    )
    ids = jnp.array([[-1, 255, 256], [511, 512, 1]], jnp.int32)
    selected = layout.gather_records((native[:128], native[128:]), ids)
    np.testing.assert_array_equal(
        selected[0, 1].view(jnp.uint16), actual[255].view(jnp.uint16)
    )
    np.testing.assert_array_equal(
        selected[0, 2].view(jnp.uint16), actual[256].view(jnp.uint16)
    )
    assert np.all(np.asarray(selected)[[0, 1], [0, 1]] == 0)


@pytest.mark.cpu_test
def test_merge_updates_both_halfplanes_and_preserves_untouched_bits():
    initial = np.random.default_rng(3).integers(
        -(2**31), 2**31 - 1, (4, 128), dtype=np.int32
    )
    ids = jnp.array([0, 1, 3, -1, 7, 8], jnp.int32)
    raw = np.random.default_rng(7).integers(0, 65536, (6, 128), dtype=np.uint16)
    keys = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.bfloat16)
    rows, updates = jax.jit(layout.merge_record_rows)(jnp.asarray(initial), ids, keys)
    expected = initial.view(np.uint32).copy()
    normalized = jax.lax.bitcast_convert_type(
        jnp.asarray(raw, jnp.uint32) << 16, jnp.float32
    ).astype(jnp.bfloat16)
    split = np.concatenate(
        (
            np.asarray(normalized.view(jnp.uint16))[:, ::2],
            np.asarray(normalized.view(jnp.uint16))[:, 1::2],
        ),
        axis=1,
    ).astype(np.uint32)
    for i, record in enumerate(np.asarray(ids)):
        if 0 <= record < 8:
            shift = np.uint32(16 * (record % 2))
            mask = np.uint32(65535) << shift
            expected[record // 2] = (expected[record // 2] & ~mask) | (
                split[i] << shift
            )
    result = (
        jnp.asarray(initial)
        .at[jnp.where(rows >= 0, rows, initial.shape[0])]
        .set(updates, mode="drop")
    )
    np.testing.assert_array_equal(np.asarray(result).view(np.uint32), expected)
    np.testing.assert_array_equal(updates[0], updates[1])
    assert int(rows[3]) == -1 and int(rows[5]) == -1


@pytest.mark.cpu_test
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("record_mask", [False, True])
def test_all_score_routes_match_dense(monkeypatch, prefetch, record_mask):
    # Exercise record masking across score windows as well as within key tiles.
    monkeypatch.setattr(scores.tuned_block_sizes, "SCORE_WINDOW", 16)
    rng = np.random.default_rng(71)
    keys = jnp.asarray(rng.integers(-8, 9, (128, 128)), jnp.bfloat16)
    native = native_pool(keys)
    q = jnp.asarray(rng.integers(-2, 3, (2, 6, 4, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.integers(-2, 3, (2, 6, 4)), jnp.float32)
    pages = jnp.array([[112, 0, -1, 64], [48, 80, 48, 16]], jnp.int32)
    ids = pages[..., None] + jnp.arange(16, dtype=jnp.int32)
    logical = keys[jnp.maximum(ids, 0)].reshape(2, 64, 128)
    records = jnp.arange(64, dtype=jnp.int32)[None]
    records = records.at[0, jnp.array([7, 9, 31, 32, 63])].set(-1)
    for function, arguments, kwargs in (
        (score_word_rows, (q[0], weights[0, ..., None]), {}),
        (
            scores.prefill_index_scores,
            (q[0], weights[0]),
            dict(
                pages=pages[:1],
                row_limit=jnp.array([64], jnp.int32),
                page_records=16,
                query_tile=3,
                key_tile=32,
                prefetch=prefetch,
                records=records if record_mask else None,
            ),
        ),
    ):
        actual = function(
            *arguments,
            native[None] if function is score_word_rows else native,
            interpret=True,
            **kwargs,
        )
        if function is score_word_rows:
            expected = dense_scores(q[0], weights[0], keys)
        else:
            expected = dense_scores(q[0], weights[0], logical[0])
            # Prefill leaves page and per-query causal masks to its caller.
            page_valid = jnp.repeat(pages[:1] >= 0, 16, axis=1)
            actual = jnp.where(page_valid, actual, -jnp.inf)
            valid = page_valid
            if record_mask:
                valid &= records >= 0
            expected = jnp.where(valid, expected, -jnp.inf)
        np.testing.assert_array_equal(actual, expected)


def test_decode_native_bitcast_on_eight_devices():
    """Exercise native bitcast under the model's manual eight-device partition."""
    if jax.default_backend() != "tpu":
        pytest.skip("native TPU lowering")
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    if len(jax.devices()) < 8:
        pytest.skip("Eight TPU devices required for model-mesh lowering")
    rng = np.random.default_rng(73)
    keys = jnp.asarray(rng.integers(-8, 9, (256, 128)), jnp.bfloat16)
    native = native_pool(keys)
    q = jnp.asarray(rng.integers(-2, 3, (8, 6, 32, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.integers(-2, 3, (8, 6, 32)), jnp.float32)
    pages = jnp.broadcast_to(jnp.array([128, 0], jnp.int32), (8, 2))
    lengths = jnp.broadcast_to(jnp.array([256, 255, 128, 17, 1, 0], jnp.int32), (8, 6))
    mesh = Mesh(np.asarray(jax.devices()[:8]), ("d",))

    def run(q, weights, pool, pages, lengths):
        return persistent_scores.grouped_decode_scores(
            q,
            weights,
            pool,
            pages,
            lengths,
            page_records=128,
        )

    mapped = jax.jit(
        jax.shard_map(
            run,
            mesh=mesh,
            in_specs=(P("d"), P("d"), (P(), P()), P("d"), P("d")),
            out_specs=P("d"),
            check_vma=False,
        )
    )
    actual = mapped(q, weights, (native[:64], native[64:]), pages, lengths)
    ids = pages[..., None] + jnp.arange(128, dtype=jnp.int32)
    logical = keys[ids].reshape(8, 256, 128)
    expected = dense_scores(q, weights, logical)
    expected = jnp.where(
        jnp.arange(256)[None, None] < lengths[..., None], expected, -jnp.inf
    )
    np.testing.assert_array_equal(actual, expected.reshape(-1, expected.shape[-1]))

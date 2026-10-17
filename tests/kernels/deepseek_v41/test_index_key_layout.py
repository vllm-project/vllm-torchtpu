# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Halfplane keys retain owner geometry and the ordinary decoder's values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout as layout
from vllm_torchtpu.kernels.deepseek_v41.index_selection import candidate, decode, scores
from vllm_torchtpu.kernels.deepseek_v41.index_word_scores import score_word_rows


def pools(keys):
    ordinary = jax.lax.bitcast_convert_type(keys.reshape(-1, 64, 2), jnp.int32).reshape(
        -1, 128
    )
    bits = layout.encode_records(keys)
    native = (bits[::2] | (bits[1::2] << 16)).astype(jnp.int32)
    return ordinary, native


@pytest.mark.cpu_test
def test_all_bf16_patterns_gather_and_inverse():
    bits = jnp.arange(65536, dtype=jnp.uint16).reshape(-1, 128)
    keys = jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
    _, native = pools(keys)
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
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_all_score_routes_match_ordinary(fused, prefetch):
    rng = np.random.default_rng(71)
    keys = jnp.asarray(rng.integers(-8, 9, (128, 128)), jnp.bfloat16)
    ordinary, native = pools(keys)
    q = jnp.asarray(rng.integers(-2, 3, (2, 6, 4, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.integers(-2, 3, (2, 6, 4, 1)), jnp.float32)
    pages = jnp.array([[112, 0, -1, 64], [48, 80, 48, 16]], jnp.int32)
    lengths = jnp.array([[0, 1, 15, 16, 17, 64], [63, 62, 48, 33, 32, 31]], jnp.int32)
    options = dict(
        page_records=16,
        key_tile=32,
        interpret=True,
        prefetch=prefetch,
        fuse_mask=fused,
        score_window=32,
    )
    reference = decode.grouped_decode_scores(
        q, weights, (ordinary[:32], ordinary[32:]), pages, lengths, **options
    )
    actual = decode.grouped_decode_scores(
        q,
        weights,
        (native[:32], native[32:]),
        pages,
        lengths,
        halfplane=True,
        **options,
    )
    np.testing.assert_array_equal(actual.view(jnp.uint32), reference.view(jnp.uint32))
    for function, arguments, kwargs in (
        (score_word_rows, (q[0], weights[0]), {}),
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
            ),
        ),
    ):
        left = function(
            *arguments,
            ordinary[None] if function is score_word_rows else ordinary,
            interpret=True,
            **kwargs,
        )
        right = function(
            *arguments,
            native[None] if function is score_word_rows else native,
            interpret=True,
            halfplane=True,
            **kwargs,
        )
        np.testing.assert_array_equal(left.view(jnp.uint32), right.view(jnp.uint32))
    unpacked = candidate.unpack_candidate_words(
        native[None], interpret=True, halfplane=True
    )
    baseline = candidate.unpack_candidate_words(ordinary[None], interpret=True)
    np.testing.assert_array_equal(unpacked.view(jnp.uint16), baseline.view(jnp.uint16))
    arguments = (
        q[:1],
        weights[:1, :, :, 0],
        jnp.arange(16, dtype=jnp.int32)[None],
        jnp.full((1, 16), 63, jnp.int32),
        lengths[:1],
    )
    left = candidate.candidate_scores(
        arguments[0],
        arguments[1],
        ordinary[None],
        *arguments[2:],
        block_size=8,
        key_tile=32,
        interpret=True,
    )
    right = candidate.candidate_scores(
        arguments[0],
        arguments[1],
        native[None],
        *arguments[2:],
        block_size=8,
        key_tile=32,
        interpret=True,
        halfplane=True,
    )
    np.testing.assert_array_equal(left.view(jnp.uint32), right.view(jnp.uint32))


def test_halfplane_decode_native_bitcast_on_eight_devices():
    """Exercise native bitcast under the model's manual eight-device partition."""
    if jax.default_backend() != "tpu":
        pytest.skip("native TPU lowering")
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    if len(jax.devices()) < 8:
        pytest.skip("Eight TPU devices required for model-mesh lowering")
    rng = np.random.default_rng(73)
    ordinary, native = pools(jnp.asarray(rng.integers(-8, 9, (64, 128)), jnp.bfloat16))
    q = jnp.asarray(rng.integers(-2, 3, (8, 6, 32, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.integers(-2, 3, (8, 6, 32, 1)), jnp.float32)
    pages = jnp.broadcast_to(jnp.array([48, 0, 32, 16], jnp.int32), (8, 4))
    lengths = jnp.broadcast_to(jnp.array([64, 63, 32, 17, 1, 0], jnp.int32), (8, 6))
    mesh = Mesh(np.asarray(jax.devices()[:8]), ("d",))
    results = []
    for halfplane, pool in ((False, ordinary), (True, native)):

        def run(q, weights, pool, pages, lengths, halfplane=halfplane):
            return decode.grouped_decode_scores(
                q,
                weights,
                pool,
                pages,
                lengths,
                page_records=16,
                key_tile=32,
                prefetch=True,
                fuse_mask=True,
                halfplane=halfplane,
                interpret=False,
            )

        mapped = jax.jit(
            jax.shard_map(
                run,
                mesh=mesh,
                in_specs=(P("d"), P("d"), P(), P("d"), P("d")),
                out_specs=P("d"),
                check_vma=False,
            )
        )
        results.append(np.asarray(mapped(q, weights, pool, pages, lengths)))
    np.testing.assert_array_equal(
        results[0].view(np.uint32), results[1].view(np.uint32)
    )

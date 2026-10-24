# SPDX-License-Identifier: Apache-2.0
"""Chunk boundaries preserve ragged requests, SWA visibility and cached KV."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.csa2 import pipeline as pipeline_module
from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import (
    encode_csa2_kv,
    prepare_swa_metadata,
)
from vllm_torchtpu.kernels.deepseek_v41.csa2.pipeline import csa2_pipeline
from vllm_torchtpu.kernels.deepseek_v41.csa2.prefill import csa2_prefill
from vllm_torchtpu.kernels.mla.paged_swa import write_swa_pages


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "starts,non_causal,banked,discontinuous,decode",
    [
        ([0, 6, 12], False, False, False, True),
        ([0, 6, 12, 12], False, False, False, True),
        ([0, 3, 12], False, False, False, False),
        ([0, 6, 11], False, False, False, False),
        ([0, 0, 12], False, False, False, False),
        ([0, 6, 12], True, False, False, False),
        ([0, 6, 12], False, True, False, False),
        ([0, 6, 12], False, False, True, False),
    ],
)
def test_six_query_dispatch_preserves_request_contract(
    monkeypatch, starts, non_causal, banked, discontinuous, decode
):
    """Shapes are identical for regular, ragged, padded and noncausal batches."""

    def decoded(
        q,
        words,
        scales,
        swa,
        positions,
        sink,
        rope,
        *,
        main_lengths,
        swa_page,
        num_tokens,
        scale,
    ):
        assert num_tokens == 6 and swa_page.shape[0] == 2
        # Include counts and positions in the output to check adapter arguments.
        return jnp.broadcast_to(
            (jnp.repeat(positions, 6) + main_lengths)[:, None, None], q.shape
        ).astype(q.dtype)

    monkeypatch.setattr(pipeline_module, "csa2_decode", decoded)
    monkeypatch.setattr(
        pipeline_module, "csa2_prefill", lambda q, *a, **k: -jnp.ones_like(q)
    )
    positions = jnp.concatenate((jnp.arange(127, 133), jnp.arange(1024, 1030)))
    if discontinuous:
        positions = positions.at[4].add(1)
    counts = jnp.arange(12, dtype=jnp.int32)
    swa = jnp.zeros((2, 64, 512), jnp.bfloat16)
    result = jax.jit(
        lambda boundaries, pos: pipeline_module.csa2_attention(
            jnp.zeros((12, 8, 512), jnp.bfloat16),
            jnp.zeros((12, 512, 128), jnp.int32),
            jnp.ones((12, 512), jnp.float32),
            (swa, swa) if banked else swa,
            boundaries,
            pos,
            jnp.zeros(8, jnp.float32),
            jnp.zeros((12, 2, 128), jnp.float32),
            swa_metadata=jnp.zeros((len(starts) - 1, 6), jnp.int32),
            swa_bounds=jnp.zeros((len(starts) - 1, 2), jnp.int32)
            if non_causal
            else None,
            main_lengths=counts,
            scale=512**-0.5,
            compress_ratio=2,
            bq=6,
        )
    )(jnp.array(starts, jnp.int32), positions)
    expected = (
        (jnp.repeat(positions[::6], 6) + counts).astype(jnp.bfloat16)
        if decode
        else -jnp.ones(12, jnp.bfloat16)
    )
    np.testing.assert_array_equal(result[:, 0, 0], expected)


@pytest.mark.parametrize("page_size", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("banked", [False, True])
def test_native_swa_publication_preserves_neighbor_tokens(page_size, banked):
    # Unaligned writes on both sides of page/bank boundaries must not clobber
    # the other half of a packed BF16 word. Invalid records cannot touch page 0.
    initial = np.full((4, page_size, 512), -3, dtype=np.float32)
    records = np.array(
        [
            -1,
            0,
            page_size - 1,
            page_size,
            page_size + 1,
            2 * page_size - 1,
            2 * page_size,
            3 * page_size + 1,
            4 * page_size,
        ],
        dtype=np.int32,
    )
    values = np.arange(len(records) * 512, dtype=np.float32).reshape(-1, 512) % 97
    expected = initial.copy().reshape(-1, 512)
    for record, value in zip(records, values):
        if page_size <= record < 4 * page_size:
            expected[record] = value
    pages = jnp.asarray(initial, jnp.bfloat16)
    owner = (pages[:2], pages[2:]) if banked else pages
    updated = write_swa_pages(
        owner, jnp.asarray(records), jnp.asarray(values, jnp.bfloat16)
    )
    actual = jnp.concatenate(updated) if banked else updated
    np.testing.assert_array_equal(actual.reshape(-1, 512), expected)


@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("banked", [False, True])
def test_native_swa_publication_contiguous_padding_and_fragmentation(
    fragmented, banked
):
    page_size = 16
    initial = np.full((8, page_size, 512), -3, dtype=np.float32)
    records = np.arange(17, 36, dtype=np.int32)
    if fragmented:
        records[8:] += 64
        records = np.random.default_rng(18).permutation(records)
    records = np.concatenate((records, np.full(5, -1, dtype=np.int32)))
    values = np.arange(records.size * 512, dtype=np.float32).reshape(-1, 512) % 97
    expected = initial.reshape(-1, 512).copy()
    for record, value in zip(records, values):
        if record >= page_size:
            expected[record] = value
    pages = jnp.asarray(initial, jnp.bfloat16)
    owner = (pages[:4], pages[4:]) if banked else pages
    updated = write_swa_pages(
        owner, jnp.asarray(records), jnp.asarray(values, jnp.bfloat16)
    )
    actual = jnp.concatenate(updated) if banked else updated
    np.testing.assert_array_equal(actual.reshape(-1, 512), expected)


@pytest.mark.parametrize(
    "dtype,shape", [(jnp.int32, (4, 2, 32, 128)), (jnp.bfloat16, (4, 32, 256))]
)
def test_native_swa_publication_rejects_non_native_owners(dtype, shape):
    with pytest.raises(ValueError, match="BF16"):
        write_swa_pages(
            jnp.zeros(shape, dtype),
            jnp.array([32], jnp.int32),
            jnp.zeros((1, 512), jnp.bfloat16),
        )


def test_native_swa_publication_preserves_bf16_bits():
    bits = jnp.resize(
        jnp.array([0, 0x8000, 1, 0x8001, 0x3F80, 0x7F80, 0xFF80, 0x7FC1], jnp.uint16),
        (3, 512),
    )
    values = jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
    # TPU input conversion canonicalizes NaNs and flushes subnormals before
    # publication. Compare the device input bits, not the host seed pattern.
    bits = jax.lax.bitcast_convert_type(values, jnp.uint16)
    owner = jnp.zeros((4, 16, 512), jnp.bfloat16)
    records = jnp.array([17, 31, 32], jnp.int32)
    updated = write_swa_pages(owner, records, values)
    actual = jax.lax.bitcast_convert_type(updated, jnp.uint16).reshape(-1, 512)
    np.testing.assert_array_equal(actual[records], bits)
    np.testing.assert_array_equal(actual[jnp.array([16, 18, 30, 33])], 0)


def test_native_swa_shared_owner_avoids_pool_sized_temporaries():
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU buffer assignment")

    def read_page(pages_ref, output_ref):
        output_ref[...] = pages_ref[0, :, :]

    read = pl.pallas_call(
        read_page,
        out_shape=jax.ShapeDtypeStruct((32, 512), jnp.bfloat16),
        in_specs=[pl.BlockSpec((1, 32, 512), lambda i: (1, 0, 0))],
        grid=(1,),
    )

    def update_and_read(pages, records, values):
        outputs = []
        for layer in range(3):
            pages = write_swa_pages(pages, records, values + layer)
            outputs.append(read(pages))
        return pages, jnp.stack(outputs)

    pages = jnp.zeros((1024, 32, 512), jnp.bfloat16)
    pool_bytes = pages.nbytes
    records = jnp.arange(288, dtype=jnp.int32) * 2 + 32
    values = jnp.ones((288, 512), jnp.bfloat16)
    compiled = (
        jax.jit(update_and_read, donate_argnums=(0,))
        .lower(pages, records, values)
        .compile()
    )
    _, outputs = compiled(pages, records, values)
    expected = np.zeros((3, 32, 512), dtype=np.float32)
    expected[:, ::2] = np.arange(1, 4)[:, None, None]
    np.testing.assert_array_equal(outputs, expected)
    # An attention read followed by another write must reuse the shared owner.
    assert compiled.memory_analysis().temp_size_in_bytes < pool_bytes


@pytest.fixture
def sparsecore_device():
    if jax.default_backend() != "tpu":
        pytest.skip("CSA2 pipeline requires a TPU with SparseCore")
    if pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("CSA2 pipeline requires SparseCore")


@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.parametrize("non_causal", [False, True])
def test_chunked_attention_preserves_requests_and_reusable_kv(
    sparsecore_device, banked, non_causal
):
    # Three requests straddle chunk boundaries, with empty requests interleaved
    # and a final chunk containing only padding. Noncausal windows include KV
    # beyond the end of the current chunk.
    tokens, heads, chunk_size, page_size = 193, 8, 64, 32
    rng = np.random.default_rng(31)

    def random(shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.2, jnp.bfloat16)

    q = random((tokens, heads, 512))
    words, scales = encode_csa2_kv(random((257, 512)))
    starts = jnp.array([0, 0, 43, 43, 103, 155, 155], jnp.int32)
    positions = jnp.array(
        [*range(160, 203), *range(240, 300), *range(330, 382)] + [0] * 38,
        jnp.int32,
    )
    counts = jnp.array(rng.integers(0, 513, tokens), jnp.int32)
    counts = jnp.where(jnp.arange(tokens) < 155, counts, 0)
    records = jnp.array(rng.integers(0, 257, (tokens, 512)), jnp.int32)
    records = jnp.where(jnp.arange(512) < counts[:, None], records, -1)
    selected = jnp.maximum(records, 0)
    expected_words = jnp.where(records[:, :, None] >= 0, words[selected], 0)
    expected_scales = jnp.where(records >= 0, scales[selected], 0)
    table = jnp.arange(1, 97, dtype=jnp.int32).reshape(6, 16)
    metadata, bounds, _ = prepare_swa_metadata(
        table,
        starts,
        positions,
        jnp.array([0, 150, 0, 200, 300, 0], jnp.int32),
        page_size=page_size,
        page_stride=page_size,
        row_offset=0,
        non_causal=non_causal,
    )
    swa = random((97, page_size, 512)).at[0].set(0)
    word_banks = (words[:113], words[113:]) if banked else words
    scale_banks = (scales[:113], scales[113:]) if banked else scales
    swa_banks = (swa[:48], swa[48:]) if banked else swa
    angles = jnp.asarray(rng.standard_normal((tokens, 64)) * 0.1)
    rope = jnp.stack(
        (
            jnp.concatenate((jnp.ones_like(angles), jnp.cos(angles)), axis=1),
            jnp.concatenate((jnp.zeros_like(angles), jnp.sin(angles)), axis=1),
        ),
        axis=1,
    )
    sink = jnp.linspace(-1, 1, heads, dtype=jnp.float32)
    kwargs = dict(scale=512**-0.5, compress_ratio=2, bq=6)

    def attend(query, selected_words, selected_scales, offset=0):
        end = offset + query.shape[0]
        return csa2_prefill(
            query,
            selected_words,
            selected_scales,
            swa_banks,
            jnp.clip(starts - offset, 0, query.shape[0]),
            positions[offset:end],
            sink,
            rope[offset:end],
            swa_metadata=metadata,
            swa_bounds=bounds if non_causal else None,
            main_lengths=counts[offset:end],
            **kwargs,
        )

    expected = attend(q, expected_words, expected_scales)
    output, gathered_words, gathered_scales = csa2_pipeline(
        q,
        word_banks,
        scale_banks,
        records,
        swa_banks,
        starts,
        positions,
        sink,
        rope,
        metadata,
        bounds if non_causal else None,
        counts,
        chunk_size=chunk_size,
        **kwargs,
    )
    np.testing.assert_allclose(output, expected, atol=2e-3, rtol=2e-2)
    np.testing.assert_array_equal(output[155:], 0)
    np.testing.assert_array_equal(gathered_words, expected_words)
    np.testing.assert_array_equal(gathered_scales, expected_scales)

    next_q = q * jnp.bfloat16(0.7)
    reused = attend(next_q, gathered_words, gathered_scales)
    np.testing.assert_allclose(
        reused, attend(next_q, expected_words, expected_scales), atol=2e-3, rtol=2e-2
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "decoder,non_causal,starts,banked,use_decode",
    [
        (False, False, [0, 16, 32], False, False),
        (False, True, [0, 16, 32], False, False),
        (True, False, [0, 16, 32], False, True),
        (True, True, [0, 16, 32], False, False),
        (True, False, [0, 8, 32], False, False),
        (True, False, [0, 16, 31], False, False),
        (True, False, [0, 16, 32], True, False),
    ],
)
def test_decoder_layer_prefill_dispatch(
    monkeypatch, decoder, non_causal, starts, banked, use_decode
):
    def decoded(q, words, scales, swa, positions, sink, rope, **kwargs):
        assert kwargs["num_tokens"] == 16
        assert kwargs["decoder_prefill"] is True
        return jnp.broadcast_to(
            (jnp.repeat(positions, 16) + kwargs["main_lengths"])[:, None, None],
            q.shape,
        ).astype(q.dtype)

    monkeypatch.setattr(pipeline_module, "csa2_decode", decoded)
    monkeypatch.setattr(
        pipeline_module, "csa2_prefill", lambda q, *args, **kwargs: -jnp.ones_like(q)
    )
    positions = jnp.concatenate((jnp.arange(128, 144), jnp.arange(512, 528)))
    counts = jnp.arange(32, dtype=jnp.int32)
    swa = jnp.zeros((2, 64, 512), jnp.bfloat16)
    result = jax.jit(
        lambda boundaries, pos: pipeline_module.csa2_attention(
            jnp.zeros((32, 8, 512), jnp.bfloat16),
            jnp.zeros((32, 512, 128), jnp.int32),
            jnp.ones((32, 512), jnp.float32),
            (swa, swa) if banked else swa,
            boundaries,
            pos,
            jnp.zeros(8, jnp.float32),
            jnp.zeros((32, 2, 128), jnp.float32),
            swa_metadata=jnp.zeros((2, 6), jnp.int32),
            swa_bounds=jnp.zeros((2, 2), jnp.int32) if non_causal else None,
            main_lengths=counts,
            scale=512**-0.5,
            compress_ratio=2,
            bq=16,
            decoder=decoder,
        )
    )(jnp.array(starts, jnp.int32), positions)
    expected = (
        (jnp.repeat(positions[::16], 16) + counts).astype(jnp.bfloat16)
        if use_decode
        else -jnp.ones(32, jnp.bfloat16)
    )
    np.testing.assert_array_equal(result[:, 0, 0], expected)

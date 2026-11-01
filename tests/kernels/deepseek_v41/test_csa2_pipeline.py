# SPDX-License-Identifier: Apache-2.0
"""Chunk boundaries preserve ragged requests, SWA visibility and cached KV."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import (
    encode_csa2_kv,
    prepare_swa_metadata,
)
from vllm_torchtpu.kernels.deepseek_v41.csa2.csa2 import csa2
from vllm_torchtpu.kernels.deepseek_v41.csa2.pipeline import csa2_pipeline
from vllm_torchtpu.kernels.mla.paged_swa import (
    plan_swa_pages,
    write_planned_swa_pages,
    write_swa_pages,
)


@pytest.mark.parametrize("page_size", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="SparseCore publication")
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
        owner,
        jnp.asarray(records),
        jnp.asarray(values, jnp.bfloat16),
    )
    actual = jnp.concatenate(updated) if banked else updated
    np.testing.assert_array_equal(actual.reshape(-1, 512), expected)


@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="SparseCore publication")
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
        owner,
        jnp.asarray(records),
        jnp.asarray(values, jnp.bfloat16),
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


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="SparseCore publication")
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


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="SparseCore publication")
def test_native_swa_publication_all_invalid_preserves_owner():
    owner = jnp.full((4, 32, 512), -3, jnp.bfloat16)
    result = write_swa_pages(
        owner,
        jnp.array([-1, 0, 128, 9999], jnp.int32),
        jnp.ones((4, 512), jnp.bfloat16),
    )
    np.testing.assert_array_equal(result, -3)


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


@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="SparseCore publication")
def test_planned_swa_publication_matches_direct_writes(page_size, banked):
    # One plan serves every layer of a stage: three layers publish different
    # values at the same fragmented records, unaligned on both pair halves.
    rng = np.random.default_rng(page_size)
    initial = rng.standard_normal((8, page_size, 512)).astype(np.float32)
    records = rng.permutation(np.arange(page_size, 8 * page_size))[:45]
    records = np.concatenate((records, [-1, 0, 8 * page_size])).astype(np.int32)

    def owner():
        # Separate owners: both writers donate theirs.
        pages = jnp.asarray(initial, jnp.bfloat16)
        return (pages[:3], pages[3:]) if banked else pages

    direct, planned = owner(), owner()
    banks = direct if banked else (direct,)
    plan = plan_swa_pages(
        jnp.asarray(records),
        page_size=page_size,
        bank_pages=tuple(bank.shape[0] for bank in banks),
    )
    for layer in range(3):
        values = jnp.asarray(
            rng.standard_normal((records.size, 512)) * (layer + 1), jnp.bfloat16
        )
        direct = write_swa_pages(direct, jnp.asarray(records), values)
        planned = write_planned_swa_pages(planned, plan, values)
    join = jnp.concatenate if banked else (lambda x: x)
    np.testing.assert_array_equal(
        jax.lax.bitcast_convert_type(join(planned), jnp.uint16),
        jax.lax.bitcast_convert_type(join(direct), jnp.uint16),
    )


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="SparseCore publication")
def test_planned_swa_publication_rejects_foreign_plans():
    records = jnp.arange(32, 64, dtype=jnp.int32)
    plan = plan_swa_pages(records, page_size=32, bank_pages=(4,))
    with pytest.raises(ValueError, match="plan"):
        write_planned_swa_pages(
            jnp.zeros((4, 32, 512), jnp.bfloat16),
            plan,
            jnp.zeros((16, 512), jnp.bfloat16),
        )
    with pytest.raises(ValueError, match="plan"):
        write_planned_swa_pages(
            (
                jnp.zeros((2, 32, 512), jnp.bfloat16),
                jnp.zeros((2, 32, 512), jnp.bfloat16),
            ),
            plan,
            jnp.zeros((32, 512), jnp.bfloat16),
        )


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
        return csa2(
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

    # The selector may return any order within the valid prefix. Compare the
    # unsorted gather pipeline with attention over the same sorted KV records.
    order = jnp.argsort(jnp.where(records >= 0, records, jnp.iinfo(jnp.int32).max))
    expected = attend(
        q,
        jnp.take_along_axis(expected_words, order[:, :, None], axis=1),
        jnp.take_along_axis(expected_scales, order, axis=1),
    )
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


def _offset_inputs(ragged):
    tokens, heads, page_size = 40, 8, 32
    rng = np.random.default_rng(7)

    def random(shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.2, jnp.bfloat16)

    if ragged:
        starts = jnp.array([0, 15, 15, 40], jnp.int32)
        positions = jnp.array([*range(100, 115), *range(300, 325)], jnp.int32)
    else:
        starts = jnp.array([0, 40], jnp.int32)
        positions = jnp.arange(200, 240, dtype=jnp.int32)
    batch = starts.shape[0] - 1
    words, scales = encode_csa2_kv(random((97, 512)))
    counts = jnp.array(rng.integers(0, 513, tokens), jnp.int32)
    records = jnp.array(rng.integers(0, 97, (tokens, 512)), jnp.int32)
    records = jnp.where(jnp.arange(512) < counts[:, None], records, 0)
    table = jnp.arange(1, 1 + batch * 16, dtype=jnp.int32).reshape(batch, 16)
    metadata, _, _ = prepare_swa_metadata(
        table,
        starts,
        positions,
        jnp.zeros(batch, jnp.int32),
        page_size=page_size,
        page_stride=page_size,
        row_offset=0,
    )
    angles = jnp.asarray(rng.standard_normal((tokens, 64)) * 0.1)
    rope = jnp.stack(
        (
            jnp.concatenate((jnp.ones_like(angles), jnp.cos(angles)), axis=1),
            jnp.concatenate((jnp.zeros_like(angles), jnp.sin(angles)), axis=1),
        ),
        axis=1,
    )
    return dict(
        main_kv=words[records],
        main_kv_scale=scales[records],
        swa_kv=random((1 + batch * 16, page_size, 512)).at[0].set(0),
        starts=starts,
        positions=positions,
        sink=jnp.linspace(-1, 1, heads, dtype=jnp.float32),
        rope_cache=rope,
        swa_metadata=metadata,
        main_lengths=counts,
    ), random


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="CSA2 attention")
def test_offset_attention_updates_only_its_owner_rows(ragged):
    # A chunk call writes attention and staged KV to rows [offset, offset + T)
    # of the aliased owners; every other owner row keeps its contents.
    inputs, random = _offset_inputs(ragged)
    tokens, offset = inputs["positions"].shape[0], 24
    q = random((96, 8, 512))
    rows = np.arange(offset, offset + tokens)
    expected = csa2(q[rows], **inputs, bq=8)
    out = random(q.shape)
    kv_out = jnp.asarray(
        np.random.default_rng(3).integers(-(2**31), 2**31, (96, 512, 128)), jnp.int32
    )
    got, got_kv = csa2(q, **inputs, bq=8, query_offset=offset, out=out, kv_out=kv_out)
    others = np.setdiff1d(np.arange(96), rows)
    np.testing.assert_array_equal(got[rows], expected)
    np.testing.assert_array_equal(got[others], out[others])
    np.testing.assert_array_equal(got_kv[rows], inputs["main_kv"])
    np.testing.assert_array_equal(got_kv[others], kv_out[others])


def test_offset_attention_requires_matching_owners():
    inputs, random = _offset_inputs(False)
    q = random((96, 8, 512))
    with pytest.raises(ValueError, match="require out and kv_out"):
        csa2(q, **inputs, query_offset=24)
    with pytest.raises(ValueError, match="together"):
        csa2(q, **inputs, query_offset=24, out=q)
    with pytest.raises(ValueError, match="excludes"):
        csa2(
            q,
            **inputs,
            query_offset=24,
            out=q,
            kv_out=jnp.zeros((96, 512, 128), jnp.int32),
            new_owners=True,
        )
    with pytest.raises(ValueError, match="within q"):
        csa2(
            q,
            **inputs,
            query_offset=60,
            out=q,
            kv_out=jnp.zeros((96, 512, 128), jnp.int32),
        )


def test_new_owner_attention_writes_offset_rows():
    # The first chunk allocates both owners; only its rows are defined.
    if jax.default_backend() != "tpu":
        pytest.skip("CSA2 attention")
    inputs, random = _offset_inputs(True)
    tokens, offset = inputs["positions"].shape[0], 24
    q = random((96, 8, 512))
    rows = np.arange(offset, offset + tokens)
    expected = csa2(q[rows], **inputs, bq=8)
    got, got_kv = csa2(q, **inputs, bq=8, query_offset=offset, new_owners=True)
    assert got.shape == q.shape and got_kv.shape == (96, 512, 128)
    np.testing.assert_array_equal(got[rows], expected)
    np.testing.assert_array_equal(got_kv[rows], inputs["main_kv"])


def test_chunked_pipeline_writes_owners_without_joins(sparsecore_device):
    # Chunked attention writes its output and gathered KV in place: no
    # whole-T update, concatenation or copy of either owner remains, also
    # when several layers' pipelines share one program.
    tokens, heads = 256, 8
    words, scales = encode_csa2_kv(jnp.zeros((257, 512), jnp.bfloat16))
    table = jnp.arange(1, 17, dtype=jnp.int32).reshape(1, 16)
    starts = jnp.array([0, tokens], jnp.int32)
    positions = jnp.arange(tokens, dtype=jnp.int32)
    metadata, _, _ = prepare_swa_metadata(
        table,
        starts,
        positions,
        jnp.zeros(1, jnp.int32),
        page_size=32,
        page_stride=32,
        row_offset=0,
    )

    def layer(q, records):
        return csa2_pipeline(
            q,
            words,
            scales,
            records,
            jnp.zeros((17, 32, 512), jnp.bfloat16),
            starts,
            positions,
            jnp.zeros(heads, jnp.float32),
            jnp.zeros((tokens, 2, 128), jnp.float32),
            metadata,
            None,
            jnp.full(tokens, 512, jnp.int32),
            chunk_size=64,
            bq=8,
            scale=512**-0.5,
            compress_ratio=2,
        )

    def two_layers(q, records):
        # Identical uninitialized owners would be merged across layers and
        # one layer's owners copied, so each layer must allocate its own.
        first = layer(q, records)
        second = layer(first[0], records)
        return first, second

    text = (
        jax.jit(two_layers)
        .lower(
            jax.ShapeDtypeStruct((tokens, heads, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, 512), jnp.int32),
        )
        .compile()
        .as_text()
    )
    owners = rf"(s32\[{tokens},512,128\]|bf16\[{tokens},{heads},512\])"
    joins = re.findall(
        rf"= {owners}\S* (fusion|concatenate|dynamic-update-slice|copy)\(", text
    )
    assert not joins, joins

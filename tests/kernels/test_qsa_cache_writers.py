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
"""Compressed pair DMA layout and native pool/state byte ownership."""

import functools
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_state
from vllm_torchtpu.kernels.qwen4_exp.tuned_block_sizes import _COMPILER_SMEM_RESERVE


def compressed_fixture(fp8, edges):
    rng = np.random.default_rng(22912443)
    packing = 4 if fp8 else 2
    raw = rng.integers(0, 120, (7, 128, 1, packing, 128), np.uint8)
    if fp8:
        pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    else:
        pool = jnp.asarray(rng.normal(0, 0.2, raw.shape), jnp.bfloat16)
    keys = jnp.asarray(rng.normal(0, 0.2, (256, 128)), jnp.bfloat16)
    counts, context = ([0, 130, 0, 123, 0], [0, 2, 0, 9, 0]) if edges else ([256], [0])
    starts = np.r_[0, np.cumsum(counts)].astype(np.int32)
    request = np.minimum(
        np.searchsorted(starts[1:], np.arange(256), side="right"), len(counts) - 1
    )
    logical = np.asarray(context)[request] + np.arange(256) - starts[request]
    valid = np.arange(256) < starts[-1]
    complete = valid & ((logical + 1) % 4 == 0)
    table = np.zeros((len(counts), 3), np.int32)
    if edges:
        table[1] = [1, 2, 3]
        table[3] = [4, 5, 6]
    else:
        table[0] = [1, 2, 3]
    expected = np.asarray(pool).copy()
    words = np.asarray(keys).view(np.uint16)
    for row in range(256):
        if not complete[row]:
            continue
        r, position = int(request[row]), int(logical[row])
        page = table[r, position // 128]
        pair, half = (position % 128) // 8, (position // 4) % 2
        if fp8:
            bits = expected.view(np.uint8)
            bits[page, pair, 0, 2 * half] = (words[row] & 255).astype(np.uint8)
            bits[page, pair, 0, 2 * half + 1] = (words[row] >> 8).astype(np.uint8)
        else:
            expected.view(np.uint16)[page, pair, 0, half] = words[row]
    args = (
        pool,
        keys,
        jnp.asarray(complete),
        jnp.asarray(request, jnp.int32),
        jnp.asarray(logical, jnp.int32),
        jnp.asarray(starts),
        jnp.asarray(context, jnp.int32),
        jnp.asarray(table),
    )
    return args, expected


@pytest.mark.parametrize("tiles", [1, 2])
@pytest.mark.parametrize("edges", [False, True])
def test_compressed_prefetch_is_bounded_and_values_are_2d(monkeypatch, tiles, edges):
    from types import SimpleNamespace

    word_bytes = 4
    words = 128 * (1 + 1 + 2) + 2
    capacity = _COMPILER_SMEM_RESERVE + tiles * words * word_bytes
    monkeypatch.setattr(
        pltpu, "get_tpu_info", lambda: SimpleNamespace(smem_capacity_bytes=capacity)
    )
    seen = []

    def capture(kernel, **kw):
        spec = kw["grid_spec"]
        assert spec.num_scalar_prefetch == 5
        assert spec.in_specs[0].block_shape == (256, 128)

        def run(first, bulk, pairs, active, masks, values, owner):
            assert values.shape == (2 * len(pairs), 128)
            metadata = sum(a.nbytes for a in (first, bulk, pairs, active, masks))
            assert metadata + _COMPILER_SMEM_RESERVE <= capacity
            seen.append(
                (
                    np.asarray(pairs),
                    np.asarray(active),
                    np.asarray(masks),
                    np.asarray(values),
                )
            )
            return owner

        return run

    monkeypatch.setattr(pl, "pallas_call", capture)
    args, _ = compressed_fixture(False, edges)
    with jax.disable_jit():
        pooled_qsa.write_compressed(*args, ratio=4)
    assert len(seen) == 2 // tiles
    # Independently recover every completed row from the captured pair writes.
    actual = {}
    context, starts, req, logical = map(
        np.asarray, (args[6], args[5], args[3], args[4])
    )
    for pairs, active, masks, flat in seen:
        values = flat.reshape(len(pairs), 2, 128)
        for row in np.flatnonzero(active):
            for half in range(2):
                if masks[row, half]:
                    actual[int(pairs[row]), half] = values[row, half]
    expected = {}
    keys, complete, table = map(np.asarray, (args[1], args[2], args[7]))
    for row in np.flatnonzero(complete):
        r, position = int(req[row]), int(logical[row])
        pair = int(table[r, position // 128]) * 128 + (position % 128) // 8
        expected[pair, (position // 4) % 2] = keys[row]
    assert actual.keys() == expected.keys()
    for location in actual:
        np.testing.assert_array_equal(actual[location], expected[location])


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("edges", [False, True])
def test_native_compressed_multigrid_preserves_every_pool_byte(
    fp8, edges, record_property
):
    args, expected = compressed_fixture(fp8, edges)
    before = [np.asarray(a).tobytes() for a in args]
    fn = jax.jit(functools.partial(pooled_qsa.write_compressed, ratio=4))
    compiled = fn.lower(*args).compile()
    memory = compiled.memory_analysis()
    assert memory is not None and memory.temp_size_in_bytes < 2 * 1024**3
    record_property(
        "memory",
        dict(
            arguments=memory.argument_size_in_bytes,
            outputs=memory.output_size_in_bytes,
            temporary=memory.temp_size_in_bytes,
            alias=memory.alias_size_in_bytes,
        ),
    )
    record_property("input_sha256", [hashlib.sha256(x).hexdigest() for x in before])
    result = compiled(*args)
    jax.block_until_ready(result)
    assert np.asarray(result).tobytes() == expected.tobytes()
    repeated = compiled(*args)
    jax.block_until_ready(repeated)
    assert np.asarray(repeated).tobytes() == np.asarray(result).tobytes()
    assert [np.asarray(a).tobytes() for a in args] == before
    record_property(
        "output_sha256", hashlib.sha256(np.asarray(result).tobytes()).hexdigest()
    )
    record_property("input_cache_repeat_exact", True)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_native_ring_tags_roundtrip_and_null_page(fp8, record_property):
    packing = 4 if fp8 else 2
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    pool = jnp.zeros((17, 128, 1, packing, 128), dtype)
    ring = (
        jnp.arange(16 * 1024, dtype=jnp.float32).reshape(16, 1024).astype(jnp.bfloat16)
        / 32
    )
    tags = jnp.arange(16 * 16, dtype=jnp.int32).reshape(16, 16) + 65539
    ids = jnp.array(list(range(1, 15)) + [0, -1], jnp.int32)
    expected = np.asarray(pool).copy()
    ring_words = np.asarray(ring).view(np.uint16)
    tag_words = np.asarray(tags).view(np.uint32)
    for row, page in enumerate(np.asarray(ids)):
        if page <= 0:
            continue
        for element, word in enumerate(ring_words[row]):
            token, slot = divmod(element, 256)
            half, lane = divmod(slot, 128)
            if fp8:
                expected.view(np.uint8)[page, token, 0, 2 * half, lane] = word & 255
                expected.view(np.uint8)[page, token, 0, 2 * half + 1, lane] = word >> 8
            else:
                expected.view(np.uint16)[page, token, 0, half, lane] = word
        for lane in range(128):
            word = int(tag_words[row, lane]) if lane < 16 else 0
            if fp8:
                for byte in range(4):
                    expected.view(np.uint8)[page, 4, 0, byte, lane] = (
                        word >> (8 * byte)
                    ) & 255
            else:
                expected.view(np.uint16)[page, 4, 0, 0, lane] = word & 65535
                expected.view(np.uint16)[page, 4, 0, 1, lane] = word >> 16
    before = [np.asarray(x).tobytes() for x in (pool, ring, tags, ids)]
    fn = jax.jit(qsa_state.write_ring_and_tags)
    compiled = fn.lower(pool, ring, tags, ids).compile()
    memory = compiled.memory_analysis()
    assert memory is not None and memory.temp_size_in_bytes < 2 * 1024**3
    record_property(
        "memory",
        dict(
            arguments=memory.argument_size_in_bytes,
            outputs=memory.output_size_in_bytes,
            temporary=memory.temp_size_in_bytes,
            alias=memory.alias_size_in_bytes,
        ),
    )
    record_property("input_sha256", [hashlib.sha256(x).hexdigest() for x in before])
    result = compiled(pool, ring, tags, ids)
    jax.block_until_ready(result)
    assert np.asarray(result).tobytes() == expected.tobytes()
    read = (
        jax.jit(
            functools.partial(
                qsa_state.read_ring_and_tags, ring_elements=1024, tag_elements=16
            )
        )
        .lower(result, ids)
        .compile()
    )
    read_memory = read.memory_analysis()
    assert read_memory is not None and read_memory.temp_size_in_bytes < 2 * 1024**3
    record_property(
        "read_memory",
        dict(
            arguments=read_memory.argument_size_in_bytes,
            outputs=read_memory.output_size_in_bytes,
            temporary=read_memory.temp_size_in_bytes,
            alias=read_memory.alias_size_in_bytes,
        ),
    )
    actual_ring, actual_tags = read(result, ids)
    jax.block_until_ready((actual_ring, actual_tags))
    np.testing.assert_array_equal(actual_ring[:14], ring[:14])
    np.testing.assert_array_equal(actual_tags[:14], tags[:14])
    np.testing.assert_array_equal(actual_ring[14:], 0)
    np.testing.assert_array_equal(actual_tags[14:], 0)
    repeated = compiled(pool, ring, tags, ids)
    jax.block_until_ready(repeated)
    assert np.asarray(repeated).tobytes() == expected.tobytes()
    np.testing.assert_array_equal(pool, 0)
    assert [np.asarray(x).tobytes() for x in (pool, ring, tags, ids)] == before
    record_property(
        "output_sha256", hashlib.sha256(np.asarray(result).tobytes()).hexdigest()
    )
    record_property("input_cache_repeat_exact", True)

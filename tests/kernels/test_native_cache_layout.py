# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native record bit ordering, padding, index scores and owner aliasing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_word_scores import score_word_rows
from vllm_torchtpu.kernels.mla.cache_layout import (
    pack_cache_values,
    packed_record_shape,
    unpack_cache_values,
)
from vllm_torchtpu.kernels.mla.word_cache import scatter_word_records


@pytest.mark.parametrize(
    "dtype,dim",
    [
        (jnp.bfloat16, 64),
        (jnp.bfloat16, 128),
        (jnp.bfloat16, 512),
        (jnp.uint8, 512),
        (jnp.float32, 1024),
    ],
)
@pytest.mark.parametrize("rows", [19, 256])
def test_native_record_preserves_bits_and_feature_order(dtype, dim, rows):
    size = jnp.dtype(dtype).itemsize
    unsigned = {1: np.uint8, 2: np.uint16, 4: np.uint32}[size]
    rng = np.random.default_rng(dim)
    bits = rng.integers(0, 2 ** (size * 8), (rows, dim), dtype=unsigned)
    # Include zero, signed zero, subnormals, infinity and NaN payloads.
    if size == 2:
        bits[0, :8] = [0, 0x8000, 1, 0x8001, 0x7F80, 0xFF80, 0x7FC1, 0xFFFF]
    values = jax.device_put(bits.view(jnp.dtype(dtype)))
    if jax.default_backend() == "tpu" and dtype == jnp.bfloat16 and dim == 64:
        from jax.experimental import pallas as pl
        from jax.experimental.layout import Format, Layout

        # device_put chooses a transposed layout for [256, 64]. Relayouting
        # BF16 canonicalizes NaNs/subnormals even before the tested kernel.
        # Transfer integer payloads, then change only the native ref dtype.
        fmt = Format(
            Layout((0, 1), tiling=((8, 128), (2, 1))),
            jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        )
        raw = jax.device_put(bits, fmt)

        def view_bits(x_ref, y_ref):
            y_ref.bitcast(jnp.uint16)[...] = x_ref[...]

        view = pl.pallas_call(
            view_bits,
            out_shape=jax.ShapeDtypeStruct(bits.shape, dtype),
            grid=(pl.cdiv(rows, 16),),
            in_specs=[pl.BlockSpec((16, dim), lambda i: (i, 0))],
            out_specs=pl.BlockSpec((16, dim), lambda i: (i, 0)),
        )
        values = jax.jit(view, in_shardings=fmt, out_shardings=fmt)(raw)
        np.testing.assert_array_equal(np.asarray(values).view(np.uint16), bits)
    words = jax.jit(pack_cache_values, in_shardings=values.format)(values)
    packing, width = packed_record_shape(dim, size)
    padded = np.pad(bits, ((0, 0), (0, packing * width - dim)))
    parts = padded.reshape(rows, packing, width).astype(np.uint32)
    expected = np.zeros((rows, width), np.uint32)
    for part in range(packing):
        expected |= parts[:, part] << (8 * size * part)
    np.testing.assert_array_equal(np.asarray(words).view(np.uint32), expected)
    restored = jax.jit(
        lambda w: unpack_cache_values(w, dim, dtype), out_shardings=values.format
    )(words)
    np.testing.assert_array_equal(np.asarray(restored).view(unsigned), bits)


@pytest.mark.parametrize("groups,tokens", [(1, 6), (2, 4)])
def test_native_index_scores_match_original_features(groups, tokens):
    rng = np.random.default_rng(42)
    keys = jnp.asarray(rng.normal(size=(groups, 32, 128)), jnp.bfloat16)
    q = jnp.asarray(rng.normal(size=(tokens, 8, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(tokens, 8, 1)), jnp.float32)
    scores = score_word_rows(
        q, weights, pack_cache_values(keys), interpret=jax.default_backend() != "tpu"
    )
    repeated = jnp.repeat(keys, tokens // groups, axis=0)
    logits = jnp.einsum("thd,tkd->thk", q, repeated, preferred_element_type=jnp.float32)
    expected = (jnp.maximum(logits, 0) * weights).sum(1)
    np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize(
    "dim,dtype",
    [(64, jnp.bfloat16), (128, jnp.bfloat16), (512, jnp.bfloat16), (1024, jnp.float32)],
)
def test_native_write_aliases_large_owner(dim, dtype):
    if jax.default_backend() != "tpu":
        pytest.skip("TPU alias and layout analysis")
    from jax.experimental.layout import Format, Layout

    owner = jax.ShapeDtypeStruct((1 << 20, 128), jnp.int32)
    records = jax.ShapeDtypeStruct((65,), jnp.int32)
    values = jax.ShapeDtypeStruct((65, dim), dtype)
    fmt = Format(
        Layout(major_to_minor=(0, 1), tiling=((8, 128),)),
        jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )
    executable = (
        jax.jit(
            lambda pool, ids, x: scatter_word_records(pool, ids, pack_cache_values(x)),
            donate_argnums=(0,),
            in_shardings=(fmt, None, None),
            out_shardings=fmt,
        )
        .lower(owner, records, values)
        .compile()
    )
    memory = executable.memory_analysis()
    assert memory.alias_size_in_bytes == owner.size * 4
    assert memory.temp_size_in_bytes < owner.size

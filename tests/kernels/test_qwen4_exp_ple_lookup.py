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
"""Encoded resident lookup checked against independent CPU row indexing."""

import functools

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.ple import (
    compact_rows,
    decode_embeddings,
    resident_lookup,
    resident_lookup_raw,
)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("head_dim", [128, 160])
@pytest.mark.parametrize("tokens", [0, 1, 9])
def test_resident_lookup_bits_decode_and_owner_mask(fp8, head_dim, tokens):
    rng = np.random.default_rng(1966)
    rows, heads, start = 257, 16, 37
    dtype = ml_dtypes.float8_e4m3fn if fp8 else ml_dtypes.bfloat16
    integer = np.int8 if fp8 else np.int16
    # 160 is the checkpoint2560-dimensional embedding /16 n-gram heads.
    physical_rows = ((rows + 127) // 128) * 128
    values = rng.normal(size=(physical_rows, head_dim)).astype(dtype)
    encoded = values.view(np.uint8 if fp8 else integer)
    storage = np.pad(encoded.reshape(-1), (0, (-encoded.size) % 128)).reshape(-1, 128)
    choices = np.array(
        [
            -1,
            start - 1,
            start,
            start + 1,
            start + rows - 1,
            start + rows,
            start + rows + 7,
        ],
        np.int32,
    )
    ids = np.resize(choices, (tokens, heads))
    owned = (ids >= start) & (ids < start + rows)
    expected_raw = np.zeros((tokens, heads, head_dim), integer)
    expected_raw[owned] = encoded[ids[owned] - start].view(integer)
    scale = np.float32(0.375)
    numeric = expected_raw.view(dtype).astype(np.float32)
    expected = (
        (
            numeric.astype(ml_dtypes.bfloat16).astype(np.float32)
            * np.asarray(scale, ml_dtypes.bfloat16).astype(np.float32)
        ).astype(ml_dtypes.bfloat16)
        if fp8
        else numeric.astype(ml_dtypes.bfloat16)
    )
    table = jax.device_put(storage)
    query = jax.device_put(ids)
    kwargs = dict(vocab_start=start, local_rows=rows, head_dim=head_dim)
    raw = jax.jit(functools.partial(resident_lookup_raw, **kwargs))(table, query)
    np.testing.assert_array_equal(
        np.asarray(raw), expected_raw.reshape(tokens, heads * head_dim)
    )
    decoded = jax.jit(decode_embeddings)(raw, jnp.asarray(scale))
    actual = jax.jit(functools.partial(resident_lookup, **kwargs))(
        table, query, jnp.asarray(scale)
    )
    for result in (decoded, actual):
        np.testing.assert_array_equal(
            np.asarray(result), expected.reshape(tokens, heads * head_dim)
        )
    # The last logical row shares storage with physically padded vocabulary rows.
    local_ids = jnp.asarray([0, 1, 3, rows - 1], jnp.int32)
    gathered = jax.jit(functools.partial(compact_rows, head_dim=head_dim))(
        table, local_ids
    )
    np.testing.assert_array_equal(np.asarray(gathered), encoded[[0, 1, 3, rows - 1]])
    np.testing.assert_array_equal(np.asarray(table), storage)

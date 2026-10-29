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
"""Mixed request selection and attention keep row order and donated owners."""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from qsa_review_reference import check_reference

from vllm_torchtpu.kernels.qwen4_exp import decode_indexer, qsa_rpa


def mixed_inputs(tokens, prefill, reverse, verify=False):
    rng = np.random.default_rng(22912436)
    counts = [1] * 255 + [prefill]
    if verify:
        counts[253:255] = [4, 4]
    if reverse:
        counts += [0] * 17
    starts = np.r_[0, np.cumsum(counts)].astype(np.int32)
    requests = len(counts)
    context = np.arange(requests, dtype=np.int32) % 7 + 96
    req = np.minimum(
        np.searchsorted(starts[1:], np.arange(tokens), side="right"), requests - 1
    )
    lengths = np.where(
        np.arange(tokens) < starts[-1],
        context[req] + np.arange(tokens) - starts[req] + 1,
        0,
    ).astype(np.int32)
    owners = rng.permutation(np.arange(1, requests + 3))
    if reverse:
        owners = owners[::-1].copy()
    table = np.zeros((requests, 8), np.int32)
    table[:, 0] = owners[:requests]
    table[255, 1:3] = owners[requests:]
    compressed = np.zeros((requests + 3, 32, 1, 128), np.float32)
    for request in range(requests):
        scores = np.arange(1, 257) if request % 2 == 0 else np.arange(256, 0, -1)
        for column, page in enumerate(table[request]):
            if page:
                compressed[page, :, 0, 0] = scores[column * 32 : (column + 1) * 32]
    q = jnp.asarray(rng.normal(0, 0.2, (tokens, 4, 128)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 1.5, q.shape), jnp.bfloat16)
    query_index = jnp.zeros((tokens, 2, 128), jnp.bfloat16).at[:, :, 0].set(1)
    pool = jnp.asarray(rng.normal(0, 0.2, (requests + 3, 128, 1, 2, 128)), jnp.bfloat16)
    index_args = (
        query_index,
        jnp.asarray(compressed, jnp.bfloat16),
        jnp.asarray(lengths),
        jnp.asarray(table),
        jnp.asarray(req, jnp.int32),
        jnp.asarray(starts),
        jnp.asarray(context),
        jnp.array([253 if verify else 255, 255, 256], jnp.int32),
    )
    return q, gate, pool, index_args


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "tokens,prefill,reverse,verify",
    [
        (257, 2, False, False),
        (512, 257, True, False),
        (512, 2, True, False),
        (1024, 257, False, True),
    ],
)
def test_native_mixed_indexer_and_attention(
    tokens, prefill, reverse, verify, record_property
):
    q, gate, pool, index_args = mixed_inputs(tokens, prefill, reverse, verify)
    before = [np.asarray(x).tobytes() for x in (q, gate, pool, *index_args)]

    def select(q, compressed, lengths, table, req, starts, context, distribution):
        return decode_indexer.select_decode_blocks(
            q,
            compressed,
            lengths,
            table,
            req,
            starts=starts,
            context=context,
            request_distribution=distribution,
            block_topk=16,
            scheduling_group_namespace=37,
        )

    compiled = jax.jit(select).lower(*index_args).compile()
    memory = compiled.memory_analysis()
    assert memory is not None and memory.temp_size_in_bytes < 2 * 1024**3
    record_property(
        "index_memory",
        dict(
            arguments=memory.argument_size_in_bytes,
            outputs=memory.output_size_in_bytes,
            temporary=memory.temp_size_in_bytes,
            alias=memory.alias_size_in_bytes,
        ),
    )
    blocks = compiled(*index_args)
    jax.block_until_ready(blocks)
    ids = np.asarray(blocks)
    _, compressed, lengths, table, req, starts, _, _ = map(np.asarray, index_args)
    for row in range(tokens):
        n = int(lengths[row]) // 4
        live = ids[row][ids[row] >= 0]
        scores = compressed[table[req[row]], :, 0, 0].reshape(-1)
        assert len(live) == min(n, 16)
        assert len(set(live)) == len(live)
        assert np.all(live < n)
        np.testing.assert_array_equal(
            np.sort(scores[live]), np.sort(scores[:n])[-min(n, 16) :] if n else []
        )
        np.testing.assert_array_equal(ids[row, len(live) :], -1)
    np.testing.assert_array_equal(compiled(*index_args), blocks)
    args = (
        q,
        gate,
        blocks,
        index_args[2],
        index_args[3][index_args[4]],
        pool,
        index_args[5],
        index_args[7],
    )

    def attention(q, gate, blocks, lengths, table, cache, starts, distribution):
        return qsa_rpa.qsa_attention(
            q,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            request_distribution=distribution,
            kv_heads=1,
            ratio=4,
        )

    attend = jax.jit(attention).lower(*args).compile()
    memory = attend.memory_analysis()
    assert memory is not None and memory.temp_size_in_bytes < 2 * 1024**3
    record_property(
        "attention_memory",
        dict(
            arguments=memory.argument_size_in_bytes,
            outputs=memory.output_size_in_bytes,
            temporary=memory.temp_size_in_bytes,
            alias=memory.alias_size_in_bytes,
        ),
    )
    result = attend(*args)
    jax.block_until_ready(result)
    check_reference(
        args[:7],
        result[0],
        kv_heads=1,
        rows=np.flatnonzero(lengths > 0),
        ratio=4,
        page_size=128,
    )
    assert np.asarray(result[1]).tobytes() == np.asarray(pool).tobytes()
    repeated = attend(*args)
    jax.block_until_ready(repeated)
    assert [np.asarray(x).tobytes() for x in repeated] == [
        np.asarray(x).tobytes() for x in result
    ]
    assert [np.asarray(x).tobytes() for x in (q, gate, pool, *index_args)] == before
    record_property("input_sha256", [hashlib.sha256(x).hexdigest() for x in before])
    record_property(
        "output_sha256", hashlib.sha256(np.asarray(result[0]).tobytes()).hexdigest()
    )
    record_property("input_cache_repeat_exact", True)

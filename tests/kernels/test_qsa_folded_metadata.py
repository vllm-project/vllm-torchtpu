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
"""GDN-folded Runner boundaries preserve the original query/page metadata."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_qsa_mixed_native as native_fixture

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

_BASE_INPUTS = native_fixture.mixed_inputs


def folded_verify_inputs():
    q, gate, pool, index_args = _BASE_INPUTS(1024, 257, False, True)
    return (
        q,
        gate,
        pool,
        (
            *index_args[:-1],
            jnp.array([253, 253, 256], jnp.int32),
        ),
    )


def test_folded_verify_changes_only_distribution():
    q, gate, pool, index_args = _BASE_INPUTS(1024, 257, False, True)
    fq, fg, fp, folded_args = folded_verify_inputs()
    for before, after in zip(
        (q, gate, pool, *index_args[:-1]), (fq, fg, fp, *folded_args[:-1])
    ):
        assert np.asarray(before).tobytes() == np.asarray(after).tobytes()
    np.testing.assert_array_equal(index_args[-1], [253, 255, 256])
    np.testing.assert_array_equal(folded_args[-1], [253, 253, 256])


def test_folded_verify_routes_rows_and_owners(monkeypatch):
    q, gate, pool, index_args = folded_verify_inputs()
    _, _, lengths, table, req, starts, _, distribution = index_args
    blocks = jnp.broadcast_to(jnp.arange(len(q))[:, None], (len(q), 16))
    pages = table[req]
    seen = []

    def attend(kind, query, weight, ids, lens, rows, owner, *args, **kw):
        for row, length in zip(np.asarray(ids[:, 0]), np.asarray(lens)):
            if length:
                seen.append((int(row), kind))
        output = (
            query.astype(jnp.float32)
            + 2 * weight.astype(jnp.float32)
            + rows[:, :1, None]
        )
        return jnp.where((lens > 0)[:, None, None], output, 0), owner

    def prefill(query, weight, ids, lens, rows, owner, local_starts, **kw):
        assert int(local_starts[0]) == 253
        return attend("prefill", query, weight, ids, lens, rows, owner, **kw)

    monkeypatch.setattr(qsa_rpa, "_prefill_attention", prefill)
    monkeypatch.setattr(
        qsa_rpa, "_rpa_attention_batch", lambda *a, **kw: attend("decode", *a, **kw)
    )
    monkeypatch.setattr(qsa_rpa.batching, "rpa_query_batch_size", lambda *a, **kw: 64)
    with jax.disable_jit():
        output, owner = qsa_rpa.qsa_attention(
            q,
            gate,
            blocks,
            lengths,
            pages,
            pool,
            starts,
            request_distribution=distribution,
            kv_heads=1,
            ratio=4,
        )
    expected = np.zeros(q.shape, np.float32)
    expected_routes = []
    for request in range(256):
        for row in range(int(starts[request]), int(starts[request + 1])):
            expected_routes.append((row, "decode" if request < 253 else "prefill"))
            expected[row] = (
                np.asarray(q[row], np.float32)
                + 2 * np.asarray(gate[row], np.float32)
                + np.asarray(table[request, 0])
            )
    assert sorted(seen) == expected_routes
    np.testing.assert_array_equal(output, expected)
    assert owner is pool


@pytest.mark.tpu_test
def test_native_gdn_folded_verify_metadata(monkeypatch, record_property):
    def inputs(tokens, prefill, reverse, verify=False):
        assert (tokens, prefill, reverse, verify) == (1024, 257, False, True)
        return folded_verify_inputs()

    monkeypatch.setattr(native_fixture, "mixed_inputs", inputs)
    native_fixture.test_native_mixed_indexer_and_attention(
        1024, 257, False, True, record_property
    )

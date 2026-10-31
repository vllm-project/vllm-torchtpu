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
"""CPU routing regression for QSA attention request partitions.

The public entry ``qsa_attention`` must organize a mixed batch by the
Runner partition ``request_distribution = [decode_end, verify_end,
active_request_end]`` — request-COUNT segment ends, not the standard
batched-RPA ``seq_range`` names, and never inferred from shapes:

- rows of requests below ``verify_end`` attend through the decode/verify
  path (the prefix);
- rows of requests in ``[verify_end, active_request_end)`` attend through
  the prefill path with suffix-masked lengths and clipped starts;
- rows beyond ``starts[active_request_end]`` (padding) carry no valid
  task even when their ``lengths`` entries are nonzero garbage;
- an empty segment contributes zero valid tasks (the wrappers may still
  be called with all-zero rows);
- the cache owner returns byte-identical, across mixed batches and two
  consecutive steps over one shared pool.

Expected values follow the per-request partition contract. Deterministic
pure-JAX stand-ins check row routing and cache ownership; native kernel
numerics are covered separately.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

RATIO = 4
KV_HEADS = 1
BATCH = 4


def partition_inputs(
    counts, distribution, *, seed=17, padding=3, pages=None, garbage_padding=False
):
    """Build one mixed batch from per-request token counts.

    ``counts`` may contain idle requests (0 tokens). ``padding`` extra rows
    are appended beyond ``starts[requests]``. By default they carry length
    0 (the Runner contract); with ``garbage_padding`` they carry nonzero
    garbage so the mixed path's suffix masking is verified against a
    partition bug that trusts ``lengths`` over the distribution.
    """
    rng = np.random.default_rng(seed)
    requests = len(counts)
    tokens = int(sum(counts)) + padding
    starts = np.zeros((requests + 1,), np.int32)
    starts[1:] = np.cumsum(counts)
    request_of_row = np.minimum(
        np.searchsorted(starts[1:], np.arange(tokens), side="right"), requests - 1
    )
    lengths = np.zeros((tokens,), np.int32)
    for row in range(tokens):
        if row < starts[requests]:
            req = request_of_row[row]
            lengths[row] = row - starts[req] + 1
    if garbage_padding:
        # Nonzero garbage that the mixed path's suffix mask must neutralize.
        lengths[int(starts[requests]) :] = 7
    table = np.zeros((requests, 4), np.int32)
    table[:, 0] = np.arange(requests) % max(1, pages or requests)
    if requests and pages:
        table[:, 1] = (np.arange(requests) + 1) % pages
    q = jnp.asarray(rng.normal(0, 0.3, (tokens, 2, 128)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 1.0, (tokens, 2, 128)), jnp.bfloat16)
    pool = jnp.asarray(
        rng.normal(0, 0.2, (pages or requests + 1, 128, 1, 2, 128)),
        jnp.bfloat16,
    )
    # Column 0 keeps the request's table value (drives the stand-in
    # arithmetic); column 3 carries a stable per-row ID so the stand-ins
    # identify ORIGINAL rows from whatever slice/reorder/mask the
    # implementation applies (invalid rows arrive masked to -1).
    blocks = np.zeros((tokens, 4), np.int32)
    for row in range(tokens):
        req = request_of_row[row]
        blocks[row] = table[req]
        blocks[row, 3] = row
    return {
        "q": q,
        "gate": gate,
        "blocks": jnp.asarray(blocks, jnp.int32),
        "lengths": jnp.asarray(lengths, jnp.int32),
        "table": jnp.asarray(table, jnp.int32),
        "pool": pool,
        "starts": jnp.asarray(starts, jnp.int32),
        "distribution": jnp.asarray(distribution, jnp.int32),
        "counts": counts,
        "request_of_row": request_of_row,
        "tokens": tokens,
    }


def standin_calls(monkeypatch, seen):
    """Replace the compute kernels with per-row deterministic stand-ins.

    decode/verify rows produce ``q + 2*gate + table[req, 0]``; prefill rows
    produce ``q + 3*gate + table[req, 0]`` — different coefficients so a
    row routed to the wrong path shows a value mismatch. Rows whose routed
    length is 0 produce 0. The owner passes through unchanged.
    """

    def row_value(query, gate, blocks, live):
        return jnp.where(
            (live > 0)[:, None, None],
            query.astype(jnp.float32)
            + 2 * gate.astype(jnp.float32)
            + blocks[:, :1, None].astype(jnp.float32),
            0.0,
        )

    def prefill_value(query, gate, blocks, live):
        return jnp.where(
            (live > 0)[:, None, None],
            query.astype(jnp.float32)
            + 3 * gate.astype(jnp.float32)
            + blocks[:, :1, None].astype(jnp.float32),
            0.0,
        )

    def routed_ids(blocks):
        ids = np.asarray(blocks)[:, 3].astype(np.int64)
        return [int(x) for x in ids]

    def rpa_batch(q, g, blocks, lens, table, owner, **kw):
        # Row identity comes from the input ID channel: whatever windowing,
        # slicing or reordering the implementation applies, each attended
        # row reports its ORIGINAL id (masked rows arrive as -1). No call
        # counting, no chunk-size assumption, and no requirement that any
        # particular RPA entry be used.
        for rid, n in zip(routed_ids(blocks), np.asarray(lens).astype(np.int64)):
            if rid >= 0 and n > 0:
                seen.append((rid, "decode"))
        return row_value(q, g, blocks, np.asarray(lens)), owner

    def rpa_attention(query, gate, blocks, lengths, table, cache, **kw):
        # The pure decode/verify path: every row with a live length attends.
        for rid, n in zip(routed_ids(blocks), np.asarray(lengths).astype(np.int64)):
            if rid >= 0 and n > 0:
                seen.append((rid, "decode"))
        result = row_value(query, gate, blocks, np.asarray(lengths))
        return jnp.where(
            (lengths > 0)[:, None, None], result, kw["initial_output"]
        ), cache

    def prefill(query, g, blocks, lens, table, cache, starts, **kw):
        for rid, n in zip(routed_ids(blocks), np.asarray(lens).astype(np.int64)):
            if rid >= 0 and n > 0:
                seen.append((rid, "prefill"))
        return prefill_value(query, g, blocks, np.asarray(lens)), cache

    monkeypatch.setattr(qsa_rpa, "_rpa_attention_batch", rpa_batch)
    monkeypatch.setattr(qsa_rpa, "rpa_attention", rpa_attention)
    monkeypatch.setattr(qsa_rpa, "_prefill_attention", prefill)
    monkeypatch.setattr(
        qsa_rpa.batching, "rpa_query_batch_size", lambda *a, **kw: BATCH
    )


def expected_reference(inp):
    """Independent per-request expectation from the partition contract."""
    q, gate, table = inp["q"], inp["gate"], inp["table"]
    counts = inp["counts"]
    decode_end, verify_end, active_end = [int(x) for x in inp["distribution"]]
    starts = np.asarray(inp["starts"])
    tokens = inp["tokens"]
    prefix_tokens = min(int(starts[verify_end]), tokens)
    active_tokens = min(int(starts[active_end]), tokens)
    expected = np.zeros((tokens, *q.shape[1:]), np.float32)
    routes = {}
    for row in range(tokens):
        req = int(inp["request_of_row"][row])
        if row < prefix_tokens and row < starts[req + 1] and counts[req] > 0:
            routes[row] = "decode"
            expected[row] = (
                np.asarray(q[row], np.float32)
                + 2 * np.asarray(gate[row], np.float32)
                + float(np.asarray(table[req, 0]))
            )
        elif prefix_tokens <= row < active_tokens and counts[req] > 0:
            routes[row] = "prefill"
            expected[row] = (
                np.asarray(q[row], np.float32)
                + 3 * np.asarray(gate[row], np.float32)
                + float(np.asarray(table[req, 0]))
            )
    return expected, routes, prefix_tokens, active_tokens


def run_attention(inp):
    return qsa_rpa.qsa_attention(
        inp["q"],
        inp["gate"],
        inp["blocks"],
        inp["lengths"],
        inp["table"],
        inp["pool"],
        inp["starts"],
        request_distribution=inp["distribution"],
        kv_heads=KV_HEADS,
        ratio=RATIO,
    )


def valid_routed_rows(seen):
    return sorted(seen)


@pytest.mark.parametrize(
    "counts,distribution",
    [
        # decode + verify + prefill mixed with an idle request between them.
        ([1, 1, 0, 5, 9], [2, 3, 5]),
        # pure decode: no verify, no prefill.
        ([1, 1, 3, 1], [4, 4, 4]),
        # pure prefill: empty decode and verify segments.
        ([3, 9, 5], [0, 0, 3]),
        # empty verify segment between decode and prefill.
        ([2, 0, 4, 7], [1, 1, 4]),
        # decode-only with an idle request (zero valid tasks from it).
        ([1, 0, 1, 2], [4, 4, 4]),
    ],
)
def test_partition_routes_rows_and_owners(monkeypatch, counts, distribution):
    # Request boundaries mask inactive rows in both attention segments.
    decode_end, verify_end, active_end = distribution
    inp = partition_inputs(
        counts,
        distribution,
        pages=len(counts) + 2,
        garbage_padding=active_end > verify_end,
    )
    before = np.asarray(inp["pool"]).tobytes()
    seen = []
    standin_calls(monkeypatch, seen)
    with jax.disable_jit():
        output, owner = run_attention(inp)
    expected, routes, prefix_tokens, active_tokens = expected_reference(inp)
    np.testing.assert_array_equal(np.asarray(output), expected)
    assert valid_routed_rows(seen) == sorted(routes.items())
    assert np.asarray(owner).tobytes() == before
    # Padding rows beyond the partition carry no valid task even though
    # their lengths entries are nonzero garbage.
    assert (np.asarray(output)[active_tokens:] == 0).all()


def test_zero_active_has_no_valid_tasks(monkeypatch):
    inp = partition_inputs([3, 5, 2], [0, 0, 0], pages=6)
    seen = []
    standin_calls(monkeypatch, seen)
    with jax.disable_jit():
        output, owner = run_attention(inp)
    assert (np.asarray(output) == 0).all()
    assert valid_routed_rows(seen) == []
    # Zero valid tasks; wrappers MAY have been called with empty rows.
    assert np.asarray(owner).tobytes() == np.asarray(inp["pool"]).tobytes()


@pytest.mark.parametrize(
    "counts,distribution", [([4, 12, 7], [2, 2, 3]), ([1, 1, 1], [1, 1, 3])]
)
def test_two_consecutive_steps_share_pool(monkeypatch, counts, distribution):
    inp = partition_inputs(counts, distribution, pages=len(counts) + 2)
    expected, routes, _, _ = expected_reference(inp)
    seen = []
    standin_calls(monkeypatch, seen)
    with jax.disable_jit():
        first_out, first_owner = run_attention(inp)
        second_out, second_owner = run_attention(inp)
    np.testing.assert_array_equal(np.asarray(first_out), expected)
    np.testing.assert_array_equal(np.asarray(second_out), expected)
    # The shared pool returns byte-identical from both steps: every
    # attention region (written or unwritten) is unchanged.
    pool_bytes = np.asarray(inp["pool"]).tobytes()
    assert np.asarray(first_owner).tobytes() == pool_bytes
    assert np.asarray(second_owner).tobytes() == pool_bytes
    assert valid_routed_rows(seen) == sorted(list(routes.items()) * 2)


@pytest.mark.parametrize(
    "counts,distribution",
    [
        ([130, 1, 65], [1, 1, 3]),  # page-crossing context + short tails
        ([1, 1, 1, 1], [2, 2, 4]),  # one token per request
        ([7, 0, 0, 9], [1, 2, 4]),  # idle requests inside verify segment
    ],
)
def test_context_and_capacity_variation(monkeypatch, counts, distribution):
    inp = partition_inputs(counts, distribution, pages=len(counts) + 2, seed=31)
    seen = []
    standin_calls(monkeypatch, seen)
    with jax.disable_jit():
        output, owner = run_attention(inp)
    expected, routes, _, _ = expected_reference(inp)
    np.testing.assert_array_equal(np.asarray(output), expected)
    assert valid_routed_rows(seen) == sorted(routes.items())
    assert np.asarray(owner).tobytes() == np.asarray(inp["pool"]).tobytes()


def test_distribution_must_be_int32_triple():
    inp = partition_inputs([1, 1], [1, 1, 2], pages=4)
    # Wrong arity (a dtype-only violation is canonicalized away on CPU
    # without x64, so the shape branch is the observable contract).
    bad = jnp.asarray([1, 1], jnp.int32)
    with (
        pytest.raises(ValueError, match="request_distribution must be int32"),
        jax.disable_jit(),
    ):
        qsa_rpa.qsa_attention(
            inp["q"],
            inp["gate"],
            inp["blocks"],
            inp["lengths"],
            inp["table"],
            inp["pool"],
            inp["starts"],
            request_distribution=bad,
            kv_heads=KV_HEADS,
            ratio=RATIO,
        )

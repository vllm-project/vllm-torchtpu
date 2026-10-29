# SPDX-License-Identifier: Apache-2.0
"""(c) DCP-2 main-KV writes: each rank writes exactly its owned rows.

The replicated ``qsa_cache_writes`` (B-token table) runs on the full pool; the
DCP adapter (default: ``qsa_cache_writes(dcp=...)``) runs per rank on a pool
holding only that rank's owned pages under the 2B-token table, every other
page NaN-poisoned, compressed-key / ring / null pages replicated. The result
must equal the projection of the replicated result byte for byte: owned rows
written, NaN poison and the partner's rows untouched, compressed keys and
rings identical to the replicated write. Writes run in Pallas interpret mode.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import (
    D,
    random_fp8,
    build_layout,
    from_raw,
    owner_of_token,
    project_rank_pool,
    rank_operand,
    raw,
)

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa

B, WIDTH, INDEX_DIM, RATIO, ROTARY = 512, 256, 128, 4, 64
K_SCALE, V_SCALE = 0.375, 2.5


def write_inputs(seed, query_lens, contexts):
    rng = np.random.default_rng(seed)
    query_lens = np.asarray(query_lens)
    contexts = np.asarray(contexts)
    requests = len(contexts)
    starts = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    tokens = max(8, -(-int(starts[-1]) // 8) * 8)
    seq = np.maximum(contexts + query_lens, 1)
    cols = int(-(-seq.max() // B))
    num_pages = int(sum(-(-seq // B)) * 2 + requests * (cols + 1) + 4)
    layout = build_layout(rng, seq, B, num_pages)
    free = iter(layout.free_pages)
    comp = np.array([[next(free) for _ in range(cols)] for _ in range(requests)], np.int32)
    ring = np.array([next(free) for _ in range(requests)], np.int32)
    keep = sorted({0, *comp.ravel().tolist(), *ring.tolist()})
    full = random_fp8(rng, (num_pages, B, 1, 4, WIDTH))
    positions = np.zeros(tokens, np.int32)
    for r in range(requests):
        positions[starts[r] : starts[r + 1]] = contexts[r] + np.arange(query_lens[r])

    def normal(*shape):
        return jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)

    step = dict(
        k=normal(tokens, 1, WIDTH),
        v=normal(tokens, 1, WIDTH),
        raw=normal(tokens, INDEX_DIM),
        positions=jnp.asarray(positions),
        nik=normal(INDEX_DIM),
        trig=jnp.pad(normal(8192, ROTARY), ((0, 0), (0, 128 - ROTARY))),
        starts=jnp.asarray(starts),
        context=jnp.asarray(contexts, jnp.int32),
        ring_ids=jnp.asarray(ring),
        compressed_table=jnp.asarray(comp),
    )
    return layout, full, keep, step, positions, starts


def call_writes(fn, pool, table, step, *, main_slot, max_query_len, dcp):
    s = step
    out = fn(
        pool, s["k"], s["v"], s["raw"], s["positions"], s["nik"], s["trig"],
        s["starts"], s["context"], s["ring_ids"], jnp.asarray(table),
        s["compressed_table"], compress_ratio=RATIO, norm_eps=1e-6,
        rotary_dim=ROTARY, k_scale=K_SCALE, v_scale=V_SCALE, main_slot=main_slot,
        max_query_len=max_query_len, **({} if dcp is None else {"dcp": dcp}),
    )
    return out[0]


def check_dcp_writes(write_fn, query_lens, contexts, *, main_slot=-1, max_query_len=1,
                     seed=0):
    """Run replicated vs DCP writes; raise AssertionError on any byte difference."""
    layout, full, keep, step, positions, starts = write_inputs(seed, query_lens, contexts)
    full_bytes = raw(full)
    after = raw(call_writes(pooled_qsa.qsa_cache_writes, full, layout.full_table, step,
                            main_slot=main_slot, max_query_len=max_query_len, dcp=None))
    assert not np.array_equal(after, full_bytes), "replicated write wrote nothing"
    owners = set()
    for r in range(len(contexts)):
        for t in positions[starts[r] : starts[r + 1]]:
            owners.add(int(owner_of_token(t, B)))
    for rank in range(D):
        initial = project_rank_pool(full_bytes, layout, rank, keep_pages=keep)
        want = project_rank_pool(after, layout, rank, keep_pages=keep)
        got = raw(call_writes(write_fn, from_raw(initial), layout.dcp_table, step,
                              main_slot=main_slot, max_query_len=max_query_len,
                              dcp=(rank_operand(rank), D)))
        if rank in owners:
            assert not np.array_equal(want, initial)
        bad = np.argwhere((got != want).reshape(got.shape[0], got.shape[1], -1).any(-1))
        if len(bad):
            page, row = bad[0]
            kind = ("dcp-main" if page in layout.dcp_table else "replicated-main"
                    if page in layout.full_table else "kept" if page in keep else "free")
            raise AssertionError(
                f"rank {rank}: {len(bad)} rows differ; first page {page} ({kind}) row {row}"
            )


# --- default adapter: the production writer with dcp=(rank, 2)
production_writes = pooled_qsa.qsa_cache_writes


# --- fault controls (each MUST be detected)
def writes_also_unowned(pool, k, v, raw_keys, positions, *args, dcp, **kw):
    """Owned write, then also the partner's rows at (table[t // 2B], t % B)."""
    pool, req, logical, valid = production_writes(pool, k, v, raw_keys, positions, *args,
                                                  dcp=dcp, **kw)
    table = args[5]
    rank = int(np.asarray(dcp[0])[0])
    unowned = valid & ((logical // B) % D != rank)
    page = jnp.where(unowned, table[req, logical // (D * B)], len(pool))
    rows = qsa_rpa.pack_rows(k, v, fp8=True, k_scale=kw["k_scale"], v_scale=kw["v_scale"])
    slot = kw.get("main_slot", -1)
    pool = pooled_qsa.write_main_cache(pool, rows, page, logical, unowned,
                                       **({"slot": slot} if slot >= 0 else {}))
    return pool, req, logical, valid


def writes_flipped_owner(*args, dcp, **kw):
    return production_writes(*args, dcp=(1 - dcp[0], dcp[1]), **kw)


def writes_replicated_span(*args, dcp, **kw):
    """Forgets DCP entirely: B-token indexing into the 2B-token table."""
    del dcp
    return production_writes(*args, **kw)


CASES = {
    # c6-ish MTP verify, idle seats, rows crossing B and 2B boundaries.
    "verify": ([4, 4, 0, 4, 4, 4], [37, 509, 0, 1022, 1020, 2045]),
    # Plain decode at every page phase.
    "decode": ([1] * 8, [0, 1, 511, 512, 1023, 1024, 2047, 3071]),
    # Unaligned prefill spans crossing B / 2B (per-row writer path).
    "prefill_edges": ([70, 37, 61], [480, 1500, 2030]),
}


@pytest.mark.parametrize("main_slot", [-1, 0, 1])
@pytest.mark.parametrize("case", sorted(CASES))
def test_owned_writes_match_replicated(interpret, case, main_slot):
    check_dcp_writes(production_writes, *CASES[case], main_slot=main_slot)


@pytest.mark.parametrize("case", ["verify", "decode"])
def test_owned_writes_fused_writer(interpret, monkeypatch, case):
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_FUSED_KV_WRITE", True)
    check_dcp_writes(production_writes, *CASES[case], main_slot=0, max_query_len=4)


def test_owned_writes_prefill_chunk_2048(interpret):
    """A 2048-token chunk on a 1152-token cached prefix (not a multiple of B;
    128-aligned so the bulk tile writer runs, as for served chunks; the
    unaligned per-row writer is ``prefill_edges``, which costs ~0.1 s/token
    in interpret mode)."""
    check_dcp_writes(production_writes, [2048], [1152], main_slot=0)


@pytest.mark.parametrize(
    "fault", [writes_also_unowned, writes_flipped_owner, writes_replicated_span],
    ids=lambda f: f.__name__,
)
def test_write_fault_controls_are_detected(interpret, fault):
    with pytest.raises(AssertionError):
        check_dcp_writes(fault, *CASES["verify"], main_slot=0)


def test_harness_projection_is_exact():
    """Self-test of the layout: each logical token appears once, on its owner,
    at (dcp_table[t // 2B], t % B), with the replicated row's bytes."""
    rng = np.random.default_rng(5)
    seq = np.array([1, B - 1, B, B + 1, 3 * B + 7, 9 * B])
    layout = build_layout(rng, seq, B, 80)
    full = rng.integers(0, 0x7E, (80, B, 2), dtype=np.uint8)
    pools = [project_rank_pool(full, layout, r) for r in range(D)]
    for r, n in enumerate(seq):
        for t in [0, n - 1, *range(0, n, 97)]:
            o = int(owner_of_token(t, B))
            src = full[layout.full_table[r, t // B], t % B]
            np.testing.assert_array_equal(pools[o][layout.dcp_table[r, t // (2 * B)], t % B], src)
            other = pools[1 - o][layout.dcp_table[r, t // (2 * B)], t % B]
            # The partner holds its own token at that slot (t +- B) or poison.
            partner = (t // (2 * B)) * 2 + (1 - o)
            if partner * B < n:
                src = full[layout.full_table[r, partner], t % B]
                np.testing.assert_array_equal(other, src)
            else:
                assert np.all(other == 0x7F)

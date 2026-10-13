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
"""Tests for DCP (decode context parallel) StreamIndex Top-K"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    cp_local_to_global,
    cp_rank_as_data,
    dcp_sc_compact,
    metadata,
    streamindex_topk,
    streamindex_topk_dcp,
)

pytestmark = pytest.mark.tpu_test

P = jax.sharding.PartitionSpec


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_candidate_all_gather_is_rank_major(dcp_size):
    """The merge resolves a winner as rank `slot // k`, position `slot % k`.

    Both readings assume this layout, and nothing else in the merge would
    notice it changing: every rank would resolve the wrong candidate and still
    return a full, plausible, silently wrong list. Asserting the full
    rank-major arange pins order and chunk width together.
    """
    devices = jax.devices()
    if len(devices) < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {len(devices)}")

    rows, k = 4, 8
    mesh = jax.sharding.Mesh(np.array(devices[:dcp_size]), (DCP_AXIS_NAME,))

    def _local(_):
        rank = cp_rank_as_data(DCP_AXIS_NAME, dcp_size)
        # Rank r writes r*k + j, so a rank-major gather reads back as arange.
        mine = rank * k + jnp.arange(k, dtype=jnp.int32)
        return jax.lax.all_gather(
            jnp.broadcast_to(mine, (rows, k)), DCP_AXIS_NAME, axis=1, tiled=True
        )

    gathered = jax.jit(
        jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=(P(),),
            out_specs=P(DCP_AXIS_NAME),
            check_vma=False,
        )
    )(jnp.zeros((rows, 1), jnp.int32))

    got = np.asarray(gathered).reshape(dcp_size, rows, dcp_size * k)
    want = np.broadcast_to(
        np.arange(dcp_size * k, dtype=np.int32), (rows, dcp_size * k)
    )
    for r in range(dcp_size):
        np.testing.assert_array_equal(
            got[r], want, err_msg=f"rank {r} saw a non-rank-major candidate axis"
        )


def _require_sparsecore():
    """Probe in a test body: a chip touch at collection breaks spawned workers."""
    try:
        if pltpu.get_tpu_info().sparse_core is not None:
            return
    except Exception:  # no TPU at all  # noqa: S110
        pass
    pytest.skip("needs SparseCore")


def _xla_pack_reference(vals):
    """The scatter the kernel replaced, unreachable in `src`, kept as an oracle."""
    num_rows, width = vals.shape
    keep = vals >= 0

    # Exclusive prefix count of kept entries gives the destination slot, < width.
    slot = jnp.cumsum(keep, axis=1, dtype=jnp.int32) - keep
    rows = jnp.arange(num_rows, dtype=jnp.int32)[:, None]

    # Column `width` is a dump for the dropped entries, sliced off at the end.
    out = jnp.full((num_rows, width + 1), -1, dtype=jnp.int32)
    out = out.at[rows, jnp.where(keep, slot, width)].set(jnp.where(keep, vals, -1))
    return out[:, :width]


def _sparse_fixture(num_rows=16, width=128, keep_rate=0.5, seed=20260906):
    """Non-negative values with `-1` holes; row 0 is full and row 1 is empty."""
    rng = np.random.default_rng(seed)
    vals = rng.integers(0, 1 << 20, size=(num_rows, width)).astype(np.int32)
    vals[rng.random((num_rows, width)) > keep_rate] = -1
    vals[0] = rng.integers(0, 1 << 20, size=width).astype(np.int32)
    if num_rows > 1:
        vals[1] = -1
    return vals


@pytest.mark.parametrize("num_rows,width", [(16, 128), (1, 16), (33, 256)])
@pytest.mark.parametrize("keep_rate", [0.1, 0.5, 0.9])
def test_sparsecore_pack_matches_xla(num_rows, width, keep_rate):
    """Elementwise, not just as a set: both packs must emit in ascending order."""
    _require_sparsecore()
    vals = _sparse_fixture(num_rows, width, keep_rate)

    want = np.asarray(_xla_pack_reference(jnp.asarray(vals)))
    got = np.asarray(dcp_sc_compact.pack_owned_winners(jnp.asarray(vals)))

    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
def test_sparsecore_fused_pack_matches_reference(dcp_size):
    """Fused SparseCore gather + pack matches take_along_axis + reference pack."""
    _require_sparsecore()
    num_rows, k = 16, 128
    rng = np.random.default_rng(20261003)
    # Slots can come from any rank [0, dcp_size * k)
    slots = np.stack(
        [
            rng.choice(dcp_size * k, size=k, replace=False).astype(np.int32)
            for _ in range(num_rows)
        ]
    )
    slots[0, -5:] = -1
    idxs = rng.integers(0, 100000, size=(num_rows, k), dtype=np.int32)
    idxs[1, -3:] = -1

    for r in range(dcp_size):
        lo = r * k
        hi = lo + k
        ours = (slots >= lo) & (slots < hi)
        safe = np.maximum(slots, 0)
        merged = np.where(ours, np.take_along_axis(idxs, safe % k, axis=1), -1)
        want = _xla_pack_reference(jnp.asarray(merged))
        got = dcp_sc_compact.pack_owned_winners(
            jnp.asarray(slots), jnp.asarray(idxs), r
        )
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def _simulate_merge(num_rows, k, span, dcp_size, interleave_c, seed=20260929):
    """Stages 1-3 of `streamindex_topk_dcp` in numpy, as `resolve` receives them.

    Returns each rank's rank-local candidate list, the merged flat slot columns,
    and the true global top-k the ranks must reassemble between them.
    """
    rng = np.random.default_rng(seed)
    # Distinct scores, so the top-k is unambiguous and the partition is exact.
    scores = rng.permutation(num_rows * span).reshape(num_rows, span).astype(np.float32)
    owner = (np.arange(span) // interleave_c) % dcp_size

    local_idxs = np.full((dcp_size, num_rows, k), -1, np.int32)
    local_scores = np.full((dcp_size, num_rows, k), -np.inf, np.float32)
    for r in range(dcp_size):
        owned = np.flatnonzero(owner == r)
        for t in range(num_rows):
            best = owned[np.argsort(-scores[t, owned])][:k]
            local_idxs[r, t, : len(best)] = np.asarray(
                metadata.cp_global_to_local(jnp.asarray(best), dcp_size, interleave_c)
            )
            local_scores[r, t, : len(best)] = scores[t, best]

    # Stage 2 is rank-major, so rank r's k candidates land at columns [r*k, r*k+k).
    gathered = np.concatenate(list(local_scores), axis=1)
    slots = np.full((num_rows, k), -1, np.int32)
    for t in range(num_rows):
        order = np.argsort(-gathered[t])[:k]
        order = order[np.isfinite(gathered[t, order])]
        slots[t, : len(order)] = order

    want = [set(np.argsort(-scores[t])[:k].tolist()) for t in range(num_rows)]
    return local_idxs, slots, want


def _resolve_owned_winners(local_idxs, slots, k, dcp_rank):
    """Stage 3 of `streamindex_topk_dcp`, which inlines it: keep and pack ours.

    The candidate all-gather is rank-major, so flat column `c` came from rank
    `c // k` at that rank's own slot `c % k`.
    """
    safe = jnp.maximum(slots, 0)
    ours = (slots >= 0) & (jnp.floor_divide(safe, k) == dcp_rank)
    mine = jnp.take_along_axis(local_idxs, jnp.mod(safe, k), axis=1)
    return dcp_sc_compact.pack_owned_winners(jnp.where(ours, mine, -1))


@pytest.mark.parametrize("dcp_size,interleave_c", [(2, 1), (4, 4), (8, 4), (8, 64)])
def test_resolve_partitions_the_global_topk(dcp_size, interleave_c):
    """The ranks' packed outputs reassemble the global top-k, each entry once."""
    _require_sparsecore()
    num_rows, k, span = 8, 32, 4096
    local_idxs, slots, want = _simulate_merge(num_rows, k, span, dcp_size, interleave_c)

    lists = np.stack(
        [
            np.asarray(
                _resolve_owned_winners(
                    jnp.asarray(local_idxs[r]), jnp.asarray(slots), k, jnp.int32(r)
                )
            )
            for r in range(dcp_size)
        ]
    )
    assert lists.shape == (dcp_size, num_rows, k)

    for t in range(num_rows):
        got = []
        for r in range(dcp_size):
            row = lists[r, t]
            for local in row[row >= 0].tolist():
                g = int(cp_local_to_global(local, r, dcp_size, interleave_c))
                # A rank must only ever keep positions it actually owns.
                assert (g // interleave_c) % dcp_size == r
                got.append(g)
        # Multiplicity, not just set equality: the LSE merge double-counts a
        # position delivered to two ranks.
        assert len(got) == len(set(got))
        assert set(got) == want[t]


# ===================================================================== #
# End-to-end against the unsharded kernel.
# ===================================================================== #

H_IDX, D_IDX, K = 32, 128, 2048
Q_LEN, KV_LEN, PAGE_SIZE = 256, 32768, 1024
WIDTH = 256


def _global_records(kv_len, seed=7):
    """One record array; both paths are views of it, so any diff is a real bug."""
    vals = jax.random.normal(jax.random.key(seed), (kv_len, D_IDX), jnp.float32)
    fp8 = jax.lax.bitcast_convert_type(
        vals.astype(jnp.float8_e4m3fn), jnp.uint8
    ).reshape(kv_len, D_IDX)
    # 127 is the e8m0 exponent bias, i.e. a scale of exactly 1.0.
    rec = jnp.concatenate([fp8, jnp.full((kv_len, 1), 127, jnp.uint8)], -1)
    return jnp.pad(rec, ((0, 0), (0, WIDTH - rec.shape[-1])))


def _need_devices(dcp_size):
    n = len(jax.devices())
    if n < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {n}")


def _dcp_case(dcp_size, seed, q_len=Q_LEN):
    """Records plus the positional args and kwargs of one sharded DCP call."""
    rec = _global_records(KV_LEN)
    kq, kw = jax.random.split(jax.random.key(seed))
    q = (jax.random.normal(kq, (q_len, H_IDX, D_IDX), jnp.float32) * 0.5).astype(
        jnp.float8_e4m3fn
    )
    weights = jax.random.normal(kw, (q_len, H_IDX), jnp.float32).astype(jnp.bfloat16)

    # interleave_size=1: global position g lives on rank g % dcp_size at local
    # index g // dcp_size, so the shards are a strided de-interleave.
    vpages = KV_LEN // (PAGE_SIZE * dcp_size)
    shards = jnp.concatenate(
        [
            rec[r::dcp_size].reshape(vpages, PAGE_SIZE // 4, 4, WIDTH)
            for r in range(dcp_size)
        ],
        0,
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp_size]), (DCP_AXIS_NAME,))
    shards = jax.device_put(shards, jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME)))

    args = (
        q,
        weights,
        shards,
        jnp.array([KV_LEN], jnp.int32),
        jnp.arange(vpages, dtype=jnp.int32),
        jnp.array([0, q_len], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
    )
    kwargs = dict(
        mesh=mesh,
        k=K,
        compression_ratio=1,
        dcp_size=dcp_size,
        interleave_size=1,
        num_kv_pages_per_block=2,
        num_queries_per_block=128,
    )
    return rec, args, kwargs


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("a2a_merge", [False, True])
def test_dcp_reassembles_the_unsharded_global_topk(dcp_size, a2a_merge):
    """Union the per-rank lists back together; it must be the global top-k.
    Both merge paths; the unsharded reference is all they must agree on."""
    _need_devices(dcp_size)
    rec, args, kwargs = _dcp_case(dcp_size, 0)
    kwargs["a2a_merge"] = a2a_merge
    q, weights = args[0], args[1]

    pages = KV_LEN // PAGE_SIZE
    reference = np.asarray(
        streamindex_topk(
            q,
            weights,
            rec.reshape(pages, PAGE_SIZE // 4, 4, WIDTH),
            jnp.array([KV_LEN], jnp.int32),
            jnp.arange(pages, dtype=jnp.int32),
            jnp.array([0, Q_LEN], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32),
            k=K,
            compression_ratio=1,
            num_kv_pages_per_block=2,
            num_queries_per_block=128,
        )
    )

    # Global leading dim is dcp_size * Q_LEN, rank-major.
    local_topk = np.asarray(streamindex_topk_dcp(*args, **kwargs)).reshape(
        dcp_size, Q_LEN, K
    )
    assert local_topk.max() < KV_LEN // dcp_size, "a rank got a foreign index"

    for t in range(Q_LEN):
        got = set()
        for r in range(dcp_size):
            row = local_topk[r, t]
            for local in row[row >= 0].tolist():
                got.add(int(cp_local_to_global(local, r, dcp_size, 1)))
        want = set(reference[t][reference[t] >= 0].tolist())
        assert got == want, (
            f"token {t}: {len(want - got)} missing, {len(got - want)} spurious"
        )


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("chunk_tokens", [None, 64])
def test_a2a_merge_matches_the_all_gather_merge(dcp_size, chunk_tokens):
    """Slot for slot, not just as a set: the packed order is the contract."""
    _need_devices(dcp_size)
    _, args, kwargs = _dcp_case(dcp_size, 5)
    kwargs["chunk_tokens"] = chunk_tokens

    # Both paths carry cp_rank_as_data's all-to-all; the a2a merge adds exactly
    # one. The default is what a bucket below TPU_DCP_INDEXER_A2A_MIN_TOKENS runs.
    def num_a2a(**kw):
        hlo = streamindex_topk_dcp.lower(*args, **kwargs, **kw).as_text()
        return len(re.findall(r"all[_-]to[_-]all", hlo))

    n_default, n_a2a = num_a2a(), num_a2a(a2a_merge=True)
    assert n_a2a == n_default + 1, (
        f"all-to-alls: default {n_default}, a2a {n_a2a}; the default merge paid"
        " the all-to-all, or the a2a path ran the all-gather merge"
    )

    gather = np.asarray(streamindex_topk_dcp(*args, **kwargs))
    a2a = np.asarray(streamindex_topk_dcp(*args, **kwargs, a2a_merge=True))
    np.testing.assert_array_equal(a2a, gather)


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_all_to_all_round_trip_preserves_token_order(dcp_size):
    """The a2a scatter and the all-gather restore must agree on block order;
    if they do not, every rank merges a full, plausible, wrong token."""
    _need_devices(dcp_size)
    rows, width = 4 * dcp_size, 8
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp_size]), (DCP_AXIS_NAME,))

    def _local(token_ids):
        # Every rank starts with the same [rows, width] of token ids.
        scattered = jax.lax.all_to_all(
            token_ids, DCP_AXIS_NAME, split_axis=0, concat_axis=1, tiled=True
        )
        # [rows/dcp, dcp * width]: every peer sent this rank the same block.
        blocks = scattered.reshape(-1, dcp_size, width)
        return jax.lax.all_gather(blocks[:, 0, :], DCP_AXIS_NAME, axis=0, tiled=True)

    token_ids = jnp.broadcast_to(
        jnp.arange(rows, dtype=jnp.int32)[:, None], (rows, width)
    )
    out = jax.jit(
        jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=(P(),),
            out_specs=P(DCP_AXIS_NAME),
            check_vma=False,
        )
    )(token_ids)

    got = np.asarray(out).reshape(dcp_size, rows, width)
    for r in range(dcp_size):
        np.testing.assert_array_equal(
            got[r],
            np.asarray(token_ids),
            err_msg=f"rank {r} came back with the token axis permuted",
        )


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("chunk_tokens", [32, 64, 128])
def test_dcp_chunking_does_not_change_the_result(chunk_tokens, dcp_size):
    _need_devices(dcp_size)
    _, args, kwargs = _dcp_case(dcp_size, 11)

    hlo = streamindex_topk_dcp.lower(
        *args, **kwargs, chunk_tokens=chunk_tokens
    ).as_text()
    assert "_scheduling_group_id" in hlo, (
        "no scheduling group annotation: the DCP path ran a single unchunked"
        " pass, so this comparison would pass trivially"
    )

    unchunked = np.asarray(streamindex_topk_dcp(*args, **kwargs))
    chunked = np.asarray(
        streamindex_topk_dcp(*args, **kwargs, chunk_tokens=chunk_tokens)
    )
    np.testing.assert_array_equal(chunked, unchunked)


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_dcp_append_and_topk_match_across_xla_has_and_sal(dcp_size):
    """DCP append + streamindex_topk_dcp produces identical local_topk across
    XLA HAS, Pallas HAS, and Pallas SAL."""
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as st_lib
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        KVLayout,
        coalesced_kv_append_dcp,
        get_dcp_page_and_slot,
    )

    _need_devices(dcp_size)
    _, args, kwargs = _dcp_case(dcp_size, 23)
    q, weights, shards_has, seq_lens, page_indices, cu_q_lens, distribution = args
    mesh = kwargs["mesh"]
    interleave_size = kwargs["interleave_size"]
    vpages = shards_has.shape[0] // dcp_size

    # New step tokens (129 bytes = 128 fp8 + 1 e8m0 scale) to append over the
    # last Q_LEN positions of the sequence.
    step_rec = _global_records(Q_LEN, seed=101)[:, : D_IDX + 1]

    sharded = jax.sharding.PartitionSpec(DCP_AXIS_NAME)
    replicated = jax.sharding.PartitionSpec()

    def _xla_dcp_append(cache_shard, k_packed, seq_lens_a, pages_a, cu_q_a):
        rank = cp_rank_as_data(DCP_AXIS_NAME, dcp_size)
        page, slot = get_dcp_page_and_slot(
            Q_LEN,
            seq_lens_a,
            pages_a,
            cu_q_a,
            rank,
            page_size=PAGE_SIZE,
            num_pages=vpages,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
        )
        k_pad = jnp.pad(k_packed, ((0, 0), (0, WIDTH - k_packed.shape[-1])))
        return cache_shard.at[page, slot // 4, slot % 4].set(k_pad)

    def _pallas_dcp_append(layout):
        def _fn(cache_shard, k_packed, seq_lens_a, pages_a, cu_q_a):
            rank = cp_rank_as_data(DCP_AXIS_NAME, dcp_size)
            return coalesced_kv_append_dcp(
                cache_shard,
                k_packed,
                seq_lens_a,
                pages_a,
                cu_q_a,
                rank,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
                kv_layout=layout,
            )

        return _fn

    in_specs = (sharded, replicated, replicated, replicated, replicated)
    cache_xla = jax.jit(
        jax.shard_map(
            _xla_dcp_append,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=sharded,
            check_vma=False,
        )
    )(shards_has, step_rec, seq_lens, page_indices, cu_q_lens)

    cache_has = jax.jit(
        jax.shard_map(
            _pallas_dcp_append(KVLayout.HEAD_ALONG_SUBLANE),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=sharded,
            check_vma=False,
        )
    )(shards_has, step_rec, seq_lens, page_indices, cu_q_lens)

    shards_sal_host = jnp.concatenate(
        [
            st_lib.convert_cache_to_seq_along_lane(
                np.asarray(shards_has).reshape(
                    dcp_size, vpages, PAGE_SIZE // 4, 4, WIDTH
                )[r],
                actual_head_dim=D_IDX,
            )
            for r in range(dcp_size)
        ],
        axis=0,
    )
    shards_sal = jax.device_put(
        shards_sal_host, jax.sharding.NamedSharding(mesh, sharded)
    )
    cache_sal = jax.jit(
        jax.shard_map(
            _pallas_dcp_append(KVLayout.SEQ_ALONG_LANE),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=sharded,
            check_vma=False,
        )
    )(shards_sal, step_rec, seq_lens, page_indices, cu_q_lens)

    np.testing.assert_array_equal(np.asarray(cache_has), np.asarray(cache_xla))

    topk_xla = np.asarray(
        streamindex_topk_dcp(
            q,
            weights,
            cache_xla,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            **kwargs,
        )
    )
    topk_has = np.asarray(
        streamindex_topk_dcp(
            q,
            weights,
            cache_has,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            **kwargs,
        )
    )
    topk_sal = np.asarray(
        streamindex_topk_dcp(
            q,
            weights,
            cache_sal,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            kv_layout=KVLayout.SEQ_ALONG_LANE,
            **kwargs,
        )
    )

    np.testing.assert_array_equal(topk_has, topk_xla)
    np.testing.assert_array_equal(topk_sal, topk_xla)


# ===================================================================== #
# overlap_fn (TPU_INDEXER_QB_OVERLAP): independent TC work pinned under the
# SparseCore top-k calls, exact.
# ===================================================================== #


def _qb_overlap_operands(seed=3, m=16, k=2048, n=16384):
    """The MLA q_b projection's operands at GLM-5.2's served shape."""
    from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

    kx, kw, ks = jax.random.split(jax.random.key(seed), 3)
    x = (jax.random.normal(kx, (m, k), jnp.float32) * 0.5).astype(jnp.bfloat16)
    w = (jax.random.normal(kw, (k, n), jnp.float32) * 0.05).astype(jnp.float8_e4m3fn)
    s = jax.random.uniform(ks, (n,), jnp.float32, 0.5, 1.5)
    # Decode rows take the skinny Pallas kernel, wider ones the XLA matmul.
    assert fp8_skinny.supports(m, w, s) == (m <= 16)
    return x, w, s


def _entry_order(compiled_text, names):
    """Positions of the first scheduled instruction whose name starts with
    each of `names`, in the compiled ENTRY computation."""
    entry = compiled_text[compiled_text.rfind("ENTRY") :]
    order = {}
    for i, line in enumerate(entry.splitlines()):
        m = re.match(r"\s*(?:ROOT )?%?([\w.-]+) = ", line)
        if not m:
            continue
        for n in names:
            if n not in order and m.group(1).startswith(n):
                order[n] = i
    return order


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 8])
@pytest.mark.parametrize("q_len", [16, 256], ids=["decode16", "mixed256"])
def test_overlap_fn_is_exact(dcp_size, q_len):
    """The overlapped q_b and the top-k both match running them apart, byte
    for byte, and the positive control (one weight byte changed) is caught."""
    from vllm_torchtpu.layers.adapter.linear_common import _fp8_linear_pallas_jax

    _need_devices(dcp_size)
    _require_sparsecore()
    _, args, kwargs = _dcp_case(dcp_size, 31, q_len=q_len)
    x, w, s = _qb_overlap_operands(m=q_len)

    want_topk = np.asarray(streamindex_topk_dcp(*args, **kwargs))
    want_q = np.asarray(jax.jit(_fp8_linear_pallas_jax)(x, w, s))

    ov = dict(overlap_fn=_fp8_linear_pallas_jax, overlap_args=(x, w, s))
    got_topk, got_q = streamindex_topk_dcp(*args, **kwargs, **ov)
    np.testing.assert_array_equal(np.asarray(got_topk), want_topk)
    np.testing.assert_array_equal(
        np.asarray(got_q).view(np.uint16), want_q.view(np.uint16)
    )

    # The model's setting: pin decode buckets only (mixed256 runs unpinned).
    got_topk, got_q = streamindex_topk_dcp(
        *args, **kwargs, **ov, overlap_pin_max_tokens=16
    )
    np.testing.assert_array_equal(np.asarray(got_topk), want_topk)
    np.testing.assert_array_equal(
        np.asarray(got_q).view(np.uint16), want_q.view(np.uint16)
    )

    w_bad = w.at[0, 0].set(jnp.float8_e4m3fn(3.0))
    _, bad_q = streamindex_topk_dcp(
        *args, **kwargs, **{**ov, "overlap_args": (x, w_bad, s)}
    )
    assert (np.asarray(bad_q).view(np.uint16) != want_q.view(np.uint16)).any(), (
        "a changed q_b weight byte left q bit-identical: the comparison is blind"
    )


@pytest.mark.multichip
def test_overlap_fn_is_scheduled_under_the_sparsecore_calls():
    """Decode bucket: q_b lands between the local top-k's call-update and its
    call-done, and the merge stage's work between the merge's update and done
    -- the windows the TensorCore otherwise idles in."""
    from vllm_torchtpu.layers.adapter.linear_common import _fp8_linear_pallas_jax

    _need_devices(2)
    _require_sparsecore()
    _, args, kwargs = _dcp_case(2, 41, q_len=16)
    x, w, s = _qb_overlap_operands(m=16)
    w_uk = jax.random.normal(jax.random.key(9), (64, 192, 512), jnp.bfloat16)

    def absorb(q):
        return jnp.einsum("thd,hdl->thl", q.reshape(16, 64, 256)[..., :192], w_uk)

    text = (
        streamindex_topk_dcp.lower(
            *args,
            **kwargs,
            overlap_fn=_fp8_linear_pallas_jax,
            overlap_args=(x, w, s),
            merge_overlap_fn=absorb,
        )
        .compile()
        .as_text()
    )
    local = "sc_topk_direct_b16_n"
    pos = _entry_order(
        text,
        [
            "fp8_skinny",
            "all_gather",
            "all-gather",
        ],
    )
    entry = text[text.rfind("ENTRY") :].splitlines()
    sc = [
        (i, ln.split("=")[0].strip().lstrip("%"))
        for i, ln in enumerate(entry)
        if local in ln.split("=")[0]
    ]
    updates = [i for i, n in sc if "update" in n]
    dones = [i for i, n in sc if n.endswith("call-done") or ".call-done" in n]
    assert len(updates) >= 2 and len(dones) >= 2, sc
    assert updates[0] < pos["fp8_skinny"] < dones[0], (sc, pos)
    convs = [
        i
        for i, ln in enumerate(entry)
        if re.search(r"= bf16\[(16,64|64,16),512\]", ln) and "fusion" in ln
    ]
    assert any(updates[1] < i < dones[1] for i in convs), (sc, convs)


@pytest.mark.multichip
def test_merge_overlap_fn_is_exact():
    """`merge_overlap_fn` sees `overlap_fn`'s output; both stages exact."""
    _need_devices(2)
    _require_sparsecore()
    _, args, kwargs = _dcp_case(2, 37, q_len=16)
    x = jax.random.normal(jax.random.key(5), (16, 512), jnp.float32)

    def first(v):
        return v * 2.0

    def second(v):
        return jnp.tanh(v) + 1.0

    want = np.asarray(jax.jit(lambda v: second(first(v)))(x))
    topk, got = streamindex_topk_dcp(
        *args, **kwargs, overlap_fn=first, overlap_args=(x,), merge_overlap_fn=second
    )
    np.testing.assert_array_equal(np.asarray(got), want)
    np.testing.assert_array_equal(
        np.asarray(topk), np.asarray(streamindex_topk_dcp(*args, **kwargs))
    )

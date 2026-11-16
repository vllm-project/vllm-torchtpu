# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_COMPRESSED_DCP_PACKED_GATHER: one all-gather of the packed
local top-k lists gives the lists that one all-gather each gives.

``qsa_op._gathered_lists`` runs on every simulated rank at once, the ranks
meeting in a threaded all-gather that knows nothing of the packing. Scores
compare as raw bits and group ids exactly, against each other and against the
lists taken straight from the ranks' inputs. Covered: the decode pair group
(world = DCP size), the sequence-parallel TP group (two token shards), rows
past the step's tokens, padding rows with no candidate, rows with fewer than
k candidates, boundary ties, and score bit patterns the indexer never writes
(NaN payloads, -0.0, subnormals, infinities). The merged selections built from
both gathers are equal, and padding rows select nothing.
"""

import threading
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import dcp_select, pooled_qsa
from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

FLAG = "VLLM_TPU_QSA_COMPRESSED_DCP_PACKED_GATHER"
K = 512  # token_topk // compress_ratio of the served model
GPP = 32  # compressed groups per main-KV page (page 128, ratio 4): the ownership unit
GROUPS = 24576  # a 98304-token context at ratio 4


def lax_topk(bits, k):
    neg_inf = jax.lax.bitcast_convert_type(jnp.float32(-jnp.inf), jnp.int32)
    slots = jax.lax.top_k(dcp_select.monotone_key(bits), k)[1].astype(jnp.int32)
    filled = jnp.take_along_axis(bits, slots, axis=1) != neg_inf
    return jnp.where(filled, slots, -1)


class _Member:
    """One rank's view of a threaded all-gather over ``world`` ranks."""

    def __init__(self, shared, rank):
        self.shared, self.rank, self.calls = shared, rank, 0

    def all_gather(self, x, dim=0):
        assert dim == 0
        slots, barrier, world = self.shared
        key = self.calls
        self.calls += 1
        slots[(key, self.rank)] = x.clone()
        barrier.wait(timeout=60)
        return torch.cat([slots[(key, r)] for r in range(world)], dim=0)


def run_ranks(world, fn):
    """``fn(rank, group)`` on ``world`` threads; returns (results, all-gathers per rank)."""
    shared = ({}, threading.Barrier(world), world)
    members = [_Member(shared, r) for r in range(world)]
    out, errors = [None] * world, []

    def body(r):
        try:
            out[r] = fn(r, members[r])
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)
            shared[1].abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return out, [m.calls for m in members]


def served_lists(rng, size, rank, rows, padding_rows):
    """A DCP rank's local top-k: owned group ids (page ownership), scores
    quantized so that many tie at the boundary (+0.0 and -0.0 among them),
    valid entries first in a shuffled order, ``EMPTY`` / -inf after. Rows in
    ``padding_rows`` have no candidate; some others have fewer than k."""
    d = rank % size
    owned = np.arange(GROUPS)[(np.arange(GROUPS) // GPP) % size == d]
    vals = np.full((rows, K), -np.inf, np.float32)
    pos = np.full((rows, K), dcp_select.EMPTY, np.int32)
    for t in range(rows):
        if t in padding_rows:
            continue
        count = K if rng.random() < 0.7 else int(rng.integers(1, K))
        ids = rng.choice(owned, size=count, replace=False)
        v = np.floor(rng.standard_normal(count) * 3).astype(np.float32) / 4
        v[rng.random(count) < 0.05] = -0.0
        vals[t, :count], pos[t, :count] = v, ids
    return vals, pos


def bit_lists(rng, rows):
    """Arbitrary words: every f32 bit pattern class and every int32."""
    words = rng.integers(-(2**31), 2**31, size=(rows, K), dtype=np.int64).astype(np.int32)
    special = np.array([0x7FC00000, 0x7FA00001, 0xFFFFFFFF, 0x80000000, 0x00000001,
                        0x807FFFFF, 0x7F800000, 0xFF800000], dtype=np.uint32).view(np.int32)
    words[:, : len(special)] = special
    pos = rng.integers(-(2**31), 2**31, size=(rows, K), dtype=np.int64).astype(np.int32)
    pos[:, :3] = [dcp_select.EMPTY, -1, -(2**31)]
    return words.view(np.float32), pos


def make_inputs(kind, world, size, rows, seed):
    rng = np.random.default_rng(seed)
    padding_rows = {rows - 1, rows // 2} if rows > 2 else set()
    lists = []
    for r in range(world):
        if kind == "served":
            lists.append(served_lists(rng, size, r, rows, padding_rows))
        else:
            lists.append(bit_lists(rng, rows))
    return lists, padding_rows


def expected_lists(lists, world, size, tokens):
    """``[size, tokens, K]``: DCP rank d's rows of every token shard, in shard order."""
    vals = np.stack([np.concatenate([lists[p * size + d][0] for p in range(world // size)])
                     for d in range(size)])[:, :tokens]
    pos = np.stack([np.concatenate([lists[p * size + d][1] for p in range(world // size)])
                    for d in range(size)])[:, :tokens]
    return vals, pos


def gather_all(monkeypatch, packed, lists, world, size, tokens):
    monkeypatch.setattr(envs, FLAG, packed, raising=False)

    def one(rank, group):
        vals, pos = (torch.from_numpy(x.copy()) for x in lists[rank])
        return qsa_op._gathered_lists(vals, pos, group, size, tokens)

    return run_ranks(world, one)


def as_bits(x):
    return x.contiguous().view(torch.int32).numpy()


# (world, size, rows, tokens): the decode pair group (world = DCP size), with and
# without rows past the tokens, DCP2, and the SP graph's TP group of 8 holding
# two token shards of the four DCP ranks.
CASES = {
    "pair4": (4, 4, 48, 48),
    "pair4_cut": (4, 4, 48, 45),
    "pair2": (2, 2, 24, 24),
    "tp8_shards": (8, 4, 16, 32),
    "tp8_shards_cut": (8, 4, 16, 29),
}


@pytest.mark.parametrize("kind", ["served", "bits"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_packed_gather_matches_two_gathers(monkeypatch, case, kind):
    world, size, rows, tokens = CASES[case]
    lists, padding_rows = make_inputs(kind, world, size, rows, seed=zlib.crc32(f"{case}/{kind}".encode()))
    want_vals, want_pos = expected_lists(lists, world, size, tokens)
    two, calls_two = gather_all(monkeypatch, False, lists, world, size, tokens)
    one, calls_one = gather_all(monkeypatch, True, lists, world, size, tokens)
    assert calls_two == [2] * world and calls_one == [1] * world
    for r in range(world):
        (v2, p2), (v1, p1) = two[r], one[r]
        assert v1.dtype == v2.dtype == torch.float32 and p1.dtype == p2.dtype == torch.int32
        assert tuple(v1.shape) == tuple(v2.shape) == tuple(p1.shape) == (size, tokens, K)
        np.testing.assert_array_equal(as_bits(v1), as_bits(v2), f"rank {r} score bits")
        np.testing.assert_array_equal(p1.numpy(), p2.numpy(), f"rank {r} group ids")
        np.testing.assert_array_equal(as_bits(v1), want_vals.view(np.int32), f"rank {r}")
        np.testing.assert_array_equal(p1.numpy(), want_pos, f"rank {r}")

    if kind != "served":
        return
    # The selections merged from both gathers, global and per rank (local
    # merge); padding rows select nothing.
    monkeypatch.setattr(dcp_select, "_sc_topk", lax_topk)
    real = sorted(set(range(tokens)) - {t for t in range(tokens)
                                        if t % rows in padding_rows})
    for r in range(size):
        (v2, p2), (v1, p1) = two[r], one[r]
        blocks2 = pooled_qsa.compressed_dcp_blocks(jnp.asarray(v2.numpy()),
                                                  jnp.asarray(p2.numpy()), K)
        blocks1 = pooled_qsa.compressed_dcp_blocks(jnp.asarray(v1.numpy()),
                                                  jnp.asarray(p1.numpy()), K)
        np.testing.assert_array_equal(np.asarray(blocks1), np.asarray(blocks2))
        rank = jnp.asarray([r], jnp.int32)
        own2 = pooled_qsa.compressed_dcp_own_blocks(jnp.asarray(v2.numpy()),
                                                   jnp.asarray(p2.numpy()), rank, K)
        own1 = pooled_qsa.compressed_dcp_own_blocks(jnp.asarray(v1.numpy()),
                                                   jnp.asarray(p1.numpy()), rank, K)
        np.testing.assert_array_equal(np.asarray(own1), np.asarray(own2))
        blocks1, own1 = np.asarray(blocks1), np.asarray(own1)
        pad = [t for t in range(tokens) if t % rows in padding_rows]
        assert (blocks1[pad] == -1).all() and (own1[pad] == -1).all()
        assert (blocks1[real] >= 0).any(axis=1).all()

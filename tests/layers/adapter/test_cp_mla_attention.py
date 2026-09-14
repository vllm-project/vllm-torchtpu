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
"""The MLA DCP head gather and log-sum-exp merge (`cp_mla_attention`), the
position-ownership rule the indexer's return leg and the KV write agree on, and
the kernel `return_lse` flag the merge is built on."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from absl.testing import parameterized

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    cp_global_to_local, cp_owner_rank)
from vllm_torchtpu.kernels.mla.kv_cache_utils import (KVCacheLayout,
                                                      KVCacheType,
                                                      SparseMLAKVCacheSpec)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla
from vllm_torchtpu.layers.core import attention_interface


class _FakeGroup:
    """Serves `all_gather`/`reduce_scatter` from a full stack of rank partials"""

    def __init__(self, outputs, lses, rank):
        self.world_size = len(lses)
        self.rank_in_group = rank
        self._outputs = outputs  # [D, T, N, V] f32
        self._lses = lses  # [D, T, N] f32

    def all_gather(self, tensor, dim=0):
        assert dim == 0 and tensor.shape[0] == 1
        return self._lses.clone()

    def _reduced(self, dtype):
        total = None
        for rank in range(self.world_size):
            peer = _FakeGroup(self._outputs, self._lses, rank)
            term = cp_mla._weight_by_lse(self._outputs[rank], self._lses[rank],
                                         peer)
            total = term if total is None else total + term
        return total.to(dtype)

    def reduce_scatter(self, tensor, dim=-1):
        total = self._reduced(tensor.dtype)
        width = total.shape[dim] // self.world_size
        return total.narrow(dim, self.rank_in_group * width, width)


def _merge_as_rank(outputs, lses, rank, monkeypatch):
    """The merge as rank `r` sees it: its own head block of the whole answer"""
    group = _FakeGroup(outputs, lses, rank)
    monkeypatch.setattr(cp_mla, "get_dcp_group", lambda: group)
    return cp_mla.merge_lse_partials_scatter_heads(outputs[rank], lses[rank])


def _merge_whole(outputs, lses, monkeypatch):
    """The full merged tensor, reassembled from every rank's scatter block"""
    return torch.cat([
        _merge_as_rank(outputs, lses, rank, monkeypatch)
        for rank in range(outputs.shape[0])
    ],
                     dim=1)


def _reference_merge(outputs, lses):
    """`o = sum_r softmax_r(lse) * o_r` in f64 numpy"""
    out = outputs.numpy().astype(np.float64)
    lse = lses.numpy().astype(np.float64)
    shift = np.where(np.isfinite(lse), lse, -np.inf).max(axis=0)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    weights = np.where(np.isfinite(lse), np.exp(lse - shift[None]), 0.0)
    denom = weights.sum(axis=0)
    weights = weights / np.where(denom > 0.0, denom, 1.0)
    return (out * weights[..., None]).sum(axis=0)


def _partials(rng, dcp_size, tokens=8, local_heads=2, vdim=16, empty=()):
    """Per-rank partials"""
    heads = dcp_size * local_heads
    outputs = torch.from_numpy(
        rng.standard_normal(
            (dcp_size, tokens, heads, vdim)).astype(np.float32))
    lses = torch.from_numpy((rng.standard_normal(
        (dcp_size, tokens, heads)) * 4).astype(np.float32))
    for rank, token in empty:
        lses[rank, token] = float("-inf")
    return outputs, lses


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
def test_torch_merge_matches_the_reference(dcp_size, monkeypatch):
    outputs, lses = _partials(np.random.default_rng(20260907), dcp_size)
    got = _merge_whole(outputs, lses, monkeypatch).numpy()
    np.testing.assert_allclose(got,
                               _reference_merge(outputs, lses),
                               rtol=1e-5,
                               atol=1e-5)


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_each_rank_gets_its_own_block_of_one_answer(rank, monkeypatch):
    """The rank index must only select which head block comes back -- never
    change the arithmetic that produced it."""
    local_heads = 3
    outputs, lses = _partials(np.random.default_rng(11),
                              4,
                              local_heads=local_heads)
    got = _merge_as_rank(outputs, lses, rank, monkeypatch).numpy()
    reference = _reference_merge(outputs, lses)
    np.testing.assert_allclose(got,
                               reference[:, rank * local_heads:(rank + 1) *
                                         local_heads],
                               rtol=1e-5,
                               atol=1e-5)


def test_ranks_owning_nothing_are_excluded(monkeypatch):
    """A `-inf` rank must contribute nothing -- not a small weight."""
    rng = np.random.default_rng(5)
    outputs, lses = _partials(rng, 4, empty=[(1, 3), (2, 3), (3, 3)])
    got = _merge_whole(outputs, lses, monkeypatch).numpy()
    # Token 3 is owned by rank 0 alone, so the merge must return rank 0's
    # output there verbatim.
    np.testing.assert_allclose(got[3],
                               outputs[0, 3].numpy(),
                               rtol=1e-6,
                               atol=1e-6)


def test_a_token_no_rank_owns_is_zero_not_nan(monkeypatch):
    """Padding rows are `-inf` everywhere; `inf - inf` must not leak out."""
    rng = np.random.default_rng(6)
    outputs, lses = _partials(rng, 4, empty=[(r, 7) for r in range(4)])
    got = _merge_whole(outputs, lses, monkeypatch).numpy()
    assert np.isfinite(got).all()
    np.testing.assert_array_equal(got[7], np.zeros_like(got[7]))


class _HeadGatherGroup:
    """Serves only `all_gather(dim=1)`, from a stack of per-rank head slices."""

    def __init__(self, per_rank, rank):
        self.world_size = len(per_rank)
        self.rank_in_group = rank
        self._per_rank = per_rank

    def all_gather(self, tensor, dim=0):
        assert dim == 1
        assert tensor.shape == self._per_rank[self.rank_in_group].shape
        return torch.cat(list(self._per_rank), dim=1)


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_head_gather_puts_this_ranks_heads_in_its_own_block(rank, monkeypatch):
    """The gather/scatter pair only lines up if block `r` belongs to rank `r`.

    `merge_lse_partials_scatter_heads` hands block `r` back to rank `r`, and
    `W_UV` downstream is that rank's TP head slice, so anything else silently
    applies the wrong up-projection.
    """
    rng = np.random.default_rng(31)
    per_rank = torch.from_numpy(
        rng.standard_normal((4, 6, 3, 8)).astype(np.float32))
    monkeypatch.setattr(cp_mla, "get_dcp_group",
                        lambda: _HeadGatherGroup(per_rank, rank))

    got = cp_mla.all_gather_heads(per_rank[rank])
    assert got.shape == (6, 12, 8)
    np.testing.assert_array_equal(got[:, rank * 3:(rank + 1) * 3].numpy(),
                                  per_rank[rank].numpy())


def test_head_gather_is_a_noop_without_a_group(monkeypatch):
    monkeypatch.setattr(cp_mla, "get_dcp_group", lambda: None)
    q = torch.ones(2, 3, 4)
    assert cp_mla.all_gather_heads(q) is q


def test_scatter_merge_is_a_noop_without_a_group(monkeypatch):
    monkeypatch.setattr(cp_mla, "get_dcp_group", lambda: None)
    out = torch.ones(2, 3, 4)
    assert cp_mla.merge_lse_partials_scatter_heads(out, torch.zeros(2,
                                                                    3)) is out


def test_merge_reproduces_an_unsplit_softmax(monkeypatch):
    rng = np.random.default_rng(99)
    dcp_size, tokens, local_heads, vdim, keys = 4, 6, 3, 8, 40
    heads = dcp_size * local_heads
    q = rng.standard_normal((tokens, heads, vdim))
    k = rng.standard_normal((keys, vdim))
    v = rng.standard_normal((keys, vdim))
    scores = np.einsum("tnd,kd->tnk", q, k)

    whole = np.exp(scores - scores.max(-1, keepdims=True))
    whole /= whole.sum(-1, keepdims=True)
    expected = np.einsum("tnk,kd->tnd", whole, v)

    outs = np.zeros((dcp_size, tokens, heads, vdim))
    lses = np.zeros((dcp_size, tokens, heads))
    for r in range(dcp_size):
        s = scores[..., r::dcp_size]
        m = s.max(-1)
        e = np.exp(s - m[..., None])
        lses[r] = m + np.log(e.sum(-1))
        outs[r] = np.einsum("tnk,kd->tnd", e / e.sum(-1, keepdims=True),
                            v[r::dcp_size])

    got = _merge_whole(
        torch.from_numpy(outs).float(),
        torch.from_numpy(lses).float(), monkeypatch).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# ===================================================================== #
# Position ownership tests.
# ===================================================================== #

_OWNERSHIP = [(d, c) for d in (2, 4, 8, 16) for c in (1, 4, 64)]


@pytest.mark.cpu_test
@pytest.mark.parametrize("dcp_size,interleave_c", _OWNERSHIP)
def test_ownership_is_a_partition_of_the_positions(dcp_size, interleave_c):
    """Every position has exactly one owner, and every rank is an owner."""
    n = 5 * dcp_size * interleave_c + 3
    owners = np.asarray(
        cp_owner_rank(jnp.arange(n, dtype=jnp.int32), dcp_size, interleave_c))
    assert owners.min() >= 0 and owners.max() < dcp_size
    assert set(owners.tolist()) == set(range(dcp_size))


@pytest.mark.cpu_test
@pytest.mark.parametrize("dcp_size,interleave_c", _OWNERSHIP)
def test_owner_counts_are_balanced_over_a_whole_cycle(dcp_size, interleave_c):
    """Over a whole interleave cycle the shards are exactly equal.

    Which is why mean per-rank occupancy of a `k`-wide top-k is `k / dcp_size`
    as an identity, not an estimate -- the spread around it comes only from
    where the winners cluster inside the cycle.
    """
    cycle = dcp_size * interleave_c
    owners = np.asarray(
        cp_owner_rank(jnp.arange(3 * cycle, dtype=jnp.int32), dcp_size,
                      interleave_c))
    counts = np.bincount(owners, minlength=dcp_size)
    assert (counts == 3 * interleave_c).all()


@pytest.mark.cpu_test
@pytest.mark.parametrize("dcp_size,interleave_c", _OWNERSHIP)
def test_local_indices_stay_inside_the_shard(dcp_size, interleave_c):
    """A rank is only ever handed an index into its own cache shard."""
    n = 5 * dcp_size * interleave_c
    gs = jnp.arange(n, dtype=jnp.int32)
    locals_ = np.asarray(cp_global_to_local(gs, dcp_size, interleave_c))
    assert locals_.min() >= 0
    assert locals_.max() < n // dcp_size


# =====================================================================
# `return_lse` on the sparse MLA kernel and the cross-shard merge it exists
# to enable.
# =====================================================================

LKV_DIM = 512
ROPE_DIM = 64
NUM_HEADS = 16
PAGE_SIZE = 32
PAGES_PER_SEQ = 4
TOTAL_PAGES = 16

KV_PACKING = sparse_mla_kernel.get_dtype_packing(jnp.float8_e4m3fn)
NOPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.NOPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, LKV_DIM, KV_PACKING)
ROPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.ROPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, ROPE_DIM, KV_PACKING)

DCP_TOKENS = 64  # multiple of TOKEN_PAD, fits PAGES_PER_SEQ * PAGE_SIZE
DCP_TOPK = 64
# The DCP shard rule from streamindex_topk.cp_local_to_global: compressed
# position `p` is owned by rank `(p // interleave_c) % dcp_size`. Anything
# below `kv_cache_utils.WORD_BYTES` would split a rope tile across ranks.
DCP_INTERLEAVE_C = 4


def _empty_pair():
    """Zeroed (nope, rope) caches in dsa_gather's native tiled layouts."""
    return (jnp.zeros(NOPE_SPEC.shape, NOPE_SPEC.jax_dtype),
            jnp.zeros(ROPE_SPEC.shape, ROPE_SPEC.jax_dtype))


def _quantize_fp8(x: np.ndarray, k_scale: float) -> jax.Array:
    return jnp.asarray(x / k_scale).astype(jnp.float8_e4m3fn)


def _dequantize(x: jax.Array, k_scale: float) -> np.ndarray:
    return np.asarray(x.astype(jnp.float32)) * k_scale


def _causal_topk(positions: list[int], topk: int) -> np.ndarray:
    """topk_indices row per token: [0..pos] then -1 padding."""
    rows = np.full((len(positions), topk), -1, np.int32)
    for i, pos in enumerate(positions):
        assert pos + 1 <= topk
        rows[i, :pos + 1] = np.arange(pos + 1)
    return rows


def _shard_topk_by_owner(topk_rows: np.ndarray, dcp_size: int,
                         interleave_c: int):
    """Partition each token's top-k list by the position's owning DCP rank"""
    num_tokens, topk = topk_rows.shape
    shards = np.full((dcp_size, num_tokens, topk), -1, np.int32)
    counts = np.zeros((dcp_size, num_tokens), np.int32)
    for t in range(num_tokens):
        row = topk_rows[t][topk_rows[t] >= 0]
        for r in range(dcp_size):
            sel = row[(row // interleave_c) % dcp_size == r]
            counts[r, t] = sel.size
            shards[r, t, :sel.size] = sel
            if sel.size == 0:
                shards[r, t, 0] = 0  # dummy; masked by `counts` at merge time
    return shards, counts


def _merge_partials(outs: np.ndarray, lses: np.ndarray,
                    counts: np.ndarray) -> np.ndarray:
    """`o = sum_r softmax_r(lse) * o_r` over the DCP axis, in f32 """
    lses = np.where(counts[:, :, None] > 0, lses, -np.inf)
    shift = lses.max(0, keepdims=True)
    # A token no shard owns would give 0/0; it cannot occur for a real top-k
    # list but the guard keeps the failure a zero rather than a NaN.
    all_empty = ~np.isfinite(shift)
    shift = np.where(all_empty, 0.0, shift)
    weights = np.exp(lses - shift[0])  # [D, T, N]
    denom = weights.sum(0)
    weights = np.where(denom[None] > 0,
                       weights / np.where(denom == 0, 1, denom)[None], 0.0)
    return (weights[..., None] * outs).sum(0)


class SparseMlaLseTest(parameterized.TestCase):
    """`return_lse` and the KV-sharded merge it exists to enable. Tolerances
    are set by the kernel's bf16 output, not the merge, which is exact in f32."""

    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(20260906)
        self.k_scale = 1.5
        self.sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
        if jax.devices()[0].platform != "tpu":
            self.skipTest("sparse MLA kernel requires a TPU (SparseCore).")

    def _prefilled(self):
        """One prefilled sequence: populated caches plus kernel-ready inputs."""
        kv_c = self.rng.standard_normal(
            (DCP_TOKENS, LKV_DIM)).astype(np.float32)
        k_pe = self.rng.standard_normal(
            (DCP_TOKENS, ROPE_DIM)).astype(np.float32)
        kv_c_fp8 = _quantize_fp8(kv_c, self.k_scale)
        k_pe_fp8 = _quantize_fp8(k_pe, self.k_scale)
        ql_nope = jnp.asarray(self.rng.standard_normal(
            (DCP_TOKENS, NUM_HEADS, LKV_DIM)).astype(np.float32),
                              dtype=jnp.bfloat16)
        q_pe = jnp.asarray(self.rng.standard_normal(
            (DCP_TOKENS, NUM_HEADS, ROPE_DIM)).astype(np.float32),
                           dtype=jnp.bfloat16)
        topk_rows = _causal_topk(list(range(DCP_TOKENS)), DCP_TOPK)
        block_tables = jnp.asarray(
            self.rng.permutation(TOTAL_PAGES)[:PAGES_PER_SEQ], dtype=jnp.int32)
        mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x", ))

        # Run the layer once purely to insert this step's rows into the caches;
        # every assertion below then drives the kernel directly, because
        # `return_lse` is a kernel-level flag.
        nope_cache, rope_cache, _ = attention_interface.sparse_mla_attention(
            ql_nope,
            q_pe,
            kv_c_fp8,
            k_pe_fp8,
            *_empty_pair(),
            jnp.asarray(topk_rows, jnp.int32),
            jnp.asarray([DCP_TOKENS], jnp.int32),
            block_tables,
            jnp.asarray([0, DCP_TOKENS], jnp.int32),
            jnp.asarray([0, 1, 1], jnp.int32),
            mesh,
            NOPE_SPEC,
            ROPE_SPEC,
            sm_scale=self.sm_scale,
            k_scale=self.k_scale)

        return dict(
            q=jnp.concatenate([ql_nope, q_pe], axis=-1),
            nope_cache=nope_cache,
            rope_cache=rope_cache,
            topk_rows=topk_rows,
            block_tables=block_tables,
            cu_q_lens=jnp.asarray([0, DCP_TOKENS], jnp.int32),
            distribution=jnp.asarray([0, 1, 1], jnp.int32),
            kv_c_deq=_dequantize(kv_c_fp8, self.k_scale),
            k_pe_deq=_dequantize(k_pe_fp8, self.k_scale),
        )

    def _attend(self, inp, topk_rows, return_lse):
        return sparse_mla_kernel.sparse_ragged_paged_attention(
            inp["q"],
            inp["nope_cache"],
            inp["rope_cache"],
            jnp.asarray(topk_rows, jnp.int32),
            inp["block_tables"],
            inp["cu_q_lens"],
            inp["distribution"],
            sm_scale=self.sm_scale,
            k_scale=self.k_scale,
            return_lse=return_lse,
        )

    def test_return_lse_does_not_perturb_the_output(self):
        """The flag is additive: the attention result must be bit-identical."""
        inp = self._prefilled()
        base = self._attend(inp, inp["topk_rows"], False)
        out, lse = self._attend(inp, inp["topk_rows"], True)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(base))
        self.assertEqual(lse.dtype, jnp.float32)
        self.assertEqual(lse.shape, (DCP_TOKENS, NUM_HEADS))

    def test_lse_matches_a_reference_logsumexp(self):
        inp = self._prefilled()
        _, lse = self._attend(inp, inp["topk_rows"], True)

        keys = np.concatenate([inp["kv_c_deq"], inp["k_pe_deq"]], -1)
        q = np.asarray(inp["q"].astype(jnp.float32))
        expected = np.zeros((DCP_TOKENS, NUM_HEADS), np.float32)
        for t in range(DCP_TOKENS):
            sel = inp["topk_rows"][t][inp["topk_rows"][t] >= 0]
            scores = q[t] @ keys[sel].T * self.sm_scale  # [N, n]
            m = scores.max(-1)
            expected[t] = m + np.log(np.exp(scores - m[:, None]).sum(-1))

        np.testing.assert_allclose(np.asarray(lse),
                                   expected,
                                   rtol=2e-2,
                                   atol=2e-2)

    @parameterized.named_parameters(
        dict(testcase_name="d2", dcp_size=2),
        dict(testcase_name="d4", dcp_size=4),
        dict(testcase_name="d8", dcp_size=8),
    )
    def test_merging_position_sharded_partials_reproduces_the_whole(
            self, dcp_size):
        """The core DCP claim, at the kernel boundary.

        Splitting a token's top-k across ranks by KV *position* and merging the
        per-rank `(output, lse)` pairs must reproduce the unsplit result. With
        `_causal_topk` the early tokens own fewer entries than there are ranks,
        so the empty-shard path is exercised too.
        """
        inp = self._prefilled()
        whole = np.asarray(
            self._attend(inp, inp["topk_rows"], False).astype(jnp.float32))

        shards, counts = _shard_topk_by_owner(inp["topk_rows"], dcp_size,
                                              DCP_INTERLEAVE_C)
        self.assertTrue(
            (counts.sum(0) == (inp["topk_rows"] >= 0).sum(-1)).all(),
            "the shard split must partition the top-k list exactly")

        outs, lses = [], []
        for r in range(dcp_size):
            out_r, lse_r = self._attend(inp, shards[r], True)
            outs.append(np.asarray(out_r.astype(jnp.float32)))
            lses.append(np.asarray(lse_r))

        merged = _merge_partials(np.stack(outs), np.stack(lses), counts)
        np.testing.assert_allclose(merged, whole, rtol=2e-2, atol=2e-2)

    def test_an_empty_shard_contributes_nothing(self):
        """A rank owning none of a token's top-k must not shift the merge.

        Its `lse` is meaningless (the kernel was handed a dummy entry), so the
        merge has to drop it on `counts`, not on the value of `lse`.
        """
        inp = self._prefilled()
        whole = np.asarray(
            self._attend(inp, inp["topk_rows"], False).astype(jnp.float32))
        out, lse = self._attend(inp, inp["topk_rows"], True)

        # Two shards: one holds the entire list, the other holds nothing.
        empty_rows = np.full_like(inp["topk_rows"], -1)
        empty_rows[:, 0] = 0
        out_e, lse_e = self._attend(inp, empty_rows, True)
        counts = np.stack([(inp["topk_rows"] >= 0).sum(-1),
                           np.zeros(DCP_TOKENS, np.int32)])

        merged = _merge_partials(
            np.stack([
                np.asarray(out.astype(jnp.float32)),
                np.asarray(out_e.astype(jnp.float32))
            ]), np.stack([np.asarray(lse), np.asarray(lse_e)]), counts)
        np.testing.assert_allclose(merged, whole, rtol=2e-2, atol=2e-2)

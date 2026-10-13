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

import concurrent.futures
import math
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from absl.testing import parameterized

from vllm_torchtpu.distributed import dcp_comm
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    cp_global_to_local,
    cp_owner_rank,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
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
            peer.reduce_scatter = lambda tensor, dim: tensor
            term = dcp_comm.cp_lse_ag_out_rs(
                self._outputs[rank], self._lses[rank], peer
            )
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
    # The reduce-scatter merge; the all-to-all one (on by default) is below.
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A", False, raising=False)
    return cp_mla.merge_lse_partials_scatter_heads(outputs[rank], lses[rank])


def _merge_whole(outputs, lses, monkeypatch):
    """The full merged tensor, reassembled from every rank's scatter block"""
    return torch.cat(
        [
            _merge_as_rank(outputs, lses, rank, monkeypatch)
            for rank in range(outputs.shape[0])
        ],
        dim=1,
    )


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
        rng.standard_normal((dcp_size, tokens, heads, vdim)).astype(np.float32)
    )
    lses = torch.from_numpy(
        (rng.standard_normal((dcp_size, tokens, heads)) * 4).astype(np.float32)
    )
    for rank, token in empty:
        lses[rank, token] = float("-inf")
    return outputs, lses


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
def test_torch_merge_matches_the_reference(dcp_size, monkeypatch):
    outputs, lses = _partials(np.random.default_rng(20260907), dcp_size)
    got = _merge_whole(outputs, lses, monkeypatch).numpy()
    np.testing.assert_allclose(
        got, _reference_merge(outputs, lses), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_each_rank_gets_its_own_block_of_one_answer(rank, monkeypatch):
    """The rank index must only select which head block comes back -- never
    change the arithmetic that produced it."""
    local_heads = 3
    outputs, lses = _partials(np.random.default_rng(11), 4, local_heads=local_heads)
    got = _merge_as_rank(outputs, lses, rank, monkeypatch).numpy()
    reference = _reference_merge(outputs, lses)
    np.testing.assert_allclose(
        got,
        reference[:, rank * local_heads : (rank + 1) * local_heads],
        rtol=1e-5,
        atol=1e-5,
    )


def test_ranks_owning_nothing_are_excluded(monkeypatch):
    """A `-inf` rank must contribute nothing -- not a small weight."""
    rng = np.random.default_rng(5)
    outputs, lses = _partials(rng, 4, empty=[(1, 3), (2, 3), (3, 3)])
    got = _merge_whole(outputs, lses, monkeypatch).numpy()
    # Token 3 is owned by rank 0 alone, so the merge must return rank 0's
    # output there verbatim.
    np.testing.assert_allclose(got[3], outputs[0, 3].numpy(), rtol=1e-6, atol=1e-6)


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
    per_rank = torch.from_numpy(rng.standard_normal((4, 6, 3, 8)).astype(np.float32))
    monkeypatch.setattr(
        cp_mla, "get_dcp_group", lambda: _HeadGatherGroup(per_rank, rank)
    )

    got = cp_mla.all_gather_heads(per_rank[rank])
    assert got.shape == (6, 12, 8)
    np.testing.assert_array_equal(
        got[:, rank * 3 : (rank + 1) * 3].numpy(), per_rank[rank].numpy()
    )

    nope_dcp, pe_dcp = cp_mla.all_gather_q_nope_pe_heads(
        per_rank[rank][..., :5], per_rank[rank][..., 5:], 5
    )
    np.testing.assert_array_equal(nope_dcp.numpy(), got[..., :5].numpy())
    np.testing.assert_array_equal(pe_dcp.numpy(), got[..., 5:].numpy())


def test_head_gather_is_a_noop_without_a_group(monkeypatch):
    monkeypatch.setattr(cp_mla, "get_dcp_group", lambda: None)
    q = torch.ones(2, 3, 4)
    assert cp_mla.all_gather_heads(q) is q
    q_pe = torch.ones(2, 3, 2)
    nope_out, pe_out = cp_mla.all_gather_q_nope_pe_heads(q, q_pe, 4)
    assert nope_out is q
    assert pe_out is q_pe


def test_scatter_merge_is_a_noop_without_a_group(monkeypatch):
    monkeypatch.setattr(cp_mla, "get_dcp_group", lambda: None)
    out = torch.ones(2, 3, 4)
    assert cp_mla.merge_lse_partials_scatter_heads(out, torch.zeros(2, 3)) is out


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
        outs[r] = np.einsum("tnk,kd->tnd", e / e.sum(-1, keepdims=True), v[r::dcp_size])

    got = _merge_whole(
        torch.from_numpy(outs).float(), torch.from_numpy(lses).float(), monkeypatch
    ).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# ===================================================================== #
# The all-to-all merge (`TPU_DCP_LSE_MERGE_A2A`).
# ===================================================================== #


class _ExchangeHub:
    """A lockstep all-to-all between threads, one thread per DCP rank."""

    def __init__(self, world, wrong_destination=False):
        self.world = world
        self.slots = [None] * world
        self.barrier = threading.Barrier(world)
        # Fault injection: rank `r` is handed the block meant for rank `r+1`.
        self.wrong_destination = wrong_destination


class _A2AGroup:
    def __init__(self, hub, rank):
        self.world_size = hub.world
        self.rank_in_group = rank
        self.device_group = None
        self._hub = hub

    def all_to_all(self, x):
        hub = self._hub
        hub.slots[self.rank_in_group] = x
        hub.barrier.wait()
        dest = (self.rank_in_group + int(hub.wrong_destination)) % hub.world
        recv = torch.stack([hub.slots[src][dest] for src in range(hub.world)])
        hub.barrier.wait()
        return recv


def _a2a_merge_whole(outputs, lses, monkeypatch, wrong_destination=False):
    """Every rank's all-to-all merge, run in lockstep, reassembled over heads."""
    monkeypatch.setattr(cp_mla, "_all_to_all", lambda x, group: group.all_to_all(x))
    world = outputs.shape[0]
    hub = _ExchangeHub(world, wrong_destination)

    def run(rank):
        return cp_mla.merge_lse_partials_all_to_all(
            outputs[rank], lses[rank], _A2AGroup(hub, rank)
        )

    with concurrent.futures.ThreadPoolExecutor(world) as pool:
        blocks = list(pool.map(run, range(world)))
    return torch.cat(blocks, dim=1)


def _hard_partials(rng, dcp_size, tokens=12, local_heads=4, vdim=64):
    """bf16 partials (the kernel's output dtype) with f32 lse that exercise
    every merge rule: large lse (a merge without the max shift overflows exp),
    tokens one rank owns alone, a token no rank owns, a rank owning nothing."""
    heads = dcp_size * local_heads
    outputs = torch.from_numpy(
        rng.standard_normal((dcp_size, tokens, heads, vdim)).astype(np.float32)
    ).to(torch.bfloat16)
    lses = torch.from_numpy(
        (rng.standard_normal((dcp_size, tokens, heads)) * 4).astype(np.float32)
    )
    lses[:, 0] += 200.0  # exp(200) overflows f32: only the shift saves it
    lses[:, 1] -= 200.0  # exp(-200) underflows f32 to 0: only the shift saves it
    lses[1:, 2] = float("-inf")  # rank 0 owns token 2 alone
    lses[:, 3] = float("-inf")  # nobody owns token 3 (a padding row)
    lses[dcp_size - 1] = float("-inf")  # the last rank owns nothing at all
    lses[0, 4, :local_heads] = float("-inf")  # a mixed row, per head
    return outputs, lses


def _bf16_close(got, reference):
    """Within bf16 rounding of the f64 answer: one bf16 ulp (2^-8 relative,
    since the bf16 mantissa has 8 bits with the implicit one) plus a floor."""
    got = got.to(torch.float32).numpy().astype(np.float64)
    return bool(
        np.isfinite(got).all()
        and (np.abs(got - reference) <= 2.0**-8 * np.abs(reference) + 1e-6).all()
    )


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
def test_all_to_all_merge_matches_the_f64_reference(dcp_size, monkeypatch):
    outputs, lses = _hard_partials(np.random.default_rng(20261002), dcp_size)
    got = _a2a_merge_whole(outputs, lses, monkeypatch)
    assert got.dtype == torch.bfloat16
    assert got.shape == outputs.shape[1:]
    reference = _reference_merge(outputs.float(), lses)
    assert _bf16_close(got, reference)
    # The `-inf` rules, stated exactly: a lone owner comes back verbatim and a
    # token nobody owns is zero, not NaN.
    torch.testing.assert_close(got[2], outputs[0, 2], rtol=0, atol=0)
    assert (got[3] == 0).all()


def test_all_to_all_merge_agrees_with_the_reduce_scatter_merge(monkeypatch):
    """Compare both paths to the reference."""
    outputs, lses = _hard_partials(np.random.default_rng(7), 8)
    reference = _reference_merge(outputs.float(), lses)
    a2a_result = _a2a_merge_whole(outputs, lses, monkeypatch)
    ag_rs_result = _merge_whole(outputs.float(), lses, monkeypatch).to(torch.bfloat16)
    assert _bf16_close(a2a_result, reference)
    assert _bf16_close(ag_rs_result, reference)


def test_positive_control_wrong_head_slice_is_caught(monkeypatch):
    """The reference check must fail when a rank merges another rank's heads."""
    outputs, lses = _hard_partials(np.random.default_rng(13), 4)
    got = _a2a_merge_whole(outputs, lses, monkeypatch, wrong_destination=True)
    assert not _bf16_close(got, _reference_merge(outputs.float(), lses))


@pytest.mark.parametrize(
    "dcp_size,tokens,local_heads,vdim",
    [(2, 12, 4, 64), (8, 16, 4, 512), (16, 16, 4, 512), (8, 64, 8, 512), (4, 5, 3, 64)],
)
def test_packed_all_to_all_merge_is_bitwise_the_two_exchange_merge(
    dcp_size, tokens, local_heads, vdim, monkeypatch
):
    """`TPU_DCP_LSE_MERGE_A2A_PACK`: one exchange carrying the lse bits after
    the partial's rows returns the two-exchange merge bit for bit, including
    the shapes whose lse words do not fill a whole row."""
    outputs, lses = _hard_partials(
        np.random.default_rng(29), dcp_size, tokens, local_heads, vdim
    )
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_PACK", False, raising=False)
    two = _a2a_merge_whole(outputs, lses, monkeypatch)
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_PACK", True, raising=False)
    calls = []
    real = cp_mla._all_to_all_packed
    monkeypatch.setattr(
        cp_mla,
        "_all_to_all_packed",
        lambda o, lse, g, *rest: calls.append(1) or real(o, lse, g, *rest),
    )
    one = _a2a_merge_whole(outputs, lses, monkeypatch)
    assert len(calls) == dcp_size  # every rank took the packed exchange
    assert torch.equal(one.view(torch.int16), two.view(torch.int16))
    assert _bf16_close(one, _reference_merge(outputs.float(), lses))


def test_positive_control_packed_lse_rows_from_the_wrong_rank_are_caught(
    monkeypatch,
):
    """The packed merge's check must fail when the lse rows (only) come from
    the neighbouring source rank."""
    outputs, lses = _hard_partials(np.random.default_rng(31), 4, 16, 4, 512)
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_PACK", True, raising=False)
    rows = 16 * 4

    def exchange(x, group):
        r = group.all_to_all(x)
        return torch.cat([r[:, :rows], torch.roll(r[:, rows:], 1, 0)], 1)

    world = 4
    hub = _ExchangeHub(world)
    monkeypatch.setattr(cp_mla, "_all_to_all", exchange)

    def run(rank):
        return cp_mla.merge_lse_partials_all_to_all(
            outputs[rank], lses[rank], _A2AGroup(hub, rank)
        )

    with concurrent.futures.ThreadPoolExecutor(world) as pool:
        got = torch.cat(list(pool.map(run, range(world))), dim=1)
    assert not _bf16_close(got, _reference_merge(outputs.float(), lses))


def _ragged_stand_in(hub, calls=None, short_by=0):
    """`cp_mla._ragged_all_to_all` over the lockstep hub: rank `d` receives
    the first `rows` rows of every source's block `d`, zeros after."""

    def exchange(x, rows, group):
        n = int(rows.reshape(-1)[0]) - short_by
        if calls is not None and group.rank_in_group == 0:
            calls.append(int(rows.reshape(-1)[0]))
        recv = group.all_to_all(x)
        out = torch.zeros_like(recv)
        out[:, :n] = recv[:, :n]
        return out

    return exchange


def _merge_real_rows(outputs, lses, monkeypatch, real, calls=None, short_by=0):
    """Every rank's packed a2a merge with TPU_DCP_LSE_MERGE_A2A_REAL_ROWS and a
    step of `real` real tokens (query_start_loc [0, real])."""
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_PACK", True, raising=False)
    monkeypatch.setattr(
        cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS", True, raising=False
    )
    world = outputs.shape[0]
    hub = _ExchangeHub(world)
    monkeypatch.setattr(
        cp_mla, "_ragged_all_to_all", _ragged_stand_in(hub, calls, short_by)
    )
    monkeypatch.setattr(cp_mla, "_all_to_all", lambda x, group: group.all_to_all(x))
    qsl = torch.tensor([0, real], dtype=torch.int32)

    def run(rank):
        return cp_mla.merge_lse_partials_all_to_all(
            outputs[rank], lses[rank], _A2AGroup(hub, rank), query_start_loc=qsl
        )

    with concurrent.futures.ThreadPoolExecutor(world) as pool:
        return torch.cat(list(pool.map(run, range(world))), dim=1)


def _merge_packed(outputs, lses, monkeypatch):
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_PACK", True, raising=False)
    monkeypatch.setattr(
        cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS", False, raising=False
    )
    return _a2a_merge_whole(outputs, lses, monkeypatch)


@pytest.mark.parametrize(
    "dcp_size,tokens,local_heads,vdim,real",
    [
        (16, 16, 4, 512, 4),  # mirror decode b1: 4 real rows of 16
        (16, 16, 4, 512, 8),  # b2
        (16, 16, 4, 512, 16),  # b4: every row real
        (8, 32, 4, 512, 12),
        (4, 16, 4, 512, 1),
        (2, 12, 4, 64, 0),  # a step with no real token
    ],
)
def test_real_rows_merge_is_bitwise_the_packed_merge_on_real_rows(
    dcp_size, tokens, local_heads, vdim, real, monkeypatch
):
    """`TPU_DCP_LSE_MERGE_A2A_REAL_ROWS`: sending only the real tokens' rows
    returns the packed merge bit for bit on every real row (hard partials:
    `-inf`/overflowing lse, a rank owning nothing, NaN partials of weight 0),
    and zero on the padding rows; the exchange is sized `extra + real * local`."""
    outputs, lses = _hard_partials(
        np.random.default_rng(37), dcp_size, tokens, local_heads, vdim
    )
    outputs = _nan_where_unowned(outputs, lses)
    want = _merge_packed(outputs, lses, monkeypatch)
    calls = []
    got = _merge_real_rows(outputs, lses, monkeypatch, real, calls)
    extra = -(-2 * tokens * local_heads // vdim)
    assert calls == [extra + real * local_heads]
    assert got.shape == want.shape and got.dtype == want.dtype
    assert torch.equal(got[:real].view(torch.int16), want[:real].view(torch.int16))
    assert (got[real:] == 0).all() and torch.isfinite(got.float()).all()


def test_real_rows_merge_without_query_start_loc_keeps_every_row(monkeypatch):
    """No query_start_loc (the drafter, eager callers): the full packed exchange."""
    outputs, lses = _hard_partials(np.random.default_rng(41), 4, 16, 4, 512)
    want = _merge_packed(outputs, lses, monkeypatch)
    monkeypatch.setattr(
        cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS", True, raising=False
    )
    monkeypatch.setattr(
        cp_mla,
        "_ragged_all_to_all",
        lambda *a: (_ for _ in ()).throw(AssertionError("ragged without qsl")),
    )
    got = _a2a_merge_whole(outputs, lses, monkeypatch)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_positive_control_real_rows_short_by_one_is_caught(monkeypatch):
    """The real-rows check must fail when the exchange drops the last real
    row (an off-by-one in the row count)."""
    outputs, lses = _hard_partials(np.random.default_rng(43), 16, 16, 4, 512)
    real = 5  # the dropped row (token 4's last head) is owned by most ranks
    want = _merge_packed(outputs, lses, monkeypatch)
    got = _merge_real_rows(outputs, lses, monkeypatch, real, short_by=1)
    assert not torch.equal(got[:real].view(torch.int16), want[:real].view(torch.int16))


def test_positive_control_dropped_lse_shift_is_caught(monkeypatch):
    """The reference check must fail when the merge skips the max shift."""

    def unshifted(o_recv, l_recv):
        weights = torch.where(
            torch.isfinite(l_recv), torch.exp(l_recv), torch.zeros_like(l_recv)
        )
        denom = weights.sum(dim=0)
        weights = weights / torch.where(denom > 0, denom, torch.ones_like(denom))
        return (o_recv.to(torch.float32) * weights.unsqueeze(-1)).sum(dim=0)

    outputs, lses = _hard_partials(np.random.default_rng(17), 4)
    monkeypatch.setattr(cp_mla, "_merge_received", unshifted)
    got = _a2a_merge_whole(outputs, lses, monkeypatch)
    assert not _bf16_close(got, _reference_merge(outputs.float(), lses))


def _nan_where_unowned(outputs, lses):
    """The partials a rank returns for rows it owns no position of: the kernel ran
    on a dummy key and its output there may be NaN (mla-dec `edge_short_rank5`)."""
    poisoned = outputs.clone()
    poisoned[~torch.isfinite(lses)] = float("nan")
    return poisoned


@pytest.mark.parametrize("dcp_size", [2, 8])
def test_all_to_all_merge_ignores_nan_zero_weight_partials(dcp_size, monkeypatch):
    """A NaN partial of weight 0 changes nothing: the merge equals, bit for bit,
    the merge of the same partials with finite values there."""
    outputs, lses = _hard_partials(np.random.default_rng(20261010), dcp_size)
    clean = _a2a_merge_whole(outputs, lses, monkeypatch)
    got = _a2a_merge_whole(_nan_where_unowned(outputs, lses), lses, monkeypatch)
    assert torch.isfinite(got.float()).all()
    assert torch.equal(got.view(torch.int16), clean.view(torch.int16))
    assert _bf16_close(got, _reference_merge(outputs.float(), lses))
    assert (got[3] == 0).all()


def test_positive_control_unguarded_merge_is_poisoned_by_nan(monkeypatch):
    """Without the guard the same input comes back NaN (the a2a bug)."""
    outputs, lses = _hard_partials(np.random.default_rng(20261010), 8)
    monkeypatch.setattr(cp_mla, "_zero_nan", lambda o: o)
    got = _a2a_merge_whole(_nan_where_unowned(outputs, lses), lses, monkeypatch)
    assert not torch.isfinite(got.float()).all()
    assert not _bf16_close(got, _reference_merge(outputs.float(), lses))


@pytest.mark.parametrize("seed", [7, 13, 20261002])
def test_nan_guard_is_bit_identical_on_finite_partials(seed, monkeypatch):
    """On finite partials the guard changes no bit of any row some rank owns,
    including the partly `-inf` and underflowing-weight rows. A row nobody owns
    is zero either way; only the sign of that zero may differ (the unguarded sum
    of `o * 0` is -0 where every partial is negative)."""
    outputs, lses = _hard_partials(np.random.default_rng(seed), 8)
    got = _a2a_merge_whole(outputs, lses, monkeypatch)
    monkeypatch.setattr(cp_mla, "_zero_nan", lambda o: o)
    old = _a2a_merge_whole(outputs, lses, monkeypatch)
    owned = torch.isfinite(lses).any(dim=0)  # [T, H]
    assert owned.any() and not owned.all()
    assert torch.equal(got[owned].view(torch.int16), old[owned].view(torch.int16))
    assert (got[~owned] == 0).all() and (old[~owned] == 0).all()


@pytest.mark.parametrize(
    "enabled,tokens,is_backbone,expect_a2a",
    [
        (False, 8, True, False),
        (True, 8, True, True),
        (True, 256, True, True),
        (True, 257, True, False),
        # The MTP drafter, traced once for every bucket, keeps the
        # reduce-scatter at every size.
        (True, 8, False, False),
        (True, 257, False, False),
        (False, 257, False, False),
    ],
)
def test_merge_picks_the_all_to_all(
    enabled, tokens, is_backbone, expect_a2a, monkeypatch
):
    calls = []
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A", enabled, raising=False)
    monkeypatch.setattr(
        cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS", 256, raising=False
    )
    monkeypatch.setattr(
        cp_mla,
        "merge_lse_partials_all_to_all",
        lambda o, lse, g, **kw: calls.append(o.shape[0]) or o[:, :2],
    )
    outputs, lses = _partials(np.random.default_rng(3), 2, tokens=tokens, vdim=4)
    group = _FakeGroup(outputs, lses, 0)
    monkeypatch.setattr(cp_mla, "get_dcp_group", lambda: group)
    cp_mla.merge_lse_partials_scatter_heads(
        outputs[0], lses[0], is_backbone=is_backbone
    )
    assert calls == ([tokens] if expect_a2a else [])


# ===================================================================== #
# The token-chunked reduce-scatter merge (`TPU_DCP_MERGE_CHUNKED`).
# ===================================================================== #


class _CollectiveGroup:
    """Lockstep `all_gather`/`reduce_scatter` between threads, one per rank,
    over whatever tensor each rank actually passes. The reduce-scatter sums in
    rank order, so two merges that hand it the same terms agree bitwise."""

    def __init__(self, hub, rank):
        self.world_size = hub.world
        self.rank_in_group = rank
        self._hub = hub

    def _exchange(self, x):
        hub = self._hub
        hub.slots[self.rank_in_group] = x
        hub.barrier.wait()
        got = list(hub.slots)
        hub.barrier.wait()
        return got

    def all_gather(self, tensor, dim=0):
        return torch.cat(self._exchange(tensor), dim=dim)

    def reduce_scatter(self, tensor, dim=-1):
        terms = self._exchange(tensor)
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        width = total.shape[dim] // self.world_size
        return total.narrow(dim, self.rank_in_group * width, width).contiguous()

    def all_to_all_blocks(self, x):
        """Block `d` of `x`'s leading axis to rank `d` (`cp_mla._all_to_all`)."""
        terms = self._exchange(x)
        return torch.stack([t[self.rank_in_group] for t in terms])


def _rs_merge_whole(outputs, lses, monkeypatch, **env):
    """Every rank's reduce-scatter merge under `env`, run in lockstep,
    reassembled over heads."""
    defaults = {
        "TPU_DCP_LSE_MERGE_A2A": False,
        "TPU_DCP_MERGE_CHUNKED": 0,
        "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS": 0,
        "TPU_DCP_MERGE_CHUNKED_LSE_AG": False,
        "TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS": 0,
        "TPU_DCP_MERGE_RS_BF16": 0,
    }
    for name, value in {**defaults, **env}.items():
        monkeypatch.setattr(cp_mla.envs, name, value, raising=False)
    world = outputs.shape[0]
    hub = _ExchangeHub(world)
    groups = [_CollectiveGroup(hub, rank) for rank in range(world)]
    monkeypatch.setattr(
        cp_mla,
        "get_dcp_group",
        lambda: groups[int(threading.current_thread().name.rsplit("_", 1)[-1])],
    )
    monkeypatch.setattr(cp_mla, "_all_to_all", lambda x, g: g.all_to_all_blocks(x))

    def run(rank):
        threading.current_thread().name = f"dcp_rank_{rank}"
        return cp_mla.merge_lse_partials_scatter_heads(outputs[rank], lses[rank])

    with concurrent.futures.ThreadPoolExecutor(world) as pool:
        blocks = list(pool.map(run, range(world)))
    return torch.cat(blocks, dim=1)


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("chunks", [2, 4])
def test_chunked_merge_is_bitwise_the_unchunked_merge(dcp_size, chunks, monkeypatch):
    """Chunking the token axis changes no per-row arithmetic, so it returns
    the unchunked merge bit for bit -- and that merge meets the f64
    reference, `-inf` rules included."""
    outputs, lses = _hard_partials(np.random.default_rng(31), dcp_size, tokens=32)
    calls = []
    real = cp_mla._merge_chunked
    monkeypatch.setattr(
        cp_mla, "_merge_chunked", lambda *a: calls.append(a[-1]) or real(*a)
    )
    today = _rs_merge_whole(outputs, lses, monkeypatch)
    assert calls == []
    got = _rs_merge_whole(outputs, lses, monkeypatch, TPU_DCP_MERGE_CHUNKED=chunks)
    assert calls == [chunks] * dcp_size
    assert got.dtype == torch.bfloat16 and got.shape == outputs.shape[1:]
    torch.testing.assert_close(got, today, rtol=0, atol=0)
    assert _bf16_close(got, _reference_merge(outputs.float(), lses))
    torch.testing.assert_close(got[2], outputs[0, 2], rtol=0, atol=0)
    assert (got[3] == 0).all()


def test_positive_control_misaligned_chunk_weights_are_caught(monkeypatch):
    """The checks above must fail when a chunk is weighted with lse rows that
    are not its own (lse shifted by one row inside every chunk)."""
    real = cp_mla.cp_lse_ag_out_rs
    monkeypatch.setattr(
        cp_mla,
        "cp_lse_ag_out_rs",
        lambda o, lse, g: real(o, torch.roll(lse, 1, dims=0), g),
    )
    outputs, lses = _hard_partials(np.random.default_rng(37), 4, tokens=32)
    got = _rs_merge_whole(outputs, lses, monkeypatch, TPU_DCP_MERGE_CHUNKED=4)
    assert not _bf16_close(got, _reference_merge(outputs.float(), lses))


@pytest.mark.parametrize("dcp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("chunks", [2, 4])
def test_collapsed_lse_gather_is_bitwise_the_chunked_merge(
    dcp_size, chunks, monkeypatch
):
    """`TPU_DCP_MERGE_CHUNKED_LSE_AG` gathers the lse of every chunk in one
    call instead of one per chunk; it returns the per-chunk-gather merge (and
    so the unchunked one) bit for bit."""
    outputs, lses = _hard_partials(np.random.default_rng(41), dcp_size, tokens=32)
    calls = []
    real = cp_mla._gather_lse
    monkeypatch.setattr(
        cp_mla, "_gather_lse", lambda x, g: calls.append(x.shape[0]) or real(x, g)
    )
    today = _rs_merge_whole(outputs, lses, monkeypatch, TPU_DCP_MERGE_CHUNKED=chunks)
    assert calls == []
    got = _rs_merge_whole(
        outputs,
        lses,
        monkeypatch,
        TPU_DCP_MERGE_CHUNKED=chunks,
        TPU_DCP_MERGE_CHUNKED_LSE_AG=True,
    )
    assert len(calls) == dcp_size
    assert sum(calls) == 32 * dcp_size  # every row gathered exactly once
    assert got.dtype == torch.bfloat16 and got.shape == outputs.shape[1:]
    torch.testing.assert_close(got, today, rtol=0, atol=0)
    unchunked = _rs_merge_whole(outputs, lses, monkeypatch)
    torch.testing.assert_close(got, unchunked, rtol=0, atol=0)
    assert _bf16_close(got, _reference_merge(outputs.float(), lses))


def test_positive_control_misaligned_collapsed_lse_gather_is_caught(monkeypatch):
    """The checks above must fail when the collapsed gather hands a chunk lse
    rows that are not its own (gathered rows shifted by one)."""
    real = cp_mla._gather_lse
    monkeypatch.setattr(
        cp_mla, "_gather_lse", lambda x, g: torch.roll(real(x, g), 1, dims=1)
    )
    outputs, lses = _hard_partials(np.random.default_rng(43), 4, tokens=32)
    got = _rs_merge_whole(
        outputs,
        lses,
        monkeypatch,
        TPU_DCP_MERGE_CHUNKED=4,
        TPU_DCP_MERGE_CHUNKED_LSE_AG=True,
    )
    assert not _bf16_close(got, _reference_merge(outputs.float(), lses))


@pytest.mark.parametrize(
    "flag,cap,tokens,expect",
    [
        (False, 0, 4096, False),
        (True, 0, 4096, True),
        (True, 2048, 2048, True),
        (True, 2048, 4096, False),  # above the cap: per-chunk gathers
    ],
)
def test_collapsed_lse_gather_per_bucket(flag, cap, tokens, expect, monkeypatch):
    monkeypatch.setattr(
        cp_mla.envs, "TPU_DCP_MERGE_CHUNKED_LSE_AG", flag, raising=False
    )
    monkeypatch.setattr(
        cp_mla.envs, "TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS", cap, raising=False
    )
    assert cp_mla._lse_gathered_once(tokens) is expect


@pytest.mark.parametrize(
    "chunks,min_tokens,tokens,is_backbone,expect",
    [
        (0, 0, 4096, True, 1),
        (1, 0, 4096, True, 1),
        (4, 1024, 4096, True, 4),
        (4, 1024, 1024, True, 4),
        (4, 1024, 512, True, 1),  # below the bucket floor
        (4, 0, 48, True, 1),  # 12-row chunks: not a multiple of 8
        (2, 0, 48, True, 2),
        (4, 1024, 4096, False, 1),  # single-trace caller (the MTP drafter)
    ],
)
def test_merge_chunks_per_bucket(
    chunks, min_tokens, tokens, is_backbone, expect, monkeypatch
):
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_LSE_MERGE_A2A", False, raising=False)
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_MERGE_CHUNKED", chunks, raising=False)
    monkeypatch.setattr(
        cp_mla.envs, "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS", min_tokens, raising=False
    )
    calls = []
    monkeypatch.setattr(
        cp_mla, "_merge_chunked", lambda o, lse, g, n: calls.append(n) or o[:, :2]
    )
    outputs, lses = _partials(np.random.default_rng(3), 2, tokens=tokens, vdim=4)
    monkeypatch.setattr(cp_mla, "get_dcp_group", lambda: _FakeGroup(outputs, lses, 0))
    cp_mla.merge_lse_partials_scatter_heads(
        outputs[0], lses[0], is_backbone=is_backbone
    )
    assert calls == ([expect] if expect > 1 else [])


# ===================================================================== #
# bf16 wire for the chunked merge (`TPU_DCP_MERGE_RS_BF16`). Not bitwise:
# bounded in bf16 ulps of the merge's condition scale sum_r w_r |o_r|.
# ===================================================================== #

# Max error bounds, in bf16 ulps of sum_r w_r |o_r|. f32 today: one final
# rounding (0.5). Mode 2 (bf16 terms, f32 sum): each term rounds by at most
# 2^-8 of itself, so all of them by at most one ulp of the scale, plus the
# final rounding: 1.5, a proof, not a fit. Mode 1 (bf16 accumulation in the
# collective; the lockstep stand-in sums in rank order, the worst ring
# order): no tight proof (dcp_size - 1 roundings of a partial sum); the bound
# is 1.5x the worst seen in a 16-rank CPU emulation (4.6 ulp max).
_RS_BF16_MAX_ULP = {0: 0.5 + 1e-3, 2: 1.5, 1: 7.0}


def _spread_partials(rng, dcp_size, tokens, local_heads, vdim, spread):
    heads = dcp_size * local_heads
    outputs = torch.from_numpy(
        rng.standard_normal((dcp_size, tokens, heads, vdim)).astype(np.float32)
    ).to(torch.bfloat16)
    lses = torch.from_numpy(
        (rng.standard_normal((dcp_size, tokens, heads)) * spread + 3).astype(np.float32)
    )
    return outputs, lses


def _scale_ulp_errors(got, outputs, lses):
    """|got - f64 reference| in bf16 ulps of sum_r w_r |o_r| (0 where no rank
    owns the row, which must then be exactly 0)."""
    out = outputs.float().numpy().astype(np.float64)
    lse = lses.numpy().astype(np.float64)
    fin = np.isfinite(lse)
    shift = np.where(fin, lse, -np.inf).max(axis=0)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    w = np.where(fin, np.exp(lse - shift[None]), 0.0)
    den = w.sum(axis=0)
    w = w / np.where(den > 0.0, den, 1.0)
    ref = (out * w[..., None]).sum(axis=0)
    scale = (np.abs(out) * w[..., None]).sum(axis=0)
    g = got.to(torch.float32).numpy().astype(np.float64)
    err = np.abs(g - ref)
    ulp = np.exp2(np.floor(np.log2(np.where(scale > 0, scale, 1.0))) - 7)
    e = np.where(scale > 0, err / ulp, np.where(err == 0, 0.0, np.inf))
    return {
        "max": float(e.max()),
        "mean": float(e.mean()),
        "frac_over_1ulp": float((e > 1.0).mean()),
    }


# (tokens, chunks, collapsed lse gather): the served c1k / c2k / c8k chunked
# buckets (the lse gather is collapsed up to 2048 tokens on the stack).
_MIXED_SHAPES = [(1024, 8, True), (2048, 8, True), (8192, 8, False)]


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("dcp_size", [8, 16])
@pytest.mark.parametrize("tokens,chunks,lse_ag", _MIXED_SHAPES)
@pytest.mark.parametrize("spread", [0.5, 4.0])
def test_bf16_wire_merge_is_within_its_bound(
    mode, dcp_size, tokens, chunks, lse_ag, spread, monkeypatch
):
    """The bf16-wire chunked merge vs the f64 reference and the f32 merge, at
    the served mixed-bucket shapes."""
    outputs, lses = _spread_partials(
        np.random.default_rng(97 + dcp_size + tokens), dcp_size, tokens, 1, 16, spread
    )
    env = dict(TPU_DCP_MERGE_CHUNKED=chunks, TPU_DCP_MERGE_CHUNKED_LSE_AG=lse_ag)
    f32 = _rs_merge_whole(outputs, lses, monkeypatch, **env)
    got = _rs_merge_whole(outputs, lses, monkeypatch, TPU_DCP_MERGE_RS_BF16=mode, **env)
    assert got.dtype == torch.bfloat16 and got.shape == outputs.shape[1:]
    e_f32 = _scale_ulp_errors(f32, outputs, lses)
    e_got = _scale_ulp_errors(got, outputs, lses)
    print(
        f"RS_BF16 mode={mode} dcp={dcp_size} T={tokens} s={spread}: "
        f"f32 {e_f32} got {e_got}"
    )
    assert e_f32["max"] <= _RS_BF16_MAX_ULP[0], e_f32
    assert e_got["max"] <= _RS_BF16_MAX_ULP[mode], e_got
    assert not torch.equal(got.view(torch.int16), f32.view(torch.int16))  # really bf16


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("dcp_size", [8, 16])
@pytest.mark.parametrize("lse_ag", [False, True])
def test_bf16_wire_merge_keeps_the_merge_rules(mode, dcp_size, lse_ag, monkeypatch):
    """`-inf` rules and exp range under the bf16 wire: a lone owner verbatim,
    a row nobody owns exactly zero, +-200 lse finite and within the bound."""
    outputs, lses = _hard_partials(np.random.default_rng(53), dcp_size, tokens=32)
    got = _rs_merge_whole(
        outputs,
        lses,
        monkeypatch,
        TPU_DCP_MERGE_CHUNKED=4,
        TPU_DCP_MERGE_CHUNKED_LSE_AG=lse_ag,
        TPU_DCP_MERGE_RS_BF16=mode,
    )
    assert torch.isfinite(got.float()).all()
    torch.testing.assert_close(got[2], outputs[0, 2], rtol=0, atol=0)
    assert (got[3] == 0).all()
    assert _scale_ulp_errors(got, outputs, lses)["max"] <= _RS_BF16_MAX_ULP[mode]


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("lse_ag", [False, True])
def test_positive_control_wrong_weight_exceeds_the_bf16_bound(
    mode, lse_ag, monkeypatch
):
    """The bound above must fail when one rank weighs its partial 25 % too
    heavily (its own lse raised by log 1.25, the merged lse left alone)."""
    real = cp_mla._weighted_reduce_scatter

    def wrong(output, lse_f32, lses, group):
        if group.rank_in_group == 1:
            lse_f32 = lse_f32 + math.log(1.25)
        return real(output, lse_f32, lses, group)

    monkeypatch.setattr(cp_mla, "_weighted_reduce_scatter", wrong)
    outputs, lses = _spread_partials(np.random.default_rng(59), 16, 1024, 1, 16, 4.0)
    got = _rs_merge_whole(
        outputs,
        lses,
        monkeypatch,
        TPU_DCP_MERGE_CHUNKED=8,
        TPU_DCP_MERGE_CHUNKED_LSE_AG=lse_ag,
        TPU_DCP_MERGE_RS_BF16=mode,
    )
    assert _scale_ulp_errors(got, outputs, lses)["max"] > _RS_BF16_MAX_ULP[mode]


def test_bf16_wire_off_is_bitwise_the_chunked_merge(monkeypatch):
    """`TPU_DCP_MERGE_RS_BF16=0` leaves the f32 chunked merge as it was."""
    outputs, lses = _hard_partials(np.random.default_rng(61), 8, tokens=32)
    for lse_ag in (False, True):
        env = dict(TPU_DCP_MERGE_CHUNKED=4, TPU_DCP_MERGE_CHUNKED_LSE_AG=lse_ag)
        a = _rs_merge_whole(outputs, lses, monkeypatch, **env)
        b = _rs_merge_whole(outputs, lses, monkeypatch, TPU_DCP_MERGE_RS_BF16=0, **env)
        unchunked = _rs_merge_whole(outputs, lses, monkeypatch)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a, unchunked, rtol=0, atol=0)


def test_bf16_wire_leaves_unchunked_buckets_alone(monkeypatch):
    """Only the chunked merge changes: an unchunked bucket (and the MTP
    drafter, which never chunks) keeps the f32 reduce-scatter bit for bit."""
    outputs, lses = _hard_partials(np.random.default_rng(67), 8, tokens=32)
    a = _rs_merge_whole(outputs, lses, monkeypatch)
    b = _rs_merge_whole(outputs, lses, monkeypatch, TPU_DCP_MERGE_RS_BF16=1)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


# ===================================================================== #
# Position ownership tests.
# ===================================================================== #

_OWNERSHIP = [(d, c) for d in (2, 4, 8, 16) for c in (1, 4, 64)]


@pytest.mark.parametrize("dcp_size,interleave_c", _OWNERSHIP)
def test_ownership_is_a_partition_of_the_positions(dcp_size, interleave_c):
    """Every position has exactly one owner, and every rank is an owner."""
    n = 5 * dcp_size * interleave_c + 3
    owners = np.asarray(
        cp_owner_rank(jnp.arange(n, dtype=jnp.int32), dcp_size, interleave_c)
    )
    assert owners.min() >= 0 and owners.max() < dcp_size
    assert set(owners.tolist()) == set(range(dcp_size))


@pytest.mark.parametrize("dcp_size,interleave_c", _OWNERSHIP)
def test_owner_counts_are_balanced_over_a_whole_cycle(dcp_size, interleave_c):
    """Over a whole interleave cycle the shards are exactly equal.

    Which is why mean per-rank occupancy of a `k`-wide top-k is `k / dcp_size`
    as an identity, not an estimate -- the spread around it comes only from
    where the winners cluster inside the cycle.
    """
    cycle = dcp_size * interleave_c
    owners = np.asarray(
        cp_owner_rank(jnp.arange(3 * cycle, dtype=jnp.int32), dcp_size, interleave_c)
    )
    counts = np.bincount(owners, minlength=dcp_size)
    assert (counts == 3 * interleave_c).all()


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
NOPE_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.NOPE,
    KVCacheLayout.TENSORCORE,
    TOTAL_PAGES,
    PAGE_SIZE,
    LKV_DIM,
    KV_PACKING,
)
ROPE_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.ROPE,
    KVCacheLayout.TENSORCORE,
    TOTAL_PAGES,
    PAGE_SIZE,
    ROPE_DIM,
    KV_PACKING,
)

DCP_TOKENS = 64  # multiple of TOKEN_PAD, fits PAGES_PER_SEQ * PAGE_SIZE
DCP_TOPK = 64
# The DCP shard rule from streamindex_topk.cp_local_to_global: compressed
# position `p` is owned by rank `(p // interleave_c) % dcp_size`. Anything
# below `kv_cache_utils.WORD_BYTES` would split a rope tile across ranks.
DCP_INTERLEAVE_C = 4


def _empty_pair():
    """Zeroed (nope, rope) caches in dsa_gather's native tiled layouts."""
    return (
        jnp.zeros(NOPE_SPEC.shape, NOPE_SPEC.jax_dtype),
        jnp.zeros(ROPE_SPEC.shape, ROPE_SPEC.jax_dtype),
    )


def _quantize_fp8(x: np.ndarray, k_scale: float) -> jax.Array:
    return jnp.asarray(x / k_scale).astype(jnp.float8_e4m3fn)


def _dequantize(x: jax.Array, k_scale: float) -> np.ndarray:
    return np.asarray(x.astype(jnp.float32)) * k_scale


def _causal_topk(positions: list[int], topk: int) -> np.ndarray:
    """topk_indices row per token: [0..pos] then -1 padding."""
    rows = np.full((len(positions), topk), -1, np.int32)
    for i, pos in enumerate(positions):
        assert pos + 1 <= topk
        rows[i, : pos + 1] = np.arange(pos + 1)
    return rows


def _shard_topk_by_owner(topk_rows: np.ndarray, dcp_size: int, interleave_c: int):
    """Partition each token's top-k list by the position's owning DCP rank"""
    num_tokens, topk = topk_rows.shape
    shards = np.full((dcp_size, num_tokens, topk), -1, np.int32)
    counts = np.zeros((dcp_size, num_tokens), np.int32)
    for t in range(num_tokens):
        row = topk_rows[t][topk_rows[t] >= 0]
        for r in range(dcp_size):
            sel = row[(row // interleave_c) % dcp_size == r]
            counts[r, t] = sel.size
            shards[r, t, : sel.size] = sel
            if sel.size == 0:
                shards[r, t, 0] = 0  # dummy; masked by `counts` at merge time
    return shards, counts


def _merge_partials(
    outs: np.ndarray, lses: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """`o = sum_r softmax_r(lse) * o_r` over the DCP axis, in f32"""
    lses = np.where(counts[:, :, None] > 0, lses, -np.inf)
    shift = lses.max(0, keepdims=True)
    # A token no shard owns would give 0/0; it cannot occur for a real top-k
    # list but the guard keeps the failure a zero rather than a NaN.
    all_empty = ~np.isfinite(shift)
    shift = np.where(all_empty, 0.0, shift)
    weights = np.exp(lses - shift[0])  # [D, T, N]
    denom = weights.sum(0)
    weights = np.where(
        denom[None] > 0, weights / np.where(denom == 0, 1, denom)[None], 0.0
    )
    return (weights[..., None] * outs).sum(0)


@pytest.mark.tpu_test
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
        kv_c = self.rng.standard_normal((DCP_TOKENS, LKV_DIM)).astype(np.float32)
        k_pe = self.rng.standard_normal((DCP_TOKENS, ROPE_DIM)).astype(np.float32)
        kv_c_fp8 = _quantize_fp8(kv_c, self.k_scale)
        k_pe_fp8 = _quantize_fp8(k_pe, self.k_scale)
        ql_nope = jnp.asarray(
            self.rng.standard_normal((DCP_TOKENS, NUM_HEADS, LKV_DIM)).astype(
                np.float32
            ),
            dtype=jnp.bfloat16,
        )
        q_pe = jnp.asarray(
            self.rng.standard_normal((DCP_TOKENS, NUM_HEADS, ROPE_DIM)).astype(
                np.float32
            ),
            dtype=jnp.bfloat16,
        )
        topk_rows = _causal_topk(list(range(DCP_TOKENS)), DCP_TOPK)
        block_tables = jnp.asarray(
            self.rng.permutation(TOTAL_PAGES)[:PAGES_PER_SEQ], dtype=jnp.int32
        )
        mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x",))

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
            k_scale=self.k_scale,
        )

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

        np.testing.assert_allclose(np.asarray(lse), expected, rtol=2e-2, atol=2e-2)

    @parameterized.named_parameters(
        dict(testcase_name="d2", dcp_size=2),
        dict(testcase_name="d4", dcp_size=4),
        dict(testcase_name="d8", dcp_size=8),
    )
    def test_merging_position_sharded_partials_reproduces_the_whole(self, dcp_size):
        """The core DCP claim, at the kernel boundary.

        Splitting a token's top-k across ranks by KV *position* and merging the
        per-rank `(output, lse)` pairs must reproduce the unsplit result. With
        `_causal_topk` the early tokens own fewer entries than there are ranks,
        so the empty-shard path is exercised too.
        """
        inp = self._prefilled()
        whole = np.asarray(
            self._attend(inp, inp["topk_rows"], False).astype(jnp.float32)
        )

        shards, counts = _shard_topk_by_owner(
            inp["topk_rows"], dcp_size, DCP_INTERLEAVE_C
        )
        self.assertTrue(
            (counts.sum(0) == (inp["topk_rows"] >= 0).sum(-1)).all(),
            "the shard split must partition the top-k list exactly",
        )

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
            self._attend(inp, inp["topk_rows"], False).astype(jnp.float32)
        )
        out, lse = self._attend(inp, inp["topk_rows"], True)

        # Two shards: one holds the entire list, the other holds nothing.
        empty_rows = np.full_like(inp["topk_rows"], -1)
        empty_rows[:, 0] = 0
        out_e, lse_e = self._attend(inp, empty_rows, True)
        counts = np.stack(
            [(inp["topk_rows"] >= 0).sum(-1), np.zeros(DCP_TOKENS, np.int32)]
        )

        merged = _merge_partials(
            np.stack(
                [
                    np.asarray(out.astype(jnp.float32)),
                    np.asarray(out_e.astype(jnp.float32)),
                ]
            ),
            np.stack([np.asarray(lse), np.asarray(lse_e)]),
            counts,
        )
        np.testing.assert_allclose(merged, whole, rtol=2e-2, atol=2e-2)

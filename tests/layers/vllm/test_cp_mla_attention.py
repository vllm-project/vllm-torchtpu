# SPDX-License-Identifier: Apache-2.0
"""The MLA DCP head gather and log-sum-exp merge (`cp_mla_attention`).

`attention_interface.merge_dcp_lse_partials` is the JAX statement of the same
arithmetic and is validated on hardware against a numpy MLA reference
(`SparseMlaDcpTest`). These tests hold the torch path to that already-checked
implementation, so the two cannot drift, and cover the collective plumbing the
JAX version has no equivalent of.

The process group is faked: every rank's partials are held in one process and
the group calls are served from that stack. That is a real test of the merge --
the arithmetic, the rank indexing, and the `-inf` masking are all exercised --
and it needs no distributed launcher.
"""

import multiprocessing

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.vllm import cp_mla_attention as cp_mla


class _FakeGroup:
    """Serves `all_gather`/`reduce_scatter` from a full stack of rank partials.

    `reduce_scatter` is where the fake has to be careful: the real collective
    sums a contribution from every rank, but only one rank's tensor is passed
    in. The other summands are produced by running the module's own
    `_weight_by_lse` as each peer -- rather than by reimplementing the
    weighting here, which would quietly replace the code under test with a
    second copy of itself and leave the real one unexercised.
    """

    def __init__(self, outputs, lses, rank):
        self.world_size = len(lses)
        self.rank_in_group = rank
        self._outputs = outputs  # [D, T, N, V] f32
        self._lses = lses  # [D, T, N] f32

    def all_gather(self, tensor, dim=0):
        assert dim == 0 and tensor.shape[0] == 1
        return self._lses.clone()

    def _reduced(self, dtype):
        """What the real all-reduce inside the reduce-scatter would sum."""
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
    """The merge as rank `r` sees it: its own head block of the whole answer."""
    group = _FakeGroup(outputs, lses, rank)
    monkeypatch.setattr(cp_mla, "get_dcp_group", lambda: group)
    return cp_mla.merge_lse_partials_scatter_heads(outputs[rank], lses[rank])


def _merge_whole(outputs, lses, monkeypatch):
    """The full merged tensor, reassembled from every rank's scatter block.

    The scatter deliberately leaves each rank only `num_heads // dcp_size`
    heads, so the whole-context tensor the arithmetic tests below want is the
    concatenation of what all `dcp_size` ranks got back.
    """
    return torch.cat([
        _merge_as_rank(outputs, lses, rank, monkeypatch)
        for rank in range(outputs.shape[0])
    ],
                     dim=1)


def _reference_merge(outputs, lses):
    """`o = sum_r softmax_r(lse) * o_r` in f64 numpy, owing nothing to the
    module -- an oracle rather than a second opinion."""
    out = outputs.numpy().astype(np.float64)
    lse = lses.numpy().astype(np.float64)
    shift = np.where(np.isfinite(lse), lse, -np.inf).max(axis=0)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    weights = np.where(np.isfinite(lse), np.exp(lse - shift[None]), 0.0)
    denom = weights.sum(axis=0)
    weights = weights / np.where(denom > 0.0, denom, 1.0)
    return (out * weights[..., None]).sum(axis=0)


def _jax_merge_worker(payload, queue):
    """Body of `_jax_merge`, run in its own process. Top-level, so it pickles."""
    outputs, lses = payload
    try:
        import jax.numpy as jnp

        from vllm_torchtpu.layers.common import attention_interface

        # The JAX twin arrives with the sparse-MLA DCP attention path; until
        # then this cross-check has nothing to compare against and the test
        # skips itself. It re-enables on its own once the symbol exists.
        jax_merge = getattr(attention_interface, "merge_dcp_lse_partials",
                            None)
        if jax_merge is None:
            queue.put(("UNAVAILABLE",
                       "attention_interface.merge_dcp_lse_partials is not "
                       "present yet"))
            return
        queue.put(
            ("OK",
             np.asarray(jax_merge(jnp.asarray(outputs), jnp.asarray(lses)))))
    except Exception:  # noqa: BLE001 - re-raised in the parent
        import traceback
        queue.put(("ERROR", traceback.format_exc()))


def _jax_merge(outputs, lses):
    """`merge_dcp_lse_partials` (the JAX twin), evaluated out of process.

    Spawned rather than called inline because importing the JAX side here
    initializes a PJRT client and takes the libtpu lockfile for the whole
    pytest process. `test_kv_scatter.py` sorts after this module and runs its
    own cases in spawned subprocesses, which then cannot get a client at all --
    so an inline call turns three unrelated tests red. Pinning JAX to the CPU
    backend does not avoid it; only keeping the parent clean does.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_jax_merge_worker,
                          args=((outputs.numpy(), lses.numpy()), queue))
    process.start()
    result = queue.get()
    process.join()
    status, payload = result
    if status == "ERROR":
        raise AssertionError(f"JAX reference merge failed:\n{payload}")
    if status == "UNAVAILABLE":
        pytest.skip(payload)
    return payload


def _partials(rng, dcp_size, tokens=8, local_heads=2, vdim=16, empty=()):
    """Per-rank partials, sized the way the kernel really produces them.

    `local_heads` is what a rank keeps *after* the scatter, so the merged head
    axis is `dcp_size * local_heads` -- which is also the divisibility the
    reduce-scatter requires.
    """
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
def test_torch_merge_matches_the_jax_merge(dcp_size, monkeypatch):
    outputs, lses = _partials(np.random.default_rng(20260907), dcp_size)
    got = _merge_whole(outputs, lses, monkeypatch).numpy()
    np.testing.assert_allclose(got,
                               _jax_merge(outputs, lses),
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
    """Single-rank and pre-init paths must not pay for, or require, DCP."""
    monkeypatch.setattr(cp_mla, "get_dcp_group", lambda: None)
    out = torch.ones(2, 3, 4)
    assert cp_mla.merge_lse_partials_scatter_heads(out, torch.zeros(2,
                                                                    3)) is out


def test_merge_reproduces_an_unsplit_softmax(monkeypatch):
    """Ground truth, independent of both implementations.

    Split one attention problem's keys across ranks, attend per shard, merge:
    the result must equal the softmax over all the keys at once.
    """
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

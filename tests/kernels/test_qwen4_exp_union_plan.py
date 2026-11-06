# SPDX-License-Identifier: Apache-2.0
"""Sort-free union plan (qsa_union_plan): integer-equal to union_plan.

``union_plan_nosort`` replaces the sort and searchsorted of
``qsa_union_prefill.union_plan`` with the served selection bitset, prefix sums
and unique-index scatters. Every output must equal the sorted reference
exactly. The edge cases are padded rows (with stale ids), rows without a
partial tail, rows shorter than one group, a row selecting its own tail,
future ids up to the last id below ``groups_cap``, -1 holes, an all-padding
group and unions that overflow ``union_cap``. Off TPU the Pallas bitset
kernel runs in interpret mode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_plan as nosort
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_prefill as up

RATIO, TOPK = 4, 512


@pytest.fixture
def kernels(monkeypatch):
    if jax.default_backend() != "tpu":
        from jax._src import tpu_info

        monkeypatch.setitem(
            tpu_info.registry,
            jax.devices()[0].device_kind,
            lambda: tpu_info._get_tpu_info_impl(tpu_info.ChipVersion.TPU_7X, 1),
        )
        original = pl.pallas_call

        def interpreted(*args, **kwargs):
            kwargs["interpret"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(bsp.pl, "pallas_call", interpreted)
    yield
    jax.clear_caches()


def _selection(seed, tokens, context, topk=TOPK):
    """Overlapping, scattered selections: local window + spans shared by nearby rows + noise."""
    rng = np.random.default_rng(seed)
    lengths = (context + np.arange(1, tokens + 1)).astype(np.int32)
    lengths[-3:] = 0
    blocks = np.full((tokens, topk), -1, np.int32)
    spans = rng.integers(0, context // RATIO - 64, size=64)
    for i, length in enumerate(lengths):
        complete = int(length) // RATIO
        if not complete:
            continue
        pool = set(range(max(0, complete - 48), complete))
        for s in spans[(i // 8) % 56: (i // 8) % 56 + 8]:
            pool |= set(range(int(s), min(int(s) + 12, complete)))
        pool |= set(rng.integers(0, complete, size=40).tolist())
        pick = rng.permutation(sorted(pool))[:topk]
        blocks[i, : len(pick)] = pick
    return blocks, lengths


def _edges(blocks, lengths, *, group, groups_cap, seed):
    """Edge cases on top of ``_selection`` (its ids are below each row's tail)."""
    rng = np.random.default_rng(seed)
    blocks, lengths = blocks.copy(), lengths.copy()
    tokens = len(lengths)
    blocks[rng.random(blocks.shape) < 0.1] = -1  # -1 holes inside rows
    lengths[0], blocks[0] = 3, -1  # shorter than one group: only its tail (id 0)
    lengths[1], blocks[1] = 4, -1  # one complete group, no tail
    blocks[1, 7] = 0
    lengths[2] = 9  # tail 2 selected explicitly, with id 0 and future ids
    blocks[2] = -1
    blocks[2, [3, 50, 51, 400]] = [2, 0, groups_cap - 1, 33]
    # Every fifth live row also selects its own tail and future ids (bits 0 and 31 included).
    for i in range(5, tokens, 5):
        if lengths[i] <= 0:
            continue
        tail = int(lengths[i]) // RATIO
        future = rng.choice(np.arange(tail + 1, groups_cap), size=6, replace=False)
        extra = np.concatenate([[tail], future, [groups_cap - 1, (tail // 32 + 1) * 32]])
        extra = np.unique(extra[extra < groups_cap])
        free = np.flatnonzero(blocks[i] < 0)[: len(extra)]
        blocks[i, rng.permutation(free)] = extra
    # Padded rows keep stale ids, which must be ignored.
    blocks[-2] = blocks[7]
    blocks[-1] = blocks[11]
    # One whole group of padding rows: an empty union.
    lengths[group: 2 * group] = 0
    return blocks, lengths


def _cap(blocks, lengths, group):
    return int(-(-up.union_sizes(blocks, lengths, group=group).max() // 32) * 32)


def _run_both(blocks, lengths, *, group, union_cap, groups_cap, shift=0, rank_kernel=False):
    want = up.union_plan(jnp.asarray(blocks), jnp.asarray(lengths), group=group,
                         union_cap=union_cap, shift=shift)
    got = nosort.union_plan_nosort(jnp.asarray(blocks), jnp.asarray(lengths), group=group,
                                   union_cap=union_cap, groups_cap=groups_cap, shift=shift,
                                   rank_kernel=rank_kernel)
    return [np.asarray(x) for x in want], [np.asarray(x) for x in got]


def _assert_equal(want, got):
    for name, w, g in zip(("union", "vblocks", "vlengths", "usize"), want, got):
        assert w.shape == g.shape and g.dtype == np.int32, name
        np.testing.assert_array_equal(g, w, err_msg=name)


@pytest.mark.parametrize("edges", [False, True])
@pytest.mark.parametrize("group", [8, 16, 32, 64])
def test_nosort_equals_union_plan(kernels, group, edges):
    tokens, context = 256, 5000
    blocks, lengths = _selection(group, tokens, context)
    # Not a multiple of 32: the last bitset word is partial.
    groups_cap = (context + tokens) // RATIO + 45
    if edges:
        blocks, lengths = _edges(blocks, lengths, group=group, groups_cap=groups_cap, seed=group)
    cap = _cap(blocks, lengths, group)
    want, got = _run_both(blocks, lengths, group=group, union_cap=cap, groups_cap=groups_cap)
    _assert_equal(want, got)
    np.testing.assert_array_equal(got[3], up.union_sizes(blocks, lengths, group=group))
    if edges:
        # The edge cases are really present in the compared outputs.
        assert (got[3] == 0).any() and (got[0] == groups_cap - 1).any()
        assert (lengths[lengths > 0] % RATIO == 0).any()
        assert (blocks[lengths > 0] == (lengths[lengths > 0] // RATIO)[:, None]).any(axis=1).sum() > 10


@pytest.mark.parametrize("group", [16, 64])
def test_nosort_union_overflow(kernels, group):
    """usize stays exact past union_cap; the truncated outputs also match union_plan."""
    tokens, context = 256, 5000
    groups_cap = (context + tokens) // RATIO + 32
    blocks, lengths = _edges(*_selection(group + 1, tokens, context), group=group,
                             groups_cap=groups_cap, seed=group + 1)
    sizes = up.union_sizes(blocks, lengths, group=group)
    cap = int(sizes[sizes > 0].min() + sizes.max()) // 2
    assert (sizes > cap).any() and (sizes[sizes > 0] <= cap).any()
    want, got = _run_both(blocks, lengths, group=group, union_cap=cap, groups_cap=groups_cap)
    np.testing.assert_array_equal(got[3], sizes)
    _assert_equal(want, got)


def test_nosort_shift_matches_union_plan(kernels):
    blocks, lengths = _selection(3, 256, 5000)
    cap = _cap(blocks, lengths, 16)
    want, got = _run_both(blocks, lengths, group=16, union_cap=cap, groups_cap=1400, shift=1)
    _assert_equal(want, got)


def test_nosort_drops_ids_at_or_above_groups_cap(kernels):
    """Ids >= groups_cap are invalid (as in _selection_words): same as if they were -1."""
    group, tokens, context = 16, 256, 5000
    groups_cap = (context + tokens) // RATIO + 3
    blocks, lengths = _selection(5, tokens, context)
    # A row whose tail id is groups_cap itself (length a multiple of the ratio,
    # so the tail is only a rank query), next to a row selecting the last id.
    lengths[20] = RATIO * groups_cap
    blocks[21, -3] = groups_cap - 1
    big = blocks.copy()
    big[::7, -1] = groups_cap + np.arange(len(big[::7]))
    big[1::7, -2] = groups_cap
    cap = _cap(blocks, lengths, group)
    want = up.union_plan(jnp.asarray(blocks), jnp.asarray(lengths), group=group, union_cap=cap)
    got = nosort.union_plan_nosort(jnp.asarray(big), jnp.asarray(lengths), group=group,
                                   union_cap=cap, groups_cap=groups_cap)
    _assert_equal([np.asarray(x) for x in want], [np.asarray(x) for x in got])


@pytest.mark.parametrize("edges", [False, True])
@pytest.mark.parametrize("group", [8, 32, 64])
def test_rank_kernel_equals_union_plan(kernels, group, edges):
    """The Pallas rank pass gives the same plan as the gathers (and as union_plan)."""
    tokens, context = 256, 5000
    blocks, lengths = _selection(group + 7, tokens, context)
    groups_cap = (context + tokens) // RATIO + 45
    if edges:
        blocks, lengths = _edges(blocks, lengths, group=group, groups_cap=groups_cap, seed=group + 7)
    cap = _cap(blocks, lengths, group)
    want, got = _run_both(blocks, lengths, group=group, union_cap=cap, groups_cap=groups_cap, rank_kernel=True)
    _assert_equal(want, got)


def test_rank_kernel_overflow_and_shift(kernels):
    tokens, context, group = 256, 5000, 16
    groups_cap = (context + tokens) // RATIO + 32
    blocks, lengths = _edges(*_selection(23, tokens, context), group=group, groups_cap=groups_cap, seed=23)
    sizes = up.union_sizes(blocks, lengths, group=group)
    cap = int(sizes[sizes > 0].min() + sizes.max()) // 2
    _assert_equal(*_run_both(blocks, lengths, group=group, union_cap=cap, groups_cap=groups_cap, rank_kernel=True))
    cap = _cap(blocks, lengths, group)
    _assert_equal(*_run_both(blocks, lengths, group=group, union_cap=cap, groups_cap=groups_cap, shift=1,
                             rank_kernel=True))

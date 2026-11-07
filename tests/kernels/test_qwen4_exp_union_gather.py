# SPDX-License-Identifier: Apache-2.0
"""SparseCore union gather, step 1: the XLA reference list and gather, and the kernel on gathered tiles.

The walk (``union_walk_kernel(fast_walk=True)``) reads each query group's union
from the paged pool one 4-token DMA per union group, with a scalar walk over the
union bitset (i1j: that walk, not the copies, is what costs). The gathered path
moves the walk out of the attention kernel: a list step compacts each group's
union into pool rows in rank order, a gather step copies those rows and their
mask lines into contiguous HBM, and the attention kernel reads each tile as two
contiguous copies. Here both steps are the XLA references; the SparseCore
kernels must produce the same arrays. Off TPU the Pallas kernels run in
interpret mode.

Checks: the list is each group's union in ascending group id, padded with word
0's first row; the kernel on gathered tiles is bitwise the walk (one page row,
per-group page rows, both line forms); mutants of the list and the lines fail.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_prefill as up
from vllm_torchtpu.kernels.qwen4_exp.qsa_union_plan import union_bits_plan

RATIO, TOPK, PAGE = 4, 512, 3200


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


def _inputs(seed, tokens=256, context=6000, pages=3):
    """As test_qwen4_exp_union_dense: overlapping scattered selections on a permuted page table."""
    rng = np.random.default_rng(seed)
    lengths = (context + np.arange(1, tokens + 1)).astype(np.int32)
    lengths[-3:] = 0
    blocks = np.full((tokens, TOPK), -1, np.int32)
    spans = rng.integers(0, context // RATIO - 64, size=64)
    for i, length in enumerate(lengths):
        complete = int(length) // RATIO
        if not complete:
            continue
        chosen = set(range(max(0, complete - 48), complete))
        for s in spans[(i // 8) % 56: (i // 8) % 56 + 8]:
            chosen |= set(range(int(s), min(int(s) + 12, complete)))
        chosen |= set(rng.integers(0, complete, size=40).tolist())
        pick = rng.permutation(sorted(chosen))[:TOPK]
        blocks[i, : len(pick)] = pick
    q = jnp.asarray(rng.normal(0, 0.5, (tokens, 24, 256)), jnp.bfloat16)
    raw = rng.integers(0, 0x40, size=(pages, PAGE, 1, 4, 256), dtype=np.uint8)
    raw |= rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    table_row = jnp.asarray(rng.permutation(pages), jnp.int32)
    return q, jnp.asarray(blocks), jnp.asarray(lengths), pool, table_row


def _plan(seed, group):
    q, blocks, lengths, pool, table_row = _inputs(seed)
    cap = int(-(-up.union_sizes(blocks, lengths, group=group).max() // 32) * 32)
    plan = union_bits_plan(blocks, lengths, group=group, union_cap=cap, groups_cap=3 * PAGE // RATIO)
    return q, blocks, lengths, pool, table_row, cap, plan


def test_list_is_the_union_in_rank_order(kernels):
    """rows/ids: each group's union groups ascending (the walk's order), their pool rows, then padding."""
    group, kv_tile = 64, 512
    q, blocks, lengths, pool, table_row, cap, plan = _plan(3, group)
    bits, ubits, usize, vlengths = plan
    rows, ids = ud.union_gather_list_reference(ubits, usize, table_row, ratio=RATIO, page_size=PAGE,
                                               union_cap=cap, kv_tile=kv_tile)
    rows, ids, table = np.asarray(rows), np.asarray(ids), np.asarray(table_row)
    blocks, lengths = np.asarray(blocks), np.asarray(lengths)
    assert rows.shape[1] % (kv_tile // RATIO) == 0 and rows.shape[1] >= cap
    for g in range(rows.shape[0]):
        rows_g = slice(g * group, (g + 1) * group)
        union = set(blocks[rows_g][blocks[rows_g] >= 0].tolist())
        live = lengths[rows_g] > 0
        partial = live & (lengths[rows_g] % RATIO != 0)
        union |= set((lengths[rows_g][partial] // RATIO).tolist())
        expect = np.asarray(sorted(union), np.int64)
        n = len(expect)
        assert int(usize[g]) == n
        np.testing.assert_array_equal(ids[g, :n], expect)
        assert (ids[g, n:] == -1).all()
        token = expect * RATIO
        np.testing.assert_array_equal(rows[g, :n], table[token // PAGE] * PAGE + token % PAGE)
        assert (rows[g, n:] == table[0] * PAGE).all()  # word 0's first row, as the walk's padding


@pytest.mark.parametrize("group,kv_tile,line_matmul,per_group_rows",
                         [(64, 512, False, False), (128, 1024, True, False), (32, 256, False, True)])
def test_gathered_kernel_is_the_walk_bit_for_bit(kernels, group, kv_tile, line_matmul, per_group_rows):
    q, blocks, lengths, pool, table_row, cap, plan = _plan(group + kv_tile, group)
    bits, ubits, usize, vlengths = plan
    ngroups = ubits.shape[0]
    rows_arg = jnp.broadcast_to(table_row, (ngroups, table_row.shape[0])) if per_group_rows else table_row
    options = dict(group=group, union_cap=cap, kv_tile=kv_tile, lean=True, fast_walk=True,
                   line_matmul=line_matmul)
    walk = np.asarray(ud.union_walk_kernel(q, *plan, pool, rows_arg, **options))
    rows, ids = ud.union_gather_list_reference(ubits, usize, rows_arg, ratio=RATIO, page_size=PAGE,
                                               union_cap=cap, kv_tile=kv_tile)
    gathered = ud.union_gather_reference(pool, bits, rows, ids, ratio=RATIO, line_matmul=line_matmul)
    jax.clear_caches()
    got = np.asarray(ud.union_walk_kernel(q, *plan, pool, rows_arg, gathered=gathered, **options))
    np.testing.assert_array_equal(got.view(np.uint16), walk.view(np.uint16))
    assert np.abs(walk.astype(np.float32)).max() > 0


def test_gathered_kernel_sees_a_wrong_list_or_wrong_lines(kernels):
    """Mutants of the gathered inputs change the output (the kernel really reads them)."""
    group, kv_tile = 64, 512
    q, blocks, lengths, pool, table_row, cap, plan = _plan(7, group)
    bits, ubits, usize, vlengths = plan
    options = dict(group=group, union_cap=cap, kv_tile=kv_tile, lean=True, fast_walk=True)
    rows, ids = ud.union_gather_list_reference(ubits, usize, table_row, ratio=RATIO, page_size=PAGE,
                                               union_cap=cap, kv_tile=kv_tile)
    good = np.asarray(ud.union_walk_kernel(
        q, *plan, pool, table_row, gathered=ud.union_gather_reference(pool, bits, rows, ids, ratio=RATIO,
                                                                       line_matmul=False), **options))
    # (a) every rank's rows one group later; (b) each rank's line from its neighbour's group.
    shifted = ud.union_gather_reference(pool, bits, rows + RATIO, ids, ratio=RATIO, line_matmul=False)
    neighbour = ud.union_gather_reference(pool, bits, rows, jnp.where(ids >= 0, ids + 1, -1), ratio=RATIO,
                                          line_matmul=False)
    for mutant in (shifted, neighbour):
        jax.clear_caches()
        bad = np.asarray(ud.union_walk_kernel(q, *plan, pool, table_row, gathered=mutant, **options))
        assert not np.array_equal(bad.view(np.uint16), good.view(np.uint16))


def test_gathered_options_are_checked(kernels):
    group, kv_tile = 64, 512
    q, blocks, lengths, pool, table_row, cap, plan = _plan(9, group)
    gathered = (jnp.zeros((8, 1, 4, 256), pool.dtype), jnp.zeros((8, 128), jnp.int32))
    for bad in (dict(fast_walk=False), dict(fast_walk=True, walk_runs=8), dict(fast_walk=True, scan_list=True)):
        with pytest.raises(ValueError, match="gathered"):
            ud.union_walk_kernel(q, *plan, pool, table_row, group=group, union_cap=cap, kv_tile=kv_tile,
                                 gathered=gathered, **bad)


# --- listed walk (SparseCore list only): the walk's per-group copies issued from a precomputed id list -----------


@pytest.mark.parametrize("group,kv_tile,line_matmul,per_group_rows",
                         [(64, 512, False, False), (128, 1024, True, False), (32, 256, False, True)])
def test_listed_kernel_is_the_walk_bit_for_bit(kernels, group, kv_tile, line_matmul, per_group_rows):
    q, blocks, lengths, pool, table_row, cap, plan = _plan(group + kv_tile + 3, group)
    bits, ubits, usize, vlengths = plan
    ngroups = ubits.shape[0]
    rows_arg = jnp.broadcast_to(table_row, (ngroups, table_row.shape[0])) if per_group_rows else table_row
    options = dict(group=group, union_cap=cap, kv_tile=kv_tile, lean=True, fast_walk=True,
                   line_matmul=line_matmul)
    walk = np.asarray(ud.union_walk_kernel(q, *plan, pool, rows_arg, **options))
    _, ids = ud.union_gather_list_reference(ubits, usize, rows_arg, ratio=RATIO, page_size=PAGE,
                                            union_cap=cap, kv_tile=kv_tile)
    jax.clear_caches()
    got = np.asarray(ud.union_walk_kernel(q, *plan, pool, rows_arg, listed=ids, **options))
    np.testing.assert_array_equal(got.view(np.uint16), walk.view(np.uint16))


def test_listed_kernel_reads_the_list(kernels):
    """A wrong list (every id one group later) changes the output; a list of the wrong shape is refused."""
    group, kv_tile = 64, 512
    q, blocks, lengths, pool, table_row, cap, plan = _plan(11, group)
    bits, ubits, usize, vlengths = plan
    options = dict(group=group, union_cap=cap, kv_tile=kv_tile, lean=True, fast_walk=True)
    _, ids = ud.union_gather_list_reference(ubits, usize, table_row, ratio=RATIO, page_size=PAGE,
                                            union_cap=cap, kv_tile=kv_tile)
    good = np.asarray(ud.union_walk_kernel(q, *plan, pool, table_row, listed=ids, **options))
    jax.clear_caches()
    bad = np.asarray(ud.union_walk_kernel(q, *plan, pool, table_row, listed=jnp.where(ids >= 0, ids + 1, -1),
                                          **options))
    assert not np.array_equal(bad.view(np.uint16), good.view(np.uint16))
    with pytest.raises(ValueError, match="listed ids"):
        ud.union_walk_kernel(q, *plan, pool, table_row, listed=ids[:, :-1], **options)
    with pytest.raises(ValueError, match="listed walk"):
        ud.union_walk_kernel(q, *plan, pool, table_row, listed=ids, walk_runs=8, **options)


@pytest.mark.parametrize("group,kv_tile,line_matmul", [(64, 512, False), (128, 1024, True)])
def test_listed_kernel_with_precomputed_lines_is_the_walk(kernels, group, kv_tile, line_matmul):
    """listed_lines: the listed walk's copies with each tile's lines read as one copy (step 4 probe): bitwise."""
    q, blocks, lengths, pool, table_row, cap, plan = _plan(group + kv_tile + 5, group)
    bits, ubits, usize, vlengths = plan
    options = dict(group=group, union_cap=cap, kv_tile=kv_tile, lean=True, fast_walk=True, line_matmul=line_matmul)
    walk = np.asarray(ud.union_walk_kernel(q, *plan, pool, table_row, **options))
    rows, ids = ud.union_gather_list_reference(ubits, usize, table_row, ratio=RATIO, page_size=PAGE,
                                               union_cap=cap, kv_tile=kv_tile)
    _, lines = ud.union_gather_reference(pool, bits, rows, ids, ratio=RATIO, line_matmul=line_matmul)
    jax.clear_caches()
    got = np.asarray(ud.union_walk_kernel(q, *plan, pool, table_row, listed=ids, listed_lines=lines, **options))
    np.testing.assert_array_equal(got.view(np.uint16), walk.view(np.uint16))
    with pytest.raises(ValueError, match="listed_lines"):
        ud.union_walk_kernel(q, *plan, pool, table_row, listed_lines=lines, **options)

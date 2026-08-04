# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the shared (sequence, tile) planner.

Pure index arithmetic, so this runs on CPU. It pins the properties kernels rely
on: that the tiles partition each sequence's tokens exactly, that ``num_tiles``
tracks real content rather than the static grid, and that ``start_seq`` /
``end_seq`` can restrict a kernel to part of a batch with no host branch.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import varlen_tiles

CHUNK = 64


def _plan(query_lens,
          *,
          max_seqs=None,
          max_tiles=16,
          seq_lens=None,
          start_seq=0,
          end_seq=None,
          chunk_size=CHUNK):
    """Plan for a batch, padding the request axis the way the runner does."""
    max_seqs = max_seqs if max_seqs is not None else len(query_lens)
    total = sum(query_lens)
    cu = [0]
    for q in query_lens:
        cu.append(cu[-1] + q)
    cu += [total] * (max_seqs - len(query_lens))  # padded tail: query_len 0
    lens = list(seq_lens if seq_lens is not None else query_lens)
    lens += [0] * (max_seqs - len(lens))
    return varlen_tiles.plan_per_seq_tiles(
        jnp.asarray(lens, jnp.int32),
        jnp.asarray(cu, jnp.int32),
        max_tiles=max_tiles,
        chunk_size=chunk_size,
        tile_size=chunk_size,
        start_seq=start_seq,
        end_seq=len(query_lens) if end_seq is None else end_seq,
    )


def _covered_rows(plan, num_tiles):
    """The token rows the first ``num_tiles`` tiles actually read."""
    rows: list[int] = []
    for p in range(num_tiles):
        base = int(plan.p_id_to_r_base[p])
        size = int(plan.p_id_to_r_size[p])
        rows.extend(range(base, base + size))
    return rows


@pytest.mark.parametrize("query_lens", [
    [64],
    [1],
    [65],
    [127],
    [64, 64],
    [1, 1, 1],
    [17, 48, 96],
    [2048],
])
def test_tiles_partition_the_token_axis(query_lens):
    """Every token is read exactly once, and no padding row is read.

    This is the property that lets the alignment gather go away: the tiles index
    the unaligned token axis directly, so together they must reproduce it.
    """
    plan = _plan(query_lens, max_tiles=64)
    num_tiles = int(plan.num_tiles)
    assert num_tiles == sum(-(-q // CHUNK) for q in query_lens)
    assert _covered_rows(plan, num_tiles) == list(range(sum(query_lens)))


def test_last_tile_is_short_when_the_length_is_not_a_chunk_multiple():
    """A ragged tail must be reported, not rounded up.

    Rounding up is what the aligned layout did by padding with zeros; here the
    kernel has to be told to fetch fewer rows instead.
    """
    plan = _plan([70])
    assert int(plan.num_tiles) == 2
    assert [int(x) for x in plan.p_id_to_r_size[:2]] == [64, 6]
    assert [int(x) for x in plan.p_id_to_r_base[:2]] == [0, 64]


def test_first_and_last_tile_flags_mark_the_state_boundaries():
    """State is read on a sequence's first tile and written on its last."""
    plan = _plan([130, 64])  # 3 tiles, then 1
    n = int(plan.num_tiles)
    assert n == 4
    assert [bool(x) for x in plan.p_id_is_first_tile[:n]
            ] == [True, False, False, True]
    assert [bool(x)
            for x in plan.p_id_is_last_tile[:n]] == [False, False, True, True]
    assert [int(x) for x in plan.p_id_to_s_idx[:n]] == [0, 0, 0, 1]


def test_a_sequence_with_no_scheduled_tokens_contributes_no_tiles():
    """Padded request slots, and requests a chunk scheduled nothing for."""
    plan = _plan([64, 0, 64], max_tiles=16)
    n = int(plan.num_tiles)
    assert n == 2
    # Sequence 1 is skipped entirely: no tile maps to it.
    assert 1 not in [int(x) for x in plan.p_id_to_s_idx[:n]]
    assert [int(x) for x in plan.p_id_to_s_idx[:n]] == [0, 2]


def test_padded_request_axis_does_not_add_tiles():
    """`num_tiles` tracks content, not the static request bucket.

    The whole point of the device-valued count: the grid is sized for the worst
    case but the cost must follow the batch.
    """
    for max_seqs in (1, 8, 160):
        plan = _plan([64], max_seqs=max_seqs, max_tiles=256)
        assert int(plan.num_tiles) == 1, max_seqs


def test_end_seq_can_exclude_every_sequence():
    """An empty range must plan zero tiles.

    This is what makes GDN-style dispatch affordable: the prefill pass is handed
    an empty range on a decode-only step and a kernel bounded by `num_tiles`
    then does nothing.
    """
    plan = _plan([1] * 8, max_seqs=8, max_tiles=32, end_seq=0)
    assert int(plan.num_tiles) == 0


def test_start_seq_rotates_the_batch():
    """Planning [start_seq, end_seq) must address the right rows.

    Splitting a batch into a decode segment and a prefill segment means the
    second pass starts partway in, and its tiles must still point at that
    segment's tokens in the *unrotated* activation layout.
    """
    query_lens = [1, 1, 128]  # two decodes, then a prefill
    plan = _plan(query_lens, max_seqs=3, max_tiles=32, start_seq=2, end_seq=3)
    n = int(plan.num_tiles)
    assert n == 2  # 128 tokens -> 2 tiles
    # The prefill's tokens start at row 2, after the two decode tokens.
    assert _covered_rows(plan, n) == list(range(2, 130))


def test_has_initial_state_is_derived_from_seq_lens():
    """Carry-in is `seq_len > query_len`, per sequence."""
    plan = _plan([8, 8], seq_lens=[8, 40], max_seqs=2)
    assert [bool(x) for x in plan.s_idx_has_initial_state[:2]] == [False, True]


def test_roll_to_start_seq_matches_the_planner():
    """Callers rotate their own payloads; this must agree with the planner."""
    x = jnp.asarray([10, 11, 12, 13], jnp.int32)
    got = varlen_tiles.roll_to_start_seq(x, 2)
    np.testing.assert_array_equal(np.asarray(got), [12, 13, 10, 11])

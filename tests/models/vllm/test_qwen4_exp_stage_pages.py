# SPDX-License-Identifier: Apache-2.0
"""StagePageMap: a pipeline stage's pages behind a larger block-ID space."""

import dataclasses
import importlib.util
import pathlib
from typing import NamedTuple

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src/vllm_torchtpu/models/vllm/qwen4_exp/stage_pages.py"
)
_spec = importlib.util.spec_from_file_location("stage_pages", _PATH)
stage_pages = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage_pages)
StagePageMap = stage_pages.StagePageMap
StagePagesExhausted = stage_pages.StagePagesExhausted


@dataclasses.dataclass
class NewReq:
    req_id: str
    block_ids: tuple


@dataclasses.dataclass
class Cached:
    req_ids: list
    new_block_ids: list
    resumed_req_ids: set = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class Out:
    scheduled_new_reqs: list
    scheduled_cached_reqs: Cached
    new_block_ids_to_zero: list | None = None
    kv_cache_block_copies: list | None = None
    block_table_updates: dict | None = None


class Copy(NamedTuple):
    src_block_id: int
    dst_block_id: int


def out(new=(), cached=(), zeros=None, copies=None):
    return Out(
        [NewReq(r, t) for r, t in new],
        Cached([r for r, _ in cached], [t for _, t in cached]),
        zeros,
        copies,
    )


# Groups 0 and 2 live on this stage, group 1 on another.
def make(physical=8, logical=32):
    return StagePageMap(physical, logical, own_groups=[0, 2], num_groups=3)


def test_own_groups_get_distinct_pages_foreign_groups_page_zero():
    m = make()
    t = m.translate(out(new=[("a", ([5, 9], [6, 10], [7]))]))
    table = t.scheduled_new_reqs[0].block_ids
    assert table[1] == [0, 0]
    pages = table[0] + table[2]
    assert len(set(pages)) == 3 and all(0 < p < 8 for p in pages)
    assert m.mapped == 3


def test_null_block_stays_page_zero():
    m = make()
    t = m.translate(out(new=[("a", ([0, 4], [0, 0], [0]))]))
    assert t.scheduled_new_reqs[0].block_ids[0][0] == 0
    assert t.scheduled_new_reqs[0].block_ids[2] == [0]
    assert m.mapped == 1


def test_prefix_hit_reuses_page_and_foreign_reallocation_frees_it():
    m = make()
    first = m.translate(out(new=[("a", ([5], [6], [7]))]))
    page5 = first.scheduled_new_reqs[0].block_ids[0][0]
    # A later request hits block 5 of group 0: same page.
    hit = m.translate(out(new=[("b", ([5, 11], [12], [13]))]))
    assert hit.scheduled_new_reqs[0].block_ids[0][0] == page5
    # The scheduler evicts block 5 and hands it to the other stage's group.
    m.translate(out(cached=[("b", ([], [5], []))]))
    assert m.page_of(5) is None
    with pytest.raises(ValueError):
        m.lookup(0, [5])
    # The freed page is reused first (LIFO).
    again = m.translate(out(cached=[("b", ([20], [], []))]))
    assert again.scheduled_cached_reqs.new_block_ids[0][0][0] == page5


def test_reassignment_between_own_groups_keeps_the_page():
    m = make()
    t = m.translate(out(new=[("a", ([5], [], []))]))
    page = t.scheduled_new_reqs[0].block_ids[0][0]
    t = m.translate(out(new=[("b", ([], [], [5]))]))
    assert t.scheduled_new_reqs[0].block_ids[2] == [page]
    assert m.lookup(2, [5]) == (page,)
    with pytest.raises(ValueError):
        m.lookup(0, [5])


def test_zero_and_copy_lists_resolve_after_allocation():
    m = make()
    t = m.translate(
        out(
            new=[("a", ([3, 4], [8], []))],
            zeros=[3, 8, 4],
            copies=[Copy(3, 4), Copy(30, 8)],
        )
    )
    p3, p4 = t.scheduled_new_reqs[0].block_ids[0]
    assert t.new_block_ids_to_zero == [p3, p4]
    # The copy into another stage's block is dropped.
    assert t.kv_cache_block_copies == [Copy(p3, p4)]


def test_cached_none_rows_pass_through():
    m = make()
    t = m.translate(out(cached=[("a", None)]))
    assert t.scheduled_cached_reqs.new_block_ids == [None]


def test_lookup_of_foreign_group_is_page_zero():
    m = make()
    assert m.lookup(1, [17, 18]) == (0, 0)


def test_exhaustion_raises_with_guidance():
    m = make(physical=3, logical=32)
    m.translate(out(new=[("a", ([1, 2], [], []))]))
    with pytest.raises(StagePagesExhausted, match="LOGICAL_BLOCKS_FACTOR"):
        m.translate(out(new=[("b", ([3], [], []))]))


def test_out_of_range_block_id_rejected():
    m = make()
    with pytest.raises(ValueError):
        m.translate(out(new=[("a", ([32], [], []))]))


def test_wrong_group_count_rejected():
    m = make()
    with pytest.raises(ValueError):
        m.translate(out(new=[("a", ([1], [2]))]))


def test_steady_state_demand_tracks_own_share():
    """24 token groups, 4 of them own: a 4x scheduler pool cycling through
    LRU-like reuse never needs more than ~1/6 of the IDs as pages."""
    import random

    groups, own = 24, [0, 1, 12, 13]
    physical, logical = 1000, 4000
    m = StagePageMap(physical, logical, own, groups)
    rng = random.Random(0)
    free = list(range(1, logical))
    rng.shuffle(free)
    live = []
    for _ in range(3000):
        # Allocate one block column: one ID per group.
        if len(free) < groups:
            for col in live[:10]:
                free.extend(col)
            live = live[10:]
        col = [free.pop(0) for _ in range(groups)]
        live.append(col)
        m.translate(out(new=[("r", tuple([b] for b in col))]))
    assert m.peak <= logical * len(own) // groups + groups

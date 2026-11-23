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
"""Tests for ``RequestSlotAllocator`` and ``RequestSlotTable``."""

from typing import Any

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner import request_slot_table
from vllm_torchtpu.runner.request_slot_table import (
    RequestSlotAllocator,
    RequestSlotTable,
)

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _eager_seed_kernel(request: pytest.FixtureRequest):
    """Run the ``@torch.compile(backend="tpu")`` seed kernel eagerly on CPU,
    except in TPU tests, which run it compiled."""
    if request.node.get_closest_marker("tpu_test"):
        yield
        return
    with torch.compiler.set_stance("force_eager"):
        yield


def _cold_rows(rows: dict[str, tuple[int, ...]], calls: list[str]):
    def compute_cold_row(req_id: str) -> tuple[int, ...]:
        calls.append(req_id)
        return rows[req_id]

    return compute_cold_row


def _row(pool: RequestSlotTable, table: torch.Tensor, req_id: str) -> list[int]:
    slot = pool.allocator.slot_for(req_id)
    assert slot is not None
    return table[slot].tolist()


def test_allocator_leases_from_slot_one_and_reuses_freed_slots() -> None:
    alloc = RequestSlotAllocator(max_num_reqs=2)
    assert alloc.max_num_reqs == 2
    assert alloc.lease("a") == 1 and alloc.lease("a") == 1
    assert alloc.lease("b") == 2
    assert set(alloc.allocated_req_ids()) == {"a", "b"}
    with pytest.raises(RuntimeError, match="all 2 slots are leased"):
        alloc.lease("c")
    alloc.release(["a", "never-leased"])
    assert alloc.slot_for("a") is None
    assert alloc.lease("c") == 1


def test_constructors_validate_their_arguments() -> None:
    with pytest.raises(ValueError, match="max_num_reqs must be positive, got 0"):
        RequestSlotAllocator(max_num_reqs=0)
    with pytest.raises(ValueError, match="max_num_reqs must be positive, got 0"):
        RequestSlotTable(max_num_reqs=0, num_cols=1, device=CPU)
    with pytest.raises(ValueError, match="num_cols must be positive"):
        RequestSlotTable(max_num_reqs=2, num_cols=0, device=CPU)
    with pytest.raises(ValueError, match=r"row \(1,\) has 1 values, not 2"):
        RequestSlotTable(max_num_reqs=2, num_cols=2, device=CPU, sentinel_row=(1,))
    not_ints: Any = (0.5,)
    with pytest.raises(TypeError):
        RequestSlotTable(max_num_reqs=2, num_cols=1, device=CPU, sentinel_row=not_ints)


def test_sync_step_leases_slots_and_seeds_staged_and_cold_rows() -> None:
    pool: RequestSlotTable[tuple[int, int]] = RequestSlotTable(
        max_num_reqs=2, num_cols=2, device=CPU, sentinel_row=(-99, 0)
    )
    assert pool.max_num_reqs == 2 and pool.device == CPU
    rows = {"a": (10, 1), "b": (20, 2), "c": (30, 3)}
    pool.mark_dirty("a", rows["a"])
    assert not pool.needs_cold_reseed(["a"], rows)
    assert pool.needs_cold_reseed(["b"], rows)
    assert not pool.needs_cold_reseed(["untracked", None], rows)

    cold_calls: list[str] = []
    cold = _cold_rows(rows, cold_calls)
    # Untracked and empty rows lease nothing and seed nothing.
    tbl = pool.sync_step(["a", None, "untracked", "b"], rows, cold)
    assert tbl is not None and tbl.shape == (3, 2)
    assert cold_calls == ["b"]
    assert tbl[0].tolist() == [-99, 0]
    assert _row(pool, tbl, "a") == [10, 1] and _row(pool, tbl, "b") == [20, 2]
    assert pool.allocator.slot_for("untracked") is None
    slots = (pool.allocator.slot_for("a"), pool.allocator.slot_for("b"))

    # Requests that keep their slot keep their row, without a cold row.
    assert pool.sync_step(["b", "a"], rows, cold) is tbl
    assert (pool.allocator.slot_for("a"), pool.allocator.slot_for("b")) == slots

    # "a" leaves the batch and loses its slot; "c" and later "a" get cold rows.
    tbl = pool.sync_step(["b", "c"], rows, cold)
    assert tbl is not None and "a" not in pool.allocator.allocated_req_ids()
    assert cold_calls == ["b", "c"]
    tbl = pool.sync_step(["a", "c"], rows, cold)
    assert tbl is not None and cold_calls == ["b", "c", "a"]
    assert pool.sync_step(["untracked"], rows, cold) is None


def test_mark_dirty_rewrites_the_row_of_a_request_that_keeps_its_slot() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=2, num_cols=1, device=CPU
    )
    rows = {"a": (7,)}
    cold_calls: list[str] = []
    cold = _cold_rows(rows, cold_calls)
    tbl = pool.sync_step(["a"], rows, cold)
    assert tbl is not None and cold_calls == ["a"]
    slot = pool.allocator.slot_for("a")

    pool.mark_dirty("a", (8,))
    tbl = pool.sync_step(["a"], rows, cold)
    assert tbl is not None and pool.allocator.slot_for("a") == slot
    assert tbl[slot].tolist() == [8]
    assert cold_calls == ["a"]


def test_a_request_listed_twice_is_handled_once() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=2, num_cols=1, device=CPU
    )
    rows = {"a": (5,)}
    cold_calls: list[str] = []
    tbl = pool.sync_step(["a", "a"], rows, _cold_rows(rows, cold_calls))
    assert tbl is not None and cold_calls == ["a"]
    assert set(pool.allocator.allocated_req_ids()) == {"a"}
    assert _row(pool, tbl, "a") == [5]


def test_untracked_only_step_releases_slots() -> None:
    """A batch without tracked requests still releases the slots of the ones
    that left, so a returning request gets a cold row instead of keeping its
    old one."""
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=2, num_cols=1, device=CPU
    )
    rows = {"a": (1,)}
    cold_calls: list[str] = []
    cold = _cold_rows(rows, cold_calls)
    assert pool.sync_step(["a"], rows, cold) is not None
    assert pool.sync_step(["x", None], rows, cold) is None
    assert not pool.allocator.allocated_req_ids()
    # "a" returns with a state that moved on while it was away.
    rows["a"] = (5,)
    tbl = pool.sync_step(["a"], rows, cold)
    assert tbl is not None and cold_calls == ["a", "a"]
    assert _row(pool, tbl, "a") == [5]


def test_sync_step_seeds_new_rows_in_one_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every seeded row has its own slot, so one scatter seeds a full pool."""
    seeded: list[int] = []
    seed = request_slot_table.seed_slot_rows

    def counting_seed(
        table: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor
    ) -> torch.Tensor:
        seeded.append(int((slots > 0).sum()))
        return seed(table, slots, rows)

    monkeypatch.setattr(request_slot_table, "seed_slot_rows", counting_seed)
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=3, num_cols=1, device=CPU
    )
    rows = {"a": (1,), "b": (2,), "c": (3,)}
    tbl = pool.sync_step(["a", "b", "c"], rows, rows.__getitem__)
    assert tbl is not None and seeded == [3]
    assert sorted(tbl[1:, 0].tolist()) == [1, 2, 3]


def test_release_frees_slots_and_staged_rows() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=3, num_cols=1, device=CPU
    )
    rows = {rid: (7,) for rid in ("a", "b", "c", "d")}
    cold_calls: list[str] = []
    cold = _cold_rows(rows, cold_calls)
    pool.sync_step(["a", "b", "c"], rows, cold)
    slot_a, slot_c = pool.allocator.slot_for("a"), pool.allocator.slot_for("c")

    # release drops the staged row too: "a" comes back with a cold row.
    pool.mark_dirty("a", (8,))
    pool.release(["a"])
    assert pool.allocator.slot_for("a") is None
    assert pool.needs_cold_reseed(["a"], rows)

    # Requests outside the active set lose their slot and staged rows.
    pool.mark_dirty("d", (9,))
    pool.release_unscheduled({"b"})
    assert set(pool.allocator.allocated_req_ids()) == {"b"}
    assert pool.needs_cold_reseed(["d"], rows)

    # Freed slots are leased again.
    cold_calls.clear()
    pool.sync_step(["b", "d"], rows, cold)
    assert cold_calls == ["d"]
    assert pool.allocator.slot_for("d") in {slot_a, slot_c}


def test_full_pool_raises() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=2, num_cols=1, device=CPU
    )
    rows = {rid: (0,) for rid in ("r0", "r1", "r2")}
    with pytest.raises(RuntimeError, match="RequestSlotAllocator is full"):
        pool.sync_step(["r0", "r1", "r2"], rows, rows.__getitem__)


def test_rows_are_validated() -> None:
    pool: RequestSlotTable[tuple[int, ...]] = RequestSlotTable(
        max_num_reqs=2, num_cols=2, device=CPU
    )
    with pytest.raises(ValueError, match=r"has 1 values, not 2"):
        pool.mark_dirty("a", (1,))
    not_ints: Any = (1, 2.5)
    with pytest.raises(TypeError):
        pool.mark_dirty("a", not_ints)
    with pytest.raises(ValueError, match="does not fit in int32"):
        pool.mark_dirty("a", (1, 2**31))
    with pytest.raises(ValueError, match=r"has 3 values, not 2"):
        pool.sync_step(["a"], {"a"}, lambda rid: (1, 2, 3))


def test_chunk_slot_ids_layout() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=4, num_cols=1, device=CPU
    )
    rows = {"a": (1,), "b": (2,)}
    batch = ["a", "x", "b", None]
    pool.sync_step(batch, rows, rows.__getitem__)
    sa, sb = pool.allocator.slot_for("a"), pool.allocator.slot_for("b")
    build = pool.allocator.build_chunk_slot_ids

    # A [max_num_reqs] int32 host array; untracked and empty rows map to 0.
    ids = build(batch, 0, 4)
    assert isinstance(ids, np.ndarray) and ids.dtype == np.int32
    assert ids.tolist() == [sa, 0, sb, 0]
    # Rows are chunk-relative, and end_idx may pass the end of the batch.
    ids = build(batch, 2, 8)
    assert ids is not None and ids.tolist() == [sb, 0, 0, 0]
    # Discarded requests map to slot 0; a chunk without active rows is None.
    ids = build(batch, 0, 3, discard_req_ids={"a"})
    assert ids is not None and ids.tolist() == [0, 0, sb, 0]
    assert build(batch, 0, 2, discard_req_ids={"a"}) is None
    assert build(["x", None], 0, 2) is None


def test_chunk_slot_ids_reject_invalid_chunks() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=2, num_cols=1, device=CPU
    )
    rows = {"a": (0,), "b": (0,)}
    pool.sync_step(["a", "b"], rows, rows.__getitem__)
    build = pool.allocator.build_chunk_slot_ids
    with pytest.raises(ValueError, match=r"invalid chunk range \[-1, 1\)"):
        build(["a", "b"], -1, 1)
    with pytest.raises(ValueError, match=r"invalid chunk range \[2, 1\)"):
        build(["a", "b"], 2, 1)
    with pytest.raises(ValueError, match=r"has 3 rows, more than max_num_reqs=2"):
        build(["a", "b", "c"], 0, 3)
    with pytest.raises(ValueError, match="lists a request twice"):
        build(["a", "a"], 0, 2)


def test_precompile_seeds_sentinel_rows_and_syncs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced: list[torch.Tensor] = []
    monkeypatch.setattr(request_slot_table, "synchronize_tensors", synced.append)
    pool: RequestSlotTable[tuple[int, int]] = RequestSlotTable(
        max_num_reqs=3, num_cols=2, device=CPU, sentinel_row=(-1, 0)
    )
    table = pool.precompile()
    assert len(synced) == 1 and synced[0] is table
    assert table.device.type == "cpu" and table.dtype == torch.int32
    assert table.tolist() == [[-1, 0]] * 4


def test_commit_table_stores_the_state_the_next_step_starts_from() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=2, num_cols=1, device=CPU
    )
    rows = {"a": (3,), "b": (4,)}
    tbl = pool.sync_step(["a"], rows, rows.__getitem__)
    slot_a = pool.allocator.slot_for("a")
    assert tbl is not None and slot_a is not None
    updated = tbl.clone()
    updated[slot_a] = 30
    pool.commit_table(updated)
    out = pool.sync_step(["a", "b"], rows, rows.__getitem__)
    assert out is not None
    assert _row(pool, out, "a") == [30] and _row(pool, out, "b") == [4]
    for bad in (
        torch.zeros((2, 1), dtype=torch.int32),
        torch.zeros((3, 1), dtype=torch.int64),
        torch.zeros((3, 2), dtype=torch.int32),
    ):
        with pytest.raises(ValueError, match=r"expected a \(3, 1\) int32 table"):
            pool.commit_table(bad)


def test_precompile_covers_the_seed_call_sync_step_makes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``precompile`` calls ``seed_slot_rows`` with the shapes, dtypes and
    devices of a seeding ``sync_step``, so serving compiles nothing new."""
    calls: list[tuple] = []
    seed = request_slot_table.seed_slot_rows

    def spy(table: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor):
        calls.append(
            tuple((t.shape, t.dtype, t.device.type) for t in (table, slots, rows))
        )
        return seed(table, slots, rows)

    monkeypatch.setattr(request_slot_table, "seed_slot_rows", spy)
    monkeypatch.setattr(request_slot_table, "synchronize_tensors", lambda t: None)
    pool: RequestSlotTable[tuple[int, int]] = RequestSlotTable(
        max_num_reqs=4, num_cols=2, device=CPU, sentinel_row=(-1, 0)
    )
    pool.precompile()
    rows = {"a": (1, 2), "b": (3, 4)}
    pool.sync_step(["a", "b"], rows, rows.__getitem__)
    assert len(calls) == 2 and calls[0] == calls[1]


@pytest.mark.tpu_test
def test_precompile_then_sync_step_on_tpu_matches_cpu() -> None:
    """The compiled ``seed_slot_rows`` that ``precompile`` warms gives
    ``sync_step`` the same table on TPU as the eager kernel on CPU."""
    import torch_tpu  # noqa: F401

    rows = {"a": (1, 2), "b": (3, 4), "c": (5, 6)}

    def run(device: torch.device) -> tuple[list[list[int]], list[int | None]]:
        pool: RequestSlotTable[tuple[int, int]] = RequestSlotTable(
            max_num_reqs=4, num_cols=2, device=device, sentinel_row=(-1, 0)
        )
        pool.precompile()
        pool.sync_step(["a", None, "b"], rows, rows.__getitem__)
        pool.mark_dirty("a", (7, 8))
        tbl = pool.sync_step(["a", "c"], rows, rows.__getitem__)
        assert tbl is not None and tbl.device.type == device.type
        slots = [pool.allocator.slot_for(rid) for rid in ("a", "b", "c")]
        return tbl.cpu().tolist(), slots

    with torch.compiler.set_stance("force_eager"):
        expected = run(CPU)
    assert run(torch.device("tpu")) == expected

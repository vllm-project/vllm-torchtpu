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
"""Tests for ``RequestSlotTable``, the request-ID-keyed device-slot pool."""

import pytest
import torch

from vllm_torchtpu.runner.request_slot_table import RequestSlotTable

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _eager_seed_kernel():
    """Run the ``@torch.compile(backend="tpu")`` seed kernel eagerly on CPU."""
    with torch.compiler.set_stance("force_eager"):
        yield


def test_constructor_validates_row_shape() -> None:
    with pytest.raises(ValueError, match="num_cols must be positive"):
        RequestSlotTable(max_num_reqs=2, num_cols=0, device=CPU)
    with pytest.raises(ValueError, match="sentinel_row length 1 != num_cols 2"):
        RequestSlotTable(max_num_reqs=2, num_cols=2, device=CPU, sentinel_row=(1,))


def test_sync_step_leases_slots_and_seeds_dirty_and_cold_rows() -> None:
    pool: RequestSlotTable[tuple[int, int]] = RequestSlotTable(
        max_num_reqs=2, num_cols=2, device=CPU, sentinel_row=(-99, 0)
    )
    assert pool.max_num_reqs == 2 and pool.device == CPU and pool.table is None
    tracked = {"a": (10, 1), "b": (20, 2), "c": (30, 3)}
    pool.mark_dirty("a", tracked["a"])
    assert not pool.needs_cold_reseed(["a"], tracked)
    assert pool.needs_cold_reseed(["b"], tracked)
    assert not pool.needs_cold_reseed(["untracked", None], tracked)

    cold_calls: list[str] = []

    def cold_row(rid: str) -> tuple[int, int]:
        cold_calls.append(rid)
        return tracked[rid]

    # Untracked and missing rows lease nothing and seed nothing.
    tbl = pool.sync_step(["a", None, "untracked", "b"], tracked, cold_row)
    assert tbl is not None and pool.table is tbl and tbl.shape == (3, 2)
    assert cold_calls == ["b"] and not pool.has_dirty("a")
    assert tuple(tbl[0].tolist()) == (-99, 0)
    sa, sb = pool.slot_for("a"), pool.slot_for("b")
    assert sa is not None and sb is not None and sa != sb
    assert tuple(tbl[sa].tolist()) == (10, 1)
    assert tuple(tbl[sb].tolist()) == (20, 2)
    assert pool.slot_for("untracked") is None

    # Leased rows keep their slot and state across steps without reseeding.
    assert pool.sync_step(["b", "a"], tracked, cold_row) is tbl
    assert (pool.slot_for("a"), pool.slot_for("b")) == (sa, sb)

    # A wave swap frees "a" and cold-seeds "c"; "a" cold-seeds again on return.
    tbl = pool.sync_step(["b", "c"], tracked, cold_row)
    assert tbl is not None and "a" not in pool.allocated_req_ids()
    assert cold_calls == ["b", "c"]
    tbl = pool.sync_step(["a", "c"], tracked, cold_row)
    assert tbl is not None and cold_calls == ["b", "c", "a"]
    assert pool.sync_step(["untracked"], tracked, cold_row) is None


def test_release_frees_slots_and_dirty_rows() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=3, num_cols=1, device=CPU
    )
    tracked = {rid: (7,) for rid in ("a", "b", "c")}
    pool.sync_step(["a", "b", "c"], tracked, tracked.__getitem__)
    slot_a, slot_c = pool.slot_for("a"), pool.slot_for("c")

    pool.mark_dirty("a", (8,))
    pool.release(["a"])
    assert pool.slot_for("a") is None and not pool.has_dirty("a")
    pool.release(None)

    # Requests outside the active set lose their slot and staged rows.
    pool.mark_dirty("d", (9,))
    pool.release_unscheduled({"b"})
    assert set(pool.allocated_req_ids()) == {"b"} and not pool.has_dirty("d")

    # Freed slots are leased again.
    pool.sync_step(["b", "d"], {**tracked, "d": (9,)}, lambda rid: (9,))
    assert pool.slot_for("d") in {slot_a, slot_c}
    assert set(pool.allocated_req_ids()) == {"b", "d"}


def test_pool_exhaustion_asserts() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=2, num_cols=1, device=CPU
    )
    tracked = {rid: (0,) for rid in ("r0", "r1", "r2")}
    with pytest.raises(AssertionError, match="RequestSlotTable pool exhausted"):
        pool.sync_step(["r0", "r1", "r2"], tracked, tracked.__getitem__)


def test_chunk_slot_ids_layout() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=4, num_cols=1, device=CPU
    )
    tracked = {"a": (1,), "b": (2,)}
    batch = ["a", "x", "b", None]
    pool.sync_step(batch, tracked, tracked.__getitem__)
    sa, sb = pool.slot_for("a"), pool.slot_for("b")

    # Width defaults to max_num_reqs; untracked and missing rows map to slot 0.
    ids = pool.build_chunk_slot_ids(batch, 0, 4)
    assert ids is not None and ids.dtype == torch.int32
    assert ids.tolist() == [sa, 0, sb, 0]
    # Rows are chunk-relative, padded to padded_num_reqs.
    ids = pool.build_chunk_slot_ids(batch, 2, 4, padded_num_reqs=2)
    assert ids is not None and ids.tolist() == [sb, 0]
    ids = pool.build_chunk_slot_ids(batch, 0, 3, padded_num_reqs=8)
    assert ids is not None and ids.tolist() == [sa, 0, sb, 0, 0, 0, 0, 0]
    # Discarded requests map to slot 0; a chunk without active rows is None.
    ids = pool.build_chunk_slot_ids(batch, 0, 3, discard_req_ids={"a"})
    assert ids is not None and ids.tolist() == [0, 0, sb, 0]
    assert pool.build_chunk_slot_ids(batch, 0, 2, discard_req_ids={"a"}) is None
    assert pool.build_chunk_slot_ids(["x", None], 0, 2) is None


def test_slot_id_cache_reuses_uploads_until_the_layout_changes() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=2, num_cols=1, device=CPU
    )
    tracked = {"r0": (0,), "r1": (0,)}
    pool.sync_step(["r0", "r1"], tracked, tracked.__getitem__)
    s1 = pool.build_chunk_slot_ids(["r0", "r1"], 0, 2, 4)
    s2 = pool.build_chunk_slot_ids(["r0", "r1"], 0, 2, 4)
    s_pad8 = pool.build_chunk_slot_ids(["r0", "r1"], 0, 2, 8)
    s_reordered = pool.build_chunk_slot_ids(["r1", "r0"], 0, 2, 4)
    assert s1 is not None and s2 is s1
    assert s_pad8 is not None and s_pad8 is not s1 and s_pad8.shape == (8,)
    assert s_reordered is not None and s_reordered is not s1
    assert s_reordered.tolist()[:2] == s1.tolist()[:2][::-1]


def test_slot_id_uploads_cached_per_chunk_across_steps() -> None:
    """A step with two chunks keeps one cached slot-ID upload per chunk."""
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=4, num_cols=1, device=CPU
    )
    tracked = {rid: (7,) for rid in ("a", "b", "c", "d")}
    batch = ["a", "b", "c", "d"]
    assert pool.sync_step(batch, tracked, tracked.__getitem__) is not None

    def chunk_uploads(req_ids: list[str]) -> list[torch.Tensor | None]:
        return [
            pool.build_chunk_slot_ids(req_ids, 0, 2, 2),
            pool.build_chunk_slot_ids(req_ids, 2, 4, 2),
        ]

    step1 = chunk_uploads(batch)
    step2 = chunk_uploads(batch)
    assert step1[0] is not None and step1[1] is not None
    assert step2[0] is step1[0] and step2[1] is step1[1]
    # Swapping two rows of the second chunk re-uploads only that chunk.
    step3 = chunk_uploads(["a", "b", "d", "c"])
    assert step3[0] is step1[0] and step3[1] is not step1[1]
    assert step3[1] is not None
    assert step3[1].tolist() == step1[1].tolist()[::-1]


def test_duplicate_slot_in_chunk_asserts() -> None:
    pool: RequestSlotTable[tuple[int]] = RequestSlotTable(
        max_num_reqs=2, num_cols=1, device=CPU
    )
    tracked = {"a": (0,)}
    pool.sync_step(["a"], tracked, tracked.__getitem__)
    with pytest.raises(AssertionError, match="Duplicate slot"):
        pool.build_chunk_slot_ids(["a", "a"], 0, 2)


def test_precompile_seeds_sentinel_rows_on_the_bound_device() -> None:
    pool: RequestSlotTable[tuple[int, int]] = RequestSlotTable(
        max_num_reqs=3, num_cols=2, device=CPU, sentinel_row=(-1, 0)
    )
    synced: list[torch.Tensor] = []
    table = pool.precompile(synced.append)
    assert synced == [table] and pool.table is table
    assert table.device.type == "cpu" and table.dtype == torch.int32
    assert table.tolist() == [[-1, 0]] * 4
    # commit_table replaces the device state without touching the leases.
    updated = table + 1
    pool.commit_table(updated)
    assert pool.table is updated and pool.ensure_table() is updated

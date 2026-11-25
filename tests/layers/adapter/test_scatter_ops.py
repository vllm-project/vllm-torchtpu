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
"""Tests for the slot-table row writes in ``scatter_ops``."""

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample.scatter_ops import (
    scatter_slot_rows,
    seed_slot_rows,
)

INT32_MIN = torch.iinfo(torch.int32).min


@pytest.fixture
def eager_seed_kernel():
    """Run the ``@torch.compile(backend="tpu")`` seed kernel eagerly on CPU."""
    with torch.compiler.set_stance("force_eager"):
        yield


def _indexed_assignment(
    table: torch.Tensor, slot_ids: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    expected = table.clone()
    active = slot_ids > 0
    expected[slot_ids[active].long()] = rows[active]
    return expected


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("trailing", [(), (3,)])
def test_scatter_slot_rows_matches_indexed_assignment(
    dtype: torch.dtype, trailing: tuple[int, ...]
) -> None:
    gen = torch.Generator().manual_seed(0)
    table = torch.randint(-50, 50, (6, *trailing), generator=gen).to(dtype)
    slot_ids = torch.tensor([4, 0, 2, 5, -1, 1], dtype=torch.int32)
    rows = torch.randint(-50, 50, (6, *trailing), generator=gen).to(dtype)
    expected = _indexed_assignment(table, slot_ids, rows)
    assert torch.equal(scatter_slot_rows(table, slot_ids, rows), expected)


def test_inactive_slots_leave_row_0_and_the_last_row_unchanged() -> None:
    """Slot 0 and negative slots are inactive: row 0 keeps its value (it is
    written back to itself) and a negative slot does not wrap to the end."""
    table = torch.full((4, 2), -1, dtype=torch.int32)
    slot_ids = torch.tensor([3, 0, 1, -1], dtype=torch.int32)
    rows = torch.tensor([[30, 31], [0, 1], [10, 11], [7, 7]], dtype=torch.int32)
    expected = [[-1, -1], [10, 11], [-1, -1], [30, 31]]
    assert scatter_slot_rows(table, slot_ids, rows).tolist() == expected
    assert table.tolist() == [[-1, -1]] * 4


@pytest.mark.parametrize(
    ("dtype", "low"),
    [
        (torch.int32, INT32_MIN),
        (torch.int64, torch.iinfo(torch.int64).min),
        (torch.float32, float("-inf")),
        (torch.bfloat16, float("-inf")),
    ],
)
def test_scatter_slot_rows_keeps_extreme_values(dtype: torch.dtype, low: float) -> None:
    table = torch.zeros((4, 2), dtype=dtype)
    slot_ids = torch.tensor([2, 1], dtype=torch.int32)
    rows = torch.tensor([[low, 7], [9, low]], dtype=dtype)
    expected = _indexed_assignment(table, slot_ids, rows)
    assert torch.equal(scatter_slot_rows(table, slot_ids, rows), expected)


def test_scatter_slot_rows_checks_its_arguments() -> None:
    table = torch.zeros((3, 2), dtype=torch.int32)
    rows = torch.zeros((2, 2), dtype=torch.int32)
    with pytest.raises(AssertionError, match="slot_ids must be 1-D"):
        scatter_slot_rows(table, torch.ones((2, 1), dtype=torch.int32), rows)
    with pytest.raises(AssertionError, match="rows must have shape"):
        scatter_slot_rows(table, torch.tensor([1, 2], dtype=torch.int32), rows[:, :1])
    with pytest.raises(AssertionError, match="rows must have the table's dtype"):
        scatter_slot_rows(table, torch.tensor([1, 2], dtype=torch.int32), rows.float())
    with pytest.raises(ValueError, match="reach past the 3 rows"):
        scatter_slot_rows(table, torch.tensor([1, 3], dtype=torch.int32), rows)
    with pytest.raises(ValueError, match="repeat an active slot"):
        scatter_slot_rows(table, torch.tensor([1, 1], dtype=torch.int32), rows)
    # Several inactive entries may target slot 0.
    out = scatter_slot_rows(table, torch.tensor([0, 0], dtype=torch.int32), rows + 1)
    assert out.tolist() == table.tolist()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_scatter_slot_rows_compiles_without_graph_breaks(dtype: torch.dtype) -> None:
    """Traced with ``fullgraph=True``: the argument checks and the CPU-only
    slot checks stay out of the graph, which matches indexed assignment."""
    table = torch.full((5, 3), -7, dtype=dtype)
    slot_ids = torch.tensor([3, 0, 1, 0], dtype=torch.int32)
    rows = torch.tensor([[5, 1, 2], [9, 9, 9], [4, 3, 6], [8, 8, 8]], dtype=dtype)
    torch._dynamo.reset()
    compiled = torch.compile(
        scatter_slot_rows, backend="eager", fullgraph=True, dynamic=False
    )
    expected = _indexed_assignment(table, slot_ids, rows)
    assert torch.equal(compiled(table, slot_ids, rows), expected)


@pytest.mark.usefixtures("eager_seed_kernel")
def test_seed_slot_rows_writes_the_active_rows() -> None:
    table = torch.full((4, 2), -1, dtype=torch.int32)
    slot_ids = torch.tensor([3, 0, 1], dtype=torch.int32)
    rows = torch.tensor([[30, 31], [0, 1], [10, 11]], dtype=torch.int32)
    assert seed_slot_rows(table, slot_ids, rows).tolist() == [
        [-1, -1],
        [10, 11],
        [-1, -1],
        [30, 31],
    ]


@pytest.mark.tpu_test
def test_compiled_seed_slot_rows_on_tpu_matches_cpu() -> None:
    """The ``backend="tpu"`` graph of ``seed_slot_rows`` matches the eager CPU
    result and indexed assignment."""
    import torch_tpu  # noqa: F401

    gen = torch.Generator().manual_seed(0)
    table = torch.randint(-50, 50, (9, 7), generator=gen, dtype=torch.int32)
    slots = torch.tensor([3, 0, 8, 1, 0, 5, 0, 2], dtype=torch.int32)
    rows = torch.randint(-50, 50, (8, 7), generator=gen, dtype=torch.int32)
    rows[2, 4] = INT32_MIN
    with torch.compiler.set_stance("force_eager"):
        expected = seed_slot_rows(table, slots, rows)
    assert torch.equal(expected, _indexed_assignment(table, slots, rows))
    tpu = torch.device("tpu")
    got = seed_slot_rows(table.to(tpu), slots.to(tpu), rows.to(tpu))
    assert torch.equal(got.cpu(), expected)

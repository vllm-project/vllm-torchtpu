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
"""Tests for the unique-index scatter helpers in ``scatter_ops``."""

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample.scatter_ops import (
    broadcast_amax_scatter,
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
    base: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    expected = base.clone()
    expected[indices[valid].long()] = values[valid]
    return expected


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("trailing", [(), (3,)])
def test_broadcast_amax_scatter_matches_indexed_assignment(
    dtype: torch.dtype, trailing: tuple[int, ...]
) -> None:
    gen = torch.Generator().manual_seed(0)
    base = torch.randint(-50, 50, (6, *trailing), generator=gen).to(dtype)
    indices = torch.tensor([4, 0, 2, 5, 1], dtype=torch.int32)
    values = torch.randint(-50, 50, (5, *trailing), generator=gen).to(dtype)
    valid = torch.tensor([True, True, False, True, False])

    expected = _indexed_assignment(base, indices, values, valid)
    assert torch.equal(broadcast_amax_scatter(base, indices, values, valid), expected)


def test_broadcast_amax_scatter_keeps_int32_min_values() -> None:
    """An active INT32_MIN value survives the int32 ``amax``."""
    base = torch.zeros((4, 2), dtype=torch.int32)
    vals = torch.tensor([[INT32_MIN, 7], [9, INT32_MIN]], dtype=torch.int32)
    out = broadcast_amax_scatter(
        base, torch.tensor([2, 0], dtype=torch.int32), vals, torch.tensor([True, False])
    )
    assert out.tolist() == [[0, 0], [0, 0], [INT32_MIN, 7], [0, 0]]


@pytest.mark.parametrize(
    ("dtype", "low"),
    [
        (torch.float32, float("-inf")),
        (torch.bfloat16, float("-inf")),
        (torch.int64, INT32_MIN - 1),
        (torch.int64, torch.iinfo(torch.int64).min),
    ],
)
def test_broadcast_amax_scatter_keeps_values_below_int32_min(
    dtype: torch.dtype, low: float
) -> None:
    """Non-matching candidates take the lowest value of ``base.dtype``, so -inf
    and int64 values below INT32_MIN survive the ``amax``."""
    base = torch.zeros((4, 2), dtype=dtype)
    indices = torch.tensor([2, 0], dtype=torch.int32)
    vals = torch.tensor([[low, 7], [9, low]], dtype=dtype)
    valid = torch.tensor([True, False])
    out = broadcast_amax_scatter(base, indices, vals, valid)
    assert torch.equal(out, _indexed_assignment(base, indices, vals, valid))


def test_broadcast_amax_scatter_rejects_non_numeric_dtypes() -> None:
    with pytest.raises(TypeError, match="floating or integer dtype, got torch.bool"):
        broadcast_amax_scatter(
            torch.zeros(3, dtype=torch.bool),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([True]),
            torch.tensor([True]),
        )


def test_broadcast_amax_scatter_without_indices_returns_base() -> None:
    base = torch.arange(4, dtype=torch.int32)
    empty = torch.zeros(0, dtype=torch.int32)
    assert (
        broadcast_amax_scatter(base, empty, empty, torch.zeros(0, dtype=torch.bool))
        is base
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_scatter_slot_rows_compiles_without_graph_breaks(dtype: torch.dtype) -> None:
    """Traced with ``fullgraph=True``: the dtype fill and the CPU-only
    duplicate check stay out of the graph, which matches indexed assignment."""
    low = float("-inf") if dtype.is_floating_point else torch.iinfo(dtype).min
    table = torch.full((5, 3), -7, dtype=dtype)
    slots = torch.tensor([3, 0, 1, 0], dtype=torch.int32)
    rows = torch.tensor([[low, 1, 2], [9, 9, 9], [4, low, 6], [8, 8, 8]], dtype=dtype)
    torch._dynamo.reset()
    compiled = torch.compile(
        scatter_slot_rows, backend="eager", fullgraph=True, dynamic=False
    )
    expected = _indexed_assignment(table, slots, rows, slots > 0)
    assert torch.equal(compiled(table, slots, rows), expected)


@pytest.mark.usefixtures("eager_seed_kernel")
def test_slot_scatters_skip_the_sentinel_slot() -> None:
    table = torch.full((4, 2), -1, dtype=torch.int32)
    slots = torch.tensor([3, 0, 1], dtype=torch.int32)
    rows = torch.tensor([[30, 31], [0, 1], [10, 11]], dtype=torch.int32)
    expected = [[-1, -1], [10, 11], [-1, -1], [30, 31]]
    assert scatter_slot_rows(table, slots, rows).tolist() == expected
    assert seed_slot_rows(table, slots, rows).tolist() == expected
    assert table.tolist() == [[-1, -1]] * 4


@pytest.mark.usefixtures("eager_seed_kernel")
def test_duplicate_active_indices_assert_on_cpu() -> None:
    table = torch.zeros((3, 2), dtype=torch.int32)
    with pytest.raises(AssertionError, match="Duplicate active indices"):
        seed_slot_rows(
            table,
            torch.tensor([1, 1], dtype=torch.int32),
            torch.zeros((2, 2), dtype=torch.int32),
        )
    # Duplicates at the inactive sentinel slot 0 are allowed.
    out = seed_slot_rows(
        table,
        torch.tensor([0, 0, 2], dtype=torch.int32),
        torch.ones((3, 2), dtype=torch.int32),
    )
    assert out.tolist() == [[0, 0], [0, 0], [1, 1]]

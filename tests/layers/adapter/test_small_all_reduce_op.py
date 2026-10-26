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
"""Which TP all-reduces the small all-reduce bridge hands to its op.

The kernel itself is tested on TPU in tests/kernels/experimental/small_collectives;
the kernel/XLA row choice is made inside the op (``_build_op``).
"""

from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.layers.adapter import small_all_reduce_op as op

pytestmark = pytest.mark.cpu_test

CORES = 8
TP = tuple(range(8, 16))
HIDDEN = 7168


@pytest.fixture
def armed(monkeypatch):
    """The op replaced by a mock, inside a compiled trace."""
    ar = Mock(name="small_tp_all_reduce")
    monkeypatch.setattr(op, "_OP", ar)
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op, "_PEERS", torch.arange(8, 16, dtype=torch.int32))
    monkeypatch.setattr(op, "_CORES", CORES)
    monkeypatch.setattr(op, "_TP_RANKS", TP)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    return ar


def _block(rows, hidden=HIDDEN, dtype=torch.bfloat16):
    return torch.randn(rows, hidden).to(dtype)


def _fallback(x):
    return x * 2


@pytest.mark.parametrize("rows", [1, 8, 16, 64, 2048])
def test_tp_block_takes_the_op_at_any_rows(armed, rows):
    x = _block(rows)
    assert op.all_reduce_or(x, TP, _fallback) is armed.return_value
    (block, rank, peers), _ = armed.call_args
    assert block is x and rank is op._RANK and peers is op._PEERS


@pytest.mark.parametrize(
    "x,ranks",
    [
        (_block(8), tuple(range(16))),  # another group (EP / DP)
        (_block(8, dtype=torch.float32), TP),
        (_block(8, hidden=HIDDEN + 128), TP),  # does not split over 8 x 128
        (torch.randn(2, 8, HIDDEN).to(torch.bfloat16), TP),
    ],
    ids=["other_group", "f32", "width", "3d"],
)
def test_other_blocks_fall_back(armed, x, ranks):
    out = op.all_reduce_or(x, ranks, _fallback)
    torch.testing.assert_close(out, _fallback(x), rtol=0, atol=0)
    armed.assert_not_called()


def test_eager_falls_back(armed, monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    x = _block(8)
    torch.testing.assert_close(op.all_reduce_or(x, TP, _fallback), _fallback(x))
    armed.assert_not_called()


def test_unarmed_falls_back(monkeypatch):
    monkeypatch.setattr(op, "_OP", None)
    x = _block(8)
    torch.testing.assert_close(op.all_reduce_or(x, TP, _fallback), _fallback(x))


def test_max_rows_follows_the_env(monkeypatch):
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 0)
    assert op.max_rows() == op.MAX_ROWS
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 32)
    assert op.max_rows() == 32

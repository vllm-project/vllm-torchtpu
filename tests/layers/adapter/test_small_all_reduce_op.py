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
"""Which calls the small all-reduce bridge hands to the kernel.

The kernel itself is tested on TPU in
tests/kernels/experimental/small_collectives; the warmup hand-over between the
two sides of the row fork in tests/compilation/test_shape_variants.py.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.layers.adapter import small_all_reduce_op as op

CORES = 8
MESH = tuple(range(CORES))
HIDDEN = 4096


@pytest.fixture
def armed(monkeypatch):
    """The op replaced by a mock, inside a warmup trace."""
    ar = Mock(name="all_reduce_op")
    monkeypatch.setattr(op, "_AR_OP", ar)
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op, "_CORES", CORES)
    monkeypatch.setattr(op, "_MESH_RANKS", MESH)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: True)
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 0)
    return ar


def _block(rows, hidden=HIDDEN, dtype=torch.bfloat16):
    return torch.randn(rows, hidden).to(dtype)


def _fallback(x):
    return x * 2


@pytest.mark.parametrize(
    "max_rows_env,rows,armed_expected",
    [
        (0, 1, True),
        (0, op.MAX_ROWS, True),
        (0, op.MAX_ROWS + 1, False),
        (32, 32, True),
        (32, 33, False),
        (32, 4096, False),
    ],
)
def test_row_cap_picks_kernel_or_collective(
    armed, monkeypatch, max_rows_env, rows, armed_expected
):
    """TPU_SMALL_ALL_REDUCE_MAX_ROWS moves the cap, 0 keeps MAX_ROWS; a block
    above it keeps vLLM's collective."""
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", max_rows_env)
    x = _block(rows)
    out = op.all_reduce_or(x, MESH, _fallback)
    if armed_expected:
        assert out is armed.return_value
        (block, rank), _ = armed.call_args
        assert block is x and rank is op._RANK
    else:
        armed.assert_not_called()
        torch.testing.assert_close(out, _fallback(x), rtol=0, atol=0)


def test_large_block_fallback_depends_on_the_rank_buffer(armed, monkeypatch):
    """The retain is a live read: a NaN rank buffer poisons the result, which a
    read the compiler could drop would not."""
    monkeypatch.setattr(op, "_RANK", torch.tensor([float("nan")]))
    out = op.all_reduce_or(_block(op.MAX_ROWS + 1), MESH, _fallback)
    assert torch.isnan(out).all()


@pytest.mark.parametrize(
    "x,ranks",
    [
        (_block(8, dtype=torch.float32), MESH),  # dtype
        (_block(8, hidden=HIDDEN + 128), MESH),  # width does not split over cores
        (_block(8).reshape(2, 4, HIDDEN), MESH),  # not 2-D
        (_block(8), (0, 1, 2, 3)),  # another group than the mesh
    ],
    ids=["fp32", "width", "3d", "group"],
)
def test_ineligible_all_reduce_falls_back_untouched(armed, x, ranks):
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, ranks, fallback) is fallback.return_value
    fallback.assert_called_once_with(x)
    armed.assert_not_called()


@pytest.mark.parametrize("outside", ["not_compiling", "no_warmup"])
def test_traces_outside_the_backbone_warmup_keep_the_collective(
    armed, monkeypatch, outside
):
    """The MTP drafter is compiled outside the warmup, which could not fork it."""
    if outside == "not_compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    else:
        monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(_block(8), MESH, fallback) is fallback.return_value
    armed.assert_not_called()


def test_unarmed_is_the_fallback(monkeypatch):
    monkeypatch.setattr(op, "_AR_OP", None)
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(_block(8), MESH, fallback) is fallback.return_value


def test_prebuild_is_a_no_op_with_the_flag_off(monkeypatch):
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", False)
    build = Mock()
    monkeypatch.setattr(op, "build_ep_mesh", build)
    monkeypatch.setattr(op, "_AR_OP", None)
    op.prebuild()
    build.assert_not_called()
    assert op._AR_OP is None


def test_prebuild_refuses_a_deployment_without_an_ep_group(monkeypatch):
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", True)
    monkeypatch.setattr(op, "build_ep_mesh", lambda: None)
    monkeypatch.setattr(op, "_AR_OP", None)
    with pytest.raises(NotImplementedError, match="TPU_SMALL_ALL_REDUCE"):
        op.prebuild()


def test_retain_is_the_identity_when_unarmed_or_outside_the_warmup(monkeypatch):
    x = _block(8)
    monkeypatch.setattr(op, "_RANK", None)
    assert op.retain(x) is x
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    assert op.retain(x) is x

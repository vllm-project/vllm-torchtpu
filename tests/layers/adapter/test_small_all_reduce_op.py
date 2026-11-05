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
"""Which calls the small-collective bridge hands to a kernel.

The kernels themselves are tested on TPU in
tests/kernels/experimental/small_collectives; the warmup hand-over between the
two sides of the row fork in tests/compilation/test_shape_variants.py.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.layers.adapter import small_all_reduce_op as op

pytestmark = pytest.mark.cpu_test

CORES = 8
MESH = tuple(range(CORES))
HIDDEN = 4096


@pytest.fixture
def armed(monkeypatch):
    """Ops replaced by mocks, inside a warmup trace."""
    ar, rs, ag = (
        Mock(name="all_reduce_op"),
        Mock(name="reduce_scatter_op"),
        Mock(name="all_gather_pair_op"),
    )
    monkeypatch.setattr(op, "_AR_OP", ar)
    monkeypatch.setattr(op, "_RS_OP", rs)
    monkeypatch.setattr(op, "_AG_OP", ag)
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op, "_CORES", CORES)
    monkeypatch.setattr(op, "_MESH_RANKS", MESH)
    monkeypatch.setattr(op, "_DP_COMBINE", True)
    monkeypatch.setattr(op, "MAX_ROWS", 128)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: True)
    return ar, rs, ag


def _block(rows, hidden=HIDDEN, dtype=torch.bfloat16):
    return torch.randn(rows, hidden).to(dtype)


def _fallback(x):
    return x * 2


def _fallback_pair(x, logits):
    return x * 2, logits * 2


@pytest.mark.parametrize("rows", [1, 8, 64, 96, 128])
def test_small_block_takes_the_all_reduce_kernel(armed, rows):
    ar, _, _ = armed
    x = _block(rows)
    assert op.all_reduce_or(x, MESH, _fallback) is ar.return_value
    (block, rank), _ = ar.call_args
    assert block is x and rank is op._RANK


@pytest.mark.parametrize("rows", [1, 16, 40])
def test_small_combine_takes_the_reduce_scatter_kernel(armed, rows):
    _, rs, _ = armed
    x = _block(rows * CORES)
    assert op.dp_combine_or(x, _fallback) is rs.return_value
    (block, rank), _ = rs.call_args
    assert block is x and rank is op._RANK


@pytest.mark.parametrize("rows", [1, 8, 64])
@pytest.mark.parametrize("logits_dtype", [torch.bfloat16, torch.float32])
def test_small_dispatch_takes_the_all_gather_pair_kernel(armed, rows, logits_dtype):
    _, _, ag = armed
    x = _block(rows)
    logits = _block(rows, hidden=512, dtype=logits_dtype)
    assert op.dp_dispatch_or(x, logits, _fallback_pair) is ag.return_value
    (got_x, got_logits, rank), _ = ag.call_args
    assert got_x is x and got_logits is logits and rank is op._RANK


@pytest.mark.parametrize(
    "call,rows",
    [
        (lambda x: op.all_reduce_or(x, MESH, _fallback), op.MAX_ROWS + 1),
        (lambda x: op.dp_combine_or(x, _fallback), CORES * op.COMBINE_MAX_ROWS + CORES),
    ],
    ids=["all_reduce", "dp_combine"],
)
def test_large_block_falls_back_and_reads_the_rank(armed, call, rows):
    ar, rs, _ = armed
    x = _block(rows)
    out = call(x)
    torch.testing.assert_close(out, _fallback(x), rtol=0, atol=0)
    ar.assert_not_called()
    rs.assert_not_called()


@pytest.mark.parametrize("rows_per_engine,armed_expected", [(40, True), (41, False)])
def test_combine_cap_is_40_rows_per_engine(armed, rows_per_engine, armed_expected):
    _, rs, _ = armed
    op.dp_combine_or(_block(rows_per_engine * CORES), _fallback)
    assert rs.called is armed_expected


@pytest.mark.parametrize("rows_per_engine,armed_expected", [(64, True), (65, False)])
def test_dispatch_cap_is_64_rows_per_engine(armed, rows_per_engine, armed_expected):
    _, _, ag = armed
    op.dp_dispatch_or(
        _block(rows_per_engine),
        _block(rows_per_engine, hidden=512, dtype=torch.float32),
        _fallback_pair,
    )
    assert ag.called is armed_expected


@pytest.mark.parametrize("rows,armed_expected", [(128, True), (129, False)])
def test_all_reduce_cap_is_128_rows(armed, rows, armed_expected):
    ar, _, _ = armed
    op.all_reduce_or(_block(rows), MESH, _fallback)
    assert ar.called is armed_expected


def test_large_block_fallback_depends_on_the_rank_buffer(armed, monkeypatch):
    """The retain is a live read: a NaN rank buffer poisons the result, which a
    read the compiler could drop would not."""
    monkeypatch.setattr(op, "_RANK", torch.tensor([float("nan")]))
    out = op.all_reduce_or(_block(op.MAX_ROWS + 1), MESH, _fallback)
    assert torch.isnan(out).all()
    out_x, out_l = op.dp_dispatch_or(
        _block(op.DISPATCH_MAX_ROWS + 1),
        _block(op.DISPATCH_MAX_ROWS + 1, hidden=512, dtype=torch.float32),
        _fallback_pair,
    )
    assert torch.isnan(out_x).all()
    assert not torch.isnan(out_l).any()


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
    ar, _, _ = armed
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, ranks, fallback) is fallback.return_value
    fallback.assert_called_once_with(x)
    ar.assert_not_called()


@pytest.mark.parametrize(
    "x,logits",
    [
        (_block(8), _block(9, hidden=512, dtype=torch.float32)),  # row mismatch
        (_block(8, dtype=torch.float32), _block(8, hidden=512)),  # x dtype
        (_block(8), _block(8, hidden=512, dtype=torch.float16)),  # logits dtype
        (_block(8, hidden=4000), _block(8, hidden=512)),  # x width off lane tile
        (_block(8), _block(8, hidden=500)),  # logits width off lane tile
        (_block(8).reshape(2, 4, HIDDEN), _block(8, hidden=512)),  # not 2-D
    ],
    ids=["rows", "x_fp32", "logits_fp16", "x_width", "logits_width", "3d"],
)
def test_ineligible_dispatch_falls_back_untouched(armed, x, logits):
    _, _, ag = armed
    fallback = Mock(return_value=(object(), object()))
    assert op.dp_dispatch_or(x, logits, fallback) is fallback.return_value
    fallback.assert_called_once_with(x, logits)
    ag.assert_not_called()


@pytest.mark.parametrize("outside", ["not_compiling", "no_warmup"])
def test_traces_outside_the_backbone_warmup_keep_the_collective(
    armed, monkeypatch, outside
):
    """The MTP drafter is compiled outside the warmup, which could not fork it."""
    ar, rs, ag = armed
    if outside == "not_compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    else:
        monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    x = _block(8)
    logits = _block(8, hidden=512)
    fallback = Mock(return_value=object())
    fallback_pair = Mock(return_value=(object(), object()))
    assert op.all_reduce_or(x, MESH, fallback) is fallback.return_value
    assert op.dp_combine_or(x, fallback) is fallback.return_value
    assert op.dp_dispatch_or(x, logits, fallback_pair) is fallback_pair.return_value
    ar.assert_not_called()
    rs.assert_not_called()
    ag.assert_not_called()


def test_combine_and_dispatch_need_the_dp_group_to_be_the_mesh(armed, monkeypatch):
    _, rs, ag = armed
    monkeypatch.setattr(op, "_DP_COMBINE", False)
    x = _block(8 * CORES)
    logits = _block(8, hidden=512)
    fallback = Mock(return_value=object())
    fallback_pair = Mock(return_value=(object(), object()))
    assert op.dp_combine_or(x, fallback) is fallback.return_value
    assert op.dp_dispatch_or(_block(8), logits, fallback_pair) is (
        fallback_pair.return_value
    )
    rs.assert_not_called()
    ag.assert_not_called()


def test_unarmed_is_the_fallback(monkeypatch):
    monkeypatch.setattr(op, "_AR_OP", None)
    monkeypatch.setattr(op, "_RS_OP", None)
    monkeypatch.setattr(op, "_AG_OP", None)
    monkeypatch.setattr(op, "_RANK", None)
    monkeypatch.setattr(op, "_CORES", 0)
    monkeypatch.setattr(op, "_MESH_RANKS", ())
    monkeypatch.setattr(op, "_DP_COMBINE", False)
    monkeypatch.setattr(op, "MAX_ROWS", 128)
    x = _block(8)
    logits = _block(8, hidden=512)
    fallback = Mock(return_value=object())
    fallback_pair = Mock(return_value=(object(), object()))
    assert op.all_reduce_or(x, MESH, fallback) is fallback.return_value
    assert op.dp_combine_or(x, fallback) is fallback.return_value
    assert op.dp_dispatch_or(x, logits, fallback_pair) is fallback_pair.return_value


def test_traced_helpers_do_not_log_or_write_globals():
    """dp_combine_or and dp_dispatch_or run inside the compiled forward."""
    import ast
    import inspect
    import textwrap

    for traced_fn in (op.dp_combine_or, op.dp_dispatch_or, op._fits_dispatch):
        tree = ast.parse(textwrap.dedent(inspect.getsource(traced_fn)))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global)]
        assert not [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "logger"
        ]


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


def test_dispatch_cap_respects_lower_max_rows(armed, monkeypatch):
    _, _, ag = armed
    monkeypatch.setattr(op, "MAX_ROWS", 16)
    op.dp_dispatch_or(_block(16), _block(16, hidden=512), _fallback_pair)
    assert ag.called is True
    ag.reset_mock()
    op.dp_dispatch_or(_block(17), _block(17, hidden=512), _fallback_pair)
    assert ag.called is False


def test_prebuild_arms_all_ops_with_ep_rank_order(monkeypatch):
    from types import SimpleNamespace

    import vllm.distributed.parallel_state as ps

    mesh = SimpleNamespace(shape={op.EP_AXIS_NAME: CORES})
    order = (0, 1, 6, 7, 2, 3, 4, 5)
    ar_op, rs_op, ag_op = Mock(), Mock(), Mock()
    build_ar = Mock(return_value=ar_op)
    build_rs = Mock(return_value=rs_op)
    build_ag = Mock(return_value=ag_op)
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", True)
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_TP", False)
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 64)
    monkeypatch.setattr(op, "_AR_OP", None)
    monkeypatch.setattr(op, "_RS_OP", None)
    monkeypatch.setattr(op, "_AG_OP", None)
    monkeypatch.setattr(op, "_RANK", None)
    monkeypatch.setattr(op, "_CORES", 0)
    monkeypatch.setattr(op, "_MESH_RANKS", ())
    monkeypatch.setattr(op, "_DP_COMBINE", False)
    monkeypatch.setattr(op, "MAX_ROWS", 128)
    monkeypatch.setattr(op, "build_ep_mesh", lambda: mesh)
    monkeypatch.setattr(op, "ep_mesh_index", lambda: 4)
    monkeypatch.setattr(op, "ep_rank_order", lambda: order)
    monkeypatch.setattr(
        op, "get_ep_group", lambda: SimpleNamespace(ranks=list(MESH))
    )
    monkeypatch.setattr(
        ps, "get_dp_group", lambda: SimpleNamespace(ranks=list(MESH), world_size=CORES)
    )
    monkeypatch.setattr(
        ps, "get_tp_group", lambda: SimpleNamespace(ranks=[2], world_size=1)
    )
    monkeypatch.setattr(op, "_build_all_reduce", build_ar)
    monkeypatch.setattr(op, "_build_reduce_scatter", build_rs)
    monkeypatch.setattr(op, "_build_all_gather_pair", build_ag)
    orig_tensor = torch.tensor
    monkeypatch.setattr(
        torch, "tensor", lambda data, **kw: orig_tensor(data, dtype=kw["dtype"])
    )
    op.prebuild()
    build_ar.assert_called_once_with(mesh)
    build_rs.assert_called_once_with(mesh, order)
    build_ag.assert_called_once_with(mesh, order)
    assert op._AR_OP is ar_op and op._RS_OP is rs_op and op._AG_OP is ag_op
    assert op._DP_COMBINE is True


def test_retain_is_the_identity_when_unarmed_or_outside_the_warmup(monkeypatch):
    x = _block(8)
    monkeypatch.setattr(op, "_RANK", None)
    assert op.retain(x) is x
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    assert op.retain(x) is x

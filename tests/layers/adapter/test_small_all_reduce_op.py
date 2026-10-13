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
    """Both ops replaced by mocks, inside a warmup trace."""
    ar, rs = Mock(name="all_reduce_op"), Mock(name="reduce_scatter_op")
    monkeypatch.setattr(op, "_AR_OP", ar)
    monkeypatch.setattr(op, "_RS_OP", rs)
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op, "_CORES", CORES)
    monkeypatch.setattr(op, "_MESH_RANKS", MESH)
    monkeypatch.setattr(op, "_DP_COMBINE", True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: True)
    # The kernel's own MAX_ROWS cap; the env default (32) is tested on its own.
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 0)
    return ar, rs


def _block(rows, hidden=HIDDEN, dtype=torch.bfloat16):
    return torch.randn(rows, hidden).to(dtype)


def _fallback(x):
    return x * 2


@pytest.mark.parametrize("rows", [1, 8, 64, 96])
def test_small_block_takes_the_all_reduce_kernel(armed, rows):
    ar, _ = armed
    x = _block(rows)
    assert op.all_reduce_or(x, MESH, _fallback) is ar.return_value
    (block, rank), _ = ar.call_args
    assert block is x and rank is op._RANK


@pytest.mark.parametrize("rows", [1, 16, 24])
def test_small_combine_takes_the_reduce_scatter_kernel(armed, rows):
    _, rs = armed
    x = _block(rows * CORES)
    assert op.dp_combine_or(x, _fallback) is rs.return_value
    (block, rank), _ = rs.call_args
    assert block is x and rank is op._RANK


@pytest.mark.parametrize(
    "call,rows",
    [
        (lambda x: op.all_reduce_or(x, MESH, _fallback), op.MAX_ROWS + 1),
        (lambda x: op.dp_combine_or(x, _fallback), CORES * op.COMBINE_MAX_ROWS + CORES),
    ],
    ids=["all_reduce", "dp_combine"],
)
def test_large_block_falls_back_and_reads_the_rank(armed, call, rows):
    ar, rs = armed
    x = _block(rows)
    out = call(x)
    torch.testing.assert_close(out, _fallback(x), rtol=0, atol=0)
    ar.assert_not_called()
    rs.assert_not_called()


@pytest.mark.parametrize("rows_per_engine,armed_expected", [(24, True), (25, False)])
def test_combine_cap_is_24_rows_per_engine(armed, rows_per_engine, armed_expected):
    _, rs = armed
    op.dp_combine_or(_block(rows_per_engine * CORES), _fallback)
    assert rs.called is armed_expected


@pytest.mark.parametrize("rows,armed_expected", [(96, True), (97, False)])
def test_all_reduce_cap_is_96_rows(armed, rows, armed_expected):
    ar, _ = armed
    op.all_reduce_or(_block(rows), MESH, _fallback)
    assert ar.called is armed_expected


@pytest.mark.parametrize(
    "rows,armed_expected", [(16, True), (32, True), (33, False), (4096, False)]
)
def test_all_reduce_cap_follows_the_env(armed, monkeypatch, rows, armed_expected):
    """TPU_SMALL_ALL_REDUCE_MAX_ROWS moves the cap; a mixed-step block above
    it keeps vLLM's collective (with the rank read)."""
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 32)
    ar, _ = armed
    x = _block(rows, hidden=2 * CORES * 128)
    out = op.all_reduce_or(x, MESH, _fallback)
    assert ar.called is armed_expected
    if not armed_expected:
        torch.testing.assert_close(out, _fallback(x), rtol=0, atol=0)


def test_all_reduce_cap_env_defaults_to_32_rows(monkeypatch):
    getter = op.envs.environment_variables["TPU_SMALL_ALL_REDUCE_MAX_ROWS"]
    monkeypatch.delenv("TPU_SMALL_ALL_REDUCE_MAX_ROWS", raising=False)
    assert getter() == 32


def test_all_reduce_cap_env_zero_keeps_the_default(armed, monkeypatch):
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 0)
    assert op.max_rows() == op.MAX_ROWS


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
    ar, _ = armed
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, ranks, fallback) is fallback.return_value
    fallback.assert_called_once_with(x)
    ar.assert_not_called()


@pytest.mark.parametrize("outside", ["not_compiling", "no_warmup"])
def test_traces_outside_the_backbone_warmup_keep_the_collective(
    armed, monkeypatch, outside
):
    """The MTP drafter is compiled outside the warmup, which could not fork it."""
    ar, rs = armed
    if outside == "not_compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    else:
        monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    x = _block(8)
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, MESH, fallback) is fallback.return_value
    assert op.dp_combine_or(x, fallback) is fallback.return_value
    ar.assert_not_called()
    rs.assert_not_called()


@pytest.fixture
def sized(armed, monkeypatch):
    """TPU_SMALL_ALL_REDUCE_DRAFT: the per-bucket op is built as well."""
    sized_op = Mock(name="all_reduce_sized_op")
    monkeypatch.setattr(op, "_AR_SIZED_OP", sized_op)
    return sized_op


@pytest.mark.parametrize("rows", [8, 16, 32, 64, 4096])
def test_draft_traces_take_the_sized_op_at_every_row_count(
    armed, sized, monkeypatch, rows
):
    """Outside the warmup the op itself picks kernel or psum per exported
    bucket, so the torch side reads no token count and leaves no guard."""
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    ar, _ = armed
    fallback = Mock()
    x = _block(rows)
    assert op.all_reduce_or(x, MESH, fallback) is sized.return_value
    (block, rank), _ = sized.call_args
    assert block is x and rank is op._RANK
    ar.assert_not_called()
    fallback.assert_not_called()


def test_draft_flag_leaves_the_backbone_warmup_path_alone(armed, sized):
    ar, _ = armed
    op.all_reduce_or(_block(16), MESH, _fallback)
    assert ar.called
    sized.assert_not_called()
    out = op.all_reduce_or(x := _block(op.MAX_ROWS + 1), MESH, _fallback)
    torch.testing.assert_close(out, _fallback(x), rtol=0, atol=0)
    sized.assert_not_called()


@pytest.mark.parametrize(
    "x,ranks",
    [
        (_block(8, dtype=torch.float32), MESH),
        (_block(8, hidden=HIDDEN + 128), MESH),
        (_block(8), (0, 1, 2, 3)),
    ],
    ids=["fp32", "width", "group"],
)
def test_draft_ineligible_blocks_keep_the_collective(
    armed, sized, monkeypatch, x, ranks
):
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, ranks, fallback) is fallback.return_value
    sized.assert_not_called()


def test_draft_flag_leaves_eager_calls_and_the_combine_alone(armed, sized, monkeypatch):
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    _, rs = armed
    fallback = Mock(return_value=object())
    assert op.dp_combine_or(_block(8 * CORES), fallback) is fallback.return_value
    rs.assert_not_called()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    assert op.all_reduce_or(_block(8), MESH, fallback) is fallback.return_value
    sized.assert_not_called()


@pytest.mark.parametrize("warmup,traces", [(False, 1), (True, 2)])
def test_draft_route_leaves_no_token_count_guard(monkeypatch, warmup, traces):
    """A dynamic trace outside the warmup serves every row count from one
    graph, as vLLM's compile of the drafter requires; the warmup route's
    row branch guards the token count, so the same calls retrace there (the
    control)."""
    monkeypatch.setattr(op, "_AR_OP", lambda x, rank: x * 3)
    monkeypatch.setattr(op, "_AR_SIZED_OP", lambda x, rank: x * 5)
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op, "_CORES", 1)
    monkeypatch.setattr(op, "_MESH_RANKS", MESH)
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: warmup)
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm

    torch._dynamo.reset()
    fn = torch.compile(
        lambda x: op.all_reduce_or(x, MESH, _fallback), backend=backend, dynamic=True
    )
    for rows in (16, 4096):
        x = _block(rows, hidden=128)
        torch._dynamo.mark_dynamic(x, 0)
        fn(x)
    torch._dynamo.reset()
    assert len(graphs) == traces


@pytest.mark.parametrize("draft", [False, True])
def test_prebuild_builds_the_sized_op_only_with_the_draft_flag(monkeypatch, draft):
    import types

    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", True)
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_DRAFT", draft)
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 32)
    for name in ("_AR_OP", "_AR_SIZED_OP", "_RS_OP", "_RANK"):
        monkeypatch.setattr(op, name, None)
    monkeypatch.setattr(
        op,
        "build_ep_mesh",
        lambda: types.SimpleNamespace(shape={op.EP_AXIS_NAME: CORES}),
    )
    monkeypatch.setattr(
        op, "get_ep_group", lambda: types.SimpleNamespace(ranks=list(MESH))
    )
    monkeypatch.setattr(op, "ep_mesh_index", lambda: 0)
    monkeypatch.setattr(op.torch, "tensor", lambda *a, **k: "rank")
    built = []
    monkeypatch.setattr(
        op,
        "_build_all_reduce",
        lambda mesh, psum_above=None: built.append(psum_above) or psum_above,
    )
    monkeypatch.setattr(op, "_build_reduce_scatter", lambda mesh: "rs")
    import vllm.distributed.parallel_state as ps

    group = types.SimpleNamespace(ranks=list(MESH))
    monkeypatch.setattr(ps, "get_dp_group", lambda: group)
    monkeypatch.setattr(ps, "get_tp_group", lambda: group)
    op.prebuild()
    assert built == ([None, 32] if draft else [None])
    assert (op._AR_SIZED_OP == 32) if draft else op._AR_SIZED_OP is None


def test_combine_needs_the_dp_group_to_be_the_mesh(armed, monkeypatch):
    _, rs = armed
    monkeypatch.setattr(op, "_DP_COMBINE", False)
    x = _block(8 * CORES)
    fallback = Mock(return_value=object())
    assert op.dp_combine_or(x, fallback) is fallback.return_value
    rs.assert_not_called()


def test_unarmed_is_the_fallback(monkeypatch):
    monkeypatch.setattr(op, "_AR_OP", None)
    monkeypatch.setattr(op, "_RS_OP", None)
    x = _block(8)
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, MESH, fallback) is fallback.return_value
    assert op.dp_combine_or(x, fallback) is fallback.return_value


def test_prebuild_is_a_no_op_with_the_flag_off(monkeypatch):
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", False)
    build = Mock()
    monkeypatch.setattr(op, "build_ep_mesh", build)
    monkeypatch.setattr(op, "_AR_OP", None)
    op.prebuild()
    build.assert_not_called()
    assert op._AR_OP is None


def test_prebuild_refuses_a_deployment_without_an_ep_group(monkeypatch):
    monkeypatch.setenv("TPU_SMALL_ALL_REDUCE", "1")
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", True)
    monkeypatch.setattr(op, "build_ep_mesh", lambda: None)
    monkeypatch.setattr(op, "_AR_OP", None)
    with pytest.raises(NotImplementedError, match="TPU_SMALL_ALL_REDUCE"):
        op.prebuild()


def test_prebuild_default_skips_a_deployment_without_an_ep_group(monkeypatch):
    monkeypatch.delenv("TPU_SMALL_ALL_REDUCE", raising=False)
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", True)
    monkeypatch.setattr(op, "build_ep_mesh", lambda: None)
    monkeypatch.setattr(op, "_AR_OP", None)
    op.prebuild()
    assert op._AR_OP is None


def test_retain_is_the_identity_when_unarmed_or_outside_the_warmup(monkeypatch):
    x = _block(8)
    monkeypatch.setattr(op, "_RANK", None)
    assert op.retain(x) is x
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    assert op.retain(x) is x


# --- TPU_SMALL_ALL_REDUCE_VALID_ROWS -----------------------------------------


@pytest.fixture
def valid_rows(armed, monkeypatch):
    """The valid-rows op armed beside the two-shot one, with a padding mask
    marking rows [5, 64) of the step as padding."""
    from vllm_torchtpu.layers.adapter import token_padding

    valid_op = Mock(name="all_reduce_valid_rows_op")
    monkeypatch.setattr(op, "_AR_VALID_OP", valid_op)
    monkeypatch.setattr(op, "_VALID_TILE_ROWS", 16)
    state = token_padding.TokenPaddingState.create(64, "cpu")
    state.update(5, 64)
    monkeypatch.setattr(token_padding, "_padding_state", state)
    return valid_op


@pytest.mark.parametrize("rows", [16, 24, 32])
def test_valid_rows_op_takes_blocks_with_a_fold_path_with_the_mask_count(
    armed, valid_rows, rows
):
    ar, _ = armed
    x = _block(rows)
    assert op.all_reduce_or(x, MESH, _fallback) is valid_rows.return_value
    (block, n_valid, rank), _ = valid_rows.call_args
    assert block is x and rank is op._RANK
    assert n_valid.dtype == torch.int32
    assert n_valid.tolist() == [5]
    ar.assert_not_called()


def test_valid_rows_count_is_the_whole_block_without_padding(
    armed, valid_rows, monkeypatch
):
    from vllm_torchtpu.layers.adapter import token_padding

    token_padding._padding_state.update(64, 64)
    op.all_reduce_or(_block(32), MESH, _fallback)
    (_, n_valid, _), _ = valid_rows.call_args
    assert n_valid.tolist() == [32]


@pytest.mark.parametrize("rows", [1, 8])
def test_valid_rows_below_a_row_tile_keeps_the_two_shot_and_reads_the_mask(
    armed, valid_rows, monkeypatch, rows
):
    """Fewer rows than a row tile have no fold path: the two-shot kernel, its
    input carrying a live zero read of the mask (a NaN mask entry poisons
    it)."""
    from vllm_torchtpu.layers.adapter import token_padding

    ar, _ = armed
    x = _block(rows)
    op.all_reduce_or(x, MESH, _fallback)
    valid_rows.assert_not_called()
    (block, rank), _ = ar.call_args
    torch.testing.assert_close(block, x, rtol=0, atol=0)
    monkeypatch.setattr(
        token_padding._padding_state,
        "local_padding_mask",
        torch.full((64,), float("nan")),
    )
    op.all_reduce_or(x, MESH, _fallback)
    (block, _), _ = ar.call_args
    assert torch.isnan(block).all()


def test_valid_rows_fallback_reads_the_mask(armed, valid_rows, monkeypatch):
    from vllm_torchtpu.layers.adapter import token_padding

    monkeypatch.setattr(
        token_padding._padding_state,
        "local_padding_mask",
        torch.full((256,), float("nan")),
    )
    out = op.all_reduce_or(_block(op.MAX_ROWS + 1), MESH, _fallback)
    assert torch.isnan(out).all()
    valid_rows.assert_not_called()


def test_valid_rows_without_a_padding_mask_is_the_two_shot(
    armed, valid_rows, monkeypatch
):
    from vllm_torchtpu.layers.adapter import token_padding

    monkeypatch.setattr(token_padding, "_padding_state", None)
    ar, _ = armed
    x = _block(16)
    assert op.all_reduce_or(x, MESH, _fallback) is ar.return_value
    valid_rows.assert_not_called()


def test_valid_rows_leaves_the_draft_traces_on_the_sized_op(
    armed, valid_rows, sized, monkeypatch
):
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    assert op.all_reduce_or(_block(16), MESH, _fallback) is sized.return_value
    valid_rows.assert_not_called()

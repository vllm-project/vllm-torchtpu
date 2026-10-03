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

import pathlib
import sys
import types
from unittest.mock import Mock

import pytest
import torch

try:
    from vllm_torchtpu.layers.adapter import small_all_reduce_op as op
except ModuleNotFoundError:
    import vllm_torchtpu

    _root = pathlib.Path(vllm_torchtpu.__file__).resolve().parent
    for _pkg, _sub in (
        ("vllm_torchtpu.layers", "layers"),
        ("vllm_torchtpu.layers.adapter", "layers/adapter"),
    ):
        if _pkg not in sys.modules:
            _mod = types.ModuleType(_pkg)
            _mod.__path__ = [str(_root / _sub)]
            sys.modules[_pkg] = _mod
    from vllm_torchtpu.layers.adapter import small_all_reduce_op as op

pytestmark = pytest.mark.cpu_test

CORES = 8
MESH = tuple(range(CORES))
HIDDEN = 4096


@pytest.fixture
def armed(monkeypatch):
    """Ops replaced by mocks, inside a warmup trace."""
    ar, rs = Mock(name="all_reduce_op"), Mock(name="reduce_scatter_op")
    ag, a2a = Mock(name="all_gather_op"), Mock(name="all_to_all_op")
    monkeypatch.setattr(op, "_AR_OP", ar)
    monkeypatch.setattr(op, "_RS_OP", rs)
    monkeypatch.setattr(op, "_AG_OP", ag)
    monkeypatch.setattr(op, "_A2A_OP", a2a)
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op, "_CORES", CORES)
    monkeypatch.setattr(op, "_MESH_RANKS", MESH)
    monkeypatch.setattr(op, "_DP_COMBINE", True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: True)
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


@pytest.mark.parametrize("rows", [1, 16, 40])
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


@pytest.mark.parametrize("rows_per_engine,armed_expected", [(40, True), (41, False)])
def test_combine_cap_is_40_rows_per_engine(armed, rows_per_engine, armed_expected):
    _, rs = armed
    op.dp_combine_or(_block(rows_per_engine * CORES), _fallback)
    assert rs.called is armed_expected


@pytest.mark.parametrize("rows,armed_expected", [(96, True), (97, False)])
def test_all_reduce_cap_is_96_rows(armed, rows, armed_expected):
    ar, _ = armed
    op.all_reduce_or(_block(rows), MESH, _fallback)
    assert ar.called is armed_expected


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
    assert op.maybe_small_all_gather(x, MESH, fallback) is fallback.return_value
    assert op.maybe_small_all_to_all(x, MESH, fallback) is fallback.return_value
    ar.assert_not_called()
    rs.assert_not_called()
    op._AG_OP.assert_not_called()
    op._A2A_OP.assert_not_called()


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
    monkeypatch.setattr(op, "_AG_OP", None)
    monkeypatch.setattr(op, "_A2A_OP", None)
    x = _block(8)
    fallback = Mock(return_value=object())
    assert op.all_reduce_or(x, MESH, fallback) is fallback.return_value
    assert op.dp_combine_or(x, fallback) is fallback.return_value
    assert op.maybe_small_all_gather(x, MESH, fallback) is fallback.return_value
    assert op.maybe_small_all_to_all(x, MESH, fallback) is fallback.return_value


def test_prebuild_is_a_no_op_with_the_flag_off(monkeypatch):
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", False)
    monkeypatch.delenv("TPU_SMALL_COLLECTIVES", raising=False)
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


@pytest.mark.parametrize("rows,armed_expected", [(1, True), (8, True), (64, True), (96, True), (97, False), (128, False)])
def test_maybe_small_all_gather_cap_is_96_rows(armed, rows, armed_expected):
    x = _block(rows, hidden=6144)
    out = op.maybe_small_all_gather(x, MESH, _fallback)
    assert op._AG_OP.called is armed_expected
    if armed_expected:
        assert out is op._AG_OP.return_value
        (block, rank), _ = op._AG_OP.call_args
        assert block is x and rank is op._RANK
    else:
        torch.testing.assert_close(out, _fallback(x), rtol=0, atol=0)


@pytest.mark.parametrize("rows,armed_expected", [(1, True), (8, True), (64, True), (96, True), (97, False), (104, False), (128, False)])
def test_maybe_small_all_to_all_cap_is_96_rows(armed, rows, armed_expected):
    x = _block(rows, hidden=6144)
    out = op.maybe_small_all_to_all(x, MESH, _fallback)
    assert op._A2A_OP.called is armed_expected
    if armed_expected:
        assert out is op._A2A_OP.return_value
        (block, rank), _ = op._A2A_OP.call_args
        assert block is x and rank is op._RANK
    else:
        torch.testing.assert_close(out, _fallback(x), rtol=0, atol=0)


@pytest.mark.parametrize(
    "x,ranks",
    [
        (_block(8, dtype=torch.float32), MESH),
        (_block(8, hidden=192), MESH),  # not a multiple of LANES=128
        (_block(8).reshape(2, 4, HIDDEN), MESH),
        (_block(8), (0, 1, 2, 3)),
    ],
    ids=["fp32", "unaligned_width", "3d", "group"],
)
def test_ineligible_all_gather_and_all_to_all_fall_back_untouched(armed, x, ranks):
    fallback = Mock(return_value=object())
    assert op.maybe_small_all_gather(x, ranks, fallback) is fallback.return_value
    assert op.maybe_small_all_to_all(x, ranks, fallback) is fallback.return_value
    assert fallback.call_count == 2
    op._AG_OP.assert_not_called()
    op._A2A_OP.assert_not_called()


@pytest.mark.parametrize("rows", [1, 2, 8, 32, 96])
def test_h6144_on_32_cores_2x2x4_arms_cleanly(armed, monkeypatch, rows):
    """GLM-5.2 hidden_size=6144 on 32 cores (2x2x4) satisfies 6144 % 128 == 0
    even though 6144 % (32 * 128) != 0, and must arm across all 4 small ops."""
    ar, rs = armed
    cores_32 = 32
    mesh_32 = tuple(range(cores_32))
    monkeypatch.setattr(op, "_CORES", cores_32)
    monkeypatch.setattr(op, "_MESH_RANKS", mesh_32)

    x = _block(rows, hidden=6144)
    assert op.all_reduce_or(x, mesh_32, _fallback) is ar.return_value
    assert op.maybe_small_all_gather(x, mesh_32, _fallback) is op._AG_OP.return_value
    assert op.maybe_small_all_to_all(x, mesh_32, _fallback) is op._A2A_OP.return_value

    if rows <= op.COMBINE_MAX_ROWS:
        x_rs = _block(rows * cores_32, hidden=6144)
        assert op.dp_combine_or(x_rs, _fallback) is rs.return_value


def test_fused_q_head_all_gather_3d_and_2d_single_launch_parity(armed, monkeypatch):
    """b/568770670: q_nope + q_rope are merged into a single all-gather launch
    and unmerged with exact bit-for-bit parity."""
    num_tokens = 6
    local_heads = 8
    nope_dim = 512
    rope_dim = 64

    # Simulate 8-rank all-gather along axis 0 of the 2D payload
    rank_q_nope = [
        torch.randn(num_tokens, local_heads, nope_dim, dtype=torch.bfloat16)
        for _ in range(CORES)
    ]
    rank_q_rope = [
        torch.randn(num_tokens, local_heads, rope_dim, dtype=torch.bfloat16)
        for _ in range(CORES)
    ]

    # Reference separate all_gather_heads along dim=1
    ref_nope_3d = torch.cat(rank_q_nope, dim=1)
    ref_rope_3d = torch.cat(rank_q_rope, dim=1)

    def fake_ag_3d(x_2d, rank):
        # Reconstruct what each rank contributed to the merged 2D tensor
        parts = [
            torch.cat([rank_q_nope[r], rank_q_rope[r]], dim=-1).reshape(
                num_tokens, local_heads * (nope_dim + rope_dim)
            )
            for r in range(CORES)
        ]
        return torch.cat(parts, dim=0)

    ag_mock = Mock(name="ag_3d", side_effect=fake_ag_3d)
    monkeypatch.setattr(op, "_AG_OP", ag_mock)

    out_nope_3d, out_rope_3d = op.fused_q_head_all_gather(
        rank_q_nope[3], rank_q_rope[3], MESH, _fallback, gather_dim=1
    )
    ag_mock.assert_called_once()
    assert torch.equal(out_nope_3d, ref_nope_3d)
    assert torch.equal(out_rope_3d, ref_rope_3d)

    # Also test 2D gather_dim=0
    q_nope_2d = rank_q_nope[3].reshape(num_tokens, local_heads * nope_dim)
    q_rope_2d = rank_q_rope[3].reshape(num_tokens, local_heads * rope_dim)
    ag_mock_2d = Mock(
        name="ag_2d",
        side_effect=lambda x_2d, rank: x_2d.repeat(CORES, 1),
    )
    monkeypatch.setattr(op, "_AG_OP", ag_mock_2d)
    out_nope_2d, out_rope_2d = op.fused_q_head_all_gather(
        q_nope_2d, q_rope_2d, MESH, lambda t: t.repeat(CORES, 1), gather_dim=0
    )
    ag_mock_2d.assert_called_once()
    assert torch.equal(out_nope_2d, q_nope_2d.repeat(CORES, 1))
    assert torch.equal(out_rope_2d, q_rope_2d.repeat(CORES, 1))


def test_fused_topk_scores_indices_all_gather_single_launch_parity(armed, monkeypatch):
    """b/568775832: candidate scores (bf16/fp32) + indices (int32) are packed
    into a single all-gather launch and unpacked with exact bit parity."""
    num_tokens = 6
    k = 128

    rank_scores = [
        torch.randn(num_tokens, k, dtype=torch.bfloat16) for _ in range(CORES)
    ]
    rank_indices = [
        torch.randint(-1, 131072, (num_tokens, k), dtype=torch.int32)
        for _ in range(CORES)
    ]
    # Include bit patterns that would look like NaN if interpreted as float32
    rank_indices[0][0, 0] = 0x7FC00001
    rank_indices[3][2, 5] = 0x7F800123

    def fake_ag_packed(packed_2d, rank):
        parts = []
        for r in range(CORES):
            s_bf16 = rank_scores[r]
            i_bf16 = rank_indices[r].view(torch.bfloat16)
            parts.append(torch.cat([s_bf16, i_bf16], dim=1))
        return torch.cat(parts, dim=0)

    ag_mock = Mock(name="ag_packed", side_effect=fake_ag_packed)
    monkeypatch.setattr(op, "_AG_OP", ag_mock)

    g_scores, g_indices = op.fused_topk_scores_indices_all_gather(
        rank_scores[3],
        rank_indices[3],
        MESH,
        fallback=lambda t: t.repeat(CORES, 1),
        gather_dim=0,
    )
    ag_mock.assert_called_once()
    assert torch.equal(g_scores, torch.cat(rank_scores, dim=0))
    assert torch.equal(g_indices, torch.cat(rank_indices, dim=0))
    assert g_indices.dtype == torch.int32

    # Also test gather_dim=1 (candidate axis concatenation [T, CORES * k])
    ag_mock.reset_mock()
    g_scores_d1, g_indices_d1 = op.fused_topk_scores_indices_all_gather(
        rank_scores[3],
        rank_indices[3],
        MESH,
        fallback=lambda t: t.repeat(CORES, 1),
        gather_dim=1,
    )
    ag_mock.assert_called_once()
    assert torch.equal(g_scores_d1, torch.cat(rank_scores, dim=1))
    assert torch.equal(g_indices_d1, torch.cat(rank_indices, dim=1))

    # Negative control: rows > ALL_GATHER_MAX_ROWS (97 rows) falls back without
    # corrupting int32 indices via float retain()
    ag_mock.reset_mock()
    large_scores = torch.randn(97, k, dtype=torch.bfloat16)
    large_indices = torch.randint(-1, 131072, (97, k), dtype=torch.int32)
    large_indices[0, 0] = 0x7FC00001
    fb_scores, fb_indices = op.fused_topk_scores_indices_all_gather(
        large_scores,
        large_indices,
        MESH,
        fallback=lambda t: t.repeat(CORES, 1),
        gather_dim=0,
    )
    ag_mock.assert_not_called()
    assert torch.equal(fb_scores, large_scores.repeat(CORES, 1))
    assert torch.equal(fb_indices, large_indices.repeat(CORES, 1))


def test_lse_a2a_post_attn_rs_and_moe_dispatch_helpers(armed, monkeypatch):
    """Verify maybe_lse_all_to_all (f32 and bf16), maybe_post_attn_reduce_scatter,
    and maybe_moe_dispatch_all_gather arming and fallback behavior."""
    _, rs = armed

    # 1. LSE all-to-all (float32 [16, 128] -> 256 bf16 lanes)
    lse_f32 = torch.randn(16, 128, dtype=torch.float32)
    a2a_mock = Mock(name="a2a_lse", side_effect=lambda w, r: w.clone())
    monkeypatch.setattr(op, "_A2A_OP", a2a_mock)
    out_lse = op.maybe_lse_all_to_all(lse_f32, MESH, _fallback)
    a2a_mock.assert_called_once()
    assert out_lse.dtype == torch.float32
    assert torch.equal(out_lse, lse_f32)

    # LSE > 96 rows declines and falls back
    a2a_mock.reset_mock()
    lse_large = torch.randn(104, 128, dtype=torch.float32)
    out_lse_large = op.maybe_lse_all_to_all(lse_large, MESH, _fallback)
    a2a_mock.assert_not_called()
    torch.testing.assert_close(out_lse_large, _fallback(lse_large), rtol=0, atol=0)

    # 2. Post-attention reduce-scatter (3D scatter_dim=1 and 2D scatter_dim=0)
    rs_3d_mock = Mock(
        name="rs_3d",
        side_effect=lambda x_2d, r: x_2d[: x_2d.shape[0] // CORES],
    )
    monkeypatch.setattr(op, "_RS_OP", rs_3d_mock)
    attn_partial_3d = torch.randn(6, CORES * 8, 512, dtype=torch.bfloat16)
    out_rs_3d = op.maybe_post_attn_reduce_scatter(
        attn_partial_3d, MESH, _fallback, scatter_dim=1
    )
    rs_3d_mock.assert_called_once()
    assert out_rs_3d.shape == (6, 8, 512)

    # 3. MoE dispatch all-gather
    ag_moe_mock = Mock(name="ag_moe", side_effect=lambda x, r: x.repeat(CORES, 1))
    monkeypatch.setattr(op, "_AG_OP", ag_moe_mock)
    moe_hs = _block(6, hidden=6144)
    out_moe = op.maybe_moe_dispatch_all_gather(moe_hs, MESH, _fallback)
    ag_moe_mock.assert_called_once()
    assert out_moe.shape == (6 * CORES, 6144)


def test_prebuild_arms_all_four_ops_and_logs_small_collectives_armed(
    monkeypatch, caplog
):
    """M3/M4 required_log_patterns: prebuild() builds all 4 Pallas small
    collective ops and logs 'Small collectives armed'."""
    import logging

    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE", True)
    monkeypatch.setattr(op, "_AR_OP", None)
    monkeypatch.setattr(op, "_RS_OP", None)
    monkeypatch.setattr(op, "_AG_OP", None)
    monkeypatch.setattr(op, "_A2A_OP", None)

    fake_mesh = types.SimpleNamespace(shape={op.EP_AXIS_NAME: 16})
    monkeypatch.setattr(op, "build_ep_mesh", lambda: fake_mesh)
    monkeypatch.setattr(
        op, "get_ep_group", lambda: types.SimpleNamespace(ranks=tuple(range(16)))
    )
    monkeypatch.setattr(op, "ep_mesh_index", lambda: 0)

    built_ops = []

    def fake_sharded_jax_op(name, fn, **kwargs):
        built_ops.append(name)
        return Mock(name=name)

    monkeypatch.setattr(op, "sharded_jax_op", fake_sharded_jax_op)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_vllm_logger.init_logger = logging.getLogger
    fake_ps = types.ModuleType("vllm.distributed.parallel_state")
    fake_ps.get_dp_group = lambda: types.SimpleNamespace(ranks=tuple(range(16)))
    fake_ps.get_tp_group = lambda: types.SimpleNamespace(ranks=tuple(range(16)))
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", fake_vllm_logger)
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", types.ModuleType("vllm.distributed")
    )
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", fake_ps)

    real_tensor = torch.tensor
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda data, dtype=None, device=None: real_tensor(data, dtype=dtype),
    )

    with caplog.at_level(logging.INFO):
        op.prebuild()

    assert built_ops == [
        "pallas::small_all_reduce",
        "pallas::small_reduce_scatter",
        "pallas::small_all_gather",
        "pallas::small_all_to_all",
    ]
    assert op._AR_OP is not None
    assert op._RS_OP is not None
    assert op._AG_OP is not None
    assert op._A2A_OP is not None
    assert any("Small collectives armed" in rec.message for rec in caplog.records)



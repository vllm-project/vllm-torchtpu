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

"""Unit tests for fused o_proj + all-reduce + RMSNorm in small_all_reduce_op."""

import types
from unittest.mock import Mock

import pytest
import torch
import vllm.distributed.parallel_state as ps
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.layers.adapter import small_all_reduce_op as op
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.linear_common import FP8_PALLAS_N_ATTR

CORES = 8
MESH = tuple(range(CORES))
K_OPROJ, N_OPROJ = 1024, 6144


class _RowParallel(RowParallelLinear):
    """A per-channel FP8 RowParallelLinear stub on the weight-streaming kernel."""

    def __init__(self, k=K_OPROJ, n=N_OPROJ, **attrs):
        torch.nn.Module.__init__(self)
        self.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.zeros(k, n).to(torch.float8_e4m3fn), requires_grad=False
            ),
        )
        self.register_parameter(
            "weight_scale",
            torch.nn.Parameter(torch.ones(n), requires_grad=False),
        )
        self.bias = None
        self.reduce_results = True
        self.tp_size = CORES
        setattr(self, FP8_PALLAS_N_ATTR, n)
        for name, value in attrs.items():
            setattr(self, name, value)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return (x[:, :1].repeat(1, self.weight.shape[1]), None)


def _block(rows, hidden=K_OPROJ, dtype=torch.bfloat16):
    return torch.randn(rows, hidden).to(dtype)


def _norm(hidden=N_OPROJ, eps=1e-6, **extra):
    attrs = {
        "has_weight": True,
        "variance_size_override": None,
        "variance_epsilon": eps,
        "weight": torch.ones(hidden, dtype=torch.bfloat16),
    }
    attrs.update(extra)
    return types.SimpleNamespace(**attrs)


@pytest.fixture
def fused(monkeypatch):
    ar = Mock(name="all_reduce_op")
    mm_ar = Mock(name="matmul_all_reduce_op")
    monkeypatch.setattr(op, "_AR_OP", ar)
    monkeypatch.setattr(op, "_AR_VALID_OP", Mock(name="ar_valid_op"))
    monkeypatch.setattr(op, "_MM_AR_OP", mm_ar)
    monkeypatch.setattr(op, "_MM_AR_ONE_ROW_OP", None)
    monkeypatch.setattr(op, "_MM_AR_RN_OPS", {})
    monkeypatch.setattr(op, "_MM_AR_RN_ONE_ROW_OPS", {})
    monkeypatch.setattr(op, "_RANK", torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(op, "_CORES", CORES)
    monkeypatch.setattr(op, "_MESH_RANKS", MESH)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: True)
    return mm_ar


@pytest.fixture
def one_row(fused, monkeypatch):
    mm_ar_one = Mock(name="matmul_all_reduce_one_row_op")
    monkeypatch.setattr(op, "_MM_AR_ONE_ROW_OP", mm_ar_one)
    state = token_padding.TokenPaddingState.create(64, "cpu")
    state.update(1, 64)
    monkeypatch.setattr(token_padding, "_padding_state", state)
    return mm_ar_one


@pytest.mark.parametrize(
    "rows,expected",
    [(4, False), (5, True), (8, True), (16, True), (64, True), (65, False)],
)
def test_should_gate_fused_rmsnorm_to_m5_through_m64(
    fused, monkeypatch, rows, expected
):
    rn_mock = Mock(name="mm_ar_rn_op")
    monkeypatch.setattr(op, "_MM_AR_RN_OPS", {1e-6: rn_mock})
    assert op.can_fuse_rmsnorm(rows, _RowParallel(), _norm()) is expected


def test_should_reject_ineligible_norm_for_fused_rmsnorm(fused, monkeypatch):
    rn_mock = Mock(name="mm_ar_rn_op")
    monkeypatch.setattr(op, "_MM_AR_RN_OPS", {1e-6: rn_mock})
    layer = _RowParallel()
    assert not op.can_fuse_rmsnorm(8, layer, _norm(eps=1e-5))
    assert not op.can_fuse_rmsnorm(8, layer, _norm(has_weight=False))
    assert not op.can_fuse_rmsnorm(8, layer, _norm(variance_size_override=64))
    assert not op.can_fuse_rmsnorm(8, layer, _norm(_prequant_slot=object()))
    assert not op.can_fuse_rmsnorm(8, layer, _norm(hidden=N_OPROJ - 128))


def test_should_build_rmsnorm_op_regardless_of_prepare_and_prebuild_order(monkeypatch):
    from jax.experimental.pallas import tpu as pltpu

    monkeypatch.setattr(op, "glm_dsa_v7x_active", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: CORES)
    monkeypatch.setattr(op, "ep_device_ids", lambda: list(range(CORES)))
    monkeypatch.setattr(op, "get_pipeline_stage", lambda: (0, 1))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: True)
    for name in (
        "_AR_OP",
        "_AR_VALID_OP",
        "_RANK",
        "_MM_AR_OP",
        "_MM_AR_ONE_ROW_OP",
        "_MESH",
    ):
        monkeypatch.setattr(op, name, None)
    monkeypatch.setattr(op, "_MM_AR_RN_OPS", {})
    monkeypatch.setattr(op, "_MM_AR_RN_ONE_ROW_OPS", {})
    monkeypatch.setattr(op, "_PENDING_RMSNORM_EPS", set())
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
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: types.SimpleNamespace(get_sublane_tiling=lambda _dt: 8),
    )
    monkeypatch.setattr(op, "_build_all_reduce", lambda mesh: "ar")
    monkeypatch.setattr(op, "_build_valid_rows_all_reduce", lambda mesh, tr: "ar_valid")
    monkeypatch.setattr(op, "_build_matmul_all_reduce", lambda mesh, waves: "mm_ar")
    monkeypatch.setattr(
        op, "_build_matmul_all_reduce_one_row", lambda mesh, waves: "mm_ar_one_row"
    )
    built = []
    monkeypatch.setattr(
        op,
        "_build_matmul_all_reduce_rmsnorm",
        lambda mesh, waves, eps, one_row: (
            built.append((eps, one_row)) or f"rn_{eps}_{one_row}"
        ),
    )
    group = types.SimpleNamespace(ranks=list(MESH))
    monkeypatch.setattr(ps, "get_tp_group", lambda: group)
    layer = _RowParallel()

    op.prepare_rmsnorm_op(1e-6)
    assert not op.can_fuse_rmsnorm(8, layer, _norm(eps=1e-6))
    assert op.prebuild() is True
    assert op.can_fuse_rmsnorm(8, layer, _norm(eps=1e-6))
    assert not op.can_fuse_rmsnorm(8, layer, _norm(eps=1e-5))

    op.prepare_rmsnorm_op(1e-5)
    assert op.can_fuse_rmsnorm(8, layer, _norm(eps=1e-5))
    assert built == [(1e-6, False), (1e-6, True), (1e-5, False), (1e-5, True)]


def test_should_keep_rank_before_norm_weight_for_bucket_handover(one_row, monkeypatch):
    rn_base = Mock(
        name="mm_ar_rn_base",
        return_value=(torch.zeros(8, N_OPROJ), torch.zeros(8, N_OPROJ)),
    )
    rn_one = Mock(
        name="mm_ar_rn_one",
        return_value=(torch.zeros(8, N_OPROJ), torch.zeros(8, N_OPROJ)),
    )
    monkeypatch.setattr(op, "_MM_AR_RN_OPS", {1e-6: rn_base})
    monkeypatch.setattr(op, "_MM_AR_RN_ONE_ROW_OPS", {1e-6: rn_one})
    token_padding._padding_state.update(6, 64)
    layer = _RowParallel()
    x = _block(8)
    residual = _block(8, hidden=N_OPROJ)
    norm_w = torch.ones(N_OPROJ, dtype=torch.bfloat16)

    op.row_parallel_or_rmsnorm(x, layer, residual, norm_w, 1e-6)
    (blk, w, w_s, n_valid, rank, res_arg, nw_arg), _ = rn_one.call_args
    torch.testing.assert_close(blk, x, rtol=0, atol=0)
    assert w is layer.weight and w_s is layer.weight_scale
    assert n_valid.tolist() == [6]
    assert rank is op._RANK
    assert res_arg is residual and nw_arg is norm_w

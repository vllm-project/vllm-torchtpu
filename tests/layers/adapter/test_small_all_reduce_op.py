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


def test_large_block_rank_read_rides_on_the_input_with_the_pallas_norm(
    armed, monkeypatch
):
    """With TPU_FUSED_ADD_RMSNORM the zero is added before the collective, where
    XLA fuses it into the producer instead of a pass over the Pallas norm's
    input."""
    monkeypatch.setattr(op.envs, "TPU_FUSED_ADD_RMSNORM", True)
    monkeypatch.setattr(op, "_RANK", torch.tensor([float("nan")]))
    fallback = Mock(side_effect=lambda x: x)
    op.all_reduce_or(_block(op.MAX_ROWS + 1), MESH, fallback)
    assert torch.isnan(fallback.call_args.args[0]).all()


# TPU_FUSED_OPROJ_AR: row_parallel_or
K_OPROJ, N_OPROJ = 1024, 6144


class _RowParallel:
    """A per-channel FP8 RowParallelLinear on the weight-streaming kernel."""

    def __init__(self, k=K_OPROJ, n=N_OPROJ, **attrs):
        from vllm_torchtpu.layers.adapter.linear_common import FP8_PALLAS_N_ATTR

        self.weight = torch.zeros(k, n).to(torch.float8_e4m3fn)
        self.weight_scale = torch.ones(n)
        self.bias = None
        self.reduce_results = True
        self.tp_size = CORES
        setattr(self, FP8_PALLAS_N_ATTR, n)
        for name, value in attrs.items():
            setattr(self, name, value)
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return (x[:, :1].repeat(1, self.weight.shape[1]), None)


@pytest.fixture
def fused(armed, monkeypatch):
    mm_ar = Mock(name="matmul_all_reduce_op")
    monkeypatch.setattr(op, "_MM_AR_OP", mm_ar)
    return mm_ar


@pytest.mark.parametrize("rows", [1, 16])
def test_decode_block_takes_the_fused_kernel(fused, rows):
    layer = _RowParallel()
    x = _block(rows, hidden=K_OPROJ)
    assert op.row_parallel_or(x, layer) is fused.return_value
    (block, w, w_s, rank), _ = fused.call_args
    torch.testing.assert_close(block, x, rtol=0, atol=0)
    assert w is layer.weight and w_s is layer.weight_scale and rank is op._RANK
    assert layer.calls == 0


@pytest.mark.parametrize(
    "rows,layer",
    [
        (17, _RowParallel()),  # above the kernel's 16 rows
        (4, _RowParallel(k=2048, n=16384)),  # not a tuned shape
        (4, _RowParallel(bias=torch.zeros(N_OPROJ))),
        (4, _RowParallel(reduce_results=False)),
        (4, _RowParallel(tp_size=CORES // 2)),  # TP group is not the mesh
        (4, _RowParallel(weight_pallas=torch.zeros(1))),  # padded kernel copy
        (4, _RowParallel(_tpu_keep_vllm_weight_layout=True)),
        (4, _RowParallel(_tpu_prequant_slot=object())),
        (4, _RowParallel(_tpu_fp8_pallas_n_out=None)),  # not on the kernel
    ],
    ids=[
        "rows17",
        "shape",
        "bias",
        "no_reduce",
        "tp_not_mesh",
        "padded",
        "vllm_layout",
        "prequant",
        "not_fp8_pallas",
    ],
)
def test_other_blocks_keep_the_layer_forward(fused, rows, layer):
    x = _block(rows, hidden=layer.weight.shape[0])
    out = op.row_parallel_or(x, layer)
    fused.assert_not_called()
    assert layer.calls == 1
    torch.testing.assert_close(out, layer(x)[0], rtol=0, atol=0)


@pytest.mark.parametrize("state", ["unarmed", "no_warmup"])
def test_unarmed_or_outside_warmup_keeps_the_layer_forward(fused, monkeypatch, state):
    if state == "unarmed":
        monkeypatch.setattr(op, "_MM_AR_OP", None)
    else:
        monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: False)
    layer = _RowParallel()
    op.row_parallel_or(_block(4, hidden=K_OPROJ), layer)
    fused.assert_not_called()
    assert layer.calls == 1


def test_fused_input_depends_on_the_rank_buffer(fused, monkeypatch):
    """The fused side reads the rank buffer, as the other side of the row fork
    does, so both warmup traces lift the same buffers."""
    monkeypatch.setattr(op, "_RANK", torch.tensor([float("nan")]))
    op.row_parallel_or(_block(4, hidden=K_OPROJ), _RowParallel())
    (block, *_), _ = fused.call_args
    assert torch.isnan(block).all()

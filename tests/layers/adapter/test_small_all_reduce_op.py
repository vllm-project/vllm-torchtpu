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
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 96)
    return ar


def _block(rows, hidden=HIDDEN, dtype=torch.bfloat16):
    return torch.randn(rows, hidden).to(dtype)


def _fallback(x):
    return x * 2


@pytest.mark.parametrize(
    "max_rows_env,rows,armed_expected",
    [
        (96, 1, True),
        (96, 96, True),
        (96, 97, False),
        (32, 32, True),
        (32, 33, False),
        (32, 4096, False),
    ],
)
def test_row_cap_picks_kernel_or_collective(
    armed, monkeypatch, max_rows_env, rows, armed_expected
):
    """TPU_SMALL_ALL_REDUCE_MAX_ROWS is the cap; a block above it keeps vLLM's
    collective."""
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
    out = op.all_reduce_or(_block(97), MESH, _fallback)
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


def test_prebuild_is_a_no_op_at_cap_zero(monkeypatch):
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 0)
    build = Mock()
    monkeypatch.setattr(op, "build_ep_mesh", build)
    monkeypatch.setattr(op, "_AR_OP", None)
    op.prebuild()
    build.assert_not_called()
    assert op._AR_OP is None


def test_prebuild_refuses_a_deployment_without_an_ep_group(monkeypatch):
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 32)
    monkeypatch.setattr(op, "build_ep_mesh", lambda: None)
    monkeypatch.setattr(op, "_AR_OP", None)
    with pytest.raises(NotImplementedError, match="TPU_SMALL_ALL_REDUCE_MAX_ROWS"):
        op.prebuild()


def test_prebuild_warns_when_the_tp_group_is_not_the_ep_group(monkeypatch):
    """With attention data parallelism the TP group is a part of the
    expert-parallel group, so the setting has no effect, and prebuild says so
    instead of logging the armed line."""
    import vllm.distributed.parallel_state as parallel_state

    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 32)
    for name in ("_AR_OP", "_RANK", "_CORES", "_MESH_RANKS"):
        monkeypatch.setattr(op, name, None if name == "_AR_OP" else getattr(op, name))
    monkeypatch.setattr(
        op, "build_ep_mesh", lambda: Mock(shape={op.EP_AXIS_NAME: CORES})
    )
    monkeypatch.setattr(op, "get_ep_group", lambda: Mock(ranks=list(MESH)))
    monkeypatch.setattr(op, "ep_mesh_index", lambda: 0)
    monkeypatch.setattr(op, "_build_all_reduce", lambda mesh: Mock())
    monkeypatch.setattr(op.torch, "tensor", Mock())  # no TPU device on the host
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: Mock(ranks=[0, 1]))
    logger = Mock()
    monkeypatch.setattr(op, "logger", logger)
    op.prebuild()
    logger.warning_once.assert_called_once()
    assert "TPU_SMALL_ALL_REDUCE_MAX_ROWS" in logger.warning_once.call_args.args[0]
    logger.info.assert_not_called()


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
    """With TPU_FUSED_ADD_RMSNORM_MIN_TOKENS > 0 the zero is added before the
    collective, where XLA fuses it into the producer instead of a pass over the
    Pallas norm's input."""
    monkeypatch.setattr(op.envs, "TPU_FUSED_ADD_RMSNORM_MIN_TOKENS", 512)
    monkeypatch.setattr(op, "_RANK", torch.tensor([float("nan")]))
    fallback = Mock(side_effect=lambda x: x)
    op.all_reduce_or(_block(97), MESH, fallback)
    assert torch.isnan(fallback.call_args.args[0]).all()


# TPU_FUSED_OPROJ_AR: row_parallel_or. The default layer has the one weight
# shape (k, n) in the kernel's tuned table.
K_TUNED, N_TUNED = 1024, 6144


class _RowParallel:
    """A per-channel FP8 RowParallelLinear on the weight-streaming kernel."""

    def __init__(self, k=K_TUNED, n=N_TUNED, **attrs):
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
    x = _block(rows, hidden=K_TUNED)
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
        (4, _RowParallel(bias=torch.zeros(N_TUNED))),
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
    op.row_parallel_or(_block(4, hidden=K_TUNED), layer)
    fused.assert_not_called()
    assert layer.calls == 1


def test_fused_input_depends_on_the_rank_buffer(fused, monkeypatch):
    """The fused side reads the rank buffer, as the other side of the row fork
    does, so both warmup traces lift the same buffers."""
    monkeypatch.setattr(op, "_RANK", torch.tensor([float("nan")]))
    op.row_parallel_or(_block(4, hidden=K_TUNED), _RowParallel())
    (block, *_), _ = fused.call_args
    assert torch.isnan(block).all()


def test_layer_from_the_fp8_load_hook_takes_the_fused_kernel(fused, monkeypatch):
    """A per-channel FP8 layer processed by vLLM's post-load pass, with its site
    in TPU_FP8_LINEAR_PALLAS_SITES, passes `_fused_layer_weights` unchanged: the
    weight is K-major [k, n], the scale is [n], and there is no padded copy.

    The layer is built as in test_post_load_hook_selects_and_pads of
    tests/layers/adapter/quantization/test_fp8_linear_pallas.py (online FP8 on a
    BF16 checkpoint). It is a ReplicatedLinear, so the row-parallel attributes
    are set by hand."""
    from types import SimpleNamespace

    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod
    from vllm_torchtpu.layers.adapter.quantization.online_fp8 import (
        attach_online_fp8,
    )

    prefix = "model.layers.0.self_attn.o_proj"
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS_SITES", "self_attn.o_proj")
    config = fp8_mod.VllmFp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [prefix],
        }
    )
    attach_online_fp8(
        config,
        SimpleNamespace(
            additional_config={
                "tpu_online_fp8": {
                    "enabled": True,
                    "targets": [prefix],
                    "weight_scheme": "per_channel",
                }
            }
        ),
    )
    layer = ReplicatedLinear(
        K_TUNED,
        N_TUNED,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix=prefix,
        disable_tp=True,
    )
    model = torch.nn.Module()
    model.add_module("o_proj", layer)
    src = torch.linspace(-3, 3, N_TUNED * K_TUNED).reshape(N_TUNED, K_TUNED)
    layer.weight.weight_loader(layer.weight, src.to(torch.bfloat16))
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        quantization=None,
        word_embeddings_untied_by_checkpoint=False,
    )
    process_weights_after_loading(model, model_config, torch.device("cpu"))
    layer.reduce_results = True
    layer.tp_size = CORES

    weights = op._fused_layer_weights(layer)
    assert weights is not None
    w, w_s = weights
    assert w is layer.weight and w_s is layer.weight_scale
    assert op.row_parallel_or(_block(4, hidden=K_TUNED), layer) is fused.return_value


def test_prebuild_warns_when_the_fused_kernel_has_no_mesh(monkeypatch):
    """TPU_FUSED_OPROJ_AR without TPU_SMALL_ALL_REDUCE_MAX_ROWS logs a warning
    at load and builds nothing."""
    monkeypatch.setattr(op.envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 0)
    monkeypatch.setattr(op.envs, "TPU_FUSED_OPROJ_AR", True)
    monkeypatch.setattr(op, "build_ep_mesh", Mock())
    monkeypatch.setattr(op, "_AR_OP", None)
    logger = Mock()
    monkeypatch.setattr(op, "logger", logger)
    op.prebuild()
    op.build_ep_mesh.assert_not_called()
    (message, *_), _ = logger.warning_once.call_args
    assert (
        "TPU_FUSED_OPROJ_AR" in message and "TPU_SMALL_ALL_REDUCE_MAX_ROWS" in message
    )

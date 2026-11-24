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
"""CPU tests for routing `o_proj` to the fused attention-output FP8 kernel."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_torchtpu.kernels.quantized_matmul.attn_out_fp8 import (
    quantize_attn_out,
    select_row_tile,
    vmem_limit_bytes,
)
from vllm_torchtpu.layers.adapter import attention as attn_mod
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackendImpl,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    ATTN_OUT_HEAD_LAYOUT_ATTR,
    KEEP_VLLM_LAYOUT_ATTR,
    attn_out_quantized_matmul,
    tag_attn_out_projections,
)
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.runner import tpu_runner as runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

# (num_kv_heads, q_per_kv, head_dim) and the matching `o_proj` shape.
_LAYOUT = (2, 2, 128)
_K, _N = 512, 128
_VMEM_64M = 64 * 1024 * 1024
_VMEM_16M = 16 * 1024 * 1024


@pytest.mark.parametrize(
    "m,layout,vmem_bytes,expected",
    [
        (2048, (8, 2, 256), _VMEM_64M, 512),
        (256, (8, 2, 256), _VMEM_64M, 256),
        (32, (8, 2, 256), _VMEM_64M, 32),
        (2048, (2, 8, 512), _VMEM_64M, 256),  # 512-row tile exceeds 28 MiB budget
        (2048, (12, 2, 512), _VMEM_64M, 128),  # K=12288 on 64 MiB VMEM
        (2048, (12, 2, 512), _VMEM_16M, 32),  # K=12288 shrinks to 32 rows on 16 MiB
        (2048, (24, 2, 512), _VMEM_16M, None),  # K=24576 exceeds 16 MiB even at 32 rows
        (16, (8, 2, 256), _VMEM_64M, None),  # below the smallest row tile
        (200, (8, 2, 256), _VMEM_64M, None),  # no row tile divides m
        (256, (8, 3, 256), _VMEM_64M, None),  # odd q_per_kv
        (256, (8, 2, 64), _VMEM_64M, None),  # head_dim not lane-aligned
        (256, (8, 2, 256), 0, None),  # non-positive VMEM capacity
    ],
)
def test_select_row_tile(m, layout, vmem_bytes, expected):
    assert select_row_tile(m, *layout, vmem_capacity_bytes=vmem_bytes) == expected


def test_vmem_limit_bytes_is_tight_and_capped():
    # Small buckets reserve only their tile working set + margin (not 32 MiB),
    # while large tiles stay within half of chip VMEM so XLA weight prefetches fit.
    assert vmem_limit_bytes(128, 8, 2, 256, _VMEM_64M) == 8 * 1024 * 1024
    assert vmem_limit_bytes(512, 8, 2, 256, _VMEM_64M) == 30 * 1024 * 1024
    assert vmem_limit_bytes(256, 2, 8, 512, _VMEM_64M) == 29 * 1024 * 1024
    assert vmem_limit_bytes(32, 12, 2, 512, _VMEM_16M) == 6 * 1024 * 1024


@pytest.mark.parametrize(
    "shape,layout,vmem_bytes,match",
    [
        ((256, 512, 1), (2, 2, 128), _VMEM_64M, "Expected 2D attention output"),
        ((256, 256), (2, 2, 128), _VMEM_64M, "must equal"),
        ((16, 512), (2, 2, 128), _VMEM_64M, "No attention-output FP8 row tile"),
        ((256, 768), (1, 3, 256), _VMEM_64M, "No attention-output FP8 row tile"),
        ((256, 512), (2, 2, 128), 1024, "No attention-output FP8 row tile"),
    ],
)
def test_quantize_attn_out_validation_errors(shape, layout, vmem_bytes, match):
    x = jnp.ones(shape, jnp.bfloat16)
    with pytest.raises(ValueError, match=match):
        quantize_attn_out(x, *layout, vmem_capacity_bytes=vmem_bytes)


@pytest.mark.parametrize(
    "rows,max_rows,w_q_dtype,w_s_shape,expect_kernel",
    [
        (256, 0, jnp.float8_e4m3fn, (_N,), True),
        (16, 0, jnp.float8_e4m3fn, (_N,), False),  # no row tile
        (64, 64, jnp.float8_e4m3fn, (_N,), False),  # W8A16 bucket
        (256, 64, jnp.float8_e4m3fn, (_N,), True),
        (256, 0, jnp.float8_e4m3fn, (1, 2, 1, _N), False),  # blockwise scales
        (256, 0, jnp.int8, (_N,), False),  # int8 weights
    ],
)
def test_kernel_or_fallback(
    monkeypatch, rows, max_rows, w_q_dtype, w_s_shape, expect_kernel
):
    calls = []

    def fake_kernel(x, w_q, w_s, num_kv_heads, q_per_kv, head_dim):
        calls.append(("kernel", (num_kv_heads, q_per_kv, head_dim)))
        return jnp.zeros((x.shape[0], w_q.shape[-1]), x.dtype)

    def fake_fallback(x, w_q, w_s, w8a16_max_rows=0):
        calls.append(("fallback", w8a16_max_rows))
        return jnp.zeros((x.shape[0], w_q.shape[-1]), x.dtype)

    monkeypatch.setattr(linear_common, "attn_out_fp8_matmul", fake_kernel)
    monkeypatch.setattr(linear_common, "_quantized_matmul_jax", fake_fallback)
    monkeypatch.setattr(
        linear_common,
        "select_row_tile",
        lambda rows, kv, qpk, d: select_row_tile(
            rows, kv, qpk, d, vmem_capacity_bytes=_VMEM_64M
        ),
    )
    out = linear_common._attn_out_quantized_matmul_jax(
        jnp.ones((rows, _K), jnp.bfloat16),
        jnp.ones((_K, _N), w_q_dtype),
        jnp.ones(w_s_shape, jnp.float32),
        *_LAYOUT,
        max_rows,
    )
    assert out.shape == (rows, _N)
    assert calls == [("kernel", _LAYOUT) if expect_kernel else ("fallback", max_rows)]


@pytest.mark.parametrize("keep_vllm_layout", [False, True])
def test_apply_routes_tagged_o_proj(monkeypatch, keep_vllm_layout):
    calls = []

    def fake_attn_out(x, w_q, w_s, head_layout, w8a16_max_rows=0):
        calls.append(("attn_out", head_layout, w8a16_max_rows))
        return torch.zeros((x.shape[0], w_q.shape[-1]), dtype=x.dtype)

    def fake_quantized_matmul(x, w_q, w_s, w8a16_max_rows=0):
        calls.append(("regular", w8a16_max_rows))
        return torch.zeros((x.shape[0], w_q.shape[-1]), dtype=x.dtype)

    monkeypatch.setattr(fp8, "attn_out_quantized_matmul", fake_attn_out)
    monkeypatch.setattr(fp8, "quantized_matmul", fake_quantized_matmul)
    monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", "16")
    method = VllmFp8LinearMethodTPU(Fp8Config(is_checkpoint_fp8_serialized=True))
    layer = torch.nn.Module()
    weight_shape = (_N, _K) if keep_vllm_layout else (_K, _N)
    layer.weight = torch.nn.Parameter(
        torch.zeros(weight_shape, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((_N,), dtype=torch.float32), requires_grad=False
    )
    setattr(layer, ATTN_OUT_HEAD_LAYOUT_ATTR, _LAYOUT)
    setattr(layer, KEEP_VLLM_LAYOUT_ATTR, keep_vllm_layout)

    out = method.apply(layer, torch.ones((64, _K), dtype=torch.bfloat16))
    assert out.shape == (64, _N)
    assert calls == [("regular", 16) if keep_vllm_layout else ("attn_out", _LAYOUT, 16)]


def test_attn_out_quantized_matmul_validation_and_3d_round_trip():
    with pytest.raises(ValueError, match="Input hidden dim"):
        attn_out_quantized_matmul(
            torch.empty((8, _K + 16), dtype=torch.bfloat16, device="meta"),
            torch.empty((_K, _N), dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty((_N,), dtype=torch.float32, device="meta"),
            _LAYOUT,
        )

    with torch.device("meta"):
        x_3d = torch.empty((4, 16, _K), dtype=torch.bfloat16)
        w_q = torch.empty((_K, _N), dtype=torch.float8_e4m3fn)
        w_s = torch.empty((_N,), dtype=torch.float32)
        out_3d = attn_out_quantized_matmul(x_3d, w_q, w_s, _LAYOUT, w8a16_max_rows=16)
    assert out_3d.shape == (4, 16, _N)
    assert out_3d.dtype == torch.bfloat16


def test_attn_out_quantized_matmul_fullgraph_compile():
    compiled = torch.compile(
        lambda x, w_q, w_s: attn_out_quantized_matmul(
            x, w_q, w_s, _LAYOUT, w8a16_max_rows=16
        ),
        backend="eager",
        fullgraph=True,
    )
    with torch.device("meta"):
        w_q = torch.empty((_K, _N), dtype=torch.float8_e4m3fn)
        w_s = torch.empty((_N,), dtype=torch.float32)
        out_2d = compiled(torch.empty((64, _K), dtype=torch.bfloat16), w_q, w_s)
        out_3d = compiled(torch.empty((2, 32, _K), dtype=torch.bfloat16), w_q, w_s)
    assert out_2d.shape == (64, _N) and out_2d.dtype == torch.bfloat16
    assert out_3d.shape == (2, 32, _N) and out_3d.dtype == torch.bfloat16


def _make_batched_impl(
    head_size: int = 256,
    *,
    pcp_streaming: bool = False,
    dcp_world_size: int = 1,
    rebind_local: bool = False,
) -> PallasBatchedRPAAttentionBackendImpl:
    impl = object.__new__(PallasBatchedRPAAttentionBackendImpl)
    impl.head_size = head_size
    impl._pcp_streaming = pcp_streaming
    impl.dcp_world_size = dcp_world_size
    impl.decode_query_size = 1
    if rebind_local:
        impl._kernel_entry = attn_mod._pallas_rpa_kernel_local
    return impl


def test_emits_head_major_batched_rpa_output():
    base_impl = object.__new__(PallasAttentionBackendImpl)
    assert not base_impl.emits_head_major_batched_rpa_output()
    assert _make_batched_impl(256).emits_head_major_batched_rpa_output()
    assert not _make_batched_impl(64).emits_head_major_batched_rpa_output()
    assert not _make_batched_impl(
        256, pcp_streaming=True
    ).emits_head_major_batched_rpa_output()
    assert not _make_batched_impl(
        256, dcp_world_size=2
    ).emits_head_major_batched_rpa_output()
    assert not _make_batched_impl(
        256, rebind_local=True
    ).emits_head_major_batched_rpa_output()


def _block(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    o_proj_in: int,
    *,
    o_proj_cls=RowParallelLinear,
    impl: object | None = None,
    quant_method_cls=VllmFp8LinearMethodTPU,
    weight_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_shape: tuple[int, ...] = (_N,),
    keep_vllm_layout: bool = False,
    qkv_num_heads: int | None = None,
) -> torch.nn.Module:
    block = torch.nn.Module()
    block.attn = MagicMock(spec=Attention)
    block.attn.impl = _make_batched_impl(head_size) if impl is None else impl
    block.attn.num_heads = num_heads
    block.attn.num_kv_heads = num_kv_heads
    block.attn.head_size = head_size
    if qkv_num_heads is not None:
        qkv = MagicMock(spec=QKVParallelLinear)
        qkv.num_heads = qkv_num_heads
        block.qkv_proj = qkv
    o_proj = MagicMock(spec=o_proj_cls)
    o_proj.input_size_per_partition = o_proj_in
    o_proj.quant_method = MagicMock(spec=quant_method_cls)
    o_proj.weight = torch.empty((o_proj_in, _N), dtype=weight_dtype, device="meta")
    o_proj.weight_scale = torch.empty(scale_shape, dtype=torch.float32, device="meta")
    if keep_vllm_layout:
        setattr(o_proj, KEEP_VLLM_LAYOUT_ATTR, True)
    block.o_proj = o_proj
    return block


def test_tag_attn_out_projections_gated_by_env_and_filters(monkeypatch):
    good = _block(16, 2, 512, 16 * 512, qkv_num_heads=16)
    # Default off: returns 0 without tagging.
    monkeypatch.delenv("TPU_FUSED_ATTN_OUT_FP8", raising=False)
    assert tag_attn_out_projections(good, vmem_capacity_bytes=_VMEM_64M) == 0
    assert not hasattr(good.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)

    monkeypatch.setenv("TPU_FUSED_ATTN_OUT_FP8", "1")
    unquantized = _block(16, 2, 512, 16 * 512, quant_method_cls=UnquantizedLinearMethod)
    other_input = _block(16, 2, 512, 4096)
    uneven_heads = _block(15, 2, 512, 15 * 512)
    odd_q_per_kv = _block(6, 2, 512, 6 * 512)
    unaligned_head_dim = _block(16, 2, 80, 16 * 80)
    blockwise_scale = _block(16, 2, 512, 16 * 512, scale_shape=(1, 4, 1, _N))
    int8_weight = _block(16, 2, 512, 16 * 512, weight_dtype=torch.int8)
    kept_layout = _block(16, 2, 512, 16 * 512, keep_vllm_layout=True)
    gated_qkv = _block(16, 2, 512, 16 * 512, qkv_num_heads=32)
    not_a_linear = _block(16, 2, 512, 16 * 512, o_proj_cls=torch.nn.Identity)
    default_rpa = _block(
        16, 2, 512, 16 * 512, impl=object.__new__(PallasAttentionBackendImpl)
    )
    pcp_streaming = _block(
        16, 2, 512, 16 * 512, impl=_make_batched_impl(512, pcp_streaming=True)
    )
    dcp_layer = _block(
        16, 2, 512, 16 * 512, impl=_make_batched_impl(512, dcp_world_size=2)
    )
    rebound_draft = _block(
        16, 2, 512, 16 * 512, impl=_make_batched_impl(512, rebind_local=True)
    )
    blocks = [
        good,
        unquantized,
        other_input,
        uneven_heads,
        odd_q_per_kv,
        unaligned_head_dim,
        blockwise_scale,
        int8_weight,
        kept_layout,
        gated_qkv,
        not_a_linear,
        default_rpa,
        pcp_streaming,
        dcp_layer,
        rebound_draft,
    ]
    model = torch.nn.ModuleList(blocks)
    assert tag_attn_out_projections(model, vmem_capacity_bytes=_VMEM_64M) == 1
    assert getattr(good.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR) == (2, 8, 512)
    for block in blocks[1:]:
        assert not hasattr(block.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)

    # A chip whose VMEM cannot fit even a 32-row tile leaves the layer untagged
    # and clears any prior tag.
    assert tag_attn_out_projections(model, vmem_capacity_bytes=1024) == 0
    assert not hasattr(good.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)


def test_runner_initializes_attention_before_tagging_projections(monkeypatch):
    target_block = _block(16, 2, 256, 16 * 256)
    draft_block = _block(16, 2, 256, 16 * 256)
    events: list[str] = []

    def init_target(_layer):
        events.append("init_target")

    def init_draft(_layer):
        events.append("init_draft")

    target_block.attn.impl.initialize_kernel = init_target
    draft_block.attn.impl.initialize_kernel = init_draft

    monkeypatch.setattr(
        runner_mod,
        "get_layers_from_vllm_config",
        lambda _cfg, _cls: {
            "model.layers.0.self_attn.attn": target_block.attn,
            "draft.layers.0.self_attn.attn": draft_block.attn,
        },
    )

    runner = object.__new__(TPUModelRunner)
    runner._attention_kernels_initialized = False
    runner.mesh = None
    runner.vllm_config = MagicMock()
    runner.speculative_config = SimpleNamespace(draft_tensor_parallel_size=1)
    runner.model = target_block
    runner.drafter = SimpleNamespace(
        _draft_attn_layer_names={"draft.layers.0.self_attn.attn"},
        draft_model=draft_block,
    )

    # Flag off: kernels initialize, nothing is tagged.
    monkeypatch.setenv("TPU_FUSED_ATTN_OUT_FP8", "0")
    runner._initialize_attention_kernels()
    assert events == ["init_target", "init_draft"]
    assert not hasattr(target_block.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)
    assert not hasattr(draft_block.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)

    # Flag on with force=True: initialize_kernel and tp=1 draft rebinding run
    # before tagging, so target is tagged while the rebound tp=1 draft is not.
    monkeypatch.setenv("TPU_FUSED_ATTN_OUT_FP8", "1")
    runner._initialize_attention_kernels(force=True)
    assert getattr(target_block.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR) == (2, 8, 256)
    assert not hasattr(draft_block.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)

    # If re-initialized into DCP mode (`dcp_world_size = 2`), the stale tag on
    # target_block is cleared rather than left behind.
    def init_target_dcp(_layer):
        target_block.attn.impl.dcp_world_size = 2

    target_block.attn.impl.initialize_kernel = init_target_dcp
    runner._initialize_attention_kernels(force=True)
    assert not hasattr(target_block.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)

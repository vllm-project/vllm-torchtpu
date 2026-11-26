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
"""CPU and TPU tests for routing `o_proj` to the fused attention-output FP8 kernel."""

import functools
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import pytest
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models.afmoe import AfmoeAttention
from vllm.model_executor.models.gemma4 import Gemma4Attention
from vllm.model_executor.models.llama import LlamaAttention
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

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
from vllm_torchtpu.layers.adapter.attn_out_fp8 import (
    FUSED_ATTN_OUT_FP8_KEY,
    fused_attn_out_fp8_enabled,
    tag_attn_out_projections,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    ATTN_OUT_HEAD_LAYOUT_ATTR,
    KEEP_VLLM_LAYOUT_ATTR,
    quantized_matmul,
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


def _mock_tpu_info(monkeypatch, vmem_bytes: int, num_lanes: int = 128) -> None:
    info = SimpleNamespace(vmem_capacity_bytes=vmem_bytes, num_lanes=num_lanes)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


@pytest.mark.parametrize(
    "m,layout,act_dtype,vmem_bytes,num_lanes,expected",
    [
        (2048, (8, 2, 256), jnp.bfloat16, _VMEM_64M, 128, 512),
        (256, (8, 2, 256), jnp.bfloat16, _VMEM_64M, 128, 256),
        (32, (8, 2, 256), jnp.bfloat16, _VMEM_64M, 128, 32),
        # 512-row tile exceeds 28 MiB budget for K=8192 bf16.
        (2048, (2, 8, 512), jnp.bfloat16, _VMEM_64M, 128, 256),
        (2048, (12, 2, 512), jnp.bfloat16, _VMEM_64M, 128, 128),
        (2048, (12, 2, 512), jnp.bfloat16, _VMEM_16M, 128, 32),
        (2048, (24, 2, 512), jnp.bfloat16, _VMEM_16M, 128, None),
        (16, (8, 2, 256), jnp.bfloat16, _VMEM_64M, 128, None),
        (200, (8, 2, 256), jnp.bfloat16, _VMEM_64M, 128, None),
        # Odd q_per_kv fails for bf16 (q_packing=2), passes for f32 (q_packing=1).
        (256, (8, 3, 256), jnp.bfloat16, _VMEM_64M, 128, None),
        (256, (8, 3, 256), jnp.float32, _VMEM_64M, 128, 256),
        # float32 uses 4B/elem I/O vs 2B/elem for bf16, so K=4096 shrinks 512->256.
        (2048, (8, 2, 256), jnp.float32, _VMEM_64M, 128, 256),
        # Lane alignment comes from pltpu.get_tpu_info().num_lanes, not hardcoded 128.
        (256, (8, 2, 64), jnp.bfloat16, _VMEM_64M, 128, None),
        (256, (8, 2, 64), jnp.bfloat16, _VMEM_64M, 64, 256),
        (256, (8, 2, 256), jnp.bfloat16, 0, 128, None),
    ],
)
def test_select_row_tile(
    monkeypatch, m, layout, act_dtype, vmem_bytes, num_lanes, expected
):
    _mock_tpu_info(monkeypatch, vmem_bytes, num_lanes)
    assert select_row_tile(m, *layout, act_dtype=act_dtype) == expected


def test_vmem_limit_bytes_is_tight_and_capped(monkeypatch):
    _mock_tpu_info(monkeypatch, _VMEM_64M)
    # Small buckets reserve only their tile working set + margin (not 32 MiB),
    # while large tiles stay within half of chip VMEM so XLA weight prefetches fit.
    assert vmem_limit_bytes(128, 8, 2, 256) == 8 * 1024 * 1024
    assert vmem_limit_bytes(512, 8, 2, 256) == 30 * 1024 * 1024
    assert vmem_limit_bytes(256, 2, 8, 512) == 29 * 1024 * 1024
    # float32 activations reserve more VMEM than bfloat16 for the same tile.
    assert vmem_limit_bytes(128, 8, 2, 256, act_dtype=jnp.float32) > (
        vmem_limit_bytes(128, 8, 2, 256, act_dtype=jnp.bfloat16)
    )
    _mock_tpu_info(monkeypatch, _VMEM_16M)
    assert vmem_limit_bytes(32, 12, 2, 512) == 6 * 1024 * 1024


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
def test_quantize_attn_out_validation_errors(
    monkeypatch, shape, layout, vmem_bytes, match
):
    _mock_tpu_info(monkeypatch, vmem_bytes)
    x = jnp.ones(shape, jnp.bfloat16)
    kv, qpk, d = layout
    with pytest.raises(ValueError, match=match):
        quantize_attn_out(x, num_kv_heads=kv, q_per_kv=qpk, head_dim=d)


@pytest.mark.parametrize(
    "vllm_config,expected",
    [
        (None, False),
        (SimpleNamespace(additional_config=None), False),
        (MagicMock(), False),
        (SimpleNamespace(additional_config={}), False),
        (SimpleNamespace(additional_config={FUSED_ATTN_OUT_FP8_KEY: False}), False),
        (SimpleNamespace(additional_config={FUSED_ATTN_OUT_FP8_KEY: True}), True),
    ],
)
def test_fused_attn_out_fp8_enabled(vllm_config, expected):
    assert fused_attn_out_fp8_enabled(vllm_config) is expected


@pytest.mark.parametrize("bad_value", ["true", 1, None])
def test_fused_attn_out_fp8_enabled_rejects_non_bool(bad_value):
    cfg = SimpleNamespace(additional_config={FUSED_ATTN_OUT_FP8_KEY: bad_value})
    with pytest.raises(ValueError, match=FUSED_ATTN_OUT_FP8_KEY):
        fused_attn_out_fp8_enabled(cfg)


@pytest.mark.parametrize(
    "rows,max_rows,layout,w_q_dtype,w_s_shape,expect_mode",
    [
        (256, 0, _LAYOUT, jnp.float8_e4m3fn, (_N,), "fused"),
        (256, 0, (0, 0, 0), jnp.float8_e4m3fn, (_N,), "regular_w8a8"),
        (16, 0, _LAYOUT, jnp.float8_e4m3fn, (_N,), "regular_w8a8"),  # no row tile
        (64, 64, _LAYOUT, jnp.float8_e4m3fn, (_N,), "regular_w8a16"),  # W8A16 bucket
        (256, 64, _LAYOUT, jnp.float8_e4m3fn, (_N,), "fused"),
        (256, 0, _LAYOUT, jnp.float8_e4m3fn, (1, 2, 1, _N), "gmm_v2"),  # blockwise
        (256, 0, _LAYOUT, jnp.int8, (_N,), "regular_w8a8"),  # int8 weights
    ],
)
def test_quantized_matmul_jax_fused_or_fallback(
    monkeypatch, rows, max_rows, layout, w_q_dtype, w_s_shape, expect_mode
):
    _mock_tpu_info(monkeypatch, _VMEM_64M)
    calls = []

    def fake_xla_mm(x, w_q, w_s, quantize_activation=True, quantize_activation_fn=None):
        calls.append(("xla", quantize_activation, quantize_activation_fn))
        return jnp.zeros((x.shape[0], w_q.shape[-1]), x.dtype)

    def fake_gmm_v2(lhs, rhs, **kwargs):
        calls.append(("gmm_v2",))
        return jnp.zeros((lhs.shape[0], rhs.shape[-1]), lhs.dtype)

    monkeypatch.setattr(linear_common, "xla_quantized_matmul", fake_xla_mm)
    monkeypatch.setattr(linear_common, "gmm_v2", fake_gmm_v2)

    out = linear_common._quantized_matmul_jax(
        jnp.ones((rows, _K), jnp.bfloat16),
        jnp.ones((_K, _N), w_q_dtype),
        jnp.ones(w_s_shape, jnp.float32),
        max_rows,
        *layout,
    )
    assert out.shape == (rows, _N)
    assert len(calls) == 1
    if expect_mode == "fused":
        tag, quant_act, fn = calls[0]
        assert tag == "xla" and quant_act is True
        assert isinstance(fn, functools.partial)
        assert fn.func is linear_common.quantize_attn_out
        assert fn.keywords == {
            "num_kv_heads": _LAYOUT[0],
            "q_per_kv": _LAYOUT[1],
            "head_dim": _LAYOUT[2],
        }
    elif expect_mode == "regular_w8a8":
        assert calls == [("xla", True, None)]
    elif expect_mode == "regular_w8a16":
        assert calls == [("xla", False, None)]
    else:
        assert calls == [("gmm_v2",)]


@pytest.mark.parametrize("keep_vllm_layout", [False, True])
def test_apply_routes_tagged_o_proj(monkeypatch, keep_vllm_layout):
    calls = []

    def fake_quantized_matmul(x, w_q, w_s, w8a16_max_rows=0, attn_out_head_layout=None):
        calls.append((w8a16_max_rows, attn_out_head_layout))
        return torch.zeros((x.shape[0], w_q.shape[-1]), dtype=x.dtype)

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
    fence_calls = 0

    def fake_fence(x):
        nonlocal fence_calls
        fence_calls += 1
        return x

    setattr(layer, ATTN_OUT_HEAD_LAYOUT_ATTR, _LAYOUT)
    setattr(layer, KEEP_VLLM_LAYOUT_ATTR, keep_vllm_layout)
    setattr(layer, fp8.ATTN_OUT_FENCE_OP_ATTR, fake_fence)

    out = method.apply(layer, torch.ones((64, _K), dtype=torch.bfloat16))
    assert out.shape == (64, _N)
    assert calls == [(16, None if keep_vllm_layout else _LAYOUT)]
    assert fence_calls == (1 if keep_vllm_layout else 0)


def test_quantized_matmul_with_head_layout_validation_and_3d_round_trip():
    with pytest.raises(ValueError, match="Input hidden dim"):
        quantized_matmul(
            torch.empty((8, _K + 16), dtype=torch.bfloat16, device="meta"),
            torch.empty((_K, _N), dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty((_N,), dtype=torch.float32, device="meta"),
            attn_out_head_layout=_LAYOUT,
        )

    with torch.device("meta"):
        x_3d = torch.empty((4, 16, _K), dtype=torch.bfloat16)
        w_q = torch.empty((_K, _N), dtype=torch.float8_e4m3fn)
        w_s = torch.empty((_N,), dtype=torch.float32)
        out_3d = quantized_matmul(
            x_3d, w_q, w_s, w8a16_max_rows=16, attn_out_head_layout=_LAYOUT
        )
    assert out_3d.shape == (4, 16, _N)
    assert out_3d.dtype == torch.bfloat16


def test_quantized_matmul_with_head_layout_fullgraph_compile():
    compiled = torch.compile(
        lambda x, w_q, w_s: quantized_matmul(
            x, w_q, w_s, w8a16_max_rows=16, attn_out_head_layout=_LAYOUT
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
    block_cls: type[torch.nn.Module] = Gemma4Attention,
    o_proj_cls=RowParallelLinear,
    impl: object | None = None,
    quant_method_cls=VllmFp8LinearMethodTPU,
    weight_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_shape: tuple[int, ...] = (_N,),
    keep_vllm_layout: bool = False,
) -> torch.nn.Module:
    block = block_cls.__new__(block_cls)
    torch.nn.Module.__init__(block)
    block.attn = MagicMock(spec=Attention)
    block.attn.impl = _make_batched_impl(head_size) if impl is None else impl
    block.attn.num_heads = num_heads
    block.attn.num_kv_heads = num_kv_heads
    block.attn.head_size = head_size
    o_proj = MagicMock(spec=o_proj_cls)
    o_proj.input_size_per_partition = o_proj_in
    o_proj.quant_method = MagicMock(spec=quant_method_cls)
    o_proj.weight = torch.empty((o_proj_in, _N), dtype=weight_dtype, device="meta")
    o_proj.weight_scale = torch.empty(scale_shape, dtype=torch.float32, device="meta")
    if keep_vllm_layout:
        setattr(o_proj, KEEP_VLLM_LAYOUT_ATTR, True)
    block.o_proj = o_proj
    return block


def test_tag_attn_out_projections_gated_and_allowlisted(monkeypatch):
    _mock_tpu_info(monkeypatch, _VMEM_64M)
    good_gemma = _block(16, 2, 512, 16 * 512, block_cls=Gemma4Attention)
    good_llama = _block(16, 2, 512, 16 * 512, block_cls=LlamaAttention)
    # Default off: returns 0 without tagging.
    assert tag_attn_out_projections(good_gemma, good_llama, enabled=False) == 0
    assert not hasattr(good_gemma.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)
    assert not hasattr(good_llama.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)

    # Gated and unknown attention classes are rejected by the allowlist even
    # when their head counts and FP8 linear shapes otherwise match.
    afmoe_gated = _block(16, 2, 512, 16 * 512, block_cls=AfmoeAttention)
    qwen3_next_gated = _block(16, 2, 512, 16 * 512, block_cls=Qwen3NextAttention)
    unknown_block = _block(16, 2, 512, 16 * 512, block_cls=torch.nn.Module)
    unquantized = _block(16, 2, 512, 16 * 512, quant_method_cls=UnquantizedLinearMethod)
    other_input = _block(16, 2, 512, 4096)
    uneven_heads = _block(15, 2, 512, 15 * 512)
    odd_q_per_kv = _block(6, 2, 512, 6 * 512)
    unaligned_head_dim = _block(16, 2, 80, 16 * 80)
    blockwise_scale = _block(16, 2, 512, 16 * 512, scale_shape=(1, 4, 1, _N))
    int8_weight = _block(16, 2, 512, 16 * 512, weight_dtype=torch.int8)
    kept_layout = _block(16, 2, 512, 16 * 512, keep_vllm_layout=True)
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
        good_gemma,
        good_llama,
        afmoe_gated,
        qwen3_next_gated,
        unknown_block,
        unquantized,
        other_input,
        uneven_heads,
        odd_q_per_kv,
        unaligned_head_dim,
        blockwise_scale,
        int8_weight,
        kept_layout,
        not_a_linear,
        default_rpa,
        pcp_streaming,
        dcp_layer,
        rebound_draft,
    ]
    model = torch.nn.ModuleList(blocks)
    assert tag_attn_out_projections(model, enabled=True) == 2
    assert getattr(good_gemma.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR) == (2, 8, 512)
    assert getattr(good_llama.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR) == (2, 8, 512)
    for block in blocks[2:]:
        assert not hasattr(block.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)

    # float32 activation derives q_packing=1, so odd q_per_kv=3 becomes eligible.
    assert (
        tag_attn_out_projections(odd_q_per_kv, enabled=True, act_dtype=jnp.float32) == 1
    )
    assert getattr(odd_q_per_kv.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR) == (2, 3, 512)

    # A chip whose VMEM cannot fit even a 32-row tile leaves layers untagged
    # and clears any prior tags.
    _mock_tpu_info(monkeypatch, 1024)
    assert tag_attn_out_projections(model, enabled=True) == 0
    assert not hasattr(good_gemma.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)
    assert not hasattr(good_llama.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)


def test_runner_initializes_attention_before_tagging_projections(monkeypatch):
    _mock_tpu_info(monkeypatch, _VMEM_64M)
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
    runner.vllm_config = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    runner.speculative_config = SimpleNamespace(draft_tensor_parallel_size=1)
    runner.model = target_block
    runner.drafter = SimpleNamespace(
        _draft_attn_layer_names={"draft.layers.0.self_attn.attn"},
        draft_model=draft_block,
    )

    # Switch off by default: kernels initialize, nothing is tagged.
    runner._initialize_attention_kernels()
    assert events == ["init_target", "init_draft"]
    assert not hasattr(target_block.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)
    assert not hasattr(draft_block.o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)

    # Switch on via additional_config with force=True: initialize_kernel and
    # tp=1 draft rebinding run before arm_runner_models, so target is tagged
    # while the rebound tp=1 draft is not.
    runner.vllm_config.additional_config = {FUSED_ATTN_OUT_FP8_KEY: True}
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


@pytest.mark.tpu_test
def test_tagged_o_proj_apply_matches_untagged_tpu(monkeypatch):
    """Run tagged vs untagged o_proj through VllmFp8LinearMethodTPU.apply on TPU."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")

    kv, qpk, d = 2, 4, 256
    m, k, n = 256, kv * qpk * d, 1024
    gen = torch.Generator().manual_seed(0)
    w_q = (torch.randn((k, n), generator=gen) * 0.05).to(torch.float8_e4m3fn).to("tpu")
    w_s = (torch.rand((n,), generator=gen) + 0.5).to(torch.float32).to("tpu") * 0.01
    x = (torch.randn((m, k), generator=gen) * 2.0).to(torch.bfloat16).to("tpu")

    def _make_layer(tagged: bool) -> torch.nn.Module:
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(w_q, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(w_s, requires_grad=False)
        if tagged:
            setattr(layer, ATTN_OUT_HEAD_LAYOUT_ATTR, (kv, qpk, d))
        return layer

    method = VllmFp8LinearMethodTPU(Fp8Config(is_checkpoint_fp8_serialized=True))
    tagged_layer = _make_layer(tagged=True)
    untagged_layer = _make_layer(tagged=False)

    quant_calls = []
    real_quantize = linear_common.quantize_attn_out

    def spy_quantize(*args, **kwargs):
        quant_calls.append(kwargs)
        return real_quantize(*args, **kwargs)

    monkeypatch.setattr(linear_common, "quantize_attn_out", spy_quantize)

    ref = method.apply(untagged_layer, x)
    assert quant_calls == []

    got = method.apply(tagged_layer, x)
    assert len(quant_calls) == 1
    assert quant_calls[0] == {"num_kv_heads": kv, "q_per_kv": qpk, "head_dim": d}
    assert got.shape == ref.shape == (m, n) and got.dtype == ref.dtype == torch.bfloat16
    assert torch.equal(got.cpu(), ref.cpu())

    # Fullgraph torch.compile of apply on the tagged layer also matches bit-for-bit.
    compiled_tagged = torch.compile(
        lambda inp: method.apply(tagged_layer, inp),
        backend="eager",
        fullgraph=True,
    )
    got_compiled = compiled_tagged(x)
    assert torch.equal(got_compiled.cpu(), ref.cpu())

# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-loading contract for GDN TP Q/K replication.

These CPU tests construct the real TPU GDN layer and call the loaders attached
to its parameters. Only distributed rank discovery and kernel registration are
stubbed. No dummy loader, replacement Linear, or proposed geometry helper is
used. Device computation belongs in the separate SPMD kernel tests.

Run: .venv/bin/python -m pytest \
    tests/layers/adapter/test_gdn_tp_head_replication.py -v
"""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor import parameter as parameter_mod
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM

from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op
from vllm_torchtpu.platforms.tpu_block_size_utils import \
    _hybrid_mamba_state_layout

pytestmark = pytest.mark.cpu_test

K_HEADS = 4
V_HEADS = 32
HEAD_DIM = 128
HIDDEN_SIZE = 128
CONV_KERNEL_SIZE = 4
TP_SIZES = [1, 2, 4, 8, 16]
QUANTIZATIONS = ["bf16", "fp8_tensor", "fp8_channel", "fp8_block"]


def _owned_heads(tp_size, tp_rank):
    """Reference: select V heads first, then the Q/K heads they depend on."""
    v_heads = torch.arange(V_HEADS).chunk(tp_size)[tp_rank]
    k_heads = (v_heads // (V_HEADS // K_HEADS)).unique_consecutive()
    return k_heads, v_heads


def _checkpoint_segments(tail_shape, dtype):
    # Exact FP8-representable values vary across heads, channels and input
    # columns, so copying the wrong half/head or transposing cannot pass.
    generator = torch.Generator().manual_seed(20260915)
    return [(torch.randint(
        -16, 17,
        (heads * HEAD_DIM, *tail_shape), generator=generator).float() /
             16).to(dtype) for heads in (K_HEADS, K_HEADS, V_HEADS, V_HEADS)]


def _select_heads(segments, tp_size, tp_rank, channels_per_head=HEAD_DIM):
    k_heads, v_heads = _owned_heads(tp_size, tp_rank)
    selected = []
    for segment, heads in zip(segments, (k_heads, k_heads, v_heads, v_heads)):
        by_head = segment.reshape(-1, channels_per_head, *segment.shape[1:])
        # CPU index_select does not support all float8 dtypes.
        selected.append(by_head.float()[heads].flatten(0, 1))
    return torch.cat(selected)


def _load_qkv_and_z(parameter, segments):
    # Qwen3.5's checkpoint mapper loads in_proj_qkv as (0, 1, 2), and
    # in_proj_z as 3, into the same merged projection parameter.
    parameter.weight_loader(parameter, torch.cat(segments[:3]), (0, 1, 2))
    parameter.weight_loader(parameter, segments[3], 3)


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_qkvz_checkpoint_load_preserves_whole_heads(make_gdn_attention,
                                                    tp_size, quantization):
    dtype = (torch.bfloat16 if quantization == "bf16" else torch.float8_e4m3fn)
    segments = _checkpoint_segments((HIDDEN_SIZE, ), dtype)
    for rank in range(tp_size):
        attention, vllm_config = make_gdn_attention(tp_size, rank,
                                                    quantization)
        assert vllm_config.device_config.device == torch.device("cpu")
        weight = attention.in_proj_qkvz.weight
        _load_qkv_and_z(weight, segments)
        expected = _select_heads(segments, tp_size, rank)
        torch.testing.assert_close(
            weight.float(),
            expected,
            rtol=0,
            atol=0,
            msg=lambda detail:
            f"QKVZ checkpoint partition: TP={tp_size}, rank={rank}\n{detail}")


@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_conv_checkpoint_uses_same_head_ownership(make_gdn_attention, tp_size):
    segments = _checkpoint_segments((1, CONV_KERNEL_SIZE), torch.bfloat16)[:3]
    for rank in range(tp_size):
        attention, _ = make_gdn_attention(tp_size, rank)
        weight = attention.conv1d.weight
        weight.weight_loader(weight, torch.cat(segments))
        expected = _select_heads(segments, tp_size, rank)
        torch.testing.assert_close(
            weight.float(),
            expected,
            rtol=0,
            atol=0,
            msg=lambda detail:
            f"Conv checkpoint partition: TP={tp_size}, rank={rank}\n{detail}")


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("quantization", QUANTIZATIONS[1:])
def test_fp8_scales_follow_projection_partition(make_gdn_attention, tp_size,
                                                quantization):
    for rank in range(tp_size):
        attention, _ = make_gdn_attention(tp_size, rank, quantization)
        scale = attention.in_proj_qkvz.weight_scale
        if quantization == "fp8_tensor":
            # QKV shares one checkpoint scale; Z has its own. All four
            # destination slots must be initialized even with replication.
            scale.weight_loader(scale, torch.tensor(0.5), (0, 1, 2))
            scale.weight_loader(scale, torch.tensor(2.0), 3)
            expected = torch.tensor([0.5, 0.5, 0.5, 2.0])
        else:
            channels_per_head = (1
                                 if quantization == "fp8_block" else HEAD_DIM)
            segments = [
                torch.arange(heads * channels_per_head, dtype=torch.float32) /
                1024 + segment + 1
                for segment, heads in enumerate((K_HEADS, K_HEADS, V_HEADS,
                                                 V_HEADS))
            ]
            if quantization == "fp8_block":
                segments = [segment[:, None] for segment in segments]
            _load_qkv_and_z(scale, segments)
            expected = _select_heads(segments, tp_size, rank,
                                     channels_per_head).flatten()
        torch.testing.assert_close(
            scale.float().flatten(),
            expected,
            rtol=0,
            atol=0,
            msg=lambda detail:
            f"FP8 scale partition: TP={tp_size}, rank={rank}\n{detail}")


@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_unified_pool_declares_complete_qk_conv_state(make_gdn_attention,
                                                      tp_size):
    attention, vllm_config = make_gdn_attention(tp_size, 0)
    k_heads, v_heads = _owned_heads(tp_size, 0)
    conv_dim = (2 * len(k_heads) + len(v_heads)) * HEAD_DIM
    taps = CONV_KERNEL_SIZE - 1
    spec = attention.get_kv_cache_spec(vllm_config)

    assert spec.shapes == ((taps, 1, conv_dim), (len(v_heads), HEAD_DIM,
                                                 HEAD_DIM))
    assert spec.dtypes == (torch.bfloat16, torch.float32)
    assert spec.page_size_bytes == (taps * conv_dim * 2 +
                                    len(v_heads) * HEAD_DIM * HEAD_DIM * 4)
    # The NHD pool used by the failing startup has 256 lanes per row.
    assert conv_dim % 256 == 0


@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_pool_budget_includes_replicated_qk_state(make_gdn_attention,
                                                  monkeypatch, tp_size):
    _, vllm_config = make_gdn_attention(tp_size, 0)
    # Exercise the real model's shape/dtype methods used before cache
    # allocation; replacing only registry lookup avoids config discovery.
    monkeypatch.setattr(ModelRegistry, "resolve_model_cls",
                        lambda *args, **kwargs: (Qwen3_5MoeForCausalLM, None))
    layout = _hybrid_mamba_state_layout(vllm_config,
                                        fa_physical_bytes_per_token=1024)
    k_heads, v_heads = _owned_heads(tp_size, 0)
    conv_dim = (2 * len(k_heads) + len(v_heads)) * HEAD_DIM
    assert layout.conv_bytes == (CONV_KERNEL_SIZE - 1) * conv_dim * 2
    assert layout.ssm_bytes == len(v_heads) * HEAD_DIM * HEAD_DIM * 4


@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_pooled_kernel_receives_tp_local_head_counts(make_gdn_attention,
                                                     monkeypatch, tp_size):
    attention, _ = make_gdn_attention(tp_size, 0)
    core = MagicMock()
    monkeypatch.setattr(gdn_attention_op, "gdn_attention_pooled_core_tpu",
                        core)
    registrations = gdn_attention_op.pallas.jax_op.call_args_list
    pooled_fn, = [
        call.args[1] for call in registrations
        if call.args[0].startswith("pallas::gdn_attention_pooled_")
    ]
    pooled_fn(*([None] * 12))
    k_heads, v_heads = _owned_heads(tp_size, 0)
    assert core.call_args.kwargs["n_kq"] == len(k_heads)
    assert core.call_args.kwargs["n_v"] == len(v_heads)
    assert attention.conv1d.weight.shape[0] == (2 * len(k_heads) +
                                                len(v_heads)) * HEAD_DIM


@pytest.mark.parametrize("tp_size", [4, 8, 16])
def test_fp8_post_load_uses_local_projection_widths(make_gdn_attention,
                                                    tp_size):
    attention, _ = make_gdn_attention(tp_size, tp_size - 1, "fp8_tensor")
    projection = attention.in_proj_qkvz
    segments = [
        torch.ones(heads * HEAD_DIM, HIDDEN_SIZE, dtype=torch.float8_e4m3fn)
        for heads in (K_HEADS, K_HEADS, V_HEADS, V_HEADS)
    ]
    _load_qkv_and_z(projection.weight, segments)
    scale = projection.weight_scale
    scale.weight_loader(scale, torch.tensor(0.5), (0, 1, 2))
    scale.weight_loader(scale, torch.tensor(2.0), 3)
    projection.quant_method.process_weights_after_loading(projection)

    # Runtime requantization must apply the shared QKV scale only to QKV,
    # keeping Z's separate scale and the [out, in] layout used by GDN.
    k_heads, v_heads = _owned_heads(tp_size, tp_size - 1)
    qkv_width = (2 * len(k_heads) + len(v_heads)) * HEAD_DIM
    z_width = len(v_heads) * HEAD_DIM
    dequantized = projection.weight.float() * projection.weight_scale[:, None]
    expected = torch.cat((torch.full(
        (qkv_width, HIDDEN_SIZE), 0.5), torch.full((z_width, HIDDEN_SIZE),
                                                   2.0)))
    torch.testing.assert_close(dequantized, expected)


@pytest.mark.parametrize("tp_size", [4, 8, 16])
@pytest.mark.parametrize("parameter_name", ["weight_scale", "input_scale"])
def test_static_fp8_scales_load_by_logical_segment(make_gdn_attention, tp_size,
                                                   parameter_name):
    # ModelOpt mixed FP8 uses static activation scales. Both kinds of
    # per-tensor scale must still initialize Q/K/V and Z independently.
    attention, _ = make_gdn_attention(tp_size,
                                      tp_size - 1,
                                      "fp8_tensor",
                                      activation_scheme="static")
    param = getattr(attention.in_proj_qkvz, parameter_name)
    param.weight_loader(param, torch.tensor(0.5), (0, 1, 2))
    param.weight_loader(param, torch.tensor(2.0), 3)
    torch.testing.assert_close(param.data, torch.tensor([0.5, 0.5, 0.5, 2.0]))


def test_loader_does_not_treat_unknown_parameters_as_scales(
        make_gdn_attention):
    attention, _ = make_gdn_attention(8, 0)
    param = parameter_mod.BasevLLMParameter(data=torch.zeros(4),
                                            weight_loader=lambda *args: None)
    with pytest.raises(AttributeError, match="output_dim"):
        attention.in_proj_qkvz.weight_loader(param, torch.ones(4))
    torch.testing.assert_close(param.data, torch.zeros(4))


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_interleaved_replication_projection_and_conv(make_gdn_attention,
                                                     tp_size, quantization):
    dtype = torch.bfloat16 if quantization == "bf16" else torch.float8_e4m3fn
    segments = _checkpoint_segments((HIDDEN_SIZE, ), dtype)
    checkpoint = torch.cat(
        [s.reshape(K_HEADS, -1, HIDDEN_SIZE) for s in segments],
        dim=1).flatten(0, 1)
    ba_segments = [
        torch.arange(V_HEADS * HIDDEN_SIZE).reshape(
            V_HEADS, HIDDEN_SIZE).float() + i for i in (0, 10000)
    ]
    ba_checkpoint = torch.cat(
        [s.reshape(K_HEADS, -1, HIDDEN_SIZE) for s in ba_segments],
        dim=1).flatten(0, 1).to(torch.bfloat16)
    conv_segments = _checkpoint_segments((1, CONV_KERNEL_SIZE),
                                         torch.bfloat16)[:3]
    for rank in range(tp_size):
        attention, _ = make_gdn_attention(tp_size,
                                          rank,
                                          quantization,
                                          gqa_interleaved_layout=True)
        weight = attention.in_proj_qkvz.weight
        weight.weight_loader(weight, checkpoint)
        kh, vh = _owned_heads(tp_size, rank)
        expected = _select_heads(segments, tp_size, rank)
        widths = [len(kh) * HEAD_DIM] * 2 + [len(vh) * HEAD_DIM] * 2
        torch.testing.assert_close(weight.float(), expected, rtol=0, atol=0)
        conv = attention.conv1d.weight
        conv.weight_loader(conv, torch.cat(conv_segments))
        torch.testing.assert_close(conv.float(),
                                   _select_heads(conv_segments, tp_size, rank))

        # B/A checkpoint groups must split on V heads, not at the midpoint
        # of the combined group. Use the BF16 layer for projection arithmetic.
        if quantization == "bf16":
            ba_weight = attention.in_proj_ba.weight
            ba_weight.weight_loader(ba_weight, ba_checkpoint)
            x = torch.ones(2, HIDDEN_SIZE)
            projected = x @ weight.float().T
            ba = x @ ba_weight.float().T
            q, k, v, z = projected.split(widths, dim=-1)
            b, a = ba.chunk(2, dim=-1)
            for actual, segment in zip((q, k, v, z), expected.split(widths)):
                torch.testing.assert_close(actual.flatten(1), x @ segment.T)
            for actual, segment in zip((b, a), ba_segments):
                torch.testing.assert_close(
                    actual, x @ segment.to(torch.bfloat16).float()[vh].T)


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("quantization", QUANTIZATIONS[1:])
def test_interleaved_scale_partition(make_gdn_attention, tp_size,
                                     quantization):
    for rank in range(tp_size):
        attention, _ = make_gdn_attention(tp_size,
                                          rank,
                                          quantization,
                                          gqa_interleaved_layout=True)
        scale = attention.in_proj_qkvz.weight_scale
        if quantization == "fp8_tensor":
            scale.weight_loader(scale, torch.tensor(0.5))
            torch.testing.assert_close(scale.data, torch.full_like(scale, 0.5))
            continue
        channels = 1 if quantization == "fp8_block" else HEAD_DIM
        segments = [
            (torch.arange(heads * channels).float() + i * 1000).reshape(-1, 1)
            for i, heads in enumerate((K_HEADS, K_HEADS, V_HEADS, V_HEADS))
        ]
        checkpoint = torch.cat([s.reshape(K_HEADS, -1, 1) for s in segments],
                               dim=1).flatten(0, 1)
        if quantization == "fp8_channel":
            checkpoint = checkpoint.flatten()
        scale.weight_loader(scale, checkpoint)
        expected = _select_heads(segments, tp_size, rank, channels).flatten()
        torch.testing.assert_close(scale.float().flatten(), expected)

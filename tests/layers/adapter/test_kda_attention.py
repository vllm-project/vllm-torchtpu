# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The shared KDA layer, with value heads wider than key heads.

Kimi K3's own tests (tests/models/vllm/test_kimi_k3.py) cover the equal-head
configuration; these use an OLMo 3.5-like one: value heads twice the key head
size, beta scaled by 2 and a biased output gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_torchtpu.layers.adapter import kda_attention
from vllm_torchtpu.layers.adapter.kda_attention import KdaAttention

pytestmark = pytest.mark.usefixtures("cpu_model_tests")

HIDDEN, HEADS, HEAD_K, HEAD_V, CONV = 16, 2, 4, 8, 3


def test_kda_state_dtype_reads_model_and_cache_config() -> None:
    """`kda_state_dtype` returns the storage dtypes for a KDA layer's two
    state tensors. The conv state is always fp32 (the fused conv1d kernel
    needs it), regardless of the model dtype or `mamba_cache_dtype`; the
    recurrent state follows vLLM's usual KDA rule."""
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
    )
    conv_dtype, recurrent_dtype = kda_attention.kda_state_dtype(vllm_config)
    assert conv_dtype == torch.float32
    assert recurrent_dtype == torch.float32


def test_load_a_log_flattens_legacy_layout_and_shards_by_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`load_a_log` is the weight loader for the KDA `A_log` parameter.
    Older checkpoints store it as `[1, 1, H, 1]` and newer ones as `[H]`;
    either way each TP rank keeps only its own slice of the H heads. With
    H=4 and a 2-head local parameter, rank 1 gets heads 2 and 3."""
    monkeypatch.setattr(kda_attention, "get_tensor_model_parallel_rank", lambda: 1)
    param = torch.empty(2)
    loaded = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1)

    kda_attention.load_a_log(param, loaded)

    torch.testing.assert_close(param, torch.tensor([2.0, 3.0]))


def test_kda_state_shape_sizes_conv_channels_by_both_head_sizes() -> None:
    """The conv state holds q | k | v channels, so a value head wider than the
    key head widens it; the recurrent state is [heads, key, value]."""
    assert kda_attention.kda_state_shape(
        num_heads=HEADS, head_k_dim=HEAD_K, head_v_dim=HEAD_V, conv_kernel_size=CONV
    ) == ((CONV - 1, 1, HEADS * (2 * HEAD_K + HEAD_V)), (HEADS, HEAD_K, HEAD_V))


class _RecordingLinear(nn.Module):
    """Stands in for vLLM's TP linear layers, which need a distributed group."""

    def __init__(self, input_size, output_size, *args, bias=False, **kwargs) -> None:
        super().__init__()
        del args, kwargs
        self.input_size = input_size
        self.output_sizes = output_size
        self.has_bias = bias


@pytest.fixture
def build_kda(monkeypatch: pytest.MonkeyPatch):
    """Build the real `KdaAttention` with stand-in linears and ops.

    Returns `build(unified_pool=False, **layer_kwargs) -> (layer, context,
    dispatched_op_kwargs)`.
    """
    for name in (
        "ColumnParallelLinear",
        "MergedColumnParallelLinear",
        "ReplicatedLinear",
        "RowParallelLinear",
    ):
        monkeypatch.setattr(kda_attention, name, _RecordingLinear)
    monkeypatch.setattr(
        kda_attention, "get_tensor_model_parallel_world_size", lambda: 1
    )
    op_kwargs = {}

    def build_dispatched(prefix, **kwargs):
        op_kwargs.update(kwargs)
        return lambda *args: None

    def build_pooled(*args, **kwargs):
        raise AssertionError("the pooled op does not support this configuration")

    monkeypatch.setattr(kda_attention, "build_kimi_dispatched_kda_op", build_dispatched)
    monkeypatch.setattr(kda_attention, "build_kimi_pooled_kda_op", build_pooled)

    def build(unified_pool: bool = False, **layer_kwargs):
        monkeypatch.setattr(
            kda_attention,
            "unified_kv_layout_enabled_for_architecture",
            lambda architecture: unified_pool,
        )
        context = {}
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(architecture="Olmo35ForCausalLM"),
            speculative_config=None,
            quant_config=None,
            compilation_config=SimpleNamespace(static_forward_context=context),
        )
        kwargs = dict(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            head_k_dim=HEAD_K,
            head_v_dim=HEAD_V,
            conv_kernel_size=CONV,
            gate_rank=HEAD_V,
            norm_eps=1e-6,
            gate_bias=True,
            beta_scale=2.0,
        )
        kwargs.update(layer_kwargs)
        # RMSNorm reads the current config; pinned to CPU, where the default
        # device detection can fail.
        with set_current_vllm_config(
            VllmConfig(device_config=DeviceConfig(device="cpu"))
        ):
            layer = KdaAttention(vllm_config, "model.layers.0.mixer", **kwargs)
        return layer, context, op_kwargs

    return build


@pytest.mark.parametrize("use_full_rank_gate", [False, True])
def test_kda_construction_with_wider_value_heads(build_kda, use_full_rank_gate) -> None:
    """Every projection, parameter and state is sized from the head size it
    belongs to: decay (f) and dt_bias by the key head, the output gate, norm
    and o_proj by the value head, the down-projections by the gate rank. The
    output gate, low-rank or full-rank, gets the bias. The beta scale reaches
    the dispatched op, and the pooled op, which supports neither, is not
    built."""
    layer, context, op_kwargs = build_kda(use_full_rank_gate=use_full_rank_gate)

    key, value = HEADS * HEAD_K, HEADS * HEAD_V
    assert layer.fused_qkvb_proj.output_sizes == [key, key, value, HEADS]
    assert (layer.f_b_proj.input_size, layer.f_b_proj.output_sizes) == (HEAD_V, key)
    assert not layer.f_b_proj.has_bias
    if use_full_rank_gate:
        assert layer.fused_fa_ga_proj is None and layer.g_b_proj is None
        assert (layer.f_a_proj.input_size, layer.f_a_proj.output_sizes) == (
            HIDDEN,
            HEAD_V,
        )
        assert not layer.f_a_proj.has_bias
        assert (layer.g_proj.input_size, layer.g_proj.output_sizes) == (HIDDEN, value)
        assert layer.g_proj.has_bias
    else:
        assert layer.f_a_proj is None and layer.g_proj is None
        assert layer.fused_fa_ga_proj.output_sizes == [HEAD_V, HEAD_V]
        assert (layer.g_b_proj.input_size, layer.g_b_proj.output_sizes) == (
            HEAD_V,
            value,
        )
        assert layer.g_b_proj.has_bias
    assert layer.o_proj.input_size == value
    assert layer.q_conv1d.weight.shape == (key, 1, CONV)
    assert layer.v_conv1d.weight.shape == (value, 1, CONV)
    assert layer.A_log.shape == (HEADS,)
    assert layer.dt_bias.shape == (key,)
    assert layer.o_norm.weight.shape == (HEAD_V,)
    assert layer.conv_weight_fused.shape == (CONV, 2 * key + value)
    assert layer.get_state_shape() == (
        (CONV - 1, 1, 2 * key + value),
        (HEADS, HEAD_K, HEAD_V),
    )
    assert op_kwargs["beta_scale"] == 2.0
    assert layer.pooled_kda_op is None
    assert context["model.layers.0.mixer"] is layer


@pytest.mark.parametrize(("head_v_dim", "beta_scale"), [(HEAD_V, 1.0), (HEAD_K, 2.0)])
def test_unified_pool_is_rejected_before_allocation(
    build_kda, head_v_dim, beta_scale
) -> None:
    """A configuration the pooled op cannot run fails at construction with
    the fix, before any weight, op or forward-context entry exists."""
    with pytest.raises(NotImplementedError, match="UNIFIED_BLOCK_POOL=0"):
        build_kda(unified_pool=True, head_v_dim=head_v_dim, beta_scale=beta_scale)


def _bare_layer(head_k_dim: int, head_v_dim: int) -> KdaAttention:
    layer = KdaAttention.__new__(KdaAttention)
    nn.Module.__init__(layer)
    layer.conv_size, layer.num_heads = CONV, HEADS
    layer.head_k_dim, layer.head_v_dim, layer.gate_rank = head_k_dim, head_v_dim, 2
    layer.local_key_size = HEADS * head_k_dim
    layer.local_value_size = HEADS * head_v_dim
    return layer


def _conv_weights(layer: KdaAttention) -> None:
    for i, (name, channels) in enumerate(
        (
            ("q_conv1d", layer.local_key_size),
            ("k_conv1d", layer.local_key_size),
            ("v_conv1d", layer.local_value_size),
        )
    ):
        conv = nn.Module()
        conv.weight = nn.Parameter(
            torch.arange(channels * CONV, dtype=torch.float32).view(channels, 1, CONV)
            + 1000 * i
        )
        setattr(layer, name, conv)


def test_conv_fusion_keeps_the_kimi_layout_for_equal_head_sizes() -> None:
    """Equal head sizes keep the `[taps, 3, heads, head_dim]` layout the AOT
    exporter shards on its head axis, with the same values as before."""
    layer = _bare_layer(HEAD_K, HEAD_K)
    _conv_weights(layer)

    layer.process_weights_after_loading(act_dtype=torch.bfloat16)

    expected = torch.stack(
        [
            w[:, 0, :].t().reshape(CONV, HEADS, HEAD_K)
            for w in (
                layer.q_conv1d.weight,
                layer.k_conv1d.weight,
                layer.v_conv1d.weight,
            )
        ],
        dim=1,
    )
    torch.testing.assert_close(layer.conv_weight_fused, expected, rtol=0, atol=0)


def test_conv_fusion_concatenates_qkv_channels_for_wider_value_heads() -> None:
    """With wider value heads the fused weight is `[taps, q | k | v]`, the
    flat channel order of `mixed_qkv` and the conv state."""
    layer = _bare_layer(HEAD_K, HEAD_V)
    _conv_weights(layer)

    layer.process_weights_after_loading(act_dtype=torch.bfloat16)

    expected = torch.cat(
        [
            w[:, 0, :].t()
            for w in (
                layer.q_conv1d.weight,
                layer.k_conv1d.weight,
                layer.v_conv1d.weight,
            )
        ],
        dim=1,
    )
    assert layer.conv_weight_fused.shape == (CONV, HEADS * (2 * HEAD_K + HEAD_V))
    torch.testing.assert_close(layer.conv_weight_fused, expected, rtol=0, atol=0)


class _ArangeProjection(nn.Module):
    def __init__(self, output_size: int, offset: float = 0.0) -> None:
        super().__init__()
        self.output_size = output_size
        self.offset = offset

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        values = torch.arange(inputs.shape[0] * self.output_size, dtype=inputs.dtype)
        return values.view(inputs.shape[0], self.output_size) + self.offset, None


class _IdentityProjection(nn.Module):
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        return inputs, None


def test_forward_splits_projections_by_key_and_value_head_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`forward` hands the op q | k | v (key, key and value widths), a
    key-wide decay gate, per-head beta and a value-wide output gate, and
    feeds the op's value-wide output to o_proj."""
    layer = _bare_layer(HEAD_K, HEAD_V)
    key, value = HEADS * HEAD_K, HEADS * HEAD_V
    layer.use_full_rank_gate = False
    layer.fused_qkvb_proj = _ArangeProjection(2 * key + value + HEADS)
    layer.fused_fa_ga_proj = _ArangeProjection(4)
    layer.f_b_proj = _ArangeProjection(key, offset=100.0)
    layer.g_b_proj = _ArangeProjection(value, offset=200.0)
    layer.o_proj = _IdentityProjection()
    layer.conv_weight_fused = torch.zeros(CONV, 2 * key + value)
    layer.A_log = nn.Parameter(torch.zeros(HEADS))
    layer.dt_bias = nn.Parameter(torch.zeros(key))
    layer.o_norm = SimpleNamespace(weight=torch.ones(HEAD_V))
    layer.kv_cache = (torch.zeros(1, CONV - 1, 1, 2 * key + value), torch.zeros(1))
    layer._metadata = lambda: SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        mamba_request_distribution=None,
        mamba_slot_read_offsets=None,
    )
    monkeypatch.setattr(
        kda_attention, "get_tp_group", lambda: SimpleNamespace(all_reduce=lambda x: x)
    )
    received = {}

    def dispatched_kda_op(mixed_qkv, raw_gate, beta, output_gate, *args):
        received.update(
            mixed_qkv=mixed_qkv, raw_gate=raw_gate, beta=beta, output_gate=output_gate
        )
        return output_gate.view(-1, HEADS, HEAD_V)

    layer.dispatched_kda_op = dispatched_kda_op

    output = layer(torch.arange(2), torch.ones(2, HIDDEN))

    projected, _ = layer.fused_qkvb_proj(torch.ones(2, HIDDEN))
    torch.testing.assert_close(received["mixed_qkv"], projected[:, : 2 * key + value])
    torch.testing.assert_close(received["beta"], projected[:, 2 * key + value :])
    assert received["raw_gate"].shape == (2, key)
    assert received["output_gate"].shape == (2, value)
    torch.testing.assert_close(output, received["output_gate"])


def test_forward_returns_value_wide_zeros_without_state() -> None:
    """Before vLLM binds the state (profiling), the output is value-wide."""
    layer = _bare_layer(HEAD_K, HEAD_V)
    layer._metadata = lambda: None
    layer.kv_cache = None
    output = layer._core_attention(
        torch.zeros(2, HEADS * (2 * HEAD_K + HEAD_V)),
        torch.zeros(2, HEADS * HEAD_K),
        torch.zeros(2, HEADS),
        torch.ones(2, HEADS * HEAD_V),
    )
    torch.testing.assert_close(output, torch.zeros(2, HEADS * HEAD_V))

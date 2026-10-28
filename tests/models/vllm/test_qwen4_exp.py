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
"""Tests for the TPU Qwen4Exp (Qwen3.8 Flash Next) text model."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen4ExpTextConfig
from vllm.config import CompilationMode
from vllm.model_executor import parameter as vllm_parameter
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers import vocab_parallel_embedding as vpe
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
)

import vllm_torchtpu.models.vllm.qwen4_exp.model as qwen4_model
from vllm_torchtpu.gdn_pool_layout import QWEN_GDN_ARCHITECTURES
from vllm_torchtpu.layers import register_layers
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.models.vllm.qwen4_exp.config import TPUQwen4ExpConfig
from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpForCausalLM
from vllm_torchtpu.models.vllm.qwen4_exp.ple import (
    PLEInputs,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
    head_vocab_sizes,
    layer_multipliers,
    ngram_ids,
    token_requests,
)

HIDDEN = 16
HC = 4
VOCAB = 32
# The checkpoint's spelling; transformers relabels full_attention as a QSA
# layer type on load. PLE may only sit on linear_attention layers.
LAYER_TYPES = ["linear_attention", "full_attention"]


@pytest.fixture(autouse=True)
def _cpu_model_tests(monkeypatch):
    # Same guard as the Kimi model tests: keep Dynamo from querying a TPU
    # device while these CPU-only tests run.
    tpu = getattr(torch, "tpu", None)
    if tpu is not None:
        monkeypatch.setattr(tpu, "is_available", lambda: False)
        monkeypatch.setattr(tpu, "current_device", lambda: 0)
        monkeypatch.setattr(tpu, "device_count", lambda: 0)
        monkeypatch.setattr(tpu, "manual_seed_all", lambda seed: None)
    if hasattr(torch, "accelerator"):
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
        monkeypatch.setattr(torch.accelerator, "device_count", lambda: 0)


def _text_config(**overrides) -> Qwen4ExpTextConfig:
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        hc_count=HC,
        hc_lowrank=8,
        num_hidden_layers=len(LAYER_TYPES),
        layer_types=LAYER_TYPES,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=3,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    kwargs.update(overrides)
    return Qwen4ExpTextConfig(**kwargs)


# ---------------------------------------------------------------------------
# Registration and config checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "arch", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_qwen4_exp_is_registered(arch: str) -> None:
    previous_model = ModelRegistry.models.get(arch)
    previous_config = MODELS_CONFIG_MAP.get(arch)
    try:
        register_layers()
        assert ModelRegistry.models[arch].load_model_cls() is Qwen4ExpForCausalLM
        assert MODELS_CONFIG_MAP[arch] is TPUQwen4ExpConfig
        assert arch in QWEN_GDN_ARCHITECTURES
    finally:
        if previous_model is None:
            ModelRegistry.models.pop(arch, None)
        else:
            ModelRegistry.models[arch] = previous_model
        if previous_config is None:
            MODELS_CONFIG_MAP.pop(arch, None)
        else:
            MODELS_CONFIG_MAP[arch] = previous_config


def _vllm_config_for_checks(
    text_config: Qwen4ExpTextConfig,
    *,
    max_model_len: int = 2048,
    speculative_config: object | None = None,
    pipeline_parallel_size: int = 1,
    use_sequence_parallel_moe: bool = False,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=text_config,
            hf_text_config=text_config,
            multimodal_config=None,
            max_model_len=max_model_len,
        ),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            ubatch_size=1,
            pipeline_parallel_size=pipeline_parallel_size,
            use_sequence_parallel_moe=use_sequence_parallel_moe,
            prefill_context_parallel_size=prefill_context_parallel_size,
            decode_context_parallel_size=decode_context_parallel_size,
        ),
        speculative_config=speculative_config,
    )


def test_config_accepts_supported_text_model_and_strips_mrope() -> None:
    text_config = _text_config(
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        }
    )
    TPUQwen4ExpConfig.verify_and_update_config(_vllm_config_for_checks(text_config))
    assert "mrope_section" not in text_config.rope_parameters
    assert "mrope_interleaved" not in text_config.rope_parameters


_QSA = dict(
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=8,
    indexer_budget=2048,
    indexer_compress_ratio=4,
)


def test_config_accepts_qsa_within_indexer_budget() -> None:
    vllm_config = _vllm_config_for_checks(_text_config(**_QSA), max_model_len=2048)
    TPUQwen4ExpConfig.verify_and_update_config(vllm_config)


def test_config_rejects_unknown_layer_type() -> None:
    text_config = _text_config()
    text_config.layer_types = ["linear_attention", "sliding_attention"]
    with pytest.raises(NotImplementedError, match="sliding_attention"):
        TPUQwen4ExpConfig.verify_and_update_config(_vllm_config_for_checks(text_config))


@pytest.mark.parametrize(
    "text_overrides,config_overrides,match",
    [
        ({"ple_layer_ids": [1], "eos_token_id": 3}, {}, "PLE"),
        (_QSA, {"max_model_len": 4096}, "indexer_budget=2048"),
        (
            {},
            {"speculative_config": SimpleNamespace(method="mtp")},
            "speculative decoding",
        ),
        ({}, {"pipeline_parallel_size": 2}, "pipeline_parallel_size=1"),
        ({}, {"use_sequence_parallel_moe": True}, "sequence-parallel MoE"),
        ({}, {"prefill_context_parallel_size": 2}, "context parallelism"),
        ({}, {"decode_context_parallel_size": 2}, "context parallelism"),
    ],
    ids=["ple", "long_context_qsa", "mtp", "pp", "sp_moe", "pcp", "dcp"],
)
def test_config_rejects_unsupported_features(
    text_overrides, config_overrides, match
) -> None:
    vllm_config = _vllm_config_for_checks(
        _text_config(**text_overrides), **config_overrides
    )
    with pytest.raises(NotImplementedError, match=match):
        TPUQwen4ExpConfig.verify_and_update_config(vllm_config)


@pytest.mark.parametrize(
    "nested,override",
    [
        (False, '{"ple_layer_ids": []}'),
        (True, '{"text_config": {"ple_layer_ids": []}}'),
    ],
    ids=["text_only", "vision_language"],
)
def test_config_ple_error_names_override(nested, override) -> None:
    text_config = _text_config(ple_layer_ids=[1], eos_token_id=3)
    vllm_config = _vllm_config_for_checks(text_config)
    if nested:
        vllm_config.model_config.hf_config = SimpleNamespace(text_config=text_config)
    with pytest.raises(NotImplementedError) as excinfo:
        TPUQwen4ExpConfig.verify_and_update_config(vllm_config)
    assert f"--hf-overrides '{override}'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Decoder wiring with stand-in sublayers
# ---------------------------------------------------------------------------
class _FakeSublayer(nn.Module):
    """A deterministic hidden_size -> hidden_size stand-in that records calls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(self, *args, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        hidden_states = (
            kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        )
        return torch.tanh(self.proj(hidden_states))


class _FakeEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, prefix: str = ""):
        super().__init__(num_embeddings, embedding_dim)


class _FakeLMHead(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, prefix: str = ""):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))


@pytest.fixture
def fake_sublayers(monkeypatch):
    for name in (
        "QwenGatedDeltaNetAttention",
        "Qwen3NextAttention",
        "Qwen3NextSparseMoeBlock",
    ):
        monkeypatch.setattr(qwen4_model, name, _FakeSublayer)
    monkeypatch.setattr(qwen4_model, "VocabParallelEmbedding", _FakeEmbedding)
    monkeypatch.setattr(qwen4_model, "ParallelLMHead", _FakeLMHead)
    monkeypatch.setattr(qwen4_model, "LogitsProcessor", lambda vocab_size: None)


def _model_vllm_config(text_config: Qwen4ExpTextConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text_config),
        cache_config=None,
        quant_config=None,
        additional_config={},
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


def _hc_connections(model):
    return [
        *(
            hc
            for layer in model.layers
            for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection)
        ),
        model.hyper_connection_mixer,
    ]


def _hc_constructor_config(options, *, calibrated=False):
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

    # Real pinned additional_config/compilation interfaces; lightweight text
    # metadata and unrelated sublayers avoid loading a checkpoint/accelerator.
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        additional_config={"qwen4_exp": options},
    )
    text = _text_config(
        hidden_size=2560 if calibrated else HIDDEN,
        hc_lowrank=320 if calibrated else 8,
    )
    config.model_config = SimpleNamespace(hf_text_config=text)
    return config


@pytest.mark.parametrize(
    "value,expected",
    [(None, "auto"), ("auto", "auto"), ("0", 0), ("128", 128), ("256", 256)],
)
def test_model_hc_reads_typed_debug_override(
    fake_sublayers, monkeypatch, value, expected
):
    from vllm.utils.torch_utils import set_default_torch_dtype

    from tests.models.vllm.test_qwen4_exp_hc_auto import (
        MEASURED,
        combine_selected,
        mix_selected,
        route_inputs,
    )
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter
    from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import GatedResidual

    if value is None:
        monkeypatch.delenv("VLLM_TPU_HC_PREFILL_TILE", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_HC_PREFILL_TILE", value)
    TPUQwen4ExpConfig.verify_and_update_config(_vllm_config_for_checks(_text_config()))

    def forbidden(*args):
        pytest.fail("metadata-only model construction executed a fused operation")

    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: MEASURED)
    monkeypatch.setattr(vllm_adapter, "build_full_mix_op", lambda *a: forbidden)
    monkeypatch.setattr(vllm_adapter, "build_gated_residual_op", lambda *a: forbidden)
    with torch.device("meta"), set_default_torch_dtype(torch.bfloat16):
        model = qwen4_model.Qwen4ExpModel(
            vllm_config=_hc_constructor_config({}, calibrated=True), prefix="model"
        )
    connections = _hc_connections(model)
    x, block = route_inputs(1536)
    for i, connection in enumerate(connections):
        assert type(connection) is GatedResidual
        assert connection.prefill_full_tile == expected
        assert connection.prefill_combine_tile == expected
        assert mix_selected(connection, x) is bool(expected)
        if i < len(connections) - 1:
            assert combine_selected(connection, block, x, x) is bool(expected)
        else:
            assert connection._prefill_combine_op is None
            assert not connection.use_combine


@pytest.mark.parametrize(
    "value", ["true", "false", "-128", "127", "129", "128.0", "atuo"]
)
def test_config_rejects_invalid_hc_debug_override(monkeypatch, value):
    monkeypatch.setenv("VLLM_TPU_HC_PREFILL_TILE", value)
    with pytest.raises(ValueError):
        TPUQwen4ExpConfig.verify_and_update_config(
            _vllm_config_for_checks(_text_config())
        )


@pytest.mark.parametrize("value", [True, False, -128, 127])
def test_config_validates_typed_hc_debug_override(monkeypatch, value):
    from vllm_torchtpu import envs

    def invalid_tile():
        return value

    monkeypatch.setitem(
        envs.environment_variables, "VLLM_TPU_HC_PREFILL_TILE", invalid_tile
    )
    with pytest.raises(ValueError, match="HC prefill"):
        TPUQwen4ExpConfig.verify_and_update_config(
            _vllm_config_for_checks(_text_config())
        )


def test_hc_cpu_fullgraph_has_no_policy_side_effects(monkeypatch):
    from tests.models.vllm.test_qwen4_exp_hc_auto import MEASURED
    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter
    from vllm_torchtpu.models.vllm.qwen4_exp import hyperconnection as hc

    def forbidden(*args, **kwargs):
        pytest.fail("CPU forward reached fused HC or logged during tracing")

    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: MEASURED)
    monkeypatch.setattr(vllm_adapter, "build_full_mix_op", lambda *a: forbidden)
    monkeypatch.setattr(vllm_adapter, "build_gated_residual_op", lambda *a: forbidden)
    with torch.device("cpu"):
        module = hc.GatedResidual(
            hc.HyperConnectionConfig(
                hidden_size=128, hc_count=3, hc_lowrank=8, hc_per_branch_norm=True
            ),
            prefill_full_tile=128,
        )
        x = torch.randn(128, 384).bfloat16()
        block = torch.randn(128, 128).bfloat16()
    monkeypatch.setattr(hc.logger, "warning_once", forbidden)
    monkeypatch.setitem(
        envs.environment_variables, "VLLM_TPU_HC_PREFILL_TILE", forbidden
    )

    def chain(x, block):
        mixed, residuals = module.mix(x)
        return mixed, module.combine(block, residuals)

    expected = chain(x, block)
    actual = torch.compile(chain, backend="eager", fullgraph=True)(x, block)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert not torch.tpu.is_initialized()
    torch.compiler.reset()


def test_model_hc_keeps_checkpoint_keys_and_single_registered_owners(
    fake_sublayers, monkeypatch
):
    from vllm.models.qwen4_exp.common import hyperconnection as upstream

    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter
    from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import GatedResidual

    def forbidden():
        pytest.fail("CPU model construction queried/initialized TPU hardware")

    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(is_initialized=lambda: False), raising=False
    )
    monkeypatch.setattr(vllm_adapter.pltpu, "get_tpu_info", forbidden)
    with torch.device("cpu"):
        model = qwen4_model.Qwen4ExpModel(
            vllm_config=_hc_constructor_config({}), prefix="model"
        )
    connections = _hc_connections(model)
    for i, connection in enumerate(connections):
        assert type(connection) is GatedResidual
        assert not connection._prefill_full_ops
        assert connection._prefill_combine_op is None
        reference = upstream.GatedResidual(
            connection.config, use_combine=i < len(connections) - 1
        )
        assert connection.state_dict().keys() == reference.state_dict().keys()
        for name, weight in connection.state_dict().items():
            assert (weight.shape, weight.dtype) == (
                reference.state_dict()[name].shape,
                reference.state_dict()[name].dtype,
            )
        owners = {
            n: (id(p), p.untyped_storage()._cdata)
            for n, p in connection.named_parameters()
        }
        checkpoint = {
            n: torch.full_like(p, 0.125) for n, p in reference.state_dict().items()
        }
        connection.load_state_dict(checkpoint, strict=True)
        assert owners == {
            n: (id(p), p.untyped_storage()._cdata)
            for n, p in connection.named_parameters()
        }
        for name, weight in connection.state_dict().items():
            torch.testing.assert_close(weight, checkpoint[name], rtol=0, atol=0)
    assert len({id(c) for c in connections}) == len(connections)
    parameters = list(model.named_parameters(remove_duplicate=False))
    assert len(parameters) == len(model.state_dict())
    assert len({id(p) for _, p in parameters}) == len(parameters)
    assert len({p.untyped_storage()._cdata for _, p in parameters}) == len(parameters)


def test_model_hc_hash_only_additional_config_keeps_defaults(fake_sublayers):
    class HashOnlyOptions:
        def compute_hash(self):
            return "test-hash-only-options"

    config = _hc_constructor_config({})
    config.additional_config = HashOnlyOptions()
    TPUQwen4ExpConfig.verify_and_update_config(
        _vllm_config_for_checks(
            _text_config(), additional_config=config.additional_config
        )
    )
    model = qwen4_model.Qwen4ExpModel(vllm_config=config, prefix="model")
    for connection in _hc_connections(model):
        assert connection.prefill_full_tile == "auto"
        assert connection.prefill_combine_tile == "auto"


def test_model_owned_hc_uses_local_fp32_gate_precision(fake_sublayers):
    from tests.models.vllm.test_qwen4_exp_math import hc_cuda_source_reference
    from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import GatedResidual

    model = qwen4_model.Qwen4ExpModel(
        vllm_config=_hc_constructor_config({}), prefix="model"
    ).bfloat16()
    x = torch.ones((1, HC * HIDDEN), dtype=torch.bfloat16)
    for connection in _hc_connections(model):
        assert type(connection) is GatedResidual
        with torch.no_grad():
            connection.hc_norm.weight.fill_(0.25)
            connection.hc_norm.weight[0] = 0
            connection.input_mix_weight_down.weight.zero_()
            connection.input_mix_weight_down.weight[0, 0] = 24
            connection.input_mix_weight_up.weight.fill_(-1)
        actual, residuals = connection.mix(x)
        # Independent fixed BF16 result from the existing CUDA precision test.
        expected = torch.tensor(0.0030975341796875, dtype=torch.bfloat16)
        torch.testing.assert_close(actual[0, 1], expected, rtol=0, atol=0)
        if connection.use_combine:
            with torch.no_grad():
                connection.block_inject_weight.weight.zero_()
                connection.block_inject_weight.weight[:, 0] = -8
            block = torch.full((1, HIDDEN), -3.125, dtype=torch.bfloat16)
            ref_mix, ref_combined = hc_cuda_source_reference(connection, x, block)
            torch.testing.assert_close(actual, ref_mix, rtol=0, atol=0)
            torch.testing.assert_close(
                connection.combine(block, residuals), ref_combined, rtol=0, atol=0
            )


def _randomize_hc_norms(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.endswith("hc_norm.weight"):
            with torch.no_grad():
                param.normal_(0.0, 0.1)


def _ref_mix(hc: nn.Module, hyper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference GatedResidual.mix with per-stream norm (hc_per_branch_norm)."""
    streams = hyper.unflatten(-1, (HC, HIDDEN))
    weight = hc.hc_norm.weight.view(HC, HIDDEN)
    eps = hc.hc_norm.variance_epsilon
    normed = streams * torch.rsqrt(streams.square().mean(-1, keepdim=True) + eps)
    normed = (normed * (1.0 + weight)).flatten(-2)
    gate = F.silu(normed @ hc.input_mix_weight_down.weight.T / HC)
    gate = torch.sigmoid(gate @ hc.input_mix_weight_up.weight.T)
    mixed = (gate * normed).unflatten(-1, (HC, HIDDEN)).mean(-2)
    return mixed, normed


def _ref_combine(
    hc: nn.Module, hyper: torch.Tensor, normed: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    inject = 2.0 * torch.sigmoid(normed @ hc.block_inject_weight.weight.T / HC)
    streams = hyper.unflatten(-1, (HC, HIDDEN))
    return (streams + out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)


def test_model_matches_hyper_connection_reference(fake_sublayers) -> None:
    torch.manual_seed(0)
    model = qwen4_model.Qwen4ExpModel(
        vllm_config=_model_vllm_config(_text_config()), prefix="model"
    )
    _randomize_hc_norms(model)
    input_ids = torch.tensor([1, 5, 7, 30, 2])
    positions = torch.arange(input_ids.numel())

    with torch.no_grad():
        output = model(input_ids, positions)

        hyper = model.embed_tokens(input_ids).repeat(1, HC)
        for layer in model.layers:
            sublayer = (
                layer.linear_attn
                if layer.layer_type == "linear_attention"
                else layer.self_attn
            )
            for hc, block in (
                (layer.attn_hyper_connection, sublayer),
                (layer.mlp_hyper_connection, layer.mlp),
            ):
                mixed, normed = _ref_mix(hc, hyper)
                block_out = torch.tanh(block.proj(mixed))
                hyper = _ref_combine(hc, hyper, normed, block_out)
        expected, _ = _ref_mix(model.hyper_connection_mixer, hyper)

    assert output.shape == (input_ids.numel(), HIDDEN)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_layer_types_select_gdn_or_dense_gated_attention(fake_sublayers) -> None:
    text_config = _text_config()
    model = qwen4_model.Qwen4ExpModel(
        vllm_config=_model_vllm_config(text_config), prefix="model"
    )
    gdn_layer, attn_layer = model.layers
    assert not hasattr(gdn_layer, "self_attn")
    assert not hasattr(attn_layer, "linear_attn")
    # Upstream Qwen4Exp, like Qwen3.5, uses the non-interleaved GDN layout.
    assert gdn_layer.linear_attn.kwargs["gqa_interleaved_layout"] is False
    assert gdn_layer.linear_attn.args[0] is text_config
    assert attn_layer.self_attn.kwargs["prefix"] == "model.layers.1.self_attn"
    # The final mixer only mixes; the checkpoint has no injection for it.
    assert not hasattr(model.hyper_connection_mixer, "block_inject_weight")

    positions = torch.arange(3)
    with torch.no_grad():
        model(torch.tensor([1, 2, 3]), positions)
    assert attn_layer.self_attn.calls[0]["positions"] is positions
    assert "positions" not in gdn_layer.linear_attn.calls[0]


def test_unknown_layer_type_is_rejected(fake_sublayers) -> None:
    text_config = _text_config()
    text_config.layer_types = ["linear_attention", "sliding_attention"]
    with pytest.raises(ValueError, match="sliding_attention"):
        qwen4_model.Qwen4ExpModel(
            vllm_config=_model_vllm_config(text_config), prefix="model"
        )


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def test_load_weights_maps_hyper_connection_names_and_skips_mtp_and_indexer(
    fake_sublayers,
) -> None:
    model = Qwen4ExpForCausalLM(vllm_config=_model_vllm_config(_text_config()))
    prefix = "model.language_model"
    skipped = (
        # MTP weights belong to the draft model.
        "mtp.fc_hidden.weight",
        # Dense attention does not use the QSA indexer.
        f"{prefix}.layers.1.self_attn.indexer.index_qk_proj.weight",
    )
    weights = {
        f"{prefix}.layers.0.attn_hyper_connection.hc_norm.weight": torch.randn(
            HC * HIDDEN
        ),
        f"{prefix}.layers.1.mlp_hyper_connection.input_mix_weight_down.weight": (
            torch.randn(8, HC * HIDDEN)
        ),
        f"{prefix}.layers.1.mlp_hyper_connection.block_inject_weight.weight": (
            torch.randn(HC, HC * HIDDEN)
        ),
        f"{prefix}.hyper_connection_mixer.input_mix_weight_up.weight": torch.randn(
            HC * HIDDEN, 8
        ),
        f"{prefix}.embed_tokens.weight": torch.randn(VOCAB, HIDDEN),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN),
        **{name: torch.randn(HIDDEN, HIDDEN) for name in skipped},
    }

    loaded = model.load_weights(iter(weights.items()))

    params = dict(model.named_parameters())
    for name, tensor in weights.items():
        if name in skipped:
            continue
        vllm_name = name.replace(f"{prefix}.", "model.")
        assert vllm_name in loaded
        param = params[vllm_name]
        torch.testing.assert_close(param, tensor.to(param.dtype))
    assert not any("mtp." in name or ".indexer." in name for name in loaded)


_LM = "model.language_model"


@pytest.mark.parametrize(
    "checkpoint_name,expected",
    [
        (f"{_LM}.embed_tokens.weight", ("model.embed_tokens.weight", None)),
        ("lm_head.weight", ("lm_head.weight", None)),
        (
            f"{_LM}.hyper_connection_mixer.hc_norm.weight",
            ("model.hyper_connection_mixer.hc_norm.weight", None),
        ),
        (
            f"{_LM}.layers.0.linear_attn.in_proj_qkv.weight",
            ("model.layers.0.linear_attn.in_proj_qkvz.weight", (0, 1, 2)),
        ),
        (
            f"{_LM}.layers.0.linear_attn.in_proj_z.weight",
            ("model.layers.0.linear_attn.in_proj_qkvz.weight", 3),
        ),
        (
            f"{_LM}.layers.0.linear_attn.in_proj_b.weight",
            ("model.layers.0.linear_attn.in_proj_ba.weight", 0),
        ),
        (
            f"{_LM}.layers.0.linear_attn.in_proj_a.weight",
            ("model.layers.0.linear_attn.in_proj_ba.weight", 1),
        ),
        (
            f"{_LM}.layers.3.self_attn.q_proj.weight",
            ("model.layers.3.self_attn.qkv_proj.weight", "q"),
        ),
        (
            f"{_LM}.layers.0.mlp.shared_expert.up_proj.weight",
            ("model.layers.0.mlp.shared_expert.gate_up_proj.weight", 1),
        ),
        (
            f"{_LM}.layers.0.mlp.experts.7.down_proj.weight_scale_inv",
            ("model.layers.0.mlp.experts.7.down_proj.weight_scale_inv", None),
        ),
        (f"{_LM}.layers.3.self_attn.indexer.index_qk_proj.weight", None),
        (
            f"{_LM}.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight",
            ("model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight", None),
        ),
        (
            f"{_LM}.layers.0.ple.key_proj.weight",
            ("model.layers.0.ple.key_proj.weight", None),
        ),
        (
            f"{_LM}.layers.0.ple.conv1d.weight",
            ("model.layers.0.ple.conv1d.weight", None),
        ),
        (f"{_LM}.layers.0.ple.ple_embedding.hashstats_ngram_counts", None),
        (f"{_LM}.layers.0.ple.ple_embedding.token_lookup", None),
        ("model.visual.blocks.0.attn.qkv.weight", None),
        ("mtp.layers.0.mlp.experts.0.gate_proj.weight", None),
    ],
)
def test_checkpoint_names_map_to_params_or_are_dropped(
    checkpoint_name: str, expected: tuple[str, object] | None
) -> None:
    mapped = list(
        Qwen4ExpForCausalLM.hf_to_vllm_mapper.apply([(checkpoint_name, torch.empty(0))])
    )
    if expected is None:
        assert mapped == []
        return
    ((name, tensor),) = mapped
    assert (name, getattr(tensor, "shard_id", None)) == expected


def test_fp8_checkpoint_quantizes_routed_experts_and_ple_table() -> None:
    not_converted = [
        "lm_head",
        f"{_LM}.embed_tokens",
        *(
            f"{_LM}.layers.0.linear_attn.{name}"
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
        ),
        f"{_LM}.layers.0.linear_attn.out_proj",
        *(
            f"{_LM}.layers.1.self_attn.{name}"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ),
        *(
            f"{_LM}.layers.{idx}.mlp.shared_expert.{name}"
            for idx in (0, 1)
            for name in ("gate_proj", "up_proj", "down_proj")
        ),
        *(
            f"{_LM}.layers.0.ple.{name}"
            for name in ("conv1d", "key_proj", "value_proj")
        ),
    ]
    quant_config = VllmFp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": not_converted,
        }
    )
    configure_quant_config(quant_config, Qwen4ExpForCausalLM)

    def skipped(prefix: str) -> bool:
        return is_layer_skipped(
            prefix,
            quant_config.ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
        )

    for prefix in (
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.0.linear_attn.in_proj_ba",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.1.self_attn.qkv_proj",
        "model.layers.1.self_attn.o_proj",
        "model.layers.0.mlp.shared_expert.gate_up_proj",
        "model.layers.1.mlp.shared_expert.down_proj",
        "model.layers.0.ple.key_proj",
        "model.layers.0.ple.value_proj",
    ):
        assert skipped(prefix), prefix
    assert not skipped("model.layers.0.mlp.experts")
    table_method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
        quant_config, "model.layers.0.ple.ple_embedding.ngram_embedding"
    )
    assert isinstance(table_method, Qwen4ExpPLEFp8EmbeddingMethod)


# ---------------------------------------------------------------------------
# N-gram PLE
# ---------------------------------------------------------------------------
_PLE = dict(
    ple_layer_ids=[1],
    ngram_size=3,
    heads_per_ngram=2,
    # The 4 heads hash into 11, 13, 17 and 19 rows: 60 rows, padded to 64.
    ngram_vocab_size_base=10,
    make_ngram_vocab_size_divisible_by=16,
    # The checkpoint splits the table into shards of 22, 22 and 20 rows.
    split_ngram_parts=3,
    ple_embed_dim=16,
    ple_conv_kernel_size=4,
    rms_norm_eps=1e-6,
    eos_token_id=3,
)
_TABLE_ROWS = 64
_HEAD_DIM = 4

# Step i runs chunk i of every request: one prefill step, chunks that start
# on and right after an EOS, and token-by-token decode.
_CHUNKS = {
    "one_step": [[12], [7]],
    "chunked": [[5, 1, 6], [2, 4, 1]],
    "decode": [[4] + [1] * 8, [1] * 7],
}


def _set_tp(monkeypatch, rank: int, size: int) -> None:
    for module in (vllm_linear, vllm_parameter, vpe):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: size
        )


def _sequences(vocab_size: int, eos: int) -> list[torch.Tensor]:
    """Two requests: one with back-to-back EOS mid-sequence, one starting
    with EOS."""
    generator = torch.Generator().manual_seed(0)
    first = torch.randint(vocab_size, (12,), generator=generator)
    first[4:6] = eos
    second = torch.randint(vocab_size, (7,), generator=generator)
    second[0] = eos
    return [first, second]


def _steps(chunks: list[list[int]]) -> list[list[tuple[int, int, int]]]:
    """(request, start, end) of chunk i of every request, for each step i."""
    return [
        [
            (request, sum(lens[:step]), sum(lens[: step + 1]))
            for request, lens in enumerate(chunks)
            if step < len(lens)
        ]
        for step in range(max(map(len, chunks)))
    ]


def _batch(
    sequences: list[torch.Tensor], step: list[tuple[int, int, int]], eos: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packs a step as the TPU runner does: padding tokens go to an empty
    trailing request, and each chunk's context is the two tokens before it."""
    chunks = [sequences[request][start:end] for request, start, end in step]
    lens = torch.tensor([len(chunk) for chunk in chunks])
    num_tokens = int(lens.sum())
    padding = torch.zeros(num_tokens // 8 * 8 + 8 - num_tokens, dtype=torch.long)
    query_start_loc = torch.tensor([0, *lens.cumsum(0).tolist(), num_tokens])
    context = []
    for request, start, _ in step:
        before = sequences[request][max(0, start - 2) : start].tolist()
        context.append([eos] * (2 - len(before)) + before)
    context.append([eos, eos])
    return torch.cat([*chunks, padding]), query_start_loc, torch.tensor(context)


# Reference math adapted from HF transformers' Qwen4Exp modeling code
# (Qwen4ExpTextNGramEmbedding and Qwen4ExpTextPLELayer without a cache), which
# the transformers version CI installs does not include.
def _ref_shift_right_ignore_eos(
    tokens: torch.Tensor, shift: int, eos: int
) -> torch.Tensor:
    positions = torch.arange(tokens.numel())
    eos_positions = torch.where(tokens == eos, positions, -1)
    previous_eos = torch.cat(
        [eos_positions.new_full((1,), -1), eos_positions.cummax(0).values[:-1]]
    )
    source = positions - shift
    valid = (positions - previous_eos - 1 >= shift) & (source >= 0)
    return torch.where(valid, tokens[source.clamp(min=0)], eos)


def _ref_ngram_ids(
    tokens: torch.Tensor,
    multipliers: torch.Tensor,
    sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    ngram_size = multipliers.numel()
    history = torch.cat([torch.full((ngram_size - 1,), eos), tokens])
    shifted = [
        _ref_shift_right_ignore_eos(history, shift, eos) for shift in range(ngram_size)
    ]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        heads = slice((ngram - 2) * heads_per_ngram, (ngram - 1) * heads_per_ngram)
        mixed = shifted[0] * multipliers[0]
        for index in range(1, ngram):
            mixed = mixed ^ (shifted[index] * multipliers[index])
        blocks.append(mixed.unsqueeze(-1) % sizes[heads] + offsets[heads])
    return torch.cat(blocks, dim=-1)[ngram_size - 1 :]


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    groups = x.unflatten(-1, (-1, HIDDEN))
    normed = groups * torch.rsqrt(groups.pow(2).mean(-1, keepdim=True) + 1e-6)
    return normed.flatten(-2) * (1.0 + weight)


def _ref_ple(
    ple: Qwen4ExpPLELayer, hidden_states: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    embedding = ple.ple_embedding
    ids = _ref_ngram_ids(
        tokens,
        embedding.layer_multipliers,
        embedding.ngram_heads_vocab_sizes,
        embedding.ngram_heads_offsets,
        _PLE["eos_token_id"],
        _PLE["heads_per_ngram"],
    )
    embeddings = embedding.ngram_embedding.weight[ids].flatten(-2)
    key = _ref_rms_norm(embeddings @ ple.key_proj.weight.T, ple.norm_key.weight)
    query = _ref_rms_norm(hidden_states, ple.norm_query.weight)
    gate = (key * query).unflatten(-1, (HC, HIDDEN)).sum(-1, keepdim=True)
    gate = gate / math.sqrt(HIDDEN)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    value = embeddings @ ple.value_proj.weight.T
    gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
    dilation = _PLE["ngram_size"]
    history = (_PLE["ple_conv_kernel_size"] - 1) * dilation
    conv_input = F.pad(_ref_rms_norm(gated, ple.norm_conv.weight).T, (history, 0))
    conv = F.conv1d(
        conv_input, ple.conv1d.weight, groups=HC * HIDDEN, dilation=dilation
    )
    return gated + F.silu(conv.T)


def _randomize_ple(ple: nn.Module) -> None:
    with torch.no_grad():
        for param in ple.parameters():
            param.normal_(0.0, 0.5)


@pytest.mark.parametrize(
    "ple_layer_index,multipliers,sizes",
    [
        # The buffers stored in the released Qwen3.8-Flash-Next checkpoint.
        (
            0,
            [23703573157769, 20109073645365, 8052911324071],
            [
                *(20000003, 20000023, 20000033, 20000047, 20000059, 20000063),
                *(20000069, 20000077, 20000081, 20000093, 20000107, 20000147),
                *(20000153, 20000159, 20000161, 20000171),
            ],
        ),
        # Upstream's _make_layer_multipliers and _make_vocab_layout for a
        # second PLE layer.
        (
            1,
            [3352040966061, 19752442147475, 10742103100073],
            [
                *(20000213, 20000221, 20000243, 20000269, 20000287, 20000297),
                *(20000303, 20000311, 20000327, 20000329, 20000339, 20000353),
                *(20000359, 20000377, 20000389, 20000429),
            ],
        ),
    ],
    ids=["released_checkpoint", "second_ple_layer"],
)
def test_hash_buffers_match_checkpoint_and_upstream(
    ple_layer_index: int, multipliers: list[int], sizes: list[int]
) -> None:
    """Config-derived multipliers and head sizes at the released checkpoint's
    vocab (248320) and ngram_vocab_size_base (20M)."""
    assert layer_multipliers(248320, 3, ple_layer_index, 1234) == multipliers
    assert head_vocab_sizes(20_000_000, 16, ple_layer_index) == sizes


@pytest.mark.parametrize(
    "query_start_loc,num_tokens,expected_request,expected_chunk_pos",
    [
        # Requests 1 and 3 are empty; the two padding tokens go to request 3.
        ([0, 2, 2, 5, 5], 7, [0, 0, 2, 2, 2, 3, 3], [0, 1, 0, 1, 2, 0, 1]),
        # A request start past the token bucket is dropped.
        ([0, 2, 6, 8], 4, [0, 0, 1, 1], [0, 1, 0, 1]),
    ],
    ids=["empty_requests_and_padding", "start_past_bucket"],
)
def test_token_requests(
    query_start_loc: list[int],
    num_tokens: int,
    expected_request: list[int],
    expected_chunk_pos: list[int],
) -> None:
    request, chunk_pos = token_requests(torch.tensor(query_start_loc), num_tokens)
    assert request.tolist() == expected_request
    assert chunk_pos.tolist() == expected_chunk_pos


@pytest.mark.parametrize("chunks", list(_CHUNKS.values()), ids=list(_CHUNKS))
def test_ngram_ids_match_reference_for_any_chunking(chunks) -> None:
    """Hashing each chunk with the two tokens before it gives the same ids as
    hashing the whole sequence, at the released checkpoint's sizes."""
    vocab_size, eos, heads_per_ngram = 248320, 248044, 8
    multipliers = torch.tensor(layer_multipliers(vocab_size, 3, 0, 1234))
    sizes = torch.tensor(head_vocab_sizes(20_000_000, 2 * heads_per_ngram, 0))
    offsets = sizes.cumsum(0) - sizes
    sequences = _sequences(vocab_size, eos)
    got: list[list[torch.Tensor]] = [[] for _ in sequences]
    for step in _steps(chunks):
        input_ids, query_start_loc, context = _batch(sequences, step, eos)
        request, chunk_pos = token_requests(query_start_loc, input_ids.numel())
        ids = ngram_ids(
            input_ids,
            context,
            request,
            chunk_pos,
            multipliers,
            sizes,
            offsets,
            eos_token_id=eos,
            heads_per_ngram=heads_per_ngram,
        )
        for index, (request, _, _) in enumerate(step):
            start, end = query_start_loc[index], query_start_loc[index + 1]
            got[request].append(ids[start:end])
    for request, tokens in enumerate(sequences):
        expected = _ref_ngram_ids(
            tokens, multipliers, sizes, offsets, eos, heads_per_ngram
        )
        assert torch.equal(torch.cat(got[request]), expected)


@pytest.mark.parametrize("chunks", list(_CHUNKS.values()), ids=list(_CHUNKS))
def test_ple_layer_matches_reference_for_any_chunking(monkeypatch, chunks) -> None:
    """Padded, chunked batches that carry each request's conv window between
    steps match the reference run over each whole sequence."""
    _set_tp(monkeypatch, 0, 1)
    torch.manual_seed(0)
    ple = Qwen4ExpPLELayer(_text_config(**_PLE), 0, None, prefix="model.layers.0.ple")
    _randomize_ple(ple)
    eos = _PLE["eos_token_id"]
    sequences = _sequences(VOCAB, eos)
    hidden_states = [torch.randn(len(tokens), HC * HIDDEN) for tokens in sequences]
    # The caller owns the conv state. As in the TPU runner, slot 0 serves the
    # padding request, and a request without history gets a zero window
    # instead of its stale slot.
    conv_state = torch.randn(len(sequences) + 1, ple.conv_state_len, HC * HIDDEN)
    got: list[list[torch.Tensor]] = [[] for _ in sequences]
    with torch.no_grad():
        for step in _steps(chunks):
            input_ids, query_start_loc, context = _batch(sequences, step, eos)
            hidden = torch.cat([hidden_states[r][start:end] for r, start, end in step])
            hidden = F.pad(hidden, (0, 0, 0, input_ids.numel() - hidden.shape[0]))
            slots = torch.tensor([r + 1 for r, _, _ in step] + [0])
            has_history = torch.tensor([s > 0 for _, s, _ in step] + [False])
            windows = torch.where(has_history.view(-1, 1, 1), conv_state[slots], 0)
            output, new_windows = ple(
                hidden, PLEInputs(input_ids, query_start_loc, context), windows
            )
            conv_state[slots] = new_windows
            for index, (request, _, _) in enumerate(step):
                start, end = query_start_loc[index], query_start_loc[index + 1]
                got[request].append(output[start:end])
        for request, tokens in enumerate(sequences):
            torch.testing.assert_close(
                torch.cat(got[request]),
                _ref_ple(ple, hidden_states[request], tokens),
                rtol=1e-5,
                atol=1e-5,
            )


@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "unquantized"])
def test_ple_table_shards_load_across_tp_ranks(monkeypatch, fp8: bool) -> None:
    """Each of two TP ranks keeps its rows of every checkpoint shard, and the
    ranks' lookups sum to the dequantized table."""
    quant_config = (
        VllmFp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            }
        )
        if fp8
        else None
    )
    table = torch.randn(_TABLE_ROWS, _HEAD_DIM)
    if fp8:
        table = table.to(torch.float8_e4m3fn)
    # The middle shard straddles the two ranks' row ranges.
    weights = {
        f"ngram_embedding.shard_{index}.weight": rows
        for index, rows in enumerate(table.split(22))
    }
    expected_loaded = {"ngram_embedding.weight"}
    if fp8:
        weights["ngram_embedding.weight_scale"] = torch.tensor(
            [0.25], dtype=torch.bfloat16
        )
        expected_loaded.add("ngram_embedding.weight_scale")
    ids = torch.randperm(_TABLE_ROWS).view(-1, 4)
    rows = torch.zeros(*ids.shape, _HEAD_DIM)
    for rank in range(2):
        _set_tp(monkeypatch, rank, 2)
        embedding = Qwen4ExpNGramEmbedding(
            _text_config(**_PLE),
            _PLE["ple_embed_dim"],
            0,
            quant_config,
            prefix="model.layers.0.ple.ple_embedding",
        )
        assert embedding.load_weights(iter(weights.items())) == expected_loaded
        rank_table = embedding.ngram_embedding
        assert rank_table.weight.shape == (_TABLE_ROWS // 2, _HEAD_DIM)
        assert rank_table.weight.dtype == table.dtype
        # Raises if an FP8 table's scale was never loaded.
        rank_table.embedding_method.process_weights_after_loading(rank_table)
        rows += rank_table.local_rows(ids)
    expected = table.float() * 0.25 if fp8 else table
    torch.testing.assert_close(rows, expected[ids])


@pytest.mark.parametrize("ple_layer_ids", [[1], []], ids=["ple", "ple_disabled"])
def test_load_weights_loads_ple_or_drops_it_when_disabled(
    fake_sublayers, monkeypatch, ple_layer_ids
) -> None:
    _set_tp(monkeypatch, 0, 1)
    text_config = _text_config(**{**_PLE, "ple_layer_ids": ple_layer_ids})
    model = Qwen4ExpForCausalLM(vllm_config=_model_vllm_config(text_config))
    prefix = f"{_LM}.layers.0.ple"
    weights = {
        f"{prefix}.key_proj.weight": torch.randn(HC * HIDDEN, _PLE["ple_embed_dim"]),
        f"{prefix}.conv1d.weight": torch.randn(
            HC * HIDDEN, 1, _PLE["ple_conv_kernel_size"]
        ),
        f"{prefix}.norm_conv.weight": torch.randn(HC * HIDDEN),
        f"{prefix}.ple_embedding.layer_multipliers": torch.tensor([3, 5, 7]),
        f"{prefix}.ple_embedding.ngram_embedding.shard_2.weight": torch.randn(
            20, _HEAD_DIM
        ),
    }

    loaded = model.load_weights(iter(weights.items()))

    if not ple_layer_ids:
        assert loaded == set()
        return
    expected = {
        name.replace(f"{_LM}.", "model."): tensor for name, tensor in weights.items()
    }
    table = "model.layers.0.ple.ple_embedding.ngram_embedding"
    last_shard = expected.pop(f"{table}.shard_2.weight")
    assert loaded == {*expected, f"{table}.weight"}
    state = model.state_dict()
    for name, tensor in expected.items():
        torch.testing.assert_close(state[name], tensor)
    last_shard_start = 2 * math.ceil(_TABLE_ROWS / _PLE["split_ngram_parts"])
    torch.testing.assert_close(state[f"{table}.weight"][last_shard_start:], last_shard)


def test_model_rejects_ple_without_runner_inputs(fake_sublayers, monkeypatch) -> None:
    _set_tp(monkeypatch, 0, 1)
    model = qwen4_model.Qwen4ExpModel(
        vllm_config=_model_vllm_config(_text_config(**_PLE)), prefix="model"
    )
    with pytest.raises(NotImplementedError, match="PLE"):
        model(torch.tensor([1, 2, 3]), torch.arange(3))


@pytest.mark.parametrize(
    "weights,match",
    [
        (
            {"ngram_embedding.shard_0.weight": torch.zeros(21, _HEAD_DIM)},
            "shard 0 has shape",
        ),
        (
            {"ngram_embedding.shard_3.weight": torch.zeros(22, _HEAD_DIM)},
            "out of range",
        ),
        (
            {"ngram_embedding.shard_x.weight": torch.zeros(22, _HEAD_DIM)},
            "shard_<index>",
        ),
        ({"ngram_embedding.weight_scale": torch.ones(1)}, "Unexpected PLE"),
        ({"layer_multipliers": torch.tensor([1 << 62, 3, 5])}, "overflows int64"),
    ],
    ids=["shard_shape", "shard_index", "shard_name", "scale_without_fp8", "overflow"],
)
def test_ngram_embedding_rejects_malformed_checkpoint_tensors(
    monkeypatch, weights: dict[str, torch.Tensor], match: str
) -> None:
    _set_tp(monkeypatch, 0, 1)
    embedding = Qwen4ExpNGramEmbedding(
        _text_config(**_PLE),
        _PLE["ple_embed_dim"],
        0,
        None,
        prefix="model.layers.0.ple.ple_embedding",
    )
    with pytest.raises(ValueError, match=match):
        embedding.load_weights(iter(weights.items()))


@pytest.mark.parametrize(
    "num_ids,context_delta,window_delta,match",
    [
        (2, 0, 0, "input ids"),
        (3, -1, 0, "ngram_context"),
        (3, 0, -1, "conv windows"),
    ],
    ids=["input_ids", "ngram_context", "conv_windows"],
)
def test_ple_layer_rejects_inconsistent_inputs(
    monkeypatch, num_ids: int, context_delta: int, window_delta: int, match: str
) -> None:
    _set_tp(monkeypatch, 0, 1)
    ple = Qwen4ExpPLELayer(_text_config(**_PLE), 0, None, prefix="model.layers.0.ple")
    inputs = PLEInputs(
        input_ids=torch.arange(num_ids) + 5,
        query_start_loc=torch.tensor([0, 3]),
        ngram_context=torch.full(
            (1, _PLE["ngram_size"] - 1 + context_delta), _PLE["eos_token_id"]
        ),
    )
    windows = torch.zeros(1, ple.conv_state_len + window_delta, HC * HIDDEN)
    with torch.no_grad(), pytest.raises(ValueError, match=match):
        ple(torch.zeros(3, HC * HIDDEN), inputs, windows)


def test_ple_layer_rejects_kernel_size_one(monkeypatch) -> None:
    _set_tp(monkeypatch, 0, 1)
    config = _text_config(**{**_PLE, "ple_conv_kernel_size": 1})
    with pytest.raises(NotImplementedError, match="ple_conv_kernel_size"):
        Qwen4ExpPLELayer(config, 0, None, prefix="model.layers.0.ple")

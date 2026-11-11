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

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn
from transformers import Qwen4ExpTextConfig
from vllm.config import DeviceConfig, VllmConfig

from tests.models.vllm.test_qwen4_exp_parallel import config
from vllm_torchtpu.models.vllm.qwen4_exp import pd_connector
from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config
from vllm_torchtpu.models.vllm.qwen4_exp.model import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpForCausalLM,
)


@pytest.fixture
def model_builder(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from vllm_torchtpu.models.vllm.qwen4_exp import model as module

    monkeypatch.setattr(
        module, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        module, "Qwen3NextSparseMoeBlock", lambda **kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        module, "QwenGatedDeltaNetAttention", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(module, "GatedResidual", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(cfg, [["LBNHC", "LBHNC"]])

    def build(kinds):
        text = Qwen4ExpTextConfig.from_dict(
            dict(
                num_hidden_layers=len(kinds),
                layer_types=kinds,
                ple_layer_ids=[],
                hidden_size=16,
                vocab_size=64,
                hc_lowrank=8,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=128,
                indexer_kv_heads=1,
                indexer_compress_ratio=4,
                indexer_n_heads=2,
                indexer_head_dim=64,
                indexer_budget=64,
                partial_rotary_factor=0.5,
            )
        )
        settings = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=text, max_model_len=32),
            kv_transfer_config=object(),
            speculative_config=None,
            additional_config={},
            device_config=SimpleNamespace(device="cpu"),
            cache_config=SimpleNamespace(cache_dtype="fp8"),
            compilation_config=cfg.compilation_config,
        )
        return Qwen4ExpForCausalLM(vllm_config=settings)

    with set_current_vllm_config(cfg):
        yield build


def test_loading_ignores_unused_keys_and_normalizes_prefix(model_builder):
    model = model_builder([])
    weights = [
        (name, torch.full_like(param, 0.25)) for name, param in model.named_parameters()
    ]
    model.load_weights(iter(weights))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    renamed = [
        (name.replace("model.", "model.language_model.", 1), value)
        for name, value in weights
    ]
    ignored = torch.arange(12).reshape(3, 4).T
    model.load_weights(
        iter([*renamed, ("model.visual.weight", ignored), ("mtp.weight", ignored + 1)])
    )
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    changed = [(name, value + 1) for name, value in weights]
    model.load_weights(iter(changed))
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, dict(changed)[name], rtol=0, atol=0)
    assert torch.equal(ignored, torch.arange(12).reshape(3, 4).T)


@pytest.mark.parametrize("fault", [None, "mismatch", "nonfinite", "empty"])
def test_actual_model_normalized_layers_bind_loaded_scale_identity(
    monkeypatch, fault, model_builder
):
    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")
    model = model_builder(
        ["linear_attention" if fault == "empty" else "full_attention"]
    )
    layer = cast(Qwen4ExpDecoderLayer, model.model.layers[0])
    if fault != "empty":
        attention = layer.self_attn
        attention.load_weights(
            [("_k_scale", torch.tensor(0.5)), ("_v_scale", torch.tensor(0.25))]
        )
        if fault == "mismatch":
            attention._k_scale_cpu.fill_(0.75)
        elif fault == "nonfinite":
            attention._v_scale_float = float("inf")
    worker = pd_connector.Qwen4ExpRaidenWorker.__new__(
        pd_connector.Qwen4ExpRaidenWorker
    )
    worker.tp_rank, worker.tp_size = 0, 1
    worker.vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    worker.vllm_config.cache_config.mamba_ssm_cache_dtype = "bfloat16"
    runner = SimpleNamespace(
        model_adapter=SimpleNamespace(
            cache=SimpleNamespace(model=model, specs={}, draft_model=None)
        )
    )
    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")
    monkeypatch.setattr(
        pd_connector,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1, cpu_group="cpu"),
    )

    def gather(output, value, *, group):
        assert group == "cpu"
        output[:] = [value]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    if fault is not None:
        message = {
            "mismatch": "differ from loaded",
            "nonfinite": "finite positive",
            "empty": "indexed-attention",
        }[fault]
        with pytest.raises(ValueError, match=message):
            worker._collect_pd_semantics(runner)
    else:
        assert layer.layer_type == "indexed_attention"
        actual = worker._collect_pd_semantics(runner)
        flags = actual["ranks"][0]["state_flags"]
        assert flags["gdn_conv_dtype"] == "torch.float32"
        assert flags["gdn_granule_bytes"] == 512
        monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "0")
        assert worker._collect_pd_semantics(runner) == actual
        attention = layer.self_attn
        assert actual["ranks"][0]["kv_scales"] == [(0, 0.5, 0.25)]
        attention._v_scale_float = 0.75
        attention._v_scale_cpu.fill_(0.75)
        assert worker._collect_pd_semantics(runner) != actual


@pytest.mark.parametrize("dp", [1, 2, 4, 8])
def test_pd_tp1_uses_engine_geometry_instead_of_a_dp_literal(monkeypatch, dp):
    for name in ("QWEN35", "KIMIK3", "GLM", "DSV4"):
        monkeypatch.setenv(f"TPU_RAIDEN_{name}_ADMISSION", "0")
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    cfg = config(tp=1, dp=dp)
    cfg.kv_transfer_config = SimpleNamespace(
        kv_connector="Qwen4ExpRaidenConnector",
        kv_connector_module_path=None,
        kv_role="kv_producer",
    )
    cfg.cache_config.cache_dtype = "fp8"
    cfg.cache_config.mamba_ssm_cache_dtype = "bfloat16"
    validate_config(cfg)
    cfg.parallel_config.tensor_parallel_size = 2
    validate_config(cfg)


@pytest.mark.parametrize("pd_enabled", [False, True])
def test_pd_uses_the_same_public_weight_loaders(model_builder, pd_enabled):
    model = model_builder([])
    model.vllm_config.kv_transfer_config = object() if pd_enabled else None
    weights = [
        (name, torch.full_like(value, 0.75)) for name, value in model.named_parameters()
    ]
    before = {name: value.clone() for name, value in weights}
    assert model.load_weights(iter(weights)) == set(dict(model.named_parameters()))
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for name, value in weights:
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_real_mtp_attention_registry_admits_prefill_only(model_builder, monkeypatch):
    from vllm.config import get_current_vllm_config, get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec

    from tests.runner.test_qwen4_exp_pool import pool_config
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    target = model_builder(["indexed_attention"])
    config = pool_config()
    config.model_config.hf_text_config = target.config
    config.model_config.max_model_len = 32
    config.model_config.head_dtype = None
    config.cache_config.enable_prefix_caching = False
    config.cache_config.mamba_cache_mode = "none"
    config.compilation_config = get_current_vllm_config().compilation_config
    config.device_config = DeviceConfig(device="cpu")
    config.quant_config = None
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=3,
        method="mtp",
        use_eagle=lambda: True,
        use_eagle_block_drop=lambda: False,
    )
    monkeypatch.setattr(mtp, "GatedResidual", lambda *args, **kwargs: nn.Identity())
    before = set(get_layers_from_vllm_config(config, AttentionLayerBase))
    from vllm.model_executor.layers import linear

    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    draft = mtp.Qwen4ExpMultiTokenPredictor(config, prefix="mtp")
    registered = get_layers_from_vllm_config(config, AttentionLayerBase)
    names = set(registered) - before
    assert names == {draft.attention_prefix}
    assert isinstance(
        registered[draft.attention_prefix].get_kv_cache_spec(config), FullAttentionSpec
    )
    assert isinstance(
        registered[draft.attention_prefix].get_auxiliary_cache_specs()["raw_ring"],
        CircularBufferSpec,
    )
    runner = SimpleNamespace(vllm_config=config, cache_config=config.cache_config)
    cache = Qwen4ExpPooledCache(runner, target)
    specs = cache.get_specs()
    proposer = object.__new__(Eagle3Proposer)
    proposer.runner = SimpleNamespace(get_kv_cache_spec=cache.get_specs)
    proposer._validate_prefill_only_draft_model(names)
    assert isinstance(specs[draft.attention_prefix], FullAttentionSpec)
    assert isinstance(specs[draft.attention_prefix + ".raw_ring"], CircularBufferSpec)

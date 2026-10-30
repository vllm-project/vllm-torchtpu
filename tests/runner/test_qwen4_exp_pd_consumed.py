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
            additional_config={},
            device_config=SimpleNamespace(device="cpu"),
            cache_config=SimpleNamespace(cache_dtype="fp8"),
            compilation_config=cfg.compilation_config,
        )
        return Qwen4ExpForCausalLM(vllm_config=settings)

    with set_current_vllm_config(cfg):
        yield build


def test_loaded_identity_ignores_unused_keys_and_normalizes_prefix(model_builder):
    model = model_builder([])
    weights = [
        (name, torch.full_like(param, 0.25)) for name, param in model.named_parameters()
    ]
    model.load_weights(iter(weights))
    before = model.pd_loaded_weight_identity
    renamed = [
        (name.replace("model.", "model.language_model.", 1), value)
        for name, value in weights
    ]
    ignored = torch.arange(12).reshape(3, 4).T
    model.load_weights(
        iter([*renamed, ("model.visual.weight", ignored), ("mtp.weight", ignored + 1)])
    )
    assert model.pd_loaded_weight_identity == before
    changed = [(name, value + 1) for name, value in weights]
    model.load_weights(iter(changed))
    assert model.pd_loaded_weight_identity != before
    assert torch.equal(ignored, torch.arange(12).reshape(3, 4).T)


@pytest.mark.parametrize("fault", [None, "mismatch", "nonfinite", "empty"])
def test_actual_model_normalized_layers_bind_loaded_scale_identity(
    monkeypatch, fault, model_builder
):
    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")
    model = model_builder(
        ["linear_attention" if fault == "empty" else "full_attention"]
    )
    model.pd_loaded_weight_identity = "a" * 64
    model.pd_loaded_ple_identity = "b" * 64
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
        model_adapter=SimpleNamespace(cache=SimpleNamespace(model=model))
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


def test_pd_disabled_uses_the_same_public_loaders_without_an_observer(
    monkeypatch, model_builder
):
    from vllm_torchtpu.models.vllm.qwen4_exp import model as module

    model = model_builder([])
    model._pd_weight_identity_enabled = False
    weights = [
        (name, torch.full_like(value, 0.75)) for name, value in model.named_parameters()
    ]

    def forbidden(*args, **kwargs):
        pytest.fail("disabled PD created a weight observer")

    monkeypatch.setattr(module, "LoadedWeightIdentity", forbidden)
    assert model.load_weights(iter(weights)) == set(dict(model.named_parameters()))
    assert model.pd_loaded_weight_identity is None
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, dict(weights)[name], rtol=0, atol=0)

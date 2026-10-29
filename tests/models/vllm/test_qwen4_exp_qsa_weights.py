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
"""Early QSA geometry checks and canonical checkpoint scale buffers."""

import pytest
import torch
from qwen4_exp_test_utils import q4exp_text_config as attention_config

from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA


@pytest.fixture(autouse=True)
def cpu_vllm_config_context():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


@pytest.fixture(autouse=True)
def local_tp(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )


def small_config(**updates):
    fields = dict(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        indexer_n_heads=2,
        indexer_head_dim=64,
        partial_rotary_factor=0.5,
        indexer_budget=64,
    )
    fields.update(updates)
    return attention_config(**fields)


@pytest.mark.parametrize(
    "updates,cache_dtype,match",
    [
        ({"hidden_size": 0}, "auto", "hidden_size"),
        ({}, "float16", "cache dtype"),
        ({"indexer_compress_ratio": 0}, "auto", "compress_ratio"),
        ({"indexer_compress_ratio": -2}, "auto", "compress_ratio"),
        ({"indexer_budget": 0}, "auto", "indexer_budget"),
        ({"indexer_budget": 14}, "auto", "indexer_budget"),
        ({"partial_rotary_factor": 0.0}, "auto", "rotary_dim"),
        ({"partial_rotary_factor": 1 / 128}, "auto", "rotary_dim"),
        ({"partial_rotary_factor": 1.0}, "auto", "rotary_dim"),
        ({"num_attention_heads": 3}, "auto", "divisible by num_key_value_heads"),
        ({"head_dim": 129}, "fp8", "even head_dim"),
        ({"rms_norm_eps": 0.0}, "auto", "norm epsilon"),
        ({"rope_parameters": {"rope_theta": float("nan")}}, "auto", "RoPE theta"),
    ],
)
def test_new_contracts_fail_before_allocation(updates, cache_dtype, match, monkeypatch):
    import vllm_torchtpu.models.vllm.qwen4_exp.qsa as qsa

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid QSA geometry reached allocation")

    monkeypatch.setattr(qsa, "ColumnParallelLinear", forbidden)
    monkeypatch.setattr(qsa, "ReplicatedLinear", forbidden)
    monkeypatch.setattr(qsa, "make_rope_cache", forbidden)
    with pytest.raises(ValueError, match=match):
        Qwen4ExpQSA(
            config=small_config(**updates),
            tp_rank=0,
            tp_size=1,
            max_model_len=16,
            cache_dtype=cache_dtype,
        )


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "fp8", "fp8_e4m3"])
def test_component_geometry_does_not_assume_a_cache_page(cache_dtype):
    layer = Qwen4ExpQSA(
        config=small_config(indexer_compress_ratio=3, indexer_budget=12),
        tp_rank=0,
        tp_size=1,
        max_model_len=16,
        cache_dtype=cache_dtype,
    )
    assert layer.compress_ratio == 3
    assert layer.q_proj.weight.shape == (1024, 16)
    assert layer.k_proj.weight.shape == (256, 16)
    assert layer.rope_cache.shape == (16, 128)


def test_canonical_scale_buffers_keep_host_and_device_values_consistent():
    k_name, v_name = "_k_scale", "_v_scale"
    layer = Qwen4ExpQSA(
        config=small_config(), tp_rank=0, tp_size=1, max_model_len=16, cache_dtype="fp8"
    )
    loaded = layer.load_weights(
        [(k_name, torch.tensor(0.375)), (v_name, torch.tensor(2.5))]
    )
    assert loaded == {"_k_scale", "_v_scale"}
    assert (
        layer._k_scale.item()
        == layer._k_scale_cpu.item()
        == layer._k_scale_float
        == 0.375
    )
    assert (
        layer._v_scale.item()
        == layer._v_scale_cpu.item()
        == layer._v_scale_float
        == 2.5
    )
    assert layer._k_scale.dtype == layer._v_scale.dtype == torch.float32
    with pytest.raises(ValueError, match="Duplicate"):
        layer.load_weights([("_k_scale", torch.tensor(1.0))])


@pytest.mark.parametrize(
    "name",
    [
        "k_proj.output_scale",
        "attn._k_scale",
        "k_proj.k_scale",
        "attn.k_scale",
        "k_scale",
        "unknown_scale",
        "k_proj.k_scale.extra",
    ],
)
def test_unknown_scale_names_are_rejected(name):
    layer = Qwen4ExpQSA(config=small_config(), tp_rank=0, tp_size=1, max_model_len=16)
    with pytest.raises(ValueError, match="Invalid BF16 QSA checkpoint tensor"):
        layer.load_weights([(name, torch.tensor(1.0))])
    assert layer._k_scale.item() == layer._k_scale_float == 1.0


@pytest.mark.parametrize(
    "weight",
    [
        torch.ones(2),
        torch.tensor(0.0),
        torch.tensor(-1.0),
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
    ],
)
def test_scale_payload_must_be_finite_positive_scalar(weight):
    layer = Qwen4ExpQSA(config=small_config(), tp_rank=0, tp_size=1, max_model_len=16)
    with pytest.raises(ValueError, match="per-tensor|finite positive"):
        layer.load_weights([("_k_scale", weight)])
    assert (
        layer._k_scale.item()
        == layer._k_scale_cpu.item()
        == layer._k_scale_float
        == 1.0
    )


def test_duplicate_scale_in_one_stream_is_rejected():
    layer = Qwen4ExpQSA(config=small_config(), tp_rank=0, tp_size=1, max_model_len=16)
    with pytest.raises(ValueError, match="Duplicate"):
        layer.load_weights(
            [
                ("_k_scale", torch.tensor(0.5)),
                ("_k_scale", torch.tensor(0.25)),
            ]
        )
    assert layer._k_scale.item() == layer._k_scale_float == 0.5


def test_checkpoint_weights_finalize_and_reject_bad_payloads():
    layer = Qwen4ExpQSA(config=small_config(), tp_rank=0, tp_size=1, max_model_len=16)
    with pytest.raises(ValueError, match="Missing"):
        layer.finalize_weights()
    for name, weight in [
        ("q_proj.weight", torch.zeros(1, 16, dtype=torch.bfloat16)),
        ("q_proj.weight", torch.zeros(1024, 16)),
        ("q_proj.typo", torch.zeros(1024, 16, dtype=torch.bfloat16)),
    ]:
        with pytest.raises(ValueError, match="Invalid BF16"):
            layer.load_weights([(name, weight)])
    weights = [
        (name, torch.full_like(parameter, (index + 1) / 16))
        for index, (name, parameter) in enumerate(layer.named_parameters())
    ]
    assert len(weights) == 9
    assert layer.load_weights(weights) == {name for name, _ in weights}
    layer.finalize_weights()
    for name, parameter in layer.named_parameters():
        torch.testing.assert_close(parameter, dict(weights)[name], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["kv_scale", "attn.kv_scale"])
def test_deprecated_shared_scale_is_rejected_without_partial_update(name):
    layer = Qwen4ExpQSA(config=small_config(), tp_rank=0, tp_size=1, max_model_len=16)
    with pytest.raises(ValueError, match="Invalid BF16 QSA checkpoint tensor"):
        layer.load_weights([(name, torch.tensor(0.5))])
    assert (
        layer._k_scale.item()
        == layer._k_scale_cpu.item()
        == layer._k_scale_float
        == 1.0
    )
    assert (
        layer._v_scale.item()
        == layer._v_scale_cpu.item()
        == layer._v_scale_float
        == 1.0
    )

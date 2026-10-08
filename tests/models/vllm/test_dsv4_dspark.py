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
"""Units for the DSV4 DSpark draft model: registry, checkpoint-name rewrite,
construction wiring, and the model-level weight load.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models import ModelRegistry

import vllm_torchtpu.models.vllm.deepseek_v4.dspark as dspark_mod
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.deepseek_v4.dspark import _rewrite_dspark_name
from vllm_torchtpu.models.vllm.deepseek_v4.model import _make_deepseek_v4_weights_mapper

HIDDEN = 64
HC = 2
VOCAB = 97
RANK = 16
N_TARGET_LAYERS = 6
TARGET_LAYER_IDS = [3, 4, 5]


def test_dspark_draft_model_registration():
    """Verify DSparkDraftModel resolves to the TPU draft class."""
    register_models()
    entry = ModelRegistry.models.get("DSparkDraftModel")
    assert entry is not None, "DSparkDraftModel is not registered"
    assert entry.load_model_cls() is dspark_mod.DSparkDeepseekV4ForCausalLM


# Checkpoint name -> DeepseekV4Model parameter path, or None for names the
# draft loader skips. Name shapes follow the DeepSeek-V4-Flash-DSpark
# inventory: three full-layer stages under mtp.{0,1,2}, the combiner on
# stage 0, the head stack and confidence head on the last stage.
@pytest.mark.parametrize(
    "expert_dtype,name,expected",
    [
        # Per-layer params go to their stage's decoder block; the target's
        # scale and rename conventions apply on top.
        ("fp4", "mtp.0.attn.wq_a.weight", "layers.0.attn.wq_a.weight"),
        ("fp4", "mtp.1.attn.wq_a.scale", "layers.1.attn.wq_a.weight_scale_inv"),
        (
            "fp4",
            "mtp.0.ffn.experts.7.w1.scale",
            "layers.0.ffn.experts.7.w1.weight_scale",
        ),
        (
            "fp8",
            "mtp.0.ffn.experts.7.w1.scale",
            "layers.0.ffn.experts.7.w1.weight_scale_inv",
        ),
        ("fp4", "mtp.2.ffn.gate.bias", "layers.2.ffn.gate.e_score_correction_bias"),
        (
            "fp4",
            "mtp.1.ffn.shared_experts.w2.weight",
            "layers.1.ffn.shared_experts.down_proj.weight",
        ),
        # Combiner and head stack live at model level; markov weights must
        # not be touched by any stacking or scale rule.
        ("fp4", "mtp.0.main_proj.scale", "main_proj.weight_scale_inv"),
        ("fp4", "mtp.2.norm.weight", "norm.weight"),
        ("fp4", "mtp.2.hc_head_fn", "hc_head_fn"),
        ("fp4", "mtp.2.markov_head.markov_w1.weight", "markov_head.markov_w1.weight"),
        # Dropped: the confidence head, and everything the target owns.
        ("fp4", "mtp.2.confidence_head.proj.weight", None),
        ("fp4", "layers.7.attn.wkv.weight", None),
        ("fp4", "embed.weight", None),
    ],
)
def test_checkpoint_name_rewrite(expert_dtype, name, expected):
    mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
    assert _rewrite_dspark_name(name, mapper) == expected


class _StubDecoderLayer(nn.Module):
    """Stands in for DeepseekV4DecoderLayer at construction time."""

    # DSV4's sliding window; the block check test shrinks it.
    window_size = 128

    def __init__(self, vllm_config, prefix="", topk_indices_buffer=None):
        super().__init__()
        del vllm_config, topk_indices_buffer
        self.prefix = prefix
        self.attn = SimpleNamespace(
            swa_cache_layer=SimpleNamespace(prefix=f"{prefix}.attn.swa"),
            window_size=type(self).window_size,
        )


def _tiny_vllm_config():
    hf_config = SimpleNamespace(
        hidden_size=HIDDEN,
        hc_mult=HC,
        hc_eps=1e-5,
        rms_norm_eps=1e-6,
        num_hidden_layers=N_TARGET_LAYERS,
        dspark_target_layer_ids=list(TARGET_LAYER_IDS),
        n_mtp_layers=3,
        index_topk=8,
        vocab_size=VOCAB,
        dspark_markov_rank=RANK,
        # Read by DeepseekV4Model.load_weights; no such tensors at toy scale.
        num_attention_heads=8,
        n_routed_experts=4,
    )
    vllm_config = VllmConfig()
    vllm_config.speculative_config = SimpleNamespace(
        num_speculative_tokens=5,
        draft_model_config=SimpleNamespace(hf_config=hf_config),
    )
    return vllm_config


@pytest.fixture(scope="module")
def dist_ctx():
    import torch.distributed as dist
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    if not dist.is_initialized():
        import portpicker

        port = portpicker.pick_unused_port()
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
    # GroupCoordinator construction consults the current vLLM config.
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)
    yield


def _use_cpu_platform(monkeypatch, request):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    request.getfixturevalue("dist_ctx")


@pytest.fixture()
def tiny_fc(monkeypatch, request):
    """A real DSparkDeepseekV4ForCausalLM at toy scale, layers stubbed."""
    _use_cpu_platform(monkeypatch, request)
    monkeypatch.setattr(dspark_mod, "DeepseekV4DecoderLayer", _StubDecoderLayer)
    vllm_config = _tiny_vllm_config()
    with set_current_vllm_config(vllm_config):
        return dspark_mod.DSparkDeepseekV4ForCausalLM(
            vllm_config=vllm_config, prefix=""
        )


def test_construction_wiring(tiny_fc):
    model = tiny_fc.model
    # vLLM prefixes continue the target's numbering so the KV-cache layer
    # names stay distinct; checkpoint params land at model.layers.{0,1,2}.
    assert [layer.prefix for layer in model.layers] == [
        f"model.layers.{N_TARGET_LAYERS + i}" for i in range(3)
    ]
    assert tiny_fc.get_draft_kv_cache_layer_names() == [
        f"model.layers.{N_TARGET_LAYERS + i}.attn.swa" for i in range(3)
    ]
    # Every draft layer runs the non-causal block mode of the SWA kernel.
    assert tiny_fc.get_draft_attn_causal() == [False, False, False]
    assert all(layer.attn.non_causal_multi_token_decode for layer in model.layers)
    # The draft shares the target's embedding and head.
    assert tiny_fc.has_own_embed_tokens is False
    assert tiny_fc.has_own_lm_head is False
    # main_proj consumes the concat of one hidden per tagged target layer.
    assert model.main_proj.weight.shape == (HIDDEN, HIDDEN * len(TARGET_LAYER_IDS))


def test_block_must_fit_the_sliding_window(monkeypatch, request):
    """The non-causal block mode only reaches keys inside the window's
    key-block walk, so construction rejects a block that cannot fit."""
    _use_cpu_platform(monkeypatch, request)
    monkeypatch.setattr(dspark_mod, "DeepseekV4DecoderLayer", _StubDecoderLayer)
    monkeypatch.setattr(_StubDecoderLayer, "window_size", 4)
    vllm_config = _tiny_vllm_config()
    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(ValueError, match="sliding window"),
    ):
        dspark_mod.DSparkDeepseekV4ForCausalLM(vllm_config=vllm_config, prefix="")


def test_weight_load_round_trip(tiny_fc):
    """mtp.* names -> rewrite -> params, reported under the wrapper's prefix."""
    model = tiny_fc.model
    ckpt = {
        "mtp.0.main_proj.weight": torch.randn_like(model.main_proj.weight),
        "mtp.0.main_norm.weight": torch.randn_like(model.main_norm.weight),
        "mtp.2.norm.weight": torch.randn_like(model.norm.weight),
        "mtp.2.hc_head_fn": torch.randn_like(model.hc_head_fn),
        "mtp.2.hc_head_base": torch.randn_like(model.hc_head_base),
        "mtp.2.hc_head_scale": torch.randn_like(model.hc_head_scale),
        "mtp.2.markov_head.markov_w1.weight": torch.randn(VOCAB, RANK),
        "mtp.2.markov_head.markov_w2.weight": torch.randn(VOCAB, RANK),
        # Dropped by design.
        "mtp.2.confidence_head.proj.weight": torch.randn(1, HIDDEN + RANK),
        # Owned by the target model; ignored here.
        "layers.0.attn.wkv.weight": torch.randn(4, 4),
    }
    loaded = tiny_fc.load_weights(iter(ckpt.items()))
    assert loaded == {
        "model.main_proj.weight",
        "model.main_norm.weight",
        "model.norm.weight",
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
        "model.markov_head.markov_w1.weight",
        "model.markov_head.markov_w2.weight",
    }
    torch.testing.assert_close(model.main_proj.weight, ckpt["mtp.0.main_proj.weight"])
    torch.testing.assert_close(model.norm.weight, ckpt["mtp.2.norm.weight"])
    torch.testing.assert_close(model.hc_head_fn, ckpt["mtp.2.hc_head_fn"])
    # Vocab-padded params round-trip their real rows.
    torch.testing.assert_close(
        model.markov_head.markov_w1.weight[:VOCAB],
        ckpt["mtp.2.markov_head.markov_w1.weight"],
    )
    torch.testing.assert_close(
        model.markov_head.markov_w2.weight[:VOCAB],
        ckpt["mtp.2.markov_head.markov_w2.weight"],
    )

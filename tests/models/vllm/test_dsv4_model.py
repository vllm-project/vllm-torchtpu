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
"""Unit tests for the DeepSeek-V4 model: registration and aux hidden-state
capture.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import DeepseekV4Config
from vllm.config import (
    CacheConfig,
    CompilationMode,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.utils import WeightsMapper

import vllm_torchtpu.models.vllm.deepseek_v4.model as model_mod
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_attention_op import (  # noqa: E501
    VllmDeepseekSparseSWABackend,
    VllmDeepseekV4SWACache,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_indexer import (
    VllmDeepseekV4Indexer,
)
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.deepseek_v4 import DeepseekV4ForCausalLM
from vllm_torchtpu.models.vllm.deepseek_v4.attention import VllmDeepseekV4MLAAttention


def test_deepseek_v4_model_registration():
    """Verify DeepseekV4ForCausalLM resolves from the vLLM ModelRegistry."""
    register_models()
    entry = ModelRegistry.models.get("DeepseekV4ForCausalLM")
    assert entry is not None, "DeepseekV4ForCausalLM is not registered"
    model_cls = entry.load_model_cls()
    assert model_cls is DeepseekV4ForCausalLM


def test_weight_loader_drops_mtp_after_mapping_and_keeps_plain_weights():
    model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(2, 3)
    model.model.mtp_adapter = nn.Linear(3, 2, bias=False)
    model.lm_head = nn.Linear(3, 2, bias=False)
    # A rename that introduces the skipped substring distinguishes filtering
    # mapped names from filtering only raw checkpoint names.
    model.hf_to_vllm_mapper |= WeightsMapper(
        orig_to_new_prefix={"draft.": "model.mtp."}
    )
    embedding = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    head = embedding + 10
    adapter = embedding + 20
    loaded = model.load_weights(
        iter(
            [
                ("mtp.0.weight", head),
                ("layers.0.mtp.weight", head),
                ("layers.0.aux_mtp.weight", head),
                ("draft.0.weight", head),
                ("embed.weight", embedding),
                ("head.weight", head),
                ("model.mtp_adapter.weight", adapter),
            ]
        )
    )
    assert loaded == {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.mtp_adapter.weight",
    }
    torch.testing.assert_close(model.model.embed_tokens.weight, embedding)
    torch.testing.assert_close(model.lm_head.weight, head)
    torch.testing.assert_close(model.model.mtp_adapter.weight, adapter)
    with pytest.raises(ValueError, match="not_in_model"):
        model.load_weights([("not_in_model.weight", head)])


# --------------------------------------------------------------------------
# Aux hidden-state capture (target-side input for eagle-family drafts)
# --------------------------------------------------------------------------

HIDDEN = 8
HC = 2
VOCAB = 16
N_LAYERS = 4
TOKENS = 3


class _StubDecoderLayer(nn.Module):
    """Stands in for DeepseekV4DecoderLayer; records what it returned."""

    def __init__(
        self, vllm_config, prefix="", topk_indices_buffer=None, aux_stream_list=None
    ):
        super().__init__()
        del vllm_config, topk_indices_buffer, aux_stream_list
        self.idx = int(prefix.rsplit(".", 1)[-1])
        self.outputs: list[tuple[torch.Tensor, ...]] = []

    def forward(
        self, x, positions, input_ids=None, post_mix=None, res_mix=None, residual=None
    ):
        del positions, input_ids, post_mix, res_mix, residual
        residual = x
        x = x * (self.idx + 2) + 1.0
        post_mix = torch.full_like(x, 0.5 * (self.idx + 1))
        res_mix = torch.full_like(x, 0.25 * (self.idx + 1))
        self.outputs.append((x, residual, post_mix, res_mix))
        return x, residual, post_mix, res_mix


class _FakePostOp:
    """Pure-torch stand-in for the MHC post op; counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, residual, post_mix, res_mix):
        self.calls += 1
        return x + residual * post_mix - res_mix


class _TorchRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _fake_collapse_head(hidden_states, hc_fn, hc_scale, hc_base, rms_norm_eps, hc_eps):
    del hc_fn, hc_scale, hc_base, rms_norm_eps, hc_eps
    return hidden_states.mean(dim=1)


def _tiny_hf_config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        hc_mult=HC,
        hc_eps=1e-5,
        rms_norm_eps=1e-6,
        num_hidden_layers=N_LAYERS,
        mlp_layer_types=["hash_moe"] + ["moe"] * (N_LAYERS - 1),
        index_topk=4,
        vocab_size=VOCAB,
    )


def _dsv4_attention_config(compress_ratio: int) -> tuple[VllmConfig, str]:
    hf_config = DeepseekV4Config(
        hidden_size=128,
        num_attention_heads=2,
        q_lora_rank=128,
        o_lora_rank=64,
        head_dim=512,
        qk_rope_head_dim=64,
        o_groups=1,
        sliding_window=128,
        num_hidden_layers=1,
        compress_ratios=[compress_ratio],
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        compress_rope_theta=10000.0,
        rope_parameters={"rope_type": "default"},
        index_topk=4,
        index_n_heads=2,
        index_head_dim=128,
    )
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        hf_config=hf_config, dtype=torch.bfloat16, max_model_len=256
    )
    # VllmConfig resolves this only when built with a model config; 5 is its
    # value for this max_model_len. The compress_ratio=4 indexer sizes its
    # prefill buffer from it.
    config.attention_config.sparse_mla_prefill_workspace_factor = 5
    config.cache_config = CacheConfig(block_size=1024, cache_dtype="fp8_e4m3")
    config.scheduler_config = SchedulerConfig(
        max_model_len=256,
        is_encoder_decoder=False,
        max_num_seqs=1,
        max_num_batched_tokens=128,
    )
    return config, "model.layers.0.self_attn"


@pytest.mark.parametrize("compress_ratio", [0, 4, 128])
def test_attention_constructs_tpu_swa_cache(monkeypatch, request, compress_ratio):
    """Exercise upstream construction, including its SWA backend argument."""
    from vllm.platforms import current_platform

    # CPU CI has no TPU device registered with PyTorch. Keep both the Gloo
    # group and the compressor's parameter allocations on CPU.
    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    request.getfixturevalue("dist_ctx")
    # Only skip Pallas op registration, which needs the worker's TPU mesh.
    monkeypatch.setattr(VllmDeepseekV4Indexer, "_build_indexer_op", lambda self: None)
    config, prefix = _dsv4_attention_config(compress_ratio)
    with set_current_vllm_config(config):
        attention = VllmDeepseekV4MLAAttention(config, prefix=prefix)
    cache = attention.swa_cache_layer
    assert isinstance(cache, VllmDeepseekV4SWACache)
    assert cache.get_attn_backend() is VllmDeepseekSparseSWABackend
    assert (
        config.compilation_config.static_forward_context[f"{prefix}.swa_cache"] is cache
    )
    spec = cache.get_kv_cache_spec(config)
    assert spec.block_size == 128
    assert spec.sliding_window == 128
    assert spec.dtype == torch.uint8


def test_attention_converts_rope_caches_when_in_graph_enabled(monkeypatch, request):
    """DSv4 rotary modules convert to in-graph cos/sin when `tpu_rope_in_graph` is on
    and stay tensor-backed when it is off, with bit-identical `cos_sin_at` rows."""
    from vllm.model_executor.layers import rotary_embedding as rope_lib
    from vllm.platforms import current_platform

    import vllm_torchtpu.layers.adapter.rotary_embedding  # noqa: F401
    from vllm_torchtpu import rope_in_graph, rope_in_graph_flag, rope_patch

    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    request.getfixturevalue("dist_ctx")
    monkeypatch.setattr(VllmDeepseekV4Indexer, "_build_indexer_op", lambda self: None)

    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    ref_config, ref_prefix = _dsv4_attention_config(4)
    ref_config.additional_config[rope_in_graph_flag.KEY] = False
    with set_current_vllm_config(ref_config):
        ref_attention = VllmDeepseekV4MLAAttention(ref_config, prefix=ref_prefix)

    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    config, prefix = _dsv4_attention_config(4)
    config.additional_config[rope_in_graph_flag.KEY] = True
    with set_current_vllm_config(config):
        attention = VllmDeepseekV4MLAAttention(config, prefix=prefix)

    assert isinstance(ref_attention.rotary_emb.cos_sin_cache, torch.Tensor)
    assert isinstance(ref_attention.indexer_rotary_emb.cos_sin_cache, torch.Tensor)

    # Tier 1 already converted DSv4 rotary modules at __init__, so Tier 2 is a no-op.
    assert (
        rope_patch.patch_in_graph_rotary_embedding(
            attention, allow_table_recovery=False
        )
        == 0
    )
    assert isinstance(
        attention.rotary_emb.cos_sin_cache, rope_in_graph.InGraphCosSinCache
    )
    assert "cos_sin_cache" not in attention.rotary_emb._buffers
    assert isinstance(
        attention.indexer_rotary_emb.cos_sin_cache, rope_in_graph.InGraphCosSinCache
    )
    assert "cos_sin_cache" not in attention.indexer_rotary_emb._buffers

    positions = torch.tensor([0, 3, 17, 63], dtype=torch.int32)
    for patched_rope, ref_rope in (
        (attention.rotary_emb, ref_attention.rotary_emb),
        (attention.indexer_rotary_emb, ref_attention.indexer_rotary_emb),
    ):
        assert patched_rope.cos_sin_cache.shape[-1] == ref_rope.cos_sin_cache.shape[-1]
        assert torch.equal(
            rope_in_graph.cos_sin_at(patched_rope, positions),
            rope_in_graph.cos_sin_at(ref_rope, positions),
        )


@pytest.fixture()
def tiny_fc(dist_ctx, monkeypatch):
    """A real DeepseekV4ForCausalLM at toy scale, kernels stubbed."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(model_mod, "DeepseekV4DecoderLayer", _StubDecoderLayer)
    monkeypatch.setattr(model_mod, "mhc_collapse_head", _fake_collapse_head)
    vllm_config = VllmConfig()
    # ``head_dtype`` is read by vLLM's LogitsProcessor at construction.
    vllm_config.model_config = SimpleNamespace(
        hf_config=_tiny_hf_config(), dtype=torch.float32, head_dtype=None
    )
    # Run the decorated backbone eagerly.
    vllm_config.compilation_config.mode = CompilationMode.NONE
    with set_current_vllm_config(vllm_config):
        fc = model_mod.DeepseekV4ForCausalLM(vllm_config=vllm_config, prefix="")
    # The post op is fetched lazily through ``mhc_post_op``; pre-seed the
    # instance slot it reads so no Pallas op is ever built.
    object.__setattr__(fc.model, "_mhc_post_op_instance", _FakePostOp())
    fc.model.norm = _TorchRMSNorm(fc.model.rms_norm_eps)
    return fc


def _forward(fc, tags):
    fc.set_aux_hidden_state_layers(tags)
    for layer in fc.model.layers:
        layer.outputs.clear()
    fc.model.mhc_post_op.calls = 0
    torch.manual_seed(1)
    ids = torch.arange(TOKENS)
    embeds = torch.randn(TOKENS, HIDDEN)
    return fc(ids, ids, inputs_embeds=embeds)


def _expected_aux(fc, tag):
    """The contract: tag ``i`` is the settled post after layer ``i - 1``,
    mean-collapsed over the hc streams."""
    x, residual, post_mix, res_mix = fc.model.layers[tag - 1].outputs[0]
    return _FakePostOp()(x, residual, post_mix, res_mix).mean(dim=1)


@pytest.mark.tpu_test
def test_untagged_forward_is_bare_tensor_and_unchanged(tiny_fc):
    out = _forward(tiny_fc, ())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (TOKENS, HIDDEN)
    # Exactly one post-op call: the fused settle of the last layer's post.
    assert tiny_fc.model.mhc_post_op.calls == 1
    x, residual, post_mix, res_mix = tiny_fc.model.layers[-1].outputs[0]
    settled = _FakePostOp()(x, residual, post_mix, res_mix)
    expected = tiny_fc.model.norm(settled.mean(dim=1))
    torch.testing.assert_close(out, expected)


@pytest.mark.tpu_test
def test_tagged_forward_captures_and_keeps_main_output(tiny_fc):
    for bad in [(0,), (N_LAYERS + 1,)]:
        with pytest.raises(ValueError, match="aux hidden-state layers"):
            tiny_fc.set_aux_hidden_state_layers(bad)

    baseline = _forward(tiny_fc, ())
    tags = (2, N_LAYERS)
    hidden, aux = _forward(tiny_fc, tags)
    assert len(aux) == len(tags)
    for got, tag in zip(aux, tags):
        assert got.shape == (TOKENS, HIDDEN)  # hc streams collapsed
        torch.testing.assert_close(got, _expected_aux(tiny_fc, tag))
    # Capturing does not perturb the main path, and a tagged final layer
    # reuses its reconstruction: one post-op call per tag, none extra.
    torch.testing.assert_close(hidden, baseline)
    assert tiny_fc.model.mhc_post_op.calls == len(tags)


def test_mla_attention_rope_wrappers_gather_positions():
    """qnorm_rope, kv_rope, and _o_proj look up cos_sin rows at positions."""
    num_tokens, rotary_dim = 4, 8
    positions = torch.tensor([0, 3, 7, 12], dtype=torch.int32)
    table = torch.randn(32, rotary_dim)
    expected = table.index_select(0, positions)

    attn = VllmDeepseekV4MLAAttention.__new__(VllmDeepseekV4MLAAttention)
    nn.Module.__init__(attn)
    attn.prefix = "model.layers.0.self_attn"
    attn.n_local_heads = 8
    attn.n_local_groups = 1
    attn.head_dim = 128
    attn.q_lora_rank = 4
    object.__setattr__(attn, "mla_attn", attn)
    attn.wo_a = SimpleNamespace(
        weight=torch.zeros((1, 1), dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones(1),
    )
    attn.wo_b = lambda z: z
    attn.rotary_emb = SimpleNamespace(cos_sin_cache=table, rotary_dim=rotary_dim)

    q_calls, kv_calls, o_calls = [], [], []
    object.__setattr__(
        attn, "_qnorm_rope_op_instance", lambda q, cs: q_calls.append(cs) or q
    )
    object.__setattr__(
        attn, "_kv_rope_op_instance", lambda kv, cs: kv_calls.append(cs) or kv
    )
    object.__setattr__(
        attn,
        "_o_proj_op_instance",
        lambda o, cs, *_a: o_calls.append(cs) or o[:, 0, :4],
    )

    attn._assert_o_proj_supported()
    attn.qnorm_rope(torch.zeros(num_tokens, 8, 128, dtype=torch.bfloat16), positions)
    attn.kv_rope(torch.zeros(num_tokens, 128, dtype=torch.bfloat16), positions)
    attn._o_proj(torch.zeros(num_tokens, 8, 128, dtype=torch.bfloat16), positions)
    assert torch.equal(q_calls[0], expected)
    assert torch.equal(kv_calls[0], expected)
    assert torch.equal(o_calls[0], expected)

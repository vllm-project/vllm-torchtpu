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
    prefix = "model.layers.0.self_attn"
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


@pytest.mark.cpu_test
def test_dsv4_mlp_and_moe_prequant_forwarding(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.deepseek_v4 import layers as layers_mod
    from vllm_torchtpu.models.vllm.deepseek_v4.layers import DeepseekV4MLP
    from vllm_torchtpu.models.vllm.deepseek_v4.moe import DeepseekV4MoE

    prequant_calls = []

    def fake_prequant(x_q, x_s, w_q, w_s):
        prequant_calls.append((x_q, x_s, w_q, w_s))
        return (x_q.float() * x_s) @ w_q.float() * w_s

    monkeypatch.setattr(layers_mod, "quantized_matmul_prequant", fake_prequant)
    monkeypatch.setattr(envs, "VLLM_DSV41_FUSED_SEAM_QUANT", True)

    mlp = DeepseekV4MLP.__new__(DeepseekV4MLP)
    nn.Module.__init__(mlp)
    mlp._prequant = [None, None]
    mlp.gate_up_proj = nn.Module()
    mlp.gate_up_proj.weight = nn.Parameter(torch.ones(8, 8))
    mlp.gate_up_proj.weight_scale = nn.Parameter(torch.full((8,), 0.5))
    mlp.gate_up_proj.bias = None
    mlp.act_fn = lambda g: g[:, :4] * g[:, 4:]
    mlp.down_proj = lambda a: (a @ torch.ones(4, 8), None)

    assert mlp.can_accept_prequant_hidden(token_sharded=False) is True
    assert mlp.can_accept_prequant_hidden(token_sharded=True) is False

    x = torch.ones(3, 8)
    x_fp8 = torch.full((3, 8), 2.0)
    x_scale = torch.full((3, 1), 0.25)

    # 1. Direct prequant call on DeepseekV4MLP
    out_direct = mlp(x, hidden_states_fp8=x_fp8, hidden_states_scale=x_scale)
    assert len(prequant_calls) == 1
    assert mlp._prequant == [None, None]
    assert out_direct.shape == (3, 8)

    # 2. Staged prequant forwarding via DeepseekV4MoE -> shared_experts
    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    nn.Module.__init__(moe)
    moe.tp_size = 1
    moe.shared_experts = mlp
    moe.gate = SimpleNamespace(tid2eid=None)
    moe.experts = lambda **kw: mlp(kw["hidden_states"])
    moe.experts.moe_config = SimpleNamespace(skip_final_all_reduce=False)

    assert moe.can_accept_prequant_hidden(token_sharded=False) is True
    assert moe.can_accept_prequant_hidden(token_sharded=True) is False

    out_moe = moe(x, hidden_states_fp8=x_fp8, hidden_states_scale=x_scale)
    assert len(prequant_calls) == 2
    assert mlp._prequant == [None, None]
    torch.testing.assert_close(out_moe, out_direct)


@pytest.mark.cpu_test
def test_dsv41_attention_qkv_prequant_and_compressor_token_sharded(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
        deepseek_v41_flash_compressor as comp_mod,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41 import attention as attn_mod

    prequant_calls = []

    def fake_prequant(x_q, x_s, w_q, w_s):
        prequant_calls.append((x_q, x_s, w_q, w_s))
        return (x_q.float() * x_s) @ w_q.float() * w_s

    monkeypatch.setattr(comp_mod, "quantized_matmul_prequant", fake_prequant)
    monkeypatch.setattr(envs, "VLLM_DSV41_FUSED_SEAM_QUANT", True)

    idx = comp_mod.VllmDeepseekV41Indexer.__new__(
        comp_mod.VllmDeepseekV41Indexer
    )
    nn.Module.__init__(idx)
    idx.wq_b = nn.Module()
    idx.wq_b.weight = nn.Parameter(torch.ones(8, 16))
    idx.wq_b.weight_scale = nn.Parameter(torch.full((16,), 0.5))
    idx.wq_b.bias = None
    assert idx.can_accept_prequant_query() is True

    q_fp8 = torch.full((4, 8), 2.0)
    q_s = torch.full((4, 128), 0.25)
    out_idx = idx._apply_wq_b(None, q_fp8, q_s)
    assert len(prequant_calls) == 1
    assert prequant_calls[0][1].shape == (4, 1)
    assert out_idx.shape == (4, 16)

    monkeypatch.setattr(
        attn_mod, "get_tensor_model_parallel_world_size", lambda: 8
    )
    attn = attn_mod.VllmDeepseekV41MLAAttention.__new__(
        attn_mod.VllmDeepseekV41MLAAttention
    )
    nn.Module.__init__(attn)
    attn.fused_wqa_wkv = nn.Module()
    attn.fused_wqa_wkv.weight = nn.Parameter(torch.ones(16, 8))
    attn.fused_wqa_wkv.weight_scale = nn.Parameter(torch.ones(8))
    attn.fused_wqa_wkv.bias = None
    attn.compressor = nn.Module()
    attn.indexer = SimpleNamespace(use_tp_indexer=lambda n: n % 8 == 0)
    attn._ced_global_from_encoder = False

    # Token-sharded prequant is accepted even on compressor layers (0 & 20).
    assert attn.can_accept_prequant_hidden(64, token_sharded=True) is True
    assert attn.can_accept_prequant_hidden(60, token_sharded=True) is False

    qr_kv = torch.zeros((4, 2048), dtype=torch.bfloat16)
    q_w = torch.ones((1536,), dtype=torch.bfloat16)
    kv_w = torch.ones((512,), dtype=torch.bfloat16)
    pos = torch.arange(4, dtype=torch.int32)
    cs = torch.zeros((16, 64), dtype=torch.float32)
    f_q_fp8, f_q_s, f_kv = attn_mod._fake_qkv_norm_rope_quant(
        qr_kv, q_w, kv_w, pos, cs
    )
    assert f_q_fp8.shape == (4, 1536) and f_q_fp8.dtype == torch.float8_e4m3fn
    assert f_q_s.shape == (4, 128) and f_q_s.dtype == torch.float32
    assert f_kv.shape == (4, 512) and f_kv.dtype == torch.bfloat16

    f_qkv = attn_mod._fake_qkv_norm_rope(qr_kv, q_w, kv_w, pos, cs)
    assert f_qkv.shape == (4, 2048) and f_qkv.dtype == torch.bfloat16
    f_qkv2, f_q_fp8_2, f_q_s_2 = attn_mod._fake_qkv_norm_rope_and_quant(
        qr_kv, q_w, kv_w, pos, cs
    )
    assert f_qkv2.shape == (4, 2048) and f_qkv2.dtype == torch.bfloat16
    assert f_q_fp8_2.shape == (4, 1536) and f_q_fp8_2.dtype == torch.float8_e4m3fn
    assert f_q_s_2.shape == (4, 128) and f_q_s_2.dtype == torch.float32

    wq_b = torch.zeros((1536, 4096), dtype=torch.float8_e4m3fn)
    wq_b_s = torch.ones((4096,), dtype=torch.float32)
    f_q = attn_mod._fake_wq_b_q_rope_prequant(
        f_q_fp8, f_q_s, pos, cs, wq_b, wq_b_s, head_dim=512
    )
    assert f_q.shape == (4, 8, 512) and f_q.dtype == torch.bfloat16

    gathered = []
    monkeypatch.setattr(
        comp_mod,
        "tp_all_gather",
        lambda x, dim=0: (gathered.append(x.shape), x.repeat(2, 1))[1],
    )
    comp = comp_mod.VllmDeepseekV41Compressor.__new__(
        comp_mod.VllmDeepseekV41Compressor
    )
    nn.Module.__init__(comp)
    comp.has_gate = False
    comp.fused_wkv_wgate = lambda h: h[:, :4]
    comp.norm = lambda z: z * 2.0
    lat = comp(torch.ones((2, 8)), torch.arange(4))
    assert gathered == [torch.Size([2, 4])]
    assert lat.shape == (4, 4)


@pytest.mark.cpu_test
def test_dsv4_moe_token_sharded_local_gate_and_packed_all_gather(monkeypatch):
    from vllm_torchtpu.layers.adapter import moe_routing
    from vllm_torchtpu.models.vllm.deepseek_v4 import moe as moe_mod
    from vllm_torchtpu.models.vllm.deepseek_v4.moe import DeepseekV4MoE

    gathered_shapes = []
    monkeypatch.setattr(
        moe_mod,
        "tp_all_gather",
        lambda x, dim=0: (gathered_shapes.append(x.shape), x.repeat(2, 1))[1],
    )
    monkeypatch.setattr(
        moe_mod,
        "tp_reduce_scatter",
        lambda x, dim=0: x[: x.shape[0] // 2],
    )

    routed_experts = SimpleNamespace(
        _precomputed_topk=[None, None],
        moe_config=SimpleNamespace(experts_per_token=2),
        custom_routing_function=None,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        use_grouped_topk=False,
        e_score_correction_bias=None,
        routed_scaling_factor=1.0,
    )

    gate_calls = []

    class _StubGate:
        tid2eid = None
        _skip_forward = [False]

        def __call__(self, x):
            if self._skip_forward[0]:
                return x[:, :1], None
            gate_calls.append(x.shape)
            return torch.arange(
                x.shape[0] * 4, dtype=torch.float32
            ).reshape(x.shape[0], 4), None

    class _StubExperts:
        is_monolithic = True
        do_naive_dispatch_combine = False
        moe_config = SimpleNamespace(skip_final_all_reduce=True)

        def __init__(self, gate, re):
            self.gate = gate
            self.routed_experts = re

        def __call__(self, *, hidden_states, router_logits, input_ids=None):
            del router_logits
            logits, _ = self.gate(hidden_states)
            assert logits.shape == (hidden_states.shape[0], 4)
            tw, tid = moe_routing.route(
                self.routed_experts, hidden_states, logits, input_ids=input_ids
            )
            assert tw.shape == (hidden_states.shape[0], 2)
            assert tid.shape == (hidden_states.shape[0], 2)
            assert tid.dtype == torch.int32
            return hidden_states * tw[:, :1].to(hidden_states.dtype)

    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    nn.Module.__init__(moe)
    moe.tp_size = 2
    moe.hidden_size = 8
    moe.n_routed_experts = 4
    moe.shared_experts = None
    moe.gate = _StubGate()
    moe.experts = _StubExperts(moe.gate, routed_experts)

    x_local = torch.ones((3, 8), dtype=torch.bfloat16)
    out = moe(
        x_local,
        return_token_sharded=True,
        input_is_token_sharded=True,
    )
    assert gate_calls == [torch.Size([6, 8])]
    assert gathered_shapes == [torch.Size([3, 8])]
    assert out.shape == (3, 8)


@pytest.mark.cpu_test
def test_dsv4_moe_token_sharded_local_shared_experts_and_prequant(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_routing
    from vllm_torchtpu.models.vllm.deepseek_v4 import layers as layers_mod
    from vllm_torchtpu.models.vllm.deepseek_v4 import moe as moe_mod
    from vllm_torchtpu.models.vllm.deepseek_v4.layers import DeepseekV4MLP
    from vllm_torchtpu.models.vllm.deepseek_v4.moe import DeepseekV4MoE

    monkeypatch.setattr(envs, "VLLM_DSV41_FUSED_SEAM_QUANT", True)
    monkeypatch.setattr(
        layers_mod, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        layers_mod,
        "tp_all_gather",
        lambda x, dim=0: torch.cat([x, x + 1], dim=dim),
    )

    mlp = DeepseekV4MLP.__new__(DeepseekV4MLP)
    nn.Module.__init__(mlp)
    mlp._prequant = [None, None]
    mlp.act_fn = lambda g: g[:, :4] * g[:, 4:]
    mlp.gate_up_proj = nn.Module()
    mlp.gate_up_proj.disable_tp = False
    mlp.gate_up_proj.input_size = 8
    mlp.gate_up_proj.weight = nn.Parameter(
        torch.ones((8, 4), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    mlp.gate_up_proj.weight_scale = nn.Parameter(
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        requires_grad=False,
    )
    mlp.gate_up_proj.bias = None
    mlp.down_proj = nn.Module()
    mlp.down_proj.output_size = 8
    mlp.down_proj.weight = nn.Parameter(
        torch.full((2, 8), 2.0, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    mlp.down_proj.weight_scale = nn.Parameter(
        torch.ones((8,), dtype=torch.float32), requires_grad=False
    )
    mlp.down_proj.bias = None

    mlp.process_weights_after_loading()
    assert mlp.has_tp_local_weights() is True
    assert mlp.can_accept_prequant_hidden(token_sharded=True) is True
    assert mlp._tp_gate_up_w.shape == (8, 8)
    assert mlp._tp_gate_up_s.shape == (8,)
    # Verify gate (w1) and up (w3) halves are gathered separately:
    torch.testing.assert_close(
        mlp._tp_gate_up_s,
        torch.tensor([1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]),
    )
    assert mlp._tp_down_w.shape == (4, 8)
    assert mlp._tp_down_s.shape == (8,)
    torch.testing.assert_close(mlp._tp_down_s, torch.full((8,), 2.0))
    # Shard 0 had scale 1.0 and weight 2.0 (0x40) -> rescaled to 1.0 under max scale 2.0;
    # Shard 1 had scale 2.0 and weight 2.25 (0x41) -> rescaled to 2.25 under max scale 2.0.
    torch.testing.assert_close(
        mlp._tp_down_w[:2].float(), torch.full((2, 8), 1.0)
    )
    torch.testing.assert_close(
        mlp._tp_down_w[2:].float(), torch.full((2, 8), 2.25)
    )

    prequant_calls = []
    down_calls = []
    monkeypatch.setattr(
        layers_mod,
        "quantized_matmul_prequant",
        lambda xq, xs, wq, ws: (
            prequant_calls.append((xq.shape, xs.shape, wq.shape, ws.shape)),
            torch.ones(
                (xq.shape[0], wq.shape[1]),
                dtype=torch.bfloat16,
                device=xq.device,
            ),
        )[1],
    )
    monkeypatch.setattr(
        layers_mod,
        "quantized_matmul",
        lambda act, wq, ws: (
            down_calls.append((act.shape, wq.shape, ws.shape)),
            torch.full(
                (act.shape[0], wq.shape[1]),
                2.0,
                dtype=act.dtype,
                device=act.device,
            ),
        )[1],
    )

    monkeypatch.setattr(
        moe_mod,
        "tp_all_gather",
        lambda x, dim=0: x.repeat(2, 1),
    )
    monkeypatch.setattr(
        moe_mod,
        "tp_reduce_scatter",
        lambda x, dim=0: x[: x.shape[0] // 2],
    )

    routed_experts = SimpleNamespace(
        moe_config=SimpleNamespace(experts_per_token=2),
        custom_routing_function=None,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        use_grouped_topk=False,
        e_score_correction_bias=None,
        routed_scaling_factor=1.0,
    )

    class _StubGate:
        tid2eid = None

        def __call__(self, x):
            return torch.ones((x.shape[0], 4), dtype=torch.float32), None

    class _StubExperts:
        is_monolithic = True
        do_naive_dispatch_combine = False
        moe_config = SimpleNamespace(skip_final_all_reduce=True)
        _skip_shared = [False]

        def __init__(self, gate, re, sh):
            self.gate = gate
            self.routed_experts = re
            self._sh = sh
            self.shared_ran_on_full = False

        def __call__(self, *, hidden_states, router_logits, input_ids=None):
            del router_logits
            if not self._skip_shared[0]:
                self.shared_ran_on_full = True
                _ = self._sh(hidden_states)
            logits, _ = self.gate(hidden_states)
            tw, _ = moe_routing.route(
                self.routed_experts, hidden_states, logits, input_ids=input_ids
            )
            return torch.ones_like(hidden_states) * tw[:, :1].to(
                hidden_states.dtype
            )

    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    nn.Module.__init__(moe)
    moe.tp_size = 2
    moe.hidden_size = 8
    moe.n_routed_experts = 4
    moe.shared_experts = mlp
    moe.gate = _StubGate()
    moe.experts = _StubExperts(moe.gate, routed_experts, mlp)
    assert moe.can_accept_prequant_hidden(token_sharded=True) is True

    x_local = torch.ones((3, 8), dtype=torch.bfloat16)
    x_fp8 = torch.ones((3, 8), dtype=torch.float8_e4m3fn)
    x_scale = torch.ones((3, 1), dtype=torch.float32)
    out = moe(
        x_local,
        return_token_sharded=True,
        hidden_states_fp8=x_fp8,
        hidden_states_scale=x_scale,
        input_is_token_sharded=True,
    )
    assert moe.experts.shared_ran_on_full is False
    assert moe.experts._skip_shared == [False]
    assert prequant_calls == [
        (
            torch.Size([3, 8]),
            torch.Size([3, 1]),
            torch.Size([8, 8]),
            torch.Size([8]),
        )
    ]
    assert down_calls == [
        (torch.Size([3, 4]), torch.Size([4, 8]), torch.Size([8]))
    ]
    assert out.shape == (3, 8)
    # Routed out is 0.5 (since 2 equal topk weights renormalize to 0.5 each),
    # and local shared_experts out is 2.0 -> sum is 2.5.
    torch.testing.assert_close(out, torch.full((3, 8), 2.5, dtype=torch.bfloat16))

    # Standalone dense DeepseekV4MLP forward with token_sharded=True dispatches
    # directly to forward_tp_local without tp_all_gather / tp_reduce_scatter.
    out_mlp_sharded = mlp(
        x_local,
        hidden_states_fp8=x_fp8,
        hidden_states_scale=x_scale,
        return_token_sharded=True,
        input_is_token_sharded=True,
    )
    torch.testing.assert_close(
        out_mlp_sharded, torch.full((3, 8), 2.0, dtype=torch.bfloat16)
    )
    assert len(prequant_calls) == 2

    # When skip_final_all_reduce is False, MoE cannot run local shared_experts,
    # so can_accept_prequant_hidden(token_sharded=True) returns False.
    moe.experts.moe_config.skip_final_all_reduce = False
    assert moe.can_accept_prequant_hidden(token_sharded=True) is False

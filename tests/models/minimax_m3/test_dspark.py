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
"""MiniMax M3 DSpark drafter on CPU: config, quantization, load and math.

The attention kernel is Pallas, so the forward tests swap the draft's
attention for a torch reference over one request's query block.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import (
    CacheConfig,
    CompilationMode,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.models import ModelRegistry, qwen3_dflash
from vllm.transformers_utils.configs.minimax_m3 import (
    MiniMaxM3Config,
    MiniMaxM3TextConfig,
)
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.layers import register_layers
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.models.minimax_m3.dspark import (
    ARCHITECTURE,
    MiniMaxM3DSparkForCausalLM,
    configure_dspark_draft,
)
from vllm_torchtpu.models.minimax_m3.hooks import TPUMiniMaxM3Config
from vllm_torchtpu.models.minimax_m3.model import (
    MiniMaxM3Model,
    MiniMaxM3SparseForCausalLM,
)
from vllm_torchtpu.models.minimax_m3.quant import (
    MiniMaxM3Mxfp8LinearMethod,
    MiniMaxM3QuantConfig,
    dequantize_mxfp8,
)

HIDDEN = 64
HEAD_DIM = 128
ROTARY_DIM = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
FFN = 64
LAYERS = 2
VOCAB = 96
RANK = 8
TARGET_LAYERS = 4
TARGET_LAYER_IDS = [1, 3]
THETA = 10000000.0
EPS = 1e-6
_CKPT = "language_model.model.dspark"


def _quant_dict() -> dict:
    return {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, 32],
        "group_size": 16,
        "moe_quant_algo": "NVFP4",
        "moe_quant_format": "nvfp4-pack-quantized",
        "ignored_layers": [
            f"{_CKPT}.confidence_head.proj",
            f"{_CKPT}.fc",
            f"{_CKPT}.markov_head.markov_w1",
            f"{_CKPT}.markov_head.markov_w2",
            "language_model.model.layers.3.block_sparse_moe.gate",
            "lm_head",
            "model.embed_tokens",
        ],
    }


def _draft_text_dict() -> dict:
    """The released dspark/config.json text_config, at toy scale."""
    dflash_config = {
        "num_hidden_layers": LAYERS,
        "num_target_layers": TARGET_LAYERS,
        "block_size": 4,
        "mask_token_id": VOCAB - 1,
        "target_layer_ids": TARGET_LAYER_IDS,
        "layer_types": ["sliding_attention"] * LAYERS,
        "sliding_window": 256,
    }
    return dict(
        architectures=[ARCHITECTURE],
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB,
        max_position_embeddings=1024,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        rotary_dim=ROTARY_DIM,
        partial_rotary_factor=0.5,
        hidden_act="swigluoai",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        moe_layer_freq=[0] * LAYERS,
        sparse_attention_config=None,
        dspark_block_size=4,
        dspark_markov_rank=RANK,
        dspark_noise_token_id=VOCAB - 1,
        dspark_use_dense_ffn=True,
        dspark_ffn_hidden_size=FFN,
        dspark_target_layer_ids=TARGET_LAYER_IDS,
        dflash_config=dflash_config,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        layer_types=["sliding_attention"] * LAYERS,
        sliding_window=256,
    )


def _upstream_draft_config() -> MiniMaxM3TextConfig:
    """The draft config as vLLM 0.32 leaves it: its MiniMax-MTP override
    promotes text_config, then the unknown DSpark architecture is tagged as
    DeepSeek-V4's draft."""
    config = MiniMaxM3TextConfig(**_draft_text_dict())
    config.quantization_config = _quant_dict()
    config.model_type = "deepseek_v4"
    config.architectures = ["DSparkDraftModel"]
    return config


def _spec_config(hf_config, method: str = "dspark") -> SimpleNamespace:
    spec = SimpleNamespace(
        method=method,
        num_speculative_tokens=4,
        enable_adaptive_verification=False,
        draft_model_config=SimpleNamespace(
            model="/ckpt/dspark", hf_config=hf_config, quantization="mxfp8"
        ),
        arch_updates=0,
    )

    def update_arch_():
        spec.arch_updates += 1

    spec.update_arch_ = update_arch_
    return spec


def test_dspark_is_registered() -> None:
    register_layers()
    entry = ModelRegistry.models[ARCHITECTURE]
    assert entry.load_model_cls() is MiniMaxM3DSparkForCausalLM
    # The draft's MXFP8 linears resolve to the TPU MiniMax config.
    assert get_quantization_config("tpu-mxfp8") is MiniMaxM3QuantConfig


@pytest.mark.parametrize("wrapped", [False, True])
def test_hook_accepts_dspark(wrapped: bool) -> None:
    from tests.models.minimax_m3.test_model import _hook_config

    if wrapped:
        # Without vLLM's text_config promotion the VL wrapper arrives as is.
        hf_config = MiniMaxM3Config(
            text_config=_draft_text_dict(), quantization_config=_quant_dict()
        )
    else:
        hf_config = _upstream_draft_config()
    spec = _spec_config(hf_config)
    config = _hook_config(speculative_config=spec)
    TPUMiniMaxM3Config.verify_and_update_config(config)

    draft = spec.draft_model_config
    assert isinstance(draft.hf_config, MiniMaxM3TextConfig)
    assert draft.hf_config.architectures == [ARCHITECTURE]
    assert draft.hf_config.model_type == "minimax_m3_dspark"
    assert draft.hf_config.dflash_config["causal"] is False
    assert draft.hf_config.markov_rank == RANK
    assert draft.hf_config.quantization_config["quant_method"] == "mxfp8"
    assert draft.quantization == "tpu-mxfp8"
    assert spec.arch_updates == 1
    # Non-causal, sliding-window draft layers.
    assert qwen3_dflash._resolve_layer_attention(draft.hf_config, 0) == (256, False)

    # The engine core reruns the hook on the already-fixed config.
    configure_dspark_draft(spec)
    assert draft.hf_config.architectures == [ARCHITECTURE]


def test_hook_rejects_other_drafts() -> None:
    with pytest.raises(NotImplementedError, match="dspark"):
        configure_dspark_draft(_spec_config(_upstream_draft_config(), "eagle3"))
    other = MiniMaxM3TextConfig(num_hidden_layers=1)
    with pytest.raises(NotImplementedError, match="dspark/"):
        configure_dspark_draft(_spec_config(other))


def test_quant_config_keeps_draft_heads_unquantized() -> None:
    config = MiniMaxM3QuantConfig.from_config(_quant_dict())
    for prefix in (
        "model.fc",
        "model.markov_head.markov_w2",
        "model.confidence_head.proj",
    ):
        assert config.is_layer_excluded(prefix), prefix
    assert not config.is_layer_excluded("model.layers.4.self_attn.qkv_proj")
    assert not config.is_layer_excluded("model.layers.4.mlp.down_proj")


def test_target_returns_residual_stream_aux_states() -> None:
    """Aux k is the residual stream entering layer k (hidden + residual of
    layer k-1), before the next norm; k = num_layers is the final one."""

    class _Layer(nn.Module):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.scale = scale

        def forward(self, positions, hidden_states, residual):
            residual = hidden_states if residual is None else hidden_states + residual
            return residual * self.scale, residual

    layers = [_Layer(float(i + 2)) for i in range(TARGET_LAYERS)]
    norm_inputs = []

    def norm(hidden_states, residual):
        norm_inputs.append(hidden_states + residual)
        return hidden_states + residual, None

    model = SimpleNamespace(
        layers=nn.ModuleList(layers),
        norm=norm,
        embed_input_ids=lambda ids: ids.float().unsqueeze(-1).repeat(1, 2),
        aux_hidden_state_layers=(),
    )
    target = SimpleNamespace(model=model)
    MiniMaxM3SparseForCausalLM.set_aux_hidden_state_layers(
        target, tuple(i + 1 for i in TARGET_LAYER_IDS)
    )
    with pytest.raises(ValueError):
        MiniMaxM3SparseForCausalLM.set_aux_hidden_state_layers(target, (5,))

    ids = torch.tensor([1, 2, 3])
    hidden, aux = MiniMaxM3Model.forward(model, ids, torch.arange(3))
    # Reference: carry (h, r) and record h + r after each layer.
    h, r, streams = ids.float().unsqueeze(-1).repeat(1, 2), None, []
    for layer in layers:
        h, r = layer(None, h, r)
        streams.append(h + r)
    assert len(aux) == len(TARGET_LAYER_IDS)
    torch.testing.assert_close(aux[0], streams[1])
    torch.testing.assert_close(aux[1], streams[3])
    torch.testing.assert_close(aux[1], norm_inputs[0])

    model.aux_hidden_state_layers = ()
    assert isinstance(MiniMaxM3Model.forward(model, ids, torch.arange(3)), torch.Tensor)


# --- Tiny drafter -------------------------------------------------------------


def _mxfp8(rows: int, cols: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    weight = torch.randn(rows, cols, generator=gen).to(torch.float8_e4m3fn)
    scale = torch.randint(120, 128, (rows, cols // 32), generator=gen)
    return {"weight": weight, "weight_scale_inv": scale.to(torch.uint8)}


def _checkpoint() -> dict[str, torch.Tensor]:
    """dspark/model-dspark-00001.safetensors names, at toy scale."""
    gen = torch.Generator().manual_seed(0)

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)

    fc_in = HIDDEN * len(TARGET_LAYER_IDS)
    ckpt = {
        f"{_CKPT}.fc.weight": randn(HIDDEN, fc_in, scale=fc_in**-0.5),
        f"{_CKPT}.hidden_norm.weight": randn(HIDDEN, scale=0.1),
        f"{_CKPT}.final_norm.weight": randn(HIDDEN, scale=0.1),
        f"{_CKPT}.markov_head.markov_w1.weight": randn(VOCAB, RANK),
        f"{_CKPT}.markov_head.markov_w2.weight": randn(VOCAB, RANK),
        f"{_CKPT}.confidence_head.proj.weight": randn(1, HIDDEN + RANK),
        f"{_CKPT}.confidence_head.proj.bias": randn(1),
    }
    shapes = {
        "self_attn.q_proj": (NUM_HEADS * HEAD_DIM, HIDDEN),
        "self_attn.k_proj": (NUM_KV_HEADS * HEAD_DIM, HIDDEN),
        "self_attn.v_proj": (NUM_KV_HEADS * HEAD_DIM, HIDDEN),
        "self_attn.o_proj": (HIDDEN, NUM_HEADS * HEAD_DIM),
        "mlp.gate_proj": (FFN, HIDDEN),
        "mlp.up_proj": (FFN, HIDDEN),
        "mlp.down_proj": (HIDDEN, FFN),
    }
    for layer in range(LAYERS):
        p = f"{_CKPT}.layers.{layer}.decoder_layer"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            ckpt[f"{p}.{norm}.weight"] = randn(HIDDEN, scale=0.1)
        for norm in ("q_norm", "k_norm"):
            ckpt[f"{p}.self_attn.{norm}.weight"] = randn(HEAD_DIM, scale=0.1)
        for proj, (rows, cols) in shapes.items():
            for key, tensor in _mxfp8(rows, cols, gen).items():
                ckpt[f"{p}.{proj}.{key}"] = tensor
    return ckpt


class _ModelConfigStub(SimpleNamespace):
    """Target model config for layer construction; other fields unset."""

    def __getattr__(self, name: str) -> None:
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    def get_total_num_hidden_layers(self) -> int:
        return TARGET_LAYERS

    def get_vocab_size(self) -> int:
        return VOCAB


@pytest.fixture
def tiny_draft(request, monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    request.getfixturevalue("dist_ctx")
    spec = _spec_config(_upstream_draft_config())
    configure_dspark_draft(spec)
    # vLLM resolves the draft's quantization ("tpu-mxfp8") from its
    # ModelConfig; build the same config from the checkpoint dict here.
    monkeypatch.setattr(
        qwen3_dflash,
        "get_draft_quant_config",
        lambda _: MiniMaxM3QuantConfig.from_config(_quant_dict()),
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = _ModelConfigStub(
        hf_config=MiniMaxM3TextConfig(), dtype=torch.bfloat16, max_model_len=1024
    )
    vllm_config.cache_config = CacheConfig(block_size=128)
    vllm_config.scheduler_config = SchedulerConfig(
        max_model_len=1024,
        is_encoder_decoder=False,
        max_num_seqs=1,
        max_num_batched_tokens=128,
    )
    vllm_config.speculative_config = spec
    monkeypatch.setattr(MiniMaxM3QuantConfig, "vllm_config", vllm_config)
    vllm_config.compilation_config.mode = CompilationMode.NONE
    resolve_kv_cache_layout(vllm_config, [["LBNHC", "LBHNC"]])
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vllm_config):
            model = MiniMaxM3DSparkForCausalLM(vllm_config=vllm_config, prefix="")
    finally:
        torch.set_default_dtype(torch.float32)
    return model


def _load(model: MiniMaxM3DSparkForCausalLM, ckpt: dict[str, torch.Tensor]) -> None:
    model.load_weights(ckpt.items())
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)


def _gemma(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return (x * (1.0 + weight.float())).to(torch.bfloat16)


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Partial NeoX RoPE over the first ROTARY_DIM dims of each head."""
    inv_freq = THETA ** (-torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM)
    freqs = positions.float()[:, None] * inv_freq
    cos, sin = freqs.cos()[:, None], freqs.sin()[:, None]
    x = x.float()
    half = ROTARY_DIM // 2
    x1, x2, rest = x[..., :half], x[..., half:ROTARY_DIM], x[..., ROTARY_DIM:]
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], dim=-1)
    return out.to(torch.bfloat16)


def _weight(ckpt: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return dequantize_mxfp8(ckpt[f"{name}.weight"], ckpt[f"{name}.weight_scale_inv"])


def _reference_kv(ckpt, layer: int, context: torch.Tensor, positions: torch.Tensor):
    p = f"{_CKPT}.layers.{layer}.decoder_layer.self_attn"
    k = (context @ _weight(ckpt, f"{p}.k_proj").T).view(-1, NUM_KV_HEADS, HEAD_DIM)
    v = (context @ _weight(ckpt, f"{p}.v_proj").T).view(-1, NUM_KV_HEADS, HEAD_DIM)
    return _rope(_gemma(k, ckpt[f"{p}.k_norm.weight"]), positions), v


def _reference_forward(ckpt, embeds: torch.Tensor, positions: torch.Tensor):
    """Draft block forward: non-causal attention within the block only."""
    h = embeds
    for layer in range(LAYERS):
        p = f"{_CKPT}.layers.{layer}.decoder_layer"
        x = _gemma(h, ckpt[f"{p}.input_layernorm.weight"])
        q = (x @ _weight(ckpt, f"{p}.self_attn.q_proj").T).view(-1, NUM_HEADS, HEAD_DIM)
        q = _rope(_gemma(q, ckpt[f"{p}.self_attn.q_norm.weight"]), positions)
        k, v = _reference_kv(ckpt, layer, x, positions)
        group = NUM_HEADS // NUM_KV_HEADS
        k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * HEAD_DIM**-0.5
        o = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
        o = o.to(torch.bfloat16).reshape(-1, NUM_HEADS * HEAD_DIM)
        h = h + o @ _weight(ckpt, f"{p}.self_attn.o_proj").T
        x = _gemma(h, ckpt[f"{p}.post_attention_layernorm.weight"])
        gate = (x @ _weight(ckpt, f"{p}.mlp.gate_proj").T).float().clamp(max=7.0)
        up = (x @ _weight(ckpt, f"{p}.mlp.up_proj").T).float().clamp(-7.0, 7.0)
        act = (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(torch.bfloat16)
        h = h + act @ _weight(ckpt, f"{p}.mlp.down_proj").T
    return _gemma(h, ckpt[f"{_CKPT}.final_norm.weight"])


class _BlockAttention(nn.Module):
    """One request's query block attending to itself, no causal mask."""

    def forward(self, q, k, v):
        q = q.view(q.shape[0], NUM_HEADS, HEAD_DIM)
        group = NUM_HEADS // NUM_KV_HEADS
        k = k.view(k.shape[0], NUM_KV_HEADS, HEAD_DIM).repeat_interleave(group, 1)
        v = v.view(v.shape[0], NUM_KV_HEADS, HEAD_DIM).repeat_interleave(group, 1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).float(),
            k.transpose(0, 1).float(),
            v.transpose(0, 1).float(),
        )
        return out.transpose(0, 1).reshape(q.shape[0], -1).to(torch.bfloat16)


def test_tiny_drafter_loads(tiny_draft) -> None:
    model = tiny_draft
    ckpt = _checkpoint()
    attn = model.model.layers[0].self_attn
    assert isinstance(attn.qkv_proj.quant_method, MiniMaxM3Mxfp8LinearMethod)
    assert isinstance(
        model.model.layers[0].mlp.down_proj.quant_method, MiniMaxM3Mxfp8LinearMethod
    )
    assert isinstance(model.model.fc.quant_method, VllmUnquantizedLinearMethod)
    # Layer names continue the target's numbering, as their KV caches do.
    assert model.get_draft_kv_cache_layer_names() == [
        f"model.layers.{TARGET_LAYERS + i}.self_attn.attn" for i in range(LAYERS)
    ]
    assert model.get_draft_attn_causal() == [False] * LAYERS
    assert attn.attn.sliding_window == 256
    assert attn.rotary_emb.rotary_dim == ROTARY_DIM

    _load(model, ckpt)
    # Embedding and LM head come from the target.
    assert not model.has_own_embed_tokens
    assert not model.has_own_lm_head
    torch.testing.assert_close(
        model.model.markov_head.markov_w1.weight,
        ckpt[f"{_CKPT}.markov_head.markov_w1.weight"],
    )
    assert model.model.confidence_head is not None
    torch.testing.assert_close(
        model.model.norm.weight, ckpt[f"{_CKPT}.final_norm.weight"]
    )
    expected = torch.cat(
        [
            _weight(ckpt, f"{_CKPT}.layers.1.decoder_layer.self_attn.{proj}")
            for proj in ("q_proj", "k_proj", "v_proj")
        ]
    )
    torch.testing.assert_close(
        model.model.layers[1].self_attn.qkv_proj.weight, expected.T
    )


def test_tiny_drafter_forward_matches_reference(tiny_draft) -> None:
    model = tiny_draft
    ckpt = _checkpoint()
    _load(model, ckpt)
    gen = torch.Generator().manual_seed(1)
    embed = torch.randn(VOCAB, HIDDEN, generator=gen).to(torch.bfloat16)
    # Padded to a multiple of 64 rows.
    model.model.embed_tokens.weight.data[:VOCAB].copy_(embed)
    for layer in model.model.layers:
        layer.self_attn.attn = _BlockAttention()

    # Anchor (last accepted token) followed by noise tokens.
    input_ids = torch.tensor([7, VOCAB - 1, VOCAB - 1, VOCAB - 1])
    positions = torch.arange(20, 24)
    with set_current_vllm_config(VllmConfig()):
        hidden = model.model(input_ids, positions)
    expected = _reference_forward(ckpt, embed[input_ids], positions)
    torch.testing.assert_close(hidden.float(), expected.float(), atol=0.1, rtol=0.05)

    # Markov chain hooks and confidence run on the drafted hidden states.
    markov = model.markov_embed(input_ids[:1])
    assert markov.shape == (1, RANK)
    confidence = model.compute_confidence(hidden[:1], markov)
    assert confidence.shape == (1,) and 0.0 <= confidence.item() <= 1.0


def test_context_kv_from_target_aux_states(tiny_draft) -> None:
    model = tiny_draft
    ckpt = _checkpoint()
    _load(model, ckpt)
    written = []
    for layer in model.model.layers:
        layer.self_attn.attn.impl = SimpleNamespace(
            forward=lambda **kwargs: written.append(kwargs)
        )
    gen = torch.Generator().manual_seed(2)
    num_tokens = 5
    aux = [
        torch.randn(num_tokens, HIDDEN, generator=gen).to(torch.bfloat16)
        for _ in TARGET_LAYER_IDS
    ]
    positions = torch.arange(num_tokens) + 3
    metadata = tuple(f"metadata{i}" for i in range(LAYERS))
    context = model.tpu_precompute_and_store_context_kv(
        torch.cat(aux, dim=-1), positions, metadata
    )

    fc = ckpt[f"{_CKPT}.fc.weight"]
    expected_context = _gemma(
        torch.cat(aux, dim=-1) @ fc.T, ckpt[f"{_CKPT}.hidden_norm.weight"]
    )
    torch.testing.assert_close(context, expected_context)
    assert [w["attn_metadata"] for w in written] == list(metadata)
    for layer, write in enumerate(written):
        k, v = _reference_kv(ckpt, layer, expected_context, positions)
        assert write["layer"] is model.model.layers[layer].self_attn.attn
        torch.testing.assert_close(
            write["key"].float(), k.float(), atol=0.05, rtol=0.02
        )
        torch.testing.assert_close(write["value"], v)
        assert write["query"].shape == (num_tokens, NUM_HEADS, HEAD_DIM)

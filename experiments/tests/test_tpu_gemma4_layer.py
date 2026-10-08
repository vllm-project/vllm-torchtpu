"""``tpu_gemma4._layer_forward`` vs the HF ``Gemma4TextDecoderLayer`` (CPU).

A tiny random Gemma-4 text config (hidden 64, 2 heads, head_dim 32, seq 16)
is run through the eager HF decoder layer in bfloat16 and through the JAX
port on the JAX CPU backend with the same weights, RoPE tables and masks.
Both layer types (sliding / full attention) and both value paths (v_proj /
``attention_k_eq_v``) are covered.

The port keeps the attention probabilities unnormalised in bfloat16 and
divides after the PV matmul, so outputs agree to a few bfloat16 ulps, not
bit-exactly; the tolerances below are set accordingly.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SPARSEGPT_JAX_CACHE", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from sparsity import tpu_gemma4, tpu_sparsegpt  # noqa: E402

_check_supported = tpu_gemma4._check_supported
_layer_forward = tpu_gemma4._layer_forward
_layer_params = tpu_gemma4._layer_params
_layer_spec = tpu_gemma4._layer_spec
_to_np = tpu_sparsegpt._to_np
_to_torch = tpu_sparsegpt._to_torch

mg = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, FFN, SEQ, WINDOW = 64, 2, 1, 32, 128, 16, 8
# Outputs are O(1) with |out| < 16; measured on the CPU backend: relative
# Frobenius error 0.55-0.74% (bf16 eps = 0.78%), max |diff| <= 0.0625.
MAX_ATOL = 0.125  # two bf16 ulps at |x| in [8, 16)
REL_FRO = 2e-2


def _tiny_config(k_eq_v: bool):
    cfg = mg.Gemma4TextConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_global_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        global_head_dim=HEAD_DIM,
        intermediate_size=FFN,
        num_hidden_layers=2,
        vocab_size=256,
        sliding_window=WINDOW,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size_per_layer_input=0,
        attention_k_eq_v=k_eq_v,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _randomise(layer: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) / p.shape[-1]**0.5)


def _build(layer_idx: int, k_eq_v: bool, seed: int):
    cfg = _tiny_config(k_eq_v)
    torch.manual_seed(seed)
    layer = mg.Gemma4TextDecoderLayer(cfg, layer_idx)
    _randomise(layer, seed)
    layer = layer.to(torch.bfloat16).eval()
    x = torch.randn(1, SEQ, HIDDEN).to(torch.bfloat16)
    pos = torch.arange(SEQ).unsqueeze(0)
    layer_type = cfg.layer_types[layer_idx]
    rot = mg.Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rot(x, pos, layer_type)
    if layer_type == "sliding_attention":
        make_mask = mg.create_sliding_window_causal_mask
    else:
        make_mask = mg.create_causal_mask
    mask = make_mask(config=cfg,
                     inputs_embeds=x,
                     attention_mask=None,
                     past_key_values=None,
                     position_ids=pos)
    return cfg, layer, x, pos, cos, sin, mask


# attention_k_eq_v only changes full-attention layers (HF: `and not sliding`).
@pytest.mark.parametrize("layer_idx,k_eq_v", [(0, False), (1, False),
                                              (1, True)],
                         ids=["sliding", "full-v_proj", "full-k_eq_v"])
def test_layer_forward_matches_hf_decoder_layer(layer_idx, k_eq_v):
    assert jax.default_backend() == "cpu"
    cfg, layer, x, pos, cos, sin, mask = _build(layer_idx, k_eq_v, seed=0)
    with torch.no_grad():
        ref = layer(x,
                    position_embeddings=(cos, sin),
                    attention_mask=mask,
                    position_ids=pos,
                    shared_kv_states={})
    ref = ref[0] if isinstance(ref, tuple) else ref
    assert ref.shape == x.shape and ref.dtype == torch.bfloat16

    spec = _layer_spec(cfg, layer_idx)
    assert spec["sliding_window"] == (WINDOW if layer_idx == 0 else 0)
    params = _layer_params(layer)
    assert ("self_attn.v_proj" in params) == (layer.self_attn.v_proj
                                              is not None)
    assert ("self_attn.v_proj" in params) == (not (k_eq_v and layer_idx))
    out, acts = _layer_forward(_to_np(x), params, _to_np(cos[0]),
                               _to_np(sin[0]), spec)
    got = _to_torch(np.asarray(out))
    assert got.shape == x.shape and got.dtype == torch.bfloat16

    ref32, got32 = ref.float(), got.float()
    diff = (got32 - ref32).abs()
    rel = (diff.norm() / ref32.norm()).item()
    assert diff.max().item() <= MAX_ATOL, f"max |jax - hf| = {diff.max():.4f}"
    assert rel <= REL_FRO, f"relative Frobenius error {rel:.4f}"

    # Linear inputs captured for the Hessians have the expected widths.
    hd = spec["head_dim"]
    assert acts["attn_in"].shape == (1, SEQ, HIDDEN)
    assert acts["o_in"].shape == (1, SEQ, HEADS * hd)
    assert acts["mlp_in"].shape == (1, SEQ, HIDDEN)
    assert acts["down_in"].shape == (1, SEQ, FFN)


def test_sliding_layer_ignores_keys_outside_window():
    """Changing a token more than WINDOW positions back must not leak in."""
    cfg, layer, x, pos, cos, sin, mask = _build(0, False, seed=1)
    spec = _layer_spec(cfg, 0)
    params = _layer_params(layer)
    out_a, _ = _layer_forward(_to_np(x), params, _to_np(cos[0]),
                              _to_np(sin[0]), spec)
    x2 = x.clone()
    x2[0, 0] = torch.randn(HIDDEN).to(torch.bfloat16)
    out_b, _ = _layer_forward(_to_np(x2), params, _to_np(cos[0]),
                              _to_np(sin[0]), spec)
    a, b = np.asarray(out_a), np.asarray(out_b)
    assert not np.array_equal(a[0, 1], b[0, 1])  # inside the window
    assert np.array_equal(a[0, WINDOW:], b[0, WINDOW:])  # outside it


def test_check_supported_rejects_unported_variants():
    cfg = _tiny_config(False)
    _check_supported(cfg)
    cfg.enable_moe_block = True
    with pytest.raises(NotImplementedError, match="enable_moe_block"):
        _check_supported(cfg)

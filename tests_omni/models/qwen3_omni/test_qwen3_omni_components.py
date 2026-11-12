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
"""Component-wise TPU vs CPU tests for the Qwen3-Omni thinker audio encoder."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import MethodType, SimpleNamespace

import pytest
import torch
from transformers import Qwen3OmniMoeAudioEncoderConfig
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

from tests_omni.omni_test_utils import assert_parity
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_STATIC_BUCKETING_PATCHED,
    _pad_rows,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = [pytest.mark.omni]

# Production chunking geometry (num_mel_bins 128, n_window 50 -> 100-frame
# windows, n_window_infer 800, three stride-2 convolutions) on a tiny encoder.
AUDIO_CONFIG = Qwen3OmniMoeAudioEncoderConfig(
    d_model=256,
    encoder_layers=1,
    encoder_attention_heads=4,
    encoder_ffn_dim=512,
    output_dim=128,
    downsample_hidden_size=32,
)

# 800 frames (8 s) is the exact 100-frame multiple the patch fixes; 801 takes
# the partial-window branch across two attention segments; 3200 hits the upper
# edge of the 32-window bucket (512 padded tokens); 3201 crosses into the
# 40-window fallback (640 padded tokens).
PARITY_LENGTHS = [800, 801, 3200, 3201]

# (max relative L2, max abs) vs float32 CPU reference (~2.5x measured on TPU v6e).
END_TO_END_LIMITS = (1.5e-2, 4.5e-3)
COMPONENT_LIMITS = {
    "conv stack + positional embedding": (5e-3, 1.5e-2),
    "layers.0.self_attn": (9e-3, 3.5e-3),
    "layers.0": (8e-3, 3e-2),
    "ln_post + proj": (1.2e-2, 4.5e-3),
}


def _init_linear_weights(module: torch.nn.Module, seed: int) -> None:
    """Seed vLLM's parallel linear layers, which ``torch.empty`` leaves to a loader."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for submodule in module.modules():
            if isinstance(submodule, LinearBase):
                for param in submodule.parameters(recurse=False):
                    if param.dim() > 1:
                        param.copy_(
                            torch.randn(param.shape, generator=generator) * 0.02
                        )
                    else:
                        param.zero_()


def _use_native_ops_on_cpu(module: torch.nn.Module) -> None:
    """Bind CPU-compatible reference methods on vLLM CustomOp instances."""

    def _attention(query, key, value, cu_seqlens=None, max_seqlen=None, **_):
        q, k, v = (t.transpose(1, 2).double() for t in (query, key, value))
        bounds = cu_seqlens.tolist() if cu_seqlens is not None else [0, q.shape[2]]
        scale = q.shape[-1] ** -0.5
        out = torch.empty_like(q)
        for start, end in zip(bounds[:-1], bounds[1:]):
            scores = q[:, :, start:end] @ k[:, :, start:end].transpose(-1, -2) * scale
            out[:, :, start:end] = scores.softmax(dim=-1) @ v[:, :, start:end]
        return out.transpose(1, 2).to(query.dtype)

    for submodule in module.modules():
        if isinstance(submodule, MMEncoderAttention):
            submodule._forward_method = _attention
        elif isinstance(submodule, CustomOp):
            submodule._forward_method = submodule.forward_native


def _sample_features(n_frames: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(n_frames)
    return (
        torch.randn(AUDIO_CONFIG.num_mel_bins, n_frames, generator=generator)
        .to(torch.bfloat16)
        .float()
    )


def _encode(
    encoder: Qwen3OmniMoeAudioEncoder,
    n_frames: int | list[int],
    forward: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    lengths = [n_frames] if isinstance(n_frames, int) else list(n_frames)
    features = torch.cat([_sample_features(n) for n in lengths], dim=1)
    param = next(encoder.parameters())
    device, dtype = param.device, param.dtype
    feature_lens = torch.tensor(lengths, dtype=torch.long, device=device)
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    run = encoder if forward is None else forward.__get__(encoder)
    with torch.no_grad():
        out = run(features.to(device=device, dtype=dtype), feature_lens, aftercnn_lens)
    assert feature_lens.device == device
    return out


def _check(actual: torch.Tensor, reference: torch.Tensor, name: str, limits) -> None:
    diff = actual.detach().cpu().double() - reference.detach().cpu().double()
    rel_l2 = float(diff.norm() / reference.detach().cpu().double().norm())
    print(
        f"parity {name:<40} rel_l2={rel_l2:.3e} max_abs={float(diff.abs().max()):.3e}"
    )
    max_rel_l2, max_abs = limits
    assert_parity(actual, reference, name=name, max_rel_l2=max_rel_l2, max_abs=max_abs)


@pytest.fixture(scope="module")
def encoders(
    omni_ctx: None,
) -> Iterator[
    tuple[
        Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoder, Callable[..., torch.Tensor]
    ]
]:
    if not torch.accelerator.is_available():
        pytest.fail("Needs a TPU: the chunk-length failure is in TPU float division.")

    cls = Qwen3OmniMoeAudioEncoder
    upstream_forward = cls.forward
    assert not getattr(upstream_forward, _AUDIO_STATIC_BUCKETING_PATCHED, False)
    own = {
        n: cls.__dict__.get(n)
        for n in (
            "forward",
            "_tpu_orig_forward",
            "prepare_encoder_metadata",
            "forward_with_encoder_metadata",
        )
    }
    try:
        config = SimpleNamespace(architectures=["Qwen3OmniMoeForConditionalGeneration"])
        assert maybe_patch_qwen3_omni_thinker(config) is True
        assert getattr(cls.forward, _AUDIO_STATIC_BUCKETING_PATCHED, False)
        assert cls._tpu_orig_forward is upstream_forward

        torch.manual_seed(0)
        cpu = cls(AUDIO_CONFIG).eval()
        # Bind the unpatched upstream forward on the CPU reference so parity
        # checks compare the patched bucketed TPU forward against upstream.
        cpu.forward = MethodType(upstream_forward, cpu)
        _init_linear_weights(cpu, seed=0)
        assert all(torch.isfinite(p).all() for p in cpu.parameters())
        _use_native_ops_on_cpu(cpu)

        tpu = cls(AUDIO_CONFIG).eval()
        tpu.load_state_dict(cpu.state_dict())
        tpu = tpu.to(torch.device("tpu")).to(torch.bfloat16)
        yield tpu, cpu, upstream_forward
    finally:
        for name, value in own.items():
            if value is None:
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, value)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="vLLM-Omni 0.32.0rc1 takes ceil(feature_lens / 100) on the device, "
    "which TPU rounds up to 9 windows for 800 frames; the split then fails",
)
def test_upstream_forward_miscounts_exact_window_multiples(encoders) -> None:
    tpu, _, upstream_forward = encoders
    try:
        out = _encode(tpu, 800, forward=upstream_forward)
    except RuntimeError as exc:
        assert "split_with_sizes expects split_sizes to sum exactly to 800" not in str(
            exc
        ), str(exc)
        raise
    assert out.shape[0] == int(_get_feat_extract_output_lengths(800))


@pytest.mark.parametrize("n_frames", PARITY_LENGTHS)
def test_audio_encoder_matches_cpu_reference(encoders, n_frames: int) -> None:
    tpu, cpu, _ = encoders
    captured: dict[str, tuple[tuple, dict, torch.Tensor]] = {}

    def _capture(name):
        def hook(module, args, kwargs, output):
            captured[name] = (args, kwargs, output)

        return hook

    hooks = [
        cpu.layers[0].register_forward_hook(_capture("layer"), with_kwargs=True),
        cpu.layers[0].self_attn.register_forward_hook(
            _capture("attn"), with_kwargs=True
        ),
        tpu.layers[0].register_forward_hook(_capture("tpu_layer"), with_kwargs=True),
    ]
    try:
        reference = _encode(cpu, n_frames)
        out = _encode(tpu, n_frames)
    finally:
        for hook in hooks:
            hook.remove()
    expected_tokens = int(_get_feat_extract_output_lengths(n_frames))
    assert out.device.type == "tpu"
    assert out.shape == (expected_tokens, AUDIO_CONFIG.output_dim)
    assert out.shape == reference.shape

    def _on_tpu(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.device("tpu"), torch.bfloat16)

    (layer_in, _, _), _, layer_out = captured["layer"]
    _, attn_kwargs, attn_out = captured["attn"]
    (tpu_layer_in, tpu_cu_seqlens), tpu_layer_kwargs, _ = captured["tpu_layer"]
    padded_tokens = 512 if n_frames <= 3200 else 640
    assert tpu_layer_in.shape == (padded_tokens, AUDIO_CONFIG.d_model)
    assert tpu_cu_seqlens.shape == (128,)
    assert tpu_layer_kwargs.get("max_seqlen") is None

    with torch.no_grad():
        tpu_attn_out = tpu.layers[0].self_attn(
            _pad_rows(_on_tpu(attn_kwargs["hidden_states"]), padded_tokens),
            tpu_cu_seqlens,
            max_seqlen=None,
        )[:expected_tokens]
        tpu_layer_out = tpu.layers[0](
            _pad_rows(_on_tpu(layer_in), padded_tokens),
            tpu_cu_seqlens,
            max_seqlen=None,
        )[:expected_tokens]
        h = tpu.ln_post(_on_tpu(layer_out))
        tpu_proj_out = tpu.proj2(tpu.act(tpu.proj1(h)[0]))[0]

    tag = f" ({n_frames} frames)"
    for name, actual, expected in [
        (
            "conv stack + positional embedding",
            tpu_layer_in[:expected_tokens],
            layer_in,
        ),
        ("layers.0.self_attn", tpu_attn_out, attn_out),
        ("layers.0", tpu_layer_out, layer_out),
        ("ln_post + proj", tpu_proj_out, reference),
    ]:
        _check(actual, expected, name + tag, COMPONENT_LIMITS[name])
    _check(out, reference, "audio encoder end to end" + tag, END_TO_END_LIMITS)


def test_audio_encoder_multi_clip_batch_matches_cpu_reference(encoders) -> None:
    """Two-clip packed batch [800, 801] via both forward and metadata APIs."""
    tpu, cpu, _ = encoders
    lengths = [800, 801]
    reference = _encode(cpu, lengths)
    out = _encode(tpu, lengths)
    _check(out, reference, "audio encoder batch [800, 801]", END_TO_END_LIMITS)

    features = torch.cat([_sample_features(n) for n in lengths], dim=1).to(
        device=torch.device("tpu"), dtype=torch.bfloat16
    )
    metadata = tpu.prepare_encoder_metadata(lengths)
    with torch.no_grad():
        out_meta = tpu.forward_with_encoder_metadata(features, metadata)
    assert metadata["aftercnn_lens"] == [
        int(_get_feat_extract_output_lengths(n)) for n in lengths
    ]
    assert torch.equal(out_meta, out)

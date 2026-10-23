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
"""Component-wise TPU vs CPU parity tests for the LTX-2 DiT and text connectors.

Each vllm-omni component is exercised directly on TPU (without end-to-end
pipeline orchestration) and compared in the same process against its
HuggingFace diffusers / transformers counterpart on CPU:

- ``LTX2AudioVideoRotaryPosEmbed`` and coordinate builders
  (``prepare_video_coords``, ``prepare_audio_coords``) for video and audio.
- Tiny ``LTX2VideoTransformer3DModel`` across ``t2v``, ``stg``, and
  ``modality`` forward variants.
- Tiny ``LTX2TextConnectors`` with ``_install_connector_attention`` on
  left-padded prompts.
- Tiny ``DistributedAutoencoderKLLTX2Video`` encode (posterior mode:
  observed fp32/bf16 ratios 0.753 / 1.420) and decode (whole video:
  0.577 / 1.103; per-frame max: 0.590 / 1.150) against
  ``AutoencoderKLLTX2Video``.
- Tiny ``AutoencoderKLLTX2Audio`` causal mel-spectrogram decode
  (observed fp32/bf16 ratios 0.559 / 1.199).
- Tiny ``LTX2Vocoder`` stereo waveform synthesis via ``_run_ltx_vocoder``
  (observed fp32/bf16 ratios 0.208 / 1.034).
- Tiny ``Gemma3ForConditionalGeneration`` ``from_pretrained`` load path and
  multi-layer hidden-state stacking on left-padded prompts (observed
  fp32/bf16 ratios 0.658 / 0.990).
"""

import copy
import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("vllm_omni", reason="vllm_omni is required for LTX-2 tests")

from diffusers import AutoencoderKLLTX2Video
from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import (
    AutoencoderKLLTX2Audio,
)
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2AudioVideoRotaryPosEmbed as RefLTX2RotaryPosEmbed,
)
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2VideoTransformer3DModel as RefLTX2Transformer3DModel,
)
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from transformers import (
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    SiglipVisionConfig,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
    DistributedAutoencoderKLLTX2Video,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    _install_connector_attention,
)
from vllm_omni.diffusion.models.ltx2.ltx2_runtime import _run_ltx_vocoder
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import (
    LTX2AudioVideoRotaryPosEmbed,
    LTX2VideoTransformer3DModel,
)
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.platforms import current_omni_platform

from .utils import (
    assert_all_loaded,
    assert_parity,
    metrics,
)

pytestmark = [pytest.mark.omni]

# RoPE cos/sin table error bounds on TPU vs CPU float64-built tables (~3% above
# observed: video max_abs 1.511e-4, rel_l2 4.60e-5; audio max_abs 9.766e-4,
# rel_l2 5.07e-5).
ROPE_MAX_ABS_ERR: dict[str, float] = {"video": 1.6e-4, "audio": 1.0e-3}
ROPE_MAX_REL_L2: dict[str, float] = {"video": 4.8e-5, "audio": 5.3e-5}

# Per-(variant, modality) error ratios (omni_rel_l2 / diffusers_cpu_bf16_rel_l2),
# tightened to ~2.5-3% above the observed TPU v7x ratios.
DIT_RATIO_GATES: dict[tuple[str, str], tuple[float, float]] = {
    ("t2v", "video"): (0.60, 1.12),
    ("t2v", "audio"): (0.54, 1.17),
    ("stg", "video"): (0.61, 1.12),
    ("stg", "audio"): (0.58, 1.15),
    ("modality", "video"): (0.63, 1.10),
    ("modality", "audio"): (0.56, 1.09),
}
CONNECTOR_RATIO_GATES: dict[str, tuple[float, float]] = {
    "video": (0.65, 1.10),
    "audio": (0.67, 1.09),
}
VIDEO_VAE_RATIO_GATES: dict[str, tuple[float, float]] = {
    "mode": (0.78, 1.47),
    "decode": (0.60, 1.14),
    "decode-frame": (0.61, 1.19),
}
AUDIO_VAE_RATIO_GATE: tuple[float, float] = (0.58, 1.24)
VOCODER_RATIO_GATE: tuple[float, float] = (0.22, 1.07)
GEMMA3_RATIO_GATE: tuple[float, float] = (0.68, 1.02)


@pytest.fixture(scope="module")
def omni_ctx():
    """Diffusion parallel groups and forward context from OmniDiffusionConfig."""
    od_config = OmniDiffusionConfig()
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = str(od_config.master_port)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", str(od_config.num_gpus))

    owns_dist = not torch.distributed.is_initialized()
    if owns_dist:
        init_distributed_environment()
    owns_mp = not model_parallel_is_initialized()
    if owns_mp:
        initialize_model_parallel()

    device = current_omni_platform.get_torch_device()
    vllm_config = create_diffusion_vllm_config(device, od_config)
    try:
        with set_forward_context(
            vllm_config=vllm_config, omni_diffusion_config=od_config
        ):
            yield
    finally:
        if owns_mp:
            destroy_model_parallel()
        if owns_dist:
            destroy_distributed_environment()


# --------------------------------------------------------------------------- #
# 1. 3D video + 1D audio RoPE and coordinate builders
# --------------------------------------------------------------------------- #
ROPE_SHARED_KWARGS: dict[str, object] = {
    "patch_size": 1,
    "patch_size_t": 1,
    "base_num_frames": 20,
    "theta": 10000.0,
    "causal_offset": 1,
    "double_precision": True,
    "rope_type": "split",
    "num_attention_heads": 32,
}


def _assert_rope_parity(
    kind: str,
    tpu_out: tuple[torch.Tensor, torch.Tensor],
    ref_out: tuple[torch.Tensor, torch.Tensor],
    expected_shape: tuple[int, int, int, int],
) -> None:
    tpu_cos, tpu_sin = tpu_out
    ref_cos, ref_sin = ref_out

    assert tpu_cos.device.type == "tpu"
    assert tpu_cos.dtype == torch.float32
    assert tpu_sin.dtype == torch.float32
    assert tpu_cos.shape == expected_shape
    assert tpu_sin.shape == expected_shape

    vs_f64 = {
        "cos": metrics(tpu_cos, ref_cos),
        "sin": metrics(tpu_sin, ref_sin),
    }
    for name, tpu_tbl, ref_tbl in (
        ("cos", tpu_cos, ref_cos),
        ("sin", tpu_sin, ref_sin),
    ):
        assert vs_f64[name]["rel_l2"] <= ROPE_MAX_REL_L2[kind], vs_f64[name]
        torch.testing.assert_close(
            tpu_tbl.cpu().double(),
            ref_tbl.double(),
            atol=ROPE_MAX_ABS_ERR[kind],
            rtol=0,
        )


@torch.no_grad()
def test_ltx2_video_rope_on_tpu_matches_diffusers() -> None:
    """LTX2AudioVideoRotaryPosEmbed on TPU matches diffusers CPU for 3D video."""
    kwargs = {
        **ROPE_SHARED_KWARGS,
        "dim": 4096,
        "scale_factors": (8, 32, 32),
        "base_height": 2048,
        "base_width": 2048,
        "modality": "video",
    }
    omni_rope = LTX2AudioVideoRotaryPosEmbed(**kwargs)
    ref_rope = RefLTX2RotaryPosEmbed(**kwargs)

    tpu_coords = omni_rope.prepare_video_coords(
        batch_size=1,
        num_frames=4,
        height=16,
        width=24,
        device=torch.device("tpu"),
        fps=24.0,
    )
    cpu_coords = ref_rope.prepare_video_coords(
        batch_size=1,
        num_frames=4,
        height=16,
        width=24,
        device=torch.device("cpu"),
        fps=24.0,
    )
    assert tpu_coords.shape == (1, 3, 1536, 2)
    torch.testing.assert_close(tpu_coords.cpu(), cpu_coords, rtol=0, atol=1e-6)

    tpu_out = omni_rope(tpu_coords)
    ref_out = ref_rope(cpu_coords)

    _assert_rope_parity(
        kind="video",
        tpu_out=tpu_out,
        ref_out=ref_out,
        expected_shape=(1, 32, 1536, 64),
    )


@torch.no_grad()
def test_ltx2_audio_rope_on_tpu_matches_diffusers() -> None:
    """LTX2AudioVideoRotaryPosEmbed on TPU matches diffusers CPU for 1D audio."""
    kwargs = {
        **ROPE_SHARED_KWARGS,
        "dim": 2048,
        "scale_factors": (4,),
        "sampling_rate": 16000,
        "hop_length": 160,
        "modality": "audio",
    }
    omni_rope = LTX2AudioVideoRotaryPosEmbed(**kwargs)
    ref_rope = RefLTX2RotaryPosEmbed(**kwargs)

    tpu_coords = omni_rope.prepare_audio_coords(
        batch_size=1, num_frames=26, device=torch.device("tpu")
    )
    cpu_coords = ref_rope.prepare_audio_coords(
        batch_size=1, num_frames=26, device=torch.device("cpu")
    )
    assert tpu_coords.shape == (1, 1, 26, 2)
    torch.testing.assert_close(tpu_coords.cpu(), cpu_coords, rtol=0, atol=1e-6)

    tpu_out = omni_rope(tpu_coords)
    ref_out = ref_rope(cpu_coords)

    _assert_rope_parity(
        kind="audio",
        tpu_out=tpu_out,
        ref_out=ref_out,
        expected_shape=(1, 32, 26, 32),
    )


# --------------------------------------------------------------------------- #
# 2. Tiny random-weight LTX2VideoTransformer3DModel (t2v, stg, modality)
# --------------------------------------------------------------------------- #
# Preserves all production structural/RoPE fields from Lightricks/LTX-2
# transformer/config.json while shrinking depth (48 -> 2) and width.
# Note: an i2v (first-frame t=0) parity variant is intentionally omitted because
# vllm-omni's cross-modal AdaLN computes av_cross_attn_video_scale_shift from
# per-token video timestep while taking av_cross_attn_video_a2v_gate from scalar
# audio_sigma (ltx2_transformer.py), whereas diffusers couples both to either
# audio_sigma (use_cross_timestep=True) or video timestep (False). In addition,
# keyframes_abs_pos_embedding is LTX-2.5-only and unused in diffusers' forward.
TINY_DIT_CONFIG: dict[str, object] = {
    "in_channels": 16,
    "out_channels": 16,
    "patch_size": 1,
    "patch_size_t": 1,
    "num_attention_heads": 2,
    "attention_head_dim": 128,
    "cross_attention_dim": 256,
    "vae_scale_factors": (8, 32, 32),
    "pos_embed_max_pos": 20,
    "base_height": 2048,
    "base_width": 2048,
    "audio_in_channels": 16,
    "audio_out_channels": 16,
    "audio_patch_size": 1,
    "audio_patch_size_t": 1,
    "audio_num_attention_heads": 2,
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": 128,
    "audio_scale_factor": 4,
    "audio_pos_embed_max_pos": 20,
    "audio_sampling_rate": 16000,
    "audio_hop_length": 160,
    "num_layers": 2,
    "activation_fn": "gelu-approximate",
    "qk_norm": "rms_norm_across_heads",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "caption_channels": 32,
    "attention_bias": True,
    "attention_out_bias": True,
    "rope_theta": 10000.0,
    "rope_double_precision": True,
    "causal_offset": 1,
    "timestep_scale_multiplier": 1000,
    "cross_attn_timestep_scale_multiplier": 1000,
    "rope_type": "split",
    "perturbed_attn": True,
}

DIT_LATENT_GRID = (4, 4, 6)  # (F, H, W) -> S_v = 96 video tokens
DIT_AUDIO_LEN = 26  # S_a = 26 audio tokens
DIT_TEXT_LEN = 32
DIT_VARIANTS = ("t2v", "stg", "modality")


def _make_dit_base_inputs(
    ref_model: RefLTX2Transformer3DModel, seed: int = 1
) -> dict[str, object]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    f_lat, h_lat, w_lat = DIT_LATENT_GRID
    seq_v = f_lat * h_lat * w_lat
    c_in = int(TINY_DIT_CONFIG["in_channels"])
    ca_in = int(TINY_DIT_CONFIG["audio_in_channels"])
    c_txt = int(TINY_DIT_CONFIG["caption_channels"])
    video_coords = ref_model.rope.prepare_video_coords(
        batch_size=1,
        num_frames=f_lat,
        height=h_lat,
        width=w_lat,
        device=torch.device("cpu"),
        fps=24.0,
    )
    audio_coords = ref_model.audio_rope.prepare_audio_coords(
        batch_size=1, num_frames=DIT_AUDIO_LEN, device=torch.device("cpu")
    )
    return {
        "hidden_states": torch.randn(1, seq_v, c_in, generator=gen),
        "audio_hidden_states": torch.randn(1, DIT_AUDIO_LEN, ca_in, generator=gen),
        "encoder_hidden_states": torch.randn(1, DIT_TEXT_LEN, c_txt, generator=gen),
        "audio_encoder_hidden_states": torch.randn(
            1, DIT_TEXT_LEN, c_txt, generator=gen
        ),
        "timestep": torch.full((1, seq_v), 900.0),
        "audio_timestep": torch.full((1, DIT_AUDIO_LEN), 900.0),
        "sigma": torch.tensor([900.0]),
        "audio_sigma": torch.tensor([900.0]),
        "num_frames": f_lat,
        "height": h_lat,
        "width": w_lat,
        "fps": 24.0,
        "audio_num_frames": DIT_AUDIO_LEN,
        "video_coords": video_coords,
        "audio_coords": audio_coords,
    }


def _cast_tensor_kwargs(
    inputs: dict[str, object], device: str, dtype: torch.dtype
) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, val in inputs.items():
        if isinstance(val, torch.Tensor):
            # Keep coordinate grids in float32 so RoPE phase products are not
            # truncated before frequency computation.
            target_dtype = (
                torch.float32 if key in ("video_coords", "audio_coords") else dtype
            )
            out[key] = val.to(device=device, dtype=target_dtype)
        else:
            out[key] = val
    return out


def _variant_ref_kwargs(
    base: dict[str, object], variant: str, dtype: torch.dtype
) -> dict[str, object]:
    kwargs = _cast_tensor_kwargs(base, "cpu", dtype)
    if variant == "stg":
        kwargs["spatio_temporal_guidance_blocks"] = [1]
        kwargs["perturbation_mask"] = torch.zeros(1, dtype=dtype)
    elif variant == "modality":
        kwargs["isolate_modalities"] = True
    return kwargs


def _variant_omni_kwargs(
    base: dict[str, object], variant: str, dtype: torch.dtype
) -> dict[str, object]:
    kwargs = _cast_tensor_kwargs(base, "tpu", dtype)
    if variant == "stg":
        zero_mask = torch.zeros(1, 1, 1, device="tpu", dtype=dtype)
        kwargs["attention_kwargs"] = {
            "ltx_perturbation_kwargs": {
                "video_self_attention_mask": zero_mask,
                "video_self_attention_blocks": [1],
                "audio_self_attention_mask": zero_mask,
                "audio_self_attention_blocks": [1],
            }
        }
    elif variant == "modality":
        zero_mask = torch.zeros(1, 1, 1, device="tpu", dtype=dtype)
        kwargs["attention_kwargs"] = {
            "ltx_perturbation_kwargs": {
                "a2v_cross_attention_mask": zero_mask,
                "v2a_cross_attention_mask": zero_mask,
            }
        }
    return kwargs


@pytest.mark.parametrize("variant", DIT_VARIANTS)
@torch.no_grad()
def test_ltx2_tiny_transformer_parity(
    variant: str,
    omni_ctx: None,
) -> None:
    """vllm-omni LTX2VideoTransformer3DModel on TPU matches diffusers on CPU."""
    torch.manual_seed(0)
    ref_model = RefLTX2Transformer3DModel(**TINY_DIT_CONFIG).eval()
    ref_model_bf16 = copy.deepcopy(ref_model).to(torch.bfloat16).eval()
    base_inputs = _make_dit_base_inputs(ref_model, seed=1)

    ref_video, ref_audio = ref_model(
        **_variant_ref_kwargs(base_inputs, variant, torch.float32), return_dict=False
    )
    ref_bf16_video, ref_bf16_audio = ref_model_bf16(
        **_variant_ref_kwargs(base_inputs, variant, torch.bfloat16), return_dict=False
    )
    if variant != "t2v":
        t2v_video, t2v_audio = ref_model(
            **_variant_ref_kwargs(base_inputs, "t2v", torch.float32),
            return_dict=False,
        )
        assert not torch.allclose(ref_video, t2v_video)
        assert not torch.allclose(ref_audio, t2v_audio)

    tpu_outputs: dict[torch.dtype, tuple[torch.Tensor, torch.Tensor]] = {}
    for dtype in (torch.float32, torch.bfloat16):
        with set_default_torch_dtype(dtype):
            omni_model = LTX2VideoTransformer3DModel(**TINY_DIT_CONFIG)
        loaded = omni_model.load_weights(ref_model.state_dict().items())
        assert_all_loaded(loaded, omni_model)
        omni_model = omni_model.to(device="tpu", dtype=dtype).eval()
        video_out, audio_out = omni_model(
            **_variant_omni_kwargs(base_inputs, variant, dtype), return_dict=False
        )
        assert video_out.shape == (1, 96, int(TINY_DIT_CONFIG["out_channels"]))
        assert audio_out.shape == (
            1,
            DIT_AUDIO_LEN,
            int(TINY_DIT_CONFIG["audio_out_channels"]),
        )
        tpu_outputs[dtype] = (video_out.float().cpu(), audio_out.float().cpu())

    for modality, ref_out, ref_bf16_out, idx in (
        ("video", ref_video, ref_bf16_video, 0),
        ("audio", ref_audio, ref_bf16_audio, 1),
    ):
        fp32_ratio, bf16_ratio = DIT_RATIO_GATES[(variant, modality)]
        assert_parity(
            ref=ref_out,
            ref_bf16=ref_bf16_out,
            tpu_fp32=tpu_outputs[torch.float32][idx],
            tpu_bf16=tpu_outputs[torch.bfloat16][idx],
            fp32_ratio=fp32_ratio,
            bf16_ratio=bf16_ratio,
        )


# --------------------------------------------------------------------------- #
# 3. Tiny random-weight LTX2TextConnectors + _install_connector_attention
# --------------------------------------------------------------------------- #
TINY_CONNECTORS_CONFIG: dict[str, object] = {
    "caption_channels": 32,
    "text_proj_in_factor": 4,
    "video_connector_num_attention_heads": 2,
    "video_connector_attention_head_dim": 16,
    "video_connector_num_layers": 2,
    "video_connector_num_learnable_registers": 16,
    "audio_connector_num_attention_heads": 2,
    "audio_connector_attention_head_dim": 16,
    "audio_connector_num_layers": 2,
    "audio_connector_num_learnable_registers": 16,
    "connector_rope_base_seq_len": 4096,
    "rope_theta": 10000.0,
    "rope_double_precision": True,
    "causal_temporal_positioning": False,
    "rope_type": "split",
}


@torch.no_grad()
def test_ltx2_tiny_connectors_parity(
    omni_ctx: None,
) -> None:
    """LTX2TextConnectors with _install_connector_attention matches diffusers CPU."""
    torch.manual_seed(0)
    ref_conn = LTX2TextConnectors(**TINY_CONNECTORS_CONFIG).eval()
    ref_conn_bf16 = copy.deepcopy(ref_conn).to(torch.bfloat16).eval()

    # Two left-padded prompts: 10 and 25 valid tokens out of 32 (multiple of
    # num_learnable_registers=16). Left-padded positions are zeroed.
    batch_size, seq_len = 2, 32
    real_lens = (10, 25)
    in_dim = int(TINY_CONNECTORS_CONFIG["caption_channels"]) * int(
        TINY_CONNECTORS_CONFIG["text_proj_in_factor"]
    )
    gen = torch.Generator(device="cpu").manual_seed(2)
    prompt_embeds = torch.randn(batch_size, seq_len, in_dim, generator=gen)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.int64)
    for row, n_real in enumerate(real_lens):
        pad = seq_len - n_real
        prompt_embeds[row, :pad] = 0.0
        attention_mask[row, pad:] = 1

    ref_video, ref_audio, ref_mask = ref_conn(
        prompt_embeds, attention_mask, padding_side="left"
    )
    ref_bf16_video, ref_bf16_audio, ref_bf16_mask = ref_conn_bf16(
        prompt_embeds.to(torch.bfloat16), attention_mask, padding_side="left"
    )
    expected_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)
    assert torch.equal(ref_mask, expected_mask)
    assert torch.equal(ref_bf16_mask, expected_mask)

    tpu_outputs: dict[torch.dtype, tuple[torch.Tensor, torch.Tensor]] = {}
    for dtype in (torch.float32, torch.bfloat16):
        tpu_conn = copy.deepcopy(ref_conn).to(device="tpu", dtype=dtype).eval()
        _install_connector_attention(tpu_conn)
        video_emb, audio_emb, out_mask = tpu_conn(
            prompt_embeds.to(device="tpu", dtype=dtype),
            attention_mask.to(device="tpu"),
            padding_side="left",
        )
        assert video_emb.shape == (
            batch_size,
            seq_len,
            int(TINY_CONNECTORS_CONFIG["caption_channels"]),
        )
        assert audio_emb.shape == (
            batch_size,
            seq_len,
            int(TINY_CONNECTORS_CONFIG["caption_channels"]),
        )
        assert torch.equal(out_mask.cpu(), expected_mask)
        tpu_outputs[dtype] = (video_emb.float().cpu(), audio_emb.float().cpu())

    for modality, ref_out, ref_bf16_out, idx in (
        ("video", ref_video, ref_bf16_video, 0),
        ("audio", ref_audio, ref_bf16_audio, 1),
    ):
        fp32_ratio, bf16_ratio = CONNECTOR_RATIO_GATES[modality]
        assert_parity(
            ref=ref_out,
            ref_bf16=ref_bf16_out,
            tpu_fp32=tpu_outputs[torch.float32][idx],
            tpu_bf16=tpu_outputs[torch.bfloat16][idx],
            fp32_ratio=fp32_ratio,
            bf16_ratio=bf16_ratio,
        )


# --------------------------------------------------------------------------- #
# 4. Tiny random-weight DistributedAutoencoderKLLTX2Video (encode + decode)
# --------------------------------------------------------------------------- #
TINY_VIDEO_VAE_CONFIG: dict[str, object] = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 8,
    "block_out_channels": (16, 32, 64, 128),
    "decoder_block_out_channels": (16, 32, 64),
    "layers_per_block": (1, 1, 1, 1, 1),
    "decoder_layers_per_block": (1, 1, 1, 1),
    "down_block_types": (
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
    ),
    "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    "upsample_type": ("spatiotemporal", "spatiotemporal", "spatiotemporal"),
    "upsample_factor": (2, 2, 2),
    "upsample_residual": (True, True, True),
    "spatio_temporal_scaling": (True, True, True, True),
    "decoder_spatio_temporal_scaling": (True, True, True),
    "decoder_inject_noise": (False, False, False, False),
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-6,
    "scaling_factor": 1.0,
    "spatial_compression_ratio": 32,
    "temporal_compression_ratio": 8,
    "timestep_conditioning": False,
    "encoder_causal": True,
    "decoder_causal": False,
    "encoder_spatial_padding_mode": "zeros",
    "decoder_spatial_padding_mode": "reflect",
}


@torch.no_grad()
def test_ltx2_tiny_video_vae_parity(
    omni_ctx: None,
) -> None:
    """DistributedAutoencoderKLLTX2Video on TPU matches diffusers on CPU."""
    torch.manual_seed(0)
    ref_vae = AutoencoderKLLTX2Video(**TINY_VIDEO_VAE_CONFIG).eval()
    ref_vae_bf16 = copy.deepcopy(ref_vae).to(torch.bfloat16).eval()

    # 9 = 8 * 1 + 1 pixel frames -> 2 latent frames; 64x64 -> 2x2 latents (32x).
    gen = torch.Generator(device="cpu").manual_seed(1)
    video = torch.randn(1, 3, 9, 64, 64, generator=gen)
    latents = torch.randn(1, 8, 2, 2, 2, generator=gen)

    ref_mode = ref_vae.encode(video).latent_dist.mode()
    ref_decode = ref_vae.decode(latents, None, return_dict=False)[0]
    ref_bf16_mode = (
        ref_vae_bf16.encode(video.to(torch.bfloat16)).latent_dist.mode().float()
    )
    ref_bf16_decode = ref_vae_bf16.decode(
        latents.to(torch.bfloat16), None, return_dict=False
    )[0].float()

    tpu_outputs: dict[torch.dtype, dict[str, torch.Tensor]] = {}
    for dtype in (torch.float32, torch.bfloat16):
        omni_vae = DistributedAutoencoderKLLTX2Video(**TINY_VIDEO_VAE_CONFIG)
        omni_vae.init_distributed()
        omni_vae.load_state_dict(ref_vae.state_dict())
        omni_vae = omni_vae.to(device="tpu", dtype=dtype).eval()
        mode_out = omni_vae.encode(
            video.to(device="tpu", dtype=dtype)
        ).latent_dist.mode()
        decode_out = omni_vae.decode(
            latents.to(device="tpu", dtype=dtype), None, return_dict=False
        )[0]
        assert mode_out.shape == (1, 8, 2, 2, 2)
        assert decode_out.shape == (1, 3, 9, 64, 64)
        tpu_outputs[dtype] = {
            "mode": mode_out.float().cpu(),
            "decode": decode_out.float().cpu(),
        }

    for output, ref_out, ref_bf16_out in (
        ("mode", ref_mode, ref_bf16_mode),
        ("decode", ref_decode, ref_bf16_decode),
    ):
        fp32_ratio, bf16_ratio = VIDEO_VAE_RATIO_GATES[output]
        assert_parity(
            ref=ref_out,
            ref_bf16=ref_bf16_out,
            tpu_fp32=tpu_outputs[torch.float32][output],
            tpu_bf16=tpu_outputs[torch.bfloat16][output],
            fp32_ratio=fp32_ratio,
            bf16_ratio=bf16_ratio,
        )

    frame_fp32_ratio, frame_bf16_ratio = VIDEO_VAE_RATIO_GATES["decode-frame"]
    for frame in range(ref_decode.shape[2]):
        assert_parity(
            ref=ref_decode[:, :, frame],
            ref_bf16=ref_bf16_decode[:, :, frame],
            tpu_fp32=tpu_outputs[torch.float32]["decode"][:, :, frame],
            tpu_bf16=tpu_outputs[torch.bfloat16]["decode"][:, :, frame],
            fp32_ratio=frame_fp32_ratio,
            bf16_ratio=frame_bf16_ratio,
        )


# --------------------------------------------------------------------------- #
# 5. Tiny random-weight AutoencoderKLLTX2Audio decode
# --------------------------------------------------------------------------- #
TINY_AUDIO_VAE_CONFIG: dict[str, object] = {
    "base_channels": 16,
    "output_channels": 2,
    "ch_mult": (1, 2, 4),
    "num_res_blocks": 1,
    "attn_resolutions": None,
    "in_channels": 2,
    "resolution": 256,
    "latent_channels": 8,
    "norm_type": "pixel",
    "causality_axis": "height",
    "dropout": 0.0,
    "mid_block_add_attention": False,
    "sample_rate": 16000,
    "mel_hop_length": 160,
    "is_causal": True,
    "mel_bins": 64,
    "double_z": True,
}


@torch.no_grad()
def test_ltx2_tiny_audio_vae_parity(
    omni_ctx: None,
) -> None:
    """AutoencoderKLLTX2Audio decode on TPU matches CPU."""
    assert torch.nn.functional.interpolate.__name__ == "_safe_interpolate"
    torch.manual_seed(0)
    ref_audio_vae = AutoencoderKLLTX2Audio(**TINY_AUDIO_VAE_CONFIG).eval()
    ref_audio_vae_bf16 = copy.deepcopy(ref_audio_vae).to(torch.bfloat16).eval()

    audio_latents = torch.randn(
        1, 8, 26, 16, generator=torch.Generator(device="cpu").manual_seed(2)
    )
    ref_mel = ref_audio_vae.decode(audio_latents, return_dict=False)[0]
    ref_bf16_mel = ref_audio_vae_bf16.decode(
        audio_latents.to(torch.bfloat16), return_dict=False
    )[0].float()

    tpu_outputs: dict[torch.dtype, torch.Tensor] = {}
    for dtype in (torch.float32, torch.bfloat16):
        tpu_audio_vae = (
            copy.deepcopy(ref_audio_vae).to(device="tpu", dtype=dtype).eval()
        )
        mel_out = tpu_audio_vae.decode(
            audio_latents.to(device="tpu", dtype=dtype), return_dict=False
        )[0]
        assert mel_out.shape == (1, 2, 101, 64)
        tpu_outputs[dtype] = mel_out.float().cpu()

    fp32_ratio, bf16_ratio = AUDIO_VAE_RATIO_GATE
    assert_parity(
        ref=ref_mel,
        ref_bf16=ref_bf16_mel,
        tpu_fp32=tpu_outputs[torch.float32],
        tpu_bf16=tpu_outputs[torch.bfloat16],
        fp32_ratio=fp32_ratio,
        bf16_ratio=bf16_ratio,
    )


# --------------------------------------------------------------------------- #
# 6. Tiny random-weight LTX2Vocoder via _run_ltx_vocoder
# --------------------------------------------------------------------------- #
TINY_VOCODER_CONFIG: dict[str, object] = {
    "in_channels": 128,
    "hidden_channels": 64,
    "out_channels": 2,
    "upsample_kernel_sizes": [16, 15, 8, 4, 4],
    "upsample_factors": [6, 5, 2, 2, 2],
    "resnet_kernel_sizes": [3, 7, 11],
    "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "leaky_relu_negative_slope": 0.1,
    "output_sampling_rate": 24000,
}


@torch.no_grad()
def test_ltx2_tiny_vocoder_parity(
    omni_ctx: None,
) -> None:
    """LTX2Vocoder via _run_ltx_vocoder on TPU matches vocoder(mel) on CPU."""
    assert torch.nn.functional.interpolate.__name__ == "_safe_interpolate"
    torch.manual_seed(0)
    ref_vocoder = LTX2Vocoder(**TINY_VOCODER_CONFIG).eval()
    ref_vocoder_bf16 = copy.deepcopy(ref_vocoder).to(torch.bfloat16).eval()

    mel = torch.randn(
        1, 2, 16, 64, generator=torch.Generator(device="cpu").manual_seed(3)
    )
    ref_wav = ref_vocoder(mel)
    ref_bf16_wav = ref_vocoder_bf16(mel.to(torch.bfloat16)).float()
    assert (ref_wav.abs() < 0.999).float().mean() > 0.9

    tpu_outputs: dict[torch.dtype, torch.Tensor] = {}
    for dtype in (torch.float32, torch.bfloat16):
        tpu_vocoder = copy.deepcopy(ref_vocoder).to(device="tpu", dtype=dtype).eval()
        wav_out = _run_ltx_vocoder(tpu_vocoder, mel.to(device="tpu", dtype=dtype))
        assert wav_out.shape == (1, 2, 3840)
        tpu_outputs[dtype] = wav_out.float().cpu()

    fp32_ratio, bf16_ratio = VOCODER_RATIO_GATE
    assert_parity(
        ref=ref_wav,
        ref_bf16=ref_bf16_wav,
        tpu_fp32=tpu_outputs[torch.float32],
        tpu_bf16=tpu_outputs[torch.bfloat16],
        fp32_ratio=fp32_ratio,
        bf16_ratio=bf16_ratio,
    )


# --------------------------------------------------------------------------- #
# 7. Tiny random-weight Gemma3ForConditionalGeneration hidden-state stack
# --------------------------------------------------------------------------- #
TINY_GEMMA3_CONFIG = Gemma3Config(
    text_config=Gemma3TextConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        hidden_activation="gelu_pytorch_tanh",
        query_pre_attn_scalar=32,
        rms_norm_eps=1e-6,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        rope_theta=1_000_000.0,
        rope_local_base_freq=10_000.0,
        rope_scaling={"factor": 8.0, "rope_type": "linear"},
        attention_bias=False,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=1,
    ),
    vision_config=SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
    ),
)


@torch.no_grad()
def test_ltx2_tiny_gemma3_parity(
    tmp_path: Path,
    omni_ctx: None,
) -> None:
    """Gemma 3 production load path and hidden-state stacking on TPU match CPU."""
    torch.manual_seed(0)
    Gemma3ForConditionalGeneration(TINY_GEMMA3_CONFIG).save_pretrained(tmp_path)

    # Two left-padded prompts: 7 and 20 real tokens out of 32, BOS=2 prepended,
    # pad=EOS=1.
    seq_len = 32
    real_lens = (7, 20)
    gen = torch.Generator(device="cpu").manual_seed(4)
    input_ids = torch.full((len(real_lens), seq_len), 1, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, n_real in enumerate(real_lens):
        tokens = torch.randint(3, 1024, (n_real - 1,), generator=gen)
        input_ids[row, seq_len - n_real :] = torch.cat([torch.tensor([2]), tokens])
        attention_mask[row, seq_len - n_real :] = 1

    with torch.device("cpu"):
        ref_gemma = Gemma3ForConditionalGeneration.from_pretrained(
            tmp_path, dtype=torch.float32
        ).eval()
        ref_gemma_bf16 = Gemma3ForConditionalGeneration.from_pretrained(
            tmp_path, dtype=torch.bfloat16
        ).eval()

    ref_hs = ref_gemma(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
    ).hidden_states
    ref_stacked = torch.stack(ref_hs, dim=-1).flatten(2, 3)
    ref_bf16_hs = ref_gemma_bf16(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
    ).hidden_states
    ref_bf16_stacked = torch.stack(ref_bf16_hs, dim=-1).flatten(2, 3).float()
    ref_real = torch.cat(
        [ref_stacked[r, seq_len - n :] for r, n in enumerate(real_lens)], dim=0
    )
    ref_bf16_real = torch.cat(
        [ref_bf16_stacked[r, seq_len - n :] for r, n in enumerate(real_lens)], dim=0
    )

    tpu_outputs: dict[torch.dtype, torch.Tensor] = {}
    for dtype in (torch.float32, torch.bfloat16):
        with torch.device("cpu"):
            tpu_gemma = Gemma3ForConditionalGeneration.from_pretrained(
                tmp_path, dtype=dtype
            )
        tpu_gemma = tpu_gemma.to(device="tpu").eval()
        tpu_hs = tpu_gemma(
            input_ids=input_ids.to("tpu"),
            attention_mask=attention_mask.to("tpu"),
            output_hidden_states=True,
        ).hidden_states
        stacked = torch.stack(tpu_hs, dim=-1).flatten(2, 3).float().cpu()
        assert stacked.shape == (len(real_lens), seq_len, 128 * 3)
        tpu_outputs[dtype] = torch.cat(
            [stacked[r, seq_len - n :] for r, n in enumerate(real_lens)], dim=0
        )

    fp32_ratio, bf16_ratio = GEMMA3_RATIO_GATE
    assert_parity(
        ref=ref_real,
        ref_bf16=ref_bf16_real,
        tpu_fp32=tpu_outputs[torch.float32],
        tpu_bf16=tpu_outputs[torch.bfloat16],
        fp32_ratio=fp32_ratio,
        bf16_ratio=bf16_ratio,
    )

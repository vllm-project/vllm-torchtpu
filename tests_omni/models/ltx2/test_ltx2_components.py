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
HuggingFace diffusers counterpart on CPU:

- ``LTX2AudioVideoRotaryPosEmbed`` and coordinate builders
  (``prepare_video_coords``, ``prepare_audio_coords``) for video and audio.
- Tiny ``LTX2VideoTransformer3DModel`` across ``t2v``, ``stg``, and
  ``modality`` forward variants.
- Tiny ``LTX2TextConnectors`` with ``_install_connector_attention`` on
  left-padded prompts.

Total runtime on TPU v7x: ~33s (6 test items).
"""

import copy

import pytest
import torch

pytest.importorskip("vllm_omni", reason="vllm_omni is required for LTX-2 tests")

from diffusers.models.transformers.transformer_ltx2 import (
    LTX2AudioVideoRotaryPosEmbed as RefLTX2RotaryPosEmbed,
)
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2VideoTransformer3DModel as RefLTX2Transformer3DModel,
)
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm_omni.diffusion.data import OmniDiffusionConfig
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
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import (
    LTX2AudioVideoRotaryPosEmbed,
    LTX2VideoTransformer3DModel,
)
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import (
    assert_all_loaded,
    assert_within_bf16_noise,
    metrics,
)

pytestmark = [pytest.mark.omni]

# RoPE cos/sin table error bounds on TPU vs CPU float64-built tables (~3% above
# observed: video max_abs 1.511e-4, rel_l2 4.60e-5; audio max_abs 9.766e-4,
# rel_l2 5.07e-5).
ROPE_MAX_ABS_ERR: dict[str, float] = {"video": 1.6e-4, "audio": 1.0e-3}
ROPE_MAX_REL_L2: dict[str, float] = {"video": 4.8e-5, "audio": 5.3e-5}

# Per-(variant, modality) error ratios (omni_rel_l2 / diffusers_cpu_bf16_rel_l2),
# ~3% above the observed TPU v7x ratios, rounded up to 0.01. The ratios repeat
# exactly across runs (vllm-omni 0.32.0rc1).
DIT_RATIO_GATES: dict[tuple[str, str], tuple[float, float]] = {
    ("t2v", "video"): (0.46, 0.99),
    ("t2v", "audio"): (0.44, 1.02),
    ("stg", "video"): (0.45, 0.94),
    ("stg", "audio"): (0.52, 1.07),
    ("modality", "video"): (0.52, 1.11),
    ("modality", "audio"): (0.53, 1.06),
}
CONNECTOR_RATIO_GATES: dict[str, tuple[float, float]] = {
    "video": (0.49, 1.05),
    "audio": (0.51, 1.05),
}


@pytest.fixture(scope="module")
def omni_ctx():
    """Diffusion parallel groups and forward context from OmniDiffusionConfig."""
    od_config = OmniDiffusionConfig()
    owns_dist = not torch.distributed.is_initialized()
    if owns_dist:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{od_config.master_port}",
        )
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
        assert_within_bf16_noise(
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
        assert_within_bf16_noise(
            ref=ref_out,
            ref_bf16=ref_bf16_out,
            tpu_fp32=tpu_outputs[torch.float32][idx],
            tpu_bf16=tpu_outputs[torch.bfloat16][idx],
            fp32_ratio=fp32_ratio,
            bf16_ratio=bf16_ratio,
        )

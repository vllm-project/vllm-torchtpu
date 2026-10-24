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
"""Component-wise TPU vs CPU parity tests for the Wan model family.

Each vllm-omni component runs on TPU and is compared, in the same process,
against a CPU reference (HuggingFace diffusers).

Tiers:
1. Kernels: WanRotaryPosEmbed vs diffusers.
2. Tiny transformers (random weights, one per architecture class: T2V, I2V+CLIP,
   I2V-A14B (36ch, no CLIP), TI2V (2-D timestep), VACE). vllm-omni loads the
   diffusers state dict, so weight mapping and numerics are both covered.
   Both vllm-omni fp32 and bf16 errors are bounded relative to diffusers' own
   bf16 error, which isolates precision-policy differences (fp32
   modulation/residuals) from expected bf16 truncation.
3. Tiny VAEs (random weights, Wan2.1 and Wan2.2 layouts): encode (posterior
   mode, as the pipelines use) and decode, with the same relative error budget
   as 2.
4. Tiny encoders (random weights): the Hugging Face UMT5 text encoder and CLIP
   vision encoder that the pipelines run on TPU, against the same modules on
   CPU, with the same relative error budget as 2.

All random inputs come from CPU generators so TPU and CPU see identical data.
Real-checkpoint accuracy is out of scope here and belongs to the end-to-end
PR/nightly tests.
"""

import copy
import dataclasses

import pytest
import torch

pytest.importorskip("vllm_omni", reason="vllm_omni is required for Wan tests")

from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.transformers.transformer_wan import (
    WanRotaryPosEmbed as RefWanRotaryPosEmbed,
)
from diffusers.models.transformers.transformer_wan import (
    WanTransformer3DModel as RefWanTransformer,
)
from diffusers.models.transformers.transformer_wan_vace import (
    WanVACETransformer3DModel as RefWanVACETransformer,
)
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    UMT5Config,
    UMT5EncoderModel,
)
from vllm.config import VllmConfig
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    load_wan_weights_with_optional_gate,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import (
    WanS2VTransformer3DModel,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    WanRotaryPosEmbed,
    WanSelfAttention,
    WanTransformer3DModel,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_vace_transformer import (
    WanVACETransformer3DModel,
)

from tests_omni.omni_test_utils import assert_all_loaded, assert_within_bf16_noise
from vllm_torchtpu.omni.patches import apply_omni_model_specific_patches
from vllm_torchtpu.omni.rope import can_fuse_qk_norm_rope

pytestmark = [pytest.mark.omni]


# Relative to diffusers' own CPU bf16 rel_l2, TPU fp32 is 0.52-0.62x (MXU uses
# bf16 passes while non-matmul ops stay fp32) and TPU bf16 is 1.00-1.28x.
FP32_ERR_RATIO = 0.65
BF16_ERR_RATIO = 1.35


@pytest.fixture(scope="module")
def omni_ctx():
    """Single-rank diffusion parallel groups plus a default forward context."""
    owns_dist_env = not torch.distributed.is_initialized()
    if owns_dist_env:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        initialize_model_parallel(tensor_parallel_size=1, pipeline_parallel_size=1)
    # vLLM CustomOps (Conv3dLayer, QKVParallelLinear) need a current VllmConfig
    # at construction time; set_forward_context installs it when provided.
    with set_forward_context(
        vllm_config=VllmConfig(), omni_diffusion_config=OmniDiffusionConfig()
    ):
        yield
    if owns_dist_env:
        destroy_distributed_env()


def _build_omni_transformer(config: dict) -> torch.nn.Module:
    model_cls = (
        WanVACETransformer3DModel if "vace_layers" in config else WanTransformer3DModel
    )
    return model_cls(**config)


def _load_omni(model: torch.nn.Module, weights) -> None:
    # AutoWeightsLoader skips the *root* module's load_weights (QKV fusion,
    # scale_shift_table remap), so nest the model as `transformer.` like the
    # pipeline does, letting the loader delegate to WanTransformer3DModel.
    parent = torch.nn.Module()
    parent.transformer = model
    loaded = load_wan_weights_with_optional_gate(
        parent, ((f"transformer.{name}", tensor) for name, tensor in weights)
    )
    loaded = {name.removeprefix("transformer.") for name in loaded}
    assert_all_loaded(loaded, model)


def _make_inputs(
    config: dict,
    latent_grid,
    text_embeds: torch.Tensor,
    per_token_timestep: bool,
    seed: int = 0,
) -> dict:
    """Diffusers forward kwargs; see _to_omni for the vllm-omni variant."""
    generator = torch.Generator().manual_seed(seed)
    frames, height, width = latent_grid
    inputs = {
        "hidden_states": torch.randn(
            1, config["in_channels"], frames, height, width, generator=generator
        ),
        "timestep": torch.tensor([900.0]),
        "encoder_hidden_states": text_embeds,
    }
    if per_token_timestep:
        # TI2V keeps the image-conditioned first latent frame at t=0. Distinct
        # per-token values make a token-order bug in the 2-D timestep path visible.
        p_t, p_h, p_w = config["patch_size"]
        tokens_per_frame = (height // p_h) * (width // p_w)
        timestep = torch.full((1, (frames // p_t) * tokens_per_frame), 900.0)
        timestep[:, :tokens_per_frame] = 0.0
        inputs["timestep"] = timestep
    if config.get("image_dim"):
        inputs["encoder_hidden_states_image"] = torch.randn(
            1, 257, config["image_dim"], generator=generator
        )
    if config.get("vace_layers"):
        inputs["control_hidden_states"] = torch.randn(
            1, config["vace_in_channels"], frames, height, width, generator=generator
        )
    return inputs


def _to(inputs: dict, device, dtype) -> dict:
    return {
        name: tensor.to(device=device, dtype=dtype) for name, tensor in inputs.items()
    }


def _to_omni(inputs: dict, dtype) -> dict:
    """Move diffusers kwargs to TPU, renaming the VACE control input."""
    inputs = _to(inputs, "tpu", dtype)
    if "control_hidden_states" in inputs:
        inputs["vace_context"] = inputs.pop("control_hidden_states")
    return inputs


# --------------------------------------------------------------------------- #
# Tier 1: kernels
# --------------------------------------------------------------------------- #
# Fused q/k RMSNorm + RoPE (single bf16 rounding) vs unpatched (intermediate
# bf16 roundings): observed max_abs is 1.95e-3 (2^-9) at output std ~0.039.
FUSED_ATTN_RTOL = 1e-2
FUSED_ATTN_ATOL = 3e-3
# TPU float32 vs diffusers CPU float64 RoPE frequency tables: observed max_abs
# is 4.3e-5 - 6.8e-5 across the 5x30x52 grid.
ROPE_FREQS_ATOL = 1e-4


def test_wan_rotary_embedding_pair_swap_bit_exact_on_tpu() -> None:
    apply_omni_model_specific_patches()

    torch.manual_seed(42)
    rope = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
    x_tpu = torch.randn(2, 128, 12, 128, dtype=torch.bfloat16, device="tpu")
    cos_tpu = torch.randn(1, 128, 1, 64, dtype=torch.bfloat16, device="tpu")
    sin_tpu = torch.randn(1, 128, 1, 64, dtype=torch.bfloat16, device="tpu")

    out = rope(x_tpu, cos_tpu, sin_tpu)
    native_out = rope.forward_native(x_tpu, cos_tpu, sin_tpu)

    assert out.device.type == "tpu"
    torch.testing.assert_close(out.cpu(), native_out.cpu(), rtol=0, atol=0)


@torch.no_grad()
def test_wan_self_attention_fused_matches_unpatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fused q/k RMSNorm + RoPE on the real WanSelfAttention on TPU."""
    apply_omni_model_specific_patches()

    torch.manual_seed(42)
    for target in (
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        "vllm_omni.diffusion.models.wan2_2.wan2_2_transformer.get_tensor_model_parallel_world_size",
        "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
    ):
        monkeypatch.setattr(target, lambda: 1)
    for target in (
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
    ):
        monkeypatch.setattr(target, lambda: 0)

    attn = WanSelfAttention(dim=256, num_heads=2, head_dim=128).to(
        device="tpu", dtype=torch.bfloat16
    )

    for p in attn.parameters():
        torch.nn.init.normal_(p, std=0.02)
    attn.norm_q.weight.copy_(
        torch.linspace(0.5, 1.5, 256, device="tpu", dtype=torch.bfloat16)
    )
    attn.norm_k.weight.copy_(
        torch.linspace(1.5, 0.5, 256, device="tpu", dtype=torch.bfloat16)
    )

    hidden = torch.randn(2, 128, 256, device="tpu", dtype=torch.bfloat16)
    cos = torch.randn(1, 128, 1, 64, device="tpu", dtype=torch.bfloat16)
    sin = torch.randn(1, 128, 1, 64, device="tpu", dtype=torch.bfloat16)
    rotary_emb = (cos, sin)

    assert can_fuse_qk_norm_rope(attn, hidden, rotary_emb)
    fused_out = attn(hidden, rotary_emb)
    unpatched_out = WanSelfAttention._tpu_orig_forward(attn, hidden, rotary_emb)

    assert fused_out.device.type == "tpu"
    assert torch.isfinite(fused_out).all()
    torch.testing.assert_close(
        fused_out.cpu().float(),
        unpatched_out.cpu().float(),
        rtol=FUSED_ATTN_RTOL,
        atol=FUSED_ATTN_ATOL,
    )


def test_wan_rope_freqs_on_tpu_match_diffusers():
    """RoPE tables built in fp32 (no float64 on TPU) vs diffusers' float64 build."""
    with torch.device("tpu"):
        rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024)
    ref_rope = RefWanRotaryPosEmbed(128, (1, 2, 2), 1024)
    # Latents [B, C, T, H, W]; the (1, 2, 2) patch gives a 5x30x52 token grid.
    latents = torch.zeros(1, 16, 5, 60, 104)
    tpu_freqs = rope(latents.to("tpu"))
    ref_freqs = ref_rope(latents)
    for tpu_table, ref_table in zip(tpu_freqs, ref_freqs, strict=True):
        torch.testing.assert_close(
            tpu_table.cpu().double(),
            ref_table.double(),
            atol=ROPE_FREQS_ATOL,
            rtol=0,
        )


# --------------------------------------------------------------------------- #
# Tier 2: tiny random-weight transformers, one per architecture class
# --------------------------------------------------------------------------- #
_TINY_TRANSFORMER_BASE_CONFIG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=128,
    in_channels=16,
    out_channels=16,
    text_dim=64,
    freq_dim=256,
    ffn_dim=512,
    num_layers=2,
    cross_attn_norm=True,
    eps=1e-6,
    rope_max_seq_len=1024,
)


@dataclasses.dataclass(frozen=True)
class TinyTransformerVariant:
    overrides: dict  # applied on top of _TINY_TRANSFORMER_BASE_CONFIG
    per_token_timestep: bool = False  # Wan2.2 TI2V: [batch, tokens] timestep


TINY_TRANSFORMER_VARIANTS = {
    "t2v": TinyTransformerVariant({}),
    "i2v-clip": TinyTransformerVariant(
        {"in_channels": 36, "image_dim": 32, "added_kv_proj_dim": 256}
    ),
    "i2v-a14b": TinyTransformerVariant({"in_channels": 36}),
    "ti2v": TinyTransformerVariant(
        {"in_channels": 48, "out_channels": 48}, per_token_timestep=True
    ),
    "vace": TinyTransformerVariant({"vace_layers": [0, 1], "vace_in_channels": 96}),
}


@pytest.mark.parametrize("arch", list(TINY_TRANSFORMER_VARIANTS))
@torch.no_grad()
def test_wan_tiny_transformer_parity(arch, omni_ctx, tmp_path):
    """vllm-omni transformer on TPU, loaded from a diffusers state dict,
    matches the diffusers forward pass on CPU."""
    variant = TINY_TRANSFORMER_VARIANTS[arch]
    config = {**_TINY_TRANSFORMER_BASE_CONFIG, **variant.overrides}
    ref_cls = RefWanVACETransformer if "vace_layers" in config else RefWanTransformer
    torch.manual_seed(0)
    ref_model = ref_cls(**config).eval()
    # Reload in bf16 so diffusers honours _keep_in_fp32_modules.
    ref_model.save_pretrained(tmp_path)
    ref_model_bf16 = ref_cls.from_pretrained(
        tmp_path, torch_dtype=torch.bfloat16
    ).eval()
    text_embeds = torch.randn(
        1, 32, config["text_dim"], generator=torch.Generator().manual_seed(1)
    )
    inputs = _make_inputs(config, (3, 16, 16), text_embeds, variant.per_token_timestep)

    ref = ref_model(**inputs, return_dict=False)[0]
    inputs_bf16 = _to(inputs, "cpu", torch.bfloat16)
    ref_bf16 = ref_model_bf16(**inputs_bf16, return_dict=False)[0]
    tpu = {}
    for dtype in (torch.float32, torch.bfloat16):
        with set_default_torch_dtype(dtype):
            omni_model = _build_omni_transformer(config)
        _load_omni(omni_model, ref_model.state_dict().items())
        omni_model = omni_model.to(device="tpu").eval()
        tpu[dtype] = omni_model(**_to_omni(inputs, dtype), return_dict=False)[0]

    assert_within_bf16_noise(
        ref=ref,
        ref_bf16=ref_bf16,
        tpu_fp32=tpu[torch.float32],
        tpu_bf16=tpu[torch.bfloat16],
        fp32_ratio=FP32_ERR_RATIO,
        bf16_ratio=BF16_ERR_RATIO,
    )


# --------------------------------------------------------------------------- #
# Tier 3: tiny random-weight VAEs (encode + decode)
# --------------------------------------------------------------------------- #
_TINY_VAE_BASE_CONFIG = dict(base_dim=8, dim_mult=[1, 2, 4, 4], num_res_blocks=1)
TINY_VAE_VARIANTS = {
    "wan2.1": {"z_dim": 4},
    "wan2.2": {  # residual down/up blocks, 2x2 patchify, 16x spatial compression
        "z_dim": 8,
        "decoder_base_dim": 16,
        "is_residual": True,
        "in_channels": 12,
        "out_channels": 12,
        "patch_size": 2,
        "scale_factor_spatial": 16,
    },
}


def _build_omni_vae(config: dict, state_dict: dict, dtype) -> torch.nn.Module:
    omni_vae = DistributedAutoencoderKLWan(**config)
    omni_vae.init_distributed()  # from_pretrained does this in the pipeline
    omni_vae.load_state_dict(state_dict)
    return omni_vae.to(device="tpu", dtype=dtype).eval()


def _run_vae(vae, video: torch.Tensor, latents: torch.Tensor, dtype) -> dict:
    """Encode to the posterior mode and decode; results in fp32 on CPU.

    The pipelines encode with retrieve_latents(sample_mode="argmax"), i.e. the
    posterior mode, so the stochastic posterior sample is not compared.
    """
    device = next(vae.parameters()).device
    mode = vae.encode(video.to(device, dtype)).latent_dist.mode()
    decoded = vae.decode(latents.to(device, dtype), return_dict=False)[0]
    return {"mode": mode.float().cpu(), "decode": decoded.float().cpu()}


@pytest.mark.parametrize("variant", list(TINY_VAE_VARIANTS))
@torch.no_grad()
def test_wan_tiny_vae_parity(variant, omni_ctx):
    """vllm-omni VAE on TPU matches diffusers on CPU for encode (mode) and decode."""
    z_dim = TINY_VAE_VARIANTS[variant]["z_dim"]
    config = {
        **_TINY_VAE_BASE_CONFIG,
        **TINY_VAE_VARIANTS[variant],
        "latents_mean": [0.0] * z_dim,
        "latents_std": [1.0] * z_dim,
    }
    torch.manual_seed(0)
    ref_vae = AutoencoderKLWan(**config).eval()
    generator = torch.Generator().manual_seed(0)
    # 9 = 1 + 4k pixel frames -> 3 latent frames.
    video = torch.rand(1, 3, 9, 64, 64, generator=generator) * 2 - 1
    latent_size = 64 // ref_vae.config.scale_factor_spatial
    latents = torch.randn(1, z_dim, 3, latent_size, latent_size, generator=generator)

    ref = _run_vae(ref_vae, video, latents, torch.float32)
    # AutoencoderKLWan has no _keep_in_fp32_modules, so a plain cast matches
    # what from_pretrained(torch_dtype=bfloat16) would produce.
    ref_vae_bf16 = copy.deepcopy(ref_vae).to(torch.bfloat16)
    ref_bf16 = _run_vae(ref_vae_bf16, video, latents, torch.bfloat16)
    tpu = {
        dtype: _run_vae(
            _build_omni_vae(config, ref_vae.state_dict(), dtype), video, latents, dtype
        )
        for dtype in (torch.float32, torch.bfloat16)
    }

    for output in ("mode", "decode"):
        assert_within_bf16_noise(
            ref=ref[output],
            ref_bf16=ref_bf16[output],
            tpu_fp32=tpu[torch.float32][output],
            tpu_bf16=tpu[torch.bfloat16][output],
            fp32_ratio=FP32_ERR_RATIO,
            bf16_ratio=BF16_ERR_RATIO,
        )

    # Whole-video metrics can average away an error confined to a single frame
    # (e.g. frame 0 causal-conv padding).
    for frame in range(ref["decode"].shape[2]):
        assert_within_bf16_noise(
            ref=ref["decode"][:, :, frame],
            ref_bf16=ref_bf16["decode"][:, :, frame],
            tpu_fp32=tpu[torch.float32]["decode"][:, :, frame],
            tpu_bf16=tpu[torch.bfloat16]["decode"][:, :, frame],
            fp32_ratio=FP32_ERR_RATIO,
            bf16_ratio=BF16_ERR_RATIO,
        )


# --------------------------------------------------------------------------- #
# Tier 4: tiny random-weight encoders (Hugging Face modules run on TPU)
# --------------------------------------------------------------------------- #
# Same layer types as Wan's google/umt5-xxl and CLIP ViT-H/14 checkpoints, at
# their head_dim (64 / 80) and the pipelines' sequence lengths (512 / 257).
_TINY_UMT5_CONFIG = dict(
    vocab_size=1024,
    d_model=128,
    d_kv=64,
    d_ff=256,
    num_heads=2,
    num_layers=2,
    feed_forward_proj="gated-gelu",
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
)
_TINY_CLIP_CONFIG = dict(
    hidden_size=160,
    intermediate_size=640,
    num_attention_heads=2,
    num_hidden_layers=3,
    image_size=224,
    patch_size=14,
    hidden_act="gelu",
)
WAN_TEXT_LEN = 512  # encode_prompt pads every prompt to max_sequence_length


def _assert_hf_encoder_parity(model_cls, path, forward) -> None:
    """Load a saved encoder the way the pipelines do, from_pretrained(torch_dtype)
    then .to(device), on CPU and TPU in fp32 and bf16; check TPU against CPU."""
    outputs = {}
    for device in ("cpu", "tpu"):
        for dtype in (torch.float32, torch.bfloat16):
            model = model_cls.from_pretrained(path, torch_dtype=dtype).to(device)
            outputs[device, dtype] = forward(model.eval(), device).float().cpu()
    assert_within_bf16_noise(
        ref=outputs["cpu", torch.float32],
        ref_bf16=outputs["cpu", torch.bfloat16],
        tpu_fp32=outputs["tpu", torch.float32],
        tpu_bf16=outputs["tpu", torch.bfloat16],
        fp32_ratio=FP32_ERR_RATIO,
        bf16_ratio=BF16_ERR_RATIO,
    )


@torch.no_grad()
def test_wan_tiny_umt5_parity(tmp_path):
    """UMT5 text encoder on TPU vs CPU, on the prompt tokens encode_prompt keeps."""
    config = UMT5Config(**_TINY_UMT5_CONFIG)
    torch.manual_seed(0)
    UMT5EncoderModel(config).save_pretrained(tmp_path)
    # Two prompts padded with id 0 to 512 tokens; 300 tokens reaches the
    # relative-position buckets beyond relative_attention_max_distance.
    seq_lens = (13, 300)
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.zeros(len(seq_lens), WAN_TEXT_LEN, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, seq_len in enumerate(seq_lens):
        input_ids[row, :seq_len] = torch.randint(
            1, config.vocab_size, (seq_len,), generator=generator
        )
        attention_mask[row, :seq_len] = 1

    def forward(model, device):
        outputs = model(input_ids.to(device), attention_mask.to(device))
        # encode_prompt keeps each prompt's real tokens and zeroes the padding.
        hidden = outputs.last_hidden_state
        return torch.cat([hidden[row, :n] for row, n in enumerate(seq_lens)])

    _assert_hf_encoder_parity(UMT5EncoderModel, tmp_path, forward)


@torch.no_grad()
def test_wan_tiny_clip_parity(tmp_path):
    """CLIP vision encoder on TPU vs CPU, on the penultimate hidden states that
    Wan2.1 I2V passes to the transformer."""
    torch.manual_seed(0)
    CLIPVisionModel(CLIPVisionConfig(**_TINY_CLIP_CONFIG)).save_pretrained(tmp_path)
    pixel_values = torch.randn(
        1, 3, 224, 224, generator=torch.Generator().manual_seed(0)
    )

    def forward(model, device):
        # encode_image casts pixel_values to the encoder's dtype.
        pixel_inputs = pixel_values.to(device=device, dtype=model.dtype)
        return model(pixel_inputs, output_hidden_states=True).hidden_states[-2]

    _assert_hf_encoder_parity(CLIPVisionModel, tmp_path, forward)


class TestWanFreshTimestepPatch:
    """The patched Wan forward gets a flushed copy of the timestep view."""

    @pytest.mark.parametrize(
        "cls",
        [WanTransformer3DModel, WanVACETransformer3DModel, WanS2VTransformer3DModel],
        ids=["wan", "vace", "s2v"],
    )
    def test_forward_gets_flushed_copy_of_timestep_view(
        self, monkeypatch: pytest.MonkeyPatch, cls: type
    ) -> None:
        """Keyword only: the wrapper names timestep, and the pipelines call
        forward by keyword."""
        from vllm_torchtpu.omni import patches

        calls = []
        monkeypatch.setattr(
            cls,
            "forward",
            lambda self, hidden_states, timestep, *args, **kwargs: calls.append(
                (hidden_states, timestep, kwargs)
            ),
        )
        synced = []
        monkeypatch.setattr(
            "vllm_torchtpu.utils.synchronize_tensors",
            lambda t, wait=True: synced.append((t, wait)),
        )
        patches._wrap_fresh_timestep(cls)

        # What the pipelines pass: t.expand(batch), a view into the schedule.
        view = torch.arange(6.0)[3].expand(2)
        hidden = torch.zeros(2, 4)
        ctx = torch.ones(2, 3)
        cls.forward(
            object(), hidden_states=hidden, timestep=view, encoder_hidden_states=ctx
        )

        assert len(calls) == 1, f"original forward ran {len(calls)} times, expected 1"
        got_hidden, got_t, got_kwargs = calls[0]
        assert got_t._base is None, (
            "forward got a view; the timestep should own its storage"
        )
        torch.testing.assert_close(
            got_t, view, msg="copied timestep values differ from the view"
        )
        assert len(synced) == 1, (
            f"synchronize_tensors called {len(synced)} times, expected 1"
        )
        assert synced[0][0] is got_t, (
            "synchronize_tensors was not called on the tensor passed to forward"
        )
        assert synced[0][1] is False, (
            "synchronize_tensors should not block (expected wait=False)"
        )
        assert got_hidden is hidden, "hidden_states was not passed through unchanged"
        assert got_kwargs["encoder_hidden_states"] is ctx, (
            "encoder_hidden_states was not passed through unchanged"
        )

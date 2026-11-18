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
"""Offline tiny random-weight pipelines for fast TPU engine tests.

Each builder returns a local model directory that
``vllm_omni.entrypoints.omni.Omni(model=...)`` loads with zero HuggingFace Hub
calls and zero upstream weights.

- Diffusers ``ModelMixin`` components are built from Python config dicts via
  ``cls.from_config(...)``; transformers ``PreTrainedModel`` components are
  built via ``cls(cls.config_class.from_dict(...))``.
- DiTs keep ``attention_head_dim`` 128, which the TPU attention kernels are tuned
  for, with 2 layers (and 2 single layers for Flux) of 4 heads.
- Text encoders keep 2 layers and a 256 hidden size; DiT text dims match it.
- Tokenizers and schedulers are constructed directly in Python and saved
  alongside the bf16 weights under ``tempfile.gettempdir()``.
- Weights are seeded on the CPU generator only: ``torch.manual_seed`` also
  seeds the TPU generators, which opens the TPU in the calling process.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import diffusers
import torch
import transformers
from diffusers import (
    AutoencoderKL,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
    UniPCMultistepScheduler,
    WanPipeline,
    WanTransformer3DModel,
)
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    OPTConfig,
    OPTForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    T5EncoderModel,
    T5TokenizerFast,
    UMT5EncoderModel,
)

_ROOT = os.path.join(tempfile.gettempdir(), "vllm-torchtpu-tiny-models")
_SEED = 0
_TEXT_HIDDEN_SIZE = 256

_QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _bpe_vocab() -> dict[str, int]:
    tokens = [
        "<|endoftext|>",
        "<|startoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<unk>",
        " ",
    ]
    tokens.extend(chr(c) for c in range(33, 127))
    tokens.append("\n")
    tokens.append("Ġ")
    return {tok: idx for idx, tok in enumerate(tokens)}


def _t5_vocab() -> list[tuple[str, float]]:
    tokens = ["<pad>", "</s>", "<unk>", "▁"]
    tokens.extend(chr(c) for c in range(33, 127))
    return [(tok, 0.0 if idx < 3 else -1.0) for idx, tok in enumerate(tokens)]


def tiny_diffusion_dir() -> str:
    """Returns an existing empty local directory for offline config-only tests."""
    model_dir = os.path.join(_ROOT, "tiny-diffusion-dir")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def tiny_flux_schnell() -> str:
    """Builds an offline tiny FLUX.1-schnell pipeline.

    Returns:
        The local model directory.
    """
    clip_vocab = _bpe_vocab()
    t5_vocab = _t5_vocab()
    components: dict[str, tuple[type, dict[str, Any]] | Any] = {
        "scheduler": FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
        ),
        "vae": (
            AutoencoderKL,
            {
                "in_channels": 3,
                "out_channels": 3,
                "down_block_types": ("DownEncoderBlock2D",) * 4,
                "up_block_types": ("UpDecoderBlock2D",) * 4,
                "block_out_channels": (32, 64, 128, 128),
                "layers_per_block": 1,
                "latent_channels": 16,
                "norm_num_groups": 32,
                "sample_size": 256,
                "scaling_factor": 0.3611,
                "shift_factor": 0.1159,
                "use_quant_conv": False,
                "use_post_quant_conv": False,
            },
        ),
        "text_encoder": (
            CLIPTextModel,
            {
                "vocab_size": len(clip_vocab),
                "hidden_size": _TEXT_HIDDEN_SIZE,
                "intermediate_size": 2 * _TEXT_HIDDEN_SIZE,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "max_position_embeddings": 77,
                "projection_dim": _TEXT_HIDDEN_SIZE,
                "bos_token_id": clip_vocab["<|startoftext|>"],
                "eos_token_id": clip_vocab["<|endoftext|>"],
                "pad_token_id": clip_vocab["<|endoftext|>"],
            },
        ),
        "tokenizer": CLIPTokenizer(
            vocab=clip_vocab,
            merges=[],
            unk_token="<unk>",
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            model_max_length=77,
        ),
        "text_encoder_2": (
            T5EncoderModel,
            {
                "vocab_size": len(t5_vocab),
                "d_model": _TEXT_HIDDEN_SIZE,
                "d_kv": 64,
                "d_ff": 2 * _TEXT_HIDDEN_SIZE,
                "num_layers": 2,
                "num_decoder_layers": 2,
                "num_heads": 4,
            },
        ),
        "tokenizer_2": T5TokenizerFast(vocab=t5_vocab, model_max_length=512),
        "transformer": (
            FluxTransformer2DModel,
            {
                "patch_size": 1,
                "in_channels": 64,
                "out_channels": 64,
                "num_layers": 2,
                "num_single_layers": 2,
                "attention_head_dim": 128,
                "num_attention_heads": 4,
                "joint_attention_dim": _TEXT_HIDDEN_SIZE,
                "pooled_projection_dim": _TEXT_HIDDEN_SIZE,
                "guidance_embeds": False,
                "axes_dims_rope": (16, 56, 56),
            },
        ),
    }
    return _build_tiny_pipeline("tiny-flux-schnell", FluxPipeline, components)


def tiny_wan2_1_t2v() -> str:
    """Builds an offline tiny Wan2.1-T2V pipeline.

    Returns:
        The local model directory.
    """
    t5_vocab = _t5_vocab()
    components: dict[str, tuple[type, dict[str, Any]] | Any] = {
        "scheduler": UniPCMultistepScheduler(
            num_train_timesteps=1000,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=3.0,
        ),
        "vae": (
            AutoencoderKLWan,
            {
                "base_dim": 32,
                "z_dim": 16,
                "dim_mult": [1, 2, 4, 4],
                "num_res_blocks": 1,
                "temperal_downsample": [False, True, True],
            },
        ),
        "text_encoder": (
            UMT5EncoderModel,
            {
                "vocab_size": len(t5_vocab),
                "d_model": _TEXT_HIDDEN_SIZE,
                "d_kv": 64,
                "d_ff": 2 * _TEXT_HIDDEN_SIZE,
                "num_layers": 2,
                "num_decoder_layers": 2,
                "num_heads": 4,
            },
        ),
        "tokenizer": T5TokenizerFast(vocab=t5_vocab, model_max_length=512),
        "transformer": (
            WanTransformer3DModel,
            {
                "patch_size": (1, 2, 2),
                "num_attention_heads": 4,
                "attention_head_dim": 128,
                "in_channels": 16,
                "out_channels": 16,
                "text_dim": _TEXT_HIDDEN_SIZE,
                "freq_dim": 256,
                "ffn_dim": 1024,
                "num_layers": 2,
                "cross_attn_norm": True,
                "qk_norm": "rms_norm_across_heads",
            },
        ),
    }
    return _build_tiny_pipeline("tiny-wan2-1-t2v", WanPipeline, components)


def tiny_qwen_image() -> str:
    """Builds an offline tiny Qwen-Image pipeline.

    Returns:
        The local model directory.
    """
    qwen_vocab = _bpe_vocab()
    components: dict[str, tuple[type, dict[str, Any]] | Any] = {
        "scheduler": FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            time_shift_type="exponential",
        ),
        "vae": (
            AutoencoderKLQwenImage,
            {
                "base_dim": 32,
                "z_dim": 16,
                "dim_mult": [1, 2, 4, 4],
                "num_res_blocks": 1,
                "temperal_downsample": [False, True, True],
            },
        ),
        "text_encoder": (
            Qwen2_5_VLForConditionalGeneration,
            {
                "text_config": {
                    "vocab_size": len(qwen_vocab),
                    "hidden_size": _TEXT_HIDDEN_SIZE,
                    "intermediate_size": 2 * _TEXT_HIDDEN_SIZE,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 128,
                    "max_position_embeddings": 32768,
                    "bos_token_id": qwen_vocab["<|im_start|>"],
                    "eos_token_id": qwen_vocab["<|im_end|>"],
                    "pad_token_id": qwen_vocab["<|endoftext|>"],
                    "rope_scaling": {
                        "type": "mrope",
                        "mrope_section": [16, 24, 24],
                    },
                    "layer_types": ["full_attention", "full_attention"],
                },
                "vision_config": {
                    "depth": 2,
                    "hidden_size": _TEXT_HIDDEN_SIZE,
                    "num_heads": 4,
                    "intermediate_size": 2 * _TEXT_HIDDEN_SIZE,
                    "out_hidden_size": _TEXT_HIDDEN_SIZE,
                    "fullatt_block_indexes": [0, 1],
                },
            },
        ),
        "tokenizer": Qwen2Tokenizer(
            vocab=qwen_vocab,
            merges=[],
            unk_token="<unk>",
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>",
            model_max_length=1024,
            chat_template=_QWEN_CHAT_TEMPLATE,
        ),
        "transformer": (
            QwenImageTransformer2DModel,
            {
                "patch_size": 2,
                "in_channels": 64,
                "out_channels": 16,
                "num_layers": 2,
                "attention_head_dim": 128,
                "num_attention_heads": 4,
                "joint_attention_dim": _TEXT_HIDDEN_SIZE,
                "axes_dims_rope": (16, 56, 56),
            },
        ),
    }
    return _build_tiny_pipeline("tiny-qwen-image", QwenImagePipeline, components)


def tiny_opt() -> str:
    """Builds an offline tiny OPT causal LM directory for AR config/arg tests.

    Returns:
        The local model directory.
    """
    model_dir = os.path.join(_ROOT, "tiny-opt")
    if os.path.isdir(model_dir):
        return model_dir
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "hello": 3, "world": 4}
    tok_obj = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    cpu_rng = torch.get_rng_state()
    try:
        torch.default_generator.manual_seed(_SEED)
        model = OPTForCausalLM(
            OPTConfig(
                vocab_size=len(vocab),
                hidden_size=64,
                ffn_dim=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                word_embed_proj_dim=64,
            )
        ).to(torch.bfloat16)
    finally:
        torch.set_rng_state(cpu_rng)
    os.makedirs(_ROOT, exist_ok=True)
    staging_dir = tempfile.mkdtemp(dir=_ROOT)
    model.save_pretrained(staging_dir)
    tokenizer.save_pretrained(staging_dir)
    os.rename(staging_dir, model_dir)
    return model_dir


def _build_tiny_pipeline(
    dir_name: str,
    pipeline_cls: type,
    components: dict[str, tuple[type, dict[str, Any]] | Any],
) -> str:
    """Constructs and saves a random-weight diffusers pipeline offline."""
    model_dir = os.path.join(_ROOT, dir_name)
    if os.path.isdir(model_dir):
        return model_dir
    init_kwargs: dict[str, Any] = {}
    cpu_rng = torch.get_rng_state()
    try:
        torch.default_generator.manual_seed(_SEED)
        for name, spec in components.items():
            if isinstance(spec, tuple):
                cls, config = spec
                if issubclass(cls, diffusers.ModelMixin):
                    init_kwargs[name] = cls.from_config(config)
                elif issubclass(cls, transformers.PreTrainedModel):
                    init_kwargs[name] = cls(cls.config_class.from_dict(config))
                else:
                    raise TypeError(f"Unsupported component class: {cls}")
            else:
                init_kwargs[name] = spec
    finally:
        torch.set_rng_state(cpu_rng)
    pipeline = pipeline_cls(**init_kwargs)
    os.makedirs(_ROOT, exist_ok=True)
    # Publish with a rename so an interrupted save is never taken as built.
    staging_dir = tempfile.mkdtemp(dir=_ROOT)
    pipeline.to(torch.bfloat16).save_pretrained(staging_dir)
    os.rename(staging_dir, model_dir)
    return model_dir

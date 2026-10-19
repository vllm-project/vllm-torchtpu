#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a synthetic Hugging Face cache folder for MiniMaxAI/MiniMax-H3.

Nothing here reads an upstream checkpoint, config file, tokenizer or weight,
and no ``diffusers`` / ``transformers`` class is used as a source of default
values. Every config value is either a constant, default or assert from the
vllm-omni source tree at commit e1d77a3 (cited as ``vllm-omni <path>:<line>``)
or is labelled SYNTHETIC. All tensors are random: normal(0, 0.02) matrices,
ones for norm gains, zeros for biases.

Layouts
  hub (default):  <out-hub-dir>/models--MiniMaxAI--MiniMax-H3/
                  {refs/main,blobs/<sha256>,snapshots/<rev>/...}
  plain:          --out-dir DIR writes the snapshot tree directly into DIR.

Imports only torch, safetensors and the standard library.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import math
import os
import shutil
import stat
import struct
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REMOTE_CODE_DIR = HERE / "remote_code"

REPO_ID = "MiniMaxAI/MiniMax-H3"
REPO_FOLDER = "models--" + REPO_ID.replace("/", "--")
REVISION_SEED = "minimax-h3-vllm-omni-only-synthetic-v2"
REVISION = hashlib.sha1(REVISION_SEED.encode("utf-8")).hexdigest()
STAMP_KEY = "h3_synthetic"
STAMP_VALUE = "apache2-random-v1"
REAL_HUB_DIR = Path("/mnt/workspace/hf_cache/hub")
# hf_utils.is_diffusion_model requires a truthy version key in model_index.json.
# This value is a synthetic marker, not the version of any library release.
SYNTHETIC_INDEX_VERSION = "0.0.0+h3.synthetic"
MANIFEST_NAME = "SYNTHETIC_MANIFEST.json"
PROVENANCE_NAME = "SYNTHETIC_PROVENANCE.json"
VLLM_OMNI_COMMIT = "e1d77a3"

BF16 = torch.bfloat16
F32 = torch.float32
_DTYPE_NAME = {BF16: "BF16", F32: "F32"}
_DTYPE_BYTES = {BF16: 2, F32: 4}
INIT_STD = 0.02

# vllm-omni source files cited in provenance (relative to the vllm-omni repo root).
PIPE = "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
DIT = "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py"
ENC = "vllm_omni/diffusion/models/minimax_h3/encoder.py"
VAE = "vllm_omni/diffusion/models/minimax_h3/vae.py"
TCH = "vllm_omni/diffusion/models/minimax_h3/temporal_chunks.py"
TREQ = "vllm_omni/diffusion/models/minimax_h3/time_request.py"
PACK = "vllm_omni/diffusion/models/minimax_h3/packed_tokens.py"
PRE = "vllm_omni/model_executor/models/minimax_h3/preprocessing.py"
ENCP = "vllm_omni/model_executor/models/minimax_h3/encoder_processing.py"
ARTE = "vllm_omni/model_executor/models/minimax_h3/text_encoder.py"
HFU = "vllm_omni/diffusion/utils/hf_utils.py"
META = "vllm_omni/diffusion/model_metadata.py"
DATA = "vllm_omni/diffusion/data.py"


# ---------------------------------------------------------------------------
# Provenance-carrying config values
# ---------------------------------------------------------------------------
class Fv:
    """A config leaf carrying its provenance."""

    __slots__ = ("value", "src")

    def __init__(self, value: Any, src: str) -> None:
        self.value = value
        self.src = src


def V(value: Any, path: str, line: str | int, note: str = "") -> Fv:
    return Fv(value, f"vllm-omni {path}:{line}" + (f" ({note})" if note else ""))


def S(value: Any, why: str) -> Fv:
    return Fv(value, f"SYNTHETIC: {why}")


def split_tree(tree: Any, prefix: str = "") -> tuple[Any, dict[str, str]]:
    """Split a tree of dicts with ``Fv`` leaves into value and provenance."""
    if isinstance(tree, Fv):
        return tree.value, {prefix: tree.src}
    if isinstance(tree, dict):
        plain: dict[str, Any] = {}
        prov: dict[str, str] = {}
        for key, value in tree.items():
            sub_plain, sub_prov = split_tree(
                value, f"{prefix}.{key}" if prefix else str(key)
            )
            plain[key] = sub_plain
            prov.update(sub_prov)
        return plain, prov
    raise TypeError(f"config value at {prefix!r} has no provenance: {tree!r}")


def stamp_field() -> Fv:
    return S(
        STAMP_VALUE,
        "marks every generated config/weight file as part of this synthetic build",
    )


# ---------------------------------------------------------------------------
# Architecture configs
# ---------------------------------------------------------------------------
def dit_config(tiny: bool) -> dict[str, Any]:
    """MiniMaxH3DiTArchConfig fields (defaults for full; overrides for --tiny)."""
    c: dict[str, Any] = {
        "_class_name": V(
            "MiniMaxH3DiTModel",
            DIT,
            1891,
            "EntryClass; avoids the Diffusers-name heuristic at :1287",
        ),
        "num_layers": V(50, DIT, 107),
        "token_refiner_num_layers": V(2, DIT, 108),
        "hidden_size": V(5376, DIT, 109),
        "num_attention_heads": V(56, DIT, 110),
        "attention_head_dim": V(128, DIT, 111),
        "ffn_hidden_size": V(14336, DIT, 112),
        "latents_dim": V(24, DIT, 113),
        "audio_latents_dim": V(32, DIT, 114),
        "patch_size": V([1, 2, 2], DIT, 115),
        "text_dim": V(5120, DIT, 116),
        "timestep_input_dim": V(256, DIT, 117),
        "time_embed_hidden_size": V(5376, DIT, 118),
        "time_embed_dim": V(2688, DIT, 119),
        "adaln_out_features": V(18 * 5376, DIT, 120),
        "final_adaln_out_features": V(2 * 5376, DIT, 121),
        "rope_inv_freq_len": V(16, DIT, 122),
        "norm_eps": V(1e-5, DIT, 123),
        "qk_norm_eps": V(1e-5, DIT, 124),
        "final_norm_eps": V(1e-5, DIT, 125),
        "rope_theta": V(
            10000.0, DIT, 1291, "default of config_mapping.get('rope_theta', 10000.0)"
        ),
        STAMP_KEY: stamp_field(),
    }
    if tiny:
        why = (
            "tiny CPU-validation override via MiniMaxH3DiTArchConfig.from_mapping "
            "(minimax_h3_transformer.py:128)"
        )
        c.update(
            num_layers=S(2, why),
            token_refiner_num_layers=S(1, why),
            hidden_size=S(256, why),
            num_attention_heads=S(
                2, why + "; head_dim kept at 128 >= rot_dim 6*16 (:448)"
            ),
            ffn_hidden_size=S(512, why),
            time_embed_hidden_size=S(256, why),
            time_embed_dim=S(128, why),
            adaln_out_features=S(18 * 256, why + "; must equal 6*hidden*3 (:801)"),
            final_adaln_out_features=S(2 * 256, why + "; must equal 2*hidden*1 (:801)"),
        )
    return c


TE_VOCAB_SIZE = 151680
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_ID = 151657
VISION_END_ID = 151658
EOS_ID = 151659
NUM_TOKENIZER_TOKENS = 151660


def vision_config() -> dict[str, Any]:
    small = (
        "small vision tower; vllm-omni pins no vision width "
        "(encoder.py:615-640 reads it from config)"
    )
    return {
        "model_type": S(
            "qwen3_vl",
            "sub-config tag of the Qwen3VLConfig class vllm-omni loads "
            "(encoder.py:1116-1118)",
        ),
        "depth": S(2, small),
        "hidden_size": S(64, small),
        "hidden_act": S(
            "gelu_pytorch_tanh",
            "vllm-omni's vision MLP uses tanh-GELU (encoder.py:527-529)",
        ),
        "intermediate_size": S(128, small),
        "num_heads": S(
            2,
            small
            + "; head_dim 32 is divisible by 4 for 2-D vision RoPE "
            + "(encoder.py:629-630)",
        ),
        "in_channels": S(3, "RGB pixel_values (encoder.py:476-477)"),
        "patch_size": S(16, "matches processor patch_size"),
        "spatial_merge_size": S(
            2, "matches processor merge_size (pipeline_minimax_h3.py:2563)"
        ),
        "temporal_patch_size": S(2, "matches processor temporal_patch_size"),
        "out_hidden_size": V(
            5120, ENC, 52, "merger output is scattered into 5120-wide text embeddings"
        ),
        "num_position_embeddings": S(
            64, "8x8 learned grid, perfect square required by encoder.py:627"
        ),
        "deepstack_visual_indexes": S(
            [0], "one DeepStack tap, index < depth (encoder.py:634-640)"
        ),
        "initializer_range": S(
            INIT_STD, "matches the generator's normal(0, 0.02) init"
        ),
    }


def text_config(tiny: bool) -> dict[str, Any]:
    c = {
        "model_type": S(
            "qwen3_vl_text",
            "text sub-config tag of the Qwen3VLConfig class (encoder.py:1116-1118)",
        ),
        "vocab_size": S(
            TE_VOCAB_SIZE,
            "smallest multiple of 128 >= synthetic tokenizer size 151660; "
            "divisible by text_encoder_tp_size<=8 (encoder.py:101)",
        ),
        "hidden_size": V(
            5120, ENC, 52, "MINIMAX_H3_QWEN3VL_HIDDEN_DIM, checked at :1338"
        ),
        "intermediate_size": S(
            256 if tiny else 1024,
            "modest MLP width, divisible by text_encoder_tp_size<=8 (encoder.py:152)",
        ),
        "num_hidden_layers": (
            S(
                2,
                "tiny CPU-validation depth "
                "(encoder keeps min(num_hidden_layers, 50), encoder.py:941)",
            )
            if tiny
            else V(
                50,
                ENC,
                "51,941",
                "MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER; deeper layers are never loaded",
            )
        ),
        "num_attention_heads": V(
            64, PIPE, "1082-1088", "text_encoder_tp_size must divide 64 query heads"
        ),
        "num_key_value_heads": V(
            8, PIPE, "1082-1088", "text_encoder_tp_size must divide 8 KV heads"
        ),
        "head_dim": V(
            128,
            ENC,
            445,
            "default mrope_section [24,20,20] covers 64 = head_dim/2 frequencies",
        ),
        "hidden_act": S("silu", "vllm-omni text MLP is silu(gate)*up (encoder.py:875)"),
        "max_position_embeddings": S(
            32768, "generous context bound for the AR text-encoder stage"
        ),
        "initializer_range": S(
            INIT_STD, "matches the generator's normal(0, 0.02) init"
        ),
        "rms_norm_eps": S(
            1e-6, "same as vllm-omni Qwen3-VL RMSNorm default eps (encoder.py:383)"
        ),
        "use_cache": S(False, "encoder-only use"),
        "tie_word_embeddings": S(
            False,
            "no lm_head is stored; the AR stage ties it itself "
            "(text_encoder.py:205-206)",
        ),
        "attention_bias": S(
            False, "q/k/v/o carry no bias in vllm-omni (encoder.py:227-235)"
        ),
        "attention_dropout": S(0.0, "inference only"),
        "rope_parameters": {
            "rope_type": S("default", "plain M-RoPE frequencies (encoder.py:447-449)"),
            "rope_theta": S(
                10000.0,
                "required key (encoder.py:444); value equals the DiT rope_theta "
                "default (minimax_h3_transformer.py:1291)",
            ),
            "mrope_section": V([24, 20, 20], ENC, 445),
            "mrope_interleaved": V(True, ENC, "424-431", "interleaved THW layout"),
        },
        "dtype": V("bfloat16", ENC, 1122),
    }
    return c


def text_encoder_config(tiny: bool) -> dict[str, Any]:
    return {
        "architectures": V(
            ["Qwen3VLForConditionalGeneration"],
            ARTE,
            179,
            "AR encoder stage subclasses this vLLM model",
        ),
        "model_type": S(
            "qwen3_vl",
            "tag of the Qwen3VLConfig class vllm-omni loads (encoder.py:1116-1118)",
        ),
        "image_token_id": V(IMAGE_TOKEN_ID, ENC, 1110),
        "video_token_id": V(VIDEO_TOKEN_ID, ENC, 1111),
        "vision_start_token_id": S(
            VISION_START_ID,
            "synthetic tokenizer id of <|vision_start|> (preprocessing.py:28)",
        ),
        "vision_end_token_id": S(
            VISION_END_ID,
            "synthetic tokenizer id of <|vision_end|> (preprocessing.py:29)",
        ),
        "tie_word_embeddings": S(False, "no lm_head is stored"),
        "dtype": V("bfloat16", ENC, 1122),
        "text_config": text_config(tiny),
        "vision_config": vision_config(),
        STAMP_KEY: stamp_field(),
    }


def partition_index(partition: str) -> dict[str, Any]:
    tasks = ["t2va", "fl2va"] if partition == "fl2va" else ["ref2va"]
    return {
        "_class_name": V(
            "MiniMaxH3Pipeline", META, 72, "registry key of the H3 pipeline"
        ),
        "_diffusers_version": V(
            SYNTHETIC_INDEX_VERSION,
            HFU,
            "154,168",
            "key must be truthy; value is a SYNTHETIC marker, not a library version",
        ),
        "_minimax_h3": {
            "partition": V(partition, PIPE, "946-949"),
            "tasks": V(
                tasks, PIPE, "951-953,964", "same as the code's fallback task set"
            ),
            "sigma_shift_scales": {
                "video": V(12.0, PIPE, 968),
                "audio": V(3.0, PIPE, 969),
            },
        },
        STAMP_KEY: stamp_field(),
    }


def root_index() -> dict[str, Any]:
    return {
        "_class_name": V(
            "MiniMaxH3Pipeline",
            META,
            72,
            "enrich_config reads _class_name (data.py:1639-1645)",
        ),
        "_diffusers_version": V(
            SYNTHETIC_INDEX_VERSION,
            HFU,
            "154,168",
            "key must be truthy; value is a SYNTHETIC marker, not a library version",
        ),
        STAMP_KEY: stamp_field(),
    }


VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
VIDEO_VAE_ARCH = {
    "latent_channels": VIDEO_LATENT_CHANNELS,
    "hidden_channels": 32,
    "spatial_ratio": 16,
    "temporal_ratio": 4,
    "clip_frames": 17,
    "tokens_per_clip": 5,
    "encode_token_drop": 3,
}
AUDIO_ENCODER_RATES = [2, 4, 5, 5, 4]
AUDIO_DECODER_RATES = [4, 5, 5, 4, 2]
AUDIO_DECODER_DIM = 64


def video_vae_config() -> dict[str, Any]:
    geo = "temporal geometry 17n+5 frames <-> 5n+2 latents"
    return {
        "latent_channels": V(
            VIDEO_LATENT_CHANNELS,
            PIPE,
            "1715,2059",
            "video latent channels pinned by the pipeline",
        ),
        "latents_mean": S(
            [0.0] * VIDEO_LATENT_CHANNELS,
            "identity normalisation; length must equal latent_channels (vae.py:85-89)",
        ),
        "latents_std": S(
            [1.0] * VIDEO_LATENT_CHANNELS,
            "identity normalisation; length must equal latent_channels (vae.py:85-89)",
        ),
        "auto_map": {
            "AutoModel": S(
                "modeling_h3_synthetic_video_vae.SyntheticH3VideoVAE",
                "our Apache-2.0 remote code; key required by vae.py:99-102",
            )
        },
        "source_class_name": S(
            "SyntheticH3VideoAutoencoder", "encoder-only loader key (vae.py:172-175)"
        ),
        "source_path": S(".", "encoder-only loader key (vae.py:182)"),
        "source_safetensors_path": S(
            "model.safetensors", "encoder-only loader key (vae.py:183)"
        ),
        "vae_clip_length": V(17, ENCP, "371,380", "17-frame causal clip; " + geo),
        "vae_token_drop": V(
            3, ENCP, "380,384", "the VAE drops the last 3 tokens; " + geo
        ),
        "vae_encoder_tiling": S(0, "untiled; key read by vae.py:187"),
        "vae_decoder_tiling": S(0, "untiled; key read by vae.py:188"),
        "vae_parallel_tiling": S(
            0, "no native tile parallelism; key read by vae.py:176,189"
        ),
        "vae_tile_size": S(0, "untiled; key read by vae.py:190"),
        "vae_tile_overlap_min": S(0, "untiled; key read by vae.py:191"),
        "vae_encoder_parallel": S(0, "key read by vae.py:192"),
        "vae_decoder_parallel": S(0, "key read by vae.py:193"),
        "vae_chunk_dim": S(
            2, "temporal axis of (B, C, T, H, W); key read by vae.py:194"
        ),
        "synthetic_architecture": {
            "latent_channels": V(VIDEO_LATENT_CHANNELS, PIPE, "1715,2059"),
            "hidden_channels": S(32, "tiny codec width"),
            "spatial_ratio": V(16, PIPE, "1669,2763-2764", "latent_h = height // 16"),
            "temporal_ratio": V(4, ENCP, 380, "token k covers frames 4k-3..4k"),
            "clip_frames": V(17, ENCP, 371),
            "tokens_per_clip": V(5, ENCP, "382-383", "token 0 + tokens 1..4 per clip"),
            "encode_token_drop": V(3, ENCP, 384),
        },
        STAMP_KEY: stamp_field(),
    }


def audio_vae_config() -> dict[str, Any]:
    return {
        "latent_channels": V(
            AUDIO_LATENT_CHANNELS, PIPE, "1731,1982", "audio latent rows are 32 wide"
        ),
        "latents_mean": S(
            [0.0] * AUDIO_LATENT_CHANNELS, "identity normalisation (vae.py:85-89)"
        ),
        "latents_std": S(
            [1.0] * AUDIO_LATENT_CHANNELS, "identity normalisation (vae.py:85-89)"
        ),
        "sample_rate": V(32000, VAE, "42,1113"),
        "audio_channels": V(2, VAE, 43, "stereo is encoded as batch 2"),
        "hop_length": V(
            800, TREQ, 36, "40 Hz latent at 32000 Hz -> 800 samples per latent frame"
        ),
        "auto_map": {
            "AutoModel": S(
                "modeling_h3_synthetic_audio_vae.SyntheticH3AudioVAE",
                "our Apache-2.0 remote code; key required by vae.py:99-102",
            )
        },
        "source_config_path": S(
            "source_config.yaml", "encoder-only loader key (vae.py:221)"
        ),
        "source_metadata_path": S(
            "source_metadata.json", "encoder-only loader key (vae.py:222)"
        ),
        "source_safetensors_path": S(
            "model.safetensors", "encoder-only loader key (vae.py:236)"
        ),
        STAMP_KEY: stamp_field(),
    }


def audio_source_config() -> dict[str, Any]:
    return {
        "model_config": {
            "decoder_dim": S(
                AUDIO_DECODER_DIM, "tiny decoder width; key read by vae.py:231"
            ),
            "vae_latent_channels": V(
                AUDIO_LATENT_CHANNELS, PIPE, "1731,1982", "key read by vae.py:232"
            ),
        },
        STAMP_KEY: stamp_field(),
    }


def audio_source_metadata() -> dict[str, Any]:
    rates = "product 800 = 32000 Hz / 40 Hz latent rate (time_request.py:34-36)"
    return {
        "metadata": {
            "kwargs": {
                "encoder_rates": S(
                    AUDIO_ENCODER_RATES, rates + "; key read by vae.py:227"
                ),
                "decoder_rates": S(
                    AUDIO_DECODER_RATES, rates + "; key read by vae.py:228"
                ),
                "attn_proj": S(
                    True,
                    "keeps a pre_block so decode-only removal of it succeeds "
                    "(vae.py:1100,1170-1171)",
                ),
                "decoder_type": S(
                    "synthetic_conv_transpose", "our decoder; key read by vae.py:230"
                ),
                "sample_rate": V(32000, VAE, 42, "key read by vae.py:233"),
            }
        },
        STAMP_KEY: stamp_field(),
    }


# Image/video processor values must agree with vision_config
# (patch 16, temporal 2, merge 2).
PROC_MIN_PIXELS = 4096
PROC_MAX_PIXELS = 262144
VIDEO_MAX_PIXELS = 2097152


def _processor_common() -> dict[str, Any]:
    return {
        "do_resize": S(True, "resize to the 32-pixel patch*merge grid"),
        "resample": S(3, "bicubic"),
        "do_rescale": S(True, "uint8 -> [0, 1]"),
        "rescale_factor": S(1.0 / 255.0, "uint8 -> [0, 1]"),
        "do_normalize": S(True, "normalise to [-1, 1]"),
        "image_mean": S([0.5, 0.5, 0.5], "symmetric normalisation"),
        "image_std": S([0.5, 0.5, 0.5], "symmetric normalisation"),
        "do_convert_rgb": S(True, "RGB input"),
        "patch_size": S(16, "equals vision_config.patch_size"),
        "temporal_patch_size": S(2, "equals vision_config.temporal_patch_size"),
        "merge_size": S(
            2,
            "equals vision_config.spatial_merge_size; "
            "read at pipeline_minimax_h3.py:2563",
        ),
        "processor_class": V("Qwen3VLProcessor", PIPE, "23,1063-1067"),
    }


def image_processor_config() -> dict[str, Any]:
    c = {
        "image_processor_type": S(
            "Qwen2VLImageProcessor",
            "patch/merge image processor used by Qwen3VLProcessor",
        ),
        "size": {
            "shortest_edge": S(
                PROC_MIN_PIXELS, "min pixel area = 4 merged 32x32 patches"
            ),
            "longest_edge": S(
                PROC_MAX_PIXELS, "max pixel area = 256 merged 32x32 patches"
            ),
        },
    }
    c.update(_processor_common())
    c[STAMP_KEY] = stamp_field()
    return c


def video_processor_config() -> dict[str, Any]:
    c = {
        "video_processor_type": S(
            "Qwen3VLVideoProcessor",
            "video processor used by Qwen3VLProcessor (pipeline_minimax_h3.py:2546)",
        ),
        "size": {
            "shortest_edge": S(PROC_MIN_PIXELS, "min pixel volume"),
            "longest_edge": S(VIDEO_MAX_PIXELS, "max pixel volume"),
        },
        "fps": S(2.0, "sampling fps when the processor samples frames itself"),
        "min_frames": S(2, "at least one temporal patch"),
        "max_frames": S(64, "small bound"),
        "do_sample_frames": V(
            False, PIPE, 2548, "pipeline always passes do_sample_frames=False"
        ),
        "return_metadata": S(False, "metadata not needed"),
    }
    c.update(_processor_common())
    c[STAMP_KEY] = stamp_field()
    return c


# ---------------------------------------------------------------------------
# Tokenizer (byte-level BPE, no merges, algorithmic filler, pinned specials)
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = {
    IMAGE_TOKEN_ID: (
        "<|image_pad|>",
        V(IMAGE_TOKEN_ID, ENC, 1110, "token text preprocessing.py:30"),
    ),
    VIDEO_TOKEN_ID: (
        "<|video_pad|>",
        V(VIDEO_TOKEN_ID, ENC, 1111, "token text preprocessing.py:31"),
    ),
    VISION_START_ID: (
        "<|vision_start|>",
        S(VISION_START_ID, "token text preprocessing.py:28; id synthetic"),
    ),
    VISION_END_ID: (
        "<|vision_end|>",
        S(VISION_END_ID, "token text preprocessing.py:29; id synthetic"),
    ),
    EOS_ID: (
        "<|endoftext|>",
        S(
            EOS_ID,
            "synthetic eos/pad; the AR stage emits eos_token_id (text_encoder.py:287)",
        ),
    ),
}


def _byte_to_unicode() -> dict[int, str]:
    """Reversible byte -> printable-unicode map used by byte BPE pre-tokenisers."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    codepoints = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {byte: chr(cp) for byte, cp in zip(printable, codepoints)}


def build_vocab() -> dict[str, int]:
    mapping = _byte_to_unicode()
    vocab: dict[str, int] = {}
    for byte in range(256):
        vocab[mapping[byte]] = byte
    for token_id in range(256, NUM_TOKENIZER_TOKENS):
        if token_id in SPECIAL_TOKENS:
            vocab[SPECIAL_TOKENS[token_id][0]] = token_id
        else:
            vocab[f"<h3filler{token_id}>"] = token_id
    if len(vocab) != NUM_TOKENIZER_TOKENS or sorted(vocab.values()) != list(
        range(NUM_TOKENIZER_TOKENS)
    ):
        raise AssertionError("synthetic vocab ids are not contiguous")
    return vocab


def _added_token(token_id: int) -> dict[str, Any]:
    return {
        "id": token_id,
        "content": SPECIAL_TOKENS[token_id][0],
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }


def tokenizer_json(vocab: dict[str, int]) -> dict[str, Any]:
    byte_level = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": True,
        "use_regex": True,
    }
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [_added_token(token_id) for token_id in sorted(SPECIAL_TOKENS)],
        "normalizer": None,
        "pre_tokenizer": dict(byte_level),
        "post_processor": dict(byte_level, trim_offsets=False),
        "decoder": dict(byte_level),
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [],
        },
    }


def tokenizer_config() -> dict[str, Any]:
    decoder = {}
    for token_id in sorted(SPECIAL_TOKENS):
        entry = _added_token(token_id)
        entry.pop("id")
        decoder[str(token_id)] = entry
    return {
        "tokenizer_class": V(
            "Qwen2Tokenizer", PIPE, "23,1058-1062", "pipeline loads Qwen2TokenizerFast"
        ),
        "processor_class": V("Qwen3VLProcessor", PIPE, "23,1063-1067"),
        "model_max_length": S(32768, "matches text_config.max_position_embeddings"),
        "bos_token": S(
            None,
            "no BOS; vllm-omni tokenises with add_special_tokens=False "
            "(preprocessing.py:182)",
        ),
        "eos_token": S(
            "<|endoftext|>", "synthetic eos (text_encoder.py:287 needs an eos id)"
        ),
        "pad_token": S("<|endoftext|>", "synthetic pad"),
        "unk_token": S(None, "byte-level vocab covers every byte"),
        "errors": S("replace", "byte-level decode error policy"),
        "add_prefix_space": S(
            False, "prompts pass through verbatim (preprocessing.py:9)"
        ),
        "clean_up_tokenization_spaces": S(False, "verbatim decode"),
        "split_special_tokens": S(False, "special tokens stay atomic"),
        "added_tokens_decoder": S(
            decoder, "ids pinned/synthetic as listed in SPECIAL_TOKENS"
        ),
        STAMP_KEY: stamp_field(),
    }


def special_tokens_map() -> dict[str, Any]:
    return {
        "eos_token": S("<|endoftext|>", "synthetic eos"),
        "pad_token": S("<|endoftext|>", "synthetic pad"),
    }


# ---------------------------------------------------------------------------
# Weight specs (names/shapes derived from the vllm-omni module code)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    init: str  # normal | ones | zeros | rope

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * _DTYPE_BYTES[self.dtype]


def dit_weight_spec(cfg: dict[str, Any]) -> list[TensorSpec]:
    """Checkpoint tensors of MiniMaxH3DiTModel (TP=1 shapes, native layout).

    Mirrors minimax_h3_transformer.py: MiniMaxH3DiTModel.__init__ (:1354-1411),
    MiniMaxH3Attention (:435-460), MiniMaxH3MLP (:751-771),
    MiniMaxH3AdalnProj (:818-826), MiniMaxH3TimeEmbedder (:349-366), the final
    layer (:1104-1132) and the persistent ``rope.inv_freq`` buffer (:314-318).
    The native loader (:1523-1595) uses parameter names as is. Fused qkv is
    stored grouped per head as [q, k, v] (:1559-1568) and fc1 as [gate; up].
    """
    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg["attention_head_dim"])
    inner = heads * head_dim
    ffn = int(cfg["ffn_hidden_size"])
    patch = [int(v) for v in cfg["patch_size"]]
    video_dim = int(cfg["latents_dim"]) * math.prod(patch)
    audio_dim = int(cfg["audio_latents_dim"])
    t_in = int(cfg["timestep_input_dim"])
    t_hidden = int(cfg["time_embed_hidden_size"])
    t_dim = int(cfg["time_embed_dim"])
    adaln = int(cfg["adaln_out_features"])
    final_adaln = int(cfg["final_adaln_out_features"])
    if adaln != 18 * hidden or final_adaln != 2 * hidden:
        raise ValueError(
            "adaln widths must be 18*hidden and 2*hidden "
            "(minimax_h3_transformer.py:801)"
        )
    if 6 * int(cfg["rope_inv_freq_len"]) > head_dim:
        raise ValueError("rot_dim 6*rope_inv_freq_len must fit in attention_head_dim")
    out: list[TensorSpec] = []

    def linear(
        name: str, out_f: int, in_f: int, dtype: torch.dtype, bias: bool
    ) -> None:
        out.append(TensorSpec(f"{name}.weight", (out_f, in_f), dtype, "normal"))
        if bias:
            out.append(TensorSpec(f"{name}.bias", (out_f,), dtype, "zeros"))

    def norm(name: str, size: int) -> None:
        out.append(TensorSpec(f"{name}.weight", (size,), BF16, "ones"))

    def attn_mlp(prefix: str) -> None:
        norm(f"{prefix}.norm1", hidden)
        norm(f"{prefix}.norm2", hidden)
        linear(f"{prefix}.attn.qkv_proj", 3 * inner, hidden, BF16, False)
        norm(f"{prefix}.attn.q_norm", head_dim)
        norm(f"{prefix}.attn.k_norm", head_dim)
        linear(f"{prefix}.attn.out_proj", hidden, inner, BF16, False)
        linear(f"{prefix}.mlp.fc1", 2 * ffn, hidden, BF16, False)
        linear(f"{prefix}.mlp.fc2", hidden, ffn, BF16, False)

    linear("video_patch_proj", hidden, video_dim, F32, True)
    linear("audio_patch_proj", hidden, audio_dim, F32, True)
    linear("condition_proj", hidden, int(cfg["text_dim"]), BF16, True)
    linear("time_embedder.proj_in", t_hidden, t_in, F32, True)
    linear("time_embedder.proj_out", t_dim, t_hidden, F32, True)
    out.append(
        TensorSpec("rope.inv_freq", (int(cfg["rope_inv_freq_len"]),), F32, "rope")
    )
    for index in range(int(cfg["token_refiner_num_layers"])):
        attn_mlp(f"token_refiner.blocks.{index}")
    norm("token_refiner.final_norm", hidden)
    for index in range(int(cfg["num_layers"])):
        attn_mlp(f"blocks.{index}")
        linear(f"blocks.{index}.adaln_proj.linear", adaln, t_dim, BF16, True)
    norm("final_layer.norm", hidden)
    linear("final_layer.adaln_proj.linear", final_adaln, t_dim, BF16, True)
    linear("final_layer.video_out", video_dim, hidden, F32, True)
    linear("final_layer.audio_out", audio_dim, hidden, F32, True)
    return out


SELECTED_LM_LAYER = 50  # encoder.py:51


def text_encoder_weight_spec(te_cfg: dict[str, Any]) -> list[TensorSpec]:
    """Checkpoint tensors of the Qwen3-VL text encoder, under HF names.

    MiniMaxH3Qwen3VLEncoder._map_weight_name (encoder.py:1147-1185) consumes
    them. Vision names mirror MiniMaxH3Qwen3VLVisionModel (encoder.py:469-640);
    text names are the unfused q/k/v and gate/up projections that the loader
    fuses. The loader skips ``lm_head``, the final norm and layers >= 50, so
    they are not stored.
    """
    vc = te_cfg["vision_config"]
    tc = te_cfg["text_config"]
    vh = int(vc["hidden_size"])
    vi = int(vc["intermediate_size"])
    merge_hidden = vh * int(vc["spatial_merge_size"]) ** 2
    out_hidden = int(vc["out_hidden_size"])
    out: list[TensorSpec] = []

    def linear(name: str, out_f: int, in_f: int, bias: bool) -> None:
        out.append(TensorSpec(f"{name}.weight", (out_f, in_f), BF16, "normal"))
        if bias:
            out.append(TensorSpec(f"{name}.bias", (out_f,), BF16, "zeros"))

    def layer_norm(name: str, size: int) -> None:
        out.append(TensorSpec(f"{name}.weight", (size,), BF16, "ones"))
        out.append(TensorSpec(f"{name}.bias", (size,), BF16, "zeros"))

    def merger(prefix: str, norm_size: int) -> None:
        layer_norm(f"{prefix}.norm", norm_size)
        linear(f"{prefix}.linear_fc1", merge_hidden, merge_hidden, True)
        linear(f"{prefix}.linear_fc2", out_hidden, merge_hidden, True)

    p = "model.visual"
    ps, tps = int(vc["patch_size"]), int(vc["temporal_patch_size"])
    out.append(
        TensorSpec(
            f"{p}.patch_embed.proj.weight",
            (vh, int(vc["in_channels"]), tps, ps, ps),
            BF16,
            "normal",
        )
    )
    out.append(TensorSpec(f"{p}.patch_embed.proj.bias", (vh,), BF16, "zeros"))
    out.append(
        TensorSpec(
            f"{p}.pos_embed.weight",
            (int(vc["num_position_embeddings"]), vh),
            BF16,
            "normal",
        )
    )
    for index in range(int(vc["depth"])):
        b = f"{p}.blocks.{index}"
        layer_norm(f"{b}.norm1", vh)
        layer_norm(f"{b}.norm2", vh)
        linear(f"{b}.attn.qkv", 3 * vh, vh, True)
        linear(f"{b}.attn.proj", vh, vh, True)
        linear(f"{b}.mlp.linear_fc1", vi, vh, True)
        linear(f"{b}.mlp.linear_fc2", vh, vi, True)
    merger(f"{p}.merger", vh)
    for index in range(len(vc["deepstack_visual_indexes"])):
        merger(f"{p}.deepstack_merger_list.{index}", merge_hidden)

    hidden = int(tc["hidden_size"])
    heads = int(tc["num_attention_heads"])
    kv_heads = int(tc["num_key_value_heads"])
    head_dim = int(tc["head_dim"])
    inter = int(tc["intermediate_size"])
    lm = "model.language_model"
    out.append(
        TensorSpec(
            f"{lm}.embed_tokens.weight", (int(tc["vocab_size"]), hidden), BF16, "normal"
        )
    )
    for index in range(min(int(tc["num_hidden_layers"]), SELECTED_LM_LAYER)):
        b = f"{lm}.layers.{index}"
        out.append(TensorSpec(f"{b}.input_layernorm.weight", (hidden,), BF16, "ones"))
        out.append(
            TensorSpec(f"{b}.post_attention_layernorm.weight", (hidden,), BF16, "ones")
        )
        linear(f"{b}.self_attn.q_proj", heads * head_dim, hidden, False)
        linear(f"{b}.self_attn.k_proj", kv_heads * head_dim, hidden, False)
        linear(f"{b}.self_attn.v_proj", kv_heads * head_dim, hidden, False)
        linear(f"{b}.self_attn.o_proj", hidden, heads * head_dim, False)
        out.append(
            TensorSpec(f"{b}.self_attn.q_norm.weight", (head_dim,), BF16, "ones")
        )
        out.append(
            TensorSpec(f"{b}.self_attn.k_norm.weight", (head_dim,), BF16, "ones")
        )
        linear(f"{b}.mlp.gate_proj", inter, hidden, False)
        linear(f"{b}.mlp.up_proj", inter, hidden, False)
        linear(f"{b}.mlp.down_proj", hidden, inter, False)
    return out


def _import_remote(module: str):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return importlib.import_module(f"remote_code.{module}")


def _module_weight_spec(model: torch.nn.Module) -> list[TensorSpec]:
    specs: list[TensorSpec] = []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}" if module_name else param_name
            if isinstance(module, torch.nn.LayerNorm) and param_name == "weight":
                init = "ones"
            elif param_name == "bias":
                init = "zeros"
            else:
                init = "normal"
            specs.append(TensorSpec(name, tuple(param.shape), F32, init))
    persistent = [name for name, _ in model.named_buffers()]
    if persistent:
        raise AssertionError(f"synthetic VAE must not carry buffers: {persistent}")
    return specs


def video_vae_weight_spec() -> list[TensorSpec]:
    mod = _import_remote("modeling_h3_synthetic_video_vae")
    with torch.device("meta"):
        model = mod.SyntheticH3VideoAutoencoder(**VIDEO_VAE_ARCH)
    return _module_weight_spec(model)


def audio_vae_weight_spec() -> list[TensorSpec]:
    mod = _import_remote("modeling_h3_synthetic_audio_vae")
    with torch.device("meta"):
        model = mod.DacAudioVAE(
            encoder_rates=AUDIO_ENCODER_RATES,
            decoder_rates=AUDIO_DECODER_RATES,
            attn_proj=True,
            decoder_type="synthetic_conv_transpose",
            decoder_dim=AUDIO_DECODER_DIM,
            vae_latent_channels=AUDIO_LATENT_CHANNELS,
            sample_rate=32000,
        )
    return _module_weight_spec(model)


# ---------------------------------------------------------------------------
# Tensor generation and streaming safetensors writer
# ---------------------------------------------------------------------------
def _tensor_seed(component_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{component_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def make_tensor(
    spec: TensorSpec, component_seed: int, rope_theta: float = 10000.0
) -> torch.Tensor:
    if spec.init == "ones":
        return torch.ones(spec.shape, dtype=spec.dtype)
    if spec.init == "zeros":
        return torch.zeros(spec.shape, dtype=spec.dtype)
    if spec.init == "rope":
        (length,) = spec.shape
        exponent = torch.arange(0, 2 * length, 2, dtype=torch.float32) / (2 * length)
        return (1.0 / (rope_theta**exponent)).to(spec.dtype)
    if spec.init != "normal":
        raise ValueError(spec.init)
    generator = torch.Generator(device="cpu").manual_seed(
        _tensor_seed(component_seed, spec.name)
    )
    values = torch.empty(spec.shape, dtype=torch.float32)
    values.normal_(0.0, INIT_STD, generator=generator)
    return values if spec.dtype == torch.float32 else values.to(spec.dtype)


def shard_specs(specs: list[TensorSpec], max_bytes: int) -> list[list[TensorSpec]]:
    shards: list[list[TensorSpec]] = []
    current: list[TensorSpec] = []
    size = 0
    for spec in specs:
        if current and size + spec.nbytes > max_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(spec)
        size += spec.nbytes
    if current:
        shards.append(current)
    return shards


def safetensors_header(specs: list[TensorSpec], metadata: dict[str, str]) -> bytes:
    header: dict[str, Any] = {"__metadata__": metadata}
    offset = 0
    for spec in specs:
        header[spec.name] = {
            "dtype": _DTYPE_NAME[spec.dtype],
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + spec.nbytes],
        }
        offset += spec.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return struct.pack("<Q", len(raw)) + raw


def write_safetensors_stream(
    path: Path,
    specs: list[TensorSpec],
    metadata: dict[str, str],
    make: Callable[[TensorSpec], torch.Tensor],
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "wb") as handle:
        head = safetensors_header(specs, metadata)
        handle.write(head)
        digest.update(head)
        size += len(head)
        for spec in specs:
            tensor = make(spec)
            if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
                raise AssertionError(
                    f"{spec.name}: generated {tuple(tensor.shape)}/{tensor.dtype}"
                )
            data = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
            view = memoryview(data)
            handle.write(view)
            digest.update(view)
            size += len(view)
            del tensor, data, view
    return digest.hexdigest(), size


# ---------------------------------------------------------------------------
# Output writer (hub layout with content-addressed blobs, or a plain directory)
# ---------------------------------------------------------------------------
class Writer:
    def __init__(self, *, hub_repo: Path | None, plain_dir: Path | None) -> None:
        if (hub_repo is None) == (plain_dir is None):
            raise ValueError("exactly one of hub_repo / plain_dir")
        self.hub_repo = hub_repo
        self.plain_dir = plain_dir
        if hub_repo is not None:
            self.root = hub_repo / "snapshots" / REVISION
            self.blobs = hub_repo / "blobs"
            self.blobs.mkdir(parents=True, exist_ok=True)
            (hub_repo / "refs").mkdir(parents=True, exist_ok=True)
        else:
            self.root = plain_dir
            self.blobs = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest: dict[str, dict[str, Any]] = {}

    def _tmp_path(self) -> Path:
        base = self.blobs if self.blobs is not None else self.root
        fd, name = tempfile.mkstemp(prefix=".h3tmp-", dir=str(base))
        os.close(fd)
        return Path(name)

    def _commit(self, rel: str, tmp: Path, sha: str, size: int, kind: str) -> None:
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.blobs is not None:
            blob = self.blobs / sha
            if blob.exists():
                tmp.unlink()
            else:
                os.replace(tmp, blob)
            if target.is_symlink() or target.exists():
                target.unlink()
            os.symlink(os.path.relpath(blob, start=target.parent), target)
        else:
            os.replace(tmp, target)
        self.manifest[rel] = {"sha256": sha, "size": size, "kind": kind}

    def write_bytes(self, rel: str, data: bytes, kind: str) -> None:
        tmp = self._tmp_path()
        tmp.write_bytes(data)
        self._commit(rel, tmp, hashlib.sha256(data).hexdigest(), len(data), kind)

    def write_json(self, rel: str, obj: Any, kind: str = "config") -> None:
        self.write_bytes(
            rel,
            (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
            kind,
        )

    def write_safetensors(
        self, rel: str, specs: list[TensorSpec], metadata: dict[str, str], make
    ) -> None:
        tmp = self._tmp_path()
        sha, size = write_safetensors_stream(tmp, specs, metadata, make)
        self._commit(rel, tmp, sha, size, "weights")

    def duplicate(self, src_rel: str, dst_rel: str) -> None:
        entry = self.manifest[src_rel]
        target = self.root / dst_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.blobs is not None:
            blob = self.blobs / entry["sha256"]
            os.symlink(os.path.relpath(blob, start=target.parent), target)
        else:
            shutil.copyfile(self.root / src_rel, target)
        self.manifest[dst_rel] = dict(entry)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
@dataclass
class BuildOptions:
    tiny: bool
    seeds: dict[str, int]
    max_shard_bytes: int
    workers: int
    configs_only: bool


def _weights_metadata(component: str, seed: int) -> dict[str, str]:
    return {
        STAMP_KEY: STAMP_VALUE,
        "format": "pt",
        "h3_component": component,
        "h3_seed": str(seed),
    }


def write_weight_component(
    writer: Writer,
    subdir: str,
    specs: list[TensorSpec],
    component: str,
    seed: int,
    opts: BuildOptions,
    provenance: dict[str, Any],
    *,
    single_file: str | None = None,
    rope_theta: float = 10000.0,
) -> None:
    metadata = _weights_metadata(component, seed)

    def make(spec: TensorSpec) -> torch.Tensor:
        return make_tensor(spec, seed, rope_theta)

    init_note = (
        f"normal(0, {INIT_STD}) matrices/convs/embeddings, "
        f"ones for norm gains, zeros for biases; "
        f"per-tensor seed = sha256('{seed}:<name>')"
    )
    if single_file is not None:
        rel = f"{subdir}/{single_file}"
        if not opts.configs_only:
            writer.write_safetensors(rel, specs, metadata, make)
        provenance["weights"][rel] = {
            "component": component,
            "seed": seed,
            "init": init_note,
            "tensors": len(specs),
        }
        return
    shards = shard_specs(specs, opts.max_shard_bytes)
    count = len(shards)
    names = [
        f"model-{index + 1:05d}-of-{count:05d}.safetensors" for index in range(count)
    ]
    weight_map = {
        spec.name: names[index] for index, shard in enumerate(shards) for spec in shard
    }
    total = sum(spec.nbytes for spec in specs)
    index_rel = f"{subdir}/model.safetensors.index.json"
    writer.write_json(
        index_rel,
        {
            "metadata": {"total_size": total, STAMP_KEY: STAMP_VALUE},
            "weight_map": weight_map,
        },
        kind="weights-index",
    )
    provenance["files"][index_rel] = {
        "__file__": "SYNTHETIC: shard index generated from the tensor spec"
    }
    if not opts.configs_only:
        jobs = [
            (f"{subdir}/{names[index]}", shard) for index, shard in enumerate(shards)
        ]
        if opts.workers > 1 and len(jobs) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=opts.workers
            ) as pool:
                futures = [
                    pool.submit(writer.write_safetensors, rel, shard, metadata, make)
                    for rel, shard in jobs
                ]
                for future in futures:
                    future.result()
        else:
            for rel, shard in jobs:
                writer.write_safetensors(rel, shard, metadata, make)
    provenance["weights"][subdir] = {
        "component": component,
        "seed": seed,
        "init": init_note,
        "tensors": len(specs),
        "bytes": total,
        "shards": count,
    }


def write_config(
    writer: Writer, rel: str, tree: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    plain, prov = split_tree(tree)
    writer.write_json(rel, plain)
    provenance["files"][rel] = prov
    return plain


def copy_remote_code(
    writer: Writer, subdir: str, files: Iterable[str], provenance: dict[str, Any]
) -> None:
    for name in files:
        data = (REMOTE_CODE_DIR / name).read_bytes()
        writer.write_bytes(f"{subdir}/{name}", data, "remote-code")
        provenance["files"][f"{subdir}/{name}"] = {
            "__file__": (
                "SYNTHETIC: Apache-2.0 remote code written for this build "
                "(pure torch.nn)"
            )
        }


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)
PROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json")


def build(writer: Writer, opts: BuildOptions) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "about": (
            "Fully synthetic MiniMax-H3-shaped checkpoint for vllm-omni. Every "
            "config value maps to a vllm-omni source location "
            f"(commit {VLLM_OMNI_COMMIT}) or is labelled SYNTHETIC. "
            "No upstream file was read."
        ),
        "vllm_omni_commit": VLLM_OMNI_COMMIT,
        "revision": REVISION,
        "revision_seed": REVISION_SEED,
        "build": "tiny" if opts.tiny else "full",
        "seeds": dict(opts.seeds),
        "geometry": {
            "fps": "vllm-omni pipeline_minimax_h3.py:189 (24)",
            "frames_to_video_latents": (
                "vllm-omni time_request.py:10-31 (17n+5 frames <-> 5n+2 latents)"
            ),
            "audio_latent_rate_hz": (
                "vllm-omni time_request.py:34-36 (40 Hz -> hop 800 at 32 kHz)"
            ),
            "spatial_ratio": (
                "vllm-omni pipeline_minimax_h3.py:1669,2763-2764 "
                "(latent = pixels // 16)"
            ),
            "video_patch": (
                "vllm-omni pipeline_minimax_h3.py:1722-1725 "
                "(patch (1,2,2), 24 channels -> 96-wide rows)"
            ),
            "audio_rows": (
                "vllm-omni pipeline_minimax_h3.py:1727-1732,2063-2068 "
                "(2 channels x audio_t rows of 32)"
            ),
        },
        "files": {},
        "weights": {},
    }
    t0 = time.time()

    # Root files.
    write_config(writer, "model_index.json", root_index(), provenance)
    dit_tree = dit_config(opts.tiny)
    dit_plain = write_config(
        writer, "FL2VA/transformer/config.json", dit_tree, provenance
    )
    te_plain = write_config(
        writer,
        "FL2VA/text_encoder/config.json",
        text_encoder_config(opts.tiny),
        provenance,
    )

    # FL2VA partition (complete).
    write_config(writer, "FL2VA/model_index.json", partition_index("fl2va"), provenance)
    write_weight_component(
        writer,
        "FL2VA/transformer",
        dit_weight_spec(dit_plain),
        "transformer_fl2va",
        opts.seeds["dit_fl2va"],
        opts,
        provenance,
        rope_theta=float(dit_plain["rope_theta"]),
    )
    write_weight_component(
        writer,
        "FL2VA/text_encoder",
        text_encoder_weight_spec(te_plain),
        "text_encoder",
        opts.seeds["text_encoder"],
        opts,
        provenance,
    )
    # Tokenizer + processor (+ copies inside text_encoder/ for the AR encoder
    # stage, which loads it as a model dir).
    vocab = build_vocab()
    tok_json = (json.dumps(tokenizer_json(vocab), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    vocab_json = (json.dumps(vocab, ensure_ascii=False) + "\n").encode("utf-8")
    writer.write_bytes("FL2VA/tokenizer/tokenizer.json", tok_json, "tokenizer")
    provenance["files"]["FL2VA/tokenizer/tokenizer.json"] = {
        "__file__": (
            "SYNTHETIC: byte-level BPE with no merges; ids 0-255 = bytes, "
            "256..151659 = algorithmic filler '<h3filler{id}>' except pinned "
            "specials <|image_pad|>=151655 (encoder.py:1110), "
            "<|video_pad|>=151656 (encoder.py:1111) and synthetic "
            "<|vision_start|>=151657, <|vision_end|>=151658, "
            "<|endoftext|>=151659 (token texts preprocessing.py:28-31)"
        )
    }
    writer.write_bytes("FL2VA/tokenizer/vocab.json", vocab_json, "tokenizer")
    provenance["files"]["FL2VA/tokenizer/vocab.json"] = {
        "__file__": "SYNTHETIC: same vocab as tokenizer.json"
    }
    writer.write_bytes("FL2VA/tokenizer/merges.txt", b"#version: 0.2\n", "tokenizer")
    provenance["files"]["FL2VA/tokenizer/merges.txt"] = {
        "__file__": "SYNTHETIC: empty merge list"
    }
    write_config(
        writer, "FL2VA/tokenizer/tokenizer_config.json", tokenizer_config(), provenance
    )
    write_config(
        writer,
        "FL2VA/tokenizer/special_tokens_map.json",
        special_tokens_map(),
        provenance,
    )
    write_config(
        writer,
        "FL2VA/processor/preprocessor_config.json",
        image_processor_config(),
        provenance,
    )
    write_config(
        writer,
        "FL2VA/processor/video_preprocessor_config.json",
        video_processor_config(),
        provenance,
    )
    for name in TOKENIZER_FILES:
        for subdir in ("FL2VA/processor", "FL2VA/text_encoder"):
            writer.duplicate(f"FL2VA/tokenizer/{name}", f"{subdir}/{name}")
            provenance["files"][f"{subdir}/{name}"] = provenance["files"][
                f"FL2VA/tokenizer/{name}"
            ]
    for name in PROCESSOR_FILES:
        writer.duplicate(f"FL2VA/processor/{name}", f"FL2VA/text_encoder/{name}")
        provenance["files"][f"FL2VA/text_encoder/{name}"] = provenance["files"][
            f"FL2VA/processor/{name}"
        ]

    # VAEs.
    write_config(writer, "FL2VA/video_vae/config.json", video_vae_config(), provenance)
    copy_remote_code(
        writer,
        "FL2VA/video_vae",
        ("modeling_h3_synthetic_video_vae.py", "parallel.py"),
        provenance,
    )
    write_weight_component(
        writer,
        "FL2VA/video_vae",
        video_vae_weight_spec(),
        "video_vae",
        opts.seeds["video_vae"],
        opts,
        provenance,
        single_file="model.safetensors",
    )
    write_config(writer, "FL2VA/audio_vae/config.json", audio_vae_config(), provenance)
    write_config(
        writer, "FL2VA/audio_vae/source_config.yaml", audio_source_config(), provenance
    )
    write_config(
        writer,
        "FL2VA/audio_vae/source_metadata.json",
        audio_source_metadata(),
        provenance,
    )
    copy_remote_code(
        writer, "FL2VA/audio_vae", ("modeling_h3_synthetic_audio_vae.py",), provenance
    )
    write_weight_component(
        writer,
        "FL2VA/audio_vae",
        audio_vae_weight_spec(),
        "audio_vae",
        opts.seeds["audio_vae"],
        opts,
        provenance,
        single_file="model.safetensors",
    )

    # Ref2VA partition: own transformer weights (distinct seed); every other
    # component is a byte-identical copy (same blob in the hub layout) so
    # Ref2VA-only startups ("Ref2VA/**", pipeline_minimax_h3.py:198-201) work.
    write_config(
        writer, "Ref2VA/model_index.json", partition_index("ref2va"), provenance
    )
    writer.duplicate("FL2VA/transformer/config.json", "Ref2VA/transformer/config.json")
    provenance["files"]["Ref2VA/transformer/config.json"] = provenance["files"][
        "FL2VA/transformer/config.json"
    ]
    write_weight_component(
        writer,
        "Ref2VA/transformer",
        dit_weight_spec(dit_plain),
        "transformer_ref2va",
        opts.seeds["dit_ref2va"],
        opts,
        provenance,
        rope_theta=float(dit_plain["rope_theta"]),
    )
    shared = ("text_encoder/", "tokenizer/", "processor/", "video_vae/", "audio_vae/")
    for rel in sorted(writer.manifest):
        if rel.startswith("FL2VA/") and rel[len("FL2VA/") :].startswith(shared):
            dst = "Ref2VA/" + rel[len("FL2VA/") :]
            writer.duplicate(rel, dst)
            if rel in provenance["files"]:
                provenance["files"][dst] = provenance["files"][rel]
    for key in [
        k
        for k in provenance["weights"]
        if k.startswith("FL2VA/") and not k.startswith("FL2VA/transformer")
    ]:
        provenance["weights"]["Ref2VA/" + key[len("FL2VA/") :]] = dict(
            provenance["weights"][key], copy_of=key
        )

    # Root transformer/config.json: read by OmniDiffusionConfig when --model
    # is the repo id / root (data.py:1575-1595 reads
    # "<model>/transformer/config.json").
    writer.duplicate("FL2VA/transformer/config.json", "transformer/config.json")
    provenance["files"]["transformer/config.json"] = provenance["files"][
        "FL2VA/transformer/config.json"
    ]

    writer.write_json(PROVENANCE_NAME, provenance, kind="provenance")
    manifest = {
        "revision": REVISION,
        "build": provenance["build"],
        "generator": "gen_minimax_h3_cache.py",
        STAMP_KEY: STAMP_VALUE,
        "files": dict(sorted(writer.manifest.items())),
    }
    writer.write_bytes(
        MANIFEST_NAME,
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
        "manifest",
    )
    return {
        "seconds": round(time.time() - t0, 1),
        "files": len(writer.manifest),
        "bytes": sum(e["size"] for e in writer.manifest.values()),
    }


def estimate(tiny: bool) -> dict[str, Any]:
    dit = split_tree(dit_config(tiny))[0]
    te = split_tree(text_encoder_config(tiny))[0]
    comps = {
        "transformer (per partition)": dit_weight_spec(dit),
        "text_encoder (shared blob)": text_encoder_weight_spec(te),
        "video_vae (shared blob)": video_vae_weight_spec(),
        "audio_vae (shared blob)": audio_vae_weight_spec(),
    }
    out = {}
    for name, specs in comps.items():
        out[name] = {
            "tensors": len(specs),
            "params": sum(math.prod(s.shape) for s in specs),
            "bytes": sum(s.nbytes for s in specs),
            "GiB": round(sum(s.nbytes for s in specs) / 2**30, 3),
        }
    unique = 2 * out["transformer (per partition)"]["bytes"] + sum(
        v["bytes"] for k, v in out.items() if not k.startswith("transformer")
    )
    out["unique_blob_bytes_total"] = unique
    out["unique_blob_GiB_total"] = round(unique / 2**30, 2)
    return out


def _make_writable(path: Path) -> None:
    for root, dirs, files in os.walk(path):
        os.chmod(root, os.stat(root).st_mode | stat.S_IWUSR)
        for name in files:
            full = os.path.join(root, name)
            if not os.path.islink(full):
                os.chmod(full, os.stat(full).st_mode | stat.S_IWUSR)


def make_readonly(paths: Iterable[Path]) -> None:
    for base in paths:
        for root, dirs, files in os.walk(base, topdown=False):
            for name in files:
                full = os.path.join(root, name)
                if not os.path.islink(full):
                    os.chmod(full, 0o444)
            os.chmod(root, 0o555)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-hub-dir", default=str(REAL_HUB_DIR), help="HF hub cache dir (hub layout)"
    )
    parser.add_argument(
        "--out-dir", default=None, help="write a plain snapshot directory here instead"
    )
    parser.add_argument(
        "--tiny", action="store_true", help="small dims for CPU validation"
    )
    parser.add_argument(
        "--configs-only",
        action="store_true",
        help="write every non-weight file (index json included)",
    )
    parser.add_argument(
        "--estimate", action="store_true", help="print per-component sizes and exit"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output produced by this generator",
    )
    parser.add_argument(
        "--allow-tiny-in-real-cache",
        action="store_true",
        help="allow a --tiny build in the default HF cache dir",
    )
    parser.add_argument(
        "--no-readonly", action="store_true", help="skip chmod -R a-w after a hub build"
    )
    parser.add_argument(
        "--max-shard-gib",
        type=float,
        default=None,
        help="max safetensors shard size in GiB (default 4, or 0.5 with --tiny)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="concurrent shard writers"
    )
    # Per-component weight seeds; main() requires them to be distinct.
    parser.add_argument("--seed-dit-fl2va", type=int, default=1001)
    parser.add_argument("--seed-dit-ref2va", type=int, default=2002)
    parser.add_argument("--seed-text-encoder", type=int, default=3003)
    parser.add_argument("--seed-video-vae", type=int, default=4004)
    parser.add_argument("--seed-audio-vae", type=int, default=5005)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.estimate:
        print(json.dumps(estimate(args.tiny), indent=2))
        return 0
    seeds = {
        "dit_fl2va": args.seed_dit_fl2va,
        "dit_ref2va": args.seed_dit_ref2va,
        "text_encoder": args.seed_text_encoder,
        "video_vae": args.seed_video_vae,
        "audio_vae": args.seed_audio_vae,
    }
    if len(set(seeds.values())) != len(seeds):
        raise SystemExit("component seeds must be distinct")
    max_gib = (
        args.max_shard_gib
        if args.max_shard_gib is not None
        else (0.5 if args.tiny else 4.0)
    )
    opts = BuildOptions(
        tiny=args.tiny,
        seeds=seeds,
        max_shard_bytes=int(max_gib * 2**30),
        workers=max(1, args.workers),
        configs_only=args.configs_only,
    )
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // opts.workers))

    if args.out_dir is not None:
        out = Path(args.out_dir)
        if out.exists() and any(out.iterdir()):
            if not args.force or not (out / MANIFEST_NAME).exists():
                raise SystemExit(
                    f"{out} is not empty (use --force only on a previous output "
                    "of this generator)"
                )
            _make_writable(out)
            shutil.rmtree(out)
        writer = Writer(hub_repo=None, plain_dir=out)
        hub_repo = None
    else:
        hub = Path(args.out_hub_dir)
        if (
            args.tiny
            and hub.resolve() == REAL_HUB_DIR.resolve()
            and not args.allow_tiny_in_real_cache
        ):
            raise SystemExit(
                "refusing to put a --tiny build into the real HF cache; "
                "pass --out-hub-dir or --out-dir"
            )
        hub_repo = hub / REPO_FOLDER
        if hub_repo.exists():
            ref = hub_repo / "refs" / "main"
            ours = ref.is_file() and ref.read_text().strip() == REVISION
            if not (args.force and ours):
                raise SystemExit(
                    f"{hub_repo} already exists "
                    f"({'synthetic' if ours else 'NOT produced by this generator'}); "
                    "refusing to touch it" + (" without --force" if ours else "")
                )
            _make_writable(hub_repo)
            shutil.rmtree(hub_repo)
        writer = Writer(hub_repo=hub_repo, plain_dir=None)

    summary = build(writer, opts)
    if hub_repo is not None:
        (hub_repo / "refs" / "main").write_text(REVISION)
        if not args.no_readonly:
            make_readonly([hub_repo / "snapshots", hub_repo / "blobs"])
    summary.update(
        layout="hub" if hub_repo is not None else "plain",
        snapshot=str(writer.root),
        revision=REVISION,
        build="tiny" if args.tiny else "full",
        configs_only=args.configs_only,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

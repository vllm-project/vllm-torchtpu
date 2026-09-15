"""
fixtures.py
-----------
Reusable fixtures for unit and integration tests.
Provides synthetic in-memory safetensors models, mock datasets, and dynamic discovery for real checkpoints.
"""

import glob
import json
import os
from typing import Dict, List

import torch
from safetensors.torch import save_file


def find_gemma4_checkpoint() -> str:
    """
    Dynamically discovers the Gemma 4 checkpoint path from environment variables,
    standard HuggingFace cache directories, or persistent storage.
    """
    # 1. Environment variable override
    env_path = os.environ.get("GEMMA4_MODEL_PATH")
    if env_path and os.path.isdir(env_path) and any(
            f.endswith(".safetensors") for f in os.listdir(env_path)):
        return env_path

    # 2. Dynamic glob over potential cache locations
    search_patterns = [
        "/mnt/pd/huggingface_cache/hub/models--google--gemma-4-31B-it/snapshots/*",
        "/mnt/pd/huggingface_cache/hub/models--google--gemma-4-31b-it/snapshots/*",
        "/mnt/pd/models/gemma-4-31b-it",
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/*"
        ),
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--google--gemma-4-31b-it/snapshots/*"
        ),
    ]
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        for match in sorted(matches, reverse=True):
            if os.path.isdir(match) and any(
                    os.path.isfile(os.path.join(match, f))
                    and f.endswith(".safetensors") for f in os.listdir(match)):
                return match
    return ""


def find_sparse_checkpoint() -> str:
    """
    Dynamically discovers the reference sparse checkpoint path from environment variables
    or common local storage directories.
    """
    env_path = os.environ.get("GEMMA4_SPARSE_MODEL_PATH")
    if env_path and os.path.isdir(env_path) and any(
            f.endswith(".safetensors") for f in os.listdir(env_path)):
        return env_path

    search_paths = [
        "/mnt/pd/sparse_checkpoints/gemma4_31b_2of4_256c4_full",
        "/mnt/pd/sparse_checkpoints/gemma4_31b_2of4_canonical_256c4_full",
        "/mnt/pd/sparse_checkpoints/gemma4_31b_2of4_full",
        "/mnt/pd/sparse_checkpoints/gemma4_31b_2of4_magnitude_full",
        os.path.expanduser("~/sparse_checkpoints/gemma4_31b_2of4_256c4_full"),
    ]
    for path in search_paths:
        if os.path.isdir(path) and any(
                f.endswith(".safetensors") for f in os.listdir(path)):
            return path
    return ""


REAL_GEMMA4_PATH = find_gemma4_checkpoint()
REAL_SPARSE_PATH = find_sparse_checkpoint()


def has_real_gemma4_weights() -> bool:
    """Returns True if a real Gemma 4 model checkpoint exists locally on disk."""
    return bool(REAL_GEMMA4_PATH) and os.path.isdir(REAL_GEMMA4_PATH) and any(
        os.path.isfile(os.path.join(REAL_GEMMA4_PATH, f))
        and f.endswith(".safetensors") for f in os.listdir(REAL_GEMMA4_PATH))


def has_real_sparse_weights() -> bool:
    """Returns True if a real Gemma 4 sparse checkpoint exists locally on disk."""
    return bool(REAL_SPARSE_PATH) and os.path.isdir(REAL_SPARSE_PATH) and any(
        f.endswith(".safetensors") for f in os.listdir(REAL_SPARSE_PATH))


def create_synthetic_model(
    output_dir: str,
    num_layers: int = 4,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_shards: int = 2,
    sparsity_ratio: float = 0.0,
) -> str:
    """
    Creates a valid synthetic transformer checkpoint with realistic tensor names and safetensors shards.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate config.json
    config = {
        "architectures": ["Gemma4ForCausalLM"],
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "torch_dtype": "bfloat16",
        "vocab_size": 256
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Distribute layers across shards
    tensors_per_shard: List[Dict[str,
                                 torch.Tensor]] = [{}
                                                   for _ in range(num_shards)]
    weight_map: Dict[str, str] = {}

    # Add embedding and final norm to shard 0
    embed_weight = torch.randn(256, hidden_size, dtype=torch.bfloat16)
    tensors_per_shard[0][
        "model.language_model.embed_tokens.weight"] = embed_weight
    weight_map[
        "model.language_model.embed_tokens.weight"] = "model-00001-of-00002.safetensors"

    final_norm = torch.ones(hidden_size, dtype=torch.bfloat16)
    tensors_per_shard[-1]["model.language_model.norm.weight"] = final_norm
    weight_map[
        "model.language_model.norm.weight"] = f"model-{num_shards:05d}-of-{num_shards:05d}.safetensors"

    for layer_idx in range(num_layers):
        shard_idx = layer_idx % num_shards
        shard_name = f"model-{shard_idx+1:05d}-of-{num_shards:05d}.safetensors"

        layer_prefix = f"model.language_model.layers.{layer_idx}"

        # Self attention weights
        q_proj = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16)
        k_proj = torch.randn(hidden_size // 2,
                             hidden_size,
                             dtype=torch.bfloat16)
        v_proj = torch.randn(hidden_size // 2,
                             hidden_size,
                             dtype=torch.bfloat16)
        o_proj = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16)

        # MLP weights
        gate_up = torch.randn(intermediate_size * 2,
                              hidden_size,
                              dtype=torch.bfloat16)
        down_proj = torch.randn(hidden_size,
                                intermediate_size,
                                dtype=torch.bfloat16)

        # 1D norms
        input_norm = torch.ones(hidden_size, dtype=torch.bfloat16)
        post_norm = torch.ones(hidden_size, dtype=torch.bfloat16)

        if sparsity_ratio > 0.0:
            # Apply 2:4 structured sparsity pattern along dim=-1
            for tensor in [q_proj, k_proj, v_proj, o_proj, gate_up, down_proj]:
                reshaped = tensor.view(-1, 4)
                mask = torch.ones_like(reshaped, dtype=torch.bool)
                mask[:, :2] = False  # Zero out 2 elements in every 4
                reshaped.masked_fill_(mask, 0.0)

        layer_weights = {
            f"{layer_prefix}.self_attn.q_proj.weight": q_proj,
            f"{layer_prefix}.self_attn.k_proj.weight": k_proj,
            f"{layer_prefix}.self_attn.v_proj.weight": v_proj,
            f"{layer_prefix}.self_attn.o_proj.weight": o_proj,
            f"{layer_prefix}.mlp.gate_up_proj.weight": gate_up,
            f"{layer_prefix}.mlp.down_proj.weight": down_proj,
            f"{layer_prefix}.input_layernorm.weight": input_norm,
            f"{layer_prefix}.post_attention_layernorm.weight": post_norm,
        }

        for k, v in layer_weights.items():
            tensors_per_shard[shard_idx][k] = v
            weight_map[k] = shard_name

    # Save safetensors shard files
    for shard_idx, tensors in enumerate(tensors_per_shard):
        shard_filename = f"model-{shard_idx+1:05d}-of-{num_shards:05d}.safetensors"
        save_file(tensors, os.path.join(output_dir, shard_filename))

    # Save model.safetensors.index.json
    index_data = {
        "metadata": {
            "total_size": 1024 * 1024
        },
        "weight_map": weight_map
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"),
              "w") as f:
        json.dump(index_data, f, indent=2)

    return output_dir

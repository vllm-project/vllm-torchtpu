"""
naive_magnitude_pruner.py
-------------------------
Standalone Naive N:M Structured Magnitude Pruning script and utility functions.
Applies data-free top-k magnitude pruning along the contracting dimension (dim=-1)
without requiring calibration data or Hessian computation.
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def prune_tensor_magnitude_nm(w: torch.Tensor,
                              prunen: int = 2,
                              prunem: int = 4) -> torch.Tensor:
    """
    Applies naive N:M structured magnitude pruning along the contracting dimension (dim=-1).
    Keeps the N weights with largest absolute magnitude in every M block.
    For 2:4: keeps top 2, zeros out 2 smallest.
    """
    orig_shape = w.shape
    orig_dtype = w.dtype

    # Reshape into contiguous M-blocks for vector ranking
    w_flat = w.contiguous().view(-1, prunem)
    w_abs = torch.abs(w_flat.float())

    # Double argsort yields rank positions (0 is smallest, prunem-1 is largest)
    assert w.shape[-1] % prunem == 0, (
        f"Contracting dimension size ({w.shape[-1]}) must be divisible by {prunem} "
        f"to avoid grouping across row boundaries.")
    # Double argsort yields rank positions (0 is smallest, prunem-1 is largest).
    # stable=True makes tie-breaks deterministic and portable across backends.
    ranks = torch.argsort(torch.argsort(w_abs, dim=-1, stable=True), dim=-1)

    # Keep the N largest: ranks >= M - N (2:4 keeps top 2, 1:4 keeps top 1)
    mask = ranks >= prunem - prunen

    # Apply mask and restore original tensor shape and dtype
    w_pruned = (w_flat * mask).view(orig_shape).to(orig_dtype)
    return w_pruned


def prune_model_magnitude(base_model_dir: str,
                          output_dir: str,
                          prunen: int = 2,
                          prunem: int = 4,
                          min_layer: int = 0,
                          max_layer: int = 60) -> None:
    """
    Streams safetensors shards from base_model_dir, applies N:M magnitude pruning to linear
    projection weights in layers [min_layer, max_layer), and saves to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=======================================================")
    print(
        f">>> [NAIVE MAGNITUDE PRUNER] {prunen}:{prunem} Structured Sparsity")
    print(f">>> Source:      {base_model_dir}")
    print(f">>> Destination: {output_dir}")
    print(f">>> Layers:      {min_layer} to {max_layer - 1}")
    print("=======================================================")

    # 1. Copy config, tokenizer, and architecture JSON files
    for fname in os.listdir(base_model_dir):
        src_path = os.path.join(base_model_dir, fname)
        if os.path.isfile(src_path) and not fname.endswith(
                ".safetensors") and not fname.endswith(".bin"):
            shutil.copy2(src_path, os.path.join(output_dir, fname))

    # 2. Check safetensors index or list shards directly
    index_path = os.path.join(base_model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shard_files = sorted(list(set(weight_map.values())))
    else:
        shard_files = sorted([
            f for f in os.listdir(base_model_dir) if f.endswith(".safetensors")
        ])

    total_pruned_tensors = 0
    total_unpruned_tensors = 0
    total_pruned_params = 0
    total_unpruned_params = 0

    # 3. Process each safetensors shard sequentially
    for shard_name in shard_files:
        src_shard_path = os.path.join(base_model_dir, shard_name)
        dst_shard_path = os.path.join(output_dir, shard_name)
        print(f"\n>>> Processing shard: {shard_name}...")

        tensors = load_file(src_shard_path)
        out_tensors = {}

        for k, v in tensors.items():
            is_target_lm_layer = False
            # Check if tensor belongs to target language model linear layers
            if "language_model.layers." in k and "weight" in k and not any(
                    skip in k
                    for skip in ["norm", "embed", "lm_head", "layer_scalar"]):
                if v.ndim == 3:
                    raise NotImplementedError(
                        f"Found 3-D tensor '{k}' (shape {list(v.shape)}) under language_model.layers. "
                        "Fused MoE expert stacks are not supported in naive magnitude pruning."
                    )
                if v.ndim == 2:
                    try:
                        parts = k.split("language_model.layers.")[1].split(".")
                        layer_idx = int(parts[0])
                        if min_layer <= layer_idx < max_layer:
                            is_target_lm_layer = True
                    except (ValueError, IndexError):
                        pass

            if is_target_lm_layer:
                v_pruned = prune_tensor_magnitude_nm(v,
                                                     prunen=prunen,
                                                     prunem=prunem)
                out_tensors[k] = v_pruned
                total_pruned_tensors += 1
                total_pruned_params += v.numel()
            else:
                out_tensors[k] = v
                total_unpruned_tensors += 1
                total_unpruned_params += v.numel()

        print(
            f">>> Saving pruned shard: {dst_shard_path} ({len(out_tensors)} tensors)"
        )
        save_file(out_tensors, dst_shard_path)

    print("\n=======================================================")
    print(">>> [SUCCESS] Magnitude Pruning Completed!")
    if total_pruned_tensors == 0:
        raise RuntimeError(
            f"No tensors matched the pruning filter in {base_model_dir}. "
            "Check the parameter-name prefix and --minlayer/--maxlayer.")

    print(
        f">>> Total Pruned Linear Projections:   {total_pruned_tensors} ({total_pruned_params / 1e9:.2f}B params)"
    )
    print(
        f">>> Total Unpruned Tensors (Dense):     {total_unpruned_tensors} ({total_unpruned_params / 1e9:.2f}B params)"
    )
    print(f">>> Saved Checkpoint:                   {output_dir}")
    print("=======================================================\n")


def main():
    """CLI entry point for standalone magnitude pruning."""
    parser = argparse.ArgumentParser(
        description="Naive N:M Magnitude Pruner for LLMs")
    parser.add_argument("--model",
                        required=True,
                        help="Base dense model directory")
    parser.add_argument("--save",
                        required=True,
                        help="Output destination directory")
    parser.add_argument("--prunen",
                        type=int,
                        default=2,
                        help="N in N:M (default 2)")
    parser.add_argument("--prunem",
                        type=int,
                        default=4,
                        help="M in N:M (default 4)")
    parser.add_argument("--minlayer",
                        type=int,
                        default=0,
                        help="Start layer index (inclusive)")
    parser.add_argument("--maxlayer",
                        type=int,
                        default=60,
                        help="End layer index (exclusive)")
    args = parser.parse_args()

    prune_model_magnitude(base_model_dir=args.model,
                          output_dir=args.save,
                          prunen=args.prunen,
                          prunem=args.prunem,
                          min_layer=args.minlayer,
                          max_layer=args.maxlayer)


if __name__ == "__main__":
    main()

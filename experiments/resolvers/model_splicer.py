import argparse
import json
import os
import re
import shutil
from typing import Dict, Optional, Set

import torch
from core.models import ModelArchitectureSpec
from safetensors import safe_open
from safetensors.torch import save_file


class ModelSplicer:

    @staticmethod
    def detect_architecture(model_dir: str) -> ModelArchitectureSpec:
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {model_dir}")

        with open(config_path, "r") as f:
            cfg = json.load(f)

        text_cfg = cfg.get("text_config", cfg)
        num_layers = text_cfg.get(
            "num_hidden_layers",
            cfg.get("num_hidden_layers", cfg.get("num_layers", 60)))
        hidden_size = text_cfg.get("hidden_size", cfg.get("hidden_size", 4096))
        model_type = cfg.get("model_type",
                             cfg.get("architectures", ["Unknown"])[0])

        return ModelArchitectureSpec(model_type=model_type,
                                     num_layers=num_layers,
                                     hidden_size=hidden_size,
                                     layer_prefix="layers",
                                     config_dict=cfg)

    @classmethod
    def extract_layer_idx(cls, key: str) -> Optional[int]:
        # Matches .layers.0., .language_model.layers.0., etc.
        match = re.search(r"\.(?:layers|h|block)\.(\d+)\.", key)
        if match:
            # Check if this is part of vision_tower
            if "vision_tower" in key or "embed_vision" in key:
                return None
            return int(match.group(1))
        return None

    @classmethod
    def parse_layer_indices_str(cls,
                                layer_str: str,
                                total_layers: int = 60) -> Set[int]:
        indices: Set[int] = set()
        for part in layer_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-")
                indices.update(range(int(start), int(end) + 1))
            else:
                indices.add(int(part))
        return {i for i in indices if 0 <= i < total_layers}

    @staticmethod
    def get_weight_map(directory: str) -> Dict[str, str]:
        index_path = os.path.join(directory, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                idx = json.load(f)
                if "weight_map" in idx and len(idx["weight_map"]) > 0:
                    return idx["weight_map"]

        # Fallback: scan safetensors headers
        weight_map = {}
        for fname in os.listdir(directory):
            if fname.endswith(".safetensors"):
                fpath = os.path.join(directory, fname)
                with safe_open(fpath, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        weight_map[k] = fname
        return weight_map

    @classmethod
    def splice_checkpoint(cls, dense_model_dir: str,
                          fully_sparse_model_dir: str,
                          sparse_layer_indices: Set[int],
                          output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        print("\n=======================================================")
        print(
            f">>> [SPLICER] Splicing Sparse Layers: {sorted(list(sparse_layer_indices))}"
        )
        print(f">>> Dense Source:  {dense_model_dir}")
        print(f">>> Sparse Source: {fully_sparse_model_dir}")
        print(f">>> Destination:   {output_dir}")
        print("=======================================================")

        # Copy non-safetensors metadata and config files
        for fname in os.listdir(dense_model_dir):
            if not fname.endswith(".safetensors") and not fname.endswith(
                    ".bin"):
                src_path = os.path.join(dense_model_dir, fname)
                dst_path = os.path.join(output_dir, fname)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)

        dense_weight_map = cls.get_weight_map(dense_model_dir)
        sparse_weight_map = cls.get_weight_map(fully_sparse_model_dir)

        dense_safetensors = sorted(list(set(dense_weight_map.values())))
        if not dense_safetensors:
            dense_safetensors = sorted([
                f for f in os.listdir(dense_model_dir)
                if f.endswith(".safetensors")
            ])

        # Pre-open all sparse shards for fast random access
        sparse_handles = {}
        for sparse_file in set(sparse_weight_map.values()):
            sparse_file_path = os.path.join(fully_sparse_model_dir,
                                            sparse_file)
            sparse_handles[sparse_file] = safe_open(sparse_file_path,
                                                    framework="pt",
                                                    device="cpu")

        replaced_count = 0
        kept_dense_count = 0

        try:
            for shard_name in dense_safetensors:
                dense_shard_path = os.path.join(dense_model_dir, shard_name)
                output_shard_path = os.path.join(output_dir, shard_name)

                merged_shard: Dict[str, torch.Tensor] = {}

                with safe_open(dense_shard_path, framework="pt",
                               device="cpu") as dense_handle:
                    for k in dense_handle.keys():
                        layer_idx = cls.extract_layer_idx(k)
                        if layer_idx is not None and layer_idx in sparse_layer_indices and k in sparse_weight_map:
                            sparse_shard_name = sparse_weight_map[k]
                            merged_shard[k] = sparse_handles[
                                sparse_shard_name].get_tensor(k)
                            replaced_count += 1
                        else:
                            merged_shard[k] = dense_handle.get_tensor(k)
                            kept_dense_count += 1

                save_file(merged_shard, output_shard_path)
                print(
                    f">>> [SPLICER] Wrote merged shard {shard_name} ({len(merged_shard)} tensors)"
                )
        finally:
            # Handles are closed on exit
            del sparse_handles

        print(
            f">>> [SPLICER] Splicing complete: {replaced_count} tensors from sparse, {kept_dense_count} tensors from dense."
        )
        print(f">>> [SPLICER] Saved into {output_dir}\n")
        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Zero-Recompute Model Layer Splicer")
    parser.add_argument("--dense-model",
                        required=True,
                        help="Path to unpruned base model")
    parser.add_argument("--sparse-model",
                        required=True,
                        help="Path to 100% sparse reference model")
    parser.add_argument("--sparse-layers",
                        required=True,
                        help="Sparse layer indices, e.g. 0-9,50-59 or 55-59")
    parser.add_argument("--output-dir",
                        required=True,
                        help="Directory to save spliced checkpoint")
    args = parser.parse_args()

    arch = ModelSplicer.detect_architecture(args.dense_model)
    sparse_indices = ModelSplicer.parse_layer_indices_str(
        args.sparse_layers, total_layers=arch.num_layers)
    ModelSplicer.splice_checkpoint(dense_model_dir=args.dense_model,
                                   fully_sparse_model_dir=args.sparse_model,
                                   sparse_layer_indices=sparse_indices,
                                   output_dir=args.output_dir)


if __name__ == "__main__":
    main()

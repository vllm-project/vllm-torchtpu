import argparse
import contextlib
import json
import os
import re
import shutil
from typing import Any, Dict, Optional, Set

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

        replaced_count = 0
        kept_dense_count = 0

        # Open sparse shard handles lazily on-demand and guarantee clean release via ExitStack.
        # This prevents holding open descriptors for unreferenced shards while avoiding
        # the performance penalty of re-opening and parsing headers per tensor.
        with contextlib.ExitStack() as stack:
            sparse_handles: Dict[str, Any] = {}

            def get_sparse_handle(file_name: str):
                if file_name not in sparse_handles:
                    sparse_file_path = os.path.join(fully_sparse_model_dir,
                                                    file_name)
                    sparse_handles[file_name] = stack.enter_context(
                        safe_open(sparse_file_path,
                                  framework="pt",
                                  device="cpu"))
                return sparse_handles[file_name]

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
                            merged_shard[k] = get_sparse_handle(
                                sparse_shard_name).get_tensor(k)
                            replaced_count += 1
                        else:
                            merged_shard[k] = dense_handle.get_tensor(k)
                            kept_dense_count += 1

                save_file(merged_shard, output_shard_path)
                print(
                    f">>> [SPLICER] Wrote merged shard {shard_name} ({len(merged_shard)} tensors)"
                )

        print(
            f">>> [SPLICER] Splicing complete: {replaced_count} tensors from sparse, {kept_dense_count} tensors from dense."
        )
        print(f">>> [SPLICER] Saved into {output_dir}\n")
        return output_dir

    @classmethod
    def save_substituted_checkpoint(
            cls,
            base_model_dir: str,
            output_dir: str,
            replacement_state_dict: Dict[str, torch.Tensor],
            target_layer_indices: Optional[Set[int]] = None) -> str:
        """Write a new checkpoint to output_dir by copying base_model_dir
        and replacing pruned layer weights from replacement_state_dict.

        Each safetensors shard in base_model_dir is read tensor by tensor:
        - If a tensor's key is in replacement_state_dict (and its layer
          index is in target_layer_indices, when specified), the new tensor
          from replacement_state_dict is written.
        - Otherwise, the original tensor from base_model_dir is copied
          unchanged (preserving unpruned layers, embeddings, and vision weights).

        Non-weight files (config.json, tokenizer files, shard index) are
        copied directly from base_model_dir.

        Raises:
            ValueError: If no tensors in the shards matched
                replacement_state_dict (for example, due to mismatched
                key prefixes), which would otherwise silently write an
                unpruned copy of the base model.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Copy non-safetensors metadata, tokenizer, and config files
        for fname in os.listdir(base_model_dir):
            if not fname.endswith(".safetensors") and not fname.endswith(
                    ".bin"):
                src_path = os.path.join(base_model_dir, fname)
                dst_path = os.path.join(output_dir, fname)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)

        # 2. Identify all safetensors shards from base model
        dense_weight_map = cls.get_weight_map(base_model_dir)
        dense_safetensors = sorted(list(set(dense_weight_map.values())))
        if not dense_safetensors:
            dense_safetensors = sorted([
                f for f in os.listdir(base_model_dir)
                if f.endswith(".safetensors")
            ])

        replaced_count = 0
        kept_dense_count = 0

        # 3. Stream each shard, substitute calibrated weights, preserve non-layer and unpruned weights
        for shard_name in dense_safetensors:
            dense_shard_path = os.path.join(base_model_dir, shard_name)
            output_shard_path = os.path.join(output_dir, shard_name)

            merged_shard: Dict[str, torch.Tensor] = {}

            with safe_open(dense_shard_path, framework="pt",
                           device="cpu") as dense_handle:
                for k in dense_handle.keys():
                    layer_idx = cls.extract_layer_idx(k)
                    should_substitute = (k in replacement_state_dict and
                                         (target_layer_indices is None or
                                          (layer_idx is not None and layer_idx
                                           in target_layer_indices)))
                    if should_substitute:
                        merged_shard[k] = replacement_state_dict[k].detach(
                        ).cpu().contiguous()
                        replaced_count += 1
                    else:
                        merged_shard[k] = dense_handle.get_tensor(k)
                        kept_dense_count += 1

            save_file(merged_shard, output_shard_path)
            print(
                f">>> [SPLICER] Wrote merged shard {shard_name} ({len(merged_shard)} tensors)"
            )

        if replaced_count == 0:
            targets = (sorted(target_layer_indices)
                       if target_layer_indices is not None else "ALL")
            raise ValueError(
                f"save_substituted_checkpoint replaced 0 tensors in "
                f"'{output_dir}': no key of replacement_state_dict matched a "
                f"shard key of '{base_model_dir}' within target layers "
                f"{targets}. The output would be a dense copy mislabelled "
                "as sparse.")

        print(
            f">>> [SPLICER] Calibrated checkpoint saved: {replaced_count} substituted, {kept_dense_count} preserved from base."
        )
        print(f">>> [SPLICER] Saved into {output_dir}\n")
        return output_dir

    @staticmethod
    def _read_shard_keys(shard_path: str) -> Set[str]:
        """
        Returns the tensor keys actually stored in one safetensors shard.

        Reads only the header, then checks that the file is exactly as long as
        the header says, so a shard cut short by a failed write is rejected
        rather than trusted. Raises ValueError if the shard is truncated or its
        header is unreadable.
        """
        file_size = os.path.getsize(shard_path)
        with open(shard_path, "rb") as f:
            prefix = f.read(8)
            if len(prefix) < 8:
                raise ValueError(
                    f"'{shard_path}' is truncated: {file_size} bytes cannot hold a safetensors header"
                )
            header_len = int.from_bytes(prefix, "little")
            if header_len > file_size - 8:
                raise ValueError(
                    f"'{shard_path}' is truncated: header declares {header_len} bytes, file has {file_size}"
                )
            try:
                header = json.loads(f.read(header_len))
            except ValueError as e:
                raise ValueError(
                    f"'{shard_path}' has an unreadable safetensors header: {e}"
                ) from e
        if not isinstance(header, dict):
            raise ValueError(
                f"'{shard_path}' has an unreadable safetensors header")

        tensors = {
            k: v
            for k, v in header.items()
            if k != "__metadata__" and isinstance(v, dict)
        }
        try:
            data_end = max((int(v["data_offsets"][1])
                            for v in tensors.values() if "data_offsets" in v),
                           default=0)
        except (TypeError, IndexError, ValueError) as e:
            raise ValueError(
                f"'{shard_path}' has an unreadable safetensors header: {e}"
            ) from e
        expected_size = 8 + header_len + data_end
        if file_size != expected_size:
            raise ValueError(
                f"'{shard_path}' is truncated: header implies {expected_size} bytes, file has {file_size}"
            )
        return set(tensors)

    @classmethod
    def assert_checkpoint_integrity(cls, base_model_dir: str,
                                    checkpoint_dir: str) -> None:
        """
        Asserts that checkpoint_dir has an identical set of safetensors keys to base_model_dir.
        Fails loudly naming all missing and extra keys before evaluation launches.

        The checkpoint's keys are read from the shard files on disk, never from
        its model.safetensors.index.json: save_substituted_checkpoint copies the
        base model's index before writing any shard, so a save that dies part
        way (e.g. ENOSPC on the last shard) leaves an index that lists every
        key. Every shard the index names must exist and be complete.
        """
        base_map = cls.get_weight_map(base_model_dir)
        base_keys = set(base_map.keys())

        index_path = os.path.join(checkpoint_dir,
                                  "model.safetensors.index.json")
        index_map: Dict[str, str] = {}
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    index_map = json.load(f).get("weight_map") or {}
                if not isinstance(index_map, dict):
                    raise ValueError("weight_map is not an object")
            except (OSError, ValueError, AttributeError) as e:
                raise ValueError(
                    f"Checkpoint integrity assertion failed for '{checkpoint_dir}': "
                    f"unreadable index {index_path}: {e}") from e
        if index_map:
            shard_names = sorted(set(index_map.values()))
        else:
            shard_names = sorted(f for f in os.listdir(checkpoint_dir)
                                 if f.endswith(".safetensors"))

        missing_shards = []
        broken_shards = []
        ckpt_keys: Set[str] = set()
        for shard_name in shard_names:
            shard_path = os.path.join(checkpoint_dir, shard_name)
            if not os.path.isfile(shard_path):
                missing_shards.append(shard_name)
                continue
            try:
                ckpt_keys |= cls._read_shard_keys(shard_path)
            except ValueError as e:
                broken_shards.append(str(e))

        missing_keys = sorted(base_keys - ckpt_keys)
        extra_keys = sorted(ckpt_keys - base_keys)

        if missing_shards or broken_shards or missing_keys or extra_keys:
            error_lines = [
                f"Checkpoint integrity assertion failed for '{checkpoint_dir}' against base '{base_model_dir}':"
            ]
            if missing_shards:
                error_lines.append(
                    f"  Missing shard files ({len(missing_shards)}): {missing_shards}"
                )
            if broken_shards:
                error_lines.append(
                    f"  Unreadable shards ({len(broken_shards)}): {broken_shards}"
                )
            if missing_keys:
                error_lines.append(
                    f"  Missing keys ({len(missing_keys)}): {missing_keys}")
            if extra_keys:
                error_lines.append(
                    f"  Extra keys ({len(extra_keys)}): {extra_keys}")
            raise ValueError("\n".join(error_lines))

        print(
            f">>> [INTEGRITY GATE] Verified {len(ckpt_keys)} safetensors keys match base model exactly."
        )


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

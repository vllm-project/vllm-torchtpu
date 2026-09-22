"""
weight_scaler_resolver.py
-------------------------
Dynamic model resolver for constant weight scaling on structured sparse checkpoints.
Applies scalar multiplication (W <- c * W) strictly to 2D linear projection weights
while preserving non-linear normalization scales, embeddings, and lm_head at c = 1.0.
"""

import glob
import os
import shutil
from typing import Any, Dict, Set, Tuple

from resolvers.base_resolver import BaseResolver
from safetensors import safe_open
from safetensors.torch import save_file


class WeightScalerResolver(BaseResolver):
    """
    Dynamically scales non-zero weights of a sparse model by a constant scalar multiplier c.
    Streams shards sequentially to bound memory consumption on Cloud TPU VMs.
    """
    CONSUMED_KEYS: Set[str] = {"scale_factor"}

    def _cleanup_old_scaled_dirs(self, output_base_path: str,
                                 target_dir: str) -> None:
        """Removes any stale scaled directories matching output_base_path_* to prevent disk overflow."""
        pattern = f"{output_base_path}_*"
        for existing_dir in glob.glob(pattern):
            if existing_dir != target_dir and os.path.isdir(existing_dir):
                print(
                    f">>> [CLEANUP] Removing stale scaled checkpoint: {existing_dir}"
                )
                shutil.rmtree(existing_dir, ignore_errors=True)

    def _scale_checkpoint(self, base_checkpoint: str, output_dir: str,
                          scale_factor: float) -> str:
        """
        Streams safetensors shards from base_checkpoint, multiplies linear weights by scale_factor,
        and saves the transformed checkpoint to output_dir.
        """
        # Ensure fresh output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print("\n=======================================================")
        print(
            f">>> [WEIGHT SCALER] Scaling sparse model by factor c = {scale_factor:.4f}"
        )
        print(f">>> Base Checkpoint:   {base_checkpoint}")
        print(f">>> Output Directory:  {output_dir}")
        print("=======================================================")

        # 1. Copy metadata, tokenizer configs, and architecture JSON files
        for fname in os.listdir(base_checkpoint):
            src = os.path.join(base_checkpoint, fname)
            dst = os.path.join(output_dir, fname)
            if not fname.endswith(".safetensors") and os.path.isfile(src):
                shutil.copy2(src, dst)

        # 2. Iterate through safetensors shards and apply scalar multiplier
        st_files = sorted([
            f for f in os.listdir(base_checkpoint)
            if f.endswith(".safetensors")
        ])
        for st_fname in st_files:
            src_st = os.path.join(base_checkpoint, st_fname)
            dst_st = os.path.join(output_dir, st_fname)
            scaled_tensors = {}

            with safe_open(src_st, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    # Scale only 2D linear projection weights in language model layers
                    if ("layers." in k and "weight" in k and tensor.ndim == 2
                            and not any(
                                skip in k for skip in
                                ["norm", "embed", "lm_head", "layer_scalar"])):
                        scaled_tensors[k] = tensor * scale_factor
                    else:
                        scaled_tensors[k] = tensor

            # Save the scaled shard to disk
            save_file(scaled_tensors, dst_st)
            print(
                f">>> [WEIGHT SCALER] Wrote scaled shard: {st_fname} ({len(scaled_tensors)} tensors)"
            )

        print(
            f">>> [WEIGHT SCALER] Successfully scaled checkpoint into {output_dir}\n"
        )
        return output_dir

    def resolve(self, current_config: Dict[str, Any],
                resolver_config: Dict[str, Any]) -> Tuple[str, Set[str]]:
        """
        Resolves the scaled checkpoint directory based on the scale_factor coordinate.
        Automatically cleans up previous temporary scaled checkpoints to preserve disk space.
        """
        scale_factor = float(current_config.get("scale_factor", 1.0))
        base_sparse_model = resolver_config.get("base_sparse_model", "")
        if not base_sparse_model:
            raise ValueError("Resolver config must specify base_sparse_model.")

        # If scale factor is 1.0, return base model directly without copying
        if abs(scale_factor - 1.0) < 1e-6:
            return base_sparse_model, self.CONSUMED_KEYS

        output_base_path = resolver_config.get(
            "output_base_path",
            "/mnt/pd/sparse_checkpoints/gemma4_2of4_scaled_c")
        scale_str = f"{scale_factor:.3f}".replace(".", "p")
        target_dir = f"{output_base_path}_{scale_str}"

        # Clean all previous scaled checkpoints matching pattern
        self._cleanup_old_scaled_dirs(output_base_path, target_dir)

        scaled_dir = self._scale_checkpoint(base_sparse_model, target_dir,
                                            scale_factor)
        return scaled_dir, self.CONSUMED_KEYS

import glob
import os
import shutil
from typing import Any, Dict, Optional, Set, Tuple

from core.env import SPARSE_CHECKPOINTS_DIR
from resolvers.base_resolver import BaseResolver
from resolvers.model_splicer import ModelSplicer


class BaseSparsityResolver(BaseResolver):
    """Unified, zero-duplication base resolver for all structured sparsity methods (SparseGPT, Magnitude, Wanda, Splicer).

    Handles:
      1. Layer range computation (first_to_last, last_to_first, middle_out).
      2. Disk checkpoint cache discovery.
      3. Fast zero-recompute cross-shard layer splicing when a reference sparse
      model is provided.
      4. Automatic cleanup of previous temporary spliced directories to preserve
      disk space.
    """

    CONSUMED_KEYS: Set[str] = {"num_sparse_layers"}
    METHOD_NAME: str = "sparsity"

    def __init__(self, splicer: Optional[ModelSplicer] = None):
        self.splicer = splicer or ModelSplicer()

    def _cleanup_old_spliced_dirs(self, base_sparse_dir: str,
                                  current_target_path: str) -> None:
        """Removes any older temporary spliced layer checkpoints to maintain ample disk headroom."""
        pattern = os.path.join(base_sparse_dir, "*_layers_*_to_*")
        for existing_dir in glob.glob(pattern):
            if existing_dir != current_target_path and os.path.isdir(
                    existing_dir):
                print(
                    f">>> [CLEANUP] Removing previous temporary spliced checkpoint: {existing_dir}"
                )
                shutil.rmtree(existing_dir, ignore_errors=True)

    def compute_layer_range(
            self, num_sparse_layers: int, strategy: str,
            total_layers: int) -> Tuple[Optional[int], Optional[int]]:
        if num_sparse_layers <= 0:
            return None, None
        if num_sparse_layers >= total_layers:
            return 0, total_layers

        if strategy == "first_to_last":
            return 0, num_sparse_layers
        elif strategy == "last_to_first":
            return total_layers - num_sparse_layers, total_layers
        elif strategy == "middle_out":
            mid = total_layers // 2
            half = num_sparse_layers // 2
            start = max(0, mid - half)
            end = min(total_layers, start + num_sparse_layers)
            return start, end
        else:
            raise ValueError(f"Unknown layer selection strategy: {strategy}")

    def generate_checkpoint_name(self, arch_type: str, prunen: int,
                                 prunem: int, strategy: str, min_layer: int,
                                 max_layer: int,
                                 resolver_config: Dict[str, Any]) -> str:
        calib_tag = f"_{resolver_config.get('nsamples')}c4" if "nsamples" in resolver_config else ""
        return f"{arch_type}_{self.METHOD_NAME}_{prunen}of{prunem}{calib_tag}_{strategy}_layers_{min_layer}_to_{max_layer}"

    def prune_in_memory(self, base_model: str, save_path: str, min_layer: int,
                        max_layer: int, prunen: int, prunem: int,
                        resolver_config: Dict[str, Any]) -> str:
        """Hook to be implemented by specific pruning algorithm subclasses."""
        raise NotImplementedError(
            "Subclasses must implement prune_in_memory fallback.")

    def resolve(self, current_config: Dict[str, Any],
                resolver_config: Dict[str, Any]) -> Tuple[str, Set[str]]:
        num_sparse_layers = int(current_config.get("num_sparse_layers", 0))
        base_model = resolver_config.get("base_model", "")
        if not base_model:
            raise ValueError("Resolver config must specify base_model.")

        # If 0 sparse layers requested, use unpruned base model directly
        if num_sparse_layers == 0:
            return base_model, self.CONSUMED_KEYS

        arch = self.splicer.detect_architecture(base_model)
        total_layers = arch.num_layers

        strategy = resolver_config.get("strategy", "first_to_last")
        prunen = resolver_config.get("prunen", 2)
        prunem = resolver_config.get("prunem", 4)
        base_sparse_dir = resolver_config.get("checkpoint_base_dir",
                                              SPARSE_CHECKPOINTS_DIR)
        reference_sparse_model = resolver_config.get("reference_sparse_model",
                                                     None)

        # If 100% sparse layers requested and full reference model exists, use reference directly
        if num_sparse_layers >= total_layers and reference_sparse_model and os.path.exists(
                reference_sparse_model):
            return reference_sparse_model, self.CONSUMED_KEYS

        min_layer, max_layer = self.compute_layer_range(
            num_sparse_layers=num_sparse_layers,
            strategy=strategy,
            total_layers=total_layers)

        ckpt_name = self.generate_checkpoint_name(
            arch_type=arch.model_type,
            prunen=prunen,
            prunem=prunem,
            strategy=strategy,
            min_layer=min_layer,
            max_layer=max_layer,
            resolver_config=resolver_config)
        ckpt_path = os.path.join(base_sparse_dir, ckpt_name)

        # 1. Cache hit on disk
        has_weights = os.path.exists(os.path.join(ckpt_path, "model.safetensors.index.json")) or \
                      (os.path.exists(ckpt_path) and any(f.endswith(".safetensors") for f in os.listdir(ckpt_path)))

        if has_weights:
            print(
                f">>> [RESOLVER] Found cached checkpoint ({num_sparse_layers} layers, {strategy}): {ckpt_path}"
            )
            return ckpt_path, self.CONSUMED_KEYS

        # Clean older temporary spliced checkpoints before creating new one
        self._cleanup_old_spliced_dirs(base_sparse_dir, ckpt_path)

        # 2. Fast cross-shard layer splicing from 100% sparse reference model
        if reference_sparse_model and os.path.exists(reference_sparse_model):
            print(
                f">>> [RESOLVER] Splicing {num_sparse_layers} layers from reference {reference_sparse_model}..."
            )
            sparse_indices = set(range(min_layer, max_layer))
            self.splicer.splice_checkpoint(
                dense_model_dir=base_model,
                fully_sparse_model_dir=reference_sparse_model,
                sparse_layer_indices=sparse_indices,
                output_dir=ckpt_path)
            return ckpt_path, self.CONSUMED_KEYS

        # 3. In-memory algorithmic pruning fallback
        print("\n=======================================================")
        print(
            f">>> [RESOLVER] Pruning in-process via {self.METHOD_NAME.upper()}: {ckpt_name} (Layers {min_layer} to {max_layer - 1})..."
        )
        print("=======================================================")

        self.prune_in_memory(base_model=base_model,
                             save_path=ckpt_path,
                             min_layer=min_layer,
                             max_layer=max_layer,
                             prunen=prunen,
                             prunem=prunem,
                             resolver_config=resolver_config)
        return ckpt_path, self.CONSUMED_KEYS

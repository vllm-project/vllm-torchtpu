import glob
import json
import os
import re
import shutil
from typing import Any, Dict, Optional, Set, Tuple

from core.env import SPARSE_CHECKPOINTS_DIR
from core.model_utils import resolve_model_path
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

    CONSUMED_KEYS: Set[str] = {
        "num_sparse_layers", "layers", "prunen", "prunem", "use_lrc", "is_lrc",
        "lrc_rank", "lrc_iters", "lrc_k", "lrc_T"
    }
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
                    f">>> [RESOLVER] Removing temporary spliced checkpoint to save disk space: {existing_dir}"
                )
                shutil.rmtree(existing_dir, ignore_errors=True)

    @staticmethod
    def assert_checkpoint_integrity(base_model_dir: str,
                                    checkpoint_dir: str) -> None:
        """Forwarding gate to ModelSplicer.assert_checkpoint_integrity."""
        ModelSplicer.assert_checkpoint_integrity(base_model_dir, checkpoint_dir)

    @staticmethod
    def _assert_spliceable(sparse_indices: Set[int], strategy: str,
                           resolver_config: Dict[str, Any]) -> None:
        """Refuse to splice a layer set that SparseGPT calibration cannot support."""
        if not sparse_indices:
            return
        if sparse_indices == set(range(len(sparse_indices))):
            return

        if resolver_config.get("allow_nonprefix_splice"):
            print(
                ">>> [RESOLVER] Proceeding because allow_nonprefix_splice is set."
            )
            return

        ordered = sorted(sparse_indices)
        raise ValueError(
            f"Refusing to splice non-prefix layer set {ordered} with strategy {strategy}. "
            "SparseGPT calibration requires sequential prefix layers. "
            "Non-prefix layer sets must be sequentially re-pruned rather than spliced."
        )

    @staticmethod
    def _assert_reference_ratio(reference_sparse_model: str, prunen: int,
                                prunem: int) -> None:
        """Asserts that reference_sparse_model matches the requested prunen/prunem ratio."""
        ref_basename = os.path.basename(os.path.normpath(reference_sparse_model))
        match = re.search(r"(\d+)(?:of|:)(\d+)", ref_basename)
        if match:
            ref_n, ref_m = int(match.group(1)), int(match.group(2))
            if (ref_n, ref_m) != (prunen, prunem):
                raise ValueError(
                    f"Reference sparse model ratio mismatch: reference at '{reference_sparse_model}' "
                    f"has ratio {ref_n}:{ref_m} ({ref_n}of{ref_m}), but sweep requested {prunen}:{prunem} ({prunen}of{prunem})."
                )

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
        name = f"{arch_type}_{self.METHOD_NAME}_{prunen}of{prunem}{calib_tag}_{strategy}_layers_{min_layer}_to_{max_layer}"
        if resolver_config.get("use_lrc") or resolver_config.get("is_lrc"):
            rank = resolver_config.get("lrc_rank", resolver_config.get("lrc_k", 128))
            iters = resolver_config.get("lrc_iters", resolver_config.get("lrc_T", 3))
            name += f"_lrc_k{rank}_T{iters}"
        return name

    def prune_in_memory(self, base_model: str, save_path: str, min_layer: int,
                        max_layer: int, prunen: int, prunem: int,
                        resolver_config: Dict[str, Any],
                        sparse_layer_indices: Optional[Set[int]] = None) -> str:
        """Hook to be implemented by specific pruning algorithm subclasses."""
        raise NotImplementedError(
            "Subclasses must implement prune_in_memory fallback.")

    def resolve(self, current_config: Dict[str, Any],
                resolver_config: Dict[str, Any]) -> Tuple[str, Set[str]]:
        raw_layers = current_config.get("layers", None)
        if raw_layers is not None:
            if isinstance(raw_layers, (list, set, tuple)):
                explicit_layer_indices: Optional[Set[int]] = set(int(x) for x in raw_layers)
                num_sparse_layers = len(explicit_layer_indices)
            else:
                raise ValueError(f"Expected list/set of layer indices in 'layers', got {type(raw_layers)}")
        else:
            explicit_layer_indices = None
            num_sparse_layers = int(current_config.get("num_sparse_layers", 0))
        base_sparse_dir = resolver_config.get("checkpoint_base_dir",
                                              SPARSE_CHECKPOINTS_DIR)
        base_model = resolve_model_path(resolver_config.get("base_model", ""))
        reference_sparse_model = resolve_model_path(resolver_config.get(
            "reference_sparse_model", ""),
                                                    base_dir=base_sparse_dir)

        if not base_model:
            raise ValueError("Resolver config must specify base_model.")

        # If 0 sparse layers requested, use unpruned base model directly
        if num_sparse_layers == 0:
            return base_model, self.CONSUMED_KEYS

        arch = self.splicer.detect_architecture(base_model)
        total_layers = arch.num_layers

        if explicit_layer_indices is not None:
            for layer_idx in sorted(explicit_layer_indices):
                if not (0 <= layer_idx < total_layers):
                    raise ValueError(
                        f"Layer index {layer_idx} out of bounds for model with "
                        f"{total_layers} layers (must satisfy 0 <= layer_idx < {total_layers})."
                    )

        strategy = resolver_config.get("strategy", "first_to_last")
        prunen = int(current_config.get("prunen", resolver_config.get("prunen", 2)))
        prunem = int(current_config.get("prunem", resolver_config.get("prunem", 4)))
        is_lrc = bool(resolver_config.get("use_lrc") or resolver_config.get("is_lrc") or current_config.get("use_lrc") or current_config.get("is_lrc"))
        is_ref_lrc = bool(reference_sparse_model and ("lrc" in os.path.basename(reference_sparse_model).lower() or resolver_config.get("reference_is_lrc", False)))

        # Check if the full reference sparse model exists on disk.
        # If absent, invoke SparseGPT in memory once across all layers to generate and cache it,
        # ensuring downstream partial-layer splices have a complete sparse source.
        if reference_sparse_model and (not is_lrc or is_ref_lrc) and not os.path.exists(
                reference_sparse_model):
            print(
                f">>> [RESOLVER] Reference sparse model not found at {reference_sparse_model}."
                f" Generating 100% sparse reference model ({total_layers} layers)..."
            )
            self.prune_in_memory(base_model=base_model,
                                 save_path=reference_sparse_model,
                                 min_layer=0,
                                 max_layer=total_layers,
                                 prunen=prunen,
                                 prunem=prunem,
                                 resolver_config=resolver_config)

        # When 100% sparse layers are requested, verify that the reference model is fully compatible
        # with the base model architecture and has all non-transformer multimodal weights
        # (e.g. model.vision_embedder.*). If compatible, reuse it directly; otherwise fall through
        # to ModelSplicer to construct a complete checkpoint using the dense base model as template.
        if num_sparse_layers >= total_layers and reference_sparse_model and (not is_lrc or is_ref_lrc) and os.path.exists(
                reference_sparse_model):
            BaseSparsityResolver._assert_reference_ratio(reference_sparse_model, prunen, prunem)
            dense_map = self.splicer.get_weight_map(base_model)
            sparse_map = self.splicer.get_weight_map(reference_sparse_model)
            # Ensure every non-layer weight (vision embeddings, embed_tokens, norm) is present in sparse model
            has_all_non_layers = all(
                k in sparse_map for k in dense_map
                if self.splicer.extract_layer_idx(k) is None)
            dense_cfg_path = os.path.join(base_model, "config.json")
            sparse_cfg_path = os.path.join(reference_sparse_model,
                                           "config.json")
            matching_arch = False
            if os.path.exists(sparse_cfg_path) and os.path.exists(
                    dense_cfg_path):
                with open(dense_cfg_path) as f_d, open(sparse_cfg_path) as f_s:
                    matching_arch = json.load(f_d).get(
                        "architectures") == json.load(f_s).get("architectures")

            if has_all_non_layers and matching_arch:
                return reference_sparse_model, self.CONSUMED_KEYS
            # If missing multimodal weights or architectures differ, fall through to splice all layers
            print(
                f">>> [RESOLVER] Reference sparse model at {reference_sparse_model} is missing "
                f"base model non-layer weights or matching config. Splicing all {total_layers} layers..."
            )

        if explicit_layer_indices is not None:
            sparse_indices = explicit_layer_indices
            min_layer = min(sparse_indices) if sparse_indices else 0
            max_layer = (max(sparse_indices) + 1) if sparse_indices else 0
        else:
            min_layer, max_layer = self.compute_layer_range(
                num_sparse_layers=num_sparse_layers,
                strategy=strategy,
                total_layers=total_layers)
            sparse_indices = set(range(min_layer, max_layer))
        can_splice = bool(reference_sparse_model) and (
            not is_lrc or is_ref_lrc) and os.path.exists(reference_sparse_model)
        is_prefix = bool(sparse_indices) and sorted(sparse_indices) == list(range(len(sparse_indices)))
        splice_from_reference = can_splice and (
            is_prefix or resolver_config.get("allow_nonprefix_splice", False))

        # Validate the layer set BEFORE the cache lookup and before
        # _cleanup_old_spliced_dirs so an invalid config can never delete
        # valid cached splices on its way to raising. Explicit non-prefix
        # layer sets bypass the reference splice (in-memory calibration)
        # and are therefore not subject to the splice guard.
        if can_splice and (splice_from_reference
                           or explicit_layer_indices is None):
            self._assert_spliceable(sparse_indices, strategy, resolver_config)

        if explicit_layer_indices is not None and not is_prefix:
            import hashlib
            calib_tag = f"_{resolver_config.get('nsamples')}c4" if "nsamples" in resolver_config else ""
            cache_payload = f"{prunen}of{prunem}:" + ",".join(str(x) for x in sorted(sparse_indices))
            if is_lrc:
                rank = resolver_config.get("lrc_rank", resolver_config.get("lrc_k", current_config.get("lrc_rank", current_config.get("lrc_k", 128))))
                iters = resolver_config.get("lrc_iters", resolver_config.get("lrc_T", current_config.get("lrc_iters", current_config.get("lrc_T", 3))))
                cache_payload += f"_lrc_k{rank}_T{iters}"
            layer_sig = hashlib.sha1(cache_payload.encode()).hexdigest()[:8]
            ckpt_name = f"{arch.model_type}_{self.METHOD_NAME}_{prunen}of{prunem}{calib_tag}_layers_{num_sparse_layers}l_{layer_sig}"
            if is_lrc:
                ckpt_name += f"_lrc_k{rank}_T{iters}"
        else:
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
        has_weights = os.path.exists(
            os.path.join(ckpt_path, "model.safetensors.index.json")) or (
                os.path.exists(ckpt_path) and any(
                    f.endswith(".safetensors") for f in os.listdir(ckpt_path)))

        if has_weights:
            try:
                self.assert_checkpoint_integrity(base_model, ckpt_path)
                print(
                    f">>> [RESOLVER] Found cached checkpoint ({num_sparse_layers} layers, {strategy}): {ckpt_path}"
                )
                return ckpt_path, self.CONSUMED_KEYS
            except ValueError as e:
                print(
                    f">>> [RESOLVER] Cached checkpoint at {ckpt_path} failed integrity check: {e}. "
                    f"Regenerating checkpoint..."
                )

        # Clean older temporary spliced checkpoints before creating new one
        self._cleanup_old_spliced_dirs(base_sparse_dir, ckpt_path)

        # 2. Fast cross-shard layer splicing from 100% sparse reference model (prefix only)
        if can_splice:
            BaseSparsityResolver._assert_reference_ratio(reference_sparse_model, prunen, prunem)
            if splice_from_reference:
                print(
                    f">>> [RESOLVER] Splicing {num_sparse_layers} layers from reference {reference_sparse_model}..."
                )
                self.splicer.splice_checkpoint(
                    dense_model_dir=base_model,
                    fully_sparse_model_dir=reference_sparse_model,
                    sparse_layer_indices=sparse_indices,
                    output_dir=ckpt_path)
                self.assert_checkpoint_integrity(base_model, ckpt_path)
                return ckpt_path, self.CONSUMED_KEYS
            elif explicit_layer_indices is not None:
                print(
                    f">>> [RESOLVER] Non-prefix layer set {sorted(sparse_indices)} requested. "
                    f"Bypassing reference splice and falling through to in-memory calibration..."
                )
        elif is_lrc and reference_sparse_model and not is_ref_lrc:
            print(
                f">>> [RESOLVER] Reference sparse model '{reference_sparse_model}' is not tagged as an LRC checkpoint. "
                "Bypassing reference model reuse and splicing, falling through directly to prune_in_memory."
            )


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
                             resolver_config=resolver_config,
                             sparse_layer_indices=sparse_indices)
        self.assert_checkpoint_integrity(base_model, ckpt_path)
        return ckpt_path, self.CONSUMED_KEYS

from typing import Any, Dict, Optional, Set, Tuple

from resolvers.base_sparsity_resolver import BaseSparsityResolver
from sparsity.naive_magnitude_pruner import prune_model_magnitude


class NaiveMagnitudeResolver(BaseSparsityResolver):
    """
    Dedicated resolver for Naive N:M Magnitude Pruning (data-free top-k selection).
    Inherits caching and cross-shard zero-recompute splicing from BaseSparsityResolver.
    """
    METHOD_NAME: str = "magnitude"

    def prune_in_memory(
            self,
            base_model: str,
            save_path: str,
            min_layer: int,
            max_layer: int,
            prunen: int,
            prunem: int,
            resolver_config: Dict[str, Any],
            sparse_layer_indices: Optional[Set[int]] = None) -> str:
        prune_model_magnitude(base_model_dir=base_model,
                              output_dir=save_path,
                              prunen=prunen,
                              prunem=prunem,
                              min_layer=min_layer,
                              max_layer=max_layer)
        return save_path


def resolve_model(current_config: Dict[str, Any],
                  resolver_config: Dict[str, Any]) -> Tuple[str, Set[str]]:
    resolver = NaiveMagnitudeResolver()
    return resolver.resolve(current_config=current_config,
                            resolver_config=resolver_config)

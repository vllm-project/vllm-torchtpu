from typing import Any, Dict, Set, Tuple

from resolvers.base_sparsity_resolver import BaseSparsityResolver
from sparsity import prune_gemma4


class SparseGptResolver(BaseSparsityResolver):
    """
    Dedicated resolver for SparseGPT Hessian-based N:M structured sparsity.
    Inherits caching and cross-shard zero-recompute splicing from BaseSparsityResolver.
    """
    METHOD_NAME: str = "sparsegpt"

    def prune_in_memory(self, base_model: str, save_path: str, min_layer: int,
                        max_layer: int, prunen: int, prunem: int,
                        resolver_config: Dict[str, Any]) -> str:
        prune_gemma4(model_path=base_model,
                     dataset=resolver_config.get("dataset", "c4"),
                     minlayer=min_layer,
                     maxlayer=max_layer,
                     prunen=prunen,
                     prunem=prunem,
                     nsamples=resolver_config.get("nsamples", 64),
                     save_path=save_path)
        return save_path


def resolve_model(current_config: Dict[str, Any],
                  resolver_config: Dict[str, Any]) -> Tuple[str, Set[str]]:
    resolver = SparseGptResolver()
    return resolver.resolve(current_config=current_config,
                            resolver_config=resolver_config)

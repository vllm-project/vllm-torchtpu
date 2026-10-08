from typing import Any, Dict, Optional, Set, Tuple

from resolvers.base_sparsity_resolver import BaseSparsityResolver
from sparsity import make_lrc_pruner, prune_gemma4


class LrcResolver(BaseSparsityResolver):
    """Dedicated resolver for Low-Rank Compensated (LRC) SparseGPT sparsity.

    Models each linear weight as W ≈ S + U V (with S N:M-sparse and U V rank-k)
    via alternating minimization in whitened activation space (sparsity/lrc.py).
    """
    METHOD_NAME: str = "lrc"
    CONSUMED_KEYS: Set[str] = BaseSparsityResolver.CONSUMED_KEYS | {
        "lrc_rank", "lrc_iters"
    }

    def _method_cache_tag(self, resolver_config: Dict[str, Any]) -> str:
        rank = int(resolver_config.get("lrc_rank", 128))
        iters = int(resolver_config.get("lrc_iters", 3))
        return f"_k{rank}_T{iters}"

    def _is_reference_compatible(self, reference_sparse_model: str,
                                 resolver_config: Dict[str, Any]) -> bool:
        import os
        return bool(reference_sparse_model and
                    ("lrc" in os.path.basename(reference_sparse_model).lower()
                     or resolver_config.get("reference_is_lrc", False)))

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
        percdamp = resolver_config.get("percdamp", 0.01)
        blocksize = resolver_config.get("blocksize", 128)
        prune_fn = make_lrc_pruner(
            prunen=prunen,
            prunem=prunem,
            percdamp=percdamp,
            blocksize=blocksize,
            T=int(resolver_config.get("lrc_iters", 3)),
            k=int(resolver_config.get("lrc_rank", 128)),
        )
        prune_gemma4(
            model_path=base_model,
            dataset=resolver_config.get("dataset", "c4"),
            minlayer=min_layer,
            maxlayer=max_layer,
            prunen=prunen,
            prunem=prunem,
            nsamples=resolver_config.get("nsamples", 64),
            seed=resolver_config.get("seed", 0),
            percdamp=percdamp,
            blocksize=blocksize,
            save_path=save_path,
            sparse_layer_indices=sparse_layer_indices,
            prune_fn=prune_fn,
        )
        return save_path


def resolve_model(current_config: Dict[str, Any],
                  resolver_config: Dict[str, Any]) -> Tuple[str, Set[str]]:
    resolver = LrcResolver()
    return resolver.resolve(current_config=current_config,
                            resolver_config=resolver_config)

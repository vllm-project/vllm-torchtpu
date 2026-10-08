from typing import Dict, Type

from core.interfaces import IModelResolver
from resolvers.lrc_resolver import LrcResolver
from resolvers.naive_magnitude_resolver import NaiveMagnitudeResolver
from resolvers.sparsegpt_resolver import SparseGptResolver
from resolvers.weight_scaler_resolver import WeightScalerResolver


class ResolverRegistry:
    """Factory and registry for dynamic model resolvers."""
    _registry: Dict[str, Type[IModelResolver]] = {
        # SparseGPT Hessian-based structured sparsity
        "sparsegpt": SparseGptResolver,
        "resolvers.sparsegpt_resolver": SparseGptResolver,
        "resolvers.gemma_sparsity_resolver": SparseGptResolver,

        # Low-Rank Compensated (LRC) SparseGPT sparsity
        "lrc": LrcResolver,
        "resolvers.lrc_resolver": LrcResolver,

        # Naive Magnitude structured sparsity (data-free)
        "magnitude": NaiveMagnitudeResolver,
        "naive_magnitude": NaiveMagnitudeResolver,
        "resolvers.naive_magnitude_resolver": NaiveMagnitudeResolver,
        "resolvers.magnitude_resolver": NaiveMagnitudeResolver,

        # Constant Weight Scaler
        "weight_scaler": WeightScalerResolver,
        "resolvers.weight_scaler_resolver": WeightScalerResolver,
    }

    @classmethod
    def register(cls, name: str):

        def decorator(subclass: Type[IModelResolver]):
            cls._registry[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str) -> IModelResolver:
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise KeyError(
                f"Resolver {name} is not registered. Available resolvers: {list(cls._registry.keys())}"
            )
        return cls._registry[name_lower]()

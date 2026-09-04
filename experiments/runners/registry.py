"""
registry.py
-----------
Central factory and registry for benchmark execution runners (profile, eval).
"""

from typing import Dict, Type

from core.interfaces import IRunner
from runners.eval_runner import TpuEvalRunner
from runners.profile_runner import TpuProfileRunner


class RunnerRegistry:
    """
    Factory and registry mapping runner mode names to concrete IRunner classes.
    """
    _registry: Dict[str, Type[IRunner]] = {
        "profile": TpuProfileRunner,
        "eval": TpuEvalRunner,
    }

    @classmethod
    def register(cls, name: str):
        """Decorator to register a custom IRunner implementation."""

        def decorator(subclass: Type[IRunner]):
            cls._registry[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str) -> IRunner:
        """Retrieves and instantiates the registered runner for the given mode name."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise KeyError(
                f"Runner {name} is not registered. Available runners: {list(cls._registry.keys())}"
            )
        return cls._registry[name_lower]()

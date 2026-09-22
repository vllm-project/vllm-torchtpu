"""
base_resolver.py
----------------
Abstract base class defining the contract for dynamic model checkpoint resolvers.
Resolvers dynamically inspect sweep coordinates and yield on-disk model directories.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Set, Tuple

from core.interfaces import IModelResolver


class BaseResolver(IModelResolver, ABC):
    """
    Abstract base class for all dynamic model checkpoint resolvers.
    Defines CONSUMED_KEYS to signal which sweep matrix keys are handled by the resolver.
    """
    CONSUMED_KEYS: Set[str] = set()

    @abstractmethod
    def resolve(self, current_config: Dict[str, Any],
                resolver_config: Dict[str, Any]) -> Tuple[str, Set[str]]:
        """
        Resolves or generates a model checkpoint for the current sweep coordinate.
        Returns:
            Tuple[str, Set[str]]: (checkpoint_directory_path, set_of_consumed_config_keys)
        """
        pass

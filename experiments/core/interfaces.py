"""
interfaces.py
-------------
Abstract base classes and contracts for the modular TPU inference and evaluation framework.
Defines core interfaces for execution runners, dynamic model resolvers, and metric recorders.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Set, Tuple

from core.models import BenchmarkResult, ExecutionContext


class IRunner(ABC):
    """Abstract base interface for benchmark execution runners (profiling, evaluation)."""

    @abstractmethod
    def execute(self, context: ExecutionContext) -> BenchmarkResult:
        """Executes a single benchmark trial for the provided execution context."""
        pass


class IModelResolver(ABC):
    """Abstract base interface for dynamic model checkpoint resolution and on-the-fly transformations."""

    @abstractmethod
    def resolve(self, current_config: Dict[str, Any],
                resolver_config: Dict[str, Any]) -> Tuple[str, Set[str]]:
        """
        Resolves or generates a model checkpoint based on sweep grid parameters.
        Returns:
            Tuple[str, Set[str]]: (checkpoint_directory_path, set_of_consumed_config_keys)
        """
        pass


class IResultRecorder(ABC):
    """Abstract base interface for logging benchmark results and metric persistence."""

    @abstractmethod
    def record(self, metadata: Dict[str, Any], metrics: Dict[str, Any],
               reproduction_cmd: str) -> None:
        """Records a completed benchmark coordinate with its metrics and reproduction command."""
        pass

    def finalize(self, status: str = "COMPLETED") -> None:
        """Marks the end of a sweep (e.g. COMPLETED or FAILED).

        Optional hook; recorders without a notion of sweep-level state
        (such as flat CSV logs) keep the default no-op.
        """
        return None


class ISparsityEngine(ABC):
    """Abstract interface for weight pruning and structured sparsity algorithms."""

    @abstractmethod
    def sparsify(self, model_path: str, output_path: str, **kwargs) -> str:
        """Applies structured sparsity to a model checkpoint and saves it to output_path."""
        pass

"""
base_runner.py
--------------
Abstract base runner for vLLM benchmark execution on Cloud TPU devices.
Initializes the runtime environment and provides shared setup routines.
"""

from abc import ABC, abstractmethod

from core.engine_builder import VllmEngineBuilder
from core.interfaces import IRunner
from core.models import BenchmarkResult, ExecutionContext


class BaseVllmRunner(IRunner, ABC):
    """
    Base runner providing shared TPU runtime initialization and vLLM environment configuration.
    All profiling and evaluation runners inherit from this class.
    """

    def __init__(self):
        """Initializes the runner and establishes required TPU environment variables."""
        VllmEngineBuilder.setup_tpu_environment()

    @abstractmethod
    def execute(self, context: ExecutionContext) -> BenchmarkResult:
        """Executes the benchmark run defined by the given ExecutionContext."""
        pass

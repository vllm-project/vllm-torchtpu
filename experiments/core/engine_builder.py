"""engine_builder.py.

Helper class responsible for constructing vLLM engine kwargs and configuring
TPU-specific runtime settings and profiler configurations.
"""

import os
from typing import Any, Dict

from core.env import HF_HOME, setup_tpu_env
from core.models import ExecutionContext
from vllm.config.profiler import ProfilerConfig


class VllmEngineBuilder:
    """Constructs vLLM runtime engine configuration dictionaries and sets up TPU platform settings."""

    @staticmethod
    def setup_tpu_environment() -> None:
        """Sets required environment variables for vLLM TPU execution."""
        setup_tpu_env()

    @staticmethod
    def build_engine_kwargs(context: ExecutionContext) -> Dict[str, Any]:
        """Assembles normalized vLLM engine arguments from the execution context.

        Sets memory utilization, parallel dimensions, precision, and profiler config.
        """
        engine_kwargs = {
            "tensor_parallel_size": context.tensor_parallel_size,
            "dtype": context.dtype,
            "disable_chunked_mm_input": True,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.85,
            # vLLM treats download_dir as a hub cache dir; point it at the
            # same HF_HOME/hub that transformers uses, or every base model
            # is downloaded twice (HF_HOME/models--* and HF_HOME/hub/models--*).
            "download_dir": os.path.join(HF_HOME, "hub"),
        }
        # Apply max model context length if specified
        if context.max_model_len:
            engine_kwargs["max_model_len"] = context.max_model_len

        # Configure distributed TPU worker profiling if trace is requested
        if context.trace:
            trace_dir = os.path.join(context.output_dir, "traces")
            os.makedirs(trace_dir, exist_ok=True)
            engine_kwargs["profiler_config"] = ProfilerConfig(
                profiler="torch", torch_profiler_dir=trace_dir)

        # Merge custom engine args from YAML sweep configuration
        if context.engine_args:
            engine_kwargs.update(context.engine_args)

        return engine_kwargs

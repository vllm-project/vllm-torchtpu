"""
models.py
---------
Canonical dataclasses and data models used across the TPU experiment pipeline.
Encapsulates runtime parameters, benchmark outputs, and model architecture metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExecutionContext:
    """
    Normalized execution context containing all configuration parameters for a benchmark run.
    Passed to IRunner implementations to execute profiling or quality evaluation.
    """
    # Base model and hardware topology
    model_path: str
    tensor_parallel_size: int = 4
    dtype: str = "bfloat16"
    max_model_len: Optional[int] = None

    # vLLM runtime engine configuration
    engine_args: Dict[str, Any] = field(default_factory=dict)
    sweep_metadata: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""
    csv_file: Optional[str] = None

    # Throughput & latency profiling parameters
    batch_size: Any = 1
    input_len: int = 128
    output_len: int = 64
    trace: bool = False
    jax_advanced_configuration: Dict[str, Any] = field(default_factory=dict)

    # Accuracy / quality evaluation parameters
    tasks: str = "mmlu_pro"
    limit: Optional[float] = None
    seed: int = 42


@dataclass
class BenchmarkResult:
    """
    Normalized result container returned by any IRunner implementation.
    Standardizes duration, parsed metrics, and log paths.
    """
    duration_s: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    log_path: Optional[str] = None
    raw_output: Optional[Any] = None


@dataclass
class ModelArchitectureSpec:
    """
    Discovered architectural specifications for an LLM (layer counts, hidden sizes, prefixes).
    Used by resolvers and splicers to route tensors accurately across layers.
    """
    model_type: str
    num_layers: int
    hidden_size: int
    layer_prefix: str
    config_dict: Dict[str, Any] = field(default_factory=dict)

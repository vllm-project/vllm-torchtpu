"""test_core.py.

Unit tests for foundational core models, interfaces, result recording, and environment resolution:
- ExecutionContext: Parameter container and override propagation
- BenchmarkResult: Metrics container and JSON serialization
- ModelArchitectureSpec: Metadata extraction
- CsvResultRecorder: Dynamic CSV schema evolution and disk persistence
- VllmEngineBuilder: Engine configuration kwargs assembly and profiler setup
- TpuEnvironmentResolution: Centralized cache directories, pythonpath, and TPU execution flags
"""

import csv
import os
import tempfile
import unittest

from core.csv_recorder import CsvResultRecorder
from core.engine_builder import VllmEngineBuilder
from core.env import (HF_HOME, ROOT_DIR, SPARSE_CHECKPOINTS_DIR,
                      TORCHINDUCTOR_CACHE_DIR, VLLM_REPO_PATH,
                      VLLM_TORCHTPU_SRC, setup_tpu_env)
from core.models import (BenchmarkResult, ExecutionContext,
                         ModelArchitectureSpec)


class TestCoreModels(unittest.TestCase):
    """Test suite validating core dataclass models and execution contexts."""

    def test_execution_context_defaults(self):
        """Arrange-Act-Assert: Verify ExecutionContext defaults and immutability."""
        context = ExecutionContext(model_path="/path/to/my_model")

        self.assertEqual(context.model_path, "/path/to/my_model")
        self.assertEqual(context.tensor_parallel_size, 4)
        self.assertEqual(context.dtype, "bfloat16")
        self.assertIsNone(context.max_model_len)
        self.assertEqual(context.batch_size, 1)
        self.assertEqual(context.input_len, 128)
        self.assertEqual(context.output_len, 64)
        self.assertFalse(context.trace)
        self.assertEqual(context.tasks, "mmlu_pro")
        self.assertIsNone(context.limit)
        self.assertEqual(context.seed, 42)
        self.assertEqual(context.engine_args, {})

    def test_execution_context_custom_overrides(self):
        """Arrange-Act-Assert: Verify ExecutionContext custom parameter overrides."""
        context = ExecutionContext(
            model_path="/path/to/custom_model",
            tensor_parallel_size=8,
            dtype="float16",
            max_model_len=4096,
            batch_size=16,
            input_len=512,
            output_len=128,
            trace=True,
            tasks="gsm8k",
            limit=0.1,
            engine_args={"gpu_memory_utilization": 0.90},
        )

        self.assertEqual(context.model_path, "/path/to/custom_model")
        self.assertEqual(context.tensor_parallel_size, 8)
        self.assertEqual(context.dtype, "float16")
        self.assertEqual(context.max_model_len, 4096)
        self.assertEqual(context.batch_size, 16)
        self.assertEqual(context.input_len, 512)
        self.assertEqual(context.output_len, 128)
        self.assertTrue(context.trace)
        self.assertEqual(context.tasks, "gsm8k")
        self.assertEqual(context.limit, 0.1)
        self.assertEqual(context.engine_args["gpu_memory_utilization"], 0.90)

    def test_benchmark_result_container(self):
        """Arrange-Act-Assert: Verify BenchmarkResult metrics containment and duration."""
        metrics = {
            "throughput_tok_s": 2450.5,
            "latency_ms": 14.2,
            "mmlu_pro_acc": 0.785,
        }
        res = BenchmarkResult(
            duration_s=25.4,
            metrics=metrics,
            log_path="/tmp/benchmark_results/run_01.json",
        )

        self.assertEqual(res.duration_s, 25.4)
        self.assertEqual(res.metrics["throughput_tok_s"], 2450.5)
        self.assertEqual(res.metrics["latency_ms"], 14.2)
        self.assertEqual(res.metrics["mmlu_pro_acc"], 0.785)
        self.assertEqual(res.log_path, "/tmp/benchmark_results/run_01.json")

    def test_model_architecture_spec(self):
        """Arrange-Act-Assert: Verify ModelArchitectureSpec attributes."""
        spec = ModelArchitectureSpec(
            model_type="Gemma4ForCausalLM",
            num_layers=60,
            hidden_size=5376,
            layer_prefix="model.language_model.layers",
        )

        self.assertEqual(spec.model_type, "Gemma4ForCausalLM")
        self.assertEqual(spec.num_layers, 60)
        self.assertEqual(spec.hidden_size, 5376)
        self.assertEqual(spec.layer_prefix, "model.language_model.layers")


class TestCsvResultRecorder(unittest.TestCase):
    """Test suite validating dynamic CSV schema evolution and result persistence."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.test_dir.name, "results.csv")
        self.recorder = CsvResultRecorder(self.csv_path)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_single_row_recording(self):
        """Arrange-Act-Assert: Record a single benchmark coordinate into CSV."""
        metadata = {
            "model": "gemma4-31b",
            "batch_size": 4,
            "sparse_layers": 10,
        }
        metrics = {"throughput_tok_s": "1250.0", "duration_s": "15.20"}

        self.recorder.record(metadata=metadata, metrics=metrics)

        self.assertTrue(os.path.exists(self.csv_path))
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 1)
            row = reader[0]
            self.assertEqual(row["model"], "gemma4-31b")
            self.assertEqual(row["batch_size"], "4")
            self.assertEqual(row["sparse_layers"], "10")
            self.assertEqual(row["throughput_tok_s"], "1250.0")
            self.assertEqual(row["duration_s"], "15.20")

    def test_dynamic_schema_evolution(self):
        """Arrange-Act-Assert: Test dynamic CSV header expansion."""
        row1_meta = {"model": "model_A", "sparse_layers": 0}
        row1_metrics = {"throughput_tok_s": "1000.0"}
        self.recorder.record(metadata=row1_meta, metrics=row1_metrics)

        row2_meta = {"model": "model_A", "sparse_layers": 10}
        row2_metrics = {
            "throughput_tok_s": "1400.0",
            "accuracy": "0.852",
            "gpu_mem_gb": "18.4",
        }
        self.recorder.record(metadata=row2_meta, metrics=row2_metrics)

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 2)
            self.assertEqual(reader[0]["model"], "model_A")
            self.assertEqual(reader[0]["sparse_layers"], "0")
            self.assertEqual(reader[0]["throughput_tok_s"], "1000.0")
            self.assertEqual(reader[0]["accuracy"], "")
            self.assertEqual(reader[0]["gpu_mem_gb"], "")

            self.assertEqual(reader[1]["model"], "model_A")
            self.assertEqual(reader[1]["sparse_layers"], "10")
            self.assertEqual(reader[1]["throughput_tok_s"], "1400.0")
            self.assertEqual(reader[1]["accuracy"], "0.852")
            self.assertEqual(reader[1]["gpu_mem_gb"], "18.4")


class TestVllmEngineBuilder(unittest.TestCase):
    """Test suite validating vLLM engine argument assembly and TPU runtime environment configuration."""

    def test_setup_tpu_environment(self):
        """Arrange-Act-Assert: Verify TPU platform environment variables."""
        VllmEngineBuilder.setup_tpu_environment()

        self.assertEqual(os.environ.get("VLLM_TARGET_DEVICE"), "tpu")
        self.assertEqual(os.environ.get("VLLM_PLATFORM"), "tpu")
        self.assertEqual(os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"), "0")

    def test_build_engine_kwargs_standard(self):
        """Arrange-Act-Assert: Verify standard vLLM engine initialization dictionary."""
        context = ExecutionContext(
            model_path="/path/to/my_model",
            tensor_parallel_size=4,
            dtype="bfloat16",
            engine_args={
                "gpu_memory_utilization": 0.88,
                "kv_cache_dtype": "fp8",
            },
        )

        kwargs = VllmEngineBuilder.build_engine_kwargs(context)

        self.assertEqual(kwargs["tensor_parallel_size"], 4)
        self.assertEqual(kwargs["dtype"], "bfloat16")
        self.assertEqual(kwargs["gpu_memory_utilization"], 0.88)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")
        self.assertTrue(kwargs["disable_chunked_mm_input"])
        self.assertEqual(kwargs["download_dir"], os.path.join(HF_HOME, "hub"))

    def test_build_engine_kwargs_with_tracing(self):
        """Arrange-Act-Assert: Verify PyTorch/JAX profiler hook configuration."""
        context = ExecutionContext(
            model_path="/path/to/my_model",
            tensor_parallel_size=4,
            trace=True,
            output_dir="/tmp",
        )

        kwargs = VllmEngineBuilder.build_engine_kwargs(context)

        self.assertIn("profiler_config", kwargs)
        profiler = kwargs["profiler_config"]
        self.assertEqual(profiler.profiler, "torch")
        self.assertEqual(profiler.torch_profiler_dir, "/tmp/traces")
        self.assertTrue(profiler.torch_profiler_with_stack)


class TestTpuEnvironmentResolution(unittest.TestCase):
    """Test suite validating centralized environment and path resolution."""

    def setUp(self):
        self.original_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_root_dir_and_paths(self):
        """Verify default root and paths derive cleanly from ROOT_DIR."""
        self.assertEqual(str(ROOT_DIR), "/mnt/pd")
        self.assertEqual(HF_HOME, "/mnt/pd/huggingface_cache")
        self.assertEqual(SPARSE_CHECKPOINTS_DIR, "/mnt/pd/sparse_checkpoints")
        self.assertEqual(TORCHINDUCTOR_CACHE_DIR,
                         "/mnt/pd/torch_compile_cache")
        self.assertEqual(VLLM_REPO_PATH, "/mnt/pd/vllm")
        self.assertEqual(VLLM_TORCHTPU_SRC, "/mnt/pd/vllm-torchtpu/src")

    def test_setup_tpu_env(self):
        """Verify setup_tpu_env configures all execution flags and module paths."""
        setup_tpu_env()
        self.assertEqual(os.environ.get("VLLM_TARGET_DEVICE"), "tpu")
        self.assertEqual(os.environ.get("VLLM_PLATFORM"), "tpu")
        self.assertEqual(os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"), "0")
        self.assertEqual(os.environ.get("ALLOW_MULTIPLE_LIBTPU_LOAD"), "1")
        self.assertEqual(os.environ.get("USE_BATCHED_RPA_KERNEL"), "1")
        self.assertEqual(os.environ.get("HF_HOME"), HF_HOME)
        self.assertEqual(os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                         TORCHINDUCTOR_CACHE_DIR)
        self.assertIn(VLLM_REPO_PATH, os.environ.get("PYTHONPATH", ""))
        self.assertIn(VLLM_TORCHTPU_SRC, os.environ.get("PYTHONPATH", ""))


if __name__ == "__main__":
    unittest.main()

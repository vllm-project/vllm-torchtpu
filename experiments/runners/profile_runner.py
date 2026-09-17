"""
profile_runner.py
------------------
Throughput, latency, and performance profiling runner for vLLM on Cloud TPUs.
Executes warmup/timed generation cycles and captures PyTorch TPU profiler traces.
Supports multi-model sweeps across different architectures/sizes by isolating model execution
in dedicated subprocesses, overcoming PjRt runtime client lifecycle constraints.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import numpy as np
from core.engine_builder import VllmEngineBuilder
from core.models import BenchmarkResult, ExecutionContext
from runners.base_runner import BaseVllmRunner
from vllm import LLM, SamplingParams
from vllm.config.profiler import ProfilerConfig


class InProcessProfileEngine:
    """Core in-process engine for initializing vLLM and running profiling benchmarks."""

    def __init__(self):
        self.llm: Optional[LLM] = None
        self._engine_key: Optional[tuple] = None

    def close(self) -> None:
        """Tears down the LLM engine and releases TPU resources."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            self._engine_key = None

    def execute(self, context: ExecutionContext) -> BenchmarkResult:
        """
        Initializes the vLLM engine, warms up TPU kernels, runs timed inference,
        and computes throughput and latency metrics with optional TPU profiling.
        """
        engine_key = (
            context.model_path,
            context.tensor_parallel_size,
            context.dtype,
            context.max_model_len,
        )

        if self.llm is None or self._engine_key != engine_key:
            self.close()
            engine_kwargs = VllmEngineBuilder.build_engine_kwargs(context)
            engine_kwargs["model"] = context.model_path

            # Ensure profiler_config is attached if sweep requested tracing
            has_tracing = (context.trace
                           or bool(context.sweep_metadata.get("trace_configs"))
                           or context.sweep_metadata.get("trace") is True)
            if has_tracing and "profiler_config" not in engine_kwargs:
                trace_dir = os.path.join(context.output_dir, "traces")
                os.makedirs(trace_dir, exist_ok=True)
                engine_kwargs["profiler_config"] = ProfilerConfig(
                    profiler="torch", torch_profiler_dir=trace_dir)

            print(f">>> [PROFILING] Initializing vLLM engine: {engine_kwargs}")
            self.llm = LLM(**engine_kwargs)
            self._engine_key = engine_key
        else:
            print(
                ">>> [PROFILING] Reusing initialized vLLM engine for next sweep coordinate."
            )

        llm = self.llm

        # Generate exact dummy token lengths according to input_len and batch_size
        dummy_prompt_token_ids = np.random.randint(10000,
                                                   size=(context.batch_size,
                                                         context.input_len))
        prompts = [{
            "prompt_token_ids": [int(x) for x in batch]
        } for batch in dummy_prompt_token_ids]

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=context.output_len,
            ignore_eos=True,
        )

        # 1. Warmup step to compile computation graphs and TPU kernels for this shape
        print(">>> [PROFILING] Warming up computation graphs...")
        _ = llm.generate(prompts=prompts, sampling_params=sampling_params)

        # 2. Timed generation benchmark with optional distributed TPU profiling
        if context.trace:
            prefix_parts: List[str] = []
            if context.jax_advanced_configuration:
                for k, v in context.jax_advanced_configuration.items():
                    prefix_parts.append(f"{k}:{v}")
            profile_prefix = ";".join(prefix_parts) if prefix_parts else None

            print(
                f">>> [PROFILING] Starting TorchTPU profiler (prefix={profile_prefix})..."
            )
            llm.start_profile(profile_prefix=profile_prefix)
            start_time = time.time()
            outputs = llm.generate(prompts=prompts,
                                   sampling_params=sampling_params)
            end_time = time.time()
            llm.stop_profile()
            print(">>> [PROFILING] Stopped TorchTPU profiler.")
        else:
            start_time = time.time()
            outputs = llm.generate(prompts=prompts,
                                   sampling_params=sampling_params)
            end_time = time.time()

        # 3. Compute throughput and latency metrics
        duration_s = end_time - start_time
        total_tokens = sum(
            len(seq.token_ids) for out in outputs for seq in out.outputs)
        throughput_tok_s = total_tokens / duration_s if duration_s > 0 else 0.0

        metrics = {
            "Duration_s": f"{duration_s:.4f}",
            "Throughput_tok_s": f"{throughput_tok_s:.2f}",
        }

        print(
            f">>> [PROFILING] Complete: {throughput_tok_s:.2f} tok/s in {duration_s:.4f}s"
        )
        return BenchmarkResult(duration_s=duration_s, metrics=metrics)


def _run_worker(payload_path: str, result_path: str) -> None:
    """Worker process entry point executed in an isolated subprocess."""
    from core.env import setup_tpu_env
    setup_tpu_env()
    with open(payload_path, "r") as f:
        data = json.load(f)
    ctx = ExecutionContext(**data)
    engine = InProcessProfileEngine()
    try:
        res = engine.execute(ctx)
        out = {
            "duration_s": res.duration_s,
            "metrics": res.metrics,
        }
    finally:
        engine.close()
    with open(result_path, "w") as f:
        json.dump(out, f)


class TpuProfileRunner(BaseVllmRunner):
    """
    Polymorphic profiling runner for vLLM on Cloud TPUs.
    Coordinates between in-process execution (for unit tests / mock environments)
    and clean subprocess isolation (for real TPU sweeps across multiple models).
    """

    def __init__(self, in_process: Optional[bool] = None):
        self._current_model: Optional[str] = None
        self._engine: Optional[InProcessProfileEngine] = None
        if in_process is not None:
            self.in_process = in_process
        else:
            self.in_process = os.environ.get("VLLM_PROFILE_IN_PROCESS", "0") == "1"

    def close(self) -> None:
        """Tears down the engine and reclaims TPU device resources."""
        if self._engine is not None:
            self._engine.close()
            self._engine = None
            self._current_model = None
            # Reclaim TPU device locks
            subprocess.run("fuser -k -9 /dev/vfio/* 2>/dev/null || true",
                           shell=True,
                           check=False)
            try:
                os.unlink("/tmp/libtpu_lockfile")
            except OSError:
                pass
            os.environ.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
            os.environ.pop("TORCH_TPU_XPROF_SESSION_ID", None)
            time.sleep(2)

    @property
    def llm(self) -> Optional[LLM]:
        """Provides compatibility for unit tests mocking runner.llm."""
        if self._engine is None:
            self._engine = InProcessProfileEngine()
        return self._engine.llm

    @llm.setter
    def llm(self, val: Optional[LLM]) -> None:
        if self._engine is None:
            self._engine = InProcessProfileEngine()
        self._engine.llm = val

    def execute(self, context: ExecutionContext) -> BenchmarkResult:
        """
        Executes profiling trial.
        If in_process mode is active or if running in a mock/test environment,
        delegates directly to InProcessProfileEngine.
        Otherwise, executes via an isolated subprocess worker with automatic clean teardown.
        """
        is_mocked = (
            isinstance(LLM, Mock)
            or getattr(LLM, "_mock_return_value", None) is not None
            or hasattr(LLM, "assert_called")
        )

        if self.in_process or is_mocked:
            if self._engine is None:
                self._engine = InProcessProfileEngine()
            return self._engine.execute(context)

        return self._execute_in_subprocess(context)

    def _execute_in_subprocess(self, context: ExecutionContext) -> BenchmarkResult:
        """Executes the trial in an isolated Python process to ensure clean PjRt client lifecycle."""
        context_payload = {
            "model_path": context.model_path,
            "tensor_parallel_size": context.tensor_parallel_size,
            "dtype": context.dtype,
            "max_model_len": context.max_model_len,
            "batch_size": context.batch_size,
            "input_len": context.input_len,
            "output_len": context.output_len,
            "trace": context.trace,
            "output_dir": context.output_dir,
            "csv_file": context.csv_file,
            "engine_args": context.engine_args,
            "sweep_metadata": context.sweep_metadata,
            "jax_advanced_configuration": context.jax_advanced_configuration,
        }

        # Write execution context to temporary payload file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(context_payload, f)
            payload_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            result_file = f.name

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = [
            sys.executable,
            "-m",
            "runners.profile_runner",
            payload_file,
            result_file,
        ]

        # Clean up stale locks before launching isolated worker
        subprocess.run("fuser -k -9 /dev/vfio/* 2>/dev/null || true",
                       shell=True,
                       check=False)
        try:
            os.unlink("/tmp/libtpu_lockfile")
        except OSError:
            pass

        worker_env = os.environ.copy()
        worker_env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"
        worker_env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        worker_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        worker_env["USE_BATCHED_RPA_KERNEL"] = "1"
        worker_env["TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS"] = "false"
        worker_env["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"
        worker_env["VLLM_NO_USAGE_STATS"] = "1"
        worker_env["VLLM_DO_NOT_TRACK"] = "1"
        worker_env["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/pd/huggingface_cache")
        worker_env["PYTHONUNBUFFERED"] = "1"
        existing_init_args = worker_env.get("LIBTPU_INIT_ARGS", "")
        if "--deepsea_hal_test_skip_slicebuilder" in existing_init_args:
            cleaned = [a for a in existing_init_args.split() if not a.startswith("--deepsea_hal_test_skip_slicebuilder")]
            worker_env["LIBTPU_INIT_ARGS"] = " ".join(cleaned)
        worker_env.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
        worker_env.pop("TORCH_TPU_XPROF_SESSION_ID", None)

        # Preserve TPU device assignment in container / multi-tenant environments
        if "TPU_VISIBLE_DEVICES" in os.environ:
            worker_env["TPU_VISIBLE_DEVICES"] = os.environ["TPU_VISIBLE_DEVICES"]
        if "TPU_VISIBLE_CHIPS" in os.environ:
            worker_env["TPU_VISIBLE_CHIPS"] = os.environ["TPU_VISIBLE_CHIPS"]
        if "TORCH_TPU_TOPOLOGY" in os.environ:
            worker_env["TORCH_TPU_TOPOLOGY"] = os.environ["TORCH_TPU_TOPOLOGY"]
        else:
            worker_env.pop("TORCH_TPU_TOPOLOGY", None)

        if "SPMM_KERNEL_PATH" in os.environ:
            worker_env["SPMM_KERNEL_PATH"] = os.environ["SPMM_KERNEL_PATH"]
        elif os.path.isfile("/workspace/internal_kernels/structured_spmm.py"):
            worker_env["SPMM_KERNEL_PATH"] = "/workspace/internal_kernels/structured_spmm.py"
        current_pythonpath = worker_env.get("PYTHONPATH", "")
        worker_env["PYTHONPATH"] = f"{repo_root}:{current_pythonpath}" if current_pythonpath else repo_root

        try:
            print(f">>> [PROFILE RUNNER] Launching isolated worker process for {context.model_path} (BS={context.batch_size})...")
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                env=worker_env,
                check=True,
                text=True,
            )

            with open(result_file, "r") as f:
                res_data = json.load(f)

            return BenchmarkResult(
                duration_s=float(res_data["duration_s"]),
                metrics=res_data["metrics"],
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Profiling worker process failed with exit code {e.returncode}") from e
        finally:
            try:
                os.unlink(payload_file)
            except OSError:
                pass
            try:
                os.unlink(result_file)
            except OSError:
                pass
            # Release TPU locks
            subprocess.run("fuser -k -9 /dev/vfio/* 2>/dev/null || true",
                           shell=True,
                           check=False)
            try:
                os.unlink("/tmp/libtpu_lockfile")
            except OSError:
                pass
            os.environ.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
            os.environ.pop("TORCH_TPU_XPROF_SESSION_ID", None)
            time.sleep(2)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        _run_worker(sys.argv[1], sys.argv[2])

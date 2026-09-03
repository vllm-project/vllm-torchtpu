"""
profile_runner.py
------------------
Throughput, latency, and performance profiling runner for vLLM on Cloud TPUs.
Executes warmup/timed generation cycles and captures PyTorch TPU profiler traces.
"""

import os
import time
from typing import List, Optional

import numpy as np
from core.engine_builder import VllmEngineBuilder
from core.models import BenchmarkResult, ExecutionContext
from runners.base_runner import BaseVllmRunner
from vllm import LLM, SamplingParams
from vllm.config.profiler import ProfilerConfig


class TpuProfileRunner(BaseVllmRunner):
    """
    Profiles throughput (tokens/sec) and generation latency on TPU v6e.
    Captures native PyTorch TPU traces and fine-grained hardware performance counters.
    Manages vLLM engine instance across sweep coordinates to avoid re-initializing PJRT.
    """

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

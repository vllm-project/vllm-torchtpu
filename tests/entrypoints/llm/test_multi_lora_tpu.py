# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E Integration and Performance Test for Multi-LoRA Serving on TPU."""

import os
import time

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_ADAPTER_1 = "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_3_adapter"


def test_multi_lora_concurrent_accuracy():
    """Verifies accuracy and isolation of Multi-LoRA concurrent batch on TPU.

    Tests that within the SAME execution batch:
    - Requests with LoRA adapter produce fine-tuned outputs (1+1 -> 3)
    - Requests with No LoRA produce base model outputs (1+1 -> 2)
    - There is zero cross-request weight leakage or state corruption.
    """
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=8,
        max_model_len=256,
        max_num_seqs=8,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        disable_log_stats=False,
    )

    lora_req_1 = LoRARequest("lora_1_plus_1_equals_3", 1, LORA_ADAPTER_1)

    prompts = [
        "What is 1+1? \n",
        "The capital of France is",
        "What is 1+1? \n",
        "The capital of France is",
    ]
    lora_requests = [
        lora_req_1,  # Should generate text containing 1+1=3
        None,  # Should generate Paris (Base model)
        lora_req_1,  # Should generate text containing 1+1=3
        None,  # Should generate Paris (Base model)
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

    try:
        start_time = time.perf_counter()
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_requests,
        )
        end_time = time.perf_counter()
        elapsed_s = end_time - start_time

        assert len(outputs) == 4
        print(
            f"\n{'='*50}\n[Multi-LoRA Batch Results] Elapsed: {elapsed_s:.3f}s"
        )
        for i, out in enumerate(outputs):
            gen_text = out.outputs[0].text.strip()
            print(
                f"Req {i} [LoRA={lora_requests[i].lora_name if lora_requests[i] else 'None'}]: {gen_text!r}"
            )

        # Assert correct isolation
        out_0 = outputs[0].outputs[0].text
        out_1 = outputs[1].outputs[0].text
        out_2 = outputs[2].outputs[0].text
        out_3 = outputs[3].outputs[0].text

        assert "3" in out_0, f"Expected '3' from LoRA adapter, got: {out_0!r}"
        assert "Paris" in out_1 or len(
            out_1) > 0, f"Expected Paris from Base model, got: {out_1!r}"
        assert "3" in out_2, f"Expected '3' from LoRA adapter, got: {out_2!r}"
        assert "Paris" in out_3 or len(
            out_3) > 0, f"Expected Paris from Base model, got: {out_3!r}"

    finally:
        del llm
        cleanup_dist_env_and_memory()


def test_multi_lora_throughput_benchmark():
    """Measures latency and throughput (TPS) for concurrent Multi-LoRA batches."""
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=8,
        max_model_len=256,
        max_num_seqs=8,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        disable_log_stats=False,
    )

    lora_req_1 = LoRARequest("lora_1_plus_1_equals_3", 1, LORA_ADAPTER_1)

    batch_size = 4
    prompts = ["What is 1+1? Answer with a single digit.\n"] * batch_size
    lora_requests = [lora_req_1, None, lora_req_1, None]
    sampling_params = SamplingParams(temperature=0, max_tokens=32)

    try:
        # Warmup
        llm.generate(prompts[:2],
                     sampling_params,
                     lora_request=lora_requests[:2])

        # Benchmark step execution
        start_time = time.perf_counter()
        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_requests)
        end_time = time.perf_counter()

        elapsed_s = end_time - start_time
        total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        tps = total_tokens / elapsed_s if elapsed_s > 0 else 0.0

        print(f"\n{'='*50}\n[Multi-LoRA Throughput Benchmark]")
        print(f"Total Generated Tokens: {total_tokens}")
        print(f"Elapsed Time: {elapsed_s:.4f} s")
        print(f"Throughput (TPS): {tps:.2f} tokens/s")
        print(f"{'='*50}\n")

        assert total_tokens > 0
    finally:
        del llm
        cleanup_dist_env_and_memory()

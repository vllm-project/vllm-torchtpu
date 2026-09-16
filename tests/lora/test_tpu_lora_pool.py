# SPDX-License-Identifier: Apache-2.0
"""Unit test for TPU Multi-LoRA Static Weight Pool and Einsum Linear Layer."""

import time

import pytest
import torch
import torch.nn as nn
from vllm.config.lora import LoRAConfig
from vllm.lora.layers import LoRAMapping

from vllm_torchtpu.lora.torch_punica_tpu import PunicaWrapperTPU
from vllm_torchtpu.lora.tpu_lora_pool import (TPUMultiLoRALinear,
                                              TPUMultiLoRAPool)


def get_test_device():
    try:
        return torch.device("tpu")
    except Exception:
        return torch.device("cpu")


def test_tpu_lora_pool_slot_management():
    pool = TPUMultiLoRAPool(max_loras=3, max_lora_rank=8, device="cpu")
    slot1 = pool.get_or_allocate_slot(101)
    slot2 = pool.get_or_allocate_slot(102)
    slot3 = pool.get_or_allocate_slot(103)
    assert slot1 == 0
    assert slot2 == 1
    assert slot3 == 2
    assert pool.get_or_allocate_slot(101) == 0

    with pytest.raises(RuntimeError):
        pool.get_or_allocate_slot(104)

    freed = pool.free_slot(102)
    assert freed == 1
    slot_new = pool.get_or_allocate_slot(105)
    assert slot_new == 1


def test_tpu_multi_lora_linear_numerical_accuracy_and_isolation():
    """Verifies that TPUMultiLoRALinear mathematically matches standard LoRA forward

    and guarantees exact isolation across concurrent mixed batches (LoRA + Base).
    """
    device = get_test_device()
    dtype = torch.bfloat16

    in_dim = 128
    out_dim = 128
    rank = 8
    max_loras = 4
    scale = 0.5

    base_linear = nn.Linear(in_dim,
                            out_dim,
                            bias=False,
                            dtype=dtype,
                            device=device)
    lora_config = LoRAConfig(max_loras=max_loras,
                             max_lora_rank=rank,
                             lora_dtype=dtype)

    lora_linear = TPUMultiLoRALinear(base_linear, max_loras, rank,
                                     lora_config).to(device)

    # Adapter 1 (slot 0, lora_indices = 1)
    lora_a_1 = torch.randn(rank, in_dim, dtype=dtype, device=device)
    lora_b_1 = torch.randn(out_dim, rank, dtype=dtype, device=device)
    lora_linear.set_lora_weights(0, lora_a_1, lora_b_1, scaling=scale)

    # Adapter 2 (slot 1, lora_indices = 2)
    lora_a_2 = torch.randn(rank, in_dim, dtype=dtype, device=device)
    lora_b_2 = torch.randn(out_dim, rank, dtype=dtype, device=device)
    lora_linear.set_lora_weights(1, lora_a_2, lora_b_2, scaling=scale)

    # 4 requests: [LoRA 1, Base (0), LoRA 2, Base (0)]
    x = torch.randn(4, in_dim, dtype=dtype, device=device)
    lora_indices = torch.tensor([1, 0, 2, 0], dtype=torch.long, device=device)

    # TPUMultiLoRALinear forward
    out_einsum = lora_linear(x, lora_indices)

    # Reference PyTorch Calculation
    base_ref = base_linear(x)
    ref_out = base_ref.clone()

    # Req 0 (LoRA 1): x[0] @ A_1.T @ B_1.T * scale
    delta_0 = (x[0:1] @ lora_a_1.T @ lora_b_1.T) * scale
    ref_out[0:1] += delta_0

    # Req 1 (Base 0): delta is exactly 0
    # Req 2 (LoRA 2): x[2] @ A_2.T @ B_2.T * scale
    delta_2 = (x[2:3] @ lora_a_2.T @ lora_b_2.T) * scale
    ref_out[2:3] += delta_2

    # Req 3 (Base 0): delta is exactly 0

    # Assert exact numerical equivalence within bfloat16 tolerance
    torch.testing.assert_close(out_einsum, ref_out, atol=1e-2, rtol=1e-2)
    # Check Base requests (index 1 and 3) have zero alteration from base_linear
    torch.testing.assert_close(out_einsum[1],
                               base_ref[1],
                               atol=1e-4,
                               rtol=1e-4)
    torch.testing.assert_close(out_einsum[3],
                               base_ref[3],
                               atol=1e-4,
                               rtol=1e-4)


def test_tpu_lora_runtime_latency_benchmark_against_baseline():
    """Directly compares execution step latency of Baseline Punica vs TPUMultiLoRALinear."""
    device = get_test_device()
    dtype = torch.bfloat16

    in_dim = 2048
    out_dim = 2048
    rank = 8
    max_loras = 4
    num_tokens = 64
    warmup_iters = 10
    bench_iters = 50

    base_linear = nn.Linear(in_dim,
                            out_dim,
                            bias=False,
                            dtype=dtype,
                            device=device)
    lora_config = LoRAConfig(max_loras=max_loras,
                             max_lora_rank=rank,
                             lora_dtype=dtype)

    # 1. Setup Baseline Punica
    punica = PunicaWrapperTPU(max_num_batched_tokens=num_tokens,
                              max_batches=num_tokens,
                              device=device)
    lora_a_stacked = (torch.randn(max_loras,
                                  1,
                                  rank,
                                  in_dim,
                                  dtype=dtype,
                                  device=device), )
    lora_b_stacked = (torch.randn(max_loras,
                                  1,
                                  out_dim,
                                  rank,
                                  dtype=dtype,
                                  device=device), )

    # 2. Setup TPUMultiLoRALinear
    tensorized_linear = TPUMultiLoRALinear(base_linear, max_loras, rank,
                                           lora_config).to(device)
    for slot in range(max_loras):
        tensorized_linear.set_lora_weights(
            slot,
            torch.randn(rank, in_dim, dtype=dtype, device=device),
            torch.randn(out_dim, rank, dtype=dtype, device=device),
            scaling=1.0,
        )

    x = torch.randn(num_tokens, in_dim, dtype=dtype, device=device)
    indices = torch.randint(0,
                            max_loras + 1, (num_tokens, ),
                            dtype=torch.long,
                            device=device)

    # --- Benchmark Baseline Punica (with per-step metadata update) ---
    for _ in range(warmup_iters):
        mapping = LoRAMapping(tuple(indices.tolist()),
                              tuple(indices.tolist()),
                              is_prefill=True)
        lora_index_to_id = list(range(1, max_loras + 1))
        punica.update_metadata(mapping, lora_index_to_id, max_loras, 32000)
        y_base = base_linear(x)
        _ = punica.add_lora_linear(y_base, x, lora_a_stacked, lora_b_stacked,
                                   1.0, (out_dim, ))
        if hasattr(device, "type") and device.type == "xla":
            import torch_xla.core.xla_model as xm
            xm.mark_step()

    t0 = time.perf_counter()
    for _ in range(bench_iters):
        mapping = LoRAMapping(tuple(indices.tolist()),
                              tuple(indices.tolist()),
                              is_prefill=True)
        lora_index_to_id = list(range(1, max_loras + 1))
        punica.update_metadata(mapping, lora_index_to_id, max_loras, 32000)
        y_base = base_linear(x)
        _ = punica.add_lora_linear(y_base, x, lora_a_stacked, lora_b_stacked,
                                   1.0, (out_dim, ))
        if hasattr(device, "type") and device.type == "xla":
            import torch_xla.core.xla_model as xm
            xm.mark_step()
    t1 = time.perf_counter()
    punica_ms = ((t1 - t0) / bench_iters) * 1000.0

    # --- Benchmark TPUMultiLoRALinear with torch.compile (XLA Fused Einsum) ---
    compiled_tensorized = torch.compile(tensorized_linear)
    for _ in range(warmup_iters):
        _ = compiled_tensorized(x, indices)
        if hasattr(device, "type") and device.type == "xla":
            import torch_xla.core.xla_model as xm
            xm.mark_step()

    t0 = time.perf_counter()
    for _ in range(bench_iters):
        _ = compiled_tensorized(x, indices)
        if hasattr(device, "type") and device.type == "xla":
            import torch_xla.core.xla_model as xm
            xm.mark_step()
    t1 = time.perf_counter()
    einsum_ms = ((t1 - t0) / bench_iters) * 1000.0

    speedup = punica_ms / max(einsum_ms, 1e-6)
    print(
        f"\n{'='*60}\n[TPU Multi-LoRA Real-Serving Latency Comparison (tokens={num_tokens})]"
    )
    print(f"Punica Baseline (Host Meta + Op) : {punica_ms:.4f} ms/step")
    print(f"Tensorized Einsum (XLA Fused)    : {einsum_ms:.4f} ms/step")
    print(f"Speedup                          : {speedup:.2f}x")
    print(f"{'='*60}\n")
    assert einsum_ms < punica_ms or speedup >= 1.0

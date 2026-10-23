# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared TPU-vs-CPU parity helpers for the LTX-2 component test suite."""

import torch

# Default error-ratio bounds (omni_rel_l2 / diffusers_cpu_bf16_rel_l2), set ~3%
# above the worst observed ratios across LTX-2 DiT and connector checks
# (fp32: 0.524x-0.651x; bf16: 1.056x-1.135x).
FP32_ERR_RATIO = 0.67
BF16_ERR_RATIO = 1.17
BF16_ERR_FLOOR = 1e-3


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Compute cosine similarity, relative L2 error, and max absolute error in fp64."""
    actual_flat = actual.detach().cpu().double().flatten()
    expected_flat = expected.detach().cpu().double().flatten()
    diff = actual_flat - expected_flat
    norm_product = actual_flat.norm() * expected_flat.norm()
    cos = torch.dot(actual_flat, expected_flat) / norm_product.clamp_min(1e-30)
    return {
        "cos": float(cos),
        "rel_l2": float(diff.norm() / expected_flat.norm().clamp_min(1e-30)),
        "max_abs": float(diff.abs().max()),
    }


def assert_parity(
    *,
    ref: torch.Tensor,
    ref_bf16: torch.Tensor,
    tpu_fp32: torch.Tensor,
    tpu_bf16: torch.Tensor,
    fp32_ratio: float = FP32_ERR_RATIO,
    bf16_ratio: float = BF16_ERR_RATIO,
    floor: float = BF16_ERR_FLOOR,
) -> dict[str, object]:
    """Assert TPU fp32 and bf16 outputs match the CPU fp32 reference.

    Both TPU error budgets are normalized against the CPU bf16 reference error
    so depth and conditioning sensitivity cancel out across components.
    """
    ref_bf16_metrics = metrics(ref_bf16, ref)
    tpu_fp32_metrics = metrics(tpu_fp32, ref)
    tpu_bf16_metrics = metrics(tpu_bf16, ref)
    ref_bf16_err = ref_bf16_metrics["rel_l2"]
    denom = max(ref_bf16_err, 1e-30)
    results: dict[str, object] = {
        "diffusers_cpu_bf16": ref_bf16_metrics,
        "omni_tpu_fp32": tpu_fp32_metrics,
        "omni_tpu_bf16": tpu_bf16_metrics,
        "fp32_ratio": tpu_fp32_metrics["rel_l2"] / denom,
        "bf16_ratio": tpu_bf16_metrics["rel_l2"] / denom,
    }
    fp32_budget = max(fp32_ratio * ref_bf16_err, floor)
    bf16_budget = max(bf16_ratio * ref_bf16_err, floor)
    assert tpu_fp32_metrics["rel_l2"] <= fp32_budget, results
    assert tpu_bf16_metrics["rel_l2"] <= bf16_budget, results
    return results


def assert_all_loaded(loaded: set[str], module: torch.nn.Module) -> None:
    """Assert that every parameter of ``module`` was populated by ``load_weights``."""
    missing = {name for name, _ in module.named_parameters()} - loaded
    assert not missing, (
        f"vllm-omni parameters not loaded from checkpoint: {sorted(missing)[:10]}"
    )

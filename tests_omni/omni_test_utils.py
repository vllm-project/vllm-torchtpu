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
"""Numerical-parity and image-metric helpers shared by the Omni tests."""

import numpy as np
import torch

# Failure messages of assert_numerical_parity for a numeric mismatch. Must-fail
# controls match on it, so other AssertionErrors, such as a shape mismatch, do
# not satisfy them.
NUMERIC_MISMATCH_PATTERN = r"Cosine similarity|Mismatched elements"


def assert_numerical_parity(
    actual: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    min_cos_sim: float = 0.999,
):
    """Assert cosine similarity and floating-point tolerance bounds."""
    a_flat = actual.flatten().float()
    r_flat = reference.flatten().float()

    norm_a = torch.linalg.norm(a_flat)
    norm_r = torch.linalg.norm(r_flat)
    assert norm_a > 0 and norm_r > 0, (
        f"Zero or NaN norm (actual {float(norm_a)}, reference {float(norm_r)}); "
        "cannot compare directions"
    )
    cos_sim = float(torch.dot(a_flat, r_flat) / (norm_a * norm_r))
    assert cos_sim >= min_cos_sim, (
        f"Cosine similarity {cos_sim:.5f} fell below minimum required {min_cos_sim}"
    )

    torch.testing.assert_close(
        actual.float(),
        reference.float(),
        atol=atol,
        rtol=rtol,
        msg=lambda m: f"TPU kernel output diverged from CPU baseline\n{m}",
    )


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Global (single-window) SSIM, mean over RGB channels."""
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    ssims = []
    for c in range(3):
        x, y = img1[..., c], img2[..., c]
        mu_x, mu_y = x.mean(), y.mean()
        var_x, var_y = x.var(), y.var()
        cov_xy = np.mean((x - mu_x) * (y - mu_y))
        num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        ssims.append(float(num / den))
    return float(np.mean(ssims))


def high_freq_energy(img: np.ndarray) -> float:
    """Mean absolute Laplacian: drops with blur, rises with noise or blocking."""
    laplacian = (
        4 * img[1:-1, 1:-1]
        - img[:-2, 1:-1]
        - img[2:, 1:-1]
        - img[1:-1, :-2]
        - img[1:-1, 2:]
    )
    return float(np.abs(laplacian).mean())


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


def assert_within_bf16_noise(
    *,
    ref: torch.Tensor,
    ref_bf16: torch.Tensor,
    tpu_fp32: torch.Tensor,
    tpu_bf16: torch.Tensor,
    fp32_ratio: float,
    bf16_ratio: float,
    floor: float = 1e-3,
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

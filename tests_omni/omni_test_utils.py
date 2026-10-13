# SPDX-License-Identifier: Apache-2.0
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

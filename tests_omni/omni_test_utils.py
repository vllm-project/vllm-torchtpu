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
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import LinearBase


def init_test_weights(module: torch.nn.Module, seed: int) -> None:
    """Seed vLLM parallel linear and Conv3d layers allocated with ``torch.empty``."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for submodule in module.modules():
            if isinstance(submodule, (LinearBase, Conv3dLayer)):
                for param in submodule.parameters(recurse=False):
                    if param.dim() > 1:
                        param.copy_(
                            torch.randn(param.shape, generator=generator) * 0.02
                        )
                    else:
                        param.zero_()


def use_native_ops_on_cpu(module: torch.nn.Module) -> None:
    """Bind CPU-compatible reference methods on vLLM CustomOp instances."""

    def _attention(query, key, value, cu_seqlens=None, max_seqlen=None, **_):
        q, k, v = (t.transpose(1, 2).double() for t in (query, key, value))
        bounds = cu_seqlens.tolist() if cu_seqlens is not None else [0, q.shape[2]]
        scale = q.shape[-1] ** -0.5
        out = torch.empty_like(q)
        for start, end in zip(bounds[:-1], bounds[1:]):
            scores = q[:, :, start:end] @ k[:, :, start:end].transpose(-1, -2) * scale
            out[:, :, start:end] = scores.softmax(dim=-1) @ v[:, :, start:end]
        return out.transpose(1, 2).to(query.dtype)

    for submodule in module.modules():
        if isinstance(submodule, MMEncoderAttention):
            submodule._forward_method = _attention
        elif isinstance(submodule, CustomOp):
            submodule._forward_method = submodule.forward_native


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


def assert_parity(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
    max_rel_l2: float | None = None,
    max_abs: float | None = None,
) -> None:
    """Assert ``actual`` matches ``reference`` on each gate that is provided.

    - ``max_rel_l2``: ``||actual - reference|| / ||reference||`` upper bound.
    - ``max_abs``: largest elementwise absolute difference upper bound.

    At least one gate is required. Metrics are computed in fp64.
    """
    assert max_rel_l2 is not None or max_abs is not None, f"{name}: no gate given"
    assert actual.shape == reference.shape, (
        f"{name}: shape ({tuple(actual.shape)}) != ({tuple(reference.shape)})"
    )
    actual_flat = actual.detach().cpu().double().flatten()
    reference_flat = reference.detach().cpu().double().flatten()
    diff = actual_flat - reference_flat
    if max_rel_l2 is not None:
        rel_l2 = float(diff.norm() / reference_flat.norm())
        assert rel_l2 <= max_rel_l2, (
            f"{name}: relative L2 error {rel_l2:.4e} exceeds max {max_rel_l2:.2e}"
        )
    if max_abs is not None:
        abs_err = float(diff.abs().max())
        assert abs_err <= max_abs, (
            f"{name}: max absolute error {abs_err:.4e} exceeds max {max_abs:.2e}"
        )


def assert_all_loaded(loaded: set[str], module: torch.nn.Module) -> None:
    """Assert that every parameter of ``module`` was populated by ``load_weights``."""
    missing = {name for name, _ in module.named_parameters()} - loaded
    assert not missing, (
        f"vllm-omni parameters not loaded from checkpoint: {sorted(missing)[:10]}"
    )

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
"""Numerical-parity, image-metric, and patch-isolation helpers for Omni tests."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention import mm_encoder_attention
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import LinearBase
from vllm.v1.attention.ops.vit_attn_wrappers import torch_sdpa_wrapper

_MISSING = object()


@contextmanager
def restore_class_attrs(cls: type, attrs: Sequence[str]) -> Generator[None]:
    """Snapshot and restore direct attributes on ``cls`` across a test or fixture."""
    saved = {attr: cls.__dict__.get(attr, _MISSING) for attr in attrs}
    try:
        yield
    finally:
        for attr, val in saved.items():
            if val is _MISSING:
                if attr in cls.__dict__:
                    delattr(cls, attr)
            else:
                setattr(cls, attr, val)


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


def compare_video_to_golden(video: np.ndarray, golden: np.ndarray) -> dict[str, float]:
    """Compare two uint8 [F, H, W, 3] videos with the Flux golden-test metrics.

    Returns the mean per-frame MAE and SSIM (single window, per channel), the
    largest per-channel mean shift, and the video-to-golden ratios of
    high-frequency energy (mean absolute Laplacian: drops with blur, rises with
    noise) and of motion (mean absolute difference between consecutive frames).
    """
    if video.shape != golden.shape:
        raise ValueError(f"video shape {video.shape} != golden {golden.shape}")
    x = video.astype(np.float64) / 255.0
    y = golden.astype(np.float64) / 255.0
    hw = (1, 2)
    mu_x, mu_y = x.mean(hw, keepdims=True), y.mean(hw, keepdims=True)
    cov = ((x - mu_x) * (y - mu_y)).mean(hw, keepdims=True)
    var_x, var_y = x.var(hw, keepdims=True), y.var(hw, keepdims=True)
    c1, c2 = 0.01**2, 0.03**2
    ssim = (
        (2 * mu_x * mu_y + c1)
        * (2 * cov + c2)
        / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    )
    hf_x, hf_y = (
        np.abs(
            4 * v[:, 1:-1, 1:-1]
            - v[:, :-2, 1:-1]
            - v[:, 2:, 1:-1]
            - v[:, 1:-1, :-2]
            - v[:, 1:-1, 2:]
        ).mean()
        for v in (x, y)
    )
    motion_x, motion_y = (np.abs(np.diff(v, axis=0)).mean() for v in (x, y))
    return {
        "mae": float(np.abs(x - y).mean()),
        "ssim": float(ssim.mean()),
        "mean_shift": float(np.abs(mu_x.mean(0) - mu_y.mean(0)).max()),
        "hf_ratio": float(hf_x / max(hf_y, 1e-9)),
        "motion_ratio": float(motion_x / max(motion_y, 1e-9)),
    }


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


def init_test_weights(module: nn.Module, seed: int = 0) -> None:
    """Seed vLLM CustomOp layers (``LinearBase``, ``Conv3dLayer``) deterministically."""
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


@contextmanager
def native_ops_on_cpu(*modules: nn.Module) -> Generator[None]:
    """Run the vLLM ``CustomOp`` layers of ``modules`` with their native CPU ops."""
    for module in modules:
        for submodule in module.modules():
            if isinstance(submodule, CustomOp):
                submodule._forward_method = submodule.forward_native
    with patch.object(
        mm_encoder_attention, "vit_torch_sdpa_wrapper", torch_sdpa_wrapper
    ):
        yield


@contextmanager
def isolate_thinker_patch_targets() -> Generator[None]:
    """Snapshot and restore Qwen3-Omni Thinker patch targets across a test."""
    from vllm_omni.model_executor.models.qwen3_omni import (
        qwen3_omni_moe_thinker as thinker_mod,
    )

    with (
        restore_class_attrs(
            thinker_mod.Qwen3OmniMoeAudioEncoder,
            ("forward", "_tpu_enforce_eager"),
        ),
        restore_class_attrs(
            thinker_mod.Qwen3Omni_VisionTransformer,
            ("forward", "_tpu_orig_forward"),
        ),
    ):
        yield

# SPDX-License-Identifier: Apache-2.0
"""TPU runtime patches for vLLM-Omni Wan 2.1 diffusion model."""

from functools import cache as run_once
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@run_once
def apply_omni_tpu_patches() -> None:
    """Extend async output timeout and patch TPU unsupported ops."""

    # Wan models need more time for inference
    from vllm_omni.diffusion import diffusion_engine
    diffusion_engine._ASYNC_OUTPUT_TIMEOUT = 1800.0
    logger.info("Applied TPU patch: extended _ASYNC_OUTPUT_TIMEOUT to 1800s.")

    # Patch 1D linear interpolation on TPU to run on CPU to avoid missing aten::upsample_linear1d
    orig_interpolate = torch.nn.functional.interpolate

    def _safe_interpolate(
        input: torch.Tensor,
        size: Any = None,
        scale_factor: Any = None,
        mode: str = "nearest",
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if input.device.type == "tpu" and mode == "linear":
            return orig_interpolate(input.cpu(), size, scale_factor, mode,
                                    *args, **kwargs).to(input.device)
        return orig_interpolate(input, size, scale_factor, mode, *args,
                                **kwargs)

    torch.nn.functional.interpolate = _safe_interpolate
    logger.info(
        "Applied TPU patch: safe torch.nn.functional.interpolate for 1D linear TPU ops."
    )

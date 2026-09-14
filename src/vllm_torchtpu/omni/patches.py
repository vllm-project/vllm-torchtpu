# SPDX-License-Identifier: Apache-2.0
"""TPU runtime patches for vLLM-Omni diffusion models."""

from functools import cache as run_once
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@run_once
def apply_omni_tpu_patches() -> None:
    """Extend async output timeout and patch TPU unsupported ops."""

    # Extend async output timeout for large model weight loading.
    # Keep deferred here: moving to module top level causes a circular import
    # during plugin discovery (rotary_embedding.mrope -> platforms.__getattr__
    # -> OmniTpuPlatform -> patches -> diffusion_engine -> stage_utils ->
    # current_omni_platform).
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


@run_once
def apply_omni_model_specific_patches() -> None:
    """Apply model-specific runtime patches."""

    # Output tensors must be moved to CPU before IPC serialization back to the
    # orchestrator process; holding TPU tensors across process boundaries causes
    # unpickling errors, PJRT re-initialization, and /tmp/libtpu_lockfile lock
    # contention in the orchestrator.
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    orig_decode_output = ltx2_runtime.LTXRuntime._decode_output

    def _safe_decode_output(self: Any, *args: Any, **kwargs: Any):
        res = orig_decode_output(self, *args, **kwargs)
        video, audio = res.output
        res.output = (
            video.cpu() if isinstance(video, torch.Tensor) else video,
            audio.cpu() if isinstance(audio, torch.Tensor) else audio,
        )
        return res

    ltx2_runtime.LTXRuntime._decode_output = _safe_decode_output
    logger.info("Applied TPU patch: safe LTX-2 CPU decode output.")

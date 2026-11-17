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
"""TPU runtime patches for Wan diffusion models."""

from functools import cache as run_once
from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.omni.patches import is_model_class_in, mark_model_patch_applied

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

# Keys of vLLM-Omni's diffusion model registry (vllm_omni/diffusion/registry.py).
# CausVid builds its transformer from the Wan 2.2 modules.
_WAN_PIPELINES = frozenset(
    {
        "WanPipeline",
        "WanDMDPipeline",
        "WanVACEPipeline",
        "WanImageToVideoPipeline",
        "WanS2VPipeline",
        "WanT2VDMD2Pipeline",
        "WanI2VDMD2Pipeline",
        "CausVidWanPipeline",
    }
)


@run_once
def _optimization_barrier_op() -> Any:
    from vllm_torchtpu.layers.adapter.custom_ops.optimization_barrier_op import (
        optimization_barrier_op,
    )

    return optimization_barrier_op()


_FFN_GELU_FENCE_PATCHED = "_tpu_ffn_gelu_fence_patched"


def _patch_wan_ffn_gelu_fence(
    od_config: "OmniDiffusionConfig | None" = None,
) -> bool | None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once XLA stops duplicating the elementwise producer
    # into the matmul operand fusion here, or torch_tpu exposes a fusion fence.
    #
    # XLA fuses the Wan FFN up-projection's bias add and tanh-GELU into the
    # operand fusion of the down-projection matmul, so the GELU of the
    # (tokens, 13824) activation is recomputed for every output tile of the
    # down-projection instead of once. torch_tpu has no way to stop a
    # producer fusion. An XLA optimization barrier on the GELU output (an
    # identity that fusion passes cannot see through) makes XLA materialize
    # it once, in bf16, which is also the value eager PyTorch produces.
    # Measured on v7x, Wan2.2-I2V-A14B: 72.72 -> 71.39 ms per block at
    # 32,760 tokens on one core; 0.939 -> 0.927 s/step end to end at
    # 480x832x81 on 8 cores (CFG 2 x Ulysses 4).
    # -------------------------------------------------------------------------
    if not is_model_class_in(od_config, _WAN_PIPELINES):
        return False
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        ColumnParallelGELU,
    )

    mark_model_patch_applied()
    if getattr(ColumnParallelGELU.forward, _FFN_GELU_FENCE_PATCHED, False):
        return
    orig_forward = getattr(
        ColumnParallelGELU, "_tpu_orig_forward", ColumnParallelGELU.forward
    )
    ColumnParallelGELU._tpu_orig_forward = orig_forward

    def _fenced_gelu_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        return _optimization_barrier_op()(orig_forward(self, x))

    setattr(_fenced_gelu_forward, _FFN_GELU_FENCE_PATCHED, True)
    ColumnParallelGELU.forward = _fenced_gelu_forward
    logger.info("Applied TPU patch: fusion fence after Wan FFN GELU.")


_FUSED_QK_NORM_ROPE_PATCHED = "_tpu_fused_qk_norm_rope_patched"


def _patch_wan_fused_qk_norm_rope(
    od_config: "OmniDiffusionConfig | None" = None,
) -> bool | None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once XLA fuses the q/k RMSNorm and RoPE into one
    # pass at HBM bandwidth, or vLLM-Omni's CustomOp gains an out-of-tree dispatch
    # hook for a fused q/k norm + RoPE op.
    #
    # WanSelfAttention normalizes q and k (RMSNorm over all heads) and then
    # rotates them. XLA splits this into a row-reduction fusion for the norm
    # statistics, an apply fusion and the RoPE fusion, each streaming the
    # (tokens, 5120) q and k through HBM; the row reduction also runs well
    # below HBM bandwidth. Together ~0.78 ms per block at 8190 tokens per
    # core on v7x. A Pallas kernel that reads q and k once and writes them
    # once takes 0.23 ms and is closer to an f64 reference (single rounding).
    # -------------------------------------------------------------------------
    if not is_model_class_in(od_config, _WAN_PIPELINES):
        return False
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        WanSelfAttention,
    )

    from vllm_torchtpu.omni.rope import (
        can_fuse_qk_norm_rope,
        wan_self_attention_fused,
    )

    mark_model_patch_applied()
    if getattr(WanSelfAttention.forward, _FUSED_QK_NORM_ROPE_PATCHED, False):
        return
    WanSelfAttention._tpu_orig_forward = getattr(
        WanSelfAttention, "_tpu_orig_forward", WanSelfAttention.forward
    )

    def _fused_self_attention_forward(
        self: Any,
        hidden_states: torch.Tensor,
        rotary_emb: Any = None,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        if not can_fuse_qk_norm_rope(self, hidden_states, rotary_emb):
            return WanSelfAttention._tpu_orig_forward(
                self, hidden_states, rotary_emb, attn_metadata
            )
        return wan_self_attention_fused(self, hidden_states, rotary_emb, attn_metadata)

    setattr(_fused_self_attention_forward, _FUSED_QK_NORM_ROPE_PATCHED, True)
    WanSelfAttention.forward = _fused_self_attention_forward
    logger.info("Applied TPU patch: fused q/k RMSNorm + RoPE for Wan attention.")


def _patch_wan_fresh_timestep(
    od_config: "OmniDiffusionConfig | None" = None,
) -> bool | None:
    # Wan: give the transformer its own copy of the timestep.
    #
    # Problem: Wan pipelines pass `t.expand(batch)`, a view into
    #   scheduler.timesteps. torch_tpu bakes a view's offset into every graph
    #   that reads it, so the transformer recompiles at every denoising step.
    # Fix: clone the timestep on entry and flush the clone right away, so only
    #   a tiny copy graph sees the view.
    # Effect (v7x, 480x832x81, 8 cores): first 12-step request 53.0 s -> 26.9 s
    #   (warm: 15.3 s); output is bit-identical.
    # Scope: WanTransformer3DModel (T2V, I2V, TI2V, both A14B experts),
    #   WanVACETransformer3DModel (overrides forward) and
    #   WanS2VTransformer3DModel (separate class). The numbers above are T2V;
    #   the VACE and S2V classes are covered but not measured on TPU.
    # TODO: remove once torch_tpu passes view offsets as runtime inputs.
    if not is_model_class_in(od_config, _WAN_PIPELINES):
        return False
    from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import (
        WanS2VTransformer3DModel,
    )
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        WanTransformer3DModel,
    )
    from vllm_omni.diffusion.models.wan2_2.wan2_2_vace_transformer import (
        WanVACETransformer3DModel,
    )

    mark_model_patch_applied()
    for cls in (
        WanTransformer3DModel,
        WanVACETransformer3DModel,
        WanS2VTransformer3DModel,
    ):
        _wrap_fresh_timestep(cls)
    logger.info(
        "Applied TPU patch: fresh timestep buffer for WanTransformer3DModel, "
        "WanVACETransformer3DModel and WanS2VTransformer3DModel."
    )


_FRESH_TIMESTEP_PATCHED = "_tpu_fresh_timestep_patched"


def _wrap_fresh_timestep(cls: type) -> None:
    # The flag lives on the forward function, not the class:
    # WanVACETransformer3DModel subclasses WanTransformer3DModel but has its
    # own forward, which still needs the wrapper.
    if getattr(cls.forward, _FRESH_TIMESTEP_PATCHED, False):
        return
    forward = cls.forward

    def _fresh_timestep_forward(
        self: Any, hidden_states: Any, timestep: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if isinstance(timestep, torch.Tensor):
            from vllm_torchtpu.utils import synchronize_tensors

            timestep = timestep.clone()
            synchronize_tensors(timestep, wait=False)
        return forward(self, hidden_states, timestep, *args, **kwargs)

    setattr(_fresh_timestep_forward, _FRESH_TIMESTEP_PATCHED, True)
    cls.forward = _fresh_timestep_forward

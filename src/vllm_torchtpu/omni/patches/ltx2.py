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
"""TPU runtime patches for LTX-2 diffusion models."""

from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.omni.patches import is_model_class_in, mark_model_patch_applied

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

# Keys of vLLM-Omni's diffusion model registry (vllm_omni/diffusion/registry.py).
_LTX2_PIPELINES = frozenset(
    {
        "LTX2Pipeline",
        "LTX2TextToAudioPipeline",
        "LTX2TwoStagePipeline",
        "LTX2DistilledOneStagePipeline",
        "LTX2DistilledTwoStagePipeline",
        "LTX2DistilledPipeline",
        "LTX2T2VDMD2Pipeline",
        "LTX2I2VDMD2Pipeline",
    }
)

_VOCODER_CPU_INIT_PATCHED = "_tpu_vocoder_cpu_init_patched"


def _patch_ltx2_vocoder_cpu_init(
    od_config: "OmniDiffusionConfig | None" = None,
) -> bool | None:
    # -------------------------------------------------------------------------
    # TODO: Remove once torch_tpu implements kaiser_window and sinc, or
    # vLLM-Omni builds the vocoder and runs
    # ltx2_runtime._restore_ltx_bwe_resampler_filter on CPU.
    #
    # Problem: DiffusersPipelineLoader._init_from_load_format constructs the
    #   pipeline under a `with torch.device("tpu:N"):` context. LTX-2.3 and
    #   LTX-2.5 use Diffusers LTX2VocoderWithBWE (`act_fn="snakebeta"`,
    #   `antialias=True`), whose __init__ builds AntiAliasAct1d -> UpSample1d
    #   -> kaiser_sinc_filter1d and calls torch.kaiser_window(...) on the
    #   active default device before weights are loaded, failing on TPU with
    #   `DispatchStub: missing kernel for tpu`. At decode,
    #   ltx2_runtime._restore_ltx_bwe_resampler_filter rebuilds the BWE
    #   resampler's FP32 Hann filter with torch.sinc on the filter's device,
    #   failing on TPU with `operator 'aten::sinc.out' is not implemented`.
    # Fix: Constructing the vocoder on CPU (matching text_encoder and
    #   latent_upsampler in ltx2_components.py) builds the filter buffers on
    #   the host before _place_aux_components moves the module to
    #   pipeline.device. The restore helper runs upstream's rebuild on a CPU
    #   stand-in filter and moves the result to the filter's device, so later
    #   decodes hit upstream's identity check and skip the rebuild.
    # Effect: LTX-2.3 loads and decodes on TPU.
    # -------------------------------------------------------------------------
    if not is_model_class_in(od_config, _LTX2_PIPELINES):
        return False
    from vllm_omni.diffusion.models.ltx2 import ltx2_components, ltx2_runtime

    mark_model_patch_applied()
    _wrap_ltx2_bwe_filter_restore(ltx2_runtime)
    if getattr(ltx2_components._load_component, _VOCODER_CPU_INIT_PATCHED, False):
        return
    orig_load_component = ltx2_components._load_component

    def _cpu_vocoder_load_component(
        component_cls: Any,
        model: str,
        subfolder: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if subfolder == "vocoder":
            with torch.device("cpu"):
                return orig_load_component(
                    component_cls, model, subfolder, *args, **kwargs
                )
        return orig_load_component(component_cls, model, subfolder, *args, **kwargs)

    setattr(_cpu_vocoder_load_component, _VOCODER_CPU_INIT_PATCHED, True)
    ltx2_components._load_component = _cpu_vocoder_load_component
    logger.info("Applied TPU patch: CPU-context init for LTX-2 vocoder.")


_BWE_FILTER_CPU_RESTORE_PATCHED = "_tpu_bwe_filter_cpu_restore_patched"


def _wrap_ltx2_bwe_filter_restore(ltx2_runtime: Any) -> None:
    # _run_ltx_vocoder resolves the helper as an ltx2_runtime global at call
    # time, so replacing the module attribute covers every caller.
    restore = ltx2_runtime._restore_ltx_bwe_resampler_filter
    if getattr(restore, _BWE_FILTER_CPU_RESTORE_PATCHED, False):
        return

    def _cpu_restore_ltx_bwe_resampler_filter(vocoder: Any) -> None:
        resampler: Any = getattr(vocoder, "resampler", None)
        current: Any = getattr(resampler, "filter", None)
        previous = getattr(resampler, "_ltx_restored_filter", None)
        if (
            not isinstance(current, torch.Tensor)
            or current.device.type == "cpu"
            or current is previous
        ):
            return restore(vocoder)
        # Upstream reads only the filter's device, never its values.
        resampler.filter = torch.empty_like(current, device="cpu")
        try:
            restore(vocoder)
        finally:
            rebuilt: Any = getattr(resampler, "_ltx_restored_filter", None)
            if rebuilt is previous or resampler.filter is not rebuilt:
                resampler.filter = current
            else:
                rebuilt = rebuilt.to(current.device)
                resampler.filter = rebuilt
                resampler._ltx_restored_filter = rebuilt

    setattr(
        _cpu_restore_ltx_bwe_resampler_filter, _BWE_FILTER_CPU_RESTORE_PATCHED, True
    )
    ltx2_runtime._restore_ltx_bwe_resampler_filter = (
        _cpu_restore_ltx_bwe_resampler_filter
    )
    logger.info("Applied TPU patch: CPU rebuild of the LTX-2 BWE resampler filter.")


_SPLIT_ROPE_PATCHED = "_tpu_split_rope_patched"


def _patch_ltx2_split_rope(
    od_config: "OmniDiffusionConfig | None" = None,
) -> bool | None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once vLLM-Omni's LTX-2 split RoPE rotates q and
    # k in their (B, T, H * D) layout instead of reshaping them per head.
    #
    # Problem: LTX2AudioVideoRotaryPosEmbed builds split-RoPE tables as
    #   (B, H, T, D/2), so apply_split_rotary_emb transposes every (B, T, H * D)
    #   q and k to (B, H, T, D) and back in each attention layer. The pipeline
    #   passes the same coords tensor at every denoising step, and the rope
    #   modules rebuild the tables each time.
    # Fix: the rope module returns the tables in the layout of q and k,
    #   (B, 1, T, H * D) with the sign of each half folded into sin
    #   (split_rope_natural_tables), and apply_split_rotary_emb rotates them
    #   elementwise. Each rope module keeps the tables of its last split-RoPE
    #   call and returns them while the key matches, until a call with
    #   another key replaces them.
    # Effect: LTX-2 T2V 704x1280x121, 40 steps, on TPU v7x: 0.770 -> 0.482
    #   s/step (-37%).
    # Scope:
    #   - T stays at dim 2, where the split-RoPE sequence-parallel (SP) plan
    #     shards the tables.
    #   - TP: _slice_rope_for_tp slices 4-D tables by heads on dim 1 and so
    #     passes these (B, 1, T, H * D) tables through whole; the rotation
    #     takes the lanes of the local heads.
    #   - Cache key: the storage of coords (its data pointer; Tensor.is_set_to
    #     is not implemented for TPU), the layout of coords on it (storage
    #     offset, shape, stride), its dtype, device and version counter, and
    #     the device and out_dtype arguments. Fresh views with the same layout,
    #     such as the cross-attention ropes' coords[:, 0:1, :] each step, hit
    #     in and out of inference mode. The cache holds the coords view, which
    #     keeps the storage alive, so its address is not reused while cached.
    #     Inference tensors have no version counter, so in-place edits of
    #     coords in inference mode are not seen (vLLM-Omni does not edit
    #     coords after building them).
    #   - Interleaved RoPE and (B, H, T, D/2) tables use the original
    #     functions.
    #   - synchronize_tensors materializes the cached tables once.
    # -------------------------------------------------------------------------
    if not is_model_class_in(od_config, _LTX2_PIPELINES):
        return False
    from vllm_omni.diffusion.models.ltx2 import ltx2_transformer

    from vllm_torchtpu.omni.rope import (
        apply_split_rotary_half_swap,
        split_rope_natural_tables,
    )
    from vllm_torchtpu.utils import synchronize_tensors

    mark_model_patch_applied()
    rope_cls = ltx2_transformer.LTX2AudioVideoRotaryPosEmbed
    if getattr(rope_cls.forward, _SPLIT_ROPE_PATCHED, False):
        return
    orig_rope_forward = rope_cls.forward
    orig_apply_split_rotary_emb = ltx2_transformer.apply_split_rotary_emb

    def _cached_rope_forward(
        self: Any,
        coords: torch.Tensor,
        device: str | torch.device | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope_type != "split":
            return orig_rope_forward(self, coords, device, out_dtype)
        storage = coords.untyped_storage().data_ptr()
        version = None if coords.is_inference() else coords._version
        key = (
            version,
            coords.storage_offset(),
            tuple(coords.shape),
            coords.stride(),
            coords.dtype,
            coords.device,
            device,
            out_dtype,
        )
        cached = getattr(self, "_tpu_split_rope_cache", None)
        if (
            cached is not None
            and storage != 0
            and cached[0].untyped_storage().data_ptr() == storage
            and cached[1] == key
        ):
            return cached[2]
        tables = orig_rope_forward(self, coords, device, out_dtype)
        if tables[0].ndim != 4:
            return tables
        tables = split_rope_natural_tables(*tables)
        synchronize_tensors(list(tables), wait=False)
        self._tpu_split_rope_cache = (coords, key, tables)
        return tables

    def _apply_split_rotary_emb(
        x: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
        *,
        head_dim: int,
    ) -> torch.Tensor:
        cos, sin = freqs
        if cos.ndim != 4 or cos.shape[1] != 1 or cos.shape[-1] == head_dim // 2:
            return orig_apply_split_rotary_emb(x, freqs, head_dim=head_dim)
        width = x.shape[-1]
        if cos.shape[-1] != width:
            tp_size = ltx2_transformer.get_tensor_model_parallel_world_size()
            if cos.shape[-1] != width * tp_size:
                raise ValueError(
                    f"LTX-2 split RoPE tables of width {cos.shape[-1]} do not "
                    f"match q/k of width {width} on {tp_size} tensor-parallel "
                    "ranks; the tables must cover all heads of all ranks."
                )
            start = ltx2_transformer.get_tensor_model_parallel_rank() * width
            cos = cos[..., start : start + width]
            sin = sin[..., start : start + width]
        return apply_split_rotary_half_swap(x, cos, sin, head_dim)

    setattr(_cached_rope_forward, _SPLIT_ROPE_PATCHED, True)
    rope_cls.forward = _cached_rope_forward
    ltx2_transformer.apply_split_rotary_emb = _apply_split_rotary_emb
    logger.info("Applied TPU patch: cached LTX-2 split RoPE on the q/k layout.")

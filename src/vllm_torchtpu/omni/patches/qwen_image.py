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
"""TPU runtime patches for Qwen-Image diffusion models."""

from typing import TYPE_CHECKING, Any

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.omni.patches import is_model_class_in, mark_model_patch_applied

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

# Keys of vLLM-Omni's diffusion model registry (vllm_omni/diffusion/registry.py).
_QWEN_IMAGE_PIPELINES = frozenset(
    {
        "QwenImagePipeline",
        "QwenImageEditPipeline",
        "QwenImageEditPlusPipeline",
        "QwenImageLayeredPipeline",
        "QwenImageDMD2Pipeline",
    }
)

_TXT_FREQS_CLONE_PATCHED = "_tpu_txt_freqs_clone_patched"


def _patch_qwen_image_rope_txt_freqs_clone(
    od_config: "OmniDiffusionConfig | None" = None,
) -> bool | None:
    # -------------------------------------------------------------------------
    # TODO: Remove this patch once torch_tpu supports torch.real/torch.imag
    # inside torch.inference_mode() on a complex tensor created outside it.
    #
    # Qwen-Image RoPE modules return txt_freqs as a slice of pos_freqs, which
    # is created outside inference mode. The attention blocks take
    # torch.real/torch.imag of that slice inside inference mode, which fails
    # on TPU with "RuntimeError: Cannot set version_counter for inference
    # tensor". vid_freqs comes from torch.cat inside inference mode and is not
    # affected. Cloning txt_freqs once per transformer forward gives the
    # attention blocks an inference tensor of the same values.
    # -------------------------------------------------------------------------
    if not is_model_class_in(od_config, _QWEN_IMAGE_PIPELINES):
        return False
    from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer

    mark_model_patch_applied()
    patched = False
    for rope_cls in (
        qwen_image_transformer.QwenEmbedRope,
        qwen_image_transformer.QwenEmbedLayer3DRope,
    ):
        if not getattr(rope_cls.forward, _TXT_FREQS_CLONE_PATCHED, False):
            rope_cls.forward = _clone_txt_freqs(rope_cls.forward)
            patched = True
    if patched:
        logger.info("Applied TPU patch: Qwen-Image RoPE txt_freqs clone.")


def _clone_txt_freqs(forward: Any) -> Any:
    def _forward(self: Any, *args: Any, **kwargs: Any):
        vid_freqs, txt_freqs = forward(self, *args, **kwargs)
        return vid_freqs, txt_freqs.clone()

    setattr(_forward, _TXT_FREQS_CLONE_PATCHED, True)
    return _forward

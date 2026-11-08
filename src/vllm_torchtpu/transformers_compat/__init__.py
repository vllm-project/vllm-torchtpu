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
"""Model classes newer than the image's transformers, registered with its Auto*.

vLLM resolves a checkpoint's config through transformers.AutoConfig before it
builds the model, so an architecture that vLLM serves but the installed
transformers does not know stops at startup ("Transformers does not recognize
this architecture"). Each entry here vendors the missing transformers module and
registers it only when transformers lacks the model type, so a newer image keeps
its own classes.
"""

import importlib.util


def register_transformers_compat() -> None:
    """Register vendored classes for model types the installed transformers lacks."""
    # MiniCPM-OCR: vLLM 0.33 serves MiniCPMOcrForConditionalGeneration;
    # transformers added minicpm_ocr in 5.21.
    if importlib.util.find_spec("transformers.models.minicpm_ocr") is None:
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoProcessor,
            AutoVideoProcessor,
        )

        from vllm_torchtpu.transformers_compat.minicpm_ocr import (
            configuration_minicpm_ocr as cfg,
        )
        from vllm_torchtpu.transformers_compat.minicpm_ocr import (
            image_processing_minicpm_ocr as image,
        )
        from vllm_torchtpu.transformers_compat.minicpm_ocr import (
            processing_minicpm_ocr as proc,
        )
        from vllm_torchtpu.transformers_compat.minicpm_ocr import (
            video_processing_minicpm_ocr as video,
        )

        AutoConfig.register("minicpm_ocr", cfg.MiniCPMOcrConfig, exist_ok=True)
        AutoConfig.register(
            "minicpm_ocr_vision", cfg.MiniCPMOcrVisionConfig, exist_ok=True
        )
        AutoImageProcessor.register(
            cfg.MiniCPMOcrConfig,
            fast_image_processor_class=image.MiniCPMOcrImageProcessor,
            exist_ok=True,
        )
        AutoVideoProcessor.register(
            cfg.MiniCPMOcrConfig, video.MiniCPMOcrVideoProcessor, exist_ok=True
        )
        AutoProcessor.register(
            cfg.MiniCPMOcrConfig, proc.MiniCPMOcrProcessor, exist_ok=True
        )

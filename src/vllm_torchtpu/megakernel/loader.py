# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``megakernel`` load format: the model without weights.

With the megakernel backend selected (``megakernel.backend``) the platform
sets this load format. vLLM still builds the model, whose attention layers
declare the KV cache spec, but on the meta device: no parameter takes memory,
nothing is loaded into it and no weight processing runs. The program streams
its own pre-shards (``MegakernelModelRunner.load_model``).
"""

import torch
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_torchtpu.megakernel.backend import LOAD_FORMAT


@register_model_loader(LOAD_FORMAT)
class MegakernelModelLoader(BaseModelLoader):
    """See the module docstring."""

    def download_model(self, model_config) -> None:
        pass  # the program reads its pre-shard directory

    def load_weights(self, model, model_config) -> None:
        pass  # the program loads its own weights

    def load_model(self, vllm_config, model_config, prefix: str = ""):
        with set_default_torch_dtype(model_config.dtype), torch.device("meta"):
            model = initialize_model(
                vllm_config=vllm_config, model_config=model_config, prefix=prefix
            )
        return model.eval()

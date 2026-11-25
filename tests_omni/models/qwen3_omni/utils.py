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
"""Shared test helpers for Qwen3-Omni Talker and Code Predictor tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
)
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import LinearBase
from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorBaseModel,
    CodePredictorWrapper,
    _RMSNorm,
    _RotaryEmbedding,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)

TINY_CP_CONFIG = Qwen3OmniMoeTalkerCodePredictorConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=64,
    num_code_groups=4,
)


def init_linear_weights(module: nn.Module, seed: int = 0) -> None:
    """Seed vLLM ``LinearBase`` parameters deterministically."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for submodule in module.modules():
            if isinstance(submodule, LinearBase):
                for param in submodule.parameters(recurse=False):
                    if param.dim() > 1:
                        param.copy_(
                            torch.randn(param.shape, generator=generator) * 0.02
                        )
                    else:
                        param.zero_()


def bind_native_custom_ops_on_cpu(module: nn.Module) -> None:
    """Bind ``forward_native`` on vLLM ``CustomOp`` instances for CPU tests."""
    for submodule in module.modules():
        if isinstance(submodule, CustomOp):
            submodule._forward_method = submodule.forward_native


@contextmanager
def isolate_talker_patch_targets() -> Iterator[None]:
    """Snapshot and restore Qwen3-Omni Talker/Code Predictor patch targets."""
    targets = (
        (_RMSNorm, "forward_tpu"),
        (_RotaryEmbedding, "forward_tpu"),
        (CodePredictorBaseModel, "forward"),
        (CodePredictorWrapper, "forward"),
        (Qwen3OmniMoeForConditionalGeneration, "talker_preprocess"),
    )
    saved = [(cls, attr, cls.__dict__.get(attr)) for cls, attr in targets]
    try:
        yield
    finally:
        for cls, attr, val in saved:
            if val is None:
                if attr in cls.__dict__:
                    delattr(cls, attr)
            else:
                setattr(cls, attr, val)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 models for the TorchTPU vLLM plugin."""

from .attention import KimiDeltaAttention, MultiHeadLatentAttention
from .dspark import K3DSparkForCausalLM, K3DSparkModel
from .model import (KimiK3ForConditionalGeneration, KimiLinearForCausalLM,
                    KimiModel)
from .moe import KimiMoE

__all__ = [
    "KimiDeltaAttention",
    "K3DSparkForCausalLM",
    "K3DSparkModel",
    "KimiK3ForConditionalGeneration",
    "KimiLinearForCausalLM",
    "KimiModel",
    "KimiMoE",
    "MultiHeadLatentAttention",
]

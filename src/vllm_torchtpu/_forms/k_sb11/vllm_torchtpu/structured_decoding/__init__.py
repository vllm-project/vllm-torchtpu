# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured decoding subsystem for vLLM on Torch TPU."""

from vllm_torchtpu.structured_decoding.structured_decoding_manager import (
    StructuredDecodingManager,
)

__all__ = ["StructuredDecodingManager"]

# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Decode MoE layer of one TP rank, shared expert and TP all-reduce included,
in one Pallas kernel (kernel.py; the all-reduce and its self-test in
all_reduce.py)."""

from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.kernels.experimental.moe_tp_decode.kernel import (
    ROW_BUCKETS,
    can_use_moe_tp_decode,
    exact_in_bf16,
    kernel_name,
    moe_tp_decode,
)

__all__ = [
    "ROW_BUCKETS",
    "all_reduce",
    "can_use_moe_tp_decode",
    "exact_in_bf16",
    "moe_tp_decode",
    "kernel_name",
]

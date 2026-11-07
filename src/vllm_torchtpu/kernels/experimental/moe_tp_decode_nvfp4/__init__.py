# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 W4A8 variant of the fused TP-MoE decode kernel.

Reuses the in-kernel all-reduce and router/shared-expert building blocks from
``moe_tp_decode`` while streaming whole K-major ``float4_e2m1fn`` experts into
grouped VMEM ring slots (``fp4 -> fp8`` widen in VMEM, grouped gate/up dots,
output-side ``f32`` block scales).
"""

from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.kernels.experimental.moe_tp_decode_nvfp4.kernel import (
    ROW_BUCKETS,
    can_use_moe_tp_decode_nvfp4,
    exact_in_bf16,
    group_size,
    kernel_name,
    moe_tp_decode_nvfp4,
)

__all__ = [
    "ROW_BUCKETS",
    "all_reduce",
    "can_use_moe_tp_decode_nvfp4",
    "exact_in_bf16",
    "group_size",
    "kernel_name",
    "moe_tp_decode_nvfp4",
]

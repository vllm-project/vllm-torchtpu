# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 fused MoE prefill kernel (W4A16 default and W4A8 secondary arm).

Streams K-major ``float4_e2m1fn`` routed experts through a whole-expert VMEM
ring (``EP2 x TP4`` primary via ``expert_start``, ``TP8`` secondary), fuses the
TP8-sharded online-FP8 shared expert as one trailing visit, and combines +
all-reduces across ranks via chunked ``reduce-scatter + all-gather``.
"""

from vllm_torchtpu.kernels.experimental.fused_moe_prefill_nvfp4.kernel import (
    STAGE_A_NVFP4,
    ActMode,
    TallRungRows,
    build_expert_kernel_nvfp4,
    can_use_fused_moe_prefill_nvfp4,
    fused_moe_prefill_nvfp4,
    fused_moe_prefill_nvfp4_stage_a,
    tall_rung_rows_for,
)

__all__ = [
    "STAGE_A_NVFP4",
    "ActMode",
    "TallRungRows",
    "build_expert_kernel_nvfp4",
    "can_use_fused_moe_prefill_nvfp4",
    "fused_moe_prefill_nvfp4",
    "fused_moe_prefill_nvfp4_stage_a",
    "tall_rung_rows_for",
]

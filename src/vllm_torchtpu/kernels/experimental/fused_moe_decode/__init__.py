# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE decode kernel (routed + shared experts + all-reduce in one call)."""

from .fused_moe_decode_kernel import fused_moe_decode, plan_weight_ring, vmem_bytes

__all__ = ["fused_moe_decode", "plan_weight_ring", "vmem_bytes"]

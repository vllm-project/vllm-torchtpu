# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE prefill kernel (routed + shared experts, combine and all-reduce)."""

from .fused_moe_prefill_kernel import fused_moe_prefill, plan_blocks

__all__ = ["fused_moe_prefill", "plan_blocks"]

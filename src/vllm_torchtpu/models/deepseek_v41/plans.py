# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default DeepSeek-V4.1 plan: slot -> ordered SUPPORTED variant names.

``vllm serve`` with no ``model_plan`` gets the first eligible entry of each
list. Support owns this file; ``validation/plan_snapshots/`` records what it
selects for each standard customer config, and CPU CI fails on any change.
"""

SUPPORTED_PLAN: dict[str, list[str]] = {
    "csa2_attention": ["tuned"],
    "indexer": ["tuned"],
    "mhc": ["tuned"],
    "engram": ["tuned"],
    "ced": ["tuned"],
    "dspark_draft": ["tuned"],
    "decode_step": ["tuned"],
    "host_gc": ["tuned"],
    "moe_kernel": ["fused_ep_v2"],
    "expert_layout": ["checkpoint"],
}

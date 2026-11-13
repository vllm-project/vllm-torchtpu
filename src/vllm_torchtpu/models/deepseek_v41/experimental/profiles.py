# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named DeepSeek-V4.1 benchmark recipes: slot overrides on top of plans.py.

Select one with
``--additional-config '{"model_plan": {"profile": "<name>"}}'``.
Active profiles pin every slot, so an edit to ``../plans.py`` never shifts
what a benchmark runs; nightly E2E keeps each active profile green.
"""

PROFILES: dict[str, dict[str, str]] = {
    # DP16/EP16 C768 decode on two v7x hosts (48 requests per rank, 288 target
    # rows, DSpark K5), the hillclimb's served configuration: 28.24 ms official
    # median step at aa4360d22 (j-f164e4eb). Needs
    # additional_config.deepseek_v41.moe=fused_ep_w4a8.
    "dsv41-dp16-c768": {
        "csa2_attention": "tuned",
        "indexer": "tuned",
        "mhc": "tuned",
        "engram": "tuned",
        "ced": "tuned",
        "dspark_draft": "tuned",
        "decode_step": "tuned",
        "host_gc": "tuned",
        "moe_kernel": "push_fused_ep",
        "expert_layout": "dp16_c768",
    },
}

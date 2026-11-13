# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXPERIMENTAL implementations of DeepSeek-V4.1 slots.

Experimental owns this catalog. Entries are reachable only through a profile
in ``profiles.py`` and never from the default plan. Graduating one moves it
into ``../variants.py`` and flips ``../plans.py`` (one entry, one snapshot
field), with the evidence the graduation gate asks for.
"""

from vllm_torchtpu.model_plan import PlanContext, Variant


def _fused_ep_with_ep(context: PlanContext) -> bool:
    return context.ep_size > 1 and context.moe == "fused_ep_w4a8"


EXPERIMENTAL_VARIANTS: dict[str, dict[str, Variant]] = {
    "moe_kernel": {
        # Push-dispatch fused EP MoE (kernels/experimental/push_fused_moe),
        # tuned at DP16/EP16 C768 decode (#2839, #2876, LAND9). Each key is a
        # keyword of `fused_ep_moe_push`; values are the ones the 28.24 ms
        # landing (aa4360d22, j-f164e4eb) served.
        "push_fused_ep": Variant(
            params={
                "impl": "push",
                "sc_prepush": 2,
                "tc_exchange": True,
                "tail_barrier": False,
                "onehot_slots": True,
                "hbm_staging": True,
                "dispatch_ahead": 25,
                "matmul_scale_repeat": True,
                "exchange_row_gather": True,
                "bf16_sort_counts": True,
                "slim_prepush_barrier": True,
                "fused_dispatch_meta": True,
                "table_kernel": True,
                "exchange_tables": True,
                "kernel_plan_tables": True,
                "light_prepush": 2,
                "single_barrier": True,
                "early_scale_scatter": 2,
                # FP8 shared expert computed inside the push op (1-ulp tier).
                "shared_expert": True,
            },
            eligible=_fused_ep_with_ep,
            description="kernel 273 -> 225 us per call at the g25 shape",
        ),
    },
}

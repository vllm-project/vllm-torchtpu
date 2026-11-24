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


# Push-dispatch fused EP MoE (kernels/experimental/push_fused_moe), tuned at
# DP16/EP16 C768 decode (#2839, #2876, LAND9). Each key is a keyword of
# `fused_ep_moe_push`; values are the ones the 28.24 ms landing (aa4360d22,
# j-f164e4eb) served.
_PUSH_LANDING = {
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
}

EXPERIMENTAL_VARIANTS: dict[str, dict[str, Variant]] = {
    "csa2_attention": {
        # `tuned` with the two-slot SparseCore gather. The three-slot gather
        # reads one zero row for every negative id, and those reads serialize:
        # 65,536 padding ids take 2.7 ms against 0.06 ms here, and a DP rank
        # padded to the 2048 bucket gathers 1M of them per KV-source layer.
        "tuned_two_slot_gather": Variant(
            params={
                "deep_gather": False,
                "blocked_rope_gather": True,
                "qb_rope_exact_rows": True,
            },
            description="padded 2048-bucket gather 42.8 -> 1.0 ms per source layer",
        ),
    },
    "moe_kernel": {
        "push_fused_ep": Variant(
            params=_PUSH_LANDING,
            eligible=_fused_ep_with_ep,
            description="kernel 273 -> 225 us per call at the g25 shape",
        ),
        # The landing settings, and a shard's trailing repeats of its last
        # row (the bucket's padding) route once instead of 6 x rows times.
        "push_fused_ep_repeated_tail": Variant(
            params={**_PUSH_LANDING, "drop_repeated_tail": True},
            eligible=_fused_ep_with_ep,
            description="padding rows of a DP-max bucket skip the experts",
        ),
        # Every run of repeated rows routes once, not only the trailing one:
        # from the first Engram layer on, a padded bucket's rows stop matching
        # the last row but still come in runs.
        "push_fused_ep_repeated_runs": Variant(
            params={**_PUSH_LANDING, "drop_repeated_runs": True},
            eligible=_fused_ep_with_ep,
            description="runs of repeated padding rows skip the experts",
        ),
        # The repeated runs as above, with the combine kernel zeroing unrouted
        # rows and copying each run's first result onto its repeats: no XLA
        # select or gather over the [rows, hidden] output. Bit-exact.
        "push_fused_ep_fused_runs": Variant(
            params={**_PUSH_LANDING, "drop_repeated_runs": True, "fused_runs": True},
            eligible=_fused_ep_with_ep,
            description="repeat copy and unrouted zeroing in the combine kernel",
        ),
    },
}

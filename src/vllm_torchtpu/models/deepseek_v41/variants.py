# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SUPPORTED implementations of each DeepSeek-V4.1 dispatch slot.

Support owns this catalog: every entry here may be selected for customers by
``plans.py``. Experimental implementations register in
``experimental/variants.py`` instead and are reachable only through a named
profile.

Most slots carry two variants. ``base`` is the code path as it was before the
DP16 decode hillclimb (every switch of the slot off); ``tuned`` is the path the
hillclimb measured and landed (#2838, #2839, #2868, #2875, #2876; evidence in
those PR bodies). Both are served code, so rolling a slot back is a one-line
change in ``plans.py``. Every variant of a slot carries the same parameter
keys; the consumers read them through :mod:`.plan`.
"""

from vllm_torchtpu.model_plan import PlanContext, Variant


def _expert_parallel(context: PlanContext) -> bool:
    return context.ep_size > 1


SUPPORTED_VARIANTS: dict[str, dict[str, Variant]] = {
    # CSA2 attention: SparseCore word/scale gather, RoPE row gather, Q-B/RoPE.
    "csa2_attention": {
        "base": Variant(
            params={
                "deep_gather": False,
                "blocked_rope_gather": False,
                "qb_rope_exact_rows": False,
            },
        ),
        "tuned": Variant(
            params={
                # Three staging slots in flight, invalid rows DMA-zeroed.
                "deep_gather": True,
                # Steps of 257 to 1024 rows gather RoPE rows as 256-row blocks.
                "blocked_rope_gather": True,
                # The fused Q-B/RoPE kernel writes only the real rows.
                "qb_rope_exact_rows": True,
            },
            description="#2868: bit-exact; RoPE/Q-B gathers -125/-39 us per step",
        ),
    },
    # Packed FP8 dense indexer scorer.
    "indexer": {
        # None keeps the kernel config's own issue depth.
        "base": Variant(params={"issue_depth": None}),
        "tuned": Variant(
            params={"issue_depth": 8},
            description="#2875: scheduling only, bit-exact; dense scorer -64 us",
        ),
    },
    # Manifold hyper-connections (mHC) around each layer.
    "mhc": {
        "base": Variant(
            params={"resident_gates": False, "early_prep": False, "shifted_t16": False}
        ),
        "tuned": Variant(
            params={"resident_gates": True, "early_prep": True, "shifted_t16": True},
            description="#2876: resident gates -126 us per step",
        ),
    },
    # Engram hashing and slot writes.
    "engram": {
        "base": Variant(
            params={
                "fused_hash": False,
                "masks_in_hash": False,
                "int32_slot_write": False,
                "tile_scatter": False,
                "fused_prep": False,
            }
        ),
        "tuned": Variant(
            params={
                "fused_hash": True,
                "masks_in_hash": True,
                "int32_slot_write": True,
                "tile_scatter": True,
                # Slot write and hash in one op call (needs int32_slot_write).
                "fused_prep": True,
            },
            description="#2876 + LAND9: same ids and cache contents",
        ),
    },
    # Compressed encoder-decoder (CED) execution.
    "ced": {
        "base": Variant(params={"identity_replay": False}),
        "tuned": Variant(
            params={"identity_replay": True},
            description="#2876: steady decode passes encoder outputs through",
        ),
    },
    # DSpark draft model.
    "dspark_draft": {
        "base": Variant(params={"pallas_logits": False}),
        "tuned": Variant(
            params={"pallas_logits": True},
            description="#2876: draft head 546 -> 433 us; drafter rounding only",
        ),
    },
    # Decode-step glue shared with the runner and the compiler.
    "decode_step": {
        "base": Variant(
            params={
                "onehot_table_lookup": False,
                "reuse_window_starts": False,
                "block_table_delta_buckets": False,
                "fused_spec_next_inputs": False,
                "inline_deduplicated_calls": False,
            }
        ),
        "tuned": Variant(
            params={
                # Per-request tables looked up by one-hot sum (integer-exact).
                "onehot_table_lookup": True,
                "reuse_window_starts": True,
                # Block-table delta uploads padded to the first fitting bucket.
                "block_table_delta_buckets": True,
                "fused_spec_next_inputs": True,
                # --xla_tpu_enable_deduplicated_calls=false (code layout only).
                "inline_deduplicated_calls": True,
            },
            description="#2876: one-hot -0.53 ms, no deduplicated calls -0.90 ms",
        ),
    },
    # CPython's cyclic collector in the DP decode step.
    "host_gc": {
        "base": Variant(params={"policy": "default", "collect_every": 0}),
        "tuned": Variant(
            # Heap frozen after warmup, automatic collection off, one explicit
            # collection every 500 DP-coordinated steps.
            params={"policy": "freeze", "collect_every": 500},
            description="#2876: stalled steps 9% -> 0.4%",
        ),
    },
    # Fused EP MoE dispatch (deepseek_v41.moe=fused_ep_w4a8 only).
    "moe_kernel": {
        "fused_ep_v2": Variant(params={"impl": None}),
    },
    # Static expert layout (#2838): where each expert's copies live. Bit-exact;
    # needs moe=fused_ep_w4a8, linear placement, no EPLB.
    "expert_layout": {
        "checkpoint": Variant(params={"layout": None}),
        "dp16_c768": Variant(
            params={"layout": "dsv41_flash_dp16_c768"}, eligible=_expert_parallel
        ),
        "dp32_c768": Variant(
            params={"layout": "dsv41_flash_dp32_c768"}, eligible=_expert_parallel
        ),
        "dp8_c384_r1": Variant(
            params={"layout": "dsv41_flash_dp8_c384_r1"}, eligible=_expert_parallel
        ),
        "dp8_c384_perm": Variant(
            params={"layout": "dsv41_flash_dp8_c384_perm"}, eligible=_expert_parallel
        ),
    },
}

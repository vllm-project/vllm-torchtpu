# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`--additional-config` keys of the opt-in r6 kernel and runtime variants.

Every key defaults to the behaviour without it. vLLM hashes
`additional_config` into the compile-cache key (`VllmConfig.compute_hash`),
so programs built with a key on and off never share a cache entry.
"""

from typing import Any

DEFAULTS: dict[str, bool | int | str] = {
    # MiniMax M3: steps of at least this many tokens run every layer on the
    # step's two token halves, so one half's residual all-reduces overlap the
    # other half's attention and FFN. 0 = off.
    "minimax_split_min_tokens": 0,
    # With the split on, the second half's layer input waits for the first
    # half's attention (so it overlaps the previous all-reduce).
    "minimax_split_attention_first": True,
    # MiniMax M3, 4-bit DCP caches over several KV heads per rank group:
    # steps of >= 2048 tokens attend their prefill requests head-parallel
    # (each rank fetches its KV head's pages and attends its own query heads,
    # no query gather or partial merge; `msa_dcp_heads`).
    "minimax_dcp_heads_attend": False,
    # MiniMax M3 MSA prefill attention: build the tiles' page unions with
    # unstable sorts (no iota tie-breaker; bit-identical unions).
    "minimax_msa_unstable_sort": False,
    # MiniMax M3 DCP prefill candidates: scan widths in steps of 32/128
    # columns instead of powers of two, picks stored per pick and looped past
    # 512 columns (same picks).
    "minimax_dcp_fine_scan_widths": False,
    # MiniMax M3 MSA glue: per-request table lookups (token positions, tile
    # tables, class columns) by a one-hot over the requests instead of a
    # gather, which XLA lowers to a compare chain.
    "minimax_msa_onehot_lookup": False,
    # MiniMax M3 indexer kernel v3: a 3-group DMA ring (2 groups ahead),
    # alternate pages on the second DMA queue, one wait per set bit of the
    # group's page count. Same scores.
    "minimax_indexer_dma_ring3": False,
    # MiniMax M3 indexer kernel v3: fp8 index caches score small tiles
    # (<= 64 rows) transposed with fp8 MXU keys, as the folded 4-bit caches do.
    "minimax_indexer_fp8_transposed": False,
    # MiniMax M3 MSA attention (DCP lse kernel): r6's kernel instead of v3's
    # (`msa_attention_lse_r6`: 4 tiles per grid step, a 3-group 4-bit page
    # prefetch ring, transposed scores for 8/16-bit full tiles; no 12-page
    # width, no 4-bit page cache). Not bit-identical on 8/16-bit caches.
    "minimax_msa_lse_r6": False,
    # MiniMax M3 DSpark drafter context K/V: fc's replicated weight split by
    # columns over the TP ranks (+ all-gather), and only the K/V columns of
    # each draft qkv_proj, when the weights are plain flipped matrices.
    "minimax_dspark_split_fc": False,
    # Raiden KV offload: local loads in host-block order (and the load's fresh
    # device blocks permuted to match), stores in device-block order, so
    # consecutive runs move as one copy per KV array. The block pool's
    # consecutive-run allocation (`contiguous_alloc`) is the connector's
    # kv_connector_extra_config key contiguous_alloc_scan_factor.
    "raiden_host_order_dma": False,
    # gmm_v2: decode-sized MoE buffers take r6's decode tiles (tile_m 128 in
    # buckets 16/32/48/128, or fp8-sized tiles up to 512 rows) ahead of the
    # skinny paths (fp8_feed_wide etc.), and kernel_mlp fuses them.
    "moe_gmm_fp8_decode_tiles": False,
    # Under synthetic rejection sampling, decide acceptance before the LM head
    # and compute logits and sample on one row per request (its first
    # rejected slot or its bonus slot) instead of every verify row.
    "spec_synthetic_one_row_verify": False,
    # Spec-decode warm-up: run the greedy mixed nr=R step at every token
    # bucket below the full budget too, so the (token bucket, logits rows)
    # programs serving's partial prefill chunks reach are compiled before
    # serving (longer warm-up).
    "spec_warmup_sweep_token_buckets": False,
    # MiniMax M3 indexer: pages per grid step of the large (prefill) tile
    # class, a positive multiple of 32 (0 = the default 32). Wider steps
    # amortize the per-step cost over more pages; same scores.
    "minimax_indexer_large_pages_per_step": 0,
}


def option(additional_config: Any, key: str) -> bool | int | str:
    """`additional_config[key]`, or the key's default when unset."""
    default = DEFAULTS[key]
    if not isinstance(additional_config, dict) or key not in additional_config:
        return default
    value = additional_config[key]
    # bool is a subclass of int: require the default's exact type.
    if type(value) is not type(default):
        raise ValueError(
            f"additional_config.{key} must be a {type(default).__name__}, got {value!r}"
        )
    return value


def check_split_disabled(additional_config: Any) -> None:
    """Refuse the r6 token-half split keys.

    The split step (steps of >= minimax_split_min_tokens tokens) produces wrong
    tokens on LongBench-v2 and leaks text across requests (real-weight check
    j-5ec6afaf-fed7-4cd6-a5ef). It stays disabled until that is fixed.
    """
    split_on = option(additional_config, "minimax_split_min_tokens") > 0
    attention_first_set = (
        isinstance(additional_config, dict)
        and "minimax_split_attention_first" in additional_config
    )
    if split_on or attention_first_set:
        raise ValueError(
            "additional_config.minimax_split_min_tokens / "
            "minimax_split_attention_first: the r6 token-half split is disabled. "
            "Its split steps leak data across requests and fail LongBench-v2 "
            "(j-5ec6afaf-fed7-4cd6-a5ef); remove these keys."
        )

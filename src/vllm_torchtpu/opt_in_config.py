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

DEFAULTS: dict[str, bool | int | float | str] = {
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
    # MiniMax M3 indexer, steps that score the large (prefill) tile class
    # under a cond (decode-sized buckets): that class's tiles, work list and
    # columns and each row's gather column outside the cond, so XLA computes
    # them once per step instead of once per layer. Same scores.
    "minimax_indexer_hoist_metadata": False,
    # MiniMax M3 indexer, token-major small scores: rows from a row gather of
    # the small class and a column gather of the large class, selected per
    # row, instead of one column gather over both (no transposed copy of the
    # small class). Same scores.
    "minimax_indexer_row_gather": False,
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
    # Raiden KV offload: refresh the host LRU for prefixes that stay hot in
    # HBM. prepare_store moves keys that are already host-resident to the
    # Most Recently Used end instead of skipping them untouched, and a
    # request finishing under the write-back store policy touches its resident
    # prefix. Changes only which host entries are evicted, never KV bytes.
    "raiden_offload_refresh_lru": False,
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
    # gmm_v2 tiles of fp4-rhs calls above FP8_DECODE_MAX_M rows:
    # "<tile_m>:<bucket>/.../<tile_m>[@<max size_m>]" ("" = off;
    # `gmm_v2.set_tile_overrides`). Same numerics per bucket.
    "gmm_v2_prefill_tiles": "",
    # gmm_v2 decode-sized gate_up / down calls:
    # "<tile_n>:<gate_up buffers>[:<down buffers>]" ("" = off). Same bits.
    "gmm_v2_decode_w13": "",
    # MiniMax M3 MSA attention, r6's lse kernel only (minimax_msa_lse_r6;
    # `msa_attention_lse_r6`), prefill-sized calls: a 12-page step width;
    # step widths outright ("<w>/<w>/...", overrides width12); a per-call
    # VMEM page cache of this many slots for 4-bit caches (0 = off); the
    # tile-union pages it covers (0 = 32); the cache unpacked to bf16 once;
    # each group in n parts of the query heads (0 or 1 = off). All bitwise.
    "minimax_msa_prefill_width12": False,
    "minimax_msa_prefill_widths": "",
    "minimax_msa_prefill_page_cache": 0,
    "minimax_msa_prefill_page_cache_tile_pages": 0,
    "minimax_msa_prefill_page_cache_bf16": False,
    "minimax_msa_prefill_row_split": 0,
    # Same kernel, prefill tile tokens / group pages (0 = 16 each). Not
    # bitwise (another summation order / group split).
    "minimax_msa_prefill_block_q": 0,
    "minimax_msa_prefill_pages_per_step": 0,
    # Same kernel, decode batches: the work list by one Pallas kernel; every
    # KV head in one kernel call. Bitwise.
    "minimax_msa_decode_work_list_kernel": False,
    "minimax_msa_decode_one_call": False,
    # EP MoE steps of at least this many tokens: combine on the TensorCore
    # (`moe_tc_combine.tc_slab_combine`, bitwise, 0 = off); flat routing
    # arrays read once (same values, 0 = off). And the v3 combine's route
    # metadata by one multi-operand sort (same values;
    # `fused_moe_gmm.set_row_movement`).
    "moe_tc_combine_min_tokens": 0,
    "moe_v3_payload_sort": False,
    "moe_flat_routing_min_tokens": 0,
    # NVFP4 GMM MoE: padded bucket rows get zero routing weights, so
    # TPU_MOE_SKIP_PADDED_TOKENS (recipe default 1) drops them from the
    # grouped matmul; and the skip applies only to batches of at least this
    # many routes (T * top_k; 0 = every size). With 2049 the one-hot-permute
    # batches keep their layout, so real rows are bitwise as with the skip off
    # (`fused_moe_gmm.set_skip_padded`).
    "moe_nvfp4_skip_padded": False,
    "moe_skip_padded_min_routes": 0,
    # NVFP4 GMM EP MoE: this many redundant expert slots per EP rank and MoE
    # layer, each a checkpoint copy of another rank's expert; decode-sized
    # steps (<= 2048 routes) assign every touched expert whole to one copy
    # (`layers.adapter.moe_redundant`). Placement: a JSON file path, inline
    # JSON, or "builtin:<name>" ("" = round-robin fill). Every rank reads all
    # experts at load; +HBM per slot. Not bitwise (the per-rank grouping of
    # each token's expert sum changes before the EP all-reduce).
    "moe_redundant_experts": 0,
    "moe_redundant_placement": "",
    # MiniMax kernels (MSA attention / select / candidates, gmm_v2, DSpark
    # draft attention): compile each pallas_call once per shape at smaller
    # VMEM limits and keep the first that compiles (`kernels.tight_vmem`);
    # only the limit changes. Candidates "<MiB>,..." ("" = "32,40,48,52").
    "tight_vmem": False,
    "tight_vmem_candidates_mib": "",
    # Worker: drop JAX's trace / lowering caches once warm-up has compiled
    # every bucket (`jax.clear_caches`). The Pallas kernels already sit in the
    # compiled programs; the cached jaxprs only serve a re-trace. Frees
    # ~0.9 GiB host heap per worker (~7 GiB per MiniMax-M3.1 host), which
    # lowers the Raiden host-pool start-up guard reading. Host-only.
    "release_jax_caches": False,
    # --- v7 MoE / grouped GEMM ---
    # gmm_v2, prefill-sized calls (> 2048 rows) with fp4 experts: buckets
    # above 48 rows that would dequantize the weights before the matmul run
    # fp8 x fp8 instead (rhs tile requantized to e4m3 per column, lhs rows to
    # e4m3 per row; `gmm_v2.uses_fp8_w8a8`). Tile choice unchanged. LOSSY.
    "gmm_fp8_w8a8": False,
    # gmm_v2 decode tiles (tile_m <= 128): rhs scale blocks 3-deep with
    # lookahead instead of double-buffered (`gmm_v2._scale_buffering`).
    # Bitwise (pipelining only; the VMEM estimate counts the extra buffer).
    "gmm_scale_lookahead": False,
    # EP one-hot MoE (<= the one-hot threshold routes): permute and combine as
    # Pallas kernels over only the 128-slot blocks of the valid local prefix
    # instead of dense one-hot matmuls over every slot, where the shapes allow
    # (`kernels/megablox/moe_onehot_bounded.py`). The permute is exact; the
    # combine sums the same nonzero products (f32 accumulation), possibly in
    # another MXU order: bitwise in the CPU tests, not guaranteed on TPU.
    "moe_bounded_onehot": False,
    # EP MoE combine (ragged path, tokens a multiple of 256) on the TensorCore
    # with r10's kernel (`kernels/megablox/tc_combine.py`: staged 16-row tiles
    # times a routing-weight matrix on the MXU) instead of the SparseCore
    # gather-reduce. Not bitwise (MXU f32 sum order; bf16 weights multiply in
    # the MXU). Excludes moe_tc_combine_min_tokens (a different, bitwise
    # TensorCore combine): setting both fails at init.
    "moe_tc_combine": False,
    # MoE steps of 1..this many tokens with NVFP4 experts (EP or TP-MoE) run
    # the permutation-free dense-expert fp4 kernel instead of sort + grouped
    # GEMM + combine (`kernels/experimental/moe_dense_expert_fp4.py`); takes
    # precedence over gmm_v2_decode_stream for those steps. Same roundings as
    # the GMM path, other f32 sum order: not bitwise. 0 = off.
    "moe_dense_fp4_max_tokens": 0,
    # NVFP4 EP MoE layers hand their shared expert (e4m3 per-channel MiniMax
    # linears) to the routed op: the dense fp4 kernel evaluates it in the
    # same call, other steps add it with XLA inside the op
    # (`layers/adapter/dense_fp4_shared.py`). Fails at init when no layer is
    # eligible (bf16 linears). LOSSY vs the separate shared-expert linears
    # (another evaluation order/rounding of the fp8 shared MLP).
    "moe_dense_fp4_shared": False,
    # NVFP4 decode MoE (cg line): gmm_v2_mlp calls of at most <max size_m>
    # rows run as one streamed kernel that feeds every hit expert's gate_up
    # then down weights as <chunk_rows>-row k-slices through <slots> VMEM
    # slots (`kernels/megablox/gmm_decode_stream.py`).
    # "<chunk_rows>:<slots>[@<max size_m>]", "on" = 768:4@512, "" = off.
    # Bitwise vs gmm_v2's fp8 rhs feed up to 512 rows (served c4/c16
    # "1536:3@512"); above 512 rows it follows the r6 fp8 decode tiles
    # (bitwise only with moe_gmm_fp8_decode_tiles; else <= 2 bf16 ulp,
    # served c48/c64 "1536:3@2048"). Refused with moe_skinny_gmm=bf16_after.
    "gmm_v2_decode_stream": "",
    # --- v7 minimax attention ---
    # Same kernel, 4-bit prefill calls with a page cache in
    # `msa_dcp.msa_attend_dcp`: "<threshold>:<block_q>:<tile_pages>" runs a
    # call whose selections hold more than <threshold> distinct pages per
    # query tile on average with <block_q>-token tiles and <tile_pages>
    # cached union pages, other calls with the configured tiles (both kernels
    # in one lax.cond). Each call is bitwise equal to its chosen config.
    "minimax_msa_prefill_adaptive": "",
    # Same kernel, the page cache table's sorts: "" (stable), "unstable",
    # "packed" (one int32 operand per sort) or "packed_search"; and
    # narrower table widths "<w>/<w>/..." below the tile pages (each layer's
    # calls sort the narrowest that holds their widest union). Same table.
    "minimax_msa_page_cache_sort": "",
    "minimax_msa_page_cache_widths": "",
    # Same kernel, head-parallel DCP prefill attention (`msa_dcp_heads`, with
    # minimax_dcp_heads_attend): "<block_q>:<pages per step>:<widths
    # w/w/...>:<page cache slots>", empty fields = the defaults (e.g.
    # "16:16::1024"). Bitwise when only widths / page cache change.
    "minimax_dcp_hp_prefill_config": "",
    # MiniMax M3 DCP steps of at most this many tokens (0 = off;
    # `models/minimax_m3/dcp_pack.py`): merge the attention partials with one
    # packed all-to-all instead of an LSE all-gather plus a reduce-scatter
    # (bitwise at DCP 2, another summation order at wider DCP); over one KV
    # head, gather the query heads and the top-k candidates in one all-gather
    # after the scoring op instead of two (bitwise).
    "minimax_dcp_merge_a2a_max_tokens": 0,
    "minimax_dcp_qcand_gather_max_tokens": 0,
    # MiniMax M3 sequence parallelism (`models/minimax_m3/model.py`) on
    # steps of at least minimax_sp_min_tokens tokens (0 = 4096) that are a
    # multiple of TP: "attn" reduce-scatters the attention output with the
    # residual, normalizes the token shard, all-gathers it as the FFN input
    # and folds the attention partial into the FFN all-reduce; "full" keeps
    # the residual stream a token shard through every layer (all-gather
    # before attention and FFN, reduce-scatter after). "" / "off" = two
    # all-reduces per layer. Not bitwise (another reduction order).
    "minimax_sp_mode": "",
    "minimax_sp_min_tokens": 0,
    # With minimax_sp_mode: the FFN-input / ("full") attention-input
    # all-gathers send e4m3 rows with a per-row f32 scale. Lossy.
    "minimax_sp_fp8_gather": False,
    "minimax_sp_fp8_attn_gather": False,
    # MiniMax M3 bias-free MXFP8 linears of the target model / of the DSpark
    # draft's MLP stay e4m3 with one power-of-two scale per output channel
    # (`models/minimax_m3/quant.py`) instead of bf16: half the weight bytes.
    # Lossy only where a block's shift onto its channel scale takes an element
    # below e4m3's normal range; otherwise the matmul equals the bf16 one.
    "minimax_fp8_linears": False,
    "minimax_fp8_draft_linears": False,
    # MiniMax M3 attention: the q / k / index_q / index_k Gemma RMSNorms and
    # the partial NeoX rope in one Pallas pass (`kernels/qk_norm_rope.py`)
    # instead of XLA fusions. Same math (f32 norm rounded to bf16, f32
    # rotation rounded once); not checked bitwise on TPU.
    "minimax_qk_norm_rope_kernel": False,
    # MiniMax M3 MSA DCP attention tiles (`kernels/msa_dcp.attend_tiles`):
    # 8-token tiles up to this many tokens (0 = 128); groups of this many
    # pages for those calls above 128 tokens (0 = 16); larger calls of the
    # default lse kernel (not r6, which has its own keys) over 8/16-bit and
    # 4-bit caches as "<block_q>:<pages per step>" ("" = 16:16). Not bitwise
    # (accumulation order).
    "minimax_dcp_small_batch_tokens": 0,
    "minimax_dcp_small_batch_wide_pages": 0,
    "minimax_dcp_prefill_tiles": "",
    "minimax_dcp_prefill_tiles_4bit": "",
    # MSA work list: dense-compare unions up to this many pairwise elements
    # (0 = 1 << 20), sorted ones above. Same unions, bitwise.
    "minimax_msa_dense_work_list_elems": 0,
    # MSA DCP candidate selection: a request class's scores (live-row grid)
    # are not copied into a padded buffer; the last block reads past them.
    # Same picks, bitwise.
    "minimax_dcp_unpadded_class_scores": False,
    # MSA DCP over one KV head (e.g. DCP 2): the attention half (query and
    # candidate gathers, attention, partial exchange, LSE merge) as one SPMD
    # op (`kernels/msa_dcp_merge.py`), permutes on a DCP 2 pair up to 1024
    # tokens, one Pallas merge pass in f32. Supersedes the packed
    # collectives above for those groups. Not bitwise (f32 merge).
    "minimax_dcp_attend_merge": False,
    # MSA DCP groups over several KV heads (e.g. DCP 8): each rank of a KV
    # head's replicas all-gathers only its 1 / replicas part of the new K / V
    # / index-query rows (the replicas' rows are equal). Bitwise.
    "minimax_dcp_split_kv_gather": False,
    # --- v7 scheduler ---
    # Scheduler: PrefillSpacingScheduler (`core/prefill_spacing_scheduler`,
    # shortest remaining prefill first over RUNNING and WAITING, on top of
    # PrefixAwareScheduler) replaces an unset or PrefixAwareScheduler
    # --scheduler-cls. Host-only; changes which tokens each step schedules,
    # not any request's tokens.
    "prefill_spacing_scheduler": False,
    # PrefillSpacingScheduler: decode-only time owed after a step that runs a
    # chunk of a long prompt, as a fraction of that step's measured time
    # (0.0 = no spacing; r10's default was 0.5).
    "prefill_spacing_frac": 0.0,
    # PrefillSpacingScheduler: a prompt is long while at least this many of
    # its prompt tokens are left (0 = 8192, r10's default).
    "prefill_spacing_min_tokens": 0,
    # PrefillSpacingScheduler: TPOT target in ms; prefill is deferred while a
    # long-chunk step would push a decoding request that can still meet it
    # past it (projected from max_tokens; 0.0 = off).
    "prefill_tpot_target_ms": 0.0,
    # PrefillSpacingScheduler: longest run of steps, in seconds, that the TPOT
    # target alone may defer prefill for (0.0 = 10 s, r10's default).
    "prefill_tpot_max_hold_s": 0.0,
    # PrefillSpacingScheduler: while the TPOT target defers prefill, still run
    # whole remaining prefills up to the largest prefill-token bin
    # (512-4096) whose measured step time fits every decoding request's slack.
    "prefill_tpot_priced": False,
    # --- v7 runner/spec ---
    # DFlash/DSpark draft forward: extra buckets above 64 tokens this many
    # apart (an eighth of the power of two apart past 8 x the step), so a
    # draft of num_reqs * block tokens pads <= ~12.5 % instead of up to ~45 %
    # (`DFlashProposer._fine_draft_buckets`). More precompiled draft programs.
    # 0 = the target's token buckets only. Bitwise per row (padding only).
    "spec_draft_bucket_step": 0,
    # Synthetic one-row verify (spec_synthetic_one_row_verify) at TP > 1 with
    # >= 4 request rows per rank: each rank gets its block of the rows over
    # an all-to-all of the LM head shards (instead of an all-gather of the
    # full logits), samples it, and the tokens are all-gathered
    # (`compute_selected_logit_rows`). Needs the model's
    # compute_logits_shard. Bitwise (same logits rows, same uniforms).
    "spec_synthetic_row_split": False,
    # Token recovery without top-k or draft probabilities (the synthetic
    # verify's sampler) on a Pallas kernel holding each row as a dense
    # [vocab / 128, 128] slab (`kernels/top_p_sample`). Same steps as the XLA
    # path, the per-row sums in another order: not bitwise (float rounding of
    # the softmax / top-p mass; a token can differ on a near-tie).
    "spec_sampler_top_p_kernel": False,
    # Runner: stop InputBatch rebuilding its (unused on TPU) device
    # sampling_metadata on every batch change: no per-num_reqs copy programs
    # compiled online, no H2D copies behind KV-offload loads. Host-side;
    # bitwise (the TPU samplers read the CPU sampling tensors).
    "skip_device_sampling_metadata": False,
    # DSpark greedy Markov chain over a vocab-parallel LM head: its K steps as
    # one Pallas kernel with remote-DMA exchanges of the (max, id) pairs
    # (`distributed.pallas_markov_argmax`, self-tested at load; buckets that
    # do not fit VMEM keep the XLA loop). Scores in f32, so a draft can differ
    # from the XLA loop's bf16-rounded scores on near-ties (r10: 0-6 % of
    # drafts); verified tokens are unchanged, only acceptance can move. Not
    # bitwise in drafts.
    "dspark_markov_kernel": False,
    # DSpark vocab-parallel argmax (XLA loop): float32 ids and the cross-rank
    # argmax as masked reductions, int32 ids along the chain. Bitwise (same ids).
    "spec_vocab_argmax_masked": False,
    # --- v7 offload ---
    # Raiden KV offload: MiB of one worker's KV that local offload loads may
    # have in flight at once (0 = no bound). A larger load is split and lands
    # over several scheduler steps, so the step's own input uploads (which
    # queue behind issued host->HBM copies) wait less. Host-only, bitwise.
    "offload_max_inflight_load_mib": 0,
    # Raiden KV offload: MiB of one worker's KV that offload saves may have
    # in flight at once (0 = no bound). A store job runs as pieces of whole
    # keys (each its own insert + save), so the next forward's KV write waits
    # behind fewer issued HBM reads. Host-only, bitwise.
    "offload_max_inflight_store_mib": 0,
}


def option(additional_config: Any, key: str) -> bool | int | float | str:
    """`additional_config[key]`, or the key's default when unset."""
    default = DEFAULTS[key]
    if not isinstance(additional_config, dict) or key not in additional_config:
        return default
    value = additional_config[key]
    # JSON has one number type: a float key also takes an int (22 -> 22.0).
    if type(default) is float and type(value) in (int, float):
        return float(value)
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

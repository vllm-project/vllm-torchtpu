# vtt-combined: file-level merge of prefill (P) + decode (D) trees

* P = `cd4eedbcb` (prefill, 197K tok/s/node). D = `1d02e4ab9` (deployed decode stack). The branches share no git history.
  This branch started at D, and every one of the 42 differing `src/` files was decided individually (table below).
* **Role selector: `VLLM_DSV41_PURE_20L_PREFILL`.** It is the only flag set exclusively on the prefill node (=1); decode leaves it unset (=0).
  Where both sides' code was live under the other side's config and not already separated by a differing flag, the
  merged file runs **P's code iff `envs.VLLM_DSV41_PURE_20L_PREFILL`** and D's code otherwise.
  * Phase-1 P/D prefill (`CED=1, PURE_20L=0`, DP8/TP1, see `dsv41/PD_DESIGN.md` §5) therefore takes the **decode**
    code paths everywhere, including the KV layout. Producer and consumer layouts are byte-identical. No file
    switches to a prefill-only layout or kernel on `CED`/`TP_ATTN` alone.
  * No new env variables were added. Every selector reads an existing, registered flag.
* Approximate 3-way bases were computed per file from both histories (blob timelines). In most files D equals P's base
  snapshot (`6c4e1a322`), so the diff is P-only. Where both sides changed a file, the hunks were classified one by one.

## Per-file decisions (42 `src/` files + 5 tests/tools)

`take-P` / `take-D` = whole file from that side. `hand-merged` = both sides' changes kept, role-gated where needed.
`identical-after-merge`: no file needed this category; every file is in one of the three groups below.

| # | File (under `src/vllm_torchtpu/`) | Decision | Rationale |
|---|---|---|---|
| 1 | `__init__.py` | hand-merged | D base. P's `POD_G_CHIP_ID` worker binding is added; it is self-gated, since the var is unset on decode. P's compact-SWA BlockPool patch in `_run_engine_core_with_tpu_patches` runs only if PURE_20L. D's global `jax_op` dynamo-disable wrap and both DecodeBench patches run only if not PURE_20L. |
| 2 | `compilation/shape_variants.py` | hand-merged | `_strict_op_handover()` (=PURE_20L) gives P's stricter op-signature bucket handover on prefill and D's rule, including the positional fallback, on decode. A 40k-case randomized equivalence check against P (flag=1) and D (flag=0) found 0 mismatches. |
| 3 | `compilation/tpu_compiler.py` | take-P | Ignores `ENABLE_DECODE_BENCH_CONNECTOR` and `VLLM_XLA_CACHE_PATH` in the compile hash. These are registered in the merged `envs.py`, so this keeps the decode hash free of them. |
| 4 | `distributed/kv_transfer/decode_bench_seeding.py` | take-D (unchanged) | D-only file: template pre-seed disabled (no host-RAM spike). Left untouched for the `pd` thread. |
| 5 | `envs.py` | take-P | Strict superset of D's registry (about 30 extra `VLLM_DSV41_*` knobs). Every env var either config sets keeps the same meaning (see the check below). |
| 6 | `kernels/deepseek_v4/core_attention/csa_gather.py` | hand-merged | P kernel for both roles (faster; adds the `max_active_per_chunk` kwarg that P's `sparse_mla` passes; decode gets a pure gather speed-up). The base/D `first_idx >= 0` guard for -1-marked union chunks is added only when not PURE_20L, so the prefill kernel body is byte-identical to P. The guard is a no-op when CSA_UNION=0. |
| 7 | `kernels/deepseek_v4/core_attention/mla_swa.py` | take-P | Decodes P's packed SWA kv-length spans (needed with P's `tp_attention` and attention op). Identical to D at decode, where nothing packs the lengths. |
| 8 | `kernels/deepseek_v4/o_projection.py` | hand-merged | P is needed for prefill (H=64, K-blocked `wo_b` scales). The only D-visible change is the VMEM vs weight-outer threshold for 256 < t_pad <= 384: 256 when PURE_20L (P), 384 otherwise (D). |
| 9 | `kernels/deepseek_v4/rope.py` | take-P | D equals P's parent, so P is D plus the **indexer sign fix** (`sign(w)*ReLU(|w| q.k)`) in `_rope_quant_body`. |
| 10 | `kernels/deepseek_v41/compressor.py` | hand-merged | P base, with `_prefill_role()` (=PURE_20L). Prefill keeps P's kernels and last-token-only state writes. Decode keeps D's per-token state writes (needed by DSpark rollback), 4-deep scatter ring, Pallas insert above 512 tokens, 16-page decode KV blocks and unpadded `wq_b` kernel. P's aliased-state path (`state_cache=None`) is kept for both roles. The **sign fix is applied to both P's and D's `wq_b` kernels** (see Risk R1). |
| 11 | `kernels/deepseek_v41/sparse_mla.py` | hand-merged (P + decodefix) | The candidate-union code is dead on both configs (`CSA_UNION=0`). Decode (T<=384) reaches a fallback branch that is byte-identical in P and D. Prefill (T=16384) needs P's 256/512-token chunking (0ca85037d). P is the API superset (`precompute_csa_union_and_gather`). D's different union implementation is dropped. Decode role: decodefix's sinks-in-running-max fix applied to all b_q<8 blocks (`_m_run`, R6). |
| 12 | `kernels/deepseek_v41/streamindex_topk.py` | take-P | D equals P's parent, so P is D plus the **indexer sign fix** (no `w/max(w)` folding before the ReLU). This must ship together with rope.py and compressor.py. |
| 13 | `kernels/deepseek_v41/tp_attention.py` | take-P | Only used with TP_ATTN at TP>1 (prefill). D's call sites are API-compatible. |
| 14 | `kernels/experimental/moe_dense_expert.py` | take-P | Disabled in both trees (`can_use_dense_expert` returns False). P adds optional kwargs used by the merged `fused_moe_gmm`. |
| 15 | `kernels/fused_moe/v2/host.py` | hand-merged | P's extra args (prefetch, VMEM override) have defaults equal to D's behaviour. D's docstrings were restored. |
| 16 | `kernels/fused_moe/v2/kernel.py` | hand-merged | P (sub-channel fp8, BF16 wire, `requantize_mid`, coalesced peer push). The coalesced-push default turns on only if PURE_20L. Otherwise it is D's kernel: same name, scratch and push schedule. |
| 17 | `kernels/fused_moe/v2/layer.py` | hand-merged | P signature. When the caller passes nothing, sub-channel / BF16-wire / coalesce default on only if PURE_20L. The routing block size uses P's formula on prefill and D's power-of-two loop otherwise. |
| 18 | `kernels/fused_moe/v2/router_ops.py` | take-D | **Decode fix:** fall back to `block_rows=R` instead of asserting. A no-op for every block size P produces. |
| 19 | `kernels/megablox/gmm_v2.py` | hand-merged | P's float32 K-loop accumulation for 4-bit weights applies only if PURE_20L. Decode keeps D's bf16 accumulation. |
| 20 | `kernels/pool_adapters.py` | take-D | P's only change is a dead `envs` import placed above the license header. |
| 21 | `layers/adapter/custom_ops/deepseek_v41/deepseek_v41_flash_attention_op.py` | take-P | Handles P's zero-row dummy main cache when caches share a buffer. Identical for D's calls. |
| 22 | `layers/adapter/custom_ops/deepseek_v41/deepseek_v41_flash_compressor.py` | take-P | Strict superset of D. The aliased-state op (`_aliased_state_to_nope`, default False), the encoder-cand mode and the decoder-TP-verify path are all inert under the decode env. |
| 23 | `layers/adapter/fused_moe.py` | take-D | D's float4 / FP8-dispatch changes sit behind `VLLM_TPU_MOE_FP8_DISPATCH`, which is off on prefill. |
| 24 | `layers/adapter/fused_moe_ep.py` | hand-merged | P base (int32-packed fp4, `op._jax_fn`, precomputed routing). `token_replica_groups=None` **unconditionally**, which satisfies both the P fix and the D fix. `_prefill_role()` is in the op cache key. On decode: capacity 128 (unless `VLLM_DSV41_MOE_V2_CAPACITY`), sub-channel fp8 off, FP8 wire, per-expert push, native `float4_e2m1fn_x2` passthrough, matching D. |
| 25 | `layers/adapter/moe_routing.py` | take-P | Adds `maybe_modulo_remap_topk_ids`, inert unless `VLLM_DSV41_MODULO_EXPERT_BALANCE` (unregistered, getattr default False). |
| 26 | `layers/adapter/quantization/fp8.py` | take-P | The only difference is a redundant top-level import. Every use site imports locally. |
| 27 | `layers/adapter/quantization/mxfp4.py` | hand-merged | `_prefill_weight_prep()` (=PURE_20L) picks P's on-TPU int32 `[E,K/8,N]` prep or D's CPU dequant to native fp4 for `process_weights_after_loading`, requantize and admission meta. The forward is P's; its extra branches are inert on decode. One reorder on decode: the fused-EP layout is admitted before the source weights are released, because P's `prebuild_fused_moe_ep` reads `w13_weight.device`. |
| 28 | `layers/core/fused_moe_gmm.py` | hand-merged | `_prefill_moe_numerics()` (=PURE_20L) runs P's code (W4A16/fp8-lhs, `_sub_tile`, 32768-row gather) on prefill and D's (W4A8 on block-512 fp4, D tiling, 16384 rows) on decode. Without the gate, P's code would switch decode to bf16 activations, because decode sets `MOE_FP4_GMM1_FP8=0`. |
| 29 | `models/vllm/deepseek_v4/attention.py` | take-P | Compile-safe `_same_cache_buffer`. DSv4.1 overrides the method anyway. |
| 30 | `models/vllm/deepseek_v4/layers.py` | hand-merged | P's 4-D shared-expert scale support (prefill TP=2, `REPL_SHARED_EXPERTS=1`) plus D's three fixes: the `quantized_matmul` import, the guarded `quant_method.process_weights_after_loading` loop and the correct `tp_shard_slice(x, tp, rank, None)` calls. |
| 31 | `models/vllm/deepseek_v4/model.py` | take-P | The modulo expert remap is inert (flag unregistered, default False). |
| 32 | `models/vllm/deepseek_v4/moe.py` | hand-merged | P base, plus D's `shared_experts(x_shard, input_is_token_sharded=True, return_token_sharded=True)` calls: identical at TP=1, and a branch prefill never reaches (it uses `forward_tp_local`). |
| 33 | `models/vllm/deepseek_v41/attention.py` | hand-merged | P base, with `_PREFILL_ROLE` (=PURE_20L) at 10 sites. Prefill gets P (fused h128 wo_a+wo_b, fused CSA input GEMM, `quantized_matmul` wqa/wkv, rank-4 q-rope scales, dummy main cache for shared buffers, DSpark draft SWA grouping, per-page-count SWA metadata, aliased-state prewarm). Decode gets D's code. Also a `hasattr` guard on the nonexistent `fused_reverse_rope_wo_a_wo_b_g1_projection` (only an 8-head crash case). |
| 34 | `models/vllm/deepseek_v41/cond_backbone.py` | take-P | Formatting plus envs-registry reads; inactive (`SINGLE_BACKBONE_COND=0` on both nodes). |
| 35 | `models/vllm/deepseek_v41/dspark.py` | hand-merged | P (envs-registry reads) with D's `>= 0` alias lift restored, so the decode-only drafter is exactly D. |
| 36 | `models/vllm/deepseek_v41/model.py` | hand-merged | P base (PURE_20L 21-layer build, encoder-only early return, `> 0` alias lift, a no-op at TP=1). D's engram fallback (on-device hash plus `engram_cpu_lookup_all`, live on decode) applies when not PURE_20L; prefill keeps P's zero-fill. D's Bug #77 extra warm-up op runs only when not PURE_20L. |
| 37 | `platforms/tpu_platform.py` | take-P | Longer env propagation list; only affects the Ray executor. |
| 38 | `runner/kv_cache_deepseek_v41_flash.py` | hand-merged | **Two layouts kept verbatim, selected by PURE_20L** (see the next section). |
| 39 | `runner/tpu_runner.py` | take-P | Three semantic hunks, all equivalent on both configs: rope truncation is a no-op for DSv4.1 (attention.py truncates at init); `ENGRAM_H2D_PREFIX` via the registry; the comb-cache key is a function of the same tuple. |
| 40 | `spec_decode/dflash.py` | take-D | Decode-only spec-decode fixes. P never changed it. |
| 41 | `spec_decode/dspark.py` | take-D | Deployed DSpark proposer, matching D's dflash and tests. P's extra `96*K` draft bucket (>64 concurrent drafts only) is not carried. |
| 42 | `worker/tpu_worker.py` | hand-merged | P base. P's gc/sync after the profile run and the HBM-stage logging run only if PURE_20L. D's KV-budget scaling for the SWA host arrays runs only for DeepseekV41 when not PURE_20L. It must switch together with the kv-cache layout. |
| t1 | `tests/kernels/fused_moe/test_fused_ep_moe_v2_tp_tokens.py` | take-P | **Must-survive prefill regression test.** Needs 8 local TPU devices; skips on CPU. |
| t2 | `tests/kernels/test_router_topk.py` | take-D | Adds tests only. |
| t3 | `tests/models/vllm/test_dsv4_model.py` | take-D | D-only test. |
| t4 | `tests/spec_decode/test_dspark_v41_cpu.py` | take-D | D-only test. It already fails to import on D (`_gather_aux_hidden_states` exists in neither tree). |
| t5 | `tools/check_env_reads.py` | take-P | Adds `POD_G_CHIP_ID` to `BOOTSTRAP_VARS`. |

Totals over the 42 `src/` files: take-P 16, take-D 6, hand-merged 20 (sparse_mla became P plus the role-gated decodefix sinks fix).

## `runner/kv_cache_deepseek_v41_flash.py` (hardest file)

The two layouts are a real design conflict, so both are kept verbatim and the role picks one:

* `_use_prefill_compact_swa_layout()` returns `envs.VLLM_DSV41_PURE_20L_PREFILL`.
* **Prefill (=1)**: `_init_prefill_compact_swa_layout`, P's code byte for byte:
  * one compact, **non-overlapping SWA ring array per SWA layer** of `max(num_swa_aux, coord_swa)+1` blocks, with the
    **768-block SWA ring fix** (`aux_partition_sizes` returns 768 for 256 <= nb < 4096);
  * compressor state **aliased** onto the CSA arrays, plus the aliasing assertion on MLA arrays;
  * main_end = nb − state_aux.
  * `_apply_compact_swa_block_pool_patch()` (fresh SWA block ids 1..cap in the scheduler BlockPool) is a no-op unless
    PURE_20L. It runs both at module import and from `_run_engine_core_with_tpu_patches`.
* **Everything else (=0)**: `_init_decode_overlay_layout`, D's code byte for byte:
  * SWA overlaid on the CSA NoPE hosts in the tail partition of the shared pool;
  * a **separate per-layer compressor state cache** `(nb, state_rows, ROW_WORDS)`;
  * UE8M0 init limited to `slice(1, main_end)`, with main_end = nb − state_aux − swa_aux matching the vllm-decode
    coordinator;
  * `tensor._main_end` set (read by `decode_bench_seeding` and the `__init__` DecodeBench patch); 24 KV arrays.
* Shared prologue and epilogue. The epilogue marks `_aliased_state_to_nope` from data pointers on every compressor
  module (True on prefill; False on decode, which equals the op's default).

## Static checks run locally
* `python3 -m py_compile` on all 38 changed files: OK. `pyflakes` over `src tests tools`: 0 undefined names.
* Import smoke test of all 42 changed modules: see the "Local verification" section of the hand-off message (CPU venv, py3.12, the deployment wheels).
* Launch-env check (`check_launch_envs.py`, output summarized here). Every env var exported by the prefill
  `LAUNCH_SH`, the decode `boot_dp8_scale_server` and the decode sitecustomize is one of:
  * registered in the merged `envs.py`, with a definition identical to P's and D's registries (`VLLM_DSV41_TP_INDEXER_MIN_TOKENS`
    differs from D only in line wrapping);
  * read raw via `os.environ` in vtt;
  * owned by vllm;
  * external (libtpu, metadata mock, OS).
* These decode-set vars are read by **neither** P nor D: `VLLM_DSV41_CSA_UNION_SC`, `HOIST_CSA_GATHER`, `HOIST_CSA_MAX_TOKENS`,
  `INDEXER_CAND_READ_16K`, `L2_CAND_K_MIN_TOKENS`, `TP_SHARD_ENGRAM`, `ULYSSES_MIN_TOKENS`, `VLLM_SPEC_DECODE_TOKENS`,
  `VLLM_SPEC_WARMUP`, `ENABLE_CONNECTOR`. They were dead before the merge too, so nothing regressed.
* Decode-set vars that D never read but P's code does read (`REPL_SHARED_EXPERTS=1`, `MOE_FP4_GMM1_FP8=0`, `SC_UNION_MAX_BLK_SHIFT=-1`,
  `DECODER_TP_ATTN_MIN_TOKENS=999999`, `ENABLE_DECODE_BENCH_CONNECTOR=1`): each is either behind the PURE_20L gate in the merged file or inert
  under decode (union off, TP=1). The analysis notes have the details.

## Risks to validate on TPU
* **R1, behaviour change on decode: indexer sign fix.** `rope.py`, `streamindex_topk.py` and `compressor.py` carry the
  cd4eedb sign fix for **both** roles (score cosine 0.212 → 0.9997 in P's ubench). It is a model-correctness fix, and
  the three files must agree. But the decode indexer top-k is no longer bit-identical to 1d02e4a whenever a head weight
  is negative.
  * Expect decode **step time unchanged**. Gate text may differ from the 1d02e4a baseline, possibly improving coherence.
  * For an exact-D A/B, revert those three hunks in compressor.py's D `wq_b` kernel (`jnp.abs(ws)`/`jnp.sign(ws)`, around L2263/L2275)
    plus rope.py and streamindex_topk.py to D. Do not revert one file alone.
* **R2.** Every prefill server **must** run with `VLLM_DSV41_PURE_20L_PREFILL=1`. Without it, the server silently gets decode
  numerics and layout. The prefill driver already sets it.
* **R3. Cold compile caches.** The larger env registry changes the compile-hash env factors, so the first prefill and decode
  starts compile cold. Pre-warm before benchmarking. Keep separate prefill and decode XLA caches, because the role-dependent kernels share op names.
* **R4. vllm.** The combined stack runs vllm-decode (common patch plus decode runtime v2) for both roles.
  * At ~2084 prefill blocks, the coordinator's aux sizes, the 16384 num_blocks cap and the pool_partition change are no-ops.
  * The bench client must keep `VLLM_BENCH_DP_SIZE=1` on prefill; the driver already does this.
* **R5. Prefill weight loading.** D's `quant_method` re-process loop (layers.py) and moe.py's shared-expert path assume every
  prefill shared expert has `has_tp_local_weights()` True and is already flipped. Check the startup log, and that 21 layers are built.
* **R6. decodefix integration: done** (cpk/dsv41-decode-fix `11e6c02`, `570c809`, `cfacfa7`; `b5665ed` was already present).
  * `11e6c02`, SWA page indices per KV-cache group:
    * `kv_cache_deepseek_v41_flash.py` tags each cache layer with `_dsv41_kv_cache_group_id`; this runs in both layouts and is harmless.
    * `attention.py` `_swa_kv_cache_group_id()` makes the SWA group bounds and their memo key per group **on the decode role only**.
      Page-index cache entries are keyed on the page count in both roles.
    * Prefill-layout analysis: the same group-agnostic sharing exists in cd4eedb, but there it is **benign**. P's compact layout
      gives every SWA layer its own private array (`overlay_hosts[len(...)]`, 21 arrays for 21 SWA layers). A layer that
      follows the leader group's block table therefore only touches its own array, and the table's block ids are unique among
      live requests, so reads and writes stay self-consistent across 16K chunks. The decode layout is different: there
      same-position layers of different groups share one array, which is the corruption decodefix fixed. Prefill therefore
      keeps cd4eedb's behaviour. Phase-1 P/D prefill (`PURE_20L=0`) uses the decode layout and gets the fix.
  * `11e6c02`, sinks in the running max: `sparse_mla.py` is P's in this tree, and in P **all four** b_q<8 unrolled blocks
    (1/2/3-chunk plus the batched kernel) omit `sinks_t` from `m_new`. `_m_run()` includes the sink on the decode role, and the
    prefill kernel body is unchanged. These blocks are only traced on the CSA-union path.
  * `cfacfa7`: per-group SWA bounds are memoized at KV init, outside any trace, on the decode layout only (otherwise the decode step is +14%).
  * `570c809`: DSpark target hidden-state capture, cherry-picked into the P-based `model.py` with one conflict resolved.
    `capture_entry_aux` defaults to False, and `_run_layer` only checks ids when aux layers are set, so the PURE_20L encoder path is unchanged.
  * `b5665ed`: the sign fix was already in combined from P. `rope.py` and `streamindex_topk.py` are byte-identical to decodefix,
    and compressor.py's `_fused_indexer_wq_b_rope_quant_decode_jax` applies `abs(ws)`/`sign(ws)`.
* **R7.** The gating assumes decode stays at TP=1 with CSA_UNION=0. TP>1 decode or CSA_UNION=1 would run P's ungated
  TP-attention or union code and needs fresh validation.
* **R8.** If decode HLO fingerprints differ from 1d02e4a, check these first: `fused_moe_ep`, `fused_moe_gmm` and `mxfp4`
  (the largest role-gated files), then compressor (R1).

### Recommended TPU validation
1. **Prefill sweep** on the combined tree with the standard prefill driver (PURE_20L=1). It must be within ±2% of 197,316 tok/s/node at 128K c=8; check all 6 workloads.
   * Also check the startup log for "KV cache: N arrays", num_blocks=2084 and the compact SWA shape `(769, ...)`.
2. **Decode c=64 synthetic** (`run_decode.py`, decode sitecustomize). The step must match 81.8 ms (±2%).
   * Expected log lines: "24 arrays" and "Truncated 0 of N rope caches".
   * 2+2 must pass on all 8 ranks. Record the USA output; it may differ from baseline because of R1.
3. Optional: an exact-D decode A/B with R1 reverted, to separate the sign-fix effect from merge regressions.

## Known follow-ups
* **Unify allocator (G8).** The prefill (PURE_20L) and decode KV layouts are still two code paths. Phase-2 P/D (PURE_20L
  producer → DP8 consumer) needs a single layout (PD_DESIGN.md §5 G8). Phase-1 (`CED=1, PURE_20L=0`) already uses the
  decode layout on both peers.
* Re-evaluate P's DSpark draft-SWA grouping and per-page SWA metadata cache (attention.py, gated off on decode) as
  candidate decode fixes, separately from this merge.
* Run `tools/check_env_reads.py --update` after the merge, and pre-commit once the tree is validated.

# ix-unified-oct23: one tree for TP8+DCP4 and DP4×TP2 decode

This branch serves both decode layouts of the Qwen3.5-397B P/D curve from one tree. The layout is chosen only by runtime flags and environment variables. Both layouts run the same code.

- **TP8 + DCP4** (bands C to H with the megakernel; bands A and B with the E6 or o6 configs): this is `shriharim/ix-curve-oct21-merged` @ `52925e2eb`. See `CURVE_README.md` in this tree for the per-band trees, environment and full commands.
- **DP4 × TP2** (bands A and B): this is `shriharim/ix-dp4tp2-oct22` @ `7bad8dfde` plus the env-gated DP4×TP2 candidate commits, the fused-EP v4 flags (`compact_message`, `reorder_chain`) and the proxy's DP-rank routing (`PROXY_DECODE_ROUTING=dp_affinity`). All of them are off by default.

## ix-unified-final: the single superset branch (2026-10-24)

`ix-unified-final` is `31750c0fd` (this README's previous tip, the gated fold) plus the two local series below and one opt-in proxy alias. Every cherry-pick applied without a conflict and is patch-id identical to its source. No file is touched by both series. Every new switch is off by default, or defaults to the behaviour of `31750c0fd`; with nothing set, a config renders and runs the code paths of `31750c0fd`.

| Commit here | Source | Change | Default |
|---|---|---|---|
| `5b8013baf` | `164abfad2` (ix-unified-port) | Host-overhead stack: `runner/fast_forward_context.py`, `runner/decode_carry.py` | off; `VLLM_TPU_FAST_FORWARD_CONTEXT=1`, `VLLM_TPU_DECODE_CARRY=1` |
| `0c1ced0f9` | `1d6e27ef2` | Megakernel SH_AT | `VLLM_TPU_QWEN35_MK_SH_AT=0` (kernel unchanged) |
| `724758d71` | `6b1683701` | Final-lineage decode admission throttle | off; `TPU_SCHED_DECODE_INFLIGHT_THROTTLE=1` |
| `6a5ffde62` | `5150929c5` | Switches `TPU_PP_QUAD_COST`, `TPU_PP_INFLIGHT_COST`, `TPU_SCHED_PAW_PROMOTE_WARMEST` | 1 (unchanged); 0 restores the old trees |
| `9032feea6` | `6b1ad14d7` | Test: the throttle acts through vLLM's hook | n/a |
| `179430263` | `9d830e80b` | Fix: a quantized draft lm-head proxy keeps its buffers on the head's device (`TPU_DRAFT_LM_HEAD_QUANT=fp4` crashed on the CPU buffers) | reached only with the draft quant env |
| `b2916cac6` | `3ebd37b21` | Fix: with fast forward context off, the call sites use their own `set_forward_context` | n/a |
| `223a873b3` | `5134714a0` | Opt-in final-lineage PP8 chunk scheduler flavour | unified's class; `TPU_PP_SCHEDULER=final` |
| `79400d892` | `fdd669a2c` (ix-unified-dpmk) | Fused EP v3 dense-slots layer for DP4×TP2 decode | off |
| `87a9ab879` | `ac256c909` | Adapter routing to dense slots: `Config.dense_slots`, `dense_slots_max_tokens` (64), `dense_slots_fused`, `dense_slots_pallas_tables`, `dense_slots_merged`, or the `TPU_MOE_FEP_DENSE_SLOTS*` env | off |
| `fa629aae8` | `7261dd71f` | GDN decode tile size through `TPU_MOE_FUSED_EP_V3_CONFIG` `"gdn_decode_tile_size"` (read only when `TPU_GDN_DECODE_TILE_SIZE` is unset) | 0 = the kernel's choice |
| `f2af3fca8` | `ed615acf3` | Test: dense slots in Pallas interpret mode on 8 fake CPU devices | n/a |
| `03cabe740` | new | Toy proxy `PROXY_PREFILL_ROUTING=prefix_hash_sticky`: prefix_hash with `PROXY_PREFIX_HASH_SATURATED_INFLIGHT` defaulting to 1 (an explicit value wins). The DP4×TP2 recipe lineage drops `PROXY_PREFIX_HASH_SATURATED_INFLIGHT` but always passes `PROXY_PREFILL_ROUTING`; the std/uport lineage passes every `PROXY_*` name, so either form works there | off; `prefix_hash` keeps 0 |

### Runtime env adopted for DP4×TP2 (no code change; render-check every name)

- Decode fused-EP JSON (v3 dense slots, gated in `j-dcfed5d3` 97.73% and with gate 64 in `j-49c0a4df` 97.73%): `DECODE_TPU_MOE_FUSED_EP_V3_CONFIG={"combine_real_tokens":true,"sort_index_compare":true,"compact_message":true,"reorder_chain":true,"dense_slots":true,"dense_slots_fused":true,"dense_slots_merged":true,"gdn_decode_tile_size":1}`.
- Gate 64: `DECODE_TPU_SP_MIN_ROWS=64`, `DECODE_TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=32` (replaces 88 / 44 in the DP4×TP2 column below).
- Prefill buckets2: `COMPILE_SIZES_PREFILL=8,16,32,64,128,256,512,1024,1536,1664,2048,2176,3840,4352,6016,6528,8192` (padding only).
- **Rejected: fp8 prefill attention** (`TPU_RPA_NATIVE_FP8_PREFILL=qkspvs`). Do not set it.
- Sticky prefill routing is under test (`j-068c5ccc`, on a tree copy). It is not adopted; to try it on this branch, set `PROXY_PREFILL_ROUTING=prefix_hash_sticky`.

### Recipes

- DP4×TP2 points: the `cloud-devkit-dp4tp2` / `cloud-devkit-pfload` lineage (passthroughs below). Its proxy env is an allowlist.
- TP8 points that use the ports (fast forward context, decode carry, SH_AT, the scheduler switches, `TPU_PP_SCHEDULER`, `TPU_DRAFT_LM_HEAD_QUANT`): the private clone `~/claude/cloud-devkit-uport` (llmdport + opt-in passthroughs `4a5b6443`, `5d012133`, `599e00ed`). Std drops every one of those names silently.
- Every 72/80 entry must be removed from `VLLM_TPU_QWEN35_MTP_MK_VERIFY_BUCKETS` (buckets above 64 silently disable the MTP megakernel on this tree).

## What the branch is

The branch is `52925e2eb` plus these cherry-picks, in order. Every cherry-pick applied without a textual conflict. Only `tpu_runner.py` needed a 3-way auto-merge, for `cf54717bc` on top of R5, and `toy_proxy_server.py` for `0aeb6fcb5` next to the llmd router. Each cherry-pick is patch-id identical to its source.

| Commit here | Source | Change | Default |
|---|---|---|---|
| `824352014` | `cf54717bc` | Greedy top tokens under EP-DP with a TP head. The target keeps full logits, and the draft head keeps the vocab-parallel argmax. | Acts only when DP>1 with EP and TP>1. |
| `3843e3c7f` | `2564cf137` | `tpu_communicator`: a replicated (below `TPU_SP_MIN_ROWS`) sequence-parallel MoE step under DP>1 gathers over DP, and combines as a DP reduce-scatter followed by a TP all-reduce. | Acts only when the DP group size is above 1. |
| `42b037cbb` | `4afe2bb67` (= `a82efefdd`, same patch-id) | `TPU_RPA_DECODE_LANES`: sets the decode lanes of the batched RPA decode kernel. | 0 = off |
| `e42a24bc1` | `54a54831f` (= `58f792c52`, same patch-id) | `TPU_DP_PACKED_ROUTER_GATHER`: one packed DP all-gather of the hidden rows and the router logits, instead of two. Lossless. | off; acts only under DP>1 |
| `fa40006aa` | `933c26ed8` | `TPU_MOE_FUSED_EP_V3_CONFIG` (JSON overrides of the fused EP v3 `Config`) and `TPU_MOE_FUSED_EP_V3_TILE_ROWS`. | empty and 128, which are the old values |
| `12c582ff9` | `815539b67` | Fused EP v3 `Config.combine_real_tokens`. | off |
| `6dbc33888` | `a994fc7a3` | Fused EP v3 `Config.sort_index_compare`. | off |
| `fd68f98a2`, `910e1a6de` | `bd10e8d5a`, `b44438a09` | Bench harness only (`p3_d_leg_bench.py`, `smallb_job.sh`, `build_prompts.py`). No engine change. | not applicable |
| `ae61a5c5c`, `8fd4dd025` | — | This README (first push, 2026-10-23). | not applicable |
| `165100aad` | `fbbcbf9ae` | Fused EP v3 `Config.compact_message` (v4): a shard of at most 64 rows sends a 64-token-per-slot routing message, so the routed pairs per shard halve (2048 → 1024 at DP4×TP2 decode). The pair order is kept, so the output is bit-identical. | off |
| `7cfe329e8` | `2379f0241` | Fused EP v3 `Config.reorder_chain` (v4): the dispatch chain's index-independent work runs as separate programs inside the SparseCore waits. Only the schedule changes. | off |
| `b482ec261`, `7fee40460` | `e627ddaf7`, `cdc03b603` | Tests for the two v4 flags: geometry (CPU) and device bit-identity under normal and hot routing (TPU, `P3_BENCH=1`). | not applicable |
| `c5b4bbad9` | `0aeb6fcb5` | Toy proxy `PROXY_DECODE_ROUTING=dp_affinity`: pins each decode request to a DP rank of its decode host (`X-data-parallel-rank`), prefix-affine with a KV-aware load guard; the prefill hop never gets a rank; tests. | off (any other routing value leaves it unused) |
| `64a613bed` | new (test only) | `test_fep_v3_compact` geometry runs on CPU: off a TPU it uses TPU7x's published sparse-core info. Before this, its 9 cases failed on CPU. | not applicable |
| `7b45f3df2` | new (test only) | Test that the four opt-in fused-EP flags (`combine_real_tokens`, `sort_index_compare`, `compact_message`, `reorder_chain`) default to off. Before this, no CPU test caught a flag defaulting on. | not applicable |

`TPU_GDN_FUSED_1DOT` is an env-only switch that was already in the merged tree.

The merged branch already contained all of o6q's lineage (`69c56a021`, prefill v2). `git cherry` matched these by patch-id: `d5c40130f`≡`c297c20e0`, `71d6e2c75`≡`4d61bcd7d`, `46e5e0103`≡`f596f8587`, `c784ad231`≡`e1c7027a7`, `09079d1d5`≡`f3c9b3a34` and `69c56a021`≡`ed366ff3a`.

## Why TP8 behaviour is unchanged

**Tree identity (measured with git):**

- The unified tree minus the DP4×TP2 candidate tree (`a994fc7a3`) is exactly the merged branch's own delta over o6q (`69c56a021..52925e2eb`). It covers the same 11 files, and the patch is identical once hunk offsets are ignored.
- The unified tree minus the merged branch (`52925e2eb`) is 17 files. Six are tests, three are bench scripts, and eight are source files.

**Code reading of the 8 source files, with every new variable unset and DP=1:**

- `tpu_communicator.py`: `_replicated_dp_group` returns `None` when the DP group has one rank. Every changed branch then returns the old expression. The packed gather is reached only through that DP path.
- `greedy_top_tokens.py`, `tpu_runner.py`, `eagle3.py`: `_dp_lockstep_enabled()` is `dp_size > 1 and enable_expert_parallel`, so it is False at DP=1. The target head then takes the old "fast" call. The draft owner is resolved with `False`, which is what it was passed before.
- `batched_rpa_decode.py`: lanes 0 returns the same kwargs as before.
- `fused_moe_ep_v3.py`: an empty override string returns the config unchanged. `tile_rows()` returns 128, which equals `TILE_ROWS`.
- `kernels/fused_moe/v3/layer.py`, `combine.py`, `config.py`: with `combine_real_tokens` and `sort_index_compare` off, `combine_tokens` equals `message_tokens` and the pair index is the same slice. `math.gcd(combine_unroll, combine_tokens_per_tile)` equals `combine_unroll`, because `Config` already rejects an unroll that does not divide the tile. `in_place=False` keeps the old combine path.
- v4 flags (`compact_message`, `reorder_chain`) off: `pallas_geometry(compact=False)` is the old geometry; `select_tokens` equals `message_tokens`, so the compact select branch is never taken; the transport finish reads `index[:LANES]` as before; the live-row mask is the old code, re-indented; the combine tile is unchanged, because `message_tokens` is a multiple of `lcm(128, tile)`. **One change remains with the flags off:** `run_kernel` is now traced after the combine's pair index instead of before it. The values are the same, but the HLO op order differs, so both layouts were gated again on this tree (below).
- `toy_proxy_server.py`: `_make_dp_router` returns `None` unless `PROXY_DECODE_ROUTING=dp_affinity`, and every new branch in the request handler is guarded by `dp_router is not None`. The test checks `round_robin`, `llmd`, `prefix_hash` and unset.

**Unit tests (fold, 2026-10-23, tree `7b45f3df2` against `8fd4dd025`)** ran in the same image and the same way. Only the 5 pre-existing failures listed below fail:

| Suite | Fold `7b45f3df2` | `8fd4dd025` |
|---|---|---|
| CURVE_README list (16 files) | 242 passed, 3 failed | 242 passed, 3 failed (the same 3) |
| Cherry-picked commits' tests (the 4 files below) | 115 passed (+1 defaults test) | 114 passed |
| New: `test_fep_v3_compact`, `test_toy_proxy_dp_rank` | 25 passed, 30 skipped (device-only, `P3_BENCH=1`) | absent |
| MoE adapters and fused MoE kernels | 298 passed, 92 skipped, 2 failed | 298 passed, 92 skipped, 2 failed (the same 2) |
| `tests/disagg` | 67 passed | 51 passed |

**Mutation checks (fold).** The control passed before and after. All 15 mutants were caught, and the tree was clean after each: compact threshold doubled (3 failed), compact flag ignored (4), compact always on (4); each of the four fused-EP flags defaulting on (1 each); dp gate always on (2), dp gate never on (2), the client's rank header loses (1), KV guard off (1), count guard off (1), `done()` not releasing (2), not sticky (1), the rank header also sent to prefill (2). `reorder_chain` and the compact slot slice need a TPU. Their evidence is the device mutants in `j-0b8fd32d` (M1, M2) and `j-88691da2` (M3), 30/30 device cases bit-identical, and engine token identity 176/176 in `j-cd93c1f9`.

**Unit tests (first push, `8fd4dd025`)** ran in the server image `vllm-torchtpu-dev@sha256:ba3be4a9…` on CPU under docker, with the tree mounted read-only as `/workspace/vllm-torchtpu`:

| Suite | Unified | Merged base `52925e2eb` |
|---|---|---|
| CURVE_README list (16 files) | 242 passed, 3 failed | 242 passed, 3 failed (the same 3) |
| Cherry-picked commits' tests (`test_tpu_communicator`, `test_batched_rpa_decode`, `test_fused_moe_ep_v3_config`, `test_greedy_top_tokens`) | 114 passed | 91 passed (the new tests are absent) |
| MoE adapters and fused MoE kernels (`test_fused_moe_ep`, `test_moe_runner`, `test_moe_routing`, `tests/kernels/fused_moe`) | 298 passed, 92 skipped, 2 failed | 298 passed, 92 skipped, 2 failed (the same 2) |

All five failures are pre-existing and fail identically on the base:

- The 3 megakernel failures: `test_qwen35_megakernel_buckets` `fake_step`, which does not accept `out_rows=`.
- `test_moe_routing.py::test_routing_path_compiles_fullgraph[sort|rowmax]`.

**Mutation checks.** Each check broke one guard in a scratch worktree, ran that commit's tests, and restored the tree. The no-edit control passed 114/114. All 12 mutations were caught:

- `2564cf137`: DP gather removed (6 failed); DP=1 guard removed (4 failed).
- `54a54831f`: packed gather never used (2 failed); packed gather on by default (3 failed).
- `cf54717bc`: target head always fast (1 failed); target head stock at TP1 (1 failed); draft lockstep passed through (1 failed).
- `4afe2bb67`: lanes env ignored (5 failed); lanes applied at 0 (5 failed).
- `933c26ed8`: overrides ignored (4 failed); unknown-field check off (1 failed); tile-rows env ignored (1 failed).

`combine_real_tokens` and `sort_index_compare` have no CPU unit test, because their kernels need a TPU. Their evidence comes from TPU runs:

- Kernel-level bit-exactness in `j-d52da5cb`.
- Engine token identity of 176/176 in `j-394f71be`.
- The engine microbench `j-50ecf56a`.

## GSM8K gates

| Tree | Layout | Job | Score | Reference |
|---|---|---|---|---|
| fold `7b45f3df2` (code of this tip) | DP4×TP2, v4 flags + `dp_affinity` + fast prefill load | `j-217490c2` | **97.73%** (se 0.41) | `j-334c6078` 97.80% |
| fold `7b45f3df2` | TP8+DCP4 1P1D (standard recipe) | `j-0cdcabb5` | **98.10%** (se 0.38) | `j-017298cb` 97.73% |
| `8fd4dd025` (first push) | DP4×TP2 candidate (rc + lc) + fast prefill load | `j-334c6078` | **97.80%** | `j-489694e3` 97.80% |
| `8fd4dd025` (first push) | TP8+DCP4 1P1D (standard recipe) | `j-017298cb` | **97.73%** | `j-e0de41ff` 97.57% |

Both fold gates mapped the frozen worktree at `7b45f3df2` (checked in the rendered recipe). In `j-217490c2` the proxy logged `Decode DP-rank routing: dp_affinity ranks=4`, and the router routed 256 requests across all 4 ranks. GSM8K prompts share no 4 KiB prefix, so every request took the cold path; the sticky path was not exercised by this gate. The sticky path matters on multi-turn traffic such as the AgentX serving runs below, but its hit rate there was not measured here. All gates use 1,319 samples, and strict-match equals flexible-extract in all of them.

The v4 flags were also gated on the candidate lineage before the fold: GSM8K 97.57% in `j-04690b1a` (frozen `klb-tree-v4`, fast prefill load).

## Best measured points on this code

These ran on the candidate lineage, not on this branch: `klb-tree-v4` / `klb-tree-v4-aff` are `a994fc7a3` plus the same v4 and `dp_affinity` commits that were cherry-picked here. This branch's own numerics check is the gate `j-217490c2` above.

- **Band A, 4P3D c560, v4 + `dp_affinity`:** 220.0 Mtok/$ @ p90 79.4 (`j-5f3c6943`); repeated 220.4 @ 79.4 (`j-ac25fda2`). The same-time v4 arm without `dp_affinity` was 214.4 @ 77.4 (`j-9a6c9516`).
- **1P1D c152, v4 + `dp_affinity`:** 211.2 Mtok/$ @ p90 96.3 (`j-c85b4fe5`).

## Flags per layout (decode leg; the prefill leg is identical across all three)

Cap = requests per decode host. Set `MAX_NUM_SEQS_DECODE`, `DECODE_TPU_SCHED_MK_MAX_RUNNING` and `PROXY_DECODE_MK_CEILING` together, as the table shows.

| Variable | TP8+DCP4, megakernel (bands C–H) | TP8+DCP4, band A E6 / 1P1D gate | DP4×TP2 (bands A–B) |
|---|---|---|---|
| `TENSOR_PARALLEL_SIZE_DECODE` | 8 | 8 | **2** |
| `DATA_PARALLEL_SIZE_DECODE` | 1 | 1 | **4** |
| `DECODE_CONTEXT_PARALLEL_SIZE_DECODE` | 4 | 4 | **1** |
| `CP_KV_CACHE_INTERLEAVE_SIZE_DECODE` | 128 | 128 | **1** |
| `DECODE_VLLM_TPU_QWEN35_MEGAKERNEL` | 1 | 1 | **0** |
| `DECODE_VLLM_TPU_QWEN35_MTP_MK` | 1 | 1 | **0** |
| `DECODE_VLLM_TPU_QWEN35_MK_MODE` | `tp` for caps 1–4, `ep` for caps ≥ 8 | `ep` | not read (MK off) |
| `DECODE_TPU_SP_MIN_ROWS` | 512 | 352 (E6) / 512 (gate) | **88** |
| `DECODE_TPU_MOE_FUSED_EP_STEP_MIN_TOKENS` | unset (64) | 44 (E6) / unset (gate) | 44 |
| `DECODE_TPU_EXPERT_PLACEMENT` (`/workspace/vllm-torchtpu/expert_placement/…`) | `…_tp8_ep8_agentx_corr_c64_mk2p.json` (C), `…_c32.json` (D–H) | `expert_placement_dp8_ep8.json` (E6) / `…_c32.json` (gate) | `expert_placement_dp8_ep8.json` |
| `MAX_NUM_SEQS_DECODE` = `DECODE_TPU_SCHED_MK_MAX_RUNNING` | cap | cap | **cap ÷ 4** (16 for cap 64, 28 for cap 112) |
| `PROXY_DECODE_MK_CEILING` | cap | cap | cap (the host total) |
| `COMPILE_SIZES_DECODE` | `4,…,384,448,512,2048` | without 448 (E6) / with 448 (gate) | `8,16,24,32,48,64,72,80,96,112,128,256,512,2048` |
| `TPU_TOKEN_BUCKET_EXTRA_DECODE` | unset | unset | `8,48,2048` |
| `KV_CACHE_MEMORY_BYTES_DECODE` | unset | unset | `39500000000` (passthrough, below) |
| `DECODE_TPU_KV_BYTES_EXPLICIT` | unset | unset | `1` (passthrough) |
| `PROXY_*_ROUTING` | `llmd` (D–H), `prefix_hash` (C) | `prefix_hash` (A) / `llmd` (gate) | `prefix_hash` |
| `DECODE_TPU_RPA_DECODE_V2` | 1 | 1 | 1 (a no-op at DCP 1) |
| **DP4×TP2 candidate:** `DECODE_TPU_RPA_DECODE_LANES` | unset | unset | `4` |
| `DECODE_TPU_DP_PACKED_ROUTER_GATHER` | unset (a no-op at DP 1) | unset | `1` |
| `DECODE_TPU_GDN_FUSED_1DOT` | unset | unset | `1` |
| `DECODE_TPU_MOE_FUSED_EP_V3_CONFIG` | unset | unset | `{"combine_real_tokens":true,"sort_index_compare":true,"compact_message":true,"reorder_chain":true}` (v4) |
| `PROXY_DECODE_ROUTING` | as `PROXY_*_ROUTING` above | as above | `dp_affinity` (pins the DP rank; `PROXY_DP_RANKS` defaults to 4) |
| `PREFILL_LOAD_FORMAT`, `PREFILL_SAFETENSORS_LOAD_STRATEGY` (prefill leg; recipe passthrough below) | unset in the TP8 gates (Runai load); `safetensors`, `prefetch` recommended | same | `safetensors`, `prefetch` (gated) |

The candidate variables have been measured only on DP4×TP2. On TP8, lanes, GDN 1-dot and the fused-EP overrides would change real kernels, and they have not been gated there. Leave them unset for TP8.

**Never set `scatter_chunk_pairs=128` with `scatter_ring=8`** in `TPU_MOE_FUSED_EP_V3_CONFIG`. That pair hangs the fused EP kernel at 8192 global rows (1024 per shard, the 2048-row SP bucket that engine warmup compiles). The hang was found in `j-d52da5cb` and hung the engines of `j-1c54d05a` and `j-87e1aff5`.

Everything else in the decode environment is common to all three layouts. It is listed in `CURVE_README.md` ("Env common to every point" and the upstream knob set).

## Recipe passthroughs

`agentx-397b-pd-baseline` on cloud-devkit-recipes master does not pass the variables below. They live in the private clone `~/claude/cloud-devkit-dp4tp2` (`4b1623bd`, `4b3470f9`, `ca4a447e`). Each is opt-in, and an unset variable renders unchanged.

| Variable | Recipe effect | Needed by |
|---|---|---|
| `KV_CACHE_MEMORY_BYTES_DECODE` | `--kv-cache-memory-bytes` on decode | DP4×TP2 (the TP2 decode KV pin) |
| `TPU_KV_BYTES_EXPLICIT` (as `DECODE_…`) | env passthrough | DP4×TP2 |
| `BENCH_MEMORY_REQUEST` | memory request on `sidecar-bench`; use 48–64Gi on hour-long replays to avoid an OOM-kill | DP4×TP2 serving and LongBench |
| `TPU_RPA_DECODE_LANES`, `TPU_DP_PACKED_ROUTER_GATHER`, `TPU_GDN_FUSED_1DOT`, `TPU_MOE_FUSED_EP_V3_CONFIG` (as `DECODE_…`) | env passthroughs | DP4×TP2 candidate |
| `PREFILL_LOAD_FORMAT`, `PREFILL_SAFETENSORS_LOAD_STRATEGY` | per-leg `--load-format` and `--safetensors-load-strategy` on the prefill only | fast prefill load (any layout) |

**Fast prefill load.** With `PREFILL_LOAD_FORMAT=safetensors PREFILL_SAFETENSORS_LOAD_STRATEGY=prefetch`, each PP8 prefill rank reads only its own stage's tensors (54.6 GB) instead of streaming the whole ~379 GB checkpoint through Runai and dropping 7/8 of it. On 2026-10-23 that took the prefill load from over 2 h to about 40 min under shared-disk contention. The two passthroughs live in the private recipe clone `~/claude/cloud-devkit-pfload` and are not yet in cloud-devkit-recipes; they will be committed with the recipe later. The fold's DP4×TP2 gate and the band A / 1P1D points above all used it.

`CURVE_README.md` lists the TP8-side passthroughs that master drops: `PREFILL_TPU_RPA_PREFILL_BLOCKS`, `PREFILL_VLLM_TPU_LIGHTWEIGHT_STEP_STATS` and `LIBTPU_EXTRA_PREFILL`.

## Launch lines

Set `TREE` to a clone of this branch.

- **TP8+DCP4.** Use the commands in `CURVE_README.md` with `--map-dir "$TREE":*:/workspace/vllm-torchtpu`. The 1P1D GSM8K gate is `j-e0de41ff`'s command with only `--map-dir` and `BENCH_NAME` changed.
- **DP4×TP2.** Start from the TP8 command of the same shape. Replace the decode variables with the DP4×TP2 column above, add the four candidate variables, and render through a recipe that has the passthroughs. The gated 1P1D form is `j-489694e3`'s command (`cdk job desc j-489694e3-fa94-4d6a-97f6`) with `--map-dir "$TREE":*:/workspace/vllm-torchtpu`, the v4 `DECODE_TPU_MOE_FUSED_EP_V3_CONFIG` above, `PROXY_DECODE_ROUTING=dp_affinity`, `PREFILL_LOAD_FORMAT=safetensors` and `PREFILL_SAFETENSORS_LOAD_STRATEGY=prefetch` added; that is the fold's gate `j-217490c2`.

## Known limits

- DP>1 with DCP>1 on a P/D KV consumer raises `NotImplementedError`. The Raiden Stage-3 admission names the layout `dp{N}tp{T}_decode`, with no DCP axis.
- DP4×TP2 loses to TP8 as served (megakernel up to 28 per host, XLA from 32) below about 31 requests per decode host. Same-harness step times, DP4×TP2 vs TP8 as served (2026-10-23, real weights, ctx 98,560): +58.5% at 8, +26.8% at 16, +3.6% at 28, −1.0% at 32, −9.3% at 64, −13.8% at 112; the cross is where TP8 leaves the megakernel (128 rows > its 112-row limit).
- Under DP4×TP2, each request's KV lands on 2 cores instead of 8. The median KV transfer is therefore 1.7–2.6× slower, and median TTFT is 0.1–0.3 s higher.
- R4 (HR + hx collectives) and the MK 112-row buckets are on by default. They live inside the megakernel, which DP4×TP2 turns off.

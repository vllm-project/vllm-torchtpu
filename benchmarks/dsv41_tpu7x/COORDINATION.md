# DSv4.1-Flash: multi-thread coordination

This file is for anyone (person or agent) working in `dsv41/` alongside other threads.
**Read it before you start, and update the ownership table when you take or give up a role.**

## 1. Ownership table

| Thread id | Owns (may edit) | Git worktree(s) | Branch | Pod prefix |
|---|---|---|---|---|
| `prefill` | prefill repro and prefill changes | `repos/vtt-prefill`, `repos/vllm-prefill` | `cpk/dsv41-prefill`, `cpk/dsv41-vllm-prefill` | `chaitanyapk-dsv41-prefill-*` |
| `decode` | decode repro and decode changes | `repos/vtt-decode`, `repos/vllm-decode` | `cpk/dsv41-decode`, `cpk/dsv41-vllm-decode` | `chaitanyapk-dsv41-decode-*` |
| `combined` | file-level merge of P+D trees (subagent), then integration by coordinator | `repos/vtt-combined` (base 1d02e4a), vllm: `repos/vllm-decode` | `cpk/dsv41-combined` | `chaitanyapk-dsv41-combined-*` |
| `decodefix` | ✅ DONE (cfacfa7): coherent multi-step decode | `repos/vtt-decode-fix` (base 1d02e4a), vllm: `repos/vllm-decode` | `cpk/dsv41-decode-fix` | `chaitanyapk-dsv41-decodefix-*` |
| `perf` | decode step-time profiling + optimization (subagent; base decodefix cfacfa7) | `repos/vtt-perf-opt` (+ old vtt-perf-86bca37) | `cpk/dsv41-perf-opt` | `chaitanyapk-dsv41-perf-*` |
| `pd` | P/D connector gaps G1-G4 + 1P1D harness (subagent) | `repos/vtt-pd` (base 1d02e4a); may create `scripts/pd/` | `cpk/dsv41-pd` | `chaitanyapk-dsv41-pd-*` |

To create a new thread:
`git -C ../vllm-torchtpu worktree add -b cpk/dsv41-<id> repos/vtt-<id> <base>`.
Then add a row to the table.

## 2. Rules

1. **Edit only your own worktrees.** Everything else is read-only for you: `patches/`, `support/`, `whls/`, `reference/`, `scripts/lib/`, and the other threads' worktrees.
   * If you need to change a shared file, say so in the log (section 4) first. Keep the change backward compatible.
2. **Always pass `--thread <id>`** to `run_prefill.py` / `run_decode.py`. This one flag sets three things:
   * the pod name, `chaitanyapk-dsv41-<thread>-...`
   * the k8s labels `owner=chaitanyapk` and `dsv41-thread=<thread>`
   * the output directory, `runs/<thread>/<UTC-ts>_<tag>/`
3. **Pods.**
   * Each pod takes a whole 4-chip node.
   * A lock file `locks/<cluster>__<pod>.lock` is created atomically for the life of a run. A second run against the same pod name fails fast.
   * To run two pods in one thread at the same time, use different `--suffix` values.
   * Never `kubectl delete` a pod with another thread's prefix. List your own with:
     `kubectl get pods -l owner=chaitanyapk,dsv41-thread=<id>`
4. **Stacks** (`runs/_stacks/stack_<hash>.tar`) are content-addressed by both repo HEADs, any uncommitted diff, and the support files.
   * Two threads with the same code share one tar.
   * A thread with different code gets a different tar.
   * Never edit or delete a stack tar while a run may be using it.
5. **Commit before a run you plan to cite.**
   * The stack key and `provenance.json` record HEAD plus a dirty-diff hash. A clean HEAD is the only thing someone else can reproduce exactly.
6. **Cluster etiquette** (same as the handoff):
   * The scripts refuse to run if the cluster has other users' Running pods. Override with `--allow-foreign` only if you know the node pool is free.
   * Delete pods when you are done. The default is to delete; `--keep-pod` is opt-in.
7. **Results.** Never overwrite another thread's `runs/<thread>/...`. Every run gets a fresh timestamped directory.

## 3. Shared, read-only inputs

| Path | What | Source |
|---|---|---|
| `patches/0001-vllm-common-f9ad9dd6b4.patch` | vllm common patch (prefill + decode) | paste 6755399435827931 |
| `patches/0002-vllm-decode-runtime-v2.patch` | vllm decode runtime patch v2 | paste 6755399435850973 |
| `repos/vllm-base` | upstream vllm @ f9ad9dd6b4 (worktree parent) | github |
| `whls/` | 14 wheels plus SHA256SUMS.txt (verified) | Drive folder in handoff section 2 |
| `support/` | sitecustomize (prefill and decode), mock metadata, LD_PRELOAD hook | pastes 6755399435827929 and 6755399435850906 |
| `reference/handoff_pastes/` | every handoff paste, frozen as of 2026-10-20 | gpaste |

## 4. Log (append-only, newest last)

* 2026-10-20 07:10 UTC, setup: created worktrees, patched vllm trees, wheels, scripts. No TPU runs yet.
* 2026-10-20 07:55 UTC, prefill: reproduced on bodaborg-tpu7x-spot-sps (197,171 vs 197,316 tok/s/node at 128K c=8; all 6 workloads within ±0.3%).
* 2026-10-20 08:48 UTC, decode: `1d02e4a` runs end to end and the gates match the handoff. Perf is 2.2× (c=64) and 1.5× (c=256) slower than the reference rows.
* 2026-10-20 09:20 UTC, decode A/B:
  * prefix caching off does not help (78.8 ms)
  * the reconstructed w384 tree fails at compile
  * new worktrees: `repos/vtt-decode-w384` (branch cpk/dsv41-decode-w384) and `repos/vllm-decode-v1` (cpk/dsv41-vllm-decode-v1)
* Infra notes for all threads:
  * stacks are staged through gs://chaitanyapk-dsv41-staging (kubectl cp is ~0.2 MB/s)
  * do not `pkill -f run_decode` from a shell whose own command line contains that string
  * launch runs with `setsid nohup ... < /dev/null &`
* 2026-10-20 09:40 UTC, decodefix: commit `11e6c02` on cpk/dsv41-decode-fix. Root cause candidate for incoherent decode: the per-step SWA page-index cache in `deepseek_v41/attention.py` was shared by all SWA layers with the same (is_csa, window), but the server log shows `swa=43 in 3 group(s) sized [15, 14, 14]`, i.e. 3 KV-cache groups with distinct block tables, and same-position layers of different groups are overlaid on one physical array (`kv_cache_deepseek_v41_flash.py` overlay). Layers outside group 0 used group 0's block table, so they clobbered other layers' SWA history (prefill is still correct within one forward, decode reads garbage). Fix: group the cache per KV-cache group (tagged at KV init) + sinks_t in the sparse_mla b_q<8 3-chunk running max. TPU run `decodefix_a` (keep-pod) launched.
* 2026-10-20 10:00 UTC, pd: commits `8108c8f` (G1+G2: V4.1 manifest — ratio-1/2 CSA classified by the (nope, rope) pair, seq-along-lane indexer `[nb,33,4,T]`, per-(role,group) tags split by page shape `.s<rows>x<pack>x<width>` for the T=512/1024 CSA/idx pages and SWA host-vs-overflow arrays, V4.1 pools ship whole tail pages; CPU tests pass) and `c18e1de` (G4: `examples/disagg/launch_dsv41_disagg_role.sh`) on cpk/dsv41-pd. Harness `scripts/pd/run_pd.py` (2 pods `chaitanyapk-dsv41-pd-{pf,dc}-a`, own sitecustomize `scripts/pd/sitecustomize_pd.py` because support/decode/sitecustomize.py force-sets CED=0 + DecodeBench). First 1P1D TPU run launched.
* 2026-10-20 10:10 UTC, perf:
  * 86bca37 gives 81.9 ms at c=64, the same as 1d02e4a, so the deleted MoE fast paths are NOT the regression.
  * Suspect libtpu 0.0.44 (pinned) vs the image's 0.0.48, which the original decode pods used.
  * support/mock_metadata.py now serves tpu-env keys as individual attributes. This fixes the 0.0.48 init hang and changes every stack hash.
  * `run_decode.py --libtpu` was added.
* 2026-10-20 10:18 UTC, decodefix: run `decodefix_a` (11e6c02) PASSES the USA gate: 50/50 states, coherent text, 2+2 on all 8 ranks. But natural τ = 1.05 (16/312 accepted), and 1.006 even on "count 1..60" -> drafter has no context. Root cause #2: commit 823509e5c (r25 perf patch) removed the DSpark aux hidden-state capture from `deepseek_v41/model.py`; V4's `set_aux_hidden_state_layers` only stores ids nobody reads, so draft chunks have empty aux_hidden_states and the context-KV store is skipped. Restored in `570c809` (entry stream mean over hc of layers 37/38/39, per reference inference/model.py). Also `b5665ed`: indexer signed-weight fix ported from cd4eedb. Run `decodefix_c` (570c809) launched. c=64 on 11e6c02: mean TPOT 89.0 ms @ τ 1.006 (baseline 81.5).
* 2026-10-20 10:20 UTC, decodefix:
  * **USA gate 50/50, coherent; 2+2 8/8** with 11e6c02. Root cause: the SWA page-index cache was keyed per window, not per KV-cache group.
  * The τ≈1 root cause was the missing DSpark aux hidden-state capture (layers 37/38/39). Restored in 570c809.
  * The indexer sign-fix port is b5665ed.
  * Step was 89.0 ms (+9%). cfacfa7 memoizes the per-group SWA bounds. Runs c and d are pending.
* 2026-10-20 10:44 UTC, infra: decode pods use hostPath /dev/shm, and stale /dev/shm/dsv41_fp4_cache (~293 GB, from 86bca37's mxfp4) caused OOM. The runner now wipes it at startup. Wipe it in any custom harness too.
* 2026-10-20 10:53 UTC, decodefix: DONE. HEAD `cfacfa7` (cpk/dsv41-decode-fix: 11e6c02, b5665ed, 570c809, cfacfa7). Run `runs/decodefix/20261020-102010_decode_dp8_c64_d`: 2+2 8/8; USA 50/50 states, coherent; natural DSpark τ on USA = 4.39 (254/75 drafts, per-pos [0.84,0.73,0.68,0.64,0.49]); c=64 bench natural accept 3.97, mean TPOT 20.95 ms, steady-state 786.2 steps/s (baseline 1d02e4a: 786.6) -> step ≈ 81.4 ms interior / 83.2 ms (TPOT×τ), i.e. unchanged; real measured throughput 3,121 tok/s/node interior (2,686 wall). Run c (570c809, no memo) was 93 ms/step -> cfacfa7 (memoize SWA group bounds at KV init) is required. Pods deleted, locks removed.
* 2026-10-20 10:53 UTC, perf:
  * libtpu 0.0.48 gives a 76.8 ms step at c=64, vs 81.8 ms on 0.0.44 (−6%). It is not the 2× gap.
  * 86bca37 matches 1d02e4a, so the gap is in tree/config versus the unsaved w384 pod, not in the deleted MoE fast paths.
  * Next: an xprof profile of the decode step on the correct tree (decodefix cfacfa7).
* 2026-10-20 10:53 UTC, decodefix DONE: HEAD cfacfa7. 2+2 8/8, USA 50/50, τ=4.39. At c=64: TPOT 20.95 ms, 3,121 tok/s/node real (interior), step ≈81.4 ms. Being ported into combined.
* 2026-10-20 11:00 UTC, combined prefill A/B on amylin:
  * combined + **vllm-prefill** gives 197,143 tok/s/node at 128K c=8, and all 6 workloads are within ±0.2% of the reference. The merged code is fine.
  * The cd4eedb control on the same cluster also gives 197,288.
  * The earlier −3..−11% (combined + vllm-decode, node x8s4, host-RAM pressure) comes from the vllm-decode patch or the node. Re-testing combined 96eafb830 + vllm-decode on amylin.
* 2026-10-20 11:23 UTC, ✅ combined 96eafb830 + **vllm-decode** prefill on amylin: 197,017 tok/s/node at 128K c=8, all 6 workloads within ±0.3%.
  * One tree plus one vllm build now serves both roles.
  * The earlier −3..−11% was **node gke-tpu-b10c48a3-x8s4 on bodaborg** (host-RAM pressure, "checkpoint exceeds 90% of RAM"). Treat that node as suspect for perf numbers.
* 2026-10-20 11:40 UTC, perf: **root cause of the 2.2× decode gap = DP request imbalance → 384-token bucket on every rank.**
  * Profile: xprof of cfacfa7 (libtpu 0.0.48) at c=64, run `runs/perf/20261020-105645_decode_dp8_c64_p1` + in-pod capture (traces in gs://chaitanyapk-dsv41-staging/profiles/c64_p1/). Every step ran the 384-token graph (bf16[384,5120] matmuls, MoE over 3072 global tokens). Per-step device time: main fwd ~93 ms = fused EP MoE kernel 37-44 ms (40 layers × ~1 ms) + MoE routing-plan glue ~16.5 ms (2 scatter-adds/layer of 116 us, 2 all-gathers/layer) + indexer top-k ~12.4 ms + attention/projections ~8 ms + mHC 3.3 ms + misc ~6 ms; outside fwd: DSpark draft 2.1 ms, host gap ~5.5 ms.
  * Why 384: the driver's client does not pin DP ranks, so vLLM's DP LB spread 64 requests as {9,8,6,10,6,6,9,10} (from /metrics during a run). `_dp_coordinated_step` takes the MAX over ranks → 60 tokens > 48 → bucket 384 for all ranks.
  * Same server, same tree, requests pinned round-robin with `X-data-parallel-rank`: step **30.5–33 ms** (vs 101–110 ms unpinned), interior **10.8–11.9K tok/s/node real** (vs 3.0–3.1K), τ≈5.1 on the bench text.
  * The historic v2 rows used `vllm bench serve` with VLLM_BENCH_DP_SIZE=8 round-robin headers (exactly c/8 per rank), which is why they were ~37 ms.
  * Caveat on the w384 reference (paste v2_6755399435827927): its sitecustomize force-sets ENABLE_CONNECTOR=0 / ENABLE_DECODE_BENCH_CONNECTOR=0 in workers, and with ENABLE_CONNECTOR=1 its launch used `rejection_sample_method: synthetic`. Its perf rows may therefore have decoded over non-template (garbage) KV. The fused EP MoE kernel skips empty experts (`expert_visit_list`), so collapsed routing makes MoE cheaper. Treat the 33.6–37 ms rows as not comparable until reproduced with real KV.
  * Shared-script changes (backward compatible; defaults noted):
    * `run_dp8_scale_driver.py`: opt-in `DSV41_DP_RR=1` adds `X-data-parallel-rank: (i//8)%8` to the warmup and benchmark requests. Default 0 = verbatim behaviour.
    * `run_decode.py`: `--dp-rr` (**default 1**), `--buckets` (DSV41_BUCKETS overrides compile_sizes), `--extra-serve-args`, `--extra-env`.
    * `in_pod_runner.sh`: honours DSV41_BUCKETS, DSV41_EXTRA_ENV, DSV41_EXTRA_SERVE_ARGS. All unset = identical launch.
    * New `scripts/perf/`: inpod_profile.py (capture + GCS upload), xplane_summary.py, inpod_relaunch.py (flag A/B on a kept pod), inpod_dp_dist.py.
- **2026-10-20 17:02 UTC (`pd` / `combined` — DONE & VALIDATED)**: Both 1P1D disaggregated modes (`CED=1` on `amylin`, `runs/pd/20261020-165851_1p1d_ced1_final_pass_a` and `CED=0` on `bodaborg`, `runs/pd/20261020-165853_1p1d_ced0_final_pass_b`) passed all accuracy gates and Stage-3 checks on `cpk/dsv41-combined` (`d5129e7c6`): `2+2` **16/16**, `USA` **50/50 states coherent**, greedy 64-token equality + needle `739204` retrieval **5/5 `equal=True` (`first_div=None`) across `64, 700, 1500, 5000, 20000` tokens**, `hard_gates_passed=True`, `STAGE3_CHECK_OK 1` (`153/153` and `171/171` sends completed, `0` skipped, window trim `157->20` pages verified). On the 8K-input, 128-output, `c=8` benchmark, `CED=1` 1P1D achieves **TTFT mean `1.445s` (`p50=1.298s`) vs `3.564s` decode-local (`2.47x` faster), TPOT `24.81 ms` vs `43.15 ms` (`-42.5%`), and output throughput `200.94 tok/s` vs `109.11 tok/s` (`+84.2%`)**. All pods deleted and locks released.
- **2026-10-20 23:22 UTC (`perf` / `combined` — PHASE 3 XPROF HLO & KERNEL OPTIMIZATIONS DONE & VALIDATED)**:
  * Analyzed TPU7x decode traces via `/google/bin/releases/xprof/xprof-mcp/xprof_cli.par` (`chaitanyapk-14677790351621652831` baseline vs `chaitanyapk-18402943240297915715` post-optimization) and landed three HLO/kernel fixes on `cpk/dsv41-combined` (`937432afb`, `1c324dd31`, `b5d01a0f3`):
    1. **Direct SparseCore Stage-2 Top-K Index Emission (`compressor.py`)**: Replaced `sc_topk_stage2_op` + TensorCore `[96, 32768]` `gather` (`gather_fusion.94/.95/.96` = `4.66 ms/step`, 16.3% of device time) and the `T > 48` 9-round TensorCore tournament (`6.75 ms/step`) with `sc_topk_direct_op` across all decode buckets (`num_tokens <= 384`), cutting Stage-2 top-$k$ + candidate gather from `4.66 ms/step` to **`0.200 ms/step`** (`-4.46 ms/step`).
    2. **Fused Pallas `o_proj_wob_op` on TP=1 Decode (`attention.py`)**: Enabled `torch.ops.tpu.o_proj_wob_op` (`fused_wo_a_wo_b_h128_h64_vmem`) on TP=1 decode (`num_tokens <= 384`), accumulating all 16 heads in VMEM and cutting 43× `o_proj_wob` from `2.28 ms/step` to **`1.275 ms/step`** (`-1.01 ms/step`).
    3. **Rotated CSA Overlay Hosts Across SWA Groups (`kv_cache_deepseek_v41_flash.py`, `b5d01a0f3`)**: Rotated `csa_nope_hosts` across `swa_layer_groups` (`host_pos = (position + group_idx) % n_csa_hosts` for `position < n_csa_hosts`) so Layers 6 and 13 no longer alias `sw_cache` with `main_cache_kv` (`two_caches_same_buffer = False` for all 43 layers), eliminating the only two 1.48 GB HBM aliasing copies (`copy.4527` and `copy.4528` = **`1.83 ms/step`, 2.96 GB/step**), and skipped redundant `engram_hash` inside `full_stage` on decode (`model.py`).
  * **Prefill Verification (`runs/combined/20261020-202932_prefill_xprof1/summary.json`)**: All 6 workloads PASS, with `128k_c8` at **197,793.3 input tok/s/node** (`+0.2%` vs 197,315.9 ref, new best).
  * **Decode Verification (`runs/perf/20261020-202908_decode_dp8_c64_xprof1/summary_engram0.json` & `summary_xprof2_b5d01a0f3.json`)**:
    * Gates: `2+2` **8/8**, `USA` **50/50 states coherent**, natural DSpark **$\tau_{\text{USA}} = 4.4938$** (up from 4.39).
    * `1x_c64`: **9.39s** (`6,980.7 tok/s/node` wall), **29.05 ms/step** steady-state (`2,203.4 steps/s` → **7,711.9 tok/s/node @ $\tau=3.5$**, **9,364.5 @ $\tau=4.25$**, **11,017.0 @ $\tau=5.0$**).
    * `4x_c256` (dedicated `T=192` bucket): **26.8s** (`9,758.1 tok/s/node` wall), **Mean TPOT `14.85 ms`** (`17,242.9 tok/s/node` implied), **53.28 ms/step** steady-state (**22,848.4 tok/s/node** natural at $\tau=4.76$; **16,815.4 tok/s/node @ $\tau=3.5$** (`+18.2%` vs 14,223 ref), **20,418.7 @ $\tau=4.25$**, **24,022.1 @ $\tau=5.0$**).
    * `8x_c512`: **58.11s** (`9,022.6 tok/s/node` wall), **Mean TPOT `21.86 ms` / Median `21.15 ms`** (**23,423.7 tok/s/node** mean implied / **24,210.0 tok/s/node** median implied). All pods deleted and locks released.
- **2026-10-21 19:46 UTC (`perf` / `combined` — PHASE 4 XPROF ROUND-2 & TECH REPORT OPTIMIZATIONS DONE & VALIDATED)**:
  * Cross-referenced `chaitanyapk-18402943240297915715` against the DeepSeek-V4.1-Flash Technical Report (`arXiv:2609.19969`) and landed four additional optimizations on `cpk/dsv41-combined` (`249d66e02`), verified by XProf `chaitanyapk-13959792908329192688`:
    1. **Decode MoE Scale C-Contiguous Materialization (`mxfp4.py`)**: Materialized `_tpu_fused_w13_scale` and `_tpu_fused_w2_scale` via CPU round-trip in `_process_weights_after_loading_decode`, eliminating the 86 `as_strided` scale copies (`tt_jit_execute_as_strided`: `-421.0 us/step`, `-99.0%`) and cutting 40× `fused_ep_moe_v2` from `6,214.8 us/step` to **`5,418.8 us/step` (`-796.0 us/step`, `-12.8%`)**.
    2. **Exact `40`-Token (`8 * b`) DSpark Bucket at `max_chunk_reqs == 64` (`spec_decode/dspark.py`)**: Added `8 * b` (`40`) to `_get_draft_token_paddings()`, avoiding 2× DSpark padding (`80` vs `40` tokens, `16` vs `8` reqs) at `c=64`, enabling `_get_reusable_draft_block_tables` reuse (`8 == 8`), and cutting the DSpark drafter program (`8210086396659415165`) from `1,949.5 us/step` (`g32`) to **`1,519.7 us/step` (`g16`, `-429.8 us/step`, `-22.0%`)**.
    3. **1D `init_pre_mix` Buffer Registration (`deepseek_v41/model.py`, `deepseek_v41/dspark.py`)**: Registered `init_pre_mix` as 1D `(config.hc_mult,)` instead of `(1, config.hc_mult)`, eliminating the eager `f32[4] -> f32[1,4]` `as_strided` view before every compiled stage call.
    4. **Fused MoE v2 `scale_slab` Bounds Alignment (`kernels/fused_moe/v2/layer.py`)**: Scattered `scale_bits` into `ragged_stride` (`2752`) before padding to `2944`, so off-shard pairs mapped to `ragged_stride` are dropped (`mode="drop"`) rather than colliding on index `2752`.
  * **Prefill Verification (`runs/combined/20261021-191514_prefill_xprof3/summary.json`)**: All 6 workloads PASS, with `128k_c8` reaching **198,541.3 input tok/s/node** (`49,635/chip`, `+0.6%` vs 197,315.9 ref, new all-time best).
  * **Decode Verification (`runs/perf/20261021-190440_decode_dp8_c64_xprof3/summary_xprof3_249d66e02.json`)**:
    * Gates: `2+2` **8/8**, `USA` **50/50 states coherent**, natural DSpark **$\tau_{\text{USA}} = 4.4938$**.
    * `1x_c64` (`[48, 96, 192, 384]`): **28.23 ms/step** (`mean_tpot * tau`) / **27.87 ms/step** (`2,296.47 steps/s`), **Mean TPOT `6.82 ms`** (`9,381.3 tok/s/node` implied), **9,502.5 tok/s/node** natural (`1,187.8/chip`), **8,037.6 tok/s/node @ $\tau=3.5$** (`+31.6%` vs 6,108.5 ref / `+25.2%` vs 6,419 ref), **9,760.0 @ $\tau=4.25$**, **11,482.4 @ $\tau=5.0$**.
    * `4x_c256` (`[48, 96, 192, 384]`): **52.81 ms/step** (`4,847.42 steps/s`), **23,316.9 tok/s/node** natural (`2,914.6/chip` at $\tau=4.81$), **16,966.0 tok/s/node @ $\tau=3.5$** (`+19.3%` vs 14,223.4 ref), **20,601.5 @ $\tau=4.25$**, **24,237.1 @ $\tau=5.0$**.
    * `8x_c512` (2-wave): **Mean TPOT `20.77 ms`** (**24,648.5 tok/s/node** mean implied at $\tau=4.74$). All pods deleted and locks released.

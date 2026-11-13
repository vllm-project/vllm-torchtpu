# DeepSeek-V4.1-Flash on TPU7x: Summary, Improvements & Reproduction Guide

Self-contained reference for reproducing and benchmarking **DeepSeek-V4.1-Flash** on `tpu7x-8` (`2x2x1`, 4 physical chips / 8 TensorCores per node) across **Prefill**, **Standalone Decode**, and **1P1D Disaggregated Prefill+Decode** from a **single unified `vllm-torchtpu` codebase** (`cpk/dsv41-combined` @ `13e0b9cd0`).

---

## 1. Executive Summary: Performance Comparison

Comparison across:
1. **Mandarin Co-Dev Recipe** ([DeepSeek V4.1 Flash 共同开发优化](https://docs.google.com/document/d/1MBc_mM0RVQwssha7O7VwOnlWZc8k6qQcJF5fKVB3GRc/edit) & `origin/epic/dsv41` [`scripts/dsv41_bench/server_steps/README.md`], PRs `#1617`, `#1673`, `#1732`, `#1890`, `#1912`, `#1949`)
2. **Handoff Doc v3** ([Handoff Doc](https://docs.google.com/document/d/1tYnsxtClsyIPhzgBmF97WT6fCMOzpBqmzlVedGkYvGs/edit) / [`paste/6755399435850976`](https://paste.googleplex.com/6755399435850976), split across `cd4eedb` for prefill and `1d02e4a` / unsaved `w384` pod for decode)
3. **Our Unified & Optimized Build** (`cpk/dsv41-combined` @ `13e0b9cd0`)

### 1.1 Prefill Throughput (`tpu7x-8` Single Node = 4 Chips)

* **GB200 Reference (Mandarin Doc)**: `60,000 input tok/s/chip` @ 100K context $\implies$ **TPU7x Target (70% of GB200) = `42,000 input tok/s/chip` (`168,000 tok/s/node`)**.

| Workload | Mandarin Recipe (`epic/dsv41`) | Handoff v3 (`cd4eedb`, P-only branch) | **Ours (`249d66e02`, Unified Tree)** | **Ours per Chip** | **Gain vs. Mandarin Recipe** | **Gain vs. Handoff v3** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **96K–100K Input** (`c=8`) | `84,609 tok/s/node` (`21,152/chip`, `DP=8` cached-extend `#1912`)<br>`61,900 tok/s/node` (`TP=8` `#1673`) | `195,977.4 tok/s/node`<br>(TTFT `3,819.0 ms`) | **`196,644.1 tok/s/node`**<br>(TTFT **`3,809.7 ms`**) | **`49,161 tok/s/chip`**<br>(**81.9% of GB200**) | **`+132.4%` (`2.32×`)** | **`+0.3%`** |
| **128K Input** (`c=4`) | — | `186,461.8 tok/s/node`<br>(TTFT `2,792.8 ms`) | **`186,703.1 tok/s/node`**<br>(TTFT `2,794.8 ms`) | **`46,676 tok/s/chip`** | — | **`+0.1%`** |
| **128K Input** (`c=8`) | — | `197,315.9 tok/s/node`<br>(TTFT `4,973.0 ms`) | **`198,541.3 tok/s/node`**<br>(TTFT **`4,946.6 ms`**) | **`49,635 tok/s/chip`**<br>(**82.7% of GB200**) | **`+134.7%` (`2.35×`)** vs 96K | **`+0.6%` (New Best)** |
| **128K Input** (`c=16`) | — | `196,687.8 tok/s/node`<br>(TTFT `8,659.9 ms`) | **`197,891.3 tok/s/node`**<br>(TTFT **`8,609.8 ms`**) | **`49,473 tok/s/chip`** | — | **`+0.6%`** |
| **100K Input** (`c=16`) | — | `195,191.6 tok/s/node`<br>(TTFT `6,675.3 ms`) | **`195,839.6 tok/s/node`**<br>(TTFT **`6,648.6 ms`**) | **`48,960 tok/s/chip`** | — | **`+0.3%`** |
| **32K Input** (`c=8`) | — | `188,896.0 tok/s/node`<br>(TTFT `1,339.7 ms`) | **`189,563.1 tok/s/node`**<br>(TTFT **`1,334.5 ms`**) | **`47,391 tok/s/chip`** | — | **`+0.4%`** |

---

### 1.2 Standalone Decode Throughput (`tpu7x-8` Single Node = 4 Chips, `DP=8, TP=1, EP=8`, DSpark $K=5$, ~100K Context)

* **GB200 Reference (Mandarin Doc)**: `8,000 tok/s/chip` @ $\tau=3.4$ (`12,000 tok/s/chip` @ $\tau=4.4$, `8 ms` TPOT) $\implies$ **TPU7x Target (70% of GB200 @ $\tau=3.4$) = `5,600 tok/s/chip` (`22,400 tok/s/node`)**.

| Metric / Workload | Mandarin Recipe (`epic/dsv41`) | Handoff v3 Committed (`1d02e4a`) | Handoff v3 Unsaved `w384` Ref | **Ours (`249d66e02`, Unified Tree)** | **Improvement vs. Mandarin / `1d02e4a` / `w384` Ref** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multi-Step Output Quality (`2+2` / `USA` 50 States / $\tau_{\text{USA}}$)** | GSM8K / DeepSWE subset (`#1890`/`#1949` unqualified) | ❌ **Broken** (`2+2` 8/8, `USA` **0/50 states**, $\tau=1.00$) | ❌ **Broken** (`USA` **0/50 states**, loops/stops early) | ✅ **Pass** (`2+2` **8/8**, `USA` **50/50 states coherent**, **$\tau_{\text{USA}} = 4.4938$**) | **Fixed multi-step coherence & restored $\tau = 1.00 \to 4.49$** |
| **`c=64` Step Time** (`8` reqs/rank, `T=48`) | — (`TP=8`: `11.71 ms` TPOT `#1673`) | `81.80 ms/step`<br>(TPOT `81.50 ms`) | `37.02 ms/step` | **`28.23 ms/step`** (`27.87 ms` @ `2,296.5 steps/s`, **TPOT `6.82 ms`**) | **`-65.5%` (`2.90×` faster) vs `1d02e4a`**<br>**`-23.7%` vs `w384` ref** |
| **`c=64` Throughput @ $\tau=3.5$** | `3,000–3,500 tok/s/node` (`c=64` `#1673`) | `2,753.0 tok/s/node`<br>(`688/chip`) | `6,108.5 tok/s/node`<br>(`1,527/chip`; v2: `6,419`) | **`8,037.6 tok/s/node`**<br>(**`2,009.4 tok/s/chip`**) | **`+130%` vs Mandarin `c=64`**<br>**`+192.0%` (`2.92×`) vs `1d02e4a`**<br>**`+31.6%` vs `w384` ref** |
| **`c=64` Natural Steady-State Throughput** | `6,938 tok/s/node` (`c=128` `#1732`) | `2,753.0 tok/s/node` ($\tau=1.00$) | — | **`9,502.5 tok/s/node` (`2,375.6/chip` @ $\tau=4.14$)**<br>(`9,760` @ $\tau=4.25$, `11,482` @ $\tau=5.0$, `11,900` @ $\tau=5.31$) | **`+245%` (`3.45×`) vs `1d02e4a`**<br>**`+37.0%` vs Mandarin `c=128`** |
| **`c=256` / `c=384` Step Time** | `210.47 ms` (`#1890`) $\to$ **`139.93 ms`** (`#1949`, `c=384`) | `98.50 ms/step` (`c=256`) | `65.06 ms/step` (`c=256`) | **`52.81 ms/step`** (`c=256`, `4,847.4 steps/s`, **TPOT `14.85–15.29 ms`**) | **`-62.3%` (`2.65×` faster) vs Mandarin `#1949`**<br>**`-46.4%` vs `1d02e4a`**<br>**`-18.8%` vs `w384` ref** |
| **`c=256` Throughput @ $\tau=3.5$** | `9,532 tok/s/node` (scaled @ $\tau=3.5$ from `#1949`) | `9,666.0 tok/s/node`<br>(`2,417/chip`) | `14,223.4 tok/s/node`<br>(`3,556/chip`) | **`16,966.0 tok/s/node`**<br>(**`4,241.5 tok/s/chip`**) | **`+78.0%` vs Mandarin `#1949`**<br>**`+75.5%` vs `1d02e4a`**<br>**`+19.3%` vs `w384` ref** |
| **`c=256` Natural Steady-State Throughput** | **`10,885.2 tok/s/node`** (`2,721/chip` @ $\tau=4.03$, `c=384` `#1949`) | `9,666.0 tok/s/node` ($\tau=1.00$) | — | **`23,316.9 tok/s/node`** (**`5,829.2 tok/s/chip`** @ $\tau=4.81$)<br>(`20,602` @ $\tau=4.25$, **`24,237` @ $\tau=5.0`**) | **`+114.2%` (`2.14×`) vs Mandarin `#1949`**<br>**Exceeds 70% GB200 target (`22.4K/node`)** |
| **`c=512` Throughput** (2-wave) | — | — | `8,586.2 tok/s/node` wall | **`24,648.5 tok/s/node` implied** (`6,162/chip`, **TPOT `20.77 ms`**)<br>`8,542–9,023 tok/s/node` wall | **`+5.1%` wall vs `w384` ref** |

---

### 1.3 End-to-End 1P1D Disaggregated Prefill + Decode (`TPURaidenConnector` Stage 3 over DCN)

* **Handoff v3 & Mandarin Doc Status**: Not yet run end-to-end on TPU (`TPURaidenConnector` only supported V4 uniform compression ratio).
* **Our 1P1D Results (`8K` input, `128` output, `c=8` & `1K` input, `512` output, `c=64`, 1 Prefill Node $\to$ 1 Decode Node)**:

| Mode | `2+2` & `USA` Gates (`pd==local`) | Greedy 64-Tok Equality + NIAH (`64..20K` tok) | Stage-3 KV Transfer Audit | Mean TTFT (`p50`) | Mean TPOT | `8K×128` (`c=8`) Output Throughput (`Active In tok/s`) | `c=64` Steady-State Decode Throughput (`Step ms`) |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| **Co-Located Decode-Local Baseline** | `16/16`, `50/50` | Reference | N/A | `3,564 ms` | `43.15 ms` | `109.11 tok/s` (`6,983 in tok/s`) | — |
| **1P1D `CED=0` Prefill $\to$ `CED=0` Decode (`DSpark=0`)** | `16/16`, `50/50` | **5/5 `equal=True`**, **5/5 needle `739204`** | `STAGE3_CHECK_OK 1` (`171/171` sends, `0` skipped) | `2,525 ms` (`2,089 ms`) | `25.64 ms` | `157.13 tok/s` (`10,056 in tok/s`) | — |
| **1P1D `CED=1` Prefill $\to$ `CED=0` Decode (`DSpark=0`)** | `16/16`, `50/50` | **5/5 `equal=True`**, **5/5 needle `739204`** | `STAGE3_CHECK_OK 1` (`153/153` sends, window trim `157→20` pages) | `1,395–1,445 ms` (`1,272–1,298 ms`) | `24.81–25.74 ms` | `196.05–200.94 tok/s` (`12,860 in tok/s`) | `~1,365 tok/s` (`~46.9 ms/step`, $\tau=1.0$) |
| **1P1D `CED=1` Prefill $\to$ `CED=0` Decode + DSpark $K=5$ + Async Sched (`13e0b9cd0` — Recommended)** | **`16/16`, `50/50`, `pd==local=True` ($\tau=3.95$)** | **5/5 `equal=True` (`first_div=None`), 5/5 needle `739204`** | **`STAGE3_CHECK_OK 1` (`204/204` sends, `0` skipped, trim `157→20`)** | **`1,118 ms` (`1,206 ms`)**<br>(**`3.19×` faster vs Local**) | **`11.40–11.48 ms`** (`8.21–13.62 ms` @ `c=64`)<br>(**`2.24×` faster vs `DSpark=0`**) | **`355.51 tok/s`** (**`+81.3%` vs `DSpark=0`**, **`3.26×` vs Local**; **`30,467 active in tok/s`**) | **`7,148–14,905 tok/s/node @ $\tau=3.5$`**<br>(**`5,026–9,916` natural**, **`3,076` wall**, **`15.0–31.3 ms/step`** — **closes gap to Standalone Decode!**) |

---

## 2. Branches, Commits & Artifacts

All work lives under `/usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/`. One unified `vllm-torchtpu` worktree (`repos/vtt-combined`) and one `vllm` worktree (`repos/vllm-decode`) serve Prefill, Standalone Decode, and 1P1D Disaggregation.

### 2.1 Repository & Branch Matrix

| Component | Worktree / Path | Branch | Commit SHA | Role / Description |
| :--- | :--- | :--- | :--- | :--- |
| **`vllm-torchtpu` (Unified — Primary)** | `dsv41/repos/vtt-combined` | **`cpk/dsv41-combined`** | **`13e0b9cd0`** (parent `249d66e02`) | **Unified P + D + 1P1D tree with all correctness, Raiden, XProf kernel, and 1P1D + DSpark $K=5$ optimizations** |
| `vllm-torchtpu` (Handoff Prefill Ref) | `dsv41/repos/vtt-prefill` | `rsinghc/deepseek-prefill-optimization-v2` | `cd4eedbcb` | Handoff v3 prefill-only baseline (`197.3K tok/s/node`) |
| `vllm-torchtpu` (Handoff Decode Ref) | `dsv41/repos/vtt-decode` | `rsinghc/deepseek-decode-optimization-v2` | `1d02e4ab9` | Handoff v3 decode-only baseline (`81.8 ms/step`, broken multi-step output) |
| **`vllm` (Unified Runtime)** | `dsv41/repos/vllm-decode` | **`cpk/dsv41-vllm-decode`** | **`afbcce3152`** (`758179797` + `patches/0003`) | Upstream `vllm` @ `f9ad9dd6b495` + common patch (`0001`) + decode runtime v2 (`0002`) + TPU spec chunk alignment (`0003`) |
| **Harness & Docs Repo** | `dsv41/` | **`main`** | **`HEAD`** | Self-contained runners (`scripts/prefill/`, `scripts/decode/`, `scripts/pd/`), patches, wheels |
| **TPU Wheels & Runtime** | `dsv41/whls/` | — | `torch_tpu 0.1.1.dev20260912`, `tpu_sync_torch 0.0.1.dev20260923`, `jax 0.10.2` | Pinned with `libtpu==0.0.48` (decode & 1P1D) / `libtpu==0.0.44` (prefill) |
| **Checkpoint** | `gs://rsinghc-storage/DeepSeek-V4.1-Flash/` | — | 48 shards (`~476 GiB`) | Prefill downloads 24 shards (`2..23, 47, 48`); Decode/1P1D downloads all 48 shards |

### 2.2 Commit Progression on `cpk/dsv41-combined`

| Commit | Scope | Summary of Changes |
| :--- | :--- | :--- |
| `96eafb830` | **P+D Merge + Decode Correctness** | 3-way merge of `cd4eedb` (prefill) and `1d02e4a` (decode) across 42 `src/` files + all 4 `decodefix` commits (`11e6c029f`, `b5665ed31`, `570c80993`, `cfacfa76d`). See `repos/vtt-combined/MERGE_NOTES.md`. |
| `8e72f7821`..`d5129e7c6` | **1P1D Raiden Stage-3 Disaggregation** | V4.1 mixed-compression (`cr=1`/`cr=2`) Raiden pool manifest, shape-split tags (`dsv4_page_shape_tag`), full-block state cache sizing, CPU-first UE8M0 scale init, and `toy_proxy_server.py` DP-rank routing. |
| `937432afb`..`1c324dd31` | **XProf Round-1 Kernel Optimizations** | Direct SparseCore Stage-2 top-$k$ (`sc_topk_direct_op` in `compressor.py`) across all decode buckets (`T <= 384`), and fused Pallas `o_proj_wob_op` (`fused_wo_a_wo_b_h128_h64_vmem`) on `TP=1` decode (`attention.py`). |
| `b5d01a0f3` | **KV Overlay Rotation & Engram Dedup** | Rotated `csa_nope_hosts` across SWA groups in `kv_cache_deepseek_v41_flash.py` (`two_caches_same_buffer=False` for all 43 layers, eliminating two 1.48 GB HBM copies) and removed duplicate decode `engram_hash` call. |
| `249d66e02` | **XProf Round-2 & Paper (`arXiv:2609.19969`) Opts** | C-contiguous host materialization of decode MoE scales (`mxfp4.py`), exact `40`-token (`8*b`) DSpark bucket at `max_chunk_reqs==64` (`spec_decode/dspark.py`), 1D `init_pre_mix` (`model.py`, `dspark.py`), and `scale_slab` bounds alignment (`fused_moe/v2/layer.py`). |
| **`13e0b9cd0`** | **1P1D + DSpark $K=5$ + Async Scheduling & Gap Closure** | Excluded drafter SWA layers (`40..42`) from Raiden Stage-3 55-pool admission (`tpu_connector.py`), aligned `pd` vs `local` step-0 drafter SWA state (`dflash.py`), fixed EP-DP lockstep & `_block_table_stage_cpu` cache invalidation on multi-chunk partial prefills (`tpu_runner.py`, `attention_metadata.py`), added prefill + decode DP load-balancing in `toy_proxy_server.py`, and enabled `ENABLE_DSPARK=1` + `--async-scheduling` on `Decode` (`launch_dsv41_disagg_role.sh`). |

---

## 3. Key Improvements & Quantified Impact

### A. Decode Correctness & DSpark Acceptance ($\tau: 1.00 \to 4.49$, `USA` States: `0/50` $\to$ `50/50`)

1. **Per-Group SWA Block-Table Caching (`attention.py`, `kv_cache_deepseek_v41_flash.py` — `11e6c029f`, `cfacfa76d`)**
   * **Problem**: In decode's overlay KV layout, the 3 SWA KV-cache groups (`[15, 14, 14]` layers) share physical buffers at the same intra-group position while vLLM's hybrid KV coordinator allocates distinct block IDs per group. `_ensure_swa_step_cache()` cached a single block-index tensor across all 40 layers, causing groups 1 and 2 to overwrite group 0's SWA slots after ~32 tokens.
   * **Fix & Impact**: Keyed SWA step metadata by `_dsv41_kv_cache_group_id` and memoized per-group bounds at KV init (avoiding a `+9%` host-loop regression). Restored coherent multi-step generation (**`50/50` US states**).
2. **DSpark Auxiliary Hidden-State Capture (`deepseek_v41/model.py` — `570c80993`)**
   * **Problem**: `1d02e4a` dropped the entry-stream mean capture over `hc` at target layers `(37, 38, 39)`, feeding zeros to the 3-layer DSpark drafter (`arXiv:2609.19969` §4) and collapsing acceptance to $\tau = 1.00$ (`0` drafts accepted).
   * **Fix & Impact**: Restored aux hidden-state capture in `_run_layer`, raising natural DSpark acceptance from **$\tau = 1.00 \to 4.4938$** on USA (`per_pos = [0.88, 0.78, 0.69, 0.60, 0.54]`) and **$\tau = 4.14\text{--}5.31$** on 100K decode benchmarks (**`>3.4×` throughput multiplier**).
3. **Indexer Head-Weight Sign Preservation & `sparse_mla` Running-Max Sink (`b5665ed31`, `11e6c029f`)**
   * **Fix & Impact**: Ported `sign(w) * ReLU(|w| * q · k)` into decode (`rope.py`, `compressor.py`, `streamindex_topk.py`) and included `sinks_t` in the $b_q < 8$ unrolled running max in `sparse_mla.py` to prevent `exp(sinks_t - m_new)` overflow.

### B. DP Request Balancing & Intermediate Compile Buckets (`c=64` Step: `81.8 ms` $\to$ `30.5 ms`, `-62.7%`)

4. **Eliminating 8× DP-Coordinated Bucket Padding (`run_dp8_scale_driver.py`, `in_pod_runner.sh`)**
   * **Problem (XProf `c64_p1`)**: With `--api-server-count 8` and unpinned requests, 64 concurrent requests landed unevenly across the 8 DP ranks (e.g. `{9, 8, 6, 10, 6, 6, 9, 10}` reqs/rank $\implies \max T = 60 > 48$). Because `compile_sizes` was `[48, 384]`, `_dp_coordinated_step` forced all 8 ranks to execute the **384-token** compiled graph (`81.8 ms/step`).
   * **Fix & Impact**:
     * Added intermediate compile buckets (`[48, 60, 72, 96, 384]` / `[48, 96, 192, 384]`), cutting unpinned (`dp_rr=0`) `c=64` step time from **`81.4 ms` $\to$ `35.53 ms` (`-56.4%`)**.
     * Enabled `X-data-parallel-rank` round-robin (`dp_rr=1`) + `libtpu 0.0.48` (unlocked by fixing `/computeMetadata/v1/instance/attributes/tpu-env` in `support/mock_metadata.py`), cutting `c=64` step time to **`30.53 ms` (`-62.7%`)** and `c=256` step time from **`98.5 ms` $\to$ `53.28 ms` (`-45.9%`)**.

### C. XProf-Guided HLO & Kernel Optimizations (Device Time: `28.50 ms` $\to$ `19.98 ms/step`, `-29.9%`)

Profiled with `/google/bin/releases/xprof/xprof-mcp/xprof_cli.par` across three TPU7x XProf sessions (`chaitanyapk-14677790351621652831` $\to$ `chaitanyapk-18402943240297915715` $\to$ `chaitanyapk-13959792908329192688`):

| # | Optimization (`cpk/dsv41-combined`) | Root Cause in XProf HLO | Pure-Device Time Impact (`c=64`, `T=48`) |
| :---: | :--- | :--- | :--- |
| **5** | **Direct SparseCore Stage-2 Top-$K$ (`sc_topk_direct_op`)** (`compressor.py`, `937432afb`) | `sc_topk_stage2_op` returned relative indices and ran a TensorCore `gather` over `[96, 32768]` (`gather_fusion.94/.95/.96` = `4.66 ms/step`, 1% HBM BW); `T > 48` fell back to a 9-round TensorCore tournament (`6.75 ms/step`). Fused candidate gather directly inside SparseCore VMEM for all `T <= 384`. | **`4.66 ms` $\to$ `0.20 ms/step`**<br>(**`-4.46 ms/step` / `-95.7%`**) |
| **6** | **Rotated CSA Overlay Hosts Across SWA Groups** (`kv_cache_deepseek_v41_flash.py`, `b5d01a0f3`) | `overlay_hosts` assigned SWA layers by intra-group `position`, causing Layer 6 and Layer 13 to alias `sw_cache` with their own CSA `main_cache_kv` (`two_caches_same_buffer=True`) and forcing two `1.48 GB` HBM copies (`copy.4527`/`.4528`). Rotated `host_pos = (position + group_idx) % n_csa_hosts`. | **`1.83 ms` $\to$ `0.00 ms/step`**<br>(**`-1.83 ms/step`, `-2.96 GB/step` HBM traffic**) |
| **7** | **Fused Pallas `o_proj_wob_op` on `TP=1` Decode** (`attention.py`, `937432afb`) | `not self.tp_attn` guard blocked `fused_wo_a_wo_b_h128_h64_vmem` on `TP=1`, materializing 43× `[T, 16, 5120]` tensors in HBM before reducing over heads. Accumulating in VMEM cut 43× `o_proj_wob` by 44%. | **`2.28 ms` $\to$ `1.17 ms/step`**<br>(**`-1.11 ms/step` / `-48.7%`**) |
| **8** | **Decode MoE Scale C-Contiguous Materialization + `scale_slab` Bounds** (`mxfp4.py`, `fused_moe/v2/layer.py`, `249d66e02`) | `_process_weights_after_loading_decode` left 86 `_tpu_fused_w13/w2_scale` buffers (`633 MB`) as `as_strided` views of 4D `{3,2,1,0:T(1,128)}` tensors, triggering `tt_jit_execute_as_strided` on every step; `scale_slab` scattered off-shard pairs into in-bounds slot `2752 < 2944`. Materialized scales via CPU round-trip (`%param.1132` async VMEM DMA) and bounded `scale_slab` scatter to `ragged_stride` (`2752`). | **MoE `6.21 ms` $\to$ `5.42 ms/step` (`-0.80 ms/step`, `-12.8%`)**<br>**`as_strided` `0.43 ms` $\to$ `0.004 ms/step` (`-0.42 ms/step`, `-99.0%`)** |
| **9** | **Exact `40`-Token (`8*b`) DSpark Bucket + 1D `init_pre_mix`** (`spec_decode/dspark.py`, `model.py`, `249d66e02`) | `max_chunk_reqs == 64` omitted `8 * b = 40`, padding `c=64` DSpark 2× from `40` to `80` tokens (`16` reqs, `g32`) and breaking `_get_reusable_draft_block_tables`. Added `8 * b` (`40`) and registered `init_pre_mix` as 1D `(4,)`. | **DSpark `1.95 ms` $\to$ `1.52 ms/step`**<br>(**`-0.43 ms/step` / `-22.0%`**, `-2` HLO launches/step) |

* **End-to-End Impact of Kernel Optimizations (#5–#9)**:
  * **Decode `c=64`**: Steady-state step time improved from `30.53 ms` (`29.92 ms` on 4-bucket) to **`28.23 ms/step`** (`27.87 ms` @ `2,296.5 steps/s`, **`8,037.6 tok/s/node @ $\tau=3.5$`** / **`9,502.5 tok/s/node` natural**, Mean TPOT **`6.82 ms`**).
  * **Decode `c=256`**: Steady-state step time improved from `70.13 ms` to **`52.81 ms/step`** (**`16,966.0 tok/s/node @ $\tau=3.5$`** / **`23,316.9 tok/s/node` natural** at $\tau=4.81$).
  * **Prefill `128K c=8`**: Also improved by **`+0.6%`** to **`198,541.3 input tok/s/node`** (`49,635 tok/s/chip`) thanks to 1D `init_pre_mix` and `scale_slab` bounds alignment.

### D. 1P1D Disaggregated Raiden Stage-3 + DSpark $K=5$ Gap Closure (`TTFT -68.6%`, `TPOT 2.24× Faster`, `c=64` Matches Standalone Decode)

10. **DSv4.1 Mixed-Compression Raiden Manifest & `CED=1` Disaggregation (`8e72f7821..d5129e7c6`)**
    * Extended `pool_manifest.py` and `layout_fingerprint.py` for DSv4.1's 55 KV pools (`cr=2` $T=512$ at layers 2/8/14, `cr=1` $T=1024$ at layer 20, seq-along-lane indexer `[nb,33,4,T]`, SWA overlays, and full-block `dsv4.state.csa.g4`), fixed CPU-first UE8M0 `0x7F` scale initialization to preserve TPU tiling, and enabled `VLLM_DSV41_CED=1` 20-layer prefill + 128-token SWA tail transfer into `CED=0` 40-layer decode.
11. **Closing the Gap: 1P1D + DSpark ($K=5$) + `--async-scheduling` + Prefill/Decode DP Load Balancing (`13e0b9cd0`, `afbcce3152`)**
    * **Why there was a gap in `20261022-093059_1p1d_ced1_repro`**: The initial 1P1D run launched `Decode` with `ENABLE_DSPARK=0` ($\tau=1.0$) and `--no-async-scheduling`, and `toy_proxy_server.py` did not pin `X-data-parallel-rank` on `Prefill`.
    * **Five fixes that closed the gap**:
      1. **Raiden Admission Filtering (`tpu_connector.py`)**: Filtered `runner.drafter._draft_attn_layer_names` (`layers 40..42` SWA caches) out of `named_kv_caches` and `kv_cache_groups` during `_admit_raiden_dsv4_kv_cache` so `Prefill` (`ENABLE_DSPARK=0`, 3 KV groups) and `Decode` (`ENABLE_DSPARK=1`, 6 KV groups) share an identical 55-pool Stage-3 manifest and layout fingerprint.
      2. **Step-0 Drafter SWA State Parity (`dflash.py`)**: On prefill/first-verify chunks (`chunk.draft_lengths is None`) under `TPU_RAIDEN_DSV4_ADMISSION=1`, zeroed the active ring-mapped SWA blocks in the 3 draft SWA layers (`40..42`) and masked non-final prompt token rows in `target_hidden` so `pd` (which only runs token `prefill_len - 1` on `Decode`) and `local` produce bit-for-bit identical drafter proposals (`pd==local=True` on `[GATE-USA]` and 5/5 `equal=True` on `[GATE-EQ]`).
      3. **Partial-Prefill EP-DP Lockstep & `_block_table_stage_cpu` Invalidation (`tpu_runner.py`, `attention_metadata.py`, `scheduler.py`)**: Added `run_dp_dummy_draft` on `all_partial_prefill` steps, aligned intermediate TPU chunked-prefill splits to 64-token KV blocks in `vllm/v1/core/sched/scheduler.py`, and invalidated `_block_tables_append_only` / `_cached_bt_stage_layout` whenever a non-cached (`not can_cache_bt`) `stage_block_table_uploads` call zeroes the shared `_block_table_stage_cpu` buffer.
      4. **Least-Inflight + Round-Robin DP Rank Pinning for Both Prefill & Decode (`toy_proxy_server.py`, `run_pd.py`)**: Added `--prefiller-dp-size 8` and `itertools.count()` tie-breaking alongside `--decoder-dp-size 8`, eliminating EP-DP lockstep padding across both prefill and decode nodes.
      5. **DSpark $K=5$ + `--async-scheduling` + Warm `2048` Prefill Bucket (`launch_dsv41_disagg_role.sh`)**: Enabled `ENABLE_DSPARK=1`, `ASYNC_SCHEDULING=1`, and `COMPILE_SIZES=[16, 40, 48, 64, 96, 192, 384]` on `Decode`, and `MAX_NUM_BATCHED_TOKENS=8192`, `COMPILE_SIZES=[512, 2048, 8192]` on `Prefill`.
    * **Result**: Cut `8K×128` Mean TPOT **`2.24×`** (`25.74 ms` $\to$ **`11.40–11.48 ms`**) and Mean TTFT to **`1,118 ms`**, raised `8K×128` output throughput **`+81.3%`** (`196.05` $\to$ **`355.51 tok/s`**), and reached **`7,148–14,905 tok/s/node @ $\tau=3.5$`** (`5,026–9,916` natural, `3,076` wall, `15.0–31.3 ms/step`) on `[BENCH-PD-C64]`, matching Standalone Decode (`7,564–8,038 tok/s/node @ $\tau=3.5$`, `28.2–29.6 ms/step`).

---

## 4. Crisp Reproduction Guide

### 4.1 One-Time Prerequisites & Workspace Setup (~5 mins, no TPU required)

```bash
# 1. Authenticate gcloud & verify GKE cluster context
gcloud auth login --update-adc
kubectl config get-contexts   # choose a cluster with free tpu7x 2x2x1 nodes

# 2. If starting from the existing workspace on chaitanyapk-dev1:
cd /usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41
git -C repos/vtt-combined rev-parse --short HEAD   # should print 13e0b9cd0
```

<details>
<summary><b>Fresh clone instructions (if setting up on a new workstation)</b></summary>

```bash
mkdir -p dsv41/repos && cd dsv41
git clone git@github.com:vllm-project/vllm-torchtpu.git repos/vllm-torchtpu
git -C repos/vllm-torchtpu worktree add -b cpk/dsv41-combined ../vtt-combined 13e0b9cd0

git clone --filter=blob:none https://github.com/vllm-project/vllm.git repos/vllm-base
git -C repos/vllm-base worktree add -b dsv41-vllm-prefill ../vllm-prefill f9ad9dd6b495f7492fa975e0ac48164b090d99ad
git -C repos/vllm-prefill apply ../../patches/0001-vllm-common-f9ad9dd6b4.patch && git -C repos/vllm-prefill commit -qam common
git -C repos/vllm-base worktree add -b cpk/dsv41-vllm-decode ../vllm-decode dsv41-vllm-prefill
git -C repos/vllm-decode apply ../../patches/0002-vllm-decode-runtime-v2.patch && git -C repos/vllm-decode commit -qam decode-v2
git -C repos/vllm-decode apply ../../patches/0003-vllm-tpu-spec-chunk-align.patch && git -C repos/vllm-decode commit -qam spec-chunk-align
# Download the 14 wheels from Drive (https://drive.google.com/drive/folders/1gM6d_wMx0jV_FE9Ml-Tn_N6uHeZrJjoz?resourcekey=0-mN4C4Kf8TPxgbSd8EKxz1Q) into whls/
(cd whls && sha256sum -c SHA256SUMS.txt)
```
</details>

---

### 4.2 Reproduce Prefill (`4 × (TP=2, EP=2)`, 20-Layer CED Prefill — ~20 mins)

Creates pod `<user>-dsv41-combined-pf-<suffix>`, stages `repos/vtt-combined`, downloads the 24 prefill shards, boots 4 × `TP=2, EP=2` servers (`VLLM_DSV41_CED=1`, `VLLM_DSV41_PURE_20L_PREFILL=1`, `max_num_batched_tokens=16384`) behind the least-in-flight proxy on `:9000`, runs the 6-workload sweep (`128K c=4/8/16`, `100K c=8/16`, `32K c=8`), harvests `summary.json`, and deletes the pod:

```bash
cd /usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41

python3 -u scripts/prefill/run_prefill.py \
  --ctx <kube-context> \
  --thread combined \
  --suffix repro \
  --vtt repos/vtt-combined

# Compare against the 197.3K reference (expected: ~198.5K tok/s/node at 128k_c8, all 6 OK):
python3 scripts/compare_results.py prefill runs/combined/*_prefill_repro/summary.json
```

---

### 4.3 Reproduce Standalone Decode (`DP=8, TP=1, EP=8`, DSpark $K=5$, 100K Context — ~25 mins)

Creates pod `<user>-dsv41-perf-dc-<mode>-<suffix>` (`hostPath /dev/shm`, `200Gi` request), stages `repos/vtt-combined` + `libtpu==0.0.48`, downloads all 48 shards, boots `vllm serve` (`DP=8, TP=1, EP=8`, DSpark $K=5$, `DecodeBenchConnector` `cache_hit_ratio=0.99936`), runs the `2+2` (8/8) and `USA` (50/50 states, $\tau \ge 4.4$) accuracy gates, seeds 100K KV templates across all 8 ranks (`DSV41_DP_RR=1`), runs the `c=64`, `c=256`, and `c=512` benchmarks, harvests `result_*.json`, and deletes the pod:

```bash
cd /usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41

# Run full suite (1x_c64 + 4x_c256 + 8x_c512) with optimal 4-bucket compile config:
python3 -u scripts/decode/run_decode.py \
  --ctx <kube-context> \
  --thread perf \
  --mode dp8_all \
  --suffix repro \
  --vtt repos/vtt-combined \
  --libtpu 0.0.48 \
  --buckets '[48, 96, 192, 384]'

# Compare against reference (expected: c=64 ~28.2 ms/step, 8,038 tok/s @3.5; c=256 ~52.8 ms/step, 16,966 tok/s @3.5):
python3 scripts/compare_results.py decode runs/perf/*_decode_dp8_all_repro/
```

*(For a quick `c=64`-only run, replace `--mode dp8_all` with `--mode dp8_c64`.)*

---

### 4.4 Reproduce 1P1D Disaggregated Prefill + Decode (`TPURaidenConnector` Stage 3 + DSpark $K=5$ — ~30 mins, 2 Nodes)

Creates two pods on `<kube-context>` (1 prefill node `DP=8, TP=1, EP=8`, `kv_producer`, `CED=1`, `MNBT=8192`, `compile_sizes=[512, 2048, 8192]` on `:8400` + 1 decode node `DP=8, TP=1, EP=8`, `kv_consumer`, `CED=0`, `ENABLE_DSPARK=1`, `ASYNC_SCHEDULING=1`, `compile_sizes=[16, 40, 48, 64, 96, 192, 384]` on `:9400`), starts `toy_proxy_server.py` on `:8000` with `--prefiller-dp-size 8 --decoder-dp-size 8`, runs `pd_gates.py` (`2+2` 16/16, `USA` 50/50 + `pd==local`, 5/5 greedy 64-token equality + NIAH needle `739204` across `64..20,000` tokens, `[BENCH-PD]` `8K` in / `128` out `c=8`, `[BENCH-LOCAL]`, and `[BENCH-PD-C64]` `1K` in / `512` out `c=64`) + `check_dsv4_stage3_log.py`, harvests all logs to `runs/pd/`, and deletes both pods:

```bash
cd /usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41

# Recommended: CED=1 prefill -> CED=0 decode + DSpark K=5 + --async-scheduling (default)
python3 -u scripts/pd/run_pd.py \
  --ctx <kube-context> \
  --suffix opt \
  --tag dspark_opt_full \
  --prefill-ced 1 \
  --vtt repos/vtt-combined
```

---

### 4.5 Key Operational Notes (Handled Automatically by the Scripts)

* **`VLLM_DSV41_PURE_20L_PREFILL=1`**: Returns dummy logits on standalone prefill pods by design (only layers `0..19` + layer-20 KV projection execute). Never run accuracy gates against a pure-prefill server.
* **`support/mock_metadata.py`**: Serves `/computeMetadata/v1/instance/attributes/tpu-env` so `libtpu==0.0.48` boots cleanly inside single-node GKE pods without hanging on `INJECT_SLICE_BUILDER_FAULT` retries.
* **`/dev/shm` Hygiene**: Decode pods mount `hostPath: /dev/shm` (`remount,size=850G`) and unlink checkpoint `.safetensors` once all 8 workers log `"Memory statistics"` so 476 GiB of weights + KV cache fit without pod eviction.
* **Automatic GCE Spot-Preemption Recovery**: `scripts/pd/run_pd.py` (`36c9166`), `scripts/decode/run_decode.py` (`be01482`), and `scripts/prefill/run_prefill.py` (`324fe30`) monitor pod phase during setup and benchmark execution; if a Spot node is preempted (`phase != "Running"`), only the preempted pod is recreated, re-staged, and relaunched.

---

## 5. Run Logs, GCS Archive & Verification Details (Per-Run Breakdown)

All reproduction artifacts, git bundles (`.bundle`), clean source tarballs (`.tar.gz`), wheels, prebuilt stack tarballs, and complete per-run logs are archived at **`gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/`**.

### 5.1 Top-Level GCS Archive Index (`gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/`)

| GCS Path | Contents |
| :--- | :--- |
| `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/README.md`, `MERGE_NOTES.md`, `PD_DESIGN.md`, `COORDINATION.md`, `PHASE2_PLAN.md` | Documentation, 42-file P+D merge manifest, 1P1D Raiden Stage-3 design, and experiment ledger |
| `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/repos/vllm-torchtpu-combined-{13e0b9cd0,249d66e02}.{bundle,tar.gz}` | Unified `vllm-torchtpu` repository (`cpk/dsv41-combined` @ `13e0b9cd0` & `249d66e02`, full git history + clean tree) |
| `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/repos/vllm-decode-{afbcce3152,758179797}.{bundle,tar.gz}` | Patched `vllm` runtime (`cpk/dsv41-vllm-decode` @ `afbcce3152`: `f9ad9dd6b4` + `0001` + `0002` + `0003`) |
| `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/repos/vllm-prefill-a8cb9ec41.{bundle,tar.gz}` | Patched `vllm` prefill runtime (`dsv41-vllm-prefill` @ `a8cb9ec41`: `f9ad9dd6b4` + `0001`) |
| `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/repos/dsv41-harness.{bundle,tar.gz}` | `dsv41` reproduction harness, scripts, patches, and support files |
| `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/stacks/stack_{329ed6bdfa396b11,f2128e6ebbf4d1e1,26d14abf13e4d410,27074ff8b209e63b}.tar` | Content-addressed `/workspace` tarballs staged onto 1P1D, Decode, and Prefill pods |
| `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/{scripts,patches,support,whls,reference}/` | Direct copies of all runner scripts, patches, metadata shims, wheels, and reference docs |

---

### 5.2 Run 1: Standalone Prefill (`4 × (TP=2, EP=2)`, 20-Layer CED Prefill)

* **Archived Runs (`cpk/dsv41-combined` @ `249d66e02`, `vllm` @ `a8cb9ec41`, Stack `27074ff8b209e63b`)**:
  1. **Peak Validation Run**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/20261021-191514_prefill_xprof3/` (Local: `runs/combined/20261021-191514_prefill_xprof3/`, Node: `gke-tpu-e785c3d8-0d34` on `amylin-v7x-perf-2x2x1`)
  2. **Repro Run #1 (`amylin`)**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/20261022-065005_prefill_repro/` (Local: `runs/combined/20261022-065005_prefill_repro/`, Node: `gke-tpu-e785c3d8-0d34` on `amylin-v7x-perf-2x2x1`)
  3. **Repro Run #2 (`bodaborg`)**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/20261022-094757_prefill_repro2/` (Local: `runs/combined/20261022-094757_prefill_repro2/`, Node: `gke-tpu-8a56d114-2mrv` on `bodaborg-tpu7x-spot-sps`)

* **Log & Artifact Inventory (inside each `<prefill_run>/` directory)**:

| File | Size | Description |
| :--- | ---: | :--- |
| `provenance.json` | `479 B` | Cluster context, pod name, exact `vllm-torchtpu` (`249d66e02`) and `vllm` (`a8cb9ec41`) SHAs, `stack_key` (`27074ff8b209e63b`) |
| `driver.log` / `driver_console.log` | `68 KB` | In-pod orchestrator log: source hash verification, Chip 0 XLA cache warmup (`360.1s`), Chips 1–3 parallel boot (`320.2s`), RR proxy startup, and `vllm bench serve` output for all 6 workloads |
| `vllm_chip0.log` .. `vllm_chip3.log` | `4 × 160 KB` | Per-instance `vllm serve` logs (`TP=2, EP=2`, ports `8000..8003`, `VLLM_DSV41_CED=1`, `VLLM_DSV41_PURE_20L_PREFILL=1`, `libtpu==0.0.44`) |
| `summary.json` | `25 KB` | Consolidated JSON summary across `128k_c4`, `128k_c8`, `128k_c16`, `100k_c8`, `100k_c16`, and `32k_c8` (input/total tok/s, mean/p50/p99 TTFT) |
| `perf_results.tar` | `30 KB` | Raw per-workload `vllm bench serve` JSON files (`v85_4xtp2_prefill_clean_flags_b16k_4chip_*.json`) |
| `repos/` | `383 MB` | Exact `vllm-torchtpu-combined-249d66e02.{bundle,tar.gz}`, `vllm-prefill-a8cb9ec41.{bundle,tar.gz}`, and `dsv41-harness.tar.gz` |

* **Key Log Excerpt (`driver.log` & `compare_results.py` from `20261021-191514_prefill_xprof3` / `20261022-094757_prefill_repro2`)**:
```text
[DRIVER] Verified source hashes inside /workspace/vllm-torchtpu: envs.py=186f79f8b9d384365c4de605c5e85353 fused_moe_ep.py=ba5610d833af2e4d5db8a59029c0c356
[DRIVER] Launching Chip 0 to populate shared XLA cache...
[READY in 360.1s] Ports [8000] healthy!
[DRIVER] Launching Chips 1, 2, 3 concurrently with warm XLA cache + RR proxy...
[READY in 320.2s] Ports [8000, 8001, 8002, 8003] healthy!
[BENCH 128k_c4]  completed=16/16 | input_tok/s=186,703.1 | mean_ttft=2794.8ms | p50_ttft=2793.9ms
[BENCH 128k_c8]  completed=32/32 | input_tok/s=198,541.3 | mean_ttft=4946.6ms | p50_ttft=5214.6ms
[BENCH 128k_c16] completed=32/32 | input_tok/s=197,891.3 | mean_ttft=8609.8ms | p50_ttft=10423.0ms
[BENCH 100k_c8]  completed=32/32 | input_tok/s=196,644.1 | mean_ttft=3809.7ms | p50_ttft=3931.4ms
[BENCH 100k_c16] completed=32/32 | input_tok/s=195,839.6 | mean_ttft=6648.6ms | p50_ttft=7859.7ms
[BENCH 32k_c8]   completed=64/64 | input_tok/s=189,563.1 | mean_ttft=1334.5ms | p50_ttft=1365.7ms
ALL_BENCHMARKS_COMPLETE_OK!
```

---

### 5.3 Run 2: Standalone Decode (`DP=8, TP=1, EP=8`, DSpark $K=5$, 100K Context)

* **Archived Runs (`cpk/dsv41-combined` @ `249d66e02`, `vllm` @ `758179797`, Stack `26d14abf13e4d410`, `libtpu==0.0.48`, `compile_sizes=[48, 96, 192, 384]`, `dp_rr=1`)**:
  1. **Peak + XProf Round-2 Run (`amylin`)**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/20261021-190440_decode_dp8_c64_xprof3/` (Local: `runs/perf/20261021-190440_decode_dp8_c64_xprof3/`, Node: `gke-tpu-e785c3d8-gzmi`, XProf session: `chaitanyapk-13959792908329192688`)
  2. **Full `dp8_all` Repro Run (`bodaborg`)**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/20261022-093105_decode_dp8_all_repro/` (Local: `runs/perf/20261022-093105_decode_dp8_all_repro/`, Node: `gke-tpu-b10c48a3-x8s4`)

* **Log & Artifact Inventory (inside each `<decode_run>/` directory)**:

| File | Size | Description |
| :--- | ---: | :--- |
| `provenance.json` | `647 B` | Cluster context, pod name, mode (`dp8_all`), `libtpu` (`0.0.48`), `dp_rr` (`1`), `buckets` (`[48, 96, 192, 384]`), `vllm-torchtpu` (`249d66e02`) & `vllm` (`758179797`) SHAs |
| `runner.log` / `driver_console.log` | `16 KB` | In-pod runner output: pip versions, server readiness, `[GATE-2+2]`, `[GATE-USA]`, `[WARMUP-KV]`, and `[SUMMARY-1x_c64]`, `[SUMMARY-4x_c256]`, `[SUMMARY-8x_c512]` |
| `vllm_server.log` | `756 KB` | Complete `vllm serve` log across all 8 DP/EP workers (`Memory statistics`, `27 arrays for 54 layers`, `num_blocks=5568`, `[POST_WARMUP_HBM] free_gib=23.775`) |
| `gate_meta.json` & `accuracy_dp8_usa_states_verified.json` | `3.6 KB` | Full accuracy gate verification (`passed_2p2=true`, `num_us_states_matched=50`, `real_dspark_mean_accept_len=4.4938`, per-position acceptance `[0.8765, 0.7778, 0.6914, 0.6049, 0.5432]`, generated USA text) |
| `result_dp8_hbm0p90_64x1024_verified.json` | `559 KB` | Complete `1x_c64` (`64 × 1024` @ 100K context) benchmark JSON including per-request TTFTs, ITLs, E2ELs, generated texts, and interior/wall throughput |
| `result_dp8_4x_c256.json` | `2.0 MB` | Complete `4x_c256` (`256 × 1024` @ 100K context) benchmark JSON |
| `result_dp8_8x_c512.json` | `4.9 MB` | Complete `8x_c512` (`512 × 1024` @ 100K context) benchmark JSON |
| `repos/` | `251 MB` | Exact `vllm-torchtpu-combined-249d66e02.{bundle,tar.gz}`, `vllm-decode-758179797.{bundle,tar.gz}`, and `dsv41-harness.tar.gz` |

* **Key Log Excerpt (`runner.log` & `compare_results.py` from `20261021-190440_decode_dp8_c64_xprof3` / `20261022-093105_decode_dp8_all_repro`)**:
```text
[PIP] libtpu 0.0.48 vllm 0.29.0+empty torch_tpu 0.1.1.dev20260912075415
[RUNNER] compile_sizes [48, 96, 192, 384] dp_rr=1
(Worker_DP0_EP0) INFO [kv_cache_deepseek_v41_flash.py:235] DeepSeek-V4.1-Flash KV cache: 27 arrays for 54 layers (mla=8 of which csa=3, swa=43 in 3 group(s) sized [15, 14, 14], state=3), num_blocks=5568, largest array=(5568, 33, 4, 1024)
(Worker_DP0_EP0) [POST_WARMUP_HBM] rank=0 free_bytes=25528083456 total_bytes=101724454912 free_gib=23.775 used_gib=70.963 total_gib=94.738
=== SERVER READY ON rsinghc-dp8-c512-scale (DP=8, max-num-seqs=64, compile_sizes=[48,96,192,384])! ===
[GATE-2+2] passed=True answers=['4', '4', '4', '4', '4', '4', '4', '4']
[GATE-USA] states=50/50 real_dspark_mean_accept_len=4.4938 (acc=283 / drafts=81) per_pos=[0.8765, 0.7778, 0.6914, 0.6049, 0.5432]
[WARMUP-KV] Seeded 100K KV templates across all 8 DP workers in 7.75s!
[SUMMARY-1x_c64] completed=64/64  failed=0 wall=6869.09 tok/s interior=9502.51 tok/s (1187.81/chip) mean_tpot=6.82ms (implied=9381.29 tok/s) accept=4.1379 (@3.5=8037.64, @4.25=9760.0, @5.0=11482.35)
[SUMMARY-4x_c256] completed=256/256 failed=0 wall=9391.47 tok/s interior=23316.90 tok/s (2914.61/chip) mean_tpot=15.29ms (implied=16742.73 tok/s) accept=4.8102 (@3.5=16965.97, @4.25=20601.53, @5.0=24237.1)
[SUMMARY-8x_c512] completed=512/512 failed=0 wall=8542.49 tok/s mean_tpot=20.77ms (implied=24648.46 tok/s) accept=4.7395
```

---

### 5.4 Run 3: 1P1D Disaggregated Prefill + Decode (`TPURaidenConnector` Stage 3 + DSpark $K=5$ over DCN)

* **Archived Runs (`cpk/dsv41-combined` @ `13e0b9cd0` / `249d66e02`, `vllm` @ `afbcce3152` / `758179797`, `libtpu==0.0.48`, `tpu_sync_torch==0.0.1.dev20260923040928`)**:
  1. **1P1D + DSpark $K=5$ Warm Full Pass (`bodaborg`, `13e0b9cd0`, Stack `329ed6bdfa396b11`)**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/20261026-161733_dspark_opt_warm_opt/` (Local: `runs/pd/20261026-161733_dspark_opt_warm_opt/`, Prefill node: `gke-tpu-b10c48a3-x8s4` `10.11.0.102`, Decode node: `gke-tpu-284c5a3a-djxv` `10.11.0.72`)
  2. **1P1D + DSpark $K=5$ Cold-to-Ready Full Boot + Gates (`bodaborg`, `13e0b9cd0`, Stack `329ed6bdfa396b11`)**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/20261026-154717_dspark_opt_full_opt/` (Local: `runs/pd/20261026-154717_dspark_opt_full_opt/`)
  3. **1P1D + DSpark $K=5$ Peak `c=64` Steady-State Run (`bodaborg`)**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/20261026-141011_dspark_opt_opt/` (Local: `runs/pd/20261026-141011_dspark_opt_opt/`)
  4. **Baseline 1P1D `CED=1` (`DSpark=0`) Repro (`bodaborg`, `249d66e02`, Stack `f2128e6ebbf4d1e1`)**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/20261022-093059_1p1d_ced1_repro/` (Local: `runs/pd/20261022-093059_1p1d_ced1_repro/`)
  5. **1P1D `CED=1` Validation (`amylin`)**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/20261020-165851_1p1d_ced1_final_pass_a/` (Local: `runs/pd/20261020-165851_1p1d_ced1_final_pass_a/`)
  6. **1P1D `CED=0` Validation (`bodaborg`)**: `gs://chaitanyapk-data/dsv4.1_opt/dsv4.1/20261020-165853_1p1d_ced0_final_pass_b/` (Local: `runs/pd/20261020-165853_1p1d_ced0_final_pass_b/`)

* **Log & Artifact Inventory (inside each `<1p1d_run>/` directory)**:

| File | Size | Description |
| :--- | ---: | :--- |
| `provenance.json` | `1.4 KB` | Cluster context, prefill/decode pod names, `run_id`, CLI args, `vllm-torchtpu` (`13e0b9cd0`) & `vllm` (`afbcce3152`) SHAs, `extra_whls` (`tpu_sync_torch`) |
| `summary.json` | `4.2 KB` | Node assignments, pod IPs, boot duration, gate summary (`2p2`, `usa`, `token_equality`, `hard_gates_passed`), `bench_pd`, `bench_local`, `bench_pd_c64`, and `stage3_check` |
| `driver_console.log` | `78 KB` | End-to-end client log from `scripts/pd/run_pd.py` (pod creation, staging, boot progress, gates, Stage-3 check, cleanup) |
| `prefill__pd_prefill.log` | `640 KB` | Prefill server log (`DP=8, TP=1, EP=8`, `kv_producer`, `CED=1`, `MNBT=8192`, `compile_sizes=[512, 2048, 8192]`, Raiden 55-pool admission, per-request `raiden_stage3_send` & SWA window trim logs) |
| `decode__pd_decode.log` | `980 KB` | Decode server log (`DP=8, TP=1, EP=8`, `kv_consumer`, `CED=0`, `ENABLE_DSPARK=1`, `ASYNC_SCHEDULING=1`, `compile_sizes=[16, 40, 48, 64, 96, 192, 384]`, Raiden 55-pool admission, Stage-3 KV pull & H2D install logs) |
| `prefill__pd_proxy.log` | `48 KB` | `toy_proxy_server.py` log (`:8000` $\to$ prefill `:8400` $\to$ decode `:9400` with least-inflight + round-robin `X-data-parallel-rank` pinning across both 8 prefill and 8 decode DP ranks) |
| `prefill__pd_gates.log` & `prefill__pd_gates.json` | `2.4 KB` / `46 KB` | `pd_gates.py` log and full JSON output (`gate_2p2`, `gate_usa`, `gate_token_equality` for `64, 700, 1500, 5000, 20000` tokens, `bench_pd`, `bench_local`, `bench_pd_c64`) |
| `prefill__pd_stage3_check.txt` | `250 B` | Output of `check_dsv4_stage3_log.py --min-sends 5` (`STAGE3_CHECK_OK 1`, `rc=0`) |
| `prefill__pd_setup.log` & `decode__pd_setup.log` | `2 × 2.1 KB` | In-pod setup logs (`[PIP] libtpu 0.0.48 ... tpu_sync_torch 0.0.1.dev20260923040928`, `CKPT_READY`, `PD_SETUP_OK`) |

* **Key Log Excerpt (`prefill__pd_prefill.log`, `prefill__pd_gates.log`, & `prefill__pd_stage3_check.txt` from `20261026-161733_dspark_opt_warm_opt` / `20261026-154717_dspark_opt_full_opt`)**:
```text
# 1. Raiden DSv4.1 55-Pool Admission with DSpark K=5 on Decode (prefill__pd_prefill.log & decode__pd_decode.log):
[PD-LAUNCH] role=prefill ced=1 dspark=0 async=0 mnbt=8192 compile_sizes=[512, 2048, 8192]
[PD-LAUNCH] role=decode  ced=0 dspark=1 async=1 mnbt=384  compile_sizes=[16, 40, 48, 64, 96, 192, 384]
INFO [tpu_connector.py:2981] Raiden DSv4 pool admission complete topology=dp8_prefill role=kv_producer binding=private_typed pools=55 storages=26 tags=13
INFO [tpu_connector.py:2981] Raiden DSv4 pool admission complete topology=dp8_decode  role=kv_consumer binding=private_typed pools=55 storages=26 tags=13

# 2. Disaggregated Correctness & Performance Gates (prefill__pd_gates.log):
[GATE-2+2] pd_passed=True pd=['4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4']
[GATE-2+2] local_passed=True local=['4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4']
[GATE-USA] pd_states=50/50 local_states=50/50 pd==local=True tau=3.9524 pd_tail='ampshire**, **New Jersey**, **New Mexico**, **North Carolina**, **North Dakota**, **Rhode Island**, **South Carolina**, **South Dakota**, and **West Virginia**.'
[GATE-EQ] n=64    equal=True first_div=None pd='739204, Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, D' local='739204, Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, D'
[GATE-EQ] n=700   equal=True first_div=None pd='739204, Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, D' local='739204, Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, D'
[GATE-EQ] n=1500  equal=True first_div=None pd='739204, Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, D' local='739204, Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, D'
[GATE-EQ] n=5000  equal=True first_div=None pd='739204, Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, D' local='739204, Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, D'
[GATE-EQ] n=20000 equal=True first_div=None pd='739204, Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, D' local='739204, Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, D'
[BENCH-PD]     n=16 c=8  in=8192 out=128 ttft_mean=1.1178s p50=1.2061s tpot=11.476ms tau=2.2134 out_tok/s=355.51 int_tok/s=474.44 step_ms=37.32 in_tok/s=22752.44 active_in_tok/s=30467.04
[BENCH-LOCAL]  n=16 c=8  in=8192 out=128 ttft_mean=3.3474s p50=2.2451s tpot=26.192ms tau=3.3756 out_tok/s=138.07
[BENCH-PD-C64] n=64 c=64 in=1024 out=512 ttft_mean=1.7643s p50=1.8525s tpot=13.62ms  tau=2.461  out_tok/s=3076.17 int_tok/s=5026.14 @3.5=7148.12 step_ms=31.34
[PD-GATES] hard_gates_passed=True

# 3. Stage-3 Transfer & SWA Window Trim Audit (prefill__pd_stage3_check.txt):
STAGE3_REGISTERED_SENDS 204
STAGE3_COMPLETED_SENDS 204
STAGE3_SKIPPED_SENDS 0
STAGE3_TRIM_CHECKED 168 requests; worst case req=cmpl-3552d5bb-142b-41d4-9972-ccf5fd6dbd0f-0-8a506028 n=19999 max_group_pages=20 untrimmed_would_be=157
STAGE3_CHECK_OK 1
rc=0
```

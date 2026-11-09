# Closing the decode gap to GB300 NVL72 — what the "TPU strikes the king" recipes buy us

Source: SemiAnalysis, *The Iron(wood) Throne* (2026-11-05, newsletter.semianalysis.com/p/tpu-strikes-the-king), plus the repos it links
(Inferact `tpu-megakernels`, `SemiAnalysisAI/InferenceX`, `google/tpu-sync`, upstream `vllm-torchtpu` main @ b08f10fdc).
Target: GB300 NVL72 Dynamo disagg on DSv4.1-Flash = **13.7–18.8K out tok/s per decode GPU @ 79–203 tok/s/user**. Ours today: 3.3K/chip @ 52/user.

> [!IMPORTANT]
> The article has **no TPU submission for DeepSeek-V4.1-Flash**. Its TPU results are Qwen3.5-397B, GLM-5.2/5.3, Kimi K3 and Muse Spark. What transfers is the *technique list*, not a config we can copy. Everything below is evaluated against our measured decode model: `step_ms = 25 + 0.163 × tokens/rank` (DP16 ICI, K=5, τ≈2.9).

## 1. Why interactivity matters for this comparison

At a fixed per-user speed `u` with acceptance τ, the step budget is `τ/u`; the sequences a rank can carry are `(τ/u − F)/(S·(K+1))`. With F=25 ms, S=0.163 ms, K=5:

| Target u (tok/s/user) | step budget | seqs/rank | out tok/s per chip (2 ranks) | GB300 Dynamo | gap |
|---|---|---|---|---|---|
| 52 (our best point) | 56 ms | 32 | 3,341 | — | — |
| 79 | 37 ms | 12 | 1,930 | 13.7K | 7× |
| 150 | 19 ms | **0** (below floor) | — | ~16K | ∞ |

So the gap to GB300 at *its* interactivity is entirely the fixed floor F and the per-token slope S. Bigger batches do not help there; they only raise the low-interactivity peak.

## 2. Article recipes → our stack

| # | Recipe (article) | Who / model | What it attacks | Status in our DSv4.1 stack | Expected effect for us |
|---|---|---|---|---|---|
| R1 | **Decode megakernel** — whole decode step in one Pallas kernel, next-layer weight prefetch, collectives inside | Inferact (Kimi K3 TP32, Qwen3.8, Muse Spark); Google (Qwen3.5) | per-step launch/idle gaps = **our 25 ms floor** | **Absent.** Upstream `vllm_torchtpu/megakernel*` is a framework with `SHORTHANDS = {}` (no model templates); Inferact's repo has Kimi/Qwen/Muse only. A DSv4.1 (DSA+CSA+SWA+MoE+Engram) megakernel is a multi-week build. | Largest lever. F 25→~5 ms ⇒ at 79 tok/s/user: 1.9K→**5.1K**/chip (2.7×); at 52/user: 3.3K→5.0K. Cannot be run tonight. |
| R2 | **DSpark acceptance** — InferenceX GB300 rows use `draft_sample_method=probabilistic`, `rejection_sample_method=block`, measured AL **4.07** (thinking off) / 3.51 (thinking on) at K=5 | InferenceX golden `deepseek-v4.1-flash__dspark__native__probabilistic-block.yaml` | tokens per step at zero step-time cost | TPU DSpark: *"only greedy drafting of a fixed block"* (`spec_decode/dspark.py`), standard per-token rejection; our τ = 2.9 (synthetic) / 3.18–3.27 (real prompts) | τ 3.2→4.07 at 79 tok/s/user ⇒ step budget 37→51 ms ⇒ seqs 12→27 ⇒ **1.9K→4.3K**/chip (2.2×). Same order as the megakernel, and it is a sampler change, not a kernel. Tonight: *measure* τ at thinking-off / temp 1.0 vs golden to size the headroom; block verification on TPU = follow-up PR. |
| R3 | **Admission geometry** — upstream's own DSv4-Flash DP8 nightly runs `MAX_NUM_SEQS=256, MNBT=1024` | upstream `scripts/vllm/benchmarking/configs/deepseek-v4-flash-dp8-ep.sh` | low-interactivity peak | Our branch dies at MNBT 768 (`ShapeSpecializationError s72 ≥ 385`). **Root cause found**: `deepseek_v41/attention.py` switches from the fused CSA decode weights (`_fused_csa_w_decode`) to plain `w_qkv_a` above 384 tokens, changing the graph's placeholder set (1251 vs 1259), which the shape-variants warmup cannot hand across. **Fixed** with `VLLM_DSV41_CSA_FUSED_MAX_TOKENS` (default 384; set = MNBT). | No gain at ≥79 tok/s/user. Raises the 25 tok/s/user peak from ~26K to an estimated 40–45K engine-total (DP16) / 5–5.5K per chip. Runs tonight. |
| R4 | **DCP (decode context parallel)** — split the shared compressed KV by sequence position, merge partials per layer | GLM-5.3 (DSA/MLA, same family as DSv4.1) | KV capacity, long-context attention time | `decode_context_parallel_size` plumbing exists; our attention cost is **context-independent** (2K vs 100K: same slope) and KV (6.0M tok/rank) is not the limiter at 20K–100K ctx | ~0 for throughput at our shapes; matters only for >100K-ctx fleets. Not run. |
| R5 | **SparseCore gather of indexer-selected KV**, 132-byte indexer layout | GLM-5.3 | indexer/attention bytes | SparseCore offload flags already on (`--xla_tpu_offload_gather_to_sparsecore`, SC collective offload); `VLLM_DSV41_INDEXER_CAND_READ_16K`, fp8 RoPE cache packing (`#2206`) present | Already harvested; attention is not on the critical path. |
| R6 | **Fused routing + experts + all-reduce + attention merge as one collective** | GLM-5.3 | MoE dispatch | `USE_MOE_FUSED_EP_KERNEL=1 v2 SHARDED_PLAN W4A8 FP8_DISPATCH` is our equivalent; upstream adds unified counting-sort routing (`#2604`) and f32 top-k (`#2673`) we lack | Slope S 0.163 → maybe 0.14 (small). Port candidates for a later A/B. |
| R7 | **4- and 8-token compile shapes** | GLM-5.3 | tiny decode steps (≤8 tok) | we run 48–384 tok/step | Irrelevant at our batch sizes. |
| R8 | **Host-DRAM KV offload (TPU-Sync)**, SGLang HiCache | Qwen3.5, Muse Spark | prefix hit rate / KV capacity on 90 %-cached AgentX | Raiden Stage-3 + `offload/raiden_store.py` exist; hit rate 67.7 % observed vs 71 % target | Prefill-side; no decode-step effect. Deferred. |
| R9 | **llm-d cache-aware routing** (exact DP-rank stickiness 56→76 % hit), **per-decode-host request cap 1…112**, xP-yD with 24–32 chips | Qwen3.5 | TTFT, interactivity curve shape | We pin ranks in the harness; 1P1D only; E2 (1P2D/1P4D) deferred by you | Multi-D gives more *points on the curve* (we showed DP16 scales linearly), not more per-chip. Revisit after R1/R2. |
| R10 | Kimi K3 TP32 across 8 hosts + megakernel, 709 tok/s single user | Inferact | the ultra-interactive end | n/a (needs megakernel) | Not a throughput lever. |

**Net:** the two things that move us toward GB300 at GB300's interactivity are the per-step floor (R1) and acceptance length (R2). R3 raises the headline peak and is cheap, so it runs first tonight; R2 is measured tonight and becomes the next code item; R1 is the strategic item and needs a dedicated effort (or Inferact/upstream delivering a DSv4 template).

## 3. Experiments — results (2026-11-08, single host `tpu7x-8` = 4 chips, DP8/EP8; the 2x2x2 slice was GCE_STOCKOUT all evening, came up for 4 min and was taken)

Raw: `runs/pd/x1_results/x1_tau_probe_sh8.*`, `x1_r3a_k5_sh8_ctx4k.*`, `x1_r3b_k0_sh8.*`, `x1_r3b_k0_sh8_ctx600.*`.

### X-R2a — DSpark acceptance headroom (K=5, 64 real coding prompts per cell, conc 32)

| thinking | temp | ours: greedy draft + standard rejection | InferenceX golden: probabilistic draft + block verification |
|---|---|---|---|
| off | 1.0 | **3.23** | **4.07** |
| on | 1.0 | **3.11** | **3.51** |
| off | 0.0 | 4.17 | — |
| on | 0.0 | 3.61 | — |

At temperature 0 we reproduce the golden AL, so the drafter is not the problem; under sampling (what InferenceX runs) the verification method costs us 0.4 (thinking on) to 0.85 (off) tokens per step: **+13 % / +26 % tokens per step** available from a TPU implementation of probabilistic-draft + block verification. At fixed 79 tok/s/user this is worth ~1.3–2.2× per-chip output (step budget grows with τ). → next code item.

### X-R3a — MNBT 768 / 128 seqs per rank, K=5, with `VLLM_DSV41_CSA_FUSED_MAX_TOKENS=768` (ctx 4K, out 1024)

Boots clean: one trace covers 40…768, no `s72 <= 384` split, no refused buckets. Fit `step = 24.6 + 0.134·t` (floor unchanged).

| seqs/rank | tok/step/rank | step ms | τ | engine tok/s (4 chips) | per chip | per user |
|---|---|---|---|---|---|---|
| 32 | 192 | 50.4 | 3.09 | 15,721 | 3,930 | 61.4 |
| **64** | **384** | **76.2** | 3.10 | **20,837** | **5,209** | **40.7** |
| 128 | 768 | admission-bound (128×6 = MNBT; TTFT 23 s) | | | | |

**New single-host best: 20.8K tok/s = 5.2K/chip @ 41 tok/s/user (+46 % over the previous 14.3K peak).** The 32-seq point also improved (3.9K vs 3.2–3.6K/chip) — shorter ctx and no prefill-pod interference.

### X-R3b — upstream geometry: MNBT 1024 / 256 seqs per rank, no spec (ctx 600, out 4096)

| seqs/rank | step ms | engine tok/s | per chip | per user |
|---|---|---|---|---|
| 64 | 40.0 | 12,791 | 3,198 | 25.0 |
| 128 (pads to 256 bucket) | 68.2 | 15,021 | 3,755 | 14.7 |
| 256 | 80.9 | **25,324** | **6,331** | 12.4 |

Peak raw throughput 25.3K/host, but at 12 tok/s/user. K=5 at 64 seqs delivers 82 % of that at 3.3× the interactivity — spec decode stays on.

### X-R1a — device xprof of the decode step (rank 0, K=0, 64 seqs/rank = 64 tok/step, 1.04 s window = 24 steps)

Raw: `runs/perf/20261109_k0_c512_xprof/xplane_rank0_k0_c512.json` (from `rank0_*.xplane.pb`, device planes present this time).

| Where the 43 ms wall step goes | ms/step | share |
|---|---|---|
| **Device idle between steps** (host: `sample_tokens → async_output.wait → device synchronize → next dispatch`) | **10.3** | 24 % |
| Device busy — one XLA module `tt_jit_pallas…custom_kernel` (the whole forward is already a single program) | 32.3 | 75 % |
| └ fused MoE v2 Pallas kernel (`fused_moe/v2/kernel.py:2358`) | 12.9 | 30 % |
| └ other Pallas kernels (attention / indexer / compressor / mHC) | 8.7 | 20 % |
| └ `all_gather` (EP) 1.8 · `scatter-add` 1.4 · `cond` 1.7 · dense `mm`/`dot_general` 1.6 · misc ~3 | ~9.5 | 22 % |
| argmax sampling module | 0.55 | 1 % |

Reading against the step model (`floor ≈ 25 ms + 0.134 ms/token`):
- **~10 ms of the floor is host-side**: the step is dispatched, the host waits for the device, samples, and only then dispatches the next step. No inter-kernel launch gaps exist inside the step (it is one XLA program), so the "megakernel removes launch gaps" argument from the article does *not* apply to us the way it does to the GPU baselines — what we are missing is **step-to-step pipelining** (dispatch step N+1 before step N's sampling completes), which is a runner/scheduler change, not a kernel.
- **~22 ms of the floor is on-device fixed cost** at 64 tokens: MoE + attention Pallas kernels whose time barely scales with tokens → per-layer prologue / weight-DMA pipeline fill across 54 layers. This is the part a cross-layer weight-prefetching megakernel (Inferact-style) attacks.
- The MoE kernel is 30 % of the step at only 64 tokens, so it is both the floor and the slope owner; routing/kernel ports (R6) target it.

Projected per-chip at 79 tok/s/user if (a) step pipelining hides the 10 ms host gap: 1.9K → 3.6K; (b) plus τ 4.07: → 6.5K; (c) plus a prefetching megakernel halving the device floor: → ~9K vs 13.7K GB300.


- Floor is still ~25 ms in every configuration → the megakernel (R1) remains the only route to GB300-class interactivity.
- Two shipped/identified software levers: admission fix (done, +46 % peak) and verification method (+13–26 % tokens/step, to implement).
- Updated per-chip projection at 79 tok/s/user with both R2 (τ 4.07) and R1 (F≈5 ms): ≈7.5K/chip vs 13.7K on GB300 — the remaining factor is the 0.13–0.16 ms/token slope (MoE path), i.e. R6-class kernel work.

| ID | Status |
|---|---|
| X-R2a τ probe | done |
| X-R3a MNBT 768 K=5 | done |
| X-R3b MNBT 1024 K=0 | done |
| X-R1a device xprof | done (K=0 engine, 64 seqs/rank) |
| X-R3 on DP16 (2x2x2) | blocked on spot capacity |
| X-R6a routing-kernel ports | not started |

## 4. Code changes (`repos/vtt-combined`, branch `cpk/dsv41-mh-decode`)

- `d591c662d fix(dsv41): make the fused-CSA decode weight cut-over configurable so MNBT > 384 compiles` — `envs.py` (+`VLLM_DSV41_CSA_FUSED_MAX_TOKENS`, default 384 = previous behaviour), `deepseek_v41/attention.py` (4 thresholds). Verified: MNBT 768 and 1024 warm up with no refused buckets and pass GATE-EQ 5/5 + STAGE3.
- Uncommitted (benchmarks tooling, to fold into the bench commit): `scripts/pd/run_pd.py` `--decode-env/--prefill-env`, `scripts/pd/x1_tau_probe.py`.

```bash
cd /usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/repos/vtt-combined && git push origin cpk/dsv41-mh-decode
```

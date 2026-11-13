# DeepSeek-V4.1-Flash FP8 PD Disaggregation (1P1D) — Empirical ISL (50:20:12:9:9) Benchmark Report (95% Prefix Cache Hit)

## 1. Executive Summary

- **Architecture & Cluster (`8` TPU7x Chips = `1P + 1D`)**:
  - **1P Node (`2x2x1` = 4 chips / 8 TensorCores)**: 4 independent per-chip `TP=2, EP=2, DP=1` instances (`VLLM_DSV41_CED=1`, `VLLM_DSV41_PURE_20L_PREFILL=1` 20-layer Context Encoder, `max_num_batched_tokens = 16,384`, `PREFILL_4X_TP2_GPU_MEM=0.92` = `1,659` blocks/partition across disjoint `main`, `state`, `swa_0`, `swa_1`, `swa_2` `BlockPool` queues).
  - **1D Node (`2x2x1` = 4 chips / 8 TensorCores)**: `DP=8, TP=1, EP=8` (`VLLM_DSV41_CED=0`, DSpark `K=5` speculative decoding, `--async-scheduling`, `max_num_batched_tokens = 384`).
  - **KV Transfer**: `TPURaidenConnector` Stage-3 DCN transfer of the 35 encoder KV pools (`Layers 0..20`, `TPU_RAIDEN_POOL_TAGS_PER_LAYER=1` for `TP=2 → TP=1` resharding) with a 128-token SWA tail handoff (`VLLM_DSV41_PD_HANDOFF_TAIL_TOKENS=128`) replayed in a single step on `1D`.
- **Sustained `~93.1%–94.2%` Realized Prefix Cache Hit Across All Concurrency Levels**: `93.89%` (`n=8`), `94.22%` (`n=16`), `93.08%` (`n=32`), `93.62%` (`n=64`), and `93.74%` (`n=128`) across 20 distinct trunks (`5` per `TP=2` chip covering all 5 ISL buckets), with structured multi-turn tool-calling completions (`finish_reason="tool_calls"`) and exact sampling temperatures from `requests_mocked.jsonl.xz`.
- **Steady-State Window Throughput**:
  - **`n = 8`**: **908.7 tok/s output** and **9,270.8 tok/s total input** (566.8 tok/s uncached prefill).
  - **`n = 16`**: **1,520.1 tok/s output** and **24,325.7 tok/s total input** (1,405.1 tok/s uncached prefill).
  - **`n = 32`**: **2,852.1 tok/s output** and **42,023.8 tok/s total input** (2,907.0 tok/s uncached prefill).
  - **`n = 64`**: **5,001.6 tok/s output** and **75,316.6 tok/s total input** (4,808.5 tok/s uncached prefill).
  - **`n = 128`**: **6,197.7 tok/s output** and **116,289.2 tok/s total input** (7,278.7 tok/s uncached prefill) at natural EOS (`mean OSL = 1,211.5`), and **7,656.7 tok/s output** / **180,255.7 tok/s total input** with fixed `max_tokens = 1,024` (`40` pinned trunks).
- **Interactive Per-Stream Decode Speed & Sub-Second TTFT Across `1–8K` to `64K+`**:
  - **`n = 8–16`**: **89.95–91.23 tok/s median per stream** (`P90_hi = 113.0–117.5 tok/s`, Per-Token ITL = `10.96–11.12 ms`, StepChunk ITL = `26.73–28.53 ms`, DSpark $\tau = 3.02\text{–}3.24$) with **0.52 s–0.60 s median TTFT** (`0.70 s–0.78 s` P90; **0.70 s–0.77 s median TTFT** and **0.73 s–0.79 s P90 TTFT** on `64K+`).
  - **`n = 32–64`**: **69.72–78.71 tok/s median per stream** (Per-Token ITL = `12.71–14.34 ms`, DSpark $\tau = 3.13\text{–}3.22$) with **0.53 s–0.59 s median TTFT** (`0.81 s–1.26 s` P90; **0.74 s–0.76 s median TTFT** on `64K+`).
  - **`n = 128`**: **0.76 s median TTFT** (**0.89 s median / 1.36 s P90** on `64K+`) with DSpark $\tau = 3.05$.

---

## 2. End-to-End 1P1D Empirical ISL (50:20:12:9:9) Summary (95% Cache Hit Rate)

### 2.1 Aggregate Throughput & Workload Summary (`180 s` Steady-State Window)

| **n (Trunks)** | **OK Reqs (Tools / In-Win)** | **Mean ISL (Cache Hit %)** | **Mean OSL** | **Win Out (Drain tok/s)** | **Win In (Uncached tok/s)** |
| --- | --- | --- | --- | --- | --- |
| **8 (20 tr)** | 88/90 (25 / 80) | 18,963 (**93.89%**) | 2,155.7 | **908.7** (634.8) | **9,270.8** (566.8) |
| **16 (20 tr)** | 237/245 (71 / 221) | 18,503 (**94.22%**) | 1,356.9 | **1,520.1** (1,292.0) | **24,325.7** (1,405.1) |
| **32 (20 tr)** | 393/401 (131 / 361) | 19,265 (**93.08%**) | 1,512.0 | **2,852.1** (1,845.0) | **42,023.8** (2,907.0) |
| **64 (20 tr)** | 721/740 (230 / 657) | 18,876 (**93.62%**) | 1,431.6 | **5,001.6** (2,941.5) | **75,316.6** (4,808.5) |
| **128 (20 tr, Natural OSL)** | 1111/1138 (357 / 983) | 18,877 (**93.74%**) | 1,211.5 | **6,197.7** (3,874.1) | **116,289.2** (7,278.7) |
| **128 (40 tr, Fixed 1,024 OSL)** | 510/510 (230 / 382) | 21,431 (**94.30%**) | 1,024.0 | **7,656.7** (6,170.4) | **180,255.7** (10,357.1) |

### 2.2 Per-Stream Decode Speed, Latency (TTFT / ITL / E2E) & DSpark Acceptance

| **n (Trunks)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** | **Per-Tok ITL (StepChunk Med/Mean)** | **DSpark τ (Pos 0..4 Accept)** | **Med TTFT (P90)** | **64K+ Med TTFT (P90)** | **Med E2E** |
| --- | --- | --- | --- | --- | --- | --- |
| **8 (20 tr)** | **91.23** (46.4 / 77.3 / 113.0) | 10.96 ms (26.73 / 32.41 ms) | **3.24** (75/56/41/30/22%) | **0.60 s** (0.78 s) | **0.77 s** (0.79 s) | 6.54 s |
| **16 (20 tr)** | **89.95** (76.5 / 78.2 / 117.5) | 11.12 ms (28.53 / 31.69 ms) | **3.02** (73/51/36/25/17%) | **0.52 s** (0.70 s) | **0.70 s** (0.73 s) | 6.71 s |
| **32 (20 tr)** | **78.71** (65.3 / 67.7 / 110.0) | 12.71 ms (33.16 / 37.27 ms) | **3.22** (75/56/41/29/21%) | **0.53 s** (0.81 s) | **0.74 s** (0.98 s) | 6.84 s |
| **64 (20 tr)** | **69.72** (37.9 / 49.8 / 91.5) | 14.34 ms (34.83 / 44.03 ms) | **3.13** (74/54/38/27/19%) | **0.59 s** (1.26 s) | **0.76 s** (1.31 s) | 8.35 s |
| **128 (20 tr, Natural OSL)** | **46.44** (29.7 / 35.2 / 62.6) | 21.53 ms (42.72 / 66.59 ms) | **3.05** (73/52/37/25/18%) | **0.76 s** (4.22 s) | **0.89 s** (1.36 s) | 10.44 s |
| **128 (40 tr, Fixed 1,024 OSL)** | **64.64** (49.9 / 53.0 / 83.9) | 15.47 ms (42.05 / 48.74 ms) | **2.72** (61/42/30/22/17%) | **1.39 s** (3.12 s) | **1.56 s** (3.16 s) | 17.41 s |

---

## 3. Per-Bucket Empirical ISL Breakdown (50:20:12:9:9) — 95% Prefix Cache Hit

### 3.1 Cross-Concurrency Per-Bucket Median TTFT, P90 TTFT & Decode Speed Matrix

| **ISL Bucket (Target Share)** | **n = 8 (20 Trunks)** | **n = 16 (20 Trunks)** | **n = 32 (20 Trunks)** | **n = 64 (20 Trunks)** | **n = 128 (20 Trunks)** |
| --- | --- | --- | --- | --- | --- |
| **1–8K (50%)** | 0.56 s (P90 0.71 s) / 88.8 tok/s (92.0% hit) | 0.49 s (P90 0.57 s) / 87.7 tok/s (93.2% hit) | 0.51 s (P90 0.74 s) / 76.6 tok/s (92.7% hit) | 0.53 s (P90 1.14 s) / 68.3 tok/s (92.8% hit) | 0.70 s (P90 3.71 s) / 45.7 tok/s (93.1% hit) |
| **8–16K (20%)** | 0.62 s (P90 1.38 s) / 95.4 tok/s (92.7% hit) | 0.53 s (P90 0.60 s) / 88.9 tok/s (93.4% hit) | 0.54 s (P90 0.82 s) / 77.3 tok/s (93.0% hit) | 0.57 s (P90 1.16 s) / 70.7 tok/s (93.3% hit) | 0.74 s (P90 3.64 s) / 46.3 tok/s (92.5% hit) |
| **16–32K (12%)** | 0.62 s (P90 0.64 s) / 98.4 tok/s (94.7% hit) | 0.55 s (P90 0.68 s) / 114.4 tok/s (94.8% hit) | 0.58 s (P90 0.71 s) / 94.5 tok/s (94.9% hit) | 0.59 s (P90 1.25 s) / 75.2 tok/s (94.9% hit) | 0.74 s (P90 3.70 s) / 55.8 tok/s (95.0% hit) |
| **32–64K (9%)** | 0.71 s (P90 2.26 s) / 105.8 tok/s (94.4% hit) | 0.65 s (P90 0.69 s) / 107.6 tok/s (94.5% hit) | 0.65 s (P90 0.89 s) / 93.8 tok/s (92.6% hit) | 0.70 s (P90 1.21 s) / 83.9 tok/s (92.6% hit) | 0.82 s (P90 4.38 s) / 57.6 tok/s (93.4% hit) |
| **64K+ (9%)** | 0.77 s (P90 0.79 s) / 95.2 tok/s (94.8% hit) | 0.70 s (P90 0.73 s) / 94.5 tok/s (94.7% hit) | 0.74 s (P90 0.98 s) / 89.8 tok/s (92.7% hit) | 0.76 s (P90 1.31 s) / 75.0 tok/s (94.6% hit) | 0.89 s (P90 1.36 s) / 54.9 tok/s (94.4% hit) |

### 3.2 n = 8 Sessions (20 Trunks: 4:4:4:4:4 Across 4 × TP=2 Chips)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 4 | 45 (51.1%) | 6,678 | 465.3 | 92.0% | **0.56 s** | 0.71 s | 5.69 s | **88.8** (41.0 / 57.7 / 100.0) |
| **8–16K** | 20% | 4 | 18 (20.5%) | 13,249 | 6,469.6 | 92.7% | **0.62 s** | 1.38 s | 22.53 s | **95.4** (73.9 / 78.5 / 131.3) |
| **16–32K** | 12% | 4 | 11 (12.5%) | 24,875 | 1,040.1 | 94.7% | **0.62 s** | 0.64 s | 7.78 s | **98.4** (83.8 / 87.6 / 116.7) |
| **32–64K** | 9% | 4 | 8 (9.1%) | 51,093 | 1,343.3 | 94.4% | **0.71 s** | 2.26 s | 4.35 s | **105.8** (86.7 / 92.0 / 141.9) |
| **64K+** | 9% | 4 | 6 (6.8%) | 74,566 | 5,021.5 | 94.8% | **0.77 s** | 0.79 s | 64.33 s | **95.2** (87.3 / 88.6 / 131.8) |

### 3.3 n = 16 Sessions (20 Trunks: 4:4:4:4:4 Across 4 × TP=2 Chips)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 4 | 123 (51.9%) | 6,590 | 496.4 | 93.2% | **0.49 s** | 0.57 s | 5.28 s | **87.7** (74.6 / 76.8 / 105.4) |
| **8–16K** | 20% | 4 | 49 (20.7%) | 12,434 | 2,838.9 | 93.4% | **0.53 s** | 0.60 s | 17.85 s | **88.9** (76.9 / 78.2 / 119.5) |
| **16–32K** | 12% | 4 | 29 (12.2%) | 24,542 | 1,171.6 | 94.8% | **0.55 s** | 0.68 s | 7.04 s | **114.4** (82.9 / 83.6 / 139.4) |
| **32–64K** | 9% | 4 | 22 (9.3%) | 53,820 | 1,907.4 | 94.5% | **0.65 s** | 0.69 s | 6.70 s | **107.6** (78.2 / 82.1 / 143.4) |
| **64K+** | 9% | 4 | 14 (5.9%) | 76,395 | 3,247.9 | 94.7% | **0.70 s** | 0.73 s | 20.75 s | **94.5** (82.8 / 83.1 / 109.6) |

### 3.4 n = 32 Sessions (20 Trunks: 4:4:4:4:4 Across 4 × TP=2 Chips)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 4 | 201 (51.1%) | 6,627 | 479.7 | 92.7% | **0.51 s** | 0.74 s | 5.10 s | **76.6** (64.3 / 66.3 / 91.9) |
| **8–16K** | 20% | 4 | 80 (20.4%) | 12,591 | 3,428.4 | 93.0% | **0.54 s** | 0.82 s | 19.82 s | **77.3** (64.6 / 68.0 / 108.2) |
| **16–32K** | 12% | 4 | 48 (12.2%) | 25,319 | 814.2 | 94.9% | **0.58 s** | 0.71 s | 7.75 s | **94.5** (70.4 / 71.7 / 126.7) |
| **32–64K** | 9% | 4 | 36 (9.2%) | 52,153 | 1,498.7 | 92.6% | **0.65 s** | 0.89 s | 5.00 s | **93.8** (68.8 / 72.3 / 126.3) |
| **64K+** | 9% | 4 | 28 (7.1%) | 76,389 | 4,660.9 | 92.7% | **0.74 s** | 0.98 s | 14.57 s | **89.8** (74.0 / 74.8 / 134.5) |

### 3.5 n = 64 Sessions (20 Trunks: 4:4:4:4:4 Across 4 × TP=2 Chips)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 4 | 370 (51.3%) | 6,620 | 467.7 | 92.8% | **0.53 s** | 1.14 s | 6.28 s | **68.3** (31.3 / 46.0 / 81.6) |
| **8–16K** | 20% | 4 | 148 (20.5%) | 12,536 | 3,379.8 | 93.3% | **0.57 s** | 1.16 s | 25.84 s | **70.7** (45.6 / 54.9 / 92.7) |
| **16–32K** | 12% | 4 | 89 (12.3%) | 24,045 | 715.9 | 94.9% | **0.59 s** | 1.25 s | 8.45 s | **75.2** (38.9 / 56.3 / 105.3) |
| **32–64K** | 9% | 4 | 67 (9.3%) | 53,402 | 1,699.7 | 92.6% | **0.70 s** | 1.21 s | 7.82 s | **83.9** (52.1 / 57.7 / 115.9) |
| **64K+** | 9% | 4 | 47 (6.5%) | 76,311 | 3,857.5 | 94.6% | **0.76 s** | 1.31 s | 29.85 s | **75.0** (45.2 / 60.7 / 117.7) |

### 3.6 n = 128 Sessions (20 Trunks: 4:4:4:4:4 Across 4 × TP=2 Chips)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 4 | 569 (51.2%) | 6,591 | 419.1 | 93.1% | **0.70 s** | 3.71 s | 8.10 s | **45.7** (27.0 / 34.3 / 54.4) |
| **8–16K** | 20% | 4 | 228 (20.5%) | 12,414 | 2,860.7 | 92.5% | **0.74 s** | 3.64 s | 27.95 s | **46.3** (32.8 / 36.1 / 60.3) |
| **16–32K** | 12% | 4 | 137 (12.3%) | 24,245 | 762.3 | 95.0% | **0.74 s** | 3.70 s | 11.39 s | **55.8** (35.4 / 39.3 / 71.2) |
| **32–64K** | 9% | 4 | 102 (9.2%) | 52,157 | 1,643.5 | 93.4% | **0.82 s** | 4.38 s | 10.03 s | **57.6** (30.6 / 38.0 / 74.0) |
| **64K+** | 9% | 4 | 75 (6.8%) | 76,666 | 2,442.3 | 94.4% | **0.89 s** | 1.36 s | 21.16 s | **54.9** (38.3 / 39.9 / 78.5) |

---

## 4. Benchmarking Methodology & Code Artifacts

### 4.1 Methodology Overview
- **Workload Distribution (`50:20:12:9:9`)**: Requests are sampled from `requests_mocked.jsonl.xz` across five empirical ISL buckets (`1–8K: 50%`, `8–16K: 20%`, `16–32K: 12%`, `32–64K: 9%`, `64K+: 9%`, weighted mean ISL `~19.0K–21.5K` tokens), preserving multi-turn tool-calling schemas and cycling the exact 30-entry `(temperature, top_p)` sampling schedule from the dataset.
- **Prefix Cache Trunk Seeding (`95%` Target Hit)**: Each request splits into a shared cached prefix trunk (`~93.1–94.3%` realized token-level hit aligned to `1,024`-token KV retention boundaries) and a unique per-request uncached suffix (`~5.7–6.9%`). Before each concurrency level `n`, `/reset_prefix_cache` clears `1P` and `1D`, and Phase-0 warmup seeds `20` distinct trunks (`5` per `TP=2` chip) pinned deterministically across the prefill instances.
- **Closed-Loop Steady-State Window & Drain**: After an initial XLA compilation warmup pass, each concurrency level `n` runs `n` persistent closed-loop session workers for a `180 s` active dispatch window (measuring steady-state window throughput and `1D` Prometheus `vllm:spec_decode_*` counters), followed by full drain of all in-flight requests.

### 4.2 Code & Data Artifacts
- **GitHub Branch (`vllm-project/vllm-torchtpu`)**: [`cpk/dsv41-combined` (`benchmarks/dsv41_tpu7x`)](https://github.com/vllm-project/vllm-torchtpu/tree/cpk/dsv41-combined/benchmarks/dsv41_tpu7x)
- **Empirical ISL Benchmark Harness**: [`bench_empirical_isl_dsv41.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/bench_empirical_isl_dsv41.py)
- **1P1D Orchestrator & Pod Launcher**: [`run_pd.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/run_pd.py), [`in_pod_serve.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/in_pod_serve.sh), and [`launch_dsv41_disagg_role.sh`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/repos/vtt-combined/examples/disagg/launch_dsv41_disagg_role.sh)
- **Disaggregated DP-Aware Proxy**: [`toy_proxy_server.py`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/repos/vtt-combined/examples/disagg/toy_proxy_server.py)
- **Consolidated JSON Summary & Logs**: [`dsv41_pd_empirical_isl_95_p2_summary.json`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/runs/pd/20261030-204736_dsv41_pd_4xtp2_20l_disjoint_p2/dsv41_pd_empirical_isl_95_p2_summary.json) and `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/phase2_4xtp2_1p1d_20261030/`

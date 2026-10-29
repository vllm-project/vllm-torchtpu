# DeepSeek-V4.1-Flash FP8 PD Disaggregation (1P1D, DP=8) — Empirical ISL (50:20:12:9:9) Benchmark Report (95% Prefix Cache Hit)

## 1. Executive Summary

- **Architecture & Cluster (`8` TPU7x Chips = `1P + 1D`)**: 1 Prefill node (`2x2x1` = 4 chips / 8 TensorCores, `DP=8, TP=1, EP=8`, `CED=1`) + 1 Decode node (`2x2x1` = 4 chips / 8 TensorCores, `DP=8, TP=1, EP=8`, `CED=0`, DSpark `K=5` speculative decoding, `--async-scheduling`) connected via `TPURaidenConnector` Stage-3 KV transfer over DCN.
- **Zero Failed Requests Across All Concurrency Sweeps**: 48/48 (`n=8`), 94/94 (`n=16`), 151/151 (`n=32`), 274/274 (`n=64`), 462/462 (`n=128`, 128 trunks), and 510/510 (`n=128`, 40 pinned trunks) with structured multi-turn tool-calling completions (`finish_reason="tool_calls"`) and exact sampling temperatures from `requests_mocked.jsonl.xz`.
- **Peak Steady-State Throughput (`n = 128`)**:
  - **Pinned HBM Trunks (40 distinct trunks, 94.3% cache hit)**: **7,656.7 tok/s output** (6,170.4 tok/s full-drain) and **180,255.7 tok/s total input** (10,357.1 tok/s uncached prefill).
  - **5-or-N Trunks (128 distinct trunks, 82.1% cache hit)**: **6,750.6 tok/s output** (5,715.9 tok/s full-drain) and **160,749.3 tok/s total input** (29,677.3 tok/s uncached prefill).
- **Interactive Per-Stream Decode Speed & TTFT**:
  - **n = 8–16**: **94.7–102.0 tok/s median per stream** (P90_hi = 122.6–126.6 tok/s, Per-Token ITL = 9.80–10.56 ms, StepChunk ITL = 27.03–28.88 ms, DSpark τ = 2.73–2.81) with **0.43 s–0.56 s median TTFT** (1.03 s–1.11 s P90; 0.39 s–0.42 s on 1–8K and 1.06 s–1.07 s on 64K+ at ~84K–90K ISL).
  - **n = 32–64**: **71.3–79.9 tok/s median per stream** (Per-Token ITL = 12.51–14.03 ms, DSpark τ = 2.75–2.76) with **0.82 s–1.10 s median TTFT** and **2,329.5–4,176.1 tok/s window output throughput**.
  - **n = 128**: **64.6–64.7 tok/s median per stream** (P95_low = 49.9–50.0 tok/s, P90_hi = 83.9–86.0 tok/s, Per-Token ITL = 15.45–15.47 ms, DSpark τ = 2.72–2.73) with **1.39 s median TTFT (3.12 s P90)** at 40 trunks and **2.01 s median TTFT (9.51 s P90)** at 128 trunks.

---

## 2. End-to-End 1P1D Empirical ISL (50:20:12:9:9) Summary (95% Cache Hit Rate)

### 2.1 Aggregate Throughput & Workload Summary

| **n (Trunks)** | **OK Reqs (Tools / In-Win)** | **Mean ISL (Cache Hit)** | **Mean OSL** | **Win Out (Drain tok/s)** | **Win In (Uncached tok/s)** |
| --- | --- | --- | --- | --- | --- |
| **8 (40 tr)** | 48/48 (17 / 40) | 20,849 (93.9%) | 1,024.0 | **769.8** (706.7) | **16,679.2** (1,012.0) |
| **16 (40 tr)** | 94/94 (31 / 78) | 21,725 (94.3%) | 1,024.0 | **1,475.5** (1,350.8) | **33,226.2** (1,908.8) |
| **32 (40 tr)** | 151/151 (63 / 119) | 21,446 (94.4%) | 1,024.0 | **2,329.5** (2,079.8) | **51,876.8** (2,946.6) |
| **64 (64 tr)** | 274/274 (115 / 210) | 21,267 (94.2%) | 1,024.0 | **4,176.1** (3,780.3) | **96,073.2** (5,602.9) |
| **128 (128 tr, 5-or-N)** | 462/462 (201 / 334) | 21,860 (82.1%) | 1,024.0 | **6,750.6** (5,715.9) | **160,749.3** (29,677.3) |
| **128 (40 tr, Pinned)** | 510/510 (230 / 382) | 21,431 (94.3%) | 1,024.0 | **7,656.7** (6,170.4) | **180,255.7** (10,357.1) |

### 2.2 Per-Stream Decode Speed, Latency (TTFT / ITL / E2E) & DSpark Acceptance

| **n (Trunks)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** | **Per-Tok ITL (StepChunk Med/Mean)** | **DSpark τ (Pos 0..4 Accept)** | **Med TTFT (P90)** | **Med E2E** |
| --- | --- | --- | --- | --- | --- |
| **8 (40 tr)** | **102.02** (79.5 / 81.4 / 126.6) | 9.80 ms (27.03 / 30.49 ms) | **2.73** (61/42/30/22/17%) | **0.43 s** (1.03 s) | 10.80 s |
| **16 (40 tr)** | **94.72** (79.2 / 83.3 / 122.6) | 10.56 ms (28.88 / 32.09 ms) | **2.81** (62/44/32/24/18%) | **0.56 s** (1.11 s) | 11.48 s |
| **32 (40 tr)** | **79.93** (57.7 / 62.4 / 106.2) | 12.51 ms (33.68 / 41.10 ms) | **2.76** (61/43/31/23/18%) | **0.82 s** (4.47 s) | 13.98 s |
| **64 (64 tr)** | **71.26** (48.8 / 50.9 / 98.9) | 14.03 ms (34.83 / 45.52 ms) | **2.75** (61/43/31/23/17%) | **1.10 s** (4.82 s) | 15.57 s |
| **128 (128 tr, 5-or-N)** | **64.71** (50.0 / 52.5 / 86.0) | 15.45 ms (42.51 / 49.81 ms) | **2.73** (61/42/31/23/17%) | **2.01 s** (9.51 s) | 18.85 s |
| **128 (40 tr, Pinned)** | **64.64** (49.9 / 53.0 / 83.9) | 15.47 ms (42.05 / 48.74 ms) | **2.72** (61/42/30/22/17%) | **1.39 s** (3.12 s) | 17.41 s |

---

## 3. Per-Bucket Empirical ISL Breakdown (50:20:12:9:9) — 95% Prefix Cache Hit

### 3.1 Cross-Concurrency Per-Bucket Median TTFT & Decode Speed Matrix

| **ISL Bucket (Target Share)** | **n = 8 (40 Trunks)** | **n = 16 (40 Trunks)** | **n = 32 (40 Trunks)** | **n = 64 (64 Trunks)** | **n = 128 (128 Trunks)** | **n = 128 (40 Pinned Trunks)** |
| --- | --- | --- | --- | --- | --- | --- |
| **1–8K (50%)** | 0.39 s / 99.1 tok/s (8 tr) | 0.42 s / 93.7 tok/s (8 tr) | 0.53 s / 76.1 tok/s (8 tr) | 0.90 s / 71.2 tok/s (32 tr) | 1.77 s / 61.6 tok/s (64 tr) | 1.24 s / 62.6 tok/s (8 tr) |
| **8–16K (20%)** | 0.41 s / 97.9 tok/s (8 tr) | 0.46 s / 99.3 tok/s (8 tr) | 0.61 s / 80.2 tok/s (8 tr) | 0.98 s / 73.4 tok/s (8 tr) | 1.83 s / 58.6 tok/s (24 tr) | 1.27 s / 60.0 tok/s (8 tr) |
| **16–32K (12%)** | 0.50 s / 104.0 tok/s (8 tr) | 0.44 s / 104.8 tok/s (8 tr) | 0.63 s / 81.8 tok/s (8 tr) | 0.93 s / 76.7 tok/s (8 tr) | 1.75 s / 77.2 tok/s (16 tr) | 1.17 s / 73.0 tok/s (8 tr) |
| **32–64K (9%)** | 0.96 s / 108.2 tok/s (8 tr) | 0.93 s / 94.4 tok/s (8 tr) | 0.97 s / 83.1 tok/s (8 tr) | 1.10 s / 79.6 tok/s (8 tr) | 2.00 s / 70.6 tok/s (16 tr) | 1.24 s / 68.7 tok/s (8 tr) |
| **64K+ (9%)** | 1.06 s / 113.0 tok/s (8 tr) | 1.07 s / 99.1 tok/s (8 tr) | 1.38 s / 86.5 tok/s (8 tr) | 1.52 s / 73.9 tok/s (8 tr) | 2.14 s / 68.6 tok/s (8 tr) | 1.56 s / 62.6 tok/s (8 tr) |

### 3.2 n = 8 Sessions (40 Trunks: 8:8:8:8:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 8 | 24 (50.0%) | 7,342 | 1,024.0 | 92.4% | **0.39 s** | 0.85 s | 11.02 s | **99.1** (75.7 / 78.5 / 125.8) |
| **8–16K** | 20% | 8 | 10 (20.8%) | 12,009 | 1,024.0 | 93.8% | **0.41 s** | 0.57 s | 10.91 s | **97.9** (84.1 / 87.3 / 111.9) |
| **16–32K** | 12% | 8 | 6 (12.5%) | 24,469 | 1,024.0 | 94.9% | **0.50 s** | 0.82 s | 10.47 s | **104.0** (81.7 / 83.9 / 136.1) |
| **32–64K** | 9% | 8 | 4 (8.3%) | 49,083 | 1,024.0 | 94.4% | **0.96 s** | 1.05 s | 10.41 s | **108.2** (91.2 / 93.6 / 128.6) |
| **64K+** | 9% | 8 | 4 (8.3%) | 90,324 | 1,024.0 | 94.1% | **1.06 s** | 1.15 s | 10.13 s | **113.0** (88.8 / 92.9 / 150.7) |

### 3.3 n = 16 Sessions (40 Trunks: 8:8:8:8:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 8 | 47 (50.0%) | 7,103 | 1,024.0 | 92.0% | **0.42 s** | 1.06 s | 11.52 s | **93.7** (78.1 / 83.5 / 121.1) |
| **8–16K** | 20% | 8 | 19 (20.2%) | 13,885 | 1,024.0 | 95.5% | **0.46 s** | 1.07 s | 10.84 s | **99.3** (77.0 / 79.6 / 115.8) |
| **16–32K** | 12% | 8 | 11 (11.7%) | 25,839 | 1,024.0 | 95.5% | **0.44 s** | 1.08 s | 10.20 s | **104.8** (84.2 / 88.8 / 134.4) |
| **32–64K** | 9% | 8 | 9 (9.6%) | 54,172 | 1,024.0 | 94.5% | **0.93 s** | 1.13 s | 11.71 s | **94.4** (83.0 / 83.3 / 105.4) |
| **64K+** | 9% | 8 | 8 (8.5%) | 84,084 | 1,024.0 | 94.2% | **1.07 s** | 1.33 s | 11.51 s | **99.1** (82.2 / 84.3 / 145.2) |

### 3.4 n = 32 Sessions (40 Trunks: 8:8:8:8:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 8 | 75 (49.7%) | 7,054 | 1,024.0 | 91.8% | **0.53 s** | 1.43 s | 14.14 s | **76.1** (57.6 / 59.6 / 106.2) |
| **8–16K** | 20% | 8 | 30 (19.9%) | 12,986 | 1,024.0 | 94.9% | **0.61 s** | 1.51 s | 13.86 s | **80.2** (60.3 / 63.6 / 95.5) |
| **16–32K** | 12% | 8 | 18 (11.9%) | 25,097 | 1,024.0 | 95.4% | **0.63 s** | 1.22 s | 13.11 s | **81.8** (66.9 / 72.2 / 120.2) |
| **32–64K** | 9% | 8 | 14 (9.3%) | 53,923 | 1,024.0 | 94.8% | **0.97 s** | 4.47 s | 13.75 s | **83.1** (66.1 / 69.0 / 107.3) |
| **64K+** | 9% | 8 | 14 (9.3%) | 79,502 | 1,024.0 | 94.7% | **1.38 s** | 4.62 s | 13.43 s | **86.5** (75.4 / 77.4 / 101.5) |

### 3.5 n = 64 Sessions (64 Trunks: 32:8:8:8:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 32 | 137 (50.0%) | 7,068 | 1,024.0 | 92.0% | **0.90 s** | 1.76 s | 15.39 s | **71.2** (48.4 / 49.8 / 90.1) |
| **8–16K** | 20% | 8 | 55 (20.1%) | 12,950 | 1,024.0 | 94.5% | **0.98 s** | 3.88 s | 15.10 s | **73.4** (49.4 / 56.6 / 95.2) |
| **16–32K** | 12% | 8 | 33 (12.0%) | 26,318 | 1,024.0 | 95.2% | **0.93 s** | 1.60 s | 14.45 s | **76.7** (51.0 / 57.8 / 117.4) |
| **32–64K** | 9% | 8 | 25 (9.1%) | 51,736 | 1,024.0 | 94.6% | **1.10 s** | 5.09 s | 14.46 s | **79.6** (62.2 / 65.4 / 96.8) |
| **64K+** | 9% | 8 | 24 (8.8%) | 82,694 | 1,024.0 | 94.4% | **1.52 s** | 5.27 s | 15.52 s | **73.9** (62.1 / 64.7 / 102.9) |

### 3.6 n = 128 Sessions (128 Trunks 5-or-N: 64:24:16:16:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 64 | 231 (50.0%) | 7,125 | 1,024.0 | 89.6% | **1.77 s** | 7.11 s | 19.00 s | **61.6** (47.4 / 51.4 / 75.7) |
| **8–16K** | 20% | 24 | 92 (19.9%) | 12,950 | 1,024.0 | 88.6% | **1.83 s** | 8.19 s | 20.25 s | **58.6** (47.1 / 49.9 / 75.5) |
| **16–32K** | 12% | 16 | 55 (11.9%) | 26,217 | 1,024.0 | 88.1% | **1.75 s** | 7.19 s | 15.54 s | **77.2** (55.5 / 60.1 / 95.0) |
| **32–64K** | 9% | 16 | 42 (9.1%) | 53,362 | 1,024.0 | 78.3% | **2.00 s** | 6.65 s | 17.64 s | **70.6** (52.7 / 56.1 / 87.8) |
| **64K+** | 9% | 8 | 42 (9.1%) | 85,218 | 1,024.0 | 76.5% | **2.14 s** | 9.51 s | 18.14 s | **68.6** (54.7 / 56.9 / 85.2) |

### 3.7 n = 128 Sessions (40 Pinned Trunks: 8:8:8:8:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 8 | 255 (50.0%) | 7,006 | 1,024.0 | 92.0% | **1.24 s** | 2.75 s | 17.94 s | **62.6** (46.0 / 49.7 / 80.1) |
| **8–16K** | 20% | 8 | 102 (20.0%) | 13,088 | 1,024.0 | 94.5% | **1.27 s** | 3.08 s | 18.62 s | **60.0** (43.8 / 46.7 / 74.9) |
| **16–32K** | 12% | 8 | 61 (12.0%) | 25,735 | 1,024.0 | 95.4% | **1.17 s** | 3.09 s | 15.78 s | **73.0** (49.4 / 52.2 / 101.7) |
| **32–64K** | 9% | 8 | 46 (9.0%) | 52,024 | 1,024.0 | 94.7% | **1.24 s** | 3.15 s | 16.71 s | **68.7** (42.0 / 52.1 / 85.9) |
| **64K+** | 9% | 8 | 46 (9.0%) | 83,595 | 1,024.0 | 94.5% | **1.56 s** | 3.16 s | 18.03 s | **62.6** (47.1 / 51.3 / 79.9) |

---

## 4. Benchmarking Methodology & Code Artifacts

### 4.1 Methodology Overview
- **Workload Distribution (`50:20:12:9:9`)**: Requests are sampled from `requests_mocked.jsonl.xz` across five empirical ISL buckets (`1–8K: 50%`, `8–16K: 20%`, `16–32K: 12%`, `32–64K: 9%`, `64K+: 9%`, weighted mean ISL `~21.5K` tokens) with `max_tokens = 1,024` (`ignore_eos = True`), preserving multi-turn tool-calling schemas and cycling the exact 30-entry `(temperature, top_p)` sampling schedule from the dataset.
- **Prefix Cache Trunk Seeding (`95%` Target Hit)**: Each request splits into a shared cached prefix trunk (`~94.3%` realized token-level hit aligned to `1,024`-token KV retention boundaries) and a unique per-request uncached suffix (`~5.7%`). Before each concurrency level `n`, `/reset_prefix_cache` clears `1P` and `1D`, and Phase-0 warmup seeds `40` (`n <= 32` or pinned `n = 128`), `64` (`n = 64`), or `128` (`n = 128` `5-or-N`) distinct trunks pinned deterministically across the 8 DP ranks via `X-data-parallel-rank`.
- **Closed-Loop Steady-State Window & Drain**: After an initial XLA compilation warmup pass, each concurrency level `n` runs `n` persistent closed-loop session workers for a `60 s` active dispatch window (measuring steady-state window throughput and `1D` Prometheus `vllm:spec_decode_*` counters), followed by full drain of all in-flight requests.

### 4.2 Code & Data Artifacts
- **GitHub Branch (`vllm-project/vllm-torchtpu`)**: [`cpk/dsv41-combined` (`benchmarks/dsv41_tpu7x`)](https://github.com/vllm-project/vllm-torchtpu/tree/cpk/dsv41-combined/benchmarks/dsv41_tpu7x)
- **Empirical ISL Benchmark Harness**: [`bench_empirical_isl_dsv41.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/bench_empirical_isl_dsv41.py)
- **1P1D Orchestrator & Pod Launcher**: [`run_pd.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/run_pd.py) and [`in_pod_serve.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/in_pod_serve.sh)
- **Disaggregated DP-Aware Proxy**: [`toy_proxy_server.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/examples/disagg/toy_proxy_server.py)
- **Consolidated JSON Summary & Logs**: [`dsv41_pd_empirical_isl_95_summary.json`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/dsv41_pd_empirical_isl_95_summary.json) and `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/empirical_isl_95_1p1d_20261029/`

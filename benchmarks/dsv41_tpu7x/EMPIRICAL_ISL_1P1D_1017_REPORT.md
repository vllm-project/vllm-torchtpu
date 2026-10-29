# DeepSeek-V4.1-Flash-1017 FP8 PD Disaggregation (1P1D, DP=8) — Empirical ISL (50:20:12:9:9) Benchmark Report (95% Prefix Cache Hit)

## 1. Executive Summary

- **Model Checkpoint**: [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`1bae0e1088993b9b82de08f59de5be5a34e42170`, 48 FP8 safetensors shards, `~476 GiB`).
- **Architecture & Cluster (`8` TPU7x Chips = `1P + 1D`)**: 1 Prefill node (`2x2x1` = 4 chips / 8 TensorCores, `DP=8, TP=1, EP=8`, `CED=1`) + 1 Decode node (`2x2x1` = 4 chips / 8 TensorCores, `DP=8, TP=1, EP=8`, `CED=0`, DSpark `K=5` speculative decoding, `--async-scheduling`) connected via `TPURaidenConnector` Stage-3 KV transfer over DCN.
- **Zero Failed Requests Across All Concurrency Sweeps**: 49/49 (`n=8`), 92/92 (`n=16`), 166/166 (`n=32`), 263/263 (`n=64`), 476/476 (`n=128`, 128 trunks), and 512/512 (`n=128`, 40 pinned trunks) with structured multi-turn tool-calling completions (`finish_reason="tool_calls"`) and exact sampling temperatures from `requests_mocked.jsonl.xz`.
- **Peak Steady-State Throughput (`n = 128`)**:
  - **Pinned HBM Trunks (40 distinct trunks, 94.3% cache hit)**: **7,736.6 tok/s output** (6,538.8 tok/s full-drain) and **182,691.2 tok/s total input** (10,471.5 tok/s uncached prefill).
  - **5-or-N Trunks (128 distinct trunks, 84.7% cache hit)**: **7,157.2 tok/s output** (6,011.1 tok/s full-drain) and **166,322.5 tok/s total input** (25,044.7 tok/s uncached prefill).
- **Interactive Per-Stream Decode Speed & TTFT**:
  - **n = 8–16**: **94.9–108.2 tok/s median per stream** (P90_hi = 126.0–129.5 tok/s, Per-Token ITL = 9.24–10.54 ms, StepChunk ITL = 27.09–28.75 ms, DSpark τ = 2.73–2.81) with **0.44 s–0.49 s median TTFT** (1.12 s–1.46 s P90; 0.41 s–0.44 s on 1–8K and 1.07 s–1.14 s on 64K+ at ~81K–87K ISL).
  - **n = 32–64**: **72.4–84.3 tok/s median per stream** (Per-Token ITL = 11.87–13.82 ms, DSpark τ = 2.76–2.82) with **0.63 s–1.00 s median TTFT** and **2,553.2–3,980.1 tok/s window output throughput**.
  - **n = 128**: **63.3–64.7 tok/s median per stream** (P95_low = 50.5–52.4 tok/s, P90_hi = 83.7–84.1 tok/s, Per-Token ITL = 15.45–15.79 ms, DSpark τ = 2.74–2.75) with **1.30 s median TTFT (3.49 s P90)** at 40 trunks and **1.91 s median TTFT (7.64 s P90)** at 128 trunks.

---

## 2. End-to-End 1P1D Empirical ISL (50:20:12:9:9) Summary (95% Cache Hit Rate)

### 2.1 Aggregate Throughput & Workload Summary

| **n (Trunks)** | **OK Reqs (Tools / In-Win)** | **Mean ISL (Cache Hit)** | **Mean OSL** | **Win Out (Drain tok/s)** | **Win In (Uncached tok/s)** |
| --- | --- | --- | --- | --- | --- |
| **8 (40 tr)** | 49/49 (26 / 41) | 21,355 (93.8%) | 1,024.0 | **788.7** (736.7) | **17,439.7** (1,072.8) |
| **16 (40 tr)** | 92/92 (47 / 76) | 21,811 (94.0%) | 1,024.0 | **1,430.6** (1,304.6) | **33,253.3** (1,970.1) |
| **32 (40 tr)** | 166/166 (100 / 134) | 21,426 (94.1%) | 1,024.0 | **2,553.2** (2,316.2) | **59,117.0** (3,479.6) |
| **64 (64 tr)** | 263/263 (139 / 199) | 21,347 (94.3%) | 1,024.0 | **3,980.1** (3,547.2) | **91,081.4** (5,202.0) |
| **128 (128 tr, 5-or-N)** | 476/476 (274 / 348) | 21,899 (84.7%) | 1,024.0 | **7,157.2** (6,011.1) | **166,322.5** (25,044.7) |
| **128 (40 tr, Pinned)** | 512/512 (284 / 384) | 21,687 (94.3%) | 1,024.0 | **7,736.6** (6,538.8) | **182,691.2** (10,471.5) |

### 2.2 Per-Stream Decode Speed, Latency (TTFT / ITL / E2E) & DSpark Acceptance

| **n (Trunks)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** | **Per-Tok ITL (StepChunk Med/Mean)** | **DSpark τ (Pos 0..4 Accept)** | **Med TTFT (P90)** | **Med E2E** |
| --- | --- | --- | --- | --- | --- |
| **8 (40 tr)** | **108.22** (82.8 / 90.3 / 126.0) | 9.24 ms (27.09 / 35.11 ms) | **2.81** (62/44/32/24/18%) | **0.44 s** (1.12 s) | 10.09 s |
| **16 (40 tr)** | **94.87** (78.1 / 80.6 / 129.5) | 10.54 ms (28.75 / 34.73 ms) | **2.73** (61/42/30/22/17%) | **0.49 s** (1.46 s) | 11.57 s |
| **32 (40 tr)** | **84.27** (67.0 / 69.5 / 104.3) | 11.87 ms (33.83 / 44.15 ms) | **2.82** (62/44/32/25/19%) | **0.63 s** (1.31 s) | 12.79 s |
| **64 (64 tr)** | **72.35** (41.8 / 46.0 / 100.9) | 13.82 ms (34.80 / 52.38 ms) | **2.76** (61/42/31/24/18%) | **1.00 s** (5.03 s) | 15.03 s |
| **128 (128 tr, 5-or-N)** | **63.32** (50.5 / 52.8 / 83.7) | 15.79 ms (42.36 / 54.02 ms) | **2.75** (60/42/31/23/18%) | **1.91 s** (7.64 s) | 18.60 s |
| **128 (40 tr, Pinned)** | **64.74** (52.4 / 54.4 / 84.1) | 15.45 ms (41.74 / 53.78 ms) | **2.74** (61/42/31/23/18%) | **1.30 s** (3.49 s) | 17.41 s |

---

## 3. Per-Bucket Empirical ISL Breakdown (50:20:12:9:9) — 95% Prefix Cache Hit

### 3.1 Cross-Concurrency Per-Bucket Median TTFT & Decode Speed Matrix

| **ISL Bucket (Target Share)** | **n = 8 (40 Trunks)** | **n = 16 (40 Trunks)** | **n = 32 (40 Trunks)** | **n = 64 (64 Trunks)** | **n = 128 (128 Trunks)** | **n = 128 (40 Pinned Trunks)** |
| --- | --- | --- | --- | --- | --- | --- |
| **1–8K (50%)** | 0.41 s / 98.3 tok/s (8 tr) | 0.44 s / 93.5 tok/s (8 tr) | 0.47 s / 82.4 tok/s (8 tr) | 0.86 s / 74.4 tok/s (32 tr) | 1.47 s / 62.4 tok/s (64 tr) | 1.17 s / 62.9 tok/s (8 tr) |
| **8–16K (20%)** | 0.44 s / 104.2 tok/s (8 tr) | 0.43 s / 86.8 tok/s (8 tr) | 0.55 s / 78.9 tok/s (8 tr) | 0.99 s / 71.3 tok/s (8 tr) | 1.52 s / 58.8 tok/s (24 tr) | 1.20 s / 61.5 tok/s (8 tr) |
| **16–32K (12%)** | 0.43 s / 118.2 tok/s (8 tr) | 0.48 s / 121.0 tok/s (8 tr) | 0.55 s / 98.4 tok/s (8 tr) | 0.71 s / 80.0 tok/s (8 tr) | 1.46 s / 73.5 tok/s (16 tr) | 1.17 s / 72.0 tok/s (8 tr) |
| **32–64K (9%)** | 0.97 s / 115.6 tok/s (8 tr) | 0.93 s / 105.9 tok/s (8 tr) | 0.96 s / 88.5 tok/s (8 tr) | 1.01 s / 75.0 tok/s (8 tr) | 1.55 s / 64.9 tok/s (16 tr) | 1.17 s / 68.0 tok/s (8 tr) |
| **64K+ (9%)** | 1.07 s / 113.0 tok/s (8 tr) | 1.14 s / 95.8 tok/s (8 tr) | 1.24 s / 84.2 tok/s (8 tr) | 1.42 s / 75.8 tok/s (8 tr) | 1.84 s / 66.2 tok/s (8 tr) | 1.49 s / 65.0 tok/s (8 tr) |

### 3.2 n = 8 Sessions (40 Trunks: 8:8:8:8:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 8 | 25 (51.0%) | 7,092 | 1,024.0 | 91.8% | **0.41 s** | 1.09 s | 10.84 s | **98.3** (83.0 / 84.3 / 121.8) |
| **8–16K** | 20% | 8 | 10 (20.4%) | 12,584 | 1,024.0 | 92.0% | **0.44 s** | 0.63 s | 10.26 s | **104.2** (87.5 / 96.9 / 119.8) |
| **16–32K** | 12% | 8 | 6 (12.2%) | 26,740 | 1,024.0 | 95.1% | **0.43 s** | 0.46 s | 9.09 s | **118.2** (106.3 / 106.4 / 121.7) |
| **32–64K** | 9% | 8 | 4 (8.2%) | 58,742 | 1,024.0 | 95.0% | **0.97 s** | 1.11 s | 9.81 s | **115.6** (96.4 / 98.5 / 133.4) |
| **64K+** | 9% | 8 | 4 (8.2%) | 86,959 | 1,024.0 | 94.2% | **1.07 s** | 1.20 s | 10.07 s | **113.0** (97.8 / 100.4 / 124.5) |

### 3.3 n = 16 Sessions (40 Trunks: 8:8:8:8:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 8 | 46 (50.0%) | 7,016 | 1,024.0 | 92.3% | **0.44 s** | 1.44 s | 11.75 s | **93.5** (79.3 / 80.7 / 114.0) |
| **8–16K** | 20% | 8 | 19 (20.7%) | 12,984 | 1,024.0 | 93.4% | **0.43 s** | 1.46 s | 12.13 s | **86.8** (75.7 / 78.0 / 110.4) |
| **16–32K** | 12% | 8 | 11 (12.0%) | 26,187 | 1,024.0 | 95.3% | **0.48 s** | 0.88 s | 9.07 s | **121.0** (82.2 / 83.4 / 160.6) |
| **32–64K** | 9% | 8 | 8 (8.7%) | 62,577 | 1,024.0 | 94.9% | **0.93 s** | 1.53 s | 10.56 s | **105.9** (86.0 / 87.1 / 135.6) |
| **64K+** | 9% | 8 | 8 (8.7%) | 81,068 | 1,024.0 | 93.9% | **1.14 s** | 1.59 s | 12.00 s | **95.8** (79.1 / 81.0 / 110.7) |

### 3.4 n = 32 Sessions (40 Trunks: 8:8:8:8:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 8 | 83 (50.0%) | 6,956 | 1,024.0 | 92.0% | **0.47 s** | 1.23 s | 13.05 s | **82.4** (66.9 / 69.7 / 102.1) |
| **8–16K** | 20% | 8 | 33 (19.9%) | 12,845 | 1,024.0 | 93.7% | **0.55 s** | 1.22 s | 13.61 s | **78.9** (66.6 / 69.6 / 95.0) |
| **16–32K** | 12% | 8 | 20 (12.0%) | 24,592 | 1,024.0 | 95.4% | **0.55 s** | 1.12 s | 10.93 s | **98.4** (70.4 / 71.6 / 116.0) |
| **32–64K** | 9% | 8 | 15 (9.0%) | 54,039 | 1,024.0 | 94.7% | **0.96 s** | 1.18 s | 12.59 s | **88.5** (66.7 / 68.0 / 102.5) |
| **64K+** | 9% | 8 | 15 (9.0%) | 83,539 | 1,024.0 | 94.3% | **1.24 s** | 1.37 s | 13.39 s | **84.2** (67.2 / 70.6 / 107.6) |

### 3.5 n = 64 Sessions (64 Trunks: 32:8:8:8:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 32 | 131 (49.8%) | 6,979 | 1,024.0 | 92.1% | **0.86 s** | 4.55 s | 14.57 s | **74.4** (42.4 / 45.6 / 96.0) |
| **8–16K** | 20% | 8 | 53 (20.2%) | 13,014 | 1,024.0 | 94.9% | **0.99 s** | 4.92 s | 15.08 s | **71.3** (40.4 / 48.4 / 96.3) |
| **16–32K** | 12% | 8 | 31 (11.8%) | 24,809 | 1,024.0 | 95.2% | **0.71 s** | 4.55 s | 13.30 s | **80.0** (45.9 / 52.6 / 114.3) |
| **32–64K** | 9% | 8 | 24 (9.1%) | 53,127 | 1,024.0 | 94.8% | **1.01 s** | 7.20 s | 15.14 s | **75.0** (51.3 / 61.1 / 95.2) |
| **64K+** | 9% | 8 | 24 (9.1%) | 81,918 | 1,024.0 | 94.5% | **1.42 s** | 8.34 s | 15.27 s | **75.8** (52.0 / 58.3 / 99.2) |

### 3.6 n = 128 Sessions (128 Trunks 5-or-N: 64:24:16:16:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 64 | 238 (50.0%) | 7,042 | 1,024.0 | 92.0% | **1.47 s** | 6.10 s | 18.41 s | **62.4** (51.1 / 53.2 / 77.4) |
| **8–16K** | 20% | 24 | 95 (20.0%) | 13,132 | 1,024.0 | 93.2% | **1.52 s** | 6.27 s | 19.24 s | **58.8** (47.8 / 50.5 / 80.1) |
| **16–32K** | 12% | 16 | 57 (12.0%) | 26,936 | 1,024.0 | 93.0% | **1.46 s** | 4.16 s | 15.67 s | **73.5** (51.8 / 54.4 / 88.0) |
| **32–64K** | 9% | 16 | 43 (9.0%) | 53,379 | 1,024.0 | 78.6% | **1.55 s** | 6.15 s | 17.51 s | **64.9** (52.6 / 54.4 / 84.6) |
| **64K+** | 9% | 8 | 43 (9.0%) | 85,346 | 1,024.0 | 78.9% | **1.84 s** | 8.45 s | 18.94 s | **66.2** (54.3 / 55.6 / 85.2) |

### 3.7 n = 128 Sessions (40 Pinned Trunks: 8:8:8:8:8 Across 8 DP Ranks)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 8 | 256 (50.0%) | 7,089 | 1,024.0 | 92.1% | **1.17 s** | 2.62 s | 17.66 s | **62.9** (50.9 / 52.9 / 77.6) |
| **8–16K** | 20% | 8 | 102 (19.9%) | 12,882 | 1,024.0 | 93.8% | **1.20 s** | 3.51 s | 18.62 s | **61.5** (48.1 / 51.9 / 74.8) |
| **16–32K** | 12% | 8 | 62 (12.1%) | 26,344 | 1,024.0 | 95.4% | **1.17 s** | 2.47 s | 15.94 s | **72.0** (52.1 / 55.2 / 91.8) |
| **32–64K** | 9% | 8 | 46 (9.0%) | 52,542 | 1,024.0 | 94.7% | **1.17 s** | 3.50 s | 17.08 s | **68.0** (50.9 / 54.5 / 84.7) |
| **64K+** | 9% | 8 | 46 (9.0%) | 85,329 | 1,024.0 | 94.7% | **1.49 s** | 3.46 s | 17.29 s | **65.0** (54.6 / 55.8 / 87.5) |

---

## 4. Benchmarking Methodology & Code Artifacts

### 4.1 Methodology Overview
- **Workload Distribution (`50:20:12:9:9`)**: Requests are sampled from `requests_mocked.jsonl.xz` across five empirical ISL buckets (`1–8K: 50%`, `8–16K: 20%`, `16–32K: 12%`, `32–64K: 9%`, `64K+: 9%`, weighted mean ISL `~21.6K` tokens) with `max_tokens = 1,024` (`ignore_eos = True`), preserving multi-turn tool-calling schemas and cycling the exact 30-entry `(temperature, top_p)` sampling schedule from the dataset.
- **Prefix Cache Trunk Seeding (`95%` Target Hit)**: Each request splits into a shared cached prefix trunk (`~94.3%` realized token-level hit aligned to `1,024`-token KV retention boundaries) and a unique per-request uncached suffix (`~5.7%`). Before each concurrency level `n`, `/reset_prefix_cache` clears `1P` and `1D`, and Phase-0 warmup seeds `40` (`n <= 32` or pinned `n = 128`), `64` (`n = 64`), or `128` (`n = 128` `5-or-N`) distinct trunks pinned deterministically across the 8 DP ranks via `X-data-parallel-rank`.
- **Closed-Loop Steady-State Window & Drain**: After an initial XLA compilation warmup pass, each concurrency level `n` runs `n` persistent closed-loop session workers for a `60 s` active dispatch window (measuring steady-state window throughput and `1D` Prometheus `vllm:spec_decode_*` counters), followed by full drain of all in-flight requests.

### 4.2 Code & Data Artifacts
- **Hugging Face Checkpoint**: [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- **GitHub Branch (`vllm-project/vllm-torchtpu`)**: [`cpk/dsv41-combined` (`benchmarks/dsv41_tpu7x`)](https://github.com/vllm-project/vllm-torchtpu/tree/cpk/dsv41-combined/benchmarks/dsv41_tpu7x)
- **Empirical ISL Benchmark Harness**: [`bench_empirical_isl_dsv41.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/bench_empirical_isl_dsv41.py)
- **1P1D Orchestrator & Pod Launcher**: [`run_pd.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/run_pd.py) and [`in_pod_serve.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/in_pod_serve.sh)
- **Disaggregated DP-Aware Proxy**: [`toy_proxy_server.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/examples/disagg/toy_proxy_server.py)
- **Consolidated JSON Summary & Logs**: [`dsv41_1017_pd_empirical_isl_95_summary.json`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/dsv41_1017_pd_empirical_isl_95_summary.json) and `gs://chaitanyapk-data/atlas/dsv4.1flash-1017/`

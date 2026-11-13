# DeepSeek-V4.1-Flash-1017 FP8 PD Disaggregation (1P1D) — Empirical ISL (50:20:12:9:9) & Uncached Prefill Benchmark Report

## 1. Executive Summary

- **Model Checkpoint**: [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`1bae0e1088993b9b82de08f59de5be5a34e42170`, 48 FP8 safetensors shards, `~476 GiB`).
- **Architecture & Cluster (`8` TPU7x Chips = `1P + 1D`)**:
  - **1P Node (`2x2x1` = 4 chips / 8 TensorCores)**: 4 independent per-chip `TP=2, EP=2, DP=1` instances (`VLLM_DSV41_CED=1`, `VLLM_DSV41_PURE_20L_PREFILL=1` 20-layer Context Encoder, `max_num_batched_tokens = 16,384`, `PREFILL_4X_TP2_GPU_MEM=0.92` = `1,659` blocks/partition across disjoint `main`, `state`, `swa_0`, `swa_1`, `swa_2` `BlockPool` queues = `849,408` tokens/node per partition).
  - **1D Node (`2x2x1` = 4 chips / 8 TensorCores)**: `DP=8, TP=1, EP=8` (`VLLM_DSV41_CED=0`, DSpark `K=5` speculative decoding, `--async-scheduling`, `VLLM_DISJOINT_KV_PARTITIONS=1` = `737,280` tokens/node per partition).
  - **KV Transfer**: `TPURaidenConnector` Stage-3 DCN transfer of the 35 encoder KV pools (`Layers 0..20`, `TPU_RAIDEN_POOL_TAGS_PER_LAYER=1` for `TP=2 → TP=1` resharding) with a 128-token SWA tail handoff (`VLLM_DSV41_PD_HANDOFF_TAIL_TOKENS=128`) replayed in a single step on `1D`.
- **Sustained `~93.6%–94.3%` Realized Prefix Cache Hit Across All Concurrency Levels**: `94.19%` (`n=8`), `94.12%` (`n=16`), `94.14%` (`n=32`), `93.76%` (`n=64`), `93.55%` (`n=128` natural OSL), and `93.75%–94.30%` (`n=128` fixed `1,024` OSL), with structured multi-turn tool-calling completions (`finish_reason="tool_calls"`) and exact sampling temperatures from `requests_mocked.jsonl.xz`.
- **Steady-State Window Throughput (`95%` Prefix Cache Hit)**:
  - **`n = 8`**: **728.3 tok/s output** (`180 s` natural OSL) / **788.7 tok/s** (`1,024` OSL) and **13,873.8–17,439.7 tok/s total input**.
  - **`n = 16`**: **1,402.9 tok/s output** (`180 s` natural OSL) / **1,430.6 tok/s** (`1,024` OSL) and **30,552.7–33,253.3 tok/s total input**.
  - **`n = 32`**: **2,362.5 tok/s output** (`180 s` natural OSL) / **2,553.2 tok/s** (`1,024` OSL) and **59,117.0–61,620.2 tok/s total input**.
  - **`n = 64`**: **4,207.7 tok/s output** and **96,184.2 tok/s total input** (5,998.3 tok/s uncached prefill).
  - **`n = 128`**: **5,372.5 tok/s output** and **135,555.7 tok/s total input** at natural EOS (`mean OSL = 906.4`), **6,376.0 tok/s output** and **154,057.0 tok/s total input** (`428/428` OK, `20` trunks, fixed `1,024` OSL), and **7,736.6 tok/s output** / **182,691.2 tok/s total input** (`512/512` OK, `40` pinned trunks, fixed `1,024` OSL).
- **Sub-Second TTFT Across `1–8K` to `64K+` (`4 × TP=2` 20L CED + Disjoint `BlockPool`)**:
  - **`n = 8–16`**: **89.45–90.01 tok/s median per stream** (`P90_hi = 114.9–123.3 tok/s`, Per-Token ITL = `11.11–11.18 ms`, StepChunk ITL = `26.83–28.26 ms`, DSpark $\tau = 2.78\text{–}2.83$) with **0.52 s–0.58 s median TTFT** (`0.70 s–0.75 s` P90; **0.68 s–0.75 s median TTFT** and **0.73 s–0.78 s P90 TTFT** on `64K+`).
  - **`n = 32–64`**: **67.86–75.40 tok/s median per stream** (Per-Token ITL = `13.26–14.74 ms`, DSpark $\tau = 2.86\text{–}2.93$) with **0.55 s–0.60 s median TTFT** (`0.77 s–0.96 s` P90; **0.72 s–0.77 s median TTFT** and **0.84 s–1.06 s P90 TTFT** on `64K+`).
  - **`n = 128`**: **0.77 s median TTFT** (`1.72 s` P90; **0.88 s median / 1.35 s P90** on `64K+`) with DSpark $\tau = 2.94$.
- **Uncached Prefill (`0%` Cache Hit, `output_len = 1`)**:
  - **`32K–128K` Long-Context Uncached Prefill**: **`180,822.9–182,224.6 tok/s` (`1P` node)** and **`176,107.4–178,323.5 tok/s` (`E2E 1P1D` node)** across `32K c=8/c=16`, `100K c=8/c=16`, and `128K c=8/c=16` (`3,754.4 ms` `1P` / `4,181.0 ms` E2E TTFT at `100K c=8`; `4,850.3 ms` `1P` / `5,382.3 ms` E2E TTFT at `128K c=8`).
  - **Empirical ISL Mixed Uncached Waves (`c = 8`)**: **`136,243.7 tok/s` (`1P`) / `134,188.4 tok/s` (`E2E 1P1D`)** on 40 requests and **`141,308.9 tok/s` (`1P`) / `139,587.2 tok/s` (`E2E 1P1D`)** on 128 requests.

---

## 2. End-to-End 1P1D Empirical ISL (50:20:12:9:9) Summary (95% Cache Hit Rate)

### 2.1 Aggregate Throughput & Workload Summary (`180 s` Steady-State Window)

| **n (Trunks)** | **OK Reqs (Tools / In-Win)** | **Mean ISL (Cache Hit %)** | **Mean OSL** | **Win Out (Drain tok/s)** | **Win In (Uncached tok/s)** |
| --- | --- | --- | --- | --- | --- |
| **8 (20 tr)** | 137/142 (59 / 129) | 18,228 (**94.19%**) | 1,031.0 | **728.3** (580.9) | **13,873.8** (806.4) |
| **16 (20 tr)** | 294/301 (109 / 278) | 18,728 (**94.12%**) | 937.4 | **1,402.9** (1,212.2) | **30,552.7** (1,795.3) |
| **32 (20 tr)** | 580/592 (251 / 548) | 19,143 (**94.14%**) | 776.8 | **2,362.5** (2,023.9) | **61,620.2** (3,610.6) |
| **64 (20 tr)** | 929/955 (352 / 865) | 18,687 (**93.76%**) | 919.1 | **4,207.7** (3,175.3) | **96,184.2** (5,998.3) |
| **128 (20 tr, Natural OSL)** | 1305/1341 (554 / 1177) | 18,827 (**93.55%**) | 906.4 | **5,372.5** (3,184.1) | **135,555.7** (8,750.4) |
| **128 (20 tr, Fixed 1,024 OSL)** | 428/428 (157 / 300) | 21,777 (**93.75%**) | 1,024.0 | **6,376.0** (5,418.5) | **154,057.0** (9,621.8) |
| **128 (40 tr, Fixed 1,024 OSL)** | 512/512 (284 / 384) | 21,687 (**94.30%**) | 1,024.0 | **7,736.6** (6,538.8) | **182,691.2** (10,471.5) |

### 2.2 Per-Stream Decode Speed, Latency (TTFT / ITL / E2E) & DSpark Acceptance

| **n (Trunks)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** | **Per-Tok ITL (StepChunk Med/Mean)** | **DSpark τ (Pos 0..4 Accept)** | **Med TTFT (P90)** | **64K+ Med TTFT (P90)** | **Med E2E** |
| --- | --- | --- | --- | --- | --- | --- |
| **8 (20 tr)** | **90.01** (72.2 / 76.3 / 114.9) | 11.11 ms (26.83 / 31.42 ms) | **2.78** (68/46/31/20/13%) | **0.58 s** (0.75 s) | **0.75 s** (0.78 s) | 5.29 s |
| **16 (20 tr)** | **89.45** (72.8 / 76.6 / 123.3) | 11.18 ms (28.26 / 32.05 ms) | **2.83** (69/47/32/21/14%) | **0.52 s** (0.70 s) | **0.68 s** (0.73 s) | 4.77 s |
| **32 (20 tr)** | **75.40** (56.4 / 63.4 / 100.1) | 13.26 ms (33.46 / 39.19 ms) | **2.86** (70/47/32/22/15%) | **0.55 s** (0.77 s) | **0.72 s** (0.84 s) | 6.07 s |
| **64 (20 tr)** | **67.86** (42.9 / 51.8 / 91.8) | 14.74 ms (34.93 / 44.59 ms) | **2.93** (71/49/34/23/16%) | **0.60 s** (0.96 s) | **0.77 s** (1.06 s) | 6.46 s |
| **128 (20 tr, Natural OSL)** | **43.32** (25.5 / 31.9 / 59.5) | 23.08 ms (43.58 / 69.99 ms) | **2.94** (71/49/34/24/16%) | **0.77 s** (1.72 s) | **0.88 s** (1.35 s) | 10.37 s |
| **128 (20 tr, Fixed 1,024 OSL)** | **53.32** (39.9 / 41.8 / 78.9) | 18.75 ms (42.36 / 61.38 ms) | **2.92** (69/48/34/24/17%) | **0.93 s** (3.53 s) | **0.92 s** (3.81 s) | 20.83 s |
| **128 (40 tr, Fixed 1,024 OSL)** | **64.74** (52.4 / 54.4 / 84.1) | 15.45 ms (41.74 / 53.78 ms) | **2.74** (61/42/31/23/18%) | **1.30 s** (3.49 s) | **1.49 s** (3.46 s) | 17.41 s |

---

## 3. Per-Bucket Empirical ISL Breakdown (50:20:12:9:9) — 95% Prefix Cache Hit

### 3.1 Cross-Concurrency Per-Bucket Median TTFT, P90 TTFT & Decode Speed Matrix

| **ISL Bucket (Target Share)** | **n = 8 (20 Trunks)** | **n = 16 (20 Trunks)** | **n = 32 (20 Trunks)** | **n = 64 (20 Trunks)** | **n = 128 (20 Trunks)** |
| --- | --- | --- | --- | --- | --- |
| **1–8K (50%)** | 0.55 s (P90 0.64 s) / 87.8 tok/s (92.2% hit) | 0.49 s (P90 0.61 s) / 88.4 tok/s (93.3% hit) | 0.51 s (P90 0.73 s) / 74.5 tok/s (93.2% hit) | 0.54 s (P90 0.89 s) / 67.6 tok/s (93.1% hit) | 0.72 s (P90 1.45 s) / 42.7 tok/s (93.2% hit) |
| **8–16K (20%)** | 0.58 s (P90 0.77 s) / 92.0 tok/s (94.1% hit) | 0.51 s (P90 0.67 s) / 85.3 tok/s (92.6% hit) | 0.54 s (P90 0.79 s) / 71.9 tok/s (93.4% hit) | 0.59 s (P90 0.93 s) / 65.3 tok/s (92.5% hit) | 0.75 s (P90 1.68 s) / 41.6 tok/s (92.9% hit) |
| **16–32K (12%)** | 0.62 s (P90 0.76 s) / 89.7 tok/s (94.8% hit) | 0.54 s (P90 0.63 s) / 104.5 tok/s (94.9% hit) | 0.57 s (P90 0.78 s) / 94.0 tok/s (95.0% hit) | 0.59 s (P90 0.91 s) / 79.3 tok/s (94.3% hit) | 0.77 s (P90 1.50 s) / 50.5 tok/s (94.3% hit) |
| **32–64K (9%)** | 0.69 s (P90 0.77 s) / 97.7 tok/s (94.8% hit) | 0.61 s (P90 0.80 s) / 105.3 tok/s (94.7% hit) | 0.63 s (P90 0.80 s) / 88.8 tok/s (94.6% hit) | 0.66 s (P90 1.03 s) / 78.1 tok/s (94.4% hit) | 0.85 s (P90 1.65 s) / 51.7 tok/s (94.0% hit) |
| **64K+ (9%)** | 0.75 s (P90 0.78 s) / 91.8 tok/s (94.7% hit) | 0.68 s (P90 0.73 s) / 99.3 tok/s (94.4% hit) | 0.72 s (P90 0.84 s) / 79.6 tok/s (94.1% hit) | 0.77 s (P90 1.06 s) / 70.6 tok/s (93.8% hit) | 0.88 s (P90 1.35 s) / 53.3 tok/s (93.2% hit) |

### 3.2 n = 8 Sessions (20 Trunks: 4:4:4:4:4 Across 4 × TP=2 Chips)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 4 | 71 (51.8%) | 6,665 | 350.3 | 92.2% | **0.55 s** | 0.64 s | 4.28 s | **87.8** (70.7 / 74.4 / 102.4) |
| **8–16K** | 20% | 4 | 28 (20.4%) | 12,595 | 2,077.5 | 94.1% | **0.58 s** | 0.77 s | 11.55 s | **92.0** (78.3 / 81.2 / 101.9) |
| **16–32K** | 12% | 4 | 17 (12.4%) | 22,935 | 610.7 | 94.8% | **0.62 s** | 0.76 s | 5.90 s | **89.7** (84.0 / 84.4 / 123.4) |
| **32–64K** | 9% | 4 | 13 (9.5%) | 53,511 | 1,546.9 | 94.8% | **0.69 s** | 0.77 s | 6.46 s | **97.7** (73.9 / 79.3 / 130.1) |
| **64K+** | 9% | 4 | 8 (5.8%) | 73,235 | 3,464.1 | 94.7% | **0.75 s** | 0.78 s | 40.87 s | **91.8** (88.0 / 88.0 / 139.5) |

### 3.3 n = 16 Sessions (20 Trunks: 4:4:4:4:4 Across 4 × TP=2 Chips)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 4 | 151 (51.4%) | 6,584 | 348.6 | 93.3% | **0.49 s** | 0.61 s | 4.27 s | **88.4** (73.0 / 76.4 / 102.3) |
| **8–16K** | 20% | 4 | 60 (20.4%) | 12,342 | 2,009.4 | 92.6% | **0.51 s** | 0.67 s | 9.63 s | **85.3** (70.9 / 73.8 / 101.1) |
| **16–32K** | 12% | 4 | 36 (12.2%) | 23,023 | 556.9 | 94.9% | **0.54 s** | 0.63 s | 4.36 s | **104.5** (78.6 / 82.7 / 153.9) |
| **32–64K** | 9% | 4 | 27 (9.2%) | 52,751 | 1,513.5 | 94.7% | **0.61 s** | 0.80 s | 5.16 s | **105.3** (77.8 / 79.1 / 140.5) |
| **64K+** | 9% | 4 | 20 (6.8%) | 75,912 | 2,073.7 | 94.4% | **0.68 s** | 0.73 s | 13.86 s | **99.3** (80.6 / 86.1 / 146.7) |

### 3.4 n = 32 Sessions (20 Trunks: 4:4:4:4:4 Across 4 × TP=2 Chips)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 4 | 296 (51.0%) | 6,591 | 378.3 | 93.2% | **0.51 s** | 0.73 s | 5.09 s | **74.5** (55.3 / 62.7 / 86.0) |
| **8–16K** | 20% | 4 | 119 (20.5%) | 11,982 | 1,285.5 | 93.4% | **0.54 s** | 0.79 s | 11.93 s | **71.9** (59.1 / 62.9 / 85.6) |
| **16–32K** | 12% | 4 | 71 (12.2%) | 26,398 | 844.2 | 95.0% | **0.57 s** | 0.78 s | 6.42 s | **94.0** (68.8 / 73.7 / 128.6) |
| **32–64K** | 9% | 4 | 53 (9.1%) | 51,158 | 969.7 | 94.6% | **0.63 s** | 0.80 s | 6.02 s | **88.8** (54.0 / 66.5 / 126.8) |
| **64K+** | 9% | 4 | 41 (7.1%) | 76,600 | 1,811.1 | 94.1% | **0.72 s** | 0.84 s | 17.78 s | **79.6** (62.8 / 65.7 / 121.5) |

### 3.5 n = 64 Sessions (20 Trunks: 4:4:4:4:4 Across 4 × TP=2 Chips)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 4 | 477 (51.3%) | 6,597 | 371.0 | 93.1% | **0.54 s** | 0.89 s | 5.61 s | **67.6** (39.2 / 49.3 / 78.8) |
| **8–16K** | 20% | 4 | 191 (20.6%) | 12,204 | 1,678.2 | 92.5% | **0.59 s** | 0.93 s | 13.13 s | **65.3** (45.6 / 52.4 / 81.0) |
| **16–32K** | 12% | 4 | 115 (12.4%) | 23,508 | 649.8 | 94.3% | **0.59 s** | 0.91 s | 6.33 s | **79.3** (47.5 / 58.7 / 122.4) |
| **32–64K** | 9% | 4 | 86 (9.3%) | 53,352 | 1,424.1 | 94.4% | **0.66 s** | 1.03 s | 6.97 s | **78.1** (48.6 / 56.3 / 112.3) |
| **64K+** | 9% | 4 | 60 (6.5%) | 76,519 | 2,652.6 | 93.8% | **0.77 s** | 1.06 s | 30.97 s | **70.6** (55.8 / 62.1 / 112.2) |

### 3.6 n = 128 Sessions (20 Trunks: 4:4:4:4:4 Across 4 × TP=2 Chips, Natural OSL)

| **ISL Bucket** | **Target Weight** | **Trunks** | **Completed Reqs** | **Mean ISL (tok)** | **Mean OSL (tok)** | **Cache Hit (%)** | **Med TTFT (s)** | **P90 TTFT (s)** | **Med E2E (s)** | **Med TPS/Strm (P95_low / P90_low / P90_hi)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1–8K** | 50% | 4 | 670 (51.3%) | 6,586 | 386.0 | 93.2% | **0.72 s** | 1.45 s | 9.15 s | **42.7** (22.1 / 29.2 / 52.1) |
| **8–16K** | 20% | 4 | 268 (20.5%) | 12,266 | 1,536.2 | 92.9% | **0.75 s** | 1.68 s | 19.73 s | **41.6** (31.4 / 34.8 / 50.9) |
| **16–32K** | 12% | 4 | 161 (12.3%) | 24,547 | 601.0 | 94.3% | **0.77 s** | 1.50 s | 10.45 s | **50.5** (28.9 / 36.6 / 74.0) |
| **32–64K** | 9% | 4 | 121 (9.3%) | 53,357 | 1,642.6 | 94.0% | **0.85 s** | 1.65 s | 12.29 s | **51.7** (31.1 / 36.3 / 74.9) |
| **64K+** | 9% | 4 | 85 (6.5%) | 76,014 | 2,552.7 | 93.2% | **0.88 s** | 1.35 s | 29.21 s | **53.3** (34.3 / 36.2 / 72.8) |

---

## 4. Long-Context & Empirical ISL Uncached Prefill (0% Cache Hit, `output_len = 1`) on `1017`

### 4.1 Long-Context Uncached Prefill (`32K–128K`, `0%` Cache Hit) — `1P` & `1P1D` vs. Standalone Prefill-Only

| **Workload (0% Cache Hit)** | **ISL** | **c** | **Reqs (n)** | **1P Node Tok/s (Per Chip)** | **E2E 1P1D Node Tok/s (Per Chip)** | **1P Latency (Mean / Med)** | **E2E 1P1D TTFT (Mean / Med)** | **Standalone Prefill-Only Ref** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`32K_c8_n64`** | 32,768 | 8 | 64 | **181,229.5 tok/s** (45,307) | **176,991.4 tok/s** (44,248) | **1,133.2 ms** (1,138.1 ms) | **1,406.7 ms** (1,387.3 ms) | 189,563.1 tok/s (1,334.5 ms) |
| **`32K_c16_n64`** | 32,768 | 16 | 64 | **180,822.9 tok/s** (45,206) | **176,539.1 tok/s** (44,135) | 2,350.5 ms (2,516.3 ms) | 2,640.8 ms (2,748.9 ms) | — |
| **`100K_c8_n32`** | 100,000 | 8 | 32 | **181,628.5 tok/s** (45,407) | **176,107.4 tok/s** (44,027) | **3,754.4 ms** (3,800.0 ms) | **4,181.0 ms** (4,142.8 ms) | 196,644.1 tok/s (3,809.7 ms) |
| **`100K_c16_n32`** | 100,000 | 16 | 32 | **182,224.6 tok/s** (45,556) | **178,319.0 tok/s** (44,580) | 6,702.8 ms (7,789.5 ms) | 7,209.2 ms (8,147.3 ms) | 195,839.6 tok/s (6,648.6 ms) |
| **`128K_c4_n16`** | 130,000 | 4 | 16 | **158,765.5 tok/s** (39,691) | **153,560.0 tok/s** (38,390) | 2,878.4 ms (2,866.3 ms) | 3,298.3 ms (3,243.5 ms) | 186,703.1 tok/s (2,794.8 ms) |
| **`128K_c8_n32`** | 130,000 | 8 | 32 | **181,559.8 tok/s** (45,390) | **178,323.5 tok/s** (44,581) | **4,850.3 ms** (5,138.6 ms) | **5,382.3 ms** (5,501.4 ms) | 198,541.3 tok/s (4,946.6 ms) |
| **`128K_c16_n32`** | 130,000 | 16 | 32 | **180,362.4 tok/s** (45,091) | **176,532.3 tok/s** (44,133) | 8,827.8 ms (10,568.8 ms) | 9,447.7 ms (10,947.2 ms) | 197,891.3 tok/s (8,609.8 ms) |

### 4.2 Empirical ISL (`50:20:12:9:9`) Per-Bucket & Mixed Uncached Prefill Waves (`c = 8`, `0%` Cache Hit)

| **ISL Bucket / Mixed Sweep** | **Mean ISL (Wave Tokens)** | **1P Latency Med (Wall)** | **KV Xfer + 1D 1st-Tok (Med)** | **E2E 1P1D TTFT Med (P90)** | **1P Node Tok/s (Per Chip)** | **E2E 1P1D Tok/s (Per Chip)** |
| --- | --- | --- | --- | --- | --- | --- |
| **64K+ (8 Reqs)** | 84,536 (676,289) | **2.71 s** (4.30 s) | 0.30 s | **3.11 s** (4.28 s) | **157,440.3 tok/s** (39,360) | **147,022.1 tok/s** (36,756) |
| **32–64K (8 Reqs)** | 53,799 (430,389) | **1.99 s** (2.85 s) | 0.24 s | **2.24 s** (2.84 s) | **151,231.6 tok/s** (37,808) | **139,123.9 tok/s** (34,781) |
| **16–32K (8 Reqs)** | 26,048 (208,387) | **1.09 s** (1.44 s) | 0.40 s | **1.66 s** (1.83 s) | **144,317.6 tok/s** (36,079) | **112,692.0 tok/s** (28,173) |
| **8–16K (8 Reqs)** | 12,922 (103,375) | **0.72 s** (1.06 s) | 0.28 s | **1.07 s** (1.32 s) | **97,452.3 tok/s** (24,363) | **73,484.6 tok/s** (18,371) |
| **1–8K (8 Reqs)** | 7,096 (56,765) | **0.58 s** (0.79 s) | 0.28 s | **0.80 s** (1.18 s) | **72,303.4 tok/s** (18,076) | **44,803.0 tok/s** (11,201) |
| **Mixed 5-Bucket (40 Reqs)** | 36,876 (1,475,051) | — (**10.83 s** wall) | — | — (**10.99 s** wall) | **136,243.7 tok/s** (34,061) | **134,188.4 tok/s** (33,547) |
| **Mixed 5-Bucket (128 Reqs)** | 21,231 (2,717,527) | — (**19.23 s** wall) | — | — (**19.47 s** wall) | **141,308.9 tok/s** (35,327) | **139,587.2 tok/s** (34,897) |

---

## 5. Benchmarking Methodology & Code Artifacts

- **Hugging Face Checkpoint**: [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- **GitHub Branch (`vllm-project/vllm-torchtpu`)**: [`cpk/dsv41-combined` (`benchmarks/dsv41_tpu7x`)](https://github.com/vllm-project/vllm-torchtpu/tree/cpk/dsv41-combined/benchmarks/dsv41_tpu7x)
- **Empirical ISL Benchmark Harness**: [`bench_empirical_isl_dsv41.py`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/scripts/pd/bench_empirical_isl_dsv41.py) and [`run_emp_uncached_waves.py`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/scripts/pd/run_emp_uncached_waves.py)
- **1P1D Orchestrator & Pod Launcher**: [`run_pd.py`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/scripts/pd/run_pd.py), [`in_pod_serve.sh`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/scripts/pd/in_pod_serve.sh), and [`launch_dsv41_disagg_role.sh`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/repos/vtt-combined/examples/disagg/launch_dsv41_disagg_role.sh)
- **Disaggregated DP-Aware Proxy**: [`toy_proxy_server.py`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/repos/vtt-combined/examples/disagg/toy_proxy_server.py)
- **Consolidated JSON Summaries & Logs**:
  - [`dsv41_1017_pd_empirical_isl_95_p2_summary.json`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/runs/pd/20261102-194124_dsv41_1017_pd_4xtp2_20l_disjoint_1017p2/dsv41_1017_pd_empirical_isl_95_p2_summary.json)
  - [`dsv41_1017_pd_empirical_isl_95_p2_n128_fixed1024.json`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/runs/pd/20261102-194124_dsv41_1017_pd_4xtp2_20l_disjoint_1017p2/dsv41_1017_pd_empirical_isl_95_p2_n128_fixed1024.json)
  - [`dsv41_1017_uncached_long_prefill.json`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/runs/pd/20261102-194124_dsv41_1017_pd_4xtp2_20l_disjoint_1017p2/dsv41_1017_uncached_long_prefill.json)
  - [`dsv41_1017_emp_uncached_waves_p2.json`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/runs/pd/20261102-194124_dsv41_1017_pd_4xtp2_20l_disjoint_1017p2/dsv41_1017_emp_uncached_waves_p2.json)
  - [`summary.json`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/runs/pd/20261102-194124_dsv41_1017_pd_4xtp2_20l_disjoint_1017p2/summary.json)
  - `gs://chaitanyapk-data/atlas/dsv4.1flash-1017/`

# DeepSeek-V4.1-Flash Uncached Prefill (0% Cache Hit) Benchmark Summary

**Hardware per Node**: `tpu7x-8` (`2x2x1` = 4 physical TPU7x chips / 8 TensorCores)  
**Model**: [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) / [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`FP8` KV cache, `VLLM_DSV41_CED=1` 20-layer Context Encoder)

## Measurement Modes Overview

- **Standalone Prefill-Only**: A single 4-chip node (`4 × TP=2, EP=2`) running only the 20-layer Context Encoder (`VLLM_DSV41_PURE_20L_PREFILL=1`) without any KV transfer connector (`TPURaidenConnector` disabled) or decode peer. Measures isolated prefill compute throughput on 1 node.
- **`1P` (Prefill Producer inside 1P1D)**: Direct HTTP requests to the 4 prefill instances (`:8400..8403`) inside a live `1P1D` deployment with `TPURaidenConnector` (`kv_producer`) active. Measures `1P` prefill compute + D2H KV staging into the Raiden host ring buffer (`kv_transfer_params` generation), before `1D` pulls the KV cache over DCN.
- **End-to-End `1P1D` (Disaggregated Prefill + Decode)**: Client requests sent through the disaggregated router (`toy_proxy_server.py` on `:8000`) across 1 Prefill node (`1P`, `4 × TP=2, EP=2`) and 1 Decode node (`1D`, `DP=8, TP=1, EP=8`). Measures the full end-to-end TTFT and throughput including `1P` prefill, Raiden Stage-3 DCN KV transfer (`TP=2 → TP=1` resharding), `1D` 128-token SWA tail replay across all 40 layers, and first-token emission from `1D`.

---

## 1. Standalone Prefill-Only (1 Node = 4 Chips, 4 × TP=2, EP=2)

- **Configuration**: 4 independent per-chip `TP=2, EP=2` instances (`VLLM_DSV41_PURE_20L_PREFILL=1`, `max_num_batched_tokens = 16,384`, `output_len = 1`, `0%` prefix cache hit).

| **Input ISL** | **Concurrency (c)** | **Total Reqs (n)** | **Node Throughput (4 Chips)** | **Per-Chip Throughput** | **Mean TTFT** |
| --- | --- | --- | --- | --- | --- |
| **32K** | 8 | 64 | **189,563.1 tok/s** | **47,391 tok/s** | **1,334.5 ms** |
| **100K** | 8 | 32 | **196,644.1 tok/s** | **49,161 tok/s** | **3,809.7 ms** |
| **100K** | 16 | 32 | **195,839.6 tok/s** | **48,960 tok/s** | **6,648.6 ms** |
| **128K** | 4 | 16 | **186,703.1 tok/s** | **46,676 tok/s** | **2,794.8 ms** |
| **128K** | 8 | 32 | **198,541.3 tok/s** | **49,635 tok/s** | **4,946.6 ms** |
| **128K** | 16 | 32 | **197,891.3 tok/s** | **49,473 tok/s** | **8,609.8 ms** |

---

## 2. 1P1D Disaggregated Prefill (1P = 4 × TP=2, EP=2 + 1D = DP=8, TP=1, EP=8)

- **Configuration**:
  - **1P Node (4 Chips)**: 4 independent per-chip `TP=2, EP=2, DP=1` servers (`VLLM_DSV41_CED=1`, `VLLM_DSV41_PURE_20L_PREFILL=1`, `max_num_batched_tokens = 16,384` / `32,768`, SparseCore collective offloads enabled, `TPU_RAIDEN_POOL_TAGS_PER_LAYER=1`).
  - **KV Transfer**: Raiden Stage-3 DCN transfer of only the **35 encoder KV pools** (18 encoder SWA + 16 `kv_b` + 1 Layer-20 global KV/state) with a **128-token SWA tail handoff** (`VLLM_DSV41_PD_HANDOFF_TAIL_TOKENS=128`).
  - **1D Node (4 Chips)**: `DP=8, TP=1, EP=8` (`VLLM_DSV41_CED=0`, DSpark `K=5` speculative decoding, `max_num_batched_tokens = 384`), which pulls the 35 encoder KV pools over DCN and replays the final 128 prompt tokens through all 40 layers in a single step to populate Layers 20–39 decoder SWA KV and emit the first token.

### 2.1 Long-Context Uncached Prefill (0% Cache Hit) — Direct Comparison vs. Standalone Prefill-Only

| **Workload (0% Cache Hit)** | **ISL** | **c** | **Reqs (n)** | **1P Node Tok/s (Per Chip)** | **E2E 1P1D Node Tok/s (Per Chip)** | **1P Latency (Mean / Med)** | **E2E 1P1D TTFT (Mean / Med)** | **Standalone Prefill-Only Ref** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`32K_c8_n64`** | 32,768 | 8 | 64 | **193,402.4 tok/s** (48,351) | **184,322.3 tok/s** (46,081) | **1,045.6 ms** (1,071.6 ms) | **1,356.5 ms** (1,351.6 ms) | 189,563.1 tok/s (1,334.5 ms) |
| **`32K_c16_n64`** | 32,768 | 16 | 64 | **192,319.7 tok/s** (48,080) | **182,762.3 tok/s** (45,691) | 2,219.4 ms (2,389.5 ms) | 2,562.9 ms (2,739.2 ms) | — |
| **`100K_c8_n32`** | 100,000 | 8 | 32 | **164,064.9 tok/s** (41,016) | **157,842.8 tok/s** (39,461) | 4,213.9 ms (4,500.3 ms) | 4,680.9 ms (4,870.1 ms) | 196,644.1 tok/s (3,809.7 ms) |
| **`100K_c16_n32`** | 100,000 | 16 | 32 | **163,903.7 tok/s** (40,976) | **157,243.8 tok/s** (39,311) | 7,585.0 ms (8,909.2 ms) | 8,187.4 ms (9,284.5 ms) | 195,839.6 tok/s (6,648.6 ms) |
| **`128K_c4_n16`** | 130,000 | 4 | 16 | **165,214.7 tok/s** (41,304) *(active 1P: **191,324 tok/s**)* | **157,613.3 tok/s** (39,403) | **2,734.2 ms** (**2,717.9 ms**) | 3,227.1 ms (3,186.2 ms) | 186,703.1 tok/s (2,794.8 ms) |
| **`128K_c8_n32`** | 130,000 | 8 | 32 | **180,398.3 tok/s** (45,100) | **173,271.8 tok/s** (43,318) | **5,019.9 ms** (5,281.8 ms) | 5,546.0 ms (5,684.6 ms) | 198,541.3 tok/s (4,946.6 ms) |
| **`128K_c16_n32`** | 130,000 | 16 | 32 | **179,289.1 tok/s** (44,822) | **172,617.7 tok/s** (43,154) | 8,951.6 ms (10,836.2 ms) | 9,667.8 ms (11,232.1 ms) | 197,891.3 tok/s (8,609.8 ms) |

### 2.2 Empirical ISL (50:20:12:9:9) Per-Bucket Uncached Prefill (c = 8, 8 Reqs per Bucket)

| **ISL Bucket (8 Reqs)** | **Mean ISL (Wave Tokens)** | **1P Latency Med (Wall)** | **KV Xfer + 1D 1st-Tok (Med)** | **E2E 1P1D TTFT Med (P90)** | **1P Node Tok/s (Per Chip)** | **E2E 1P1D Tok/s (Per Chip)** |
| --- | --- | --- | --- | --- | --- | --- |
| **64K+** | 84,536 (676,289) | **2.70 s** (4.34 s) | 1.49 s | **3.95 s** (5.68 s) | **155,684.8 tok/s** (38,921) | **110,942.6 tok/s** (27,736) |
| **32–64K** | 53,799 (430,389) | **1.91 s** (2.65 s) | 0.29 s | **2.19 s** (2.86 s) | **162,424.4 tok/s** (40,606) | **144,910.0 tok/s** (36,228) |
| **16–32K** | 26,048 (208,387) | **1.16 s** (1.62 s) | 0.34 s | **1.44 s** (3.81 s) | **128,776.9 tok/s** (32,194) | **54,412.6 tok/s** (13,603) |
| **8–16K** | 12,922 (103,375) | **0.70 s** (1.00 s) | 0.31 s | **0.98 s** (1.30 s) | **103,050.4 tok/s** (25,763) | **78,933.8 tok/s** (19,734) |
| **1–8K** | 7,096 (56,765) | **0.56 s** (0.74 s) | 2.24 s | **2.91 s** (2.98 s) | **76,494.8 tok/s** (19,124) | **19,018.6 tok/s** (4,755) |

### 2.3 Empirical ISL (50:20:12:9:9) Mixed 5-Bucket Uncached Sweeps (c = 8)

| **Mixed Sweep (0% Cache Hit, c = 8)** | **Mean ISL (Total Tokens)** | **1P Wall** | **E2E Wall** | **1P Node Tok/s (Per Chip)** | **E2E 1P1D Tok/s (Per Chip)** |
| --- | --- | --- | --- | --- | --- |
| **Mixed 5-Bucket (40 Reqs)** | 36,876 (1,475,051) | **11.60 s** | **11.77 s** | **127,123.0 tok/s** (31,781) | **125,287.6 tok/s** (31,322) |
| **Mixed 5-Bucket (128 Reqs)** | 21,231 (2,717,527) | **19.59 s** | **19.95 s** | **138,722.9 tok/s** (34,681) | **136,240.2 tok/s** (34,060) |

---

## 3. Summary of Key Architectural Optimizations on 1P1D

1. **Asymmetric Topology (`4 × TP=2, EP=2` `1P` → `1 × DP=8, TP=1, EP=8` `1D`)**: Runs 4 independent 1-chip `TP=2, EP=2` prefill servers (`POD_G_CHIP_ID=0..3`, `TORCH_TPU_TOPOLOGY=1,1,1,2`), avoiding 4-chip ICI synchronization on prefill and enabling SparseCore collective offloads (`--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=true`).
2. **True 20-Layer Context Encoder (`VLLM_DSV41_PURE_20L_PREFILL=1`) + 128-Token SWA Tail Handoff**: `1P` executes only Layers `0..19` + Layer-20 global KV/state write (`max_num_batched_tokens = 16,384` / `32,768`). `TPURaidenConnector` transfers only Layers `0..20` (`TPU_RAIDEN_DSV4_MAX_TRANSFER_LAYER=20`, 35 pools) and truncates `num_computed_tokens` by 128 tokens (`VLLM_DSV41_PD_HANDOFF_TAIL_TOKENS=128`) so `1D` replays the final 128 prompt tokens locally in one step (`w384`) to populate Layers `20..39` SWA KV.
3. **Heterogeneous `TP=2 → TP=1` KV Resharding (`TPU_RAIDEN_POOL_TAGS_PER_LAYER=1`)**: Per-layer Raiden pool tag splitting (`split_layer_tag`) and `SEQ_ALONG_LANE` / `HEAD_ACROSS_LANES` layout overrides reassemble `TP=2` producer KV shards cleanly into `TP=1` consumer KV caches over DCN Stage 3.
4. **Compact SWA Ring Layout without HBM Aliasing**: Preserves the `128`-block compact SWA ring on `1P` while disabling `raw_tensors` HBM aliasing when `TPURaidenConnector` is active so concurrent D2H transfers cannot overlap in-place SWA ring writes.
5. **Disjoint `BlockPool` Partition Isolation (`main`, `state`, `swa_0`, `swa_1`, `swa_2`) on `1P` (`PREFILL_4X_TP2_GPU_MEM=0.92`)**: Allocates 36 separate HBM arrays across the 5 KV cache groups (`g0..g4`), each of length `num_blocks = 1,659` (`212,352` tokens/chip = `849,408` tokens/node per partition), and isolates each group (`main`, `state`, `swa_0`, `swa_1`, `swa_2`) into its own `1,658`-block `FreeKVCacheBlockQueue` keyed by `(block.pool_partition, block.block_id)`.

---

## 4. 1P1D Empirical ISL (50:20:12:9:9) Decode Benchmark (95% Prefix Cache Hit, 180s Window)

| **Sessions (n)** | **Trunks (4 × TP=2)** | **Mean ISL (Cache Hit %)** | **Mean OSL** | **Win Out Tok/s (Drain)** | **Win In Tok/s (Uncached)** | **Med TPS/Strm (P90_low / P90_hi)** | **Per-Tok ITL (StepChunk)** | **DSpark τ** | **Med TTFT (P90)** | **64K+ Med TTFT (P90)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **8** | 20 (5/chip) | 18,963 (**93.89%**) | 2,155.7 | **908.7** (634.8) | **9,270.8** (566.8) | **91.23** (77.3 / 113.0) | 10.96 ms (26.73 ms) | **3.24** | **0.60 s** (0.78 s) | **0.77 s** (0.79 s) |
| **16** | 20 (5/chip) | 18,503 (**94.22%**) | 1,356.9 | **1,520.1** (1,292.0) | **24,325.7** (1,405.1) | **89.95** (78.2 / 117.5) | 11.12 ms (28.53 ms) | **3.02** | **0.52 s** (0.70 s) | **0.70 s** (0.73 s) |
| **32** | 20 (5/chip) | 19,265 (**93.08%**) | 1,512.0 | **2,852.1** (1,845.0) | **42,023.8** (2,907.0) | **78.71** (67.7 / 110.0) | 12.71 ms (33.16 ms) | **3.22** | **0.53 s** (0.81 s) | **0.74 s** (0.98 s) |
| **64** | 20 (5/chip) | 18,876 (**93.62%**) | 1,431.6 | **5,001.6** (2,941.5) | **75,316.6** (4,808.5) | **69.72** (49.8 / 91.5) | 14.34 ms (34.83 ms) | **3.13** | **0.59 s** (1.26 s) | **0.76 s** (1.31 s) |
| **128** | 20 (5/chip) | 18,877 (**93.74%**) | 1,211.5 | **6,197.7** (3,874.1) | **116,289.2** (7,278.7) | **46.44** (35.2 / 62.6) | 21.53 ms (42.72 ms) | **3.05** | **0.76 s** (4.22 s) | **0.89 s** (1.36 s) |

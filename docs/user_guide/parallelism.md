# Parallelism & Scaling

`vllm-torchtpu` supports **6D composable parallelism** across single-host and multi-host Cloud TPU slices:

1. **Tensor Parallelism (TP)**: `--tensor-parallel-size`
2. **Data Parallelism (DP)**: `--data-parallel-size`
3. **Pipeline Parallelism (PP)**: `--pipeline-parallel-size`
4. **Expert Parallelism (EP)**: `--enable-expert-parallel`
5. **Prefill Context Parallelism (PCP)**: `--prefill-context-parallel-size`
6. **Decode Context Parallelism (DCP)**: `--decode-context-parallel-size`

---

## 1. Parallelism Dimensions Overview

| Dimension | CLI Flag | Primary Use Case | Mesh / Collective Behavior |
| :--- | :--- | :--- | :--- |
| **Tensor Parallelism (TP)** | `--tensor-parallel-size N` | Shard attention heads and dense MLP projections within a high-bandwidth ICI domain. | All-reduce / Reduce-scatter across `TP` ranks per layer. |
| **Data Parallelism (DP)** | `--data-parallel-size N` | Replicate attention layers across independent engines while sharing a global Expert Parallel MoE mesh (`TP * DP` experts). | Zero attention collective overhead when `TP=1`; All-to-All across `TP * DP` for MoE layers when `--enable-expert-parallel` is enabled. |
| **Expert Parallelism (EP)** | `--enable-expert-parallel` | Shard Mixture-of-Experts (MoE) routed experts across the `TP * DP` device mesh instead of slicing each expert's weights. | Uses Pallas SparseCore / GMM ragged dispatch and combine collectives across the EP mesh (`src/vllm_torchtpu/distributed/ep_mesh.py`). |
| **Pipeline Parallelism (PP)** | `--pipeline-parallel-size N` | Partition transformer layers sequentially across stages (`PP` stages). | Point-to-point activation transfers with dynamic microbatch chunking (`TPU_PP_DYNAMIC_CHUNKS=1`). |
| **Prefill Context Parallelism (PCP)** | `--prefill-context-parallel-size N` | Split long prefill sequences across `PCP` ranks to parallelize prefill compute and reduce time-to-first-token (TTFT). | Topology-aware ring/mesh assignment (`TPU_PCP_TOPOLOGY_AWARE_MESH=1`) minimizing ICI hop count between consecutive PCP ranks. |
| **Decode Context Parallelism (DCP)** | `--decode-context-parallel-size N` | Shard the KV cache along the sequence dimension across `DCP` ranks for long-context decode. | Online LSE (log-sum-exp) attention merge across `DCP` ranks (`TPU_DCP_LSE_MERGE_A2A`). |

> [!IMPORTANT]
> **Total Device Count**: For an in-process or multi-host deployment, the total number of logical TPU devices consumed by an engine group is `DP * TP * PP * PCP`. On **TPU v7x (Ironwood)**, each physical chip exposes **2 logical TensorCore devices**, so a 4-chip host provides **8 logical devices** (for example, `TP=8`, `DP=8, TP=1`, or `DP=4, TP=2`).

---

## 2. Choosing Between `TP` and `DP + EP` for MoE Models

For large Mixture-of-Experts (MoE) models with GQA or Multi-head Latent Attention (MLA)—such as `Qwen/Qwen3.5-397B-A17B`, `deepseek-ai/DeepSeek-V4`, or `zai-org/GLM-5.2-FP8`—combining **Data Parallelism (`DP`)** with **Expert Parallelism (`--enable-expert-parallel`)** often delivers significantly higher throughput than pure Tensor Parallelism (`TP`):

- **Attention stays chip-local (`TP=1` or `TP=2`)**: Avoiding wide 8-way or 16-way attention all-reduces on every layer.
- **MoE experts shard across the full slice (`EP = TP * DP`)**: Every device holds `num_experts / (TP * DP)` experts, keeping full HBM capacity for weights and KV cache.

### Example: `DP=8, TP=1 + EP` on a Single TPU v7x Host (8 Devices)

```bash
export USE_MOE_SPARSE_CORE=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768

vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 10240 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --block-size 256
```

### Example: `DP=4, TP=2 + EP` Hybrid Layout

When per-device HBM is tighter or higher single-stream concurrency is needed, pair `TP=2` (across the 2 logical cores on the same physical TPU v7x chip) with `DP=4`:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --async-scheduling
```

---

## 3. Hierarchical Expert Parallelism & MoE Kernel Tuning

On multi-chip and multi-host slices, `vllm-torchtpu` provides specialized MoE execution paths controlled via environment variables:

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `USE_MOE_SPARSE_CORE` | `1` | Offloads MoE token routing/gather-reduce collectives to TPU SparseCores. |
| `USE_MOE_FUSED_EP_KERNEL` | `0` | Enables the fused Expert-Parallel Pallas kernel (`v1` or `v2` via `MOE_FUSED_EP_KERNEL_IMPL`). |
| `TPU_MOE_HIERARCHICAL_EP` | `0` | Enables two-level (intra-chip / inter-chip) hierarchical EP token dispatch on dual-core TPU v7x topologies. |
| `TPU_MOE_SEQUENCE_PARALLEL` | `0` | Keeps activations sequence-parallel entering MoE layers to replace all-reduce with reduce-scatter + all-gather. |
| `TPU_LATENT_PROJ_INTRA_CHIP_TP` | `0` | Shards MLA latent projections across the 2 cores of each physical TPU v7x chip even when `TP=1`. |

### Static Expert Placement

With `--enable-expert-parallel`, each EP rank holds an equal, contiguous block of every MoE layer's routed experts, so when the traffic favours a few experts, the ranks that hold them carry more of each layer's rows than the others. A static expert placement chooses, per layer, which rank holds which expert, from routing counts collected on the served traffic. Two `--additional-config` keys control it:

| Key | Value | Description |
| :--- | :--- | :--- |
| `moe_expert_load_stats` | `true` or `false` (default) | Counts the routed rows per expert of every MoE layer on the steps the sequence-parallel MoE prefill path serves (`TPU_MOE_PREFILL_SEQUENCE_PARALLEL`), padding excluded. |
| `moe_expert_placement` | path to a placement table | Loads logical expert `e` of layer `L` as physical expert `slots[L][e]` of the table, and maps the router's selected experts through the table after the top-k. Without it, experts load in checkpoint order. |

The worker method `dump_moe_expert_load(stats_path, table_path)`, a collective RPC, reads the counts: it sums them over the TP group, writes them to `stats_path` and their placement table to `table_path` on every host, and zeroes them. The OpenAI-compatible server exposes collective RPCs at `POST /collective_rpc` when started with `VLLM_SERVER_DEV_MODE=1`; an offline `LLM` calls `llm.collective_rpc("dump_moe_expert_load", args=(stats_path, table_path))`. The table builder also runs offline on a saved dump:

```bash
python -m vllm_torchtpu.distributed.expert_placement stats.json table.json
```

It places each layer's experts longest first: an expert goes to the EP group (the ranks that split an expert's weights between them) with the fewest rows among those with room left, then to the group member with the fewest rows.

**Calibrate, then serve.** Start the server as it will be served, plus `VLLM_SERVER_DEV_MODE=1` and `--additional-config '{"moe_expert_load_stats": true}'`. When it is ready, dump once to drop the counts of precompilation and warmup:

```bash
curl -X POST http://localhost:8000/collective_rpc -H 'Content-Type: application/json' \
  -d '{"method": "dump_moe_expert_load", "args": ["/tmp/warmup_stats.json", "/tmp/warmup_table.json"]}'
```

Then send traffic representative of the deployment and dump again, to the paths the placed server will read (for example `"args": ["/data/placement/stats.json", "/data/placement/table.json"]`); every host writes its own copy. Restart without `moe_expert_load_stats` and `VLLM_SERVER_DEV_MODE`, with the same model and parallel configuration and `--additional-config '{"moe_expert_placement": "/data/placement/table.json"}'`.

The counts need the token padding mask (`TPU_MOE_SKIP_PADDED_TOKENS`, on by default) and a router with sigmoid scores and a score-correction bias; grouped top-k is refused. A placement needs `--enable-expert-parallel`, `--load-format runai_streamer`, data parallel and prefill context parallel sizes of 1, the linear expert placement strategy, and a DeepSeek-V2-family MoE model whose router has a score-correction bias. It is refused with EPLB, with `USE_MOE_FUSED_EP_KERNEL=1` or `FUSED_MOE_DECODE_KERNEL=1`, and when the fused-norm prequantized MoE path is armed (`TPU_FUSED_ADD_RMSNORM_QUANT_MOE=1` with `FUSED_MOE_PREFILL_KERNEL_V2=1`). The server validates the table when it loads the model, and fails to start on a table for another expert count, EP size, EP x TP split or set of MoE layers, or on a layer that is not a permutation. The table's path is part of the compile cache key, so a table at a new path compiles again.

Experts are always selected in the router's own (logical) order, so a placed server selects the same experts, with the same weights, as an unplaced one, ties included. Steps on the sequence-parallel MoE prefill path are bit-identical to an unplaced server for any table. Smaller steps, decode included, sum each rank's experts before the all-reduce, so when the table moves an expert to another rank they match an unplaced server within floating-point tolerance rather than bit for bit.

---

## 4. Pipeline Parallelism (`--pipeline-parallel-size`)

Pipeline Parallelism partitions model layers across `PP` stages and is composable with `TP`:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --max-model-len 8192
```

By default, `TPU_PP_DYNAMIC_CHUNKS=1` dynamically sizes microbatch chunks to keep pipeline bubbles minimal without recompiling static token buckets.

---

## 5. Context Parallelism (`PCP` and `DCP`)

For long-context workloads (32K–256K+ tokens) or Prefill/Decode disaggregated serving:

- **Prefill Context Parallelism (`--prefill-context-parallel-size`)**: Distributes prefill sequence chunks across `PCP` ranks. `TPU_PCP_TOPOLOGY_AWARE_MESH=1` (enabled by default) binds consecutive PCP ranks to adjacent physical chips on the ICI torus to minimize ring-attention hop latency.
- **Decode Context Parallelism (`--decode-context-parallel-size`)**: Distributes KV cache pages across `DCP` ranks during decode.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 2 \
  --prefill-context-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 65536
```

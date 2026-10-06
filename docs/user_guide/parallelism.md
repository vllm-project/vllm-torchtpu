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

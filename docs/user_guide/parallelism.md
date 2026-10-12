# Parallelism & Scaling

`vllm-torchtpu` supports six parallelism dimensions across single-host and multi-host Cloud TPU slices. Not every combination composes; see the limits in the [Pipeline Parallelism](#4-pipeline-parallelism-pipeline-parallel-size) and [Context Parallelism](#5-context-parallelism-pcp-and-dcp) sections:

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
| **Pipeline Parallelism (PP)** | `--pipeline-parallel-size N` | Partition transformer layers sequentially across stages (`PP` stages). | Point-to-point activation transfers, with prefill chunks sized by a per-step time budget (`TPU_PP_DYNAMIC_CHUNKS=1`). |
| **Prefill Context Parallelism (PCP)** | `--prefill-context-parallel-size N` | Split long prefill sequences across `PCP` ranks to parallelize prefill compute and reduce time-to-first-token (TTFT). | Topology-aware ring/mesh assignment (`TPU_PCP_TOPOLOGY_AWARE_MESH=1`) minimizing ICI hop count between consecutive PCP ranks. |
| **Decode Context Parallelism (DCP)** | `--decode-context-parallel-size N` | Shard the KV cache along the sequence dimension across `DCP` ranks for long-context decode. | LSE (log-sum-exp) attention merge across `DCP` ranks (reduce-scatter by default; `TPU_DCP_LSE_MERGE_A2A=1` opts in to an all-to-all merge for small MLA decode steps). |

> [!IMPORTANT]
> **Total Device Count**: For an in-process or multi-host deployment, the total number of logical TPU devices consumed by an engine group is `DP * TP * PP * PCP`. On **TPU v7x (Ironwood)**, each physical chip exposes **2 logical TensorCore devices**, so a 4-chip host provides **8 logical devices** (for example, `TP=8`; `DP=8 × TP=1`; or `DP=4 × TP=2`).

---

## 2. Choosing Between `TP` and `DP + EP` for MoE Models

For large Mixture-of-Experts (MoE) models with GQA or Multi-head Latent Attention (MLA)—such as `Qwen/Qwen3.5-397B-A17B`, `deepseek-ai/DeepSeek-V4`, or `zai-org/GLM-5.2-FP8`—combining **Data Parallelism (`DP`)** with **Expert Parallelism (`--enable-expert-parallel`)** often delivers significantly higher throughput than pure Tensor Parallelism (`TP`):

- **Attention stays chip-local (`TP=1` or `TP=2`)**: Avoiding wide 8-way or 16-way attention all-reduces on every layer.
- **MoE experts shard across the full slice (`EP = TP * DP`)**: Every device holds `num_experts / (TP * DP)` experts, keeping full HBM capacity for weights and KV cache.

> [!WARNING]
> Use `DP > 1` with MoE models and `--enable-expert-parallel`. All DP engines share one TPU slice, so an idle engine must keep issuing the collectives of the busy ones. The idle-rank pairing that does this runs only with expert parallelism, so dense-model DP can deadlock once the engines drain unevenly.

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
| `USE_MOE_FUSED_EP_KERNEL` | `0` | Enables the fused Expert-Parallel Pallas kernel. Setting `MOE_FUSED_EP_KERNEL_IMPL` (`v2` or `adaptive`) also enables it. |
| `TPU_MOE_HIERARCHICAL_EP` | `0` | Hierarchical MoE parallelism: expert-parallel across chips, tensor-parallel across the two cores of a TPU v7x chip. Helps only at low concurrency. |
| `TPU_MOE_SEQUENCE_PARALLEL` | `0` | Keeps activations sequence-parallel entering MoE layers to replace all-reduce with reduce-scatter + all-gather. Qwen3.5 MoE with EP and `TP > 1` only. |
| `TPU_LATENT_PROJ_INTRA_CHIP_TP` | `0` | Shards Kimi-K3's unquantized latent-MoE down/up projections across the 2 cores of each TPU v7x chip instead of replicating them. |

---

## 4. Pipeline Parallelism (`--pipeline-parallel-size`)

Pipeline Parallelism partitions model layers across `PP` stages:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --no-async-scheduling \
  --max-model-len 8192
```

When `PP > 1`, the following are not supported yet and are rejected at startup:

- Tensor parallelism inside a stage: use `--tensor-parallel-size 1`.
- Stages that span several devices (`PCP > 1`, or `DP > 1` with `--enable-expert-parallel`), unless the installed torch_tpu builds Pallas meshes over a stage's process group.
- Async scheduling: pass `--no-async-scheduling`.
- Speculative decoding.
- KV transfer connectors other than `TPURaidenConnector`. With `TPURaidenConnector`, every stage must hold at least one full-attention layer (checked for hybrid models that declare `layer_types`).
- The legacy Ray executor: set `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` when using Ray.

By default, `TPU_PP_DYNAMIC_CHUNKS=1` sizes prefill chunks by a per-step time budget instead of a token budget. Set it to `0` to keep token budgets.

---

## 5. Context Parallelism (`PCP` and `DCP`)

For long-context workloads (32K–256K+ tokens) or Prefill/Decode disaggregated serving:

- **Prefill Context Parallelism (`--prefill-context-parallel-size`)**: Distributes prefill sequence chunks across `PCP` ranks. `TPU_PCP_TOPOLOGY_AWARE_MESH=1` (enabled by default) binds consecutive PCP ranks to adjacent physical chips on the ICI torus to minimize ring-attention hop latency.
- **Decode Context Parallelism (`--decode-context-parallel-size`)**: Distributes KV cache pages across `DCP` ranks during decode.

PCP limits: it cannot be combined with DCP, with `PP > 1` it is subject to the PP limits above, MoE models require `--enable-expert-parallel`, and speculative decoding is limited to `"method": "mtp"` with `num_speculative_tokens=1`.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 2 \
  --prefill-context-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 65536
```

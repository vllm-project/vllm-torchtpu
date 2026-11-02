# Multi-Host Serving

When a model or parallelism configuration spans multiple TPU VM hosts in a pod slice (for example, `v7x-16`, `v7x-32`, `v7x-64`, or `v6e-16`), `vllm-torchtpu` coordinates the hosts across the high-speed Inter-Chip Interconnect (ICI) mesh.

`vllm-torchtpu` supports two multi-host execution backends controlled by `TPU_MULTIHOST_BACKEND`:

1. **Multiprocessing (`mp`) Backend (Default)**: Each TPU host runs its own `vllm serve` process and exchanges slice addresses over a `torch.distributed` `TCPStore` rendezvous (`src/vllm_torchtpu/distributed/tpu_mp_multihost.py`). No Ray cluster is required.
2. **Ray (`ray`) Backend (`TPU_MULTIHOST_BACKEND=ray`)**: A Ray cluster spans all TPU hosts in the slice, and a single `vllm serve` driver on the head node schedules `RayDistributedExecutor` workers across the cluster.

---

## Option 1: Native Multiprocessing (`mp`) Multi-Host Backend

The `mp` backend supports two multi-host deployment modes without Ray:

### Mode A: Multi-Node Tensor / Pipeline / Context Parallelism (`--nnodes`)

When TP, PP, or PCP spans multiple hosts (`--nnodes N`), launch `vllm serve` on every host in the slice. Host 0 serves the OpenAI HTTP entrypoint, while ranks `1..N-1` pass `--headless`:

=== "Host 0 (Leader)"

    ```bash
    vllm serve Qwen/Qwen3.5-397B-A17B \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --nnodes 2 \
      --node-rank 0 \
      --master-addr <HOST_0_INTERNAL_IP> \
      --master-port 29500 \
      --port 8000
    ```

=== "Host 1..N-1 (Headless Workers)"

    ```bash
    vllm serve Qwen/Qwen3.5-397B-A17B \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --nnodes 2 \
      --node-rank 1 \
      --master-addr <HOST_0_INTERNAL_IP> \
      --master-port 29500 \
      --headless
    ```

### Mode B: Multi-Host Data Parallelism (`--data-parallel-size`)

When `data_parallel_size` is wider than `data_parallel_size_local` (for example, `DP=16` across two 8-device TPU v7x hosts with `DP_LOCAL=8`), each host runs `vllm serve` for its local slice of DP ranks:

=== "Host 0 (`--data-parallel-start-rank 0`)"

    ```bash
    vllm serve Qwen/Qwen3.5-397B-A17B \
      --tensor-parallel-size 1 \
      --data-parallel-size 16 \
      --data-parallel-size-local 8 \
      --data-parallel-start-rank 0 \
      --data-parallel-address <HOST_0_INTERNAL_IP> \
      --data-parallel-rpc-port 13345 \
      --enable-expert-parallel \
      --port 8000
    ```

=== "Host 1 (`--data-parallel-start-rank 8`)"

    ```bash
    vllm serve Qwen/Qwen3.5-397B-A17B \
      --tensor-parallel-size 1 \
      --data-parallel-size 16 \
      --data-parallel-size-local 8 \
      --data-parallel-start-rank 8 \
      --data-parallel-address <HOST_0_INTERNAL_IP> \
      --data-parallel-rpc-port 13345 \
      --enable-expert-parallel \
      --headless
    ```

During `VllmConfig` initialization, `vllm-torchtpu` automatically runs a `TCPStore` rendezvous across all hosts to populate `TORCH_TPU_SLICEBUILDER_ADDRESSES` and `TORCH_TPU_TOPOLOGY` for the full multi-host ICI mesh.

---

## Option 2: Ray (`ray`) Multi-Host Backend on GCE

Set `TPU_MULTIHOST_BACKEND=ray` to use the Ray distributed executor across multiple TPU VMs. The instructions below are included from `examples/multihost/README.md`:

---8<--- "examples/multihost/README.md:5"

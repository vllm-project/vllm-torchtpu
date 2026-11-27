# Multi-Host Serving

When a model or parallelism configuration spans multiple TPU VM hosts in a pod slice (for example, `v7x-16`, `v7x-32`, `v7x-64`, or `v6e-16`), `vllm-torchtpu` coordinates the hosts across the high-speed Inter-Chip Interconnect (ICI) mesh.

`vllm-torchtpu` supports two multi-host execution backends controlled by `TPU_MULTIHOST_BACKEND`:

1. **Multiprocessing (`mp`) Backend (Default)**: Each TPU host runs its own `vllm serve` process and exchanges slice addresses over a `torch.distributed` `TCPStore` rendezvous (`src/vllm_torchtpu/distributed/tpu_mp_multihost.py`). No Ray cluster is required.
2. **Ray (`ray`) Backend (`TPU_MULTIHOST_BACKEND=ray`)**: A Ray cluster spans all TPU hosts in the slice, and a single `vllm serve` driver on the head node schedules `RayDistributedExecutorV2` workers across the cluster.

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

---

## Option 3: Kubernetes LeaderWorkerSet (LWS) and Bare-Metal IPv6 Clusters

On bare-metal or on-prem Kubernetes clusters managed by [LeaderWorkerSet (LWS)](https://lws.sigs.k8s.io/), GCE instance metadata is unavailable and pods may communicate over IPv6. Both the `mp` and `ray` multi-host backends support LWS deployments using the orchestrator-injected `LWS_LEADER_ADDRESS` and `LWS_WORKER_INDEX` environment variables:

- **Worker identity without GCE metadata**: When `LWS_WORKER_INDEX` or `LWS_LEADER_ADDRESS` is set, `tpu_info.get_node_worker_id()` reads `TPU_WORKER_ID`, then `LWS_WORKER_INDEX` (defaulting to `0` when only `LWS_LEADER_ADDRESS` is set), and `tpu_info.get_node_name()` reads `TPU_NAME` or falls back to `socket.gethostname()`, so LWS pods never query the GCE metadata server. Both values only appear in the startup log.
- **Host order**: With `mp`, hosts are ordered by `--node-rank` (or `--data-parallel-start-rank` for multi-host DP). With `ray`, hosts are ordered by the `ray.io/tpu-worker-id` node label (when contiguous `0..N-1`); when that label is missing and `LWS_LEADER_ADDRESS` is set, by each node's `LWS_WORKER_INDEX` (when contiguous `0..N-1`); otherwise by node IP, with a warning that the order may not match the ICI topology.
- **IPv6 slicebuilder hostnames**: Because TorchTPU currently parses `TORCH_TPU_SLICEBUILDER_ADDRESSES` on `:` (tracked in `google-pytorch/torch_tpu#7040`), each IPv6 host resolves and publishes its own colon-free, forward-resolvable, non-loopback hostname (LWS pod DNS from `LWS_LEADER_ADDRESS` + `LWS_WORKER_INDEX`, local FQDN/hostname, or reverse DNS PTR). With `mp`, each peer publishes its hostname alongside its IP in the `TCPStore` rendezvous and validates all published hostnames; with `ray`, the driver probes each remote node once (failing startup if the probe fails or times out after 120 s) and validates all published hostnames. Validation rejects loopback or duplicate hostnames and verifies that every hostname resolves to a non-loopback address (retrying up to 30 s for DNS propagation).
- **Kubernetes device-plugin `TPU_VISIBLE_DEVICE_PATHS`**: `libtpu` prioritizes `TPU_VISIBLE_DEVICE_PATHS` over `TPU_VISIBLE_CHIPS` and opens every listed path in each worker process. `TPUWorker` clears `TPU_VISIBLE_DEVICE_PATHS` whenever multiple slice ranks share a host, more than one path is listed, or `VLLM_TPU_DEVICE_ORDER`, `TPU_LOCAL_RANK_OFFSET`, or `DEBUG_TPU_LOCAL_RANK_OFFSET` is set.

### Known Limitations

- **Ray resource over-counting on privileged SR-IOV pods**: On privileged bare-metal pods where `/dev/vfio` exposes SR-IOV management functions alongside compute VFs, Ray's automatic TPU resource detection can over-report device counts (tracked in `ray-project/ray#68548`). Start Ray on every host with an explicit TPU count, for example `ray start --resources '{"TPU": <usable_chips_per_host>}' ...`.
- **No sub-host partitioning via multi-path `TPU_VISIBLE_DEVICE_PATHS` with `TP > 1`**: Because `TPUWorker` clears pod-wide `TPU_VISIBLE_DEVICE_PATHS` when more than one rank shares a host or multiple paths are present, do not rely on a multi-path `TPU_VISIBLE_DEVICE_PATHS` list to partition a single host across concurrent `TP > 1` servers. Use separate pods or explicit `TPU_LOCAL_RANK_OFFSET` instead.
- **Unique forward-resolvable hostnames required on IPv6 slices**: Every host in an IPv6 slice must resolve to a distinct, non-loopback, colon-free hostname reachable by all peers; duplicate or loopback hostnames are rejected at startup.

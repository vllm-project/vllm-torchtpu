# Disaggregated Serving (Prefill / Decode)

Prefill/Decode (P/D) disaggregation separates compute-bound **Prefill** (`kv_producer`) and memory-bandwidth-bound **Decode** (`kv_consumer`) onto independent TPU engine pools. This prevents long-prompt prefills from stalling ongoing decode steps (eliminating TPOT spikes) and allows each pool to use the parallelism layout (`TP`, `DP`, `PCP`, `EP`) best suited to its phase.

`vllm-torchtpu` supports two KV cache transfer connectors:

1. **Raiden KV Connector (`TpuRaidenConnector`)**: High-bandwidth PJRT DMA/RDMA engine powered by `tpu-sync-torch` (`tpu_sync`) with a central coordinator (`run_raiden_controller.py`) and support for **heterogeneous P/D resharding** (e.g., Prefill with `PCP=4, TP=2` transferring KV cache into Decode with `DP=8, TP=1`).
2. **Host TCP/Side-Channel Connector (`TPUConnector`)**: Lightweight single-host or multi-host connector for functional testing and local disaggregation (`examples/disagg/run_disagg_single_host.sh`).

---

## 1. Architecture Overview (Raiden P/D Disaggregation)

A production Raiden P/D deployment consists of four components:

1. **Raiden Controller (`examples/disagg/run_raiden_controller.py`)**: Coordinates KV transfer metadata and pool registration across Prefill and Decode workers.
2. **Prefill Engine Pool (`kv_role="kv_producer"`)**: Computes prompt prefill, writes KV blocks into the unified/Raiden block pool, and streams KV blocks to the Decode pool.
3. **Decode Engine Pool (`kv_role="kv_consumer"`)**: Receives KV blocks into its local HBM KV cache (resharding on the fly if P and D meshes differ) and generates output tokens.
4. **Disaggregated Proxy Router (`examples/disagg/toy_proxy_server.py`)**: Accepts client OpenAI HTTP requests, routes a `max_tokens=1` prefill request to a Prefill instance, and forwards the request with KV transfer parameters to a Decode instance.

---

## 2. Single-Host Disaggregated Serving Quickstart

On an 8-device TPU host, you can launch a complete Prefill + Decode + Proxy pipeline using `examples/disagg/run_disagg_single_host.sh`:

```bash
MODEL="Qwen/Qwen3-0.6B" \
INPUT_LEN=512 \
OUTPUT_LEN=128 \
NUM_PROMPTS=200 \
bash examples/disagg/run_disagg_single_host.sh
```

The script pins separate `/dev/vfio/<N>` devices to the Prefill (`kv_producer`) and Decode (`kv_consumer`) `vllm serve` processes, starts `examples/disagg/toy_proxy_server.py` on port `8000`, and runs `vllm bench serve` against the proxy.

---

## 3. Multi-Host Disaggregated Serving with Raiden (`TpuRaidenConnector`)

For large models such as `Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3.5-35B-A3B-FP8`, `moonshotai/Kimi-K3`, and `deepseek-ai/DeepSeek-V4`, use `TpuRaidenConnector` with `TPU_USE_RAIDEN_KV_CACHE_MANAGER=1`.

### Step 1: Start the Raiden Controller

Run the Raiden controller on a reachable host/port (default `50055`):

```bash
python examples/disagg/run_raiden_controller.py \
  --host 0.0.0.0 \
  --port 50055
```

### Step 2: Launch Prefill Instance (`kv_producer`)

```bash
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_CONTROLLER_ADDRESS="<CONTROLLER_IP>:50055"
export TPU_RAIDEN_JOB_NAME="prefill"
export TPU_RAIDEN_ENGINE_ID="0"
export TPU_RAIDEN_QWEN35_ADMISSION=1

vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --port 8400 \
  --tensor-parallel-size 4 \
  --prefill-context-parallel-size 2 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --attention-backend CUSTOM \
  --kv-transfer-config '{
    "kv_connector": "TpuRaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.raiden.raiden_connector",
    "kv_role": "kv_producer"
  }'
```

### Step 3: Launch Decode Instance (`kv_consumer`)

```bash
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_CONTROLLER_ADDRESS="<CONTROLLER_IP>:50055"
export TPU_RAIDEN_JOB_NAME="decode"
export TPU_RAIDEN_ENGINE_ID="0"
export TPU_RAIDEN_QWEN35_ADMISSION=1

vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --port 9400 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --attention-backend CUSTOM \
  --kv-transfer-config '{
    "kv_connector": "TpuRaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.raiden.raiden_connector",
    "kv_role": "kv_consumer"
  }'
```

### Step 4: Launch the Disaggregated Proxy Server

```bash
python examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts <PREFILL_HOST_IP> \
  --prefiller-ports 8400 \
  --decoder-hosts <DECODE_HOST_IP> \
  --decoder-ports 9400
```

---

## 4. Reference Launch Scripts & GKE Manifests

| Script / Manifest | Description |
| :--- | :--- |
| `examples/disagg/launch_qwen35_disagg_multihost.sh` | Turnkey launcher for Qwen3.5 P/D disaggregation supporting `ROLE=controller\|prefill\|decode\|proxy\|all`. |
| `examples/disagg/launch_dsv4_disagg_multihost.sh` | Turnkey launcher for DeepSeek-V4 multi-host P/D disaggregation. |
| `examples/disagg/run_qwen35_raiden_pcp8_dp8_live_pair.sh` | Live pair script demonstrating heterogeneous resharding from Prefill (`PCP=8`) to Decode (`DP=8`). |
| `.buildkite/kubernetes/manifests/workloads/qwen3.5-397b-1p1d.yaml` | GKE JobSet manifest for `Qwen3.5-397B-A17B` 1P1D disaggregated serving. |
| `.buildkite/kubernetes/manifests/workloads/kimi-k3-1p1d-4dp8tp.yaml` | GKE JobSet manifest for `Kimi-K3` 1P1D disaggregated serving. |

---

## 5. Key Raiden Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TPU_USE_RAIDEN_KV_CACHE_MANAGER` | `0` | Enables the Raiden-compatible KV cache block layout and manager. |
| `TPU_RAIDEN_CONTROLLER_ADDRESS` | `""` | `host:port` of the Raiden coordinator (`run_raiden_controller.py`). |
| `TPU_RAIDEN_JOB_NAME` | `""` | Logical role identifier (`prefill` or `decode`) registered with the Raiden controller. |
| `TPU_RAIDEN_ENGINE_ID` | `"0"` | Unique engine replica ID within the job role. |
| `TPU_RAIDEN_TRANSFER_PARALLELISM` | `8` | Number of concurrent DMA transfer streams per worker. |
| `TPU_RAIDEN_RESHARD_IMPL` | `"controller"` | KV resharding implementation (`controller` or `host`) when Prefill and Decode shardings differ. |
| `TPU_KV_RESHARD_TRANSPORT` | `"zmq"` | Transport protocol for host-side KV resharding (`zmq` or `tcp`). |

---

## 6. Both Legs Must Cache the Same Layout

Each worker registers a layout fingerprint with the Raiden controller: a hash of its KV cache page layout and of anything else a peer needs to read its pages. The controller refuses a transfer between a Prefill unit and a Decode unit that registered different fingerprints, so the Decode never loads pages it would misread.

For GLM (`TPU_RAIDEN_GLM_ADMISSION=1`), the fingerprint includes the rope channel order of the cached MLA `k_pe` and of the indexer key:

- With `TPU_MLA_ROPE_NEOX_PERMUTE` / `TPU_INDEXER_ROPE_NEOX_PERMUTE` on (the default), an engine permutes its rope projection columns after load and caches those channels in half-split order. The payload then declares `"mla_rope_order":"half_split"` / `"indexer_rope_order":"half_split"`.
- With them off, or on a tree without the permute, the engine caches the checkpoint's interleaved order and declares nothing.
- A pipeline stage whose layers hold no indexer declares the order an indexer layer of its own would cache: half-split when `TPU_INDEXER_ROPE_NEOX_PERMUTE` is on and the model config sets `indexer_rope_interleave`.

The page geometry is the same in both orders, so run both legs with the same two settings: a half-split leg and an interleaved leg register different fingerprints, and every transfer between them is refused. The refusal itself comes from the controller. The Decode worker logs one line per refused request, at ERROR, carrying the controller's message, and then waits for the native transfer's terminal state:

```text
(Worker_TP0_DCP0_EP0 pid=<pid>) ERROR <date> <time> [tpu_connector.py:6201] Stage-3 controller transfer outcome uncertain; waiting for native terminal state req_id=<request id> destination_req_id=<request id> uuid=<transfer id>: Remote Controller Server execution failed: Layout fingerprint mismatch between source and destination
```

Each worker logs its registration in one line with the fingerprint and its payload, so the two legs' lines show the field that differs:

```text
Raiden work unit registered controller=<addr> unit=<unit> data=<addr> listener=<addr> page_tokens=1024 transfer_rank=0/8 interleave_tokens=<n> layout_fingerprint=<sha256> layout_fingerprint_payload={"indexer_rope_order":"half_split",...,"mla_rope_order":"half_split",...}
```

# Disaggregated Serving (Prefill / Decode)

Prefill/Decode (P/D) disaggregation separates compute-bound **Prefill** (`kv_producer`) and memory-bandwidth-bound **Decode** (`kv_consumer`) onto independent TPU engine pools. This prevents long-prompt prefills from stalling ongoing decode steps (eliminating TPOT spikes) and allows each pool to use the parallelism layout (`TP`, `DP`, `PCP`, `EP`) best suited to its phase.

`vllm-torchtpu` supports two KV cache transfer connectors:

1. **Raiden KV Connector (`TPURaidenConnector`)**: High-bandwidth PJRT DMA/RDMA engine powered by `tpu-sync-torch` (`tpu_sync`) with a reshard control plane (in-engine store or external controller) and support for **heterogeneous P/D resharding** (e.g., Prefill with `PCP=4, TP=2` transferring KV cache into Decode with `DP=8, TP=1`).
2. **Host TCP/Side-Channel Connector (`TPUConnector`)**: Lightweight single-host or multi-host connector for functional testing and local disaggregation (`examples/disagg/run_disagg_single_host.sh`).

---

## 1. Architecture Overview (Raiden P/D Disaggregation)

A production Raiden P/D deployment consists of four components:

1. **Reshard Control Plane**: Coordinates KV transfer metadata and pool registration across Prefill and Decode workers. In store mode (`TPU_RAIDEN_RESHARD_IMPL=store`) it runs inside each engine; in controller mode it is an external per-role controller (`examples/disagg/run_raiden_controller.py`).
2. **Prefill Engine Pool (`kv_role="kv_producer"`)**: Computes prompt prefill, writes KV blocks into the unified/Raiden block pool, and streams KV blocks to the Decode pool.
3. **Decode Engine Pool (`kv_role="kv_consumer"`)**: Receives KV blocks into its local HBM KV cache (resharding on the fly if P and D meshes differ) and generates output tokens.
4. **Disaggregated Proxy Router (`examples/disagg/toy_proxy_server.py`)**: Accepts client OpenAI HTTP requests, routes a `max_tokens=1` prefill request to a Prefill instance, and forwards the request with KV transfer parameters to a Decode instance.

---

## 2. Single-Host Disaggregated Serving Quickstart

On an 8-device TPU host, you can launch a complete Prefill + Decode + Proxy pipeline with the `TPUConnector` using `examples/disagg/run_disagg_single_host.sh`:

```bash
MODEL="Qwen/Qwen3-0.6B" \
INPUT_LEN=512 \
OUTPUT_LEN=128 \
NUM_PROMPTS=200 \
bash examples/disagg/run_disagg_single_host.sh
```

The script pins separate `/dev/vfio/<N>` devices to the Prefill (`kv_producer`) and Decode (`kv_consumer`) `vllm serve` processes, starts `examples/disagg/toy_proxy_server.py` on port `8000`, and runs `vllm bench serve` against the proxy.

---

## 3. Multi-Host Disaggregated Serving with Raiden (`TPURaidenConnector`)

For large models such as `Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3.5-35B-A3B-FP8`, `moonshotai/Kimi-K3`, `deepseek-ai/DeepSeek-V4`, and GLM-5.2, use `TPURaidenConnector` with `TPU_USE_RAIDEN_KV_CACHE_MANAGER=1`, `TPU_KV_RESHARD_TRANSPORT=raiden`, and the model's admission flag: `TPU_RAIDEN_QWEN35_ADMISSION`, `TPU_RAIDEN_KIMIK3_ADMISSION`, `TPU_RAIDEN_DSV4_ADMISSION`, or `TPU_RAIDEN_GLM_ADMISSION`.

### Step 1: Choose the Reshard Control Plane

The steps below use **store mode** (`TPU_RAIDEN_RESHARD_IMPL=store`), as the reference launchers do: each engine's rank-0 worker hosts the reshard service in-process, so no extra process is needed. Engine `dp_rank` `i` binds `TPU_RAIDEN_RESHARD_PORT_BASE + i` and `TPU_RAIDEN_STORE_DISPATCH_PORT_BASE + i` on `TPU_RAIDEN_ADVERTISE_HOST`.

In controller mode (`TPU_RAIDEN_RESHARD_IMPL=controller`, the default), each role instead dials an external controller at `TPU_RAIDEN_CONTROLLER_ADDRESS`, for example one started with `python examples/disagg/run_raiden_controller.py --port 27000`.

### Step 2: Launch Prefill Instance (`kv_producer`)

```bash
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_QWEN35_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_ADVERTISE_HOST="<PREFILL_HOST_IP>"
export TPU_RAIDEN_RESHARD_PORT_BASE=27000
export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100
export TPU_RAIDEN_JOB_NAME="prefill"
export TPU_RAIDEN_ENGINE_ID="prefill-engine"
export TPU_KV_TRANSFER_PORT=9100

vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --port 8400 \
  --tensor-parallel-size 4 \
  --prefill-context-parallel-size 2 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --attention-backend CUSTOM \
  --kv-transfer-config '{
    "kv_connector": "TPURaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
    "kv_role": "kv_producer",
    "kv_port": 14579
  }'
```

### Step 3: Launch Decode Instance (`kv_consumer`)

```bash
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_QWEN35_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_ADVERTISE_HOST="<DECODE_HOST_IP>"
export TPU_RAIDEN_RESHARD_PORT_BASE=28000
export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28100
export TPU_RAIDEN_JOB_NAME="decode"
export TPU_RAIDEN_ENGINE_ID="decode-engine"
export TPU_KV_TRANSFER_PORT=9200

vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --port 9400 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --attention-backend CUSTOM \
  --kv-transfer-config '{
    "kv_connector": "TPURaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
    "kv_role": "kv_consumer",
    "kv_port": 14579
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
| `examples/disagg/launch_qwen35_disagg_multihost.sh` | Turnkey launcher for Qwen3.5 P/D disaggregation (store mode) supporting `ROLE=prefill\|decode\|all`. |
| `examples/disagg/launch_dsv4_disagg_multihost.sh` | Turnkey launcher for DeepSeek-V4 multi-host P/D disaggregation. |
| `examples/disagg/run_qwen35_raiden_pcp8_dp8_live_pair.sh` | Live pair script demonstrating heterogeneous resharding from Prefill (`PCP=8`) to Decode (`DP=8`). |
| `.buildkite/kubernetes/manifests/workloads/qwen3.5-397b-1p1d.yaml` | GKE JobSet manifest for `Qwen3.5-397B-A17B` 1P1D disaggregated serving. |
| `.buildkite/kubernetes/manifests/workloads/kimi-k3-1p1d-4dp8tp.yaml` | GKE JobSet manifest for `Kimi-K3` 1P1D disaggregated serving. |

---

## 5. Key Raiden Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TPU_USE_RAIDEN_KV_CACHE_MANAGER` | `0` | Enables the Raiden-compatible KV cache block layout and manager. |
| `TPU_RAIDEN_QWEN35_ADMISSION` / `_KIMIK3_` / `_DSV4_` / `_GLM_` | `0` | Per-model Raiden pool admission; set the one matching the model on both roles. |
| `TPU_RAIDEN_CONTROLLER_ADDRESS` | `""` | `host:port` of the external Raiden controller (`run_raiden_controller.py`); used when `TPU_RAIDEN_RESHARD_IMPL=controller`. |
| `TPU_RAIDEN_JOB_NAME` | `""` | Logical role identifier (`prefill` or `decode`) registered with the Raiden controller. |
| `TPU_RAIDEN_ENGINE_ID` | `"0"` | Unique engine replica ID within the job role. |
| `TPU_RAIDEN_TRANSFER_PARALLELISM` | `8` | Number of concurrent DMA transfer streams per worker. |
| `TPU_RAIDEN_RESHARD_IMPL` | `"controller"` | Reshard control-plane hosting: `controller` (external controller) or `store` (in-process in each engine's rank-0 worker). |
| `TPU_RAIDEN_ADVERTISE_HOST` | `""` | Routable host advertised for the in-process reshard services; required in store mode. |
| `TPU_RAIDEN_RESHARD_PORT_BASE` | `27000` | Store mode: engine `dp_rank` `i` binds its reshard service at this port + `i`. |
| `TPU_RAIDEN_STORE_DISPATCH_PORT_BASE` | `27100` | Store mode: engine `dp_rank` `i` binds its dispatch controller at this port + `i`. |
| `TPU_KV_TRANSFER_PORT` | `"9100"` | TCP port the KV wire transfer listens on. |
| `TPU_KV_RESHARD_TRANSPORT` | `"zmq"` | KV reshard transport: `zmq` (v1 connector protocol) or `raiden` (Raiden pool resharding). |

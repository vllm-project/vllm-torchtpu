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

## 4. Decode Capacity Preemption & Error Contract

When a Decode instance (`kv_role="kv_consumer"`) runs with `kv_load_failure_policy="fail"` (the upstream vLLM default), `vllm-torchtpu` aborts `RUNNING` requests that are evicted due to KV cache capacity pressure instead of returning them to `waiting` to recompute prompts locally on the decode pool. Administrative prefix cache resets (`reset_prefix_cache`) continue to requeue requests without aborting them.

### Proxy & Router Error Contract

Aborted victims emit a single terminal `EngineCoreOutput` once the preemption step's `finished_req_ids` drains:

* `finish_reason`: `"error"` (`FinishReason.ERROR`)
* `stop_reason`: `"kv_cache_preempted"`
* `kv_transfer_params`:

```json
{
  "kv_transfer_error": {
    "status_code": 500,
    "status_name": "model.KVCachePreempt",
    "status_message": "D-side request was preempted due to KV cache capacity."
  }
}
```

Preemption metrics (`vllm:num_preemptions`, `vllm:request_num_preemptions`, and `EngineCoreEventType.PREEMPTED`) are still recorded alongside `vllm:request_success{finished_reason="error"}` so capacity evictions remain observable in engine logs and dashboards.

### Configuration & Override

Capacity-preemption abort follows `kv_load_failure_policy` by default (`"fail"` enables abort; `"recompute"` keeps local recomputation). To override this behavior independently of `kv_load_failure_policy`, set `additional_config.abort_on_preemption` (`bool`) on the `kv_consumer` instance:

```bash
# Keep preempt-and-recompute on capacity eviction while keeping kv_load_failure_policy="fail"
--additional-config '{"abort_on_preemption": false}'

# Enable capacity-preemption abort even when kv_load_failure_policy="recompute"
--additional-config '{"abort_on_preemption": true}'
```

---

## 5. Reference Launch Scripts & GKE Manifests

| Script / Manifest | Description |
| :--- | :--- |
| `examples/disagg/launch_qwen35_disagg_multihost.sh` | Turnkey launcher for Qwen3.5 P/D disaggregation supporting `ROLE=controller\|prefill\|decode\|proxy\|all`. |
| `examples/disagg/launch_dsv4_disagg_multihost.sh` | Turnkey launcher for DeepSeek-V4 multi-host P/D disaggregation. |
| `examples/disagg/run_qwen35_raiden_pcp8_dp8_live_pair.sh` | Live pair script demonstrating heterogeneous resharding from Prefill (`PCP=8`) to Decode (`DP=8`). |
| `.buildkite/kubernetes/manifests/workloads/qwen3.5-397b-1p1d.yaml` | GKE JobSet manifest for `Qwen3.5-397B-A17B` 1P1D disaggregated serving. |
| `.buildkite/kubernetes/manifests/workloads/kimi-k3-1p1d-4dp8tp.yaml` | GKE JobSet manifest for `Kimi-K3` 1P1D disaggregated serving. |

---

## 6. Key Raiden Environment Variables

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

## 7. Raiden P/D Completion Signaling & Failure Handling

### Producer Send-Completion Signaling (`raiden_pd_send_completion`)

By default, a Prefill (`kv_producer`) instance finishes the HTTP request as soon as prefill token generation completes and the KV blocks are registered with the Raiden controller, while the actual P-to-D transfer proceeds asynchronously before P frees its KV blocks.

Routers and proxies that want the Prefill response to stay open until the KV transfer reaches a terminal state can opt in via `--additional-config '{"raiden_pd_send_completion": true}'` (supported on Stage-3 `TPU_KV_RESHARD_TRANSPORT=raiden` `kv_producer` instances with `TP=1, DP=1, PP=1` and a single `TPURaidenConnector`):

* **Initial prefill completion (`kv_transfer_pending`)**: When prefill finishes on P, `kv_transfer_params` includes `"kv_transfer_pending": true` alongside the transfer routing metadata (`uuid`, `req_id`, `num_tokens`, `src_controller_address`, etc.). In streaming mode (`stream=True`), this is emitted as an unfinished chunk so the router can immediately dispatch the Decode request in parallel.
* **Successful transfer completion (`kv_transfer_done`)**: Once all producer ranks report `finished_sending` with a successful transfer outcome, P closes the request with its original `finish_reason` (`length` or `stop`) and `kv_transfer_params` containing the transfer metadata plus `"kv_transfer_done": true`.
* **Producer transfer failure (`kv_transfer_error`)**: If any producer rank fails to send or the registration expires unclaimed before Decode pulls it (`TPU_KVCACHE_P2P_WAIT_PULL_TIMEOUT`), P terminates the request with `finish_reason="error"` and a structured `kv_transfer_error` payload:
  * `model.KVTPrefillWriteFailure`: P-side failed to write KV cache data (or reached the completion barrier without a sender outcome).
  * `model.KVTPrefillTransferTimeout`: P-side timed out waiting for KV cache transfer completion before the registration was claimed.

### Decode Load-Failure Policy (`kv_load_failure_policy`)

On the Decode (`kv_consumer`) side, Raiden reports asynchronous KV load failures (`TPU_RAIDEN_INLINE_LOAD=0`) per request through `KVConnectorTransferResults.failed_recving` (supporting both single-group and hybrid multi-group KV caches) and synchronous inline load failures (`TPU_RAIDEN_INLINE_LOAD=1`) through `invalid_block_ids`. Behavior is controlled by `kv_transfer_config.kv_load_failure_policy`:

* `"recompute"` (default): Releases the failed remote KV blocks and recomputes the prompt locally on the Decode instance.
* `"fail"`: Immediately terminates the affected Decode request with `finish_reason="error"` while retaining its allocated HBM receive buffers until the connector's `finished_recving` barrier completes.

# Disaggregated Serving (Prefill / Decode)

Prefill/Decode (P/D) disaggregation separates compute-bound **Prefill** (`kv_producer`) and memory-bandwidth-bound **Decode** (`kv_consumer`) onto independent TPU engine pools. This prevents long-prompt prefills from stalling ongoing decode steps (eliminating TPOT spikes) and allows each pool to use the parallelism layout (`TP`, `DP`, `PCP`, `EP`) best suited to its phase.

`vllm-torchtpu` transfers KV cache through the **Raiden KV Connector (`TPURaidenConnector`)**: a high-bandwidth PJRT DMA/RDMA engine powered by `tpu-sync-torch` (`tpu_sync`) with a central coordinator (`run_raiden_controller.py`) and support for **heterogeneous P/D resharding** (e.g., Prefill with `PCP=4, TP=2` transferring KV cache into Decode with `DP=8, TP=1`). `TPUConnector` is a compatibility alias of `TPURaidenConnector` and uses the same Raiden Stage-3 backend; the legacy ZMQ transport has been removed.

---

## 1. Architecture Overview (Raiden P/D Disaggregation)

A production Raiden P/D deployment consists of four components:

1. **Raiden Controller (`examples/disagg/run_raiden_controller.py`)**: Coordinates KV transfer metadata and pool registration across Prefill and Decode workers.
2. **Prefill Engine Pool (`kv_role="kv_producer"`)**: Computes prompt prefill, writes KV blocks into the unified/Raiden block pool, and streams KV blocks to the Decode pool.
3. **Decode Engine Pool (`kv_role="kv_consumer"`)**: Receives KV blocks into its local HBM KV cache (resharding on the fly if P and D meshes differ) and generates output tokens.
4. **Disaggregated Proxy Router (`examples/disagg/toy_proxy_server.py`)**: Accepts client OpenAI HTTP requests, routes a `max_tokens=1` prefill request to a Prefill instance, and forwards the request with KV transfer parameters to a Decode instance.

---

## 2. Multi-Host Disaggregated Serving with Raiden (`TPURaidenConnector`)

For large models such as `Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3.5-35B-A3B-FP8`, `moonshotai/Kimi-K3`, and `deepseek-ai/DeepSeek-V4`, use `TPURaidenConnector` with `TPU_USE_RAIDEN_KV_CACHE_MANAGER=1`.

> [!IMPORTANT]
> `TPURaidenConnector` (and its `TPUConnector` alias) only supports models with a
> Raiden Stage-3 admission path. Every Prefill and Decode instance must set
> `TPU_USE_RAIDEN_KV_CACHE_MANAGER=1` and exactly one of the model admission flags:
>
> | Model family | Admission flag |
> | :--- | :--- |
> | Qwen3.5 | `TPU_RAIDEN_QWEN35_ADMISSION=1` |
> | Kimi-K3 | `TPU_RAIDEN_KIMIK3_ADMISSION=1` |
> | GLM | `TPU_RAIDEN_GLM_ADMISSION=1` |
> | DeepSeek-V4 | `TPU_RAIDEN_DSV4_ADMISSION=1` |
>
> `TpuPlatform.check_and_update_config` rejects the configuration before model
> load if the cache manager is disabled, no admission flag is set, or more than one
> admission flag is set. Plain attention models without a Stage-3 admission path
> (for example `Qwen/Qwen3-0.6B`) and 1-chip-per-role Qwen3.5 pairs are not
> supported. `kv_connector_extra_config["use_raiden_connector"]` has been removed
> and is rejected at startup.

### Single-Host Disaggregated Serving

To run Prefill and Decode on a single host, use
`examples/disagg/launch_qwen35_p4d2_v2_baseline.sh`. It launches a Qwen3.5
Prefill/Decode pair on one host with `TPU_USE_RAIDEN_KV_CACHE_MANAGER=1` and
`TPU_RAIDEN_QWEN35_ADMISSION=1`, combining `TPURaidenConnector` with
`TPURaidenOffloadingConnector` through `TPUMultiConnector`. The multi-host steps
below show the same configuration split across hosts.

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
    "kv_connector": "TPURaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
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
    "kv_connector": "TPURaidenConnector",
    "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
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

## 3. Decode Capacity Preemption & Error Contract

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

## 4. Reference Launch Scripts & GKE Manifests

| Script / Manifest | Description |
| :--- | :--- |
| `examples/disagg/launch_qwen35_p4d2_v2_baseline.sh` | Single-host Qwen3.5 P/D launcher (Prefill and Decode on one host, `TPURaidenConnector` + offloading via `TPUMultiConnector`). |
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
| `TPU_RAIDEN_TRANSFER_PARALLELISM` | `8` | Number of source ranks that transfer KV blocks in parallel (source rank transfer parallelism). Distinct from `TPU_RAIDEN_TRANSFER_STREAMS`, which sets the TCP stream count of each push. |
| `TPU_RAIDEN_RESHARD_IMPL` | `"controller"` | KV resharding implementation (`controller` or `host`) when Prefill and Decode shardings differ. |
| `TPU_KV_RESHARD_TRANSPORT` | `"raiden"` | Transport selector for KV cache resharding. Defaults to and only accepts `"raiden"` (case-insensitive); kept for launch-config compatibility. Any other value (for example the removed `"zmq"`) is rejected at startup. |

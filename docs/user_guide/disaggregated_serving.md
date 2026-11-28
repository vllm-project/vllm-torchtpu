# Disaggregated Serving (Prefill / Decode)

Prefill/Decode (P/D) disaggregation separates compute-bound **Prefill** (`kv_producer`) and memory-bandwidth-bound **Decode** (`kv_consumer`) onto independent TPU engine pools. This prevents long-prompt prefills from stalling ongoing decode steps (eliminating TPOT spikes) and allows each pool to use the parallelism layout (`TP`, `DP`, `PCP`, `EP`) best suited to its phase.

`vllm-torchtpu` supports two KV cache transfer connectors:

1. **Raiden KV Connector (`TPURaidenConnector`)**: High-bandwidth PJRT DMA/RDMA engine powered by `tpu-sync-torch` (`tpu_sync`) with a central coordinator (`run_raiden_controller.py`) and support for **heterogeneous P/D resharding** (e.g., Prefill with `PCP=4, TP=2` transferring KV cache into Decode with `DP=8, TP=1`).
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

## 3. Multi-Host Disaggregated Serving with Raiden (`TPURaidenConnector`)

For large models such as `Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3.5-35B-A3B-FP8`, `moonshotai/Kimi-K3`, and `deepseek-ai/DeepSeek-V4`, use `TPURaidenConnector` with `TPU_USE_RAIDEN_KV_CACHE_MANAGER=1`.

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

With `--attention-backend CUSTOM` and the seq-along-lane KV cache layout
(`VLLM_KV_CACHE_LAYOUT=HND`), the Raiden layout fingerprint includes the
attention kernel page, and vLLM picks the largest supported page that divides
each engine's block size: 128 or 256 tokens, or up to 4096 tokens on an engine
with prefill context parallelism. The setting below applies only to an engine
with a KV transfer config, this backend and this layout; any other engine logs a
warning and keeps vLLM's page. When the prefill and decode block sizes differ
(for example 2304 and 640 tokens), the two engines page differently and every
transfer is refused with "Layout fingerprint mismatch". Give both engines the
same page:
`--additional-config '{"tpu_kv_transfer": {"kernel_page_tokens": 128}}'`.
Prefill (`kv_producer`) and decode (`kv_consumer`) engines of the
`Qwen/Qwen3.5-397B-A17B` checkpoint shape (its BF16 and FP8 releases) with the
seq-along-lane layout and no prefill context parallelism default to 128.

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

## 7. Raiden Connector Options

These opt-in options go in a connector's own `kv_connector_extra_config` in `--kv-transfer-config` (for a `TPUMultiConnector`, in the child's entry of `connectors`). Unset, each connector behaves as it does without the option. `vllm_torchtpu/distributed/kv_transfer/connector_options.py` reads them: a value must be a JSON boolean or a non-negative JSON integer as listed, and an option set on another connector than the one below is refused at start-up. Other keys of `kv_connector_extra_config` (such as `cpu_bytes_to_use`) belong to their own readers.

| Option | Connector | Default | Role | Effect | Limits |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `async_complete` | `TPURaidenConnector` | `false` | KV producer | Issues the post-transfer `complete_request_blocks` RPC on a background thread instead of inside the engine step. | Refused at start-up on a `kv_consumer` engine. A failed call is retried on that thread, up to 5 attempts 0.1 s apart; the next poll raises a call that failed every attempt. |
| `async_register` | `TPURaidenConnector` | `false` | KV producer | Registers a step's request blocks on a side thread that starts once the forward is dispatched (or after 10 ms). | Refused at start-up on a `kv_consumer` engine. Only the default admission path; DSv4, GLM and Kimi-K3 admission register synchronously. |
| `late_poll` | `TPURaidenConnector` | `false` | KV consumer | Reports Stage-3 loads that land during a step with that step's output instead of the next one's. | Refused at start-up on a `kv_producer` or `kv_both` engine, without `--async-scheduling`, and with `TPU_RAIDEN_INLINE_LOAD`. Under DP, a zero-token step's output waits for the next DP dummy batch, at most 50 ms. |
| `host_tier_recency` | `TPURaidenOffloadingConnector` | `false` | Any | Refreshes a request's host-tier keys on every lookup and store (FullAttention tail to chunk 0, then the recurrent and sliding-window state, then partial tails), skips partial tails at or below the local HBM hit, offers FullAttention chunks for storing from chunk 0 again, and launches fence-acknowledged stores largest first. | Scheduler side only; lookups and stores are upstream's either way. |
| `max_async_loads` | `TPURaidenOffloadingConnector` | `0` (no cap) | Any | Defers a request with a host-tier hit to a later step while this many host-tier loads are in flight. | Non-negative integer. |
| `producer_divergent_hits` | `TPUMultiConnector` | `false` | KV producer | Reports divergent local hybrid hits (FullAttention deeper than the recurrent state) when any child can complete them, instead of upstream's every-child rule, which is false on a producer. | Refused at start-up on a `kv_consumer` engine. The offloading child completes them only with `TPU_RAIDEN_PREFIX_AWARE_LOAD=1`; the pipeline chunk scheduler with `tpu_pp_scheduler.prefix_aware` asks that child how much of the hit to keep. |

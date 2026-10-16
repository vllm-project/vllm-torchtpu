# Disaggregated serving (Prefill/Decode with Raiden3)

> [!NOTE]
> Raiden3 (`TPURaidenConnector`) Prefill/Decode (PD) disaggregation profiling is
> currently experimental and under active development. The configuration and
> workflow below have been verified against commit
> [`7bb6a4a`](https://github.com/vllm-project/vllm-torchtpu/commit/7bb6a4ae2a8379f2e6cf9f6d5922ac5ef8ca8c15).

In a disaggregated Prefill/Decode (PD) deployment, prompt ingestion and
autoregressive token generation run on **two separate sets of TPU pods**
interconnected via the native C++ **Raiden3** (`tpu_sync`) KV cache transfer
engine:

* **Prefill cluster (`kv_producer`)** — computes the prompt's KV cache (and any
  hybrid Mamba/GDN state blocks) and registers the resulting block handles with
  its local `RaidenControllerServer` (`register_request_blocks`), returning the
  transfer metadata (`uuid`, source controller address, and block IDs) in
  `kv_transfer_params`.
* **Decode cluster (`kv_consumer`)** — receives `kv_transfer_params`, coordinates
  an asynchronous cross-slice pull/reshard transfer from the Prefill store using
  the shared `uuid`, scatters the KV blocks into its local HBM, and generates
  the completion tokens.

Because Prefill and Decode run in separate pods with independent TPU workers,
profiling a disaggregated request requires arming `/start_profile` on **both**
services before sending traffic, then disarming `/stop_profile` on both and
extracting traces from each pod's persistent volume.

## KV transfer `TraceAnnotation` markers

[`TPURaidenConnectorWorker`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/distributed/kv_transfer/tpu_connector.py)
emits `TraceAnnotation` scopes carrying the shared Raiden transfer `uuid` and
`request_id` so you can correlate a single disaggregated request across both the
Prefill and Decode XPlane timelines:

| Cluster Role | TraceAnnotation Name | Annotated Metadata | Description |
|---|---|---|---|
| Prefill (`kv_producer`) | `KV_Cache_Prefill_Register_Blocks` | `uuid`, `request_id`, `num_tokens`, `num_blocks` | Registers computed KV cache blocks and state spans in the local Raiden store. |
| Prefill (`kv_producer`) | `KV_Cache_Prefill_Send_Complete` | `uuid`, `request_id`, `num_tokens` | Marks completion and release of producer KV blocks after consumer pull completes. |
| Decode (`kv_consumer`) | `KV_Cache_Decode_Submit_Load` | `uuid`, `request_id`, `source_req_id`, `num_tokens`, `num_blocks` | Submits the cross-slice reshard/pull request to fetch KV blocks from the producer. |
| Decode (`kv_consumer`) | `KV_Cache_Decode_Recv_Complete` | `uuid`, `request_id`, `source_req_id`, `num_tokens`, `reshard_e2e_latency_ms` | Records completion of the KV transfer and logs end-to-end reshard latency in ms. |

## Step 1: Configure and deploy Prefill and Decode pods on GKE

Both the Prefill and Decode GKE deployments must enable the Raiden3 store-based
KV cache manager in their `env` blocks and configure `--profiler-config` to
write traces to a mounted Persistent Volume Claim (such as
`/usr/vllm/vllm_profile`) so captured traces persist across container restarts:

```yaml
env:
  - name: TPU_USE_RAIDEN_KV_CACHE_MANAGER
    value: "1"
  - name: TPU_KV_RESHARD_TRANSPORT
    value: "raiden"
  - name: TPU_RAIDEN_RESHARD_IMPL
    value: "store"
  - name: USE_PHASED_PROFILER
    value: "true"  # set to "false" or omit for continuous non-phased profiling
```

Pass `--profiler-config` and `--kv-transfer-config` to `vllm serve` (with
`"kv_role": "kv_producer"` on the Prefill deployment and
`"kv_role": "kv_consumer"` on the Decode deployment):

```yaml
args:
  - --profiler-config='{"profiler": "torch", "torch_profiler_dir": "/usr/vllm/vllm_profile", "ignore_frontend": true}'
  - --kv-transfer-config='{"kv_connector": "TPURaidenConnector", "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector", "kv_role": "kv_producer", "kv_port": 14579}'
```

Complete reference GKE `Deployment` and `Service` manifests for both clusters
are available in `docs/developers_guide/examples/`:

* **Prefill Pod Manifest (`TPU v6e 2x2, PCP=4, TP=1`)**:
  [`docs/developers_guide/examples/raiden_pd_prefill.yaml`](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/developers_guide/examples/raiden_pd_prefill.yaml)
* **Decode Pod Manifest (`TPU v6e 2x2, DP=4, TP=1`)**:
  [`docs/developers_guide/examples/raiden_pd_decode.yaml`](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs/developers_guide/examples/raiden_pd_decode.yaml)

Deploy both manifests into your target namespace (e.g., `disagg-1-1`):

```bash
kubectl apply -f docs/developers_guide/examples/raiden_pd_prefill.yaml
kubectl apply -f docs/developers_guide/examples/raiden_pd_decode.yaml
```

## Step 2: Coordinate `/start_profile`, KV transfer, and `/stop_profile`

Port-forward the Prefill service to local port `8000` and the Decode service to
local port `8001`:

```bash
kubectl -n disagg-1-1 port-forward svc/vllm-service-prefill 8000:8000 &
kubectl -n disagg-1-1 port-forward svc/vllm-service-decode 8001:8000 &
```

Arm `/start_profile` simultaneously on **both** clusters before triggering the
Prefill-to-Decode `uuid` transfer, then call `/stop_profile` on both clusters
once generation finishes:

```python
import json
import time
import urllib.request

PREFILL_HOST = "http://127.0.0.1:8000"
DECODE_HOST = "http://127.0.0.1:8001"
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
PROMPT = "San Francisco is a major cultural and commercial center in " * 64


def post_json(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else b""
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


# 1. Start profiling on both Prefill and Decode clusters
post_json(f"{PREFILL_HOST}/start_profile")
post_json(f"{DECODE_HOST}/start_profile")
time.sleep(1)

# 2. Run Prefill (kv_producer) and extract Raiden kv_transfer_params (UUID)
print("1/2 Executing Prefill on producer cluster...")
prefill_resp = post_json(
    f"{PREFILL_HOST}/v1/completions",
    {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 1,
        "temperature": 0,
        "kv_transfer_params": {
            "do_remote_decode": True,
            "do_remote_prefill": False,
        },
    },
)
kv_transfer_params = prefill_resp.get("kv_transfer_params", {})
print("Extracted kv_transfer_params:", json.dumps(kv_transfer_params, indent=2))

# 3. Run Decode (kv_consumer) using the returned kv_transfer_params
print("2/2 Executing Decode on consumer cluster...")
decode_resp = post_json(
    f"{DECODE_HOST}/v1/completions",
    {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 32,
        "temperature": 0,
        "kv_transfer_params": kv_transfer_params,
    },
)

# 4. Stop profiling and flush XPlane traces on both clusters
post_json(f"{PREFILL_HOST}/stop_profile")
post_json(f"{DECODE_HOST}/stop_profile")
print("Profiling completed. XPlane traces saved on both Prefill and Decode pods!")
```

## Step 3: Extract and inspect the Prefill and Decode traces

With `USE_PHASED_PROFILER=true`, the Prefill pod writes its captures under
`/usr/vllm/vllm_profile/prefill_only/` and the Decode pod writes its captures
under `/usr/vllm/vllm_profile/decode_only/`. Copy both directories from their
respective pods to your local machine:

```bash
export NAMESPACE="disagg-1-1"
export PREFILL_POD=$(kubectl -n "$NAMESPACE" get pods -l app=vllm-prefill -o jsonpath="{.items[0].metadata.name}")
export DECODE_POD=$(kubectl -n "$NAMESPACE" get pods -l app=vllm-decode -o jsonpath="{.items[0].metadata.name}")

mkdir -p ./phased_traces
kubectl cp "${NAMESPACE}/${PREFILL_POD}:/usr/vllm/vllm_profile/prefill_only" ./phased_traces/prefill_only -c vllm-tpu
kubectl cp "${NAMESPACE}/${DECODE_POD}:/usr/vllm/vllm_profile/decode_only" ./phased_traces/decode_only -c vllm-decode
```

Launch XProf pointing at `./phased_traces` to inspect both `prefill_only` and
`decode_only` runs, and search the Trace Viewer for the shared `uuid` on the
`KV_Cache_Prefill_Register_Blocks`, `KV_Cache_Decode_Submit_Load`, and
`KV_Cache_Decode_Recv_Complete` events:

```bash
xprof --logdir ./phased_traces
```

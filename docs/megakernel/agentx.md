# Kimi-K3 P/D with Docker for AgentX Reproduction

These commands reproduce the nine TPU v7x points of the Kimi-K3 AgentX results on [InferenceX](https://inferencex.semianalysis.com/inference/kimi-k3). Each point serves Kimi-K3 on the megakernel backend as one prefill instance and one decode instance (1P1D), each on its own TPU v7x 2x2x4 slice (4 hosts, 16 chips, 32 devices), and runs the one-hour AgentX benchmark at the point's concurrency. Points 1 to 6 decode with DSpark speculative decoding and the MLA draft `Inferact/Kimi-K3-DSpark`, with synthetic acceptance at fixed mean acceptance lengths, so throughput does not depend on the prompts. Points 7 to 9 decode without speculation. Every command runs from the image `vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c`, which has vLLM, this plugin, the Rust frontend `vllm-rs`, `vllm-router` and the tpu-sync binaries.

Assume every host has the same `/models` (shared storage, or a copy on each host): the `moonshotai/Kimi-K3` checkpoint in `/models/Kimi-K3` and the `Inferact/Kimi-K3-DSpark` draft in `/models/Kimi-K3-DSpark`. Each prefill host needs 384 GiB (points 1 to 4) or 512 GiB (points 5 to 9) of free DRAM for the host KV store. Each decode host needs 384 GiB of free DRAM for its own host KV store (points 1 and 2) or 512 GiB for its store server (points 3 to 9).

Set these on every host:

```bash
PREFILL_IP=10.0.0.10   # host 0 of the prefill slice
DECODE_IP=10.0.1.10    # host 0 of the decode slice
NODE_RANK=0            # 0, 1, 2, 3 within the host's slice
```

## Points

| Point | AgentX concurrency | Decode | Decode `--max-num-seqs` | Drafts by decode batch size | Synthetic acceptance length | Commands |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | 1 | DCP32 | 1 | 7 | 3.78 | section 2 |
| 2 | 4 | DCP32 | 2 | 7 | 3.78 | section 2 |
| 3 | 8 | DCP32 | 8 | 7 at 1, 3 at 2-8 | 3.78 (7 drafts), 2.87 (3 drafts) | section 3 |
| 4 | 16 | DCP32 | 8 | 7 at 1, 3 at 2-8 | 3.78 (7 drafts), 2.87 (3 drafts) | section 3 |
| 5 | 32 | DCP32 | 16 | 7 at 1, 3 at 2-16 | 3.78 (7 drafts), 2.87 (3 drafts) | section 3 |
| 6 | 48 | attention DP2 x DCP16 | 32 | 3 | 2.87 | section 4 |
| 7 | 48 | attention DP2 x DCP16 | 128 | no speculation | - | section 5 |
| 8 | 72 | attention DP2 x DCP16 | 128 | no speculation | - | section 5 |
| 9 | 48 | DCP32 | 64 | no speculation | - | section 5 |

Points 3 to 9 add a store server on each decode host. It holds the KV blocks that the prefill evicts from its own host store.

## 1. Pre-shard the weights

On one TPU v7x host, before the servers start. The conversion uses only the CPU and writes the TP32 pre-shard into `/models/kimi-k3-tp32`.

```bash
docker run --rm --privileged -v /models:/models -e MEGAKERNEL_TOOL_WORKER=1 vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c bash -c '
  python -m tools.megakernel preshard k3 --checkpoint /models/Kimi-K3 --output /models/kimi-k3-tp32 --ranks $(seq -s, 0 31) &&
  python -m tools.megakernel preshard prefill --checkpoint /models/Kimi-K3 --output /models/kimi-k3-tp32 &&
  python -m tools.megakernel preshard dspark --draft /models/Kimi-K3-DSpark --output /models/kimi-k3-tp32'
```

## 2. Points 1 and 2: DCP32 decode

Both roles keep their own host KV store. Set `MNS=1` for point 1 or `MNS=2` for point 2 on the decode hosts.

Prefill, on each of the 4 prefill hosts:

```bash
docker run --rm --privileged --net=host --ipc=host -v /models:/models \
  -e TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 -e TPU_HOST_BOUNDS=1,1,4 -e TPU_NODE_ID="$NODE_RANK" -e TPU_SKIP_MDS_QUERY=1 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e TCMALLOC_AGGRESSIVE_DECOMMIT=true -e NPROC=2 \
  -e LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7" \
  -e VLLM_USE_RUST_FRONTEND=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800 -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
  -e TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 -e TPU_RAIDEN_KIMIK3_ADMISSION=1 -e TPU_KV_RESHARD_TRANSPORT=raiden -e TPU_RAIDEN_RESHARD_IMPL=store \
  -e TPU_RAIDEN_ADVERTISE_HOST="$PREFILL_IP" -e TPU_RAIDEN_PREFIX_AWARE_LOAD=1 -e TPU_RAIDEN_TRANSFER_PARALLELISM=32 -e TPU_RAIDEN_DST_SHARDS=32 \
  -e TPU_RAIDEN_TRANSFER_NUM_SLOTS=1 -e TPU_RAIDEN_POOL_STAGING_LEASES=16 -e TPU_P2P_WAIT_PULL_TIMEOUT=900 \
  -e RAIDEN_NUMA_POLICY=bind -e RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=1800 -e TPU_K3_MOE_REDUCE_DTYPE=bfloat16 \
  vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c \
  vllm serve /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --language-model-only \
  --tensor-parallel-size 32 --enable-expert-parallel --decode-context-parallel-size 32 \
  --nnodes 4 --node-rank "$NODE_RANK" --master-addr "$PREFILL_IP" --distributed-timeout-seconds 1800 $([ "$NODE_RANK" = 0 ] || echo --headless) \
  --max-model-len 1048576 --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512 \
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details --load-format megakernel --kv-cache-memory-bytes 19327352832 \
  --model-loader-extra-config '{"presharded":"/models/kimi-k3-tp32","draft":"/models/Kimi-K3-DSpark"}' \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --kv-transfer-config '{"kv_connector":"TPUMultiConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer"},{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":1649267441664,"store_server_ip":"'"$PREFILL_IP"'"}}]}}'
```

Decode, on each of the 4 decode hosts:

```bash
docker run --rm --privileged --net=host --ipc=host -v /models:/models \
  -e TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 -e TPU_HOST_BOUNDS=1,1,4 -e TPU_NODE_ID="$NODE_RANK" -e TPU_SKIP_MDS_QUERY=1 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e TCMALLOC_AGGRESSIVE_DECOMMIT=true -e NPROC=2 \
  -e LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7" \
  -e VLLM_USE_RUST_FRONTEND=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800 -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
  -e TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 -e TPU_RAIDEN_KIMIK3_ADMISSION=1 -e TPU_KV_RESHARD_TRANSPORT=raiden -e TPU_RAIDEN_RESHARD_IMPL=store \
  -e TPU_RAIDEN_ADVERTISE_HOST="$DECODE_IP" -e TPU_RAIDEN_PREFIX_AWARE_LOAD=1 -e TPU_RAIDEN_TRANSFER_PARALLELISM=32 -e TPU_RAIDEN_DST_SHARDS=32 \
  -e TPU_RAIDEN_TRANSFER_NUM_SLOTS=1 -e TPU_RAIDEN_POOL_STAGING_LEASES=16 -e TPU_P2P_WAIT_PULL_TIMEOUT=900 \
  -e RAIDEN_NUMA_POLICY=bind -e RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=1800 -e TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT=0 \
  vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c \
  vllm serve /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --language-model-only \
  --tensor-parallel-size 32 --enable-expert-parallel --decode-context-parallel-size 32 \
  --nnodes 4 --node-rank "$NODE_RANK" --master-addr "$DECODE_IP" --distributed-timeout-seconds 1800 $([ "$NODE_RANK" = 0 ] || echo --headless) \
  --max-model-len 1048576 --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512 \
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details --load-format megakernel --kv-cache-memory-bytes 19327352832 \
  --model-loader-extra-config '{"presharded":"/models/kimi-k3-tp32"}' \
  --max-num-seqs "$MNS" --max-num-batched-tokens 512 \
  --speculative-config '{"method":"dspark","model":"/models/Kimi-K3-DSpark","num_speculative_tokens":7,"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.78}' \
  --kv-transfer-config '{"kv_connector":"TPUMultiConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer"},{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":1649267441664,"store_server_ip":"'"$DECODE_IP"'"}}]}}'
```

## 3. Points 3 to 5: DCP32 decode with store servers

Set the point's values on every host:

```bash
# Point 3 (concurrency 8) and point 4 (concurrency 16):
PREFILL_STORE=1649267441664 PREFILL_KV=19327352832 PREFILL_FLAGS=
DECODE_KV=19327352832 MNS=8 SCHEDULE='[[1,1,7],[2,8,3]]' DEFERRED_SUBMIT=0
# Point 5 (concurrency 32):
PREFILL_STORE=2199023255552 PREFILL_KV=23622320128 PREFILL_FLAGS='--long-prefill-token-threshold 4096 --numa-bind --numa-bind-nodes 0 0 0 0 1 1 1 1'
DECODE_KV=30064771072 MNS=16 SCHEDULE='[[1,1,7],[2,16,3]]' DEFERRED_SUBMIT=1
```

On prefill host 0, start the global registry first, and keep it running:

```bash
docker run --rm --net=host vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c global_registry_server --port=27900
```

Prefill, on each of the 4 prefill hosts:

```bash
docker run --rm --privileged --net=host --ipc=host -v /models:/models \
  -e TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 -e TPU_HOST_BOUNDS=1,1,4 -e TPU_NODE_ID="$NODE_RANK" -e TPU_SKIP_MDS_QUERY=1 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e TCMALLOC_AGGRESSIVE_DECOMMIT=true -e NPROC=2 \
  -e LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7" \
  -e VLLM_USE_RUST_FRONTEND=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800 -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
  -e TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 -e TPU_RAIDEN_KIMIK3_ADMISSION=1 -e TPU_KV_RESHARD_TRANSPORT=raiden -e TPU_RAIDEN_RESHARD_IMPL=store \
  -e TPU_RAIDEN_ADVERTISE_HOST="$PREFILL_IP" -e TPU_RAIDEN_PREFIX_AWARE_LOAD=1 -e TPU_RAIDEN_TRANSFER_PARALLELISM=32 -e TPU_RAIDEN_DST_SHARDS=32 \
  -e TPU_RAIDEN_TRANSFER_NUM_SLOTS=1 -e TPU_RAIDEN_POOL_STAGING_LEASES=16 -e TPU_P2P_WAIT_PULL_TIMEOUT=900 \
  -e RAIDEN_NUMA_POLICY=bind -e RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=1800 -e TPU_K3_MOE_REDUCE_DTYPE=bfloat16 \
  -e PYTHONHASHSEED=0 -e RAIDEN_ENABLE_STORE_MONITOR=1 -e RAIDEN_ENABLE_EVICT_SWEEP=1 \
  vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c \
  vllm serve /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --language-model-only \
  --tensor-parallel-size 32 --enable-expert-parallel --decode-context-parallel-size 32 \
  --nnodes 4 --node-rank "$NODE_RANK" --master-addr "$PREFILL_IP" --distributed-timeout-seconds 1800 $([ "$NODE_RANK" = 0 ] || echo --headless) \
  --max-model-len 1048576 --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512 \
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details --load-format megakernel --kv-cache-memory-bytes "$PREFILL_KV" \
  --model-loader-extra-config '{"presharded":"/models/kimi-k3-tp32","draft":"/models/Kimi-K3-DSpark"}' \
  --max-num-seqs 4 --max-num-batched-tokens 8192 $PREFILL_FLAGS \
  --kv-transfer-config '{"kv_connector":"TPUMultiConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer"},{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"$PREFILL_STORE"',"store_server_ip":"'"$PREFILL_IP"'","global_registry_address":"'"$PREFILL_IP"':27900","kv_pool_group":"k3-host-store"}}]}}'
```

Decode, on each of the 4 decode hosts. The decode keeps no host store of its own:

```bash
docker run --rm --privileged --net=host --ipc=host -v /models:/models \
  -e TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 -e TPU_HOST_BOUNDS=1,1,4 -e TPU_NODE_ID="$NODE_RANK" -e TPU_SKIP_MDS_QUERY=1 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e TCMALLOC_AGGRESSIVE_DECOMMIT=true -e NPROC=2 \
  -e LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7" \
  -e VLLM_USE_RUST_FRONTEND=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800 -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
  -e TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 -e TPU_RAIDEN_KIMIK3_ADMISSION=1 -e TPU_KV_RESHARD_TRANSPORT=raiden -e TPU_RAIDEN_RESHARD_IMPL=store \
  -e TPU_RAIDEN_ADVERTISE_HOST="$DECODE_IP" -e TPU_RAIDEN_PREFIX_AWARE_LOAD=1 -e TPU_RAIDEN_TRANSFER_PARALLELISM=32 -e TPU_RAIDEN_DST_SHARDS=32 \
  -e TPU_RAIDEN_TRANSFER_NUM_SLOTS=1 -e TPU_RAIDEN_POOL_STAGING_LEASES=16 -e TPU_P2P_WAIT_PULL_TIMEOUT=900 \
  -e RAIDEN_NUMA_POLICY=bind -e RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=1800 -e TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT="$DEFERRED_SUBMIT" \
  vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c \
  vllm serve /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --language-model-only \
  --tensor-parallel-size 32 --enable-expert-parallel --decode-context-parallel-size 32 \
  --nnodes 4 --node-rank "$NODE_RANK" --master-addr "$DECODE_IP" --distributed-timeout-seconds 1800 $([ "$NODE_RANK" = 0 ] || echo --headless) \
  --max-model-len 1048576 --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512 \
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details --load-format megakernel --kv-cache-memory-bytes "$DECODE_KV" \
  --model-loader-extra-config '{"presharded":"/models/kimi-k3-tp32","synthetic_acceptance_lengths":{"3":2.87,"7":3.78}}' \
  --max-num-seqs "$MNS" --max-num-batched-tokens 512 \
  --speculative-config '{"method":"dspark","model":"/models/Kimi-K3-DSpark","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":'"$SCHEDULE"',"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.78}' \
  --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer"}'
```

When the decode serves (`http://$DECODE_IP:8000/health` returns 200), start a store server on each of the 4 decode hosts, with `NODE_IP` that host's IP:

```bash
docker run --rm --privileged --net=host -e RAIDEN_NUMA_POLICY=bind vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c \
  kv_cache_host_store_node_main --dram_budget_bytes=549755813888 \
  --global_registry_address="$PREFILL_IP:27900" --kv_pool_group=k3-host-store \
  --store_server_ip="$NODE_IP" --job_name=k3-host-store-node --job_replica_id="$NODE_RANK" --parallelism=16
```

## 4. Point 6: attention DP2 x DCP16 decode

The global registry and the store servers are the same as in section 3. The prefill is the point-5 prefill with the transfer resharded to the decode's DCP16 page layout.

Prefill, on each of the 4 prefill hosts:

```bash
docker run --rm --privileged --net=host --ipc=host -v /models:/models \
  -e TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 -e TPU_HOST_BOUNDS=1,1,4 -e TPU_NODE_ID="$NODE_RANK" -e TPU_SKIP_MDS_QUERY=1 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e TCMALLOC_AGGRESSIVE_DECOMMIT=true -e NPROC=2 \
  -e LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7" \
  -e VLLM_USE_RUST_FRONTEND=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800 -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
  -e TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 -e TPU_RAIDEN_KIMIK3_ADMISSION=1 -e TPU_KV_RESHARD_TRANSPORT=raiden -e TPU_RAIDEN_RESHARD_IMPL=store \
  -e TPU_RAIDEN_ADVERTISE_HOST="$PREFILL_IP" -e TPU_RAIDEN_PREFIX_AWARE_LOAD=1 -e TPU_RAIDEN_TRANSFER_PARALLELISM=32 -e TPU_RAIDEN_DST_SHARDS=32 \
  -e TPU_RAIDEN_TRANSFER_NUM_SLOTS=1 -e TPU_RAIDEN_POOL_STAGING_LEASES=16 -e TPU_P2P_WAIT_PULL_TIMEOUT=900 \
  -e RAIDEN_NUMA_POLICY=bind -e RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=1800 -e TPU_K3_MOE_REDUCE_DTYPE=bfloat16 \
  -e PYTHONHASHSEED=0 -e RAIDEN_ENABLE_STORE_MONITOR=1 -e RAIDEN_ENABLE_EVICT_SWEEP=1 \
  vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c \
  vllm serve /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --language-model-only \
  --tensor-parallel-size 32 --enable-expert-parallel --decode-context-parallel-size 32 \
  --nnodes 4 --node-rank "$NODE_RANK" --master-addr "$PREFILL_IP" --distributed-timeout-seconds 1800 $([ "$NODE_RANK" = 0 ] || echo --headless) \
  --max-model-len 1048576 --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512 \
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details --load-format megakernel --kv-cache-memory-bytes 23622320128 \
  --model-loader-extra-config '{"presharded":"/models/kimi-k3-tp32","draft":"/models/Kimi-K3-DSpark"}' \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 4096 --numa-bind --numa-bind-nodes 0 0 0 0 1 1 1 1 \
  --kv-transfer-config '{"kv_connector":"TPUMultiConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer","kv_connector_extra_config":{"destination_decode_context_parallel_size":16}},{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":2199023255552,"store_server_ip":"'"$PREFILL_IP"'","global_registry_address":"'"$PREFILL_IP"':27900","kv_pool_group":"k3-host-store"}}]}}'
```

Decode, on each of the 4 decode hosts:

```bash
docker run --rm --privileged --net=host --ipc=host -v /models:/models \
  -e TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 -e TPU_HOST_BOUNDS=1,1,4 -e TPU_NODE_ID="$NODE_RANK" -e TPU_SKIP_MDS_QUERY=1 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e TCMALLOC_AGGRESSIVE_DECOMMIT=true -e NPROC=2 \
  -e LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7" \
  -e VLLM_USE_RUST_FRONTEND=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800 -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
  -e TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 -e TPU_RAIDEN_KIMIK3_ADMISSION=1 -e TPU_KV_RESHARD_TRANSPORT=raiden -e TPU_RAIDEN_RESHARD_IMPL=store \
  -e TPU_RAIDEN_ADVERTISE_HOST="$DECODE_IP" -e TPU_RAIDEN_PREFIX_AWARE_LOAD=1 -e TPU_RAIDEN_TRANSFER_PARALLELISM=32 -e TPU_RAIDEN_DST_SHARDS=32 \
  -e TPU_RAIDEN_TRANSFER_NUM_SLOTS=1 -e TPU_RAIDEN_POOL_STAGING_LEASES=16 -e TPU_P2P_WAIT_PULL_TIMEOUT=900 \
  -e RAIDEN_NUMA_POLICY=bind -e RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=1800 \
  vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c \
  vllm serve /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --language-model-only \
  --tensor-parallel-size 32 --enable-expert-parallel --decode-context-parallel-size 16 \
  --nnodes 4 --node-rank "$NODE_RANK" --master-addr "$DECODE_IP" --distributed-timeout-seconds 1800 $([ "$NODE_RANK" = 0 ] || echo --headless) \
  --max-model-len 1048576 --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512 \
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details --load-format megakernel --kv-cache-memory-bytes 30064771072 \
  --model-loader-extra-config '{"presharded":"/models/kimi-k3-tp32","attention_dp":2,"synthetic_acceptance_lengths":{"3":2.87}}' \
  --max-num-seqs 32 --max-num-batched-tokens 512 \
  --speculative-config '{"method":"dspark","model":"/models/Kimi-K3-DSpark","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,32,3]],"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.87}' \
  --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer"}'
```

## 5. Points 7 to 9: decode without speculation

The global registry and the store servers are the same as in section 3. The prefill is the point-5 prefill without the draft. For the attention DP2 x DCP16 decode, the transfer reshards to the decode's DCP16 page layout.

Set the point's values on every host:

```bash
# Point 7 (concurrency 48) and point 8 (concurrency 72), attention DP2 x DCP16:
DST=',"kv_connector_extra_config":{"destination_decode_context_parallel_size":16}'
DECODE_DCP=16 DECODE_LOADER='{"presharded":"/models/kimi-k3-tp32","attention_dp":2}' MNS=128
# Point 9 (concurrency 48), DCP32:
DST= DECODE_DCP=32 DECODE_LOADER='{"presharded":"/models/kimi-k3-tp32"}' MNS=64
```

Prefill, on each of the 4 prefill hosts:

```bash
docker run --rm --privileged --net=host --ipc=host -v /models:/models \
  -e TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 -e TPU_HOST_BOUNDS=1,1,4 -e TPU_NODE_ID="$NODE_RANK" -e TPU_SKIP_MDS_QUERY=1 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e TCMALLOC_AGGRESSIVE_DECOMMIT=true -e NPROC=2 \
  -e LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7" \
  -e VLLM_USE_RUST_FRONTEND=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800 -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
  -e TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 -e TPU_RAIDEN_KIMIK3_ADMISSION=1 -e TPU_KV_RESHARD_TRANSPORT=raiden -e TPU_RAIDEN_RESHARD_IMPL=store \
  -e TPU_RAIDEN_ADVERTISE_HOST="$PREFILL_IP" -e TPU_RAIDEN_PREFIX_AWARE_LOAD=1 -e TPU_RAIDEN_TRANSFER_PARALLELISM=32 -e TPU_RAIDEN_DST_SHARDS=32 \
  -e TPU_RAIDEN_TRANSFER_NUM_SLOTS=1 -e TPU_RAIDEN_POOL_STAGING_LEASES=16 -e TPU_P2P_WAIT_PULL_TIMEOUT=900 \
  -e RAIDEN_NUMA_POLICY=bind -e RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=1800 -e TPU_K3_MOE_REDUCE_DTYPE=bfloat16 \
  -e PYTHONHASHSEED=0 -e RAIDEN_ENABLE_STORE_MONITOR=1 -e RAIDEN_ENABLE_EVICT_SWEEP=1 \
  vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c \
  vllm serve /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --language-model-only \
  --tensor-parallel-size 32 --enable-expert-parallel --decode-context-parallel-size 32 \
  --nnodes 4 --node-rank "$NODE_RANK" --master-addr "$PREFILL_IP" --distributed-timeout-seconds 1800 $([ "$NODE_RANK" = 0 ] || echo --headless) \
  --max-model-len 1048576 --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512 \
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details --load-format megakernel --kv-cache-memory-bytes 23622320128 \
  --model-loader-extra-config '{"presharded":"/models/kimi-k3-tp32"}' \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 4096 --numa-bind --numa-bind-nodes 0 0 0 0 1 1 1 1 \
  --kv-transfer-config '{"kv_connector":"TPUMultiConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer"'"$DST"'},{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":2199023255552,"store_server_ip":"'"$PREFILL_IP"'","global_registry_address":"'"$PREFILL_IP"':27900","kv_pool_group":"k3-host-store"}}]}}'
```

Decode, on each of the 4 decode hosts:

```bash
docker run --rm --privileged --net=host --ipc=host -v /models:/models \
  -e TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 -e TPU_HOST_BOUNDS=1,1,4 -e TPU_NODE_ID="$NODE_RANK" -e TPU_SKIP_MDS_QUERY=1 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e TCMALLOC_AGGRESSIVE_DECOMMIT=true -e NPROC=2 \
  -e LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7" \
  -e VLLM_USE_RUST_FRONTEND=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800 -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
  -e TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 -e TPU_RAIDEN_KIMIK3_ADMISSION=1 -e TPU_KV_RESHARD_TRANSPORT=raiden -e TPU_RAIDEN_RESHARD_IMPL=store \
  -e TPU_RAIDEN_ADVERTISE_HOST="$DECODE_IP" -e TPU_RAIDEN_PREFIX_AWARE_LOAD=1 -e TPU_RAIDEN_TRANSFER_PARALLELISM=32 -e TPU_RAIDEN_DST_SHARDS=32 \
  -e TPU_RAIDEN_TRANSFER_NUM_SLOTS=1 -e TPU_RAIDEN_POOL_STAGING_LEASES=16 -e TPU_P2P_WAIT_PULL_TIMEOUT=900 \
  -e RAIDEN_NUMA_POLICY=bind -e RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=1800 \
  vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c \
  vllm serve /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --language-model-only \
  --tensor-parallel-size 32 --enable-expert-parallel --decode-context-parallel-size "$DECODE_DCP" \
  --nnodes 4 --node-rank "$NODE_RANK" --master-addr "$DECODE_IP" --distributed-timeout-seconds 1800 $([ "$NODE_RANK" = 0 ] || echo --headless) \
  --max-model-len 1048576 --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512 \
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details --load-format megakernel --kv-cache-memory-bytes 30064771072 \
  --model-loader-extra-config "$DECODE_LOADER" \
  --max-num-seqs "$MNS" --max-num-batched-tokens 512 \
  --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer"}'
```

## 6. Put a P/D proxy in front of the two servers

On the prefill head, once `http://$PREFILL_IP:8000/health` and `http://$DECODE_IP:8000/health` return 200. The router listens on `localhost:30000`.

```bash
docker run --rm --net=host vllm/vllm-torchtpu:nightly-20261019-k3-megakernel-c78529c \
  vllm-router --vllm-pd-disaggregation --prefill "http://$PREFILL_IP:8000" --decode "http://$DECODE_IP:8000" \
  --request-timeout-secs 5000 --disable-retries
```

## 7. Run the AgentX benchmark

On the prefill head, with `aiperf` from [agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness) at `56a0cf7`, one hour at the point's concurrency `CONC` (see [Points](#points)):

```bash
AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800 \
aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:30000 --endpoint-type chat --streaming \
  --model moonshotai/Kimi-K3 --tokenizer /models/Kimi-K3 --tokenizer-trust-remote-code \
  --public-dataset semianalysis_cc_traces_weka_062126 --num-dataset-entries 393 \
  --concurrency "$CONC" --benchmark-duration 3600 --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --warmup-grace-period 1800 --trace-idle-gap-cap-seconds 300 \
  --use-server-token-count --no-gpu-telemetry
```

## Timeouts

| Setting | Default | Why it is raised |
| :---- | :---- | :---- |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | 600 s | The Rust frontend waits this long for the engine's READY, which comes after the weight load and the compilation warmup. |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800` | 300 s | A step that compiles a new program on a cold cache can take longer than the default. |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120` | 5 s | Longer than the router's idle connection pool, so the router never reuses a connection that the server closed. |
| `--distributed-timeout-seconds 1800` | unset (the process-group backend's default) | The process groups wait this long for the ranks of all four hosts to join, and the hosts can start minutes apart. |
| `RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=1800` | 120 s | The host store waits for all 32 ranks to register, which they do only after their weight load. |
| `TPU_P2P_WAIT_PULL_TIMEOUT=900` | 120 s | The limit on one request's KV transfer from the prefill. It also caps the wait for the prefill's block registration. |
| `vllm-router --request-timeout-secs 5000` | 1800 s | The router limit on one request, which includes its whole decode. With `--disable-retries`, a failed hop fails the request instead of sending it to the prefill a second time. |
| `AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800`, `AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800` | 300 s, 600 s | The first build of the AgentX dataset takes several minutes. |

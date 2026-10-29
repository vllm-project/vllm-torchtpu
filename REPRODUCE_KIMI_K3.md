# Reproducing Kimi-K3 TPU v7x Track 2 Disaggregated `1P1D` Megakernel (`tpu-7x-mega-pd`, 32 Chips)

## Topology & Base Container
- **Hardware**: `2x tpu7x 2x2x4` (`16` prefill chips + `16` decode chips = `32` total chips).
- **Base image**: `gcr.io/cloud-tpu-shared-capacity/jianpingsyu/vllm-torchtpu:k3engine-v0.29.0` with `PYTHONPATH` pointing to `src/` from this branch (`rishabhmanoj/k3-mega-ovdp2`).
- **Inputs**:
  - `MODEL_DIR`: Kimi-K3 HF metadata/tokenizer directory (`config.json`, tokenizer files).
  - `PRESHARDED_DIR`: `TP=32` presharded megakernel checkpoint directory (containing `layout.json` and `prefill-sp-cache/layout.json`).

## Environment Variables (`ROLE=prefill` or `ROLE=decode`, `INDEX=0..3`)

```bash
export VLLM_TARGET_DEVICE=tpu VLLM_NO_USAGE_STATS=1 VLLM_SERVER_DEV_MODE=1
export VLLM_HOST_IP="${POD_IP}" VLLM_ENGINE_READY_TIMEOUT_S=14400 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" TPU_HOST_BOUNDS="1,1,4" TPU_NODE_ID="${INDEX}"
export TPU_SKIP_MDS_QUERY=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TCMALLOC_AGGRESSIVE_DECOMMIT=true
export TPU_MULTIHOST_BACKEND=mp
export TPU_MOE_HIERARCHICAL_EP=1 TPU_LATENT_PROJ_INTRA_CHIP_TP=1 ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1 TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 TPU_RAIDEN_KIMIK3_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden TPU_RAIDEN_RESHARD_IMPL=store TPU_RAIDEN_TRANSFER_PARALLELISM=32
export TPU_RAIDEN_PREFIX_AWARE_LOAD=1 TPU_RAIDEN_TRANSFER_NUM_SLOTS=1 TPU_RAIDEN_DST_SHARDS=32
export TPU_RAIDEN_POOL_STAGING_LEASES=16 TPU_P2P_WAIT_PULL_TIMEOUT=900 RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=1800
export TPU_RAIDEN_ADVERTISE_HOST="${ROLE_HEAD_IP}"
export TPU_KV_TRANSFER_NAMESPACE=k3-mk-0
export TPU_RAIDEN_JOB_NAME="k3-mk-0-${ROLE}" TPU_RAIDEN_ENGINE_ID="k3-mk-0-${ROLE}-engine"
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7"

if [[ "${ROLE}" == "prefill" ]]; then
  export TPU_K3_MOE_REDUCE_DTYPE=bfloat16
  export TPU_RAIDEN_CONTROLLER_ADDRESS="${ROLE_HEAD_IP}:27340" TPU_RAIDEN_RESHARD_PORT_BASE=27340
  export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27440 TPU_KV_TRANSFER_PORT=29340 TPU_SIDE_CHANNEL_PORT=29740
else
  export TPU_K3_MOE_REDUCE_DTYPE=float32
  export TPU_RAIDEN_CONTROLLER_ADDRESS="${ROLE_HEAD_IP}:28340" TPU_RAIDEN_RESHARD_PORT_BASE=28340
  export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28440 TPU_KV_TRANSFER_PORT=30340 TPU_SIDE_CHANNEL_PORT=30740
  export TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT=0
fi
```

## Server & Disagg Proxy Launch (`ATTN_DP=2`, `DEC_DCP=16`, `DEC_MNS=128`)
Common flags for both slices (rank `0` uses `--host 0.0.0.0 --port 8000`; ranks `1..3` use `--headless`):

```bash
COMMON_ARGS=(
  "${MODEL_DIR}" --served-model-name moonshotai/Kimi-K3
  --tensor-parallel-size 32 --enable-expert-parallel
  --distributed-timeout-seconds 1800 --max-model-len 1048576
  --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512
  --trust-remote-code --language-model-only
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details
  --load-format megakernel
)
RAIDEN='"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector"'
```

- **Prefill slice (`16` chips, `DCP=32 -> destination_decode_context_parallel_size=16`)**:

```bash
vllm serve "${COMMON_ARGS[@]}" \
  --decode-context-parallel-size 32 \
  --model-loader-extra-config "{\"presharded\":\"${PRESHARDED_DIR}\"}" \
  --max-num-seqs 4 --max-num-batched-tokens 8192 --kv-cache-memory-bytes 23622320128 \
  --kv-transfer-config "{${RAIDEN},\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"destination_decode_context_parallel_size\":16}}" \
  --nnodes 4 --node-rank "${INDEX}" --master-addr "${PREFILL_HEAD_IP}"
```

- **Decode slice (`16` chips, `attention_dp=2`, `DCP=16`, `max_num_seqs=128`)**:

```bash
vllm serve "${COMMON_ARGS[@]}" \
  --decode-context-parallel-size 16 \
  --model-loader-extra-config "{\"presharded\":\"${PRESHARDED_DIR}\",\"attention_dp\":2}" \
  --max-num-seqs 128 --max-num-batched-tokens 512 --kv-cache-memory-bytes 30064771072 \
  --async-scheduling \
  --kv-transfer-config "{${RAIDEN},\"kv_role\":\"kv_consumer\"}" \
  --nnodes 4 --node-rank "${INDEX}" --master-addr "${DECODE_HEAD_IP}"
```

- **Disagg Proxy** (run on prefill head once both `/health` return `200`):

```bash
python3 examples/disagg/toy_proxy_server.py --host 0.0.0.0 --port 8100 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 8000 \
  --decoder-hosts "${DECODE_HEAD_IP}" --decoder-ports 8000
```

## Benchmark Client (`vllm bench serve`, `0` Prefix Cache Hits Verified)
Run a short warmup with a disjoint `--seed`, then `POST /reset_prefix_cache` on both prefill (`127.0.0.1:8000`) and decode (`${DECODE_HEAD_IP}:8000`) engines immediately before measuring, and verify `0` prefix cache hits via `/metrics`:

```bash
curl -s -X POST "http://127.0.0.1:8000/reset_prefix_cache"
curl -s -X POST "http://${DECODE_HEAD_IP}:8000/reset_prefix_cache"

vllm bench serve --backend vllm \
  --model moonshotai/Kimi-K3 --served-model-name moonshotai/Kimi-K3 \
  --tokenizer "${MODEL_DIR}" --trust-remote-code \
  --host 127.0.0.1 --port 8100 --endpoint /v1/completions \
  --dataset-name random --random-input-len "${ISL}" --random-output-len "${OSL}" \
  --random-range-ratio 0.0 --random-prefix-len 0 \
  --num-prompts "$(( 2 * C ))" --num-warmups 0 \
  --max-concurrency "${C}" --request-rate inf --ignore-eos --seed "${SEED}"
```

Divide `output_throughput` by **`32`** (`16` prefill + `16` decode = `32` total chips) for `out tok/s/chip`.

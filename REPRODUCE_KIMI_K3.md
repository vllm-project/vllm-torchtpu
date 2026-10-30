# Reproducing Kimi-K3 TPU v7x Track 1 Aggregated Serving (`tpu-7x-agg`, 16 Chips)

## Topology & Base Container
- **Hardware**: `1x tpu7x 2x2x4` (`4` hosts × `4` chips = `16` chips / `32` TensorCores, `TP=32, EP=32`).
- **Base image**: `gcr.io/cloud-tpu-shared-capacity/jianpingsyu/vllm-torchtpu:k3engine-v0.29.0` with this branch (`rishabhmanoj/k3-track1-v5d5`) installed via `pip install --no-deps --no-build-isolation -e .` (or preserving `/workspace/vllm-torchtpu/src/*.egg-info` alongside `PYTHONPATH=/workspace/vllm-torchtpu/src`; `cd /` first if replacing `/workspace/vllm-torchtpu`).

## Environment Variables (All 4 Hosts)

```bash
export VLLM_TARGET_DEVICE=tpu
export VLLM_HOST_IP="${POD_IP}"
export VLLM_SERVER_DEV_MODE=1
export VLLM_NO_USAGE_STATS=1
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-cloud-tpu-shared-capacity}"
export TPU_SKIP_MDS_QUERY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TCMALLOC_AGGRESSIVE_DECOMMIT=true
export ALLOW_MULTIPLE_LIBTPU_LOAD=1
export RUNAI_STREAMER_CONCURRENCY=8
export RUNAI_STREAMER_MEMORY_LIMIT=17179869184
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0

# Track 1 operator & memory optimizations
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_K3_FP8_LATENT_PROJ=1
export TPU_MOE_COMPACT_SCALE=e8m0
export TPU_MOE_COUNTING_SORT=1
export TPU_KDA_GPROJ_LAYOUT=head_major
export TPU_MOE_OWNER_OUTPUT_MODE=on
export TPU_MOE_OWNER_OUTPUT_MIN_ROUTES=4096
export TPU_MOE_ROUTER_TOPK=rowmax
export VLLM_TPU_ENABLE_TOPK_OPTIMIZATION=1
export DISTRIBUTED_SAMPLING_MAX_TOP_K=64
export TPU_K3_SP_PREFILL=0

# d5 scheduler (omit both lines for pure FCFS)
export TPU_SCHED_DECODE_INTERLEAVE_STEPS=4
export TPU_SCHED_PREFILL_SRPT=1

export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7"
```

## Server Launch (`vllm serve`)
Use `MAX_NUM_SEQS=128` (`COMPILE_SIZES="[1,2,4,8,16,32,64,128,1024,2048,4096,8192]"`) for `c <= 128`, or `MAX_NUM_SEQS=192` (`COMPILE_SIZES="[1,2,4,8,16,32,64,128,192,1024,2048,4096,8192]"`, pure FCFS recommended at `c=192`) for `c=192`:

```bash
# Head node (rank 0): add --port 8000
# Worker nodes (ranks 1..3): add --headless
vllm serve "${SERVE_MODEL_GCS_URI}" \
  --served-model-name moonshotai/Kimi-K3 \
  --tokenizer "${LOCAL_OR_FUSE_TOKENIZER_DIR}" \
  --trust-remote-code \
  --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --tensor-parallel-size 32 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --load-format runai_streamer \
  --model-loader-extra-config '{"memory_limit":17179869184}' \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 9472 \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens 8192 \
  --attention-backend CUSTOM \
  --no-enable-prefix-caching \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --compilation-config "{\"compile_sizes\":${COMPILE_SIZES}}" \
  --nnodes 4 --node-rank "${RANK}" --master-addr "${HEAD_IP}" --master-port 29500
```

## Benchmark Client (`vllm bench serve`)
Protocol: `N = 2 * c`, `0%` prefix cache (`--no-enable-prefix-caching`), `0` preemptions, `--ignore-eos`, `--random-range-ratio 0.0` (run a short warmup first so XLA buckets are warm):

```bash
vllm bench serve --backend vllm \
  --model moonshotai/Kimi-K3 \
  --served-model-name moonshotai/Kimi-K3 \
  --tokenizer "${LOCAL_OR_FUSE_TOKENIZER_DIR}" \
  --trust-remote-code \
  --host 127.0.0.1 --port 8000 --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len "${ISL}" --random-output-len "${OSL}" \
  --random-range-ratio 0.0 --random-prefix-len 0 \
  --num-prompts "$(( 2 * C ))" --num-warmups 0 \
  --max-concurrency "${C}" --request-rate inf --ignore-eos
```

Divide `output_throughput` by `16` (`CHIPS=16`) for `out tok/s/chip`.

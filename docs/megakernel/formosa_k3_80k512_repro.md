# Reproducing the Formosa Kimi-K3 `80128/512` results (16× TPU7x aggregate)

This page is the end-to-end recipe behind the `tpu-7x-mega-agg` rows in
go/formosa-poc-exp: Kimi-K3 served by the megakernel **aggregate** role
(prefill and decode programs on one engine) on a single 16-chip TPU7x slice,
benchmarked with `vllm bench serve` at fixed `ISL/OSL = 80128/512`.

Everything needed is public in this branch or produced by its tools; nothing
depends on a private artifact.

| Item | Value |
|---|---|
| Branch / commit | `rishabhmanoj/k3-formosa-1030` @ `f43847168` |
| Container | `gcr.io/cloud-tpu-shared-capacity/jianpingsyu/vllm-torchtpu:k3engine-v0.29.0` (vLLM 0.29) with this branch's `src/vllm_torchtpu` overlaid |
| Hardware | 1 × TPU7x `2x2x4` slice = 4 hosts × 4 chips = 16 chips = 32 TensorCores (`world_size=32`) |
| Model | `moonshotai/Kimi-K3` (MXFP4 weights, FP8 KV), language model only |
| Sharding | `TP32 DCP32`, routed experts `EP8 × TP4` shared between the prefill and decode programs |

## Expected results (`0` prefix-cache hits, `0` preemptions)

`Tot tok/s/chip = total_token_throughput / 16` from `vllm bench serve`, N = 2c
prompts, `--ignore-eos`, exact lengths. Two independent runs (fresh boots,
separate clients and seeds) agreed to within ±0.2 %.

| c | Tot tok/s/chip | Out tok/s/chip | Med / Mean TTFT (ms) | Med TPOT (ms) |
|--:|--:|--:|--:|--:|
| 1 | 794 | 5.04 | 4,268 / 4,268 | 4.06 |
| 2 | 907 | 5.76 | 6,354 / 6,366 | 9.26 |
| 4 | 980 | 6.22 | 7,645 / 8,705 | 25.2 |
| 8 | 1,056 | 6.70 | 7,289 / 13,009 | 60.0 |
| 16 | 1,086 | 6.90 | 6,101 / 21,114 | 131.6 |
| 32 | 1,115 | 7.08 | 5,737 / 38,263 | 266.2 |
| 64 | 1,116 | 7.09 | 78,535 / 109,989 | 404.0 |

Notes on reading them:

- Prefill is serial per request (`K3_SERIAL_PREFILL=1`, ~4.3 s for an 80k
  prompt), so TTFT is bimodal at `c >= 8`: compare the median with the mean.
- At `c=64` the engine admits only ~48 running requests at the 14 GiB/core KV
  budget (`--max-num-seqs 64`), so that row is effectively ~48 concurrent plus
  queueing; it has `0` preemptions because the extra requests simply wait.
- Quality on the same server configuration: GSM8K `96.89 %` strict-match
  (1,319 questions, 5-shot greedy, thinking off), MMLU-Pro `83.36 %` (1,400
  questions = 100/subject × 14, 5-shot), `32/32` MMLU answer agreement with the
  unoptimised reference build on the 1,024-token greedy gate, and the greedy
  smoke prompt `"The capital of France is"` → `" Paris. ..."`.

## 1. Weights

Pre-shard the HF checkpoint into the kernels' per-rank layout once (CPU only,
one process per host; see `docs/megakernel/README.md` → Tools):

```bash
python -m tools.megakernel preshard k3      --checkpoint /ckpt/Kimi-K3 --output /models/k3pre
python -m tools.megakernel preshard prefill --checkpoint /ckpt/Kimi-K3 --output /models/k3pre
```

The result is `layout.json` (`format: 1, ranks: 32`), `rank00..rank31/` and
`prefill-sp-cache/` (~1.7 TB). `/models/k3meta` is the plain HF directory
(`config.json`, tokenizer) used for `vllm serve <model>` and `--tokenizer`.
The Formosa PoC bucket already holds this exact pre-shard
(`gs://formosa-poc/kimi-k3/presharded/kimi-k3-tp32/`, built by
`tools.megakernel preshard` at `3af67bd47` from
`gs://tpu-commons-ci/moonshootai/kimi/k3`; see its `PROVENANCE.txt`) — the
measured runs used a byte-for-byte copy of it, and it is the same directory
that serves the `tpu-7x-mega-pd` rows.

## 2. Source overlay

The image ships vLLM 0.29 and a vllm-torchtpu build; replace only the package:

```bash
git archive f43847168 src/vllm_torchtpu | tar -x -C /workspace/vllm-torchtpu
export PYTHONPATH=/workspace/vllm-torchtpu/src:$PYTHONPATH
```

## 3. Server (identical on all 4 hosts; host 0 serves, hosts 1–3 `--headless`)

```bash
# multi-host TPU7x slice
export TPU_MULTIHOST_BACKEND=mp TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,4 TPU_NODE_ID=$INDEX
export TPU_SKIP_MDS_QUERY=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TCMALLOC_AGGRESSIVE_DECOMMIT=true
export VLLM_TARGET_DEVICE=tpu VLLM_NO_USAGE_STATS=1 VLLM_SERVER_DEV_MODE=1      # DEV_MODE exposes /reset_prefix_cache
export VLLM_HOST_IP=$POD_IP VLLM_ENGINE_READY_TIMEOUT_S=14400 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7"
# engine knobs
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1 TPU_MOE_HIERARCHICAL_EP=1 TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768 TPU_K3_MOE_REDUCE_DTYPE=float32
# K3 aggregate role (this branch)
export K3_AGGREGATE_EP16=0            # share the EP8/TP4 expert tensors between prefill and decode (fits 16 chips)
export K3_PREFILL_PREFIX_FP8=1        # FP8 prefix-KV tile reads in context-parallel prefill attention
export K3_PREFILL_SJF=1               # shortest-job-first prefill queue
export K3_SERIAL_PREFILL=1            # one request prefills all its windows before the next starts
export K3_MAX_PREFILL_STEP_TOKENS=8192

ROLE_FLAGS=(--host 0.0.0.0 --port 8000); [[ "$INDEX" != "0" ]] && ROLE_FLAGS=(--headless)

vllm serve /models/k3meta \
  --served-model-name moonshotai/Kimi-K3 \
  --tensor-parallel-size 32 --decode-context-parallel-size 32 --enable-expert-parallel \
  --distributed-timeout-seconds 1800 \
  --max-model-len 163840 \
  --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512 \
  --trust-remote-code --language-model-only \
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details \
  --load-format megakernel \
  --model-loader-extra-config '{"presharded":"/models/k3pre"}' \
  --max-num-seqs 64 --max-num-batched-tokens 8256 \
  --kv-cache-memory-bytes 15032385536 \
  --async-scheduling \
  --nnodes 4 --node-rank "$INDEX" --master-addr "$HEAD_IP" \
  "${ROLE_FLAGS[@]}"
```

`--kv-cache-memory-bytes 15032385536` (14.0 GiB/core) is the calibrated budget
for this layout: resident weights are ~71 GiB/core (49.7 decode incl. the
shared experts + 21.2 prefill projections) of 94.7 GiB. Prefix caching has to
be on for `--mamba-cache-mode align`; the benchmark below resets it before
every point and verifies `0` hits. There is no speculative decoding and no KV
offloading. First boot compiles for ~25 min (a persisted
`TORCH_TPU_CACHE_DIR`/`VLLM_XLA_CACHE_PATH` cuts that to ~15 min).

## 4. Benchmark client (on host 0)

Warm up once, then for each concurrency reset the prefix cache, snapshot
`/metrics`, run the point with its own seed, and snapshot again:

```bash
EP=http://127.0.0.1:8000
curl -s -X POST $EP/reset_prefix_cache
vllm bench serve --backend openai --host 127.0.0.1 --port 8000 --endpoint /v1/completions \
  --model moonshotai/Kimi-K3 --tokenizer /models/k3meta --trust-remote-code \
  --dataset-name random --random-input-len 80128 --random-output-len 32 --random-range-ratio 0.0 \
  --ignore-eos --request-rate inf --num-prompts 2 --max-concurrency 1 --num-warmups 0 --seed 90031 --temperature 0.0

for c in 1 2 4 8 16 32 64; do
  curl -s -X POST $EP/reset_prefix_cache
  curl -s $EP/metrics > c${c}.before.prom
  vllm bench serve --backend openai --host 127.0.0.1 --port 8000 --endpoint /v1/completions \
    --model moonshotai/Kimi-K3 --tokenizer /models/k3meta --trust-remote-code \
    --dataset-name random --random-input-len 80128 --random-output-len 512 --random-range-ratio 0.0 \
    --ignore-eos --request-rate inf --max-concurrency $c --num-prompts $((2*c)) \
    --num-warmups 0 --seed $((50512 + c*131)) --temperature 0.0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 25,50,75,90,95,99 \
    --save-result --save-detailed --result-filename k3_80128_512_c${c}.json
  curl -s $EP/metrics > c${c}.after.prom
done
```

Acceptance checks per point (all held in both measured runs):

- `completed == 2c`, `failed == 0`, every request `input_len == 80128` and
  `output_len == 512` (from the detailed result JSON);
- Prometheus deltas: `vllm:prompt_tokens_total == 2c × 80128`,
  `vllm:generation_tokens_total == 2c × 512`, `vllm:prefix_cache_hits_total`
  delta `0`, `vllm:num_preemptions_total` delta `0`;
- `Tot tok/s/chip = total_token_throughput / 16`.

## 5. Quality gates (same server)

```bash
# smoke
curl -s $EP/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"moonshotai/Kimi-K3","prompt":"The capital of France is","max_tokens":32,"temperature":0.0}'
# accuracy (lm-eval-harness, chat completions, thinking off; tasks' default few-shot)
lm_eval --tasks gsm8k --seed "0,1234,None,1234" --log_samples \
  --model local-chat-completions --apply_chat_template \
  --model_args "model=moonshotai/Kimi-K3,base_url=$EP/v1/chat/completions,num_concurrent=32,max_retries=3,timeout=3600" \
  --gen_kwargs '{"chat_template_kwargs": {"thinking": false}, "max_gen_toks": 1024, "temperature": 0.0}'
lm_eval --tasks mmlu_pro --seed "0,1234,None,1234" --log_samples --limit 100 \
  --model local-chat-completions --apply_chat_template \
  --model_args "model=moonshotai/Kimi-K3,base_url=$EP/v1/chat/completions,num_concurrent=16,max_retries=3,timeout=3600" \
  --gen_kwargs '{"chat_template_kwargs": {"thinking": false}}'
```

## What is in this branch relative to `dev/k3-aggregate`

- `KimiAggregate` passes `tp_rank=self.decoder.rank` to the prefiller so prefill
  page ownership and KDA head order match the decoder on multi-host slices
  (`megakernel/kimi_k3/aggregate.py`); without it the output is incoherent on a
  `2x2x4` slice even though throughput looks normal.
- Owned-page `PoolWriter._write_op`, SP-local `kv_latent` RMSNorm before the
  all-gather, FP8 prefix tiles in prefill MLA, shared EP8/TP4 experts between
  the two programs, and the serial / shortest-job-first prefill scheduler knobs
  (`K3_SERIAL_PREFILL`, `K3_PREFILL_SJF`, `K3_MAX_PREFILL_STEP_TOKENS`).

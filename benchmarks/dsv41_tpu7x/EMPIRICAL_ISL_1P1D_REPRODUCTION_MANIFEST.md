# DeepSeek-V4.1-Flash FP8 PD Disaggregation (`1P1D`) — Empirical ISL (`50:20:12:9:9`) & Uncached Prefill Reproduction Manifest

## 1. 1P1D Server & Proxy Launch ([`run_pd.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/run_pd.py))

```bash
PREFILL_4X_TP2_GPU_MEM=0.92 python3 -u scripts/pd/run_pd.py \
  --ctx "$CTX" \
  --suffix p2 \
  --tag dsv41_pd_4xtp2_20l_disjoint \
  --vtt repos/vtt-combined \
  --vllm repos/vllm-decode \
  --prefill-mode 4x_tp2 \
  --prefill-ced 1 \
  --prefill-mnbt 16384 \
  --decode-mnbt 96 \
  --enable-dspark 1 \
  --allow-foreign \
  --keep-pods
```

This provisions (via [`in_pod_serve.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/in_pod_serve.sh), [`in_pod_setup.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/in_pod_setup.sh), and [`launch_dsv41_disagg_role.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/examples/disagg/launch_dsv41_disagg_role.sh)):
- **Prefill Pod (`1P`)**: `2x2x1` TPU7x (`4` chips / `8` TensorCores) running **4 independent `TP=2, EP=2` 20-layer Context Encoder (`VLLM_DSV41_CED=1`, `VLLM_DSV41_PURE_20L_PREFILL=1`) servers** on ports `8400, 8401, 8402, 8403` (`1` chip per server), with `VLLM_DISJOINT_KV_PARTITIONS=1`, `PREFILL_4X_TP2_GPU_MEM=0.92` (`1,659` blocks/partition = `212,352` tokens/chip = `849,408` tokens/node per partition across disjoint `main`, `state`, `swa_0`, `swa_1`, `swa_2` `BlockPool` queues), `--kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_role":"kv_producer"}'`, `--enable-prefix-caching --prefix-cache-retention-interval 1024 --max-num-batched-tokens 16384 --max-num-seqs 8 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice`.
- **Decode Pod (`1D`)**: `2x2x1` TPU7x (`DP=8, TP=1, EP=8`), port `9400`, `--kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_role":"kv_consumer"}'` with `VLLM_DISJOINT_KV_PARTITIONS=1` (`720` blocks/partition = `92,160` tokens/rank = `737,280` tokens/node per partition), `ENABLE_DSPARK=1` (`method="dflash"`, `num_speculative_tokens=5`), `--async-scheduling --enable-prefix-caching --prefix-cache-retention-interval 1024 --max-num-batched-tokens 96 --max-num-seqs 16 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice`.
- **Disaggregated Proxy ([`toy_proxy_server.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/examples/disagg/toy_proxy_server.py))**: Port `8000` on the prefill pod (`--prefiller-ports 8400,8401,8402,8403 --prefiller-dp-size 1 --decoder-ports 9400 --decoder-dp-size 8 --prefill-max-inflight 8`), routing DP rank `r` (`0..7`) deterministically to prefill producer `8400 + (r // 2)` and decode DP rank `r`, and broadcasting `/reset_prefix_cache` across all 4 `1P` producers and `1D`.

---

## 2. Empirical ISL (`50:20:12:9:9`, 95% Prefix Cache Hit) Benchmark Execution ([`bench_empirical_isl_dsv41.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/bench_empirical_isl_dsv41.py))

```bash
# 95% Cache Hit — 5 Trunks/Rank (40 Trunks Total across 8 Decode DP Ranks / 4 Prefill Producers)
# Runs inside the 1P pod against the disaggregated proxy at http://127.0.0.1:8000
COMPACT_WARMUP_SUFFIX=1 python3 -u /tmp/bench_empirical_isl_dsv41.py \
  --base-url http://127.0.0.1:8000 \
  --prefill-url http://127.0.0.1:8400,http://127.0.0.1:8401,http://127.0.0.1:8402,http://127.0.0.1:8403 \
  --decode-url http://<DECODE_POD_IP>:9400 \
  --dataset /data/requests_mocked.jsonl.xz \
  --sessions 8,16,32,64,128 \
  --duration-per-n 180 \
  --cache-hit-rate 0.95 \
  --max-tokens 1024 \
  --ignore-eos \
  --dp-size 8 \
  --trunks-above-n16 40 \
  --warmup-pass-s 30 \
  --output-json /tmp/dsv41_pd_empirical_isl_95_summary.json
```

---

## 3. Uncached Prefill (`0%` Cache Hit, `output_len = 1`) Benchmark Execution on `1P` & `1P1D`

```bash
# Example: 128K input, c=8, n=32 on 1P (round-robin across ports 8400..8403) and E2E 1P1D (port 8000)
for WL in "131072 8 32" "131072 16 32" "131072 4 16" "102400 8 32" "102400 16 32" "32768 8 64"; do
  set -- $WL
  ISL=$1; CONC=$2; PROMPTS=$3
  # Reset prefix cache across 1P (8400..8403) + 1D (9400) via proxy
  curl -s -X POST http://127.0.0.1:8000/reset_prefix_cache
  vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model /dev/shm/DeepSeek-V4.1-Flash \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len 1 \
    --random-range-ratio 0.0 \
    --num-prompts "$PROMPTS" \
    --max-concurrency "$CONC" \
    --request-rate inf \
    --ignore-eos
done
```

---

## 4. Public GitHub & GCS Provenance Links

- **Unified Branch (`cpk/dsv41-combined`)**: [`vllm-project/vllm-torchtpu` (`cpk/dsv41-combined`)](https://github.com/vllm-project/vllm-torchtpu/tree/cpk/dsv41-combined/benchmarks/dsv41_tpu7x)
- **Benchmark Harness & Orchestration Scripts**:
  - [`bench_empirical_isl_dsv41.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/bench_empirical_isl_dsv41.py)
  - [`run_pd.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/run_pd.py)
  - [`launch_dsv41_disagg_role.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/examples/disagg/launch_dsv41_disagg_role.sh)
  - [`in_pod_serve.sh`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/in_pod_serve.sh)
  - [`pd_gates.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/scripts/pd/pd_gates.py)
  - [`toy_proxy_server.py`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/examples/disagg/toy_proxy_server.py)
- **Benchmark Reports & Consolidated JSON Summaries (GitHub)**:
  - [`EMPIRICAL_ISL_1P1D_REPORT.md`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/EMPIRICAL_ISL_1P1D_REPORT.md)
  - [`UNCACHED_PREFILL_BENCHMARK_SUMMARY.md`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/UNCACHED_PREFILL_BENCHMARK_SUMMARY.md)
  - [`EMPIRICAL_ISL_1P1D_REPRODUCTION_MANIFEST.md`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/EMPIRICAL_ISL_1P1D_REPRODUCTION_MANIFEST.md)
  - [`dsv41_pd_empirical_isl_95_summary.json`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/dsv41_pd_empirical_isl_95_summary.json)
  - [`dsv41_pd_uncached_prefill_summary.json`](https://github.com/vllm-project/vllm-torchtpu/blob/cpk/dsv41-combined/benchmarks/dsv41_tpu7x/dsv41_pd_uncached_prefill_summary.json)
- **GCS Archives**:
  - `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/phase2_4xtp2_1p1d_20261030/20261030-204736_dsv41_pd_4xtp2_20l_disjoint_p2/`
  - `gs://chaitanyapk-data/atlas/dsv4.1_vllm_torchtpu/phase2_4xtp2_1p1d_20261030/20261030-074500_dsv41_pd_4xtp2_20l_phase2/`
  - `gs://chaitanyapk-dsv41-staging/runs/pd/20261030-204736_dsv41_pd_4xtp2_20l_disjoint_p2/`
  - `gs://chaitanyapk-dsv41-staging/runs/pd/20261030-074500_dsv41_pd_4xtp2_20l_phase2/`
